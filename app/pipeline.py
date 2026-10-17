"""Safety-first discharge-question pipeline and nurse escalation."""

import asyncio
import logging
import re
import time
from typing import Any

from app.banned_content import check_banned_content
from app.config import get_settings
from app.evolus import get_backend
from app.gate import calculate_confidence, should_answer
from app.generate import generate_answer
from app.grounding import check_grounding, embedding_similarity
from app.handoff import create_ticket
from app.language import (
    detect_language_async,
    translate_from_english,
    translate_to_english,
    urgent_message,
)
from app.metrics import metrics
from app.redflag import screen_red_flags
from app.schemas import ChatResult, Citation, GeneratedAnswer

_CLASSIFIER_OUTAGE = "classifier_unavailable"

logger = logging.getLogger(__name__)
PATIENTS: dict[str, dict[str, str]] | None = None


def load_patients() -> dict[str, dict[str, str]]:
    """Load and validate the synthetic patient-to-document mapping."""
    import json

    from app.config import get_settings

    raw = json.loads(get_settings().patients_file.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("The patients file must contain a JSON list.")
    patients: dict[str, dict[str, str]] = {}
    for item in raw:
        if not isinstance(item, dict) or not item.get("patient_id") or not item.get("doc_id"):
            raise ValueError("Each synthetic patient must define patient_id and doc_id.")
        patient_id = str(item["patient_id"])
        if patient_id in patients:
            raise ValueError(f"Duplicate patient_id {patient_id!r}.")
        patients[patient_id] = {key: str(value) for key, value in item.items()}
    return patients


def set_patients(patients: dict[str, dict[str, str]] | None) -> None:
    """Set the process patient map or reset it for tests."""
    global PATIENTS
    PATIENTS = patients


def get_patients() -> dict[str, dict[str, str]]:
    """Return the patient map, loading it when needed."""
    global PATIENTS
    if PATIENTS is None:
        PATIENTS = load_patients()
    return PATIENTS


def _clean_message(message: str) -> str:
    """Remove unsafe control characters and normalize whitespace."""
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", message)
    return re.sub(r"\s+", " ", cleaned).strip()


async def _consistency(question_en: str, chunks: list[dict[str, Any]], samples: int) -> float:
    """Sample answers concurrently and compute their pairwise semantic agreement."""
    answers = await asyncio.gather(
        *[generate_answer(question_en, chunks, temperature=0.7) for _ in range(samples)],
        return_exceptions=True,
    )
    valid = [answer for answer in answers if isinstance(answer, GeneratedAnswer)]
    if len(valid) != samples:
        return 0.0
    if any(not answer.answerable for answer in valid):
        return 0.0
    if len(valid) < 2:
        return 1.0
    comparisons = [
        embedding_similarity(valid[i].answer, valid[j].answer)
        for i in range(len(valid))
        for j in range(i + 1, len(valid))
    ]
    return sum(comparisons) / len(comparisons)


async def _escalate(
    *,
    patient_id: str,
    session_id: str,
    language: str,
    original_message: str,
    english_message: str,
    reason: str,
    chunks: list[dict[str, Any]],
    draft_answer: str | None,
    confidence: float,
    breakdown: dict[str, float],
    started: float,
    timings: dict[str, float],
    red_flag: bool = False,
) -> ChatResult:
    """Persist review context and return a no-answer escalation response."""

    ticket_payload = {
        "priority": "urgent" if red_flag else "normal",
        "patient_id": patient_id,
        "language": language,
        "original_message": original_message,
        "english_message": english_message,
        "retrieved_chunks": chunks,
        "draft_answer": draft_answer,
        "confidence": confidence,
        "breakdown": breakdown,
        "reason": reason,
    }
    ticket_id = create_ticket(ticket_payload)
    elapsed = (time.perf_counter() - started) * 1000
    timings["total"] = elapsed
    status = "red_flag" if red_flag else "escalated"
    metrics.record(status, elapsed)
    if red_flag:
        answer = urgent_message(language, get_settings().emergency_contact_text)
    elif reason and _CLASSIFIER_OUTAGE in reason:
        # Classifier was unavailable but no rule matched — do not use emergency wording.
        answer = (
            "Our AI safety check is temporarily unavailable. "
            "A nurse will review your message shortly."
        )
    else:
        answer = (
            "I cannot safely answer from the discharge information available. "
            "A nurse will follow up."
        )
    return ChatResult(
        status=status,
        run_mode=get_settings().run_mode,
        language=language,
        answer=answer,
        confidence=confidence,
        breakdown=breakdown,
        ticket_id=ticket_id,
        latency_ms=elapsed,
        timings=timings,
    )


async def handle_message(
    patient_id: str,
    message: str,
    session_id: str,
    *,
    gate_enabled: bool | None = None,
) -> ChatResult:
    """Run safety screening, translation, document retrieval, generation, and gating."""
    started = time.perf_counter()
    timings: dict[str, float] = {}
    cleaned = _clean_message(message)
    if not cleaned:
        raise ValueError("Message must contain readable text.")
    max_length = get_settings().max_message_length
    if len(cleaned) > max_length:
        raise ValueError(f"Message must be at most {max_length:,} characters.")
    patient = get_patients().get(patient_id)
    if patient is None:
        raise KeyError(f"Unknown demo patient {patient_id!r}.")
    language = await detect_language_async(cleaned)
    step = time.perf_counter()

    try:
        english = await translate_to_english(cleaned, language)
    except Exception as exc:
        logger.warning(
            "Question translation failed (%s); escalating without an answer.",
            type(exc).__name__,
        )
        try:
            flagged, category, _ = await screen_red_flags(cleaned)
        except Exception:
            flagged, category = True, "classifier_unavailable"
        timings["translation"] = (time.perf_counter() - step) * 1000
        return await _escalate(
            patient_id=patient_id,
            session_id=session_id,
            language=language,
            original_message=cleaned,
            english_message="",
            reason=f"Question translation failed: {type(exc).__name__}",
            chunks=[],
            draft_answer=None,
            confidence=0.0,
            breakdown={"translation_consistency": 0.0},
            started=started,
            timings=timings,
            red_flag=flagged,
        )
    timings["translation"] = (time.perf_counter() - step) * 1000
    step = time.perf_counter()
    try:
        flagged, category, _ = await screen_red_flags(cleaned, english)
    except Exception:
        logger.exception("Red-flag screening failed; escalating fail-safe.")
        # Unexpected exception (not LLMError) — treat as outage, not confirmed emergency.
        flagged, category = False, _CLASSIFIER_OUTAGE
    timings["redflag"] = (time.perf_counter() - step) * 1000
    if flagged:
        # True red-flag: rule match or classifier-positive → urgent ticket + emergency wording.
        return await _escalate(
            patient_id=patient_id,
            session_id=session_id,
            language=language,
            original_message=cleaned,
            english_message=english,
            reason=f"Urgent red-flag screen: {category}",
            chunks=[],
            draft_answer=None,
            confidence=0.0,
            breakdown={"red_flag": 1.0},
            started=started,
            timings=timings,
            red_flag=True,
        )
    if category == _CLASSIFIER_OUTAGE:
        # Classifier was unavailable and no rule matched.
        # Normal escalation — nurse sees it, patient gets service-outage message (not emergency).
        return await _escalate(
            patient_id=patient_id,
            session_id=session_id,
            language=language,
            original_message=cleaned,
            english_message=english,
            reason=f"Safety classifier unavailable: {_CLASSIFIER_OUTAGE}",
            chunks=[],
            draft_answer=None,
            confidence=0.0,
            breakdown={"classifier_outage": 1.0},
            started=started,
            timings=timings,
            red_flag=False,
        )

    step = time.perf_counter()
    chunks = get_backend().search_knowledge(patient["doc_id"], english)
    timings["retrieval"] = (time.perf_counter() - step) * 1000
    step = time.perf_counter()
    generation_error: str | None = None
    _s = get_settings()
    try:
        generated = await generate_answer(english, chunks, temperature=0.0)
        if generated.answerable:
            cited = [chunk for chunk in chunks if chunk.get("chunk_id") in generated.citations]
            grounding = await check_grounding(
                generated.answer,
                cited,
                stage_timeout=_s.llm_grounding_timeout_seconds or None,
            )
            consistency = await _consistency(
                english,
                chunks,
                _s.consistency_samples,
            )
        else:
            cited = []
            grounding = None
            consistency = 0.0
    except Exception as exc:
        logger.warning("Answer generation or grounding failed; escalating.", exc_info=True)
        generation_error = f"Model processing failed: {type(exc).__name__}"
        generated = GeneratedAnswer(
            answerable=False,
            answer="",
            citations=[],
            reasoning_note=generation_error,
        )
        cited = []
        grounding = None
        consistency = 0.0
    timings["generation_grounding"] = (time.perf_counter() - step) * 1000

    scores = [float(chunk.get("score", 0.0)) for chunk in chunks]
    top = scores[0] if scores else 0.0
    margin = top - (scores[1] if len(scores) > 1 else 0.0)
    retrieval_strength = min(1.0, max(0.0, 0.8 * top + 0.2 * max(0.0, margin)))
    supported = grounding.supported_fraction if grounding else 0.0
    translation_similarity: float | None = None
    translated_answer: str | None = None
    translation_error: str | None = None
    if generated.answerable and language != "en":
        try:
            translated_answer = await translate_from_english(generated.answer, language)
            back_translation = await translate_to_english(translated_answer, language)
            translation_similarity = embedding_similarity(generated.answer, back_translation)
        except Exception as exc:
            translation_error = f"Answer translation failed: {type(exc).__name__}"
            translation_similarity = 0.0
    confidence, breakdown = calculate_confidence(
        retrieval_strength,
        supported,
        consistency,
        translation_consistency=translation_similarity,
        answerable_flag=generated.answerable,
    )
    grounding_unsafe = bool(
        grounding is None or grounding.supported_fraction < 0.95 or grounding.unsupported_claims
    )
    if (
        translation_error
        or generation_error
        or not generated.answerable
        or grounding_unsafe
        or not should_answer(confidence, generated.answerable, enabled=gate_enabled)
    ):
        reason = (
            translation_error
            or generation_error
            or (
                "Grounding check found unsupported or insufficiently supported claims."
                if grounding_unsafe
                else generated.reasoning_note
            )
        )
        if (
            generated.answerable
            and not grounding_unsafe
            and not translation_error
            and not generation_error
        ):
            reason = f"Confidence {confidence:.3f} is below the configured answer threshold."
        return await _escalate(
            patient_id=patient_id,
            session_id=session_id,
            language=language,
            original_message=cleaned,
            english_message=english,
            reason=reason,
            chunks=chunks,
            draft_answer=generated.answer or None,
            confidence=confidence,
            breakdown=breakdown,
            started=started,
            timings=timings,
        )

    # Hard safety check: ensure the answer contains no prohibited advice
    banned_found, banned_reason = check_banned_content(generated.answer)
    if not banned_found and translated_answer is not None:
        banned_found, banned_reason = check_banned_content(translated_answer)
    if banned_found:
        return await _escalate(
            patient_id=patient_id,
            session_id=session_id,
            language=language,
            original_message=cleaned,
            english_message=english,
            reason=f"Banned medical advice detected in draft answer: {banned_reason}",
            chunks=chunks,
            draft_answer=generated.answer,
            confidence=0.0,
            breakdown=breakdown,
            started=started,
            timings=timings,
        )

    marker_map = {chunk["chunk_id"]: index + 1 for index, chunk in enumerate(cited)}
    answer_en = generated.answer.rstrip()
    for chunk_id in generated.citations:
        marker = marker_map.get(chunk_id)
        if marker is not None:
            answer_en += f" [{marker}]"
    answer = translated_answer if translated_answer is not None else answer_en
    if translated_answer is not None:
        markers = re.findall(r"\[\d+\]", answer_en)
        if any(marker not in translated_answer for marker in markers):
            return await _escalate(
                patient_id=patient_id,
                session_id=session_id,
                language=language,
                original_message=cleaned,
                english_message=english,
                reason="Answer translation did not preserve citation markers.",
                chunks=chunks,
                draft_answer=generated.answer,
                confidence=0.0,
                breakdown=breakdown,
                started=started,
                timings=timings,
            )
    citations: list[Citation] = []
    for chunk_id in generated.citations:
        chunk = next((item for item in cited if item.get("chunk_id") == chunk_id), None)
        if chunk is None:
            continue
        snippet = None
        if language != "en":
            try:
                snippet = await translate_from_english(str(chunk["text"]), language)
            except Exception:
                return await _escalate(
                    patient_id=patient_id,
                    session_id=session_id,
                    language=language,
                    original_message=cleaned,
                    english_message=english,
                    reason=(
                        "Citation translation failed; no unverified translated answer was shown."
                    ),
                    chunks=chunks,
                    draft_answer=generated.answer,
                    confidence=0.0,
                    breakdown=breakdown | {"translation_consistency": 0.0},
                    started=started,
                    timings=timings,
                )
        citations.append(
            Citation(
                chunk_id=chunk_id,
                section=str(chunk["section"]),
                source_text=str(chunk["text"]),
                translated_snippet=snippet,
            )
        )
    elapsed = (time.perf_counter() - started) * 1000
    timings["total"] = elapsed
    metrics.record("answered", elapsed)
    return ChatResult(
        status="answered",
        run_mode=get_settings().run_mode,
        language=language,
        answer=answer,
        citations=citations,
        confidence=confidence,
        breakdown=breakdown,
        latency_ms=elapsed,
        timings=timings,
    )
