"""Grounded plain-language answer generation."""

import logging
import re
from typing import Any

from app.llm import complete_json
from app.schemas import GeneratedAnswer

logger = logging.getLogger(__name__)
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "can",
    "do",
    "does",
    "for",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "with",
    "you",
    "your",
    "should",
    "may",
    "am",
    "be",
    "after",
    "before",
    "sheet",
    "document",
    "instructions",
    "instruction",
    "says",
    "say",
    "here",
}
UNSUPPORTED_TOPICS = {
    "alcohol",
    "travel",
    "flight",
    "driving",
    "drive",
    "work",
    "restart",
    "dose",
    "dosage",
    "date",
    "time",
    "tablets",
    "pills",
    "swimming",
    "pool",
    "bath",
    "exercise",
    "therapy",
}


def _terms(text: str) -> set[str]:
    """Return normalized content words for a conservative local demo answer."""
    terms = set(re.findall(r"[a-z0-9]+", text.lower())) - STOP_WORDS
    normalized = set()
    for term in terms:
        if len(term) > 5 and term.endswith("ing"):
            term = term[:-3]
        elif len(term) > 4 and term.endswith("ies"):
            term = f"{term[:-3]}y"
        elif len(term) > 5 and term.endswith("ed"):
            term = term[:-2]
        elif len(term) > 4 and term.endswith("s"):
            term = term[:-1]
        normalized.add(term)
    return normalized


def _local_answer(payload: dict[str, Any]) -> dict[str, Any]:
    """Create an extractive answer only when an exact topical sentence exists."""
    question = str(payload.get("question", ""))
    chunks = payload.get("chunks", [])
    qterms = _terms(question)
    if "understand" in qterms:
        qterms |= {"missing", "confusing", "clarify"}
    if "meal" in qterms:
        qterms.add("food")
    if qterms & {"incision", "wound"} and "change" in qterms:
        qterms.discard("change")
        qterms.discard("around")
        qterms |= {"redness", "warmth", "swelling", "drainage", "pain"}
    medication_terms = {"medicine", "medicines", "medication", "tablet", "tablets", "pill", "pills"}
    medication_clarification = {"understand", "list", "pharmacist", "clarify", "written"}
    medication_request = bool(qterms & medication_terms) and not bool(
        qterms & medication_clarification
    )
    if not qterms or qterms & UNSUPPORTED_TOPICS or medication_request:
        return {
            "answerable": False,
            "answer": "",
            "citations": [],
            "reasoning_note": (
                "The local demo does not have a complete explicit instruction for this topic."
            ),
        }
    candidates: list[tuple[float, int, str, str]] = []
    for chunk in chunks:
        text = str(chunk.get("text", ""))
        chunk_id = str(chunk.get("chunk_id", ""))
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            overlap = qterms & _terms(sentence)
            coverage = len(overlap) / len(qterms)
            if overlap and coverage >= 0.5:
                candidates.append((coverage, len(overlap), chunk_id, sentence.strip()))
    if not candidates:
        return {
            "answerable": False,
            "answer": "",
            "citations": [],
            "reasoning_note": "No retrieved sentence directly answers the question.",
        }
    candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
    _, _, chunk_id, sentence = candidates[0]
    # The local mode never improvises: return a verbatim source sentence.
    return {
        "answerable": True,
        "answer": sentence,
        "citations": [chunk_id],
        "reasoning_note": "Extracted verbatim from a retrieved synthetic passage.",
    }


async def generate_answer(
    question_en: str, chunks: list[dict[str, Any]], temperature: float = 0.0
) -> GeneratedAnswer:
    """Generate an answer using only the supplied retrieved passages."""
    result = await complete_json(
        "Answer in plain language for grade 6-8 using only the retrieved chunks. "
        "Never diagnose, recommend treatment, or change medications. If unsupported, "
        "set answerable=false. Cite every factual claim with provided chunk IDs.",
        {"question": question_en, "chunks": chunks},
        GeneratedAnswer,
        temperature=temperature,
        local_handler=_local_answer,
    )
    citations = {str(chunk.get("chunk_id")) for chunk in chunks}
    if not result.answerable:
        return result
    if (
        not result.answer.strip()
        or not result.citations
        or any(citation not in citations for citation in result.citations)
    ):
        return GeneratedAnswer(
            answerable=False,
            answer="",
            citations=[],
            reasoning_note="The generated answer did not cite valid retrieved passages.",
        )
    return result
