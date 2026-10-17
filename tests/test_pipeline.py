"""Offline end-to-end safety pipeline behavior."""

import pytest

from app.handoff import get_ticket
from app.pipeline import handle_message
from app.schemas import GroundingResult


@pytest.mark.asyncio
async def test_answerable_question_returns_cited_answer():
    """A supported shower-care question is answered with a source passage."""
    result = await handle_message("demo-appendectomy", "Can I shower?", "session-a")
    assert result.status == "answered"
    assert result.answer
    assert result.citations
    assert "shower" in result.answer.lower()
    assert 0 <= result.confidence <= 1


@pytest.mark.asyncio
async def test_incisions_changes_question_returns_source_warning():
    """A symptom-intent question is answered from the wound warning sentence."""
    result = await handle_message(
        "demo-appendectomy",
        "What changes around the incision should I report?",
        "session-wound-changes",
    )
    assert result.status == "answered"
    assert "redness" in result.answer.lower()
    assert result.citations


@pytest.mark.asyncio
async def test_unsupported_question_is_escalated_with_context():
    """Medication restart questions are not answered and create a review ticket."""
    result = await handle_message("demo-knee", "When do I restart my tablets?", "session-b")
    assert result.status == "escalated"
    assert result.ticket_id
    ticket = get_ticket(result.ticket_id)
    assert ticket is not None
    assert ticket["original_message"] == "When do I restart my tablets?"
    assert ticket["retrieved_chunks"]
    assert ticket["reason"]
    assert ticket["status"] == "open"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("patient", "message"),
    [
        ("demo-appendectomy", "I have chest pain"),
        ("demo-cataract", "अचानक नजर चली गई"),
        ("demo-knee", "నాకు ఛాతి నొప్పి ఉంది"),
    ],
)
async def test_redflag_paths_create_urgent_ticket_without_medical_answer(patient, message):
    """English, Telugu, and Hindi urgent messages never receive a medical answer."""
    result = await handle_message(patient, message, "session-red")
    assert result.status == "red_flag"
    assert (
        "नर्स" in result.answer or "నర్సు" in result.answer or "nurse team" in result.answer.lower()
    )
    assert result.ticket_id
    ticket = get_ticket(result.ticket_id)
    assert ticket["priority"] == "urgent"
    assert ticket["draft_answer"] is None


@pytest.mark.asyncio
async def test_translation_failure_escalates(monkeypatch):
    """A translation failure never falls through to retrieval or an answer."""

    async def fail(*args, **kwargs):
        raise RuntimeError("translation unavailable")

    monkeypatch.setattr("app.pipeline.translate_to_english", fail)
    result = await handle_message("demo-cataract", "What does this mean?", "session-trans")
    assert result.status in {"escalated", "red_flag"}
    assert result.answer


@pytest.mark.asyncio
async def test_unsupported_claims_never_pass_even_with_high_confidence(monkeypatch):
    """A positive score cannot override the claim-support hard safety rule."""

    async def unsupported(*args, **kwargs):
        return GroundingResult(
            supported_fraction=1.0,
            unsupported_claims=["A claim not present in the cited document."],
        )

    monkeypatch.setattr("app.pipeline.check_grounding", unsupported)
    result = await handle_message("demo-appendectomy", "Can I shower?", "session-ground")
    assert result.status == "escalated"
    ticket = get_ticket(result.ticket_id)
    assert "Grounding check" in ticket["reason"]


@pytest.mark.asyncio
async def test_unknown_patient_is_rejected():
    """Unmapped patient IDs cannot retrieve any document."""
    with pytest.raises(KeyError):
        await handle_message("patient-not-mapped", "Can I shower?", "session-x")


@pytest.mark.asyncio
async def test_pipeline_classifier_outage_with_rule_match_creates_urgent_ticket(monkeypatch):
    """If classifier fails but a rule matches, pipeline escalates as an urgent red flag."""
    from app.llm import LLMError

    async def fail(*args, **kwargs):
        raise LLMError("classifier timeout")

    monkeypatch.setattr("app.redflag.complete_json", fail)
    result = await handle_message(
        "demo-appendectomy", "I have severe chest pain", "session-outage-rule"
    )
    assert result.status == "red_flag"
    ticket = get_ticket(result.ticket_id)
    assert ticket["priority"] == "urgent"
    assert "chest_pain" in ticket["reason"]
    assert "emergency" in result.answer.lower() or "nurse" in result.answer.lower()


@pytest.mark.asyncio
async def test_pipeline_classifier_outage_without_rule_match_creates_normal_escalation(
    monkeypatch,
):
    """If classifier fails and no rule matches, pipeline creates normal escalation."""
    from app.llm import LLMError

    async def fail(*args, **kwargs):
        raise LLMError("classifier timeout")

    monkeypatch.setattr("app.redflag.complete_json", fail)
    result = await handle_message(
        "demo-appendectomy", "Can I shower today?", "session-outage-norule"
    )
    assert result.status == "escalated"
    assert result.status != "red_flag"
    ticket = get_ticket(result.ticket_id)
    assert ticket["priority"] == "normal"
    assert "classifier_unavailable" in ticket["reason"]
    # Does not contain urgent emergency wording
    assert "temporarily unavailable" in result.answer.lower()

