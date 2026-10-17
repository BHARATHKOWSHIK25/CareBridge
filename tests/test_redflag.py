"""Multilingual safety-screen tests."""

import pytest

from app.llm import LLMError
from app.redflag import screen_red_flags


@pytest.mark.parametrize(
    ("message", "category"),
    [
        ("I have severe chest pain", "chest_pain"),
        ("I have severe stomach pain with vomiting", "severe_abdominal_pain"),
        ("నాకు ఛాతి నొప్పి ఉంది", "chest_pain"),
        ("सीने में दर्द हो रहा है", "chest_pain"),
        ("seene mein dard ho raha hai", "chest_pain"),
        ("मैं खुद को नुकसान पहुंचाना चाहता हूँ", "self_harm"),
        ("నన్ను నేను గాయపరచుకోవాలనిపిస్తోంది", "self_harm"),
        ("The dressing is dry and I feel comfortable", "none"),
    ],
)
async def test_multilingual_redflag_screen(message, category):
    """Flag explicit urgent symptoms in supported scripts and Latin transliteration."""
    flagged, found_category, _ = await screen_red_flags(message)
    assert flagged is (category != "none")
    assert found_category == category


async def test_classifier_outage_with_rule_match_returns_urgent_redflag(monkeypatch):
    """When classifier is down but a rule matches, return urgent red flag."""

    async def fail(*args, **kwargs):
        raise LLMError("timeout")

    monkeypatch.setattr("app.redflag.complete_json", fail)
    flagged, category, confidence = await screen_red_flags("I have severe chest pain")
    assert flagged is True
    assert category == "chest_pain"
    assert confidence == 1.0


async def test_classifier_outage_without_rule_match_returns_normal_escalation_sentinel(
    monkeypatch,
):
    """When classifier is down and no rule matches, return outage sentinel without red flag."""

    async def fail(*args, **kwargs):
        raise LLMError("timeout")

    monkeypatch.setattr("app.redflag.complete_json", fail)
    flagged, category, confidence = await screen_red_flags("Can I shower today?")
    assert flagged is False
    assert category == "classifier_unavailable"
    assert confidence == 0.0
