"""Confidence-gate behavior."""

from types import SimpleNamespace

from app.gate import calculate_confidence, should_answer


def test_confidence_is_monotonic_for_positive_signals():
    """Higher retrieval and grounding signals must not lower confidence."""
    low, _ = calculate_confidence(0.2, 0.3, 0.4)
    high, _ = calculate_confidence(0.8, 0.9, 0.9)
    assert high > low


def test_answerability_is_a_hard_gate():
    """A model's non-answerable decision cannot be overridden by high signals."""
    confidence, breakdown = calculate_confidence(1, 1, 1, answerable_flag=False)
    assert confidence <= 0.2
    assert breakdown["answerable_flag"] == 0


def test_threshold_and_no_gate_baseline(monkeypatch):
    """The threshold escalates low confidence unless the baseline disables the gate."""
    monkeypatch.setattr(
        "app.gate.get_settings",
        lambda: SimpleNamespace(
            gate_weights_file="missing.json", gate_enabled=True, gate_threshold=0.65
        ),
    )
    assert should_answer(0.7, True)
    assert not should_answer(0.4, True)
    assert not should_answer(1.0, False)
    monkeypatch.setattr(
        "app.gate.get_settings",
        lambda: SimpleNamespace(
            gate_weights_file="missing.json", gate_enabled=False, gate_threshold=0.65
        ),
    )
    assert should_answer(0.01, True)
