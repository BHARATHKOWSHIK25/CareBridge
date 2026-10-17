"""Confidence calculation and calibrated answer/no-answer policy."""

import json
import logging
import math
from pathlib import Path

from app.config import get_settings

logger = logging.getLogger(__name__)
DEFAULT_WEIGHTS = {
    "bias": -2.2,
    "retrieval_strength": 2.2,
    "grounding": 2.0,
    "consistency": 1.0,
    "translation_consistency": 0.7,
}


def _weights() -> dict[str, float]:
    """Read validated gate weights or return safe defaults."""
    path = get_settings().gate_weights_file
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, dict) and all(
            key in data and isinstance(data[key], float | int) for key in DEFAULT_WEIGHTS
        ):
            return {key: float(data[key]) for key in DEFAULT_WEIGHTS}
    except (OSError, json.JSONDecodeError, TypeError):
        logger.info("Gate weights unavailable; using configured defaults.")
    return DEFAULT_WEIGHTS.copy()


def _calibrated_threshold() -> float:
    """Use an explicitly configured threshold or the threshold fitted from dev data."""
    settings = get_settings()
    if "gate_threshold" in getattr(settings, "model_fields_set", set()):
        return settings.gate_threshold
    path = Path(settings.gate_weights_file)
    try:
        value = json.loads(path.read_text(encoding="utf-8")).get("suggested_threshold")
        if isinstance(value, int | float) and 0 <= value <= 1:
            return float(value)
    except (OSError, json.JSONDecodeError, TypeError):
        logger.info("Calibrated threshold unavailable; using configured default.")
    return settings.gate_threshold


def calculate_confidence(
    retrieval_strength: float,
    grounding: float,
    consistency: float,
    *,
    translation_consistency: float | None = None,
    answerable_flag: bool = True,
) -> tuple[float, dict[str, float]]:
    """Combine normalized confidence signals through a logistic model."""
    signals = {
        "retrieval_strength": min(1.0, max(0.0, retrieval_strength)),
        "grounding": min(1.0, max(0.0, grounding)),
        "consistency": min(1.0, max(0.0, consistency)),
    }
    if translation_consistency is not None:
        signals["translation_consistency"] = min(1.0, max(0.0, translation_consistency))
    weights = _weights()
    active_weights = {name: weights[name] for name in signals}
    # Renormalize omitted language-specific signals while preserving the learned intercept.
    weighted = sum(active_weights[name] * value for name, value in signals.items())
    if translation_consistency is None:
        configured = sum(
            weights[name]
            for name in (
                "retrieval_strength",
                "grounding",
                "consistency",
                "translation_consistency",
            )
        )
        present = sum(weights[name] for name in active_weights)
        if present:
            weighted *= configured / present
    z = weights["bias"] + weighted
    confidence = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
    if not answerable_flag:
        confidence = min(confidence, 0.2)
    breakdown = signals | {"answerable_flag": 1.0 if answerable_flag else 0.0}
    return confidence, breakdown


def should_answer(confidence: float, answerable_flag: bool, enabled: bool | None = None) -> bool:
    """Apply the configured threshold and hard answerability condition."""
    settings = get_settings()
    if not answerable_flag:
        return False
    gate_enabled = settings.gate_enabled if enabled is None else enabled
    return not gate_enabled or confidence >= _calibrated_threshold()
