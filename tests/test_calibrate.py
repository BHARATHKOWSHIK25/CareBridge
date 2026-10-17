"""Dev-only calibration helper tests."""

import json

import pytest

from eval.calibrate import calibrate


def test_calibration_fits_weights_and_selects_dev_threshold(tmp_path):
    """Calibration writes a valid coefficient set using only explicitly marked dev rows."""
    source = tmp_path / "dev.jsonl"
    destination = tmp_path / "weights.json"
    rows = []
    for index in range(12):
        correct = index >= 6
        rows.append(
            {
                "split": "dev",
                "answer_correct": correct,
                "breakdown": {
                    "retrieval_strength": index / 11,
                    "grounding": index / 11,
                    "consistency": index / 11,
                    "translation_consistency": 0.5,
                },
            }
        )
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    result = calibrate(source, destination, target_error_rate=0.05)
    assert 0 <= result["suggested_threshold"] <= 1
    assert {"bias", "retrieval_strength", "grounding", "consistency"} <= result.keys()
    assert destination.exists()


def test_calibration_rejects_test_split_rows(tmp_path):
    """Test-set rows cannot be used to fit or choose the gate."""
    source = tmp_path / "test.jsonl"
    source.write_text(
        json.dumps(
            {
                "split": "test",
                "answer_correct": True,
                "breakdown": {},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="dev split"):
        calibrate(source, tmp_path / "weights.json")
