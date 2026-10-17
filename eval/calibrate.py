"""Fit confidence-gate coefficients and a target-error coverage threshold on dev only."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from app.config import ROOT_DIR

logger = logging.getLogger(__name__)
FEATURES = [
    "retrieval_strength",
    "grounding",
    "consistency",
    "translation_consistency",
]


def calibrate(
    input_path: Path,
    output_path: Path,
    target_error_rate: float = 0.05,
) -> dict[str, float]:
    """Fit logistic weights and select maximum-coverage threshold on dev predictions."""
    if not 0 <= target_error_rate < 1:
        raise ValueError("target_error_rate must be in [0, 1).")
    records = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("No dev predictions found; run eval.run_eval --split dev first.")
    if any(record.get("split") != "dev" for record in records):
        raise ValueError("Calibration input must contain dev split rows only.")
    rows = [
        record
        for record in records
        if isinstance(record.get("answer_correct"), bool)
        and isinstance(record.get("breakdown"), dict)
    ]
    labels = np.asarray([int(row["answer_correct"]) for row in rows])
    if len(rows) < 4 or len(set(labels.tolist())) < 2:
        raise ValueError(
            "Calibration needs at least four dev rows with both correct and incorrect labels."
        )
    features = np.asarray(
        [[float(row["breakdown"].get(name, 0.5)) for name in FEATURES] for row in rows],
        dtype=float,
    )
    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    model.fit(features, labels)
    probabilities = model.predict_proba(features)[:, 1]
    candidates = sorted({0.0, 1.0, *(float(value) for value in probabilities)})
    feasible: list[tuple[int, float]] = []
    for threshold in candidates:
        selected = probabilities >= threshold
        coverage = int(selected.sum())
        if not coverage:
            continue
        error_rate = 1 - float(labels[selected].mean())
        if error_rate <= target_error_rate:
            feasible.append((coverage, threshold))
    suggested_threshold = (
        min(feasible, key=lambda item: (-item[0], item[1]))[1] if feasible else 1.0
    )
    result: dict[str, float] = {
        "bias": float(model.intercept_[0]),
        **{name: float(weight) for name, weight in zip(FEATURES, model.coef_[0], strict=True)},
        "suggested_threshold": float(suggested_threshold),
        "target_error_rate": float(target_error_rate),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    """Fit the calibration artifact from the dev-only evaluation file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT_DIR / "eval" / "dev_predictions.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT_DIR / "eval" / "gate_weights.json")
    parser.add_argument("--target-error-rate", type=float, default=0.05)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    result = calibrate(args.input, args.output, args.target_error_rate)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
