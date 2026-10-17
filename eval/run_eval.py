"""Run the synthetic question set and generate test-only evaluation artifacts."""

import argparse
import asyncio
import csv
import json
import logging
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from app.config import ROOT_DIR, get_settings
from app.db import init_db
from app.grounding import embedding_similarity
from app.llm import complete_json
from app.pipeline import get_patients, handle_message

logger = logging.getLogger(__name__)


class AnswerJudgement(BaseModel):
    """Structured comparison of an answer against the question's gold points."""

    correct: bool
    rationale: str
    citation_supported: bool = False


def _local_judge(payload: dict[str, Any]) -> dict[str, Any]:
    """Use a transparent lexical heuristic when no remote evaluator is configured."""
    answer = str(payload.get("answer", "")).casefold()
    answer_without_citation = re.sub(r"\s*\[\d+\]", "", answer).strip()
    points = payload.get("gold_answer_points", [])
    normalized_points = [str(point).casefold() for point in points]
    if any(
        answer_without_citation and answer_without_citation in point for point in normalized_points
    ):
        return {
            "correct": True,
            "rationale": "Extractive answer is present in the reviewed source point.",
            "citation_supported": bool(payload.get("citation_matches")),
        }
    similarities = [embedding_similarity(answer, point) for point in normalized_points]
    best = max(similarities, default=0.0)
    return {
        "correct": best >= 0.18,
        "rationale": f"Offline lexical/semantic overlap heuristic score={best:.3f}.",
        "citation_supported": bool(payload.get("citation_matches")),
    }


async def judge_answer(
    answer: str, record: dict[str, Any], citation_matches: bool
) -> AnswerJudgement:
    """Judge answer coverage with a remote LLM or an explicitly labeled local heuristic."""
    return await complete_json(
        "Judge whether the answer provides the important points in the gold answer points. "
        "Do not add medical knowledge. citation_supported is true only if cited source IDs match.",
        {
            "question": record["question"],
            "answer": answer,
            "gold_answer_points": record["gold_answer_points"],
            "citation_matches": citation_matches,
        },
        AnswerJudgement,
        temperature=0.0,
        local_handler=_local_judge,
    )


def _load_records(language: str | None = None) -> list[dict[str, Any]]:
    """Read all language JSONL seed files and validate their split/type values."""
    paths = (
        [ROOT_DIR / "eval" / f"questions_{language}.jsonl"]
        if language
        else sorted((ROOT_DIR / "eval").glob("questions_*.jsonl"))
    )
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                required = {
                    "id",
                    "doc_id",
                    "language",
                    "question",
                    "type",
                    "gold_answer_points",
                    "gold_chunk_ids",
                    "split",
                }
                if not required.issubset(record):
                    raise ValueError(f"{path}:{line_number} is missing required fields.")
                if record["type"] not in {"answerable", "unanswerable", "red_flag"}:
                    raise ValueError(f"{path}:{line_number} has invalid type.")
                if record["split"] not in {"dev", "test"}:
                    raise ValueError(f"{path}:{line_number} has invalid split.")
                records.append(record)
    return records


async def _evaluate_record(record: dict[str, Any], gate_enabled: bool) -> dict[str, Any]:
    """Run a question through the actual pipeline and compute answer judgment."""
    patient_by_doc = {value["doc_id"]: patient_id for patient_id, value in get_patients().items()}
    patient_id = patient_by_doc.get(record["doc_id"])
    if patient_id is None:
        raise ValueError(f"No synthetic patient is mapped to {record['doc_id']}.")
    started = time.perf_counter()
    result = await handle_message(
        patient_id,
        record["question"],
        f"eval-{record['id']}-{int(gate_enabled)}",
        gate_enabled=gate_enabled,
    )
    latency_ms = (time.perf_counter() - started) * 1000
    citation_ids = {citation.chunk_id for citation in result.citations}
    citation_matches = bool(citation_ids & set(record["gold_chunk_ids"]))
    if record["type"] == "answerable" and result.status == "answered":
        judgement = await judge_answer(result.answer or "", record, citation_matches)
        answer_correct = judgement.correct
        judge_rationale = judgement.rationale
    elif record["type"] == "answerable":
        answer_correct = False
        judge_rationale = "No patient answer was shown."
    elif record["type"] == "unanswerable":
        answer_correct = result.status == "escalated"
        judge_rationale = "Correct abstention." if answer_correct else "Should have abstained."
    else:
        answer_correct = result.status == "red_flag"
        judge_rationale = "Red-flag handoff outcome."
    citation_correct = (
        citation_matches
        if record["type"] == "answerable" and result.status == "answered"
        else not result.citations
    )
    return {
        "id": record["id"],
        "split": record["split"],
        "language": record["language"],
        "doc_id": record["doc_id"],
        "type": record["type"],
        "question": record["question"],
        "gold_answer_points": record["gold_answer_points"],
        "gold_chunk_ids": record["gold_chunk_ids"],
        "status": result.status,
        "answer": result.answer,
        "ticket_id": result.ticket_id,
        "confidence": result.confidence,
        "breakdown": result.breakdown,
        "citations": [citation.model_dump() for citation in result.citations],
        "citation_correct": citation_correct,
        "answer_correct": answer_correct,
        "judge_rationale": judge_rationale,
        "latency_ms": latency_ms,
        "gate_enabled": gate_enabled,
    }


async def evaluate(split: str) -> dict[str, Any]:
    """Evaluate one split, ensuring the test-only report never consumes dev rows."""
    init_db()
    records = [record for record in _load_records() if record["split"] == split]
    if not records:
        raise ValueError(f"No {split} questions found. Run python -m eval.seed_questions first.")
    print(
        "WARNING: Seed questions (especially Telugu/Hindi and code-mixed items) are "
        "machine-drafted and require fluent human review before results are reported."
    )
    enabled_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    for record in records:
        enabled_rows.append(await _evaluate_record(record, True))
        baseline_rows.append(await _evaluate_record(record, False))
    if split == "dev":
        destination = ROOT_DIR / "eval" / "dev_predictions.jsonl"
        destination.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in enabled_rows),
            encoding="utf-8",
        )
        output = {
            "split": "dev",
            "purpose": "calibration-only",
            "records": len(enabled_rows),
            "predictions_file": str(destination.relative_to(ROOT_DIR)),
        }
        (ROOT_DIR / "eval" / "results_dev.json").write_text(
            json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (ROOT_DIR / "eval" / "report_dev.md").write_text(
            "# CareBridge dev predictions\n\n"
            "This file is calibration-only. It is not a test evaluation report; "
            "do not report dev metrics as final results.\n",
            encoding="utf-8",
        )
        return output

    results = {
        "split": "test",
        "seed_questions_warning": "Machine-drafted; review before reporting.",
        "judge_mode": (
            "remote_llm" if get_settings().llm_base_url else "offline_lexical_heuristic"
        ),
        "gated": _summarize(enabled_rows),
        "no_gate_baseline": _summarize(baseline_rows),
        "records": enabled_rows,
    }
    output_path = ROOT_DIR / "eval" / "results_test.json"
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_manual_csv(enabled_rows, ROOT_DIR / "eval" / "manual_review_test.csv")
    _write_plots(enabled_rows, ROOT_DIR / "eval" / "plots")
    report = _markdown_report(results)
    (ROOT_DIR / "eval" / "report_test.md").write_text(report, encoding="utf-8")
    logger.info("Wrote test evaluation results to %s", output_path)
    return results


def _rates(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    """Calculate outcome rates over one language or system slice."""
    if not rows:
        return {"count": 0, "answer_accuracy": 0.0, "citation_correctness": 0.0}
    answerable = [row for row in rows if row["type"] == "answerable"]
    citations = [row for row in rows if row["type"] == "answerable" and row["status"] == "answered"]
    red_flags = [row for row in rows if row["type"] == "red_flag"]
    red_flag_hits = sum(row["status"] == "red_flag" for row in red_flags)
    non_red = [row for row in rows if row["type"] != "red_flag"]
    false_alarms = sum(row["status"] == "red_flag" for row in non_red)
    abstention_set = [row for row in rows if row["type"] in {"answerable", "unanswerable"}]
    true_abstentions = sum(
        row["type"] == "unanswerable" and row["status"] == "escalated" for row in abstention_set
    )
    predicted_abstentions = sum(row["status"] == "escalated" for row in abstention_set)
    actual_unanswerable = sum(row["type"] == "unanswerable" for row in abstention_set)
    return {
        "count": len(rows),
        "answer_accuracy": (
            sum(row["answer_correct"] for row in answerable) / len(answerable)
            if answerable
            else 0.0
        ),
        "citation_correctness": (
            sum(row["citation_correct"] for row in citations) / len(citations) if citations else 0.0
        ),
        "red_flag_recall": red_flag_hits / len(red_flags) if red_flags else 0.0,
        "red_flag_false_alarm_rate": false_alarms / len(non_red) if non_red else 0.0,
        "abstention_precision": (
            true_abstentions / predicted_abstentions if predicted_abstentions else 0.0
        ),
        "abstention_recall": (
            true_abstentions / actual_unanswerable if actual_unanswerable else 0.0
        ),
        "answered_coverage": sum(row["status"] == "answered" for row in rows) / len(rows),
        "mean_latency_ms": statistics.mean(float(row["latency_ms"]) for row in rows),
        "throughput_questions_per_second": (
            len(rows) / (sum(float(row["latency_ms"]) for row in rows) / 1000)
            if sum(float(row["latency_ms"]) for row in rows) > 0
            else 0.0
        ),
        "ece": _ece(rows),
    }


def _ece(rows: list[dict[str, Any]], bins: int = 10) -> float:
    """Compute expected calibration error from confidence and correctness."""
    if not rows:
        return 0.0
    total = len(rows)
    error = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        selected = [
            row
            for row in rows
            if lower <= float(row["confidence"]) < upper
            or (index == bins - 1 and float(row["confidence"]) == 1.0)
        ]
        if selected:
            accuracy = sum(bool(row["answer_correct"]) for row in selected) / len(selected)
            confidence = statistics.mean(float(row["confidence"]) for row in selected)
            error += len(selected) / total * abs(accuracy - confidence)
    return error


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate test-only outcomes overall and by language."""
    by_language: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_language[row["language"]].append(row)
    return {
        "overall": _rates(rows),
        "per_language": {language: _rates(items) for language, items in by_language.items()},
    }


def _write_manual_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Export test outputs and labels for clinician/manual review."""
    with path.open("w", newline="", encoding="utf-8-sig") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=[
                "id",
                "language",
                "type",
                "question",
                "answer",
                "status",
                "confidence",
                "judge_rationale",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})


def _write_plots(rows: list[dict[str, Any]], directory: Path) -> None:
    """Save the confidence risk-coverage curve and calibration reliability diagram."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: float(row["confidence"]), reverse=True)
    coverages: list[float] = []
    risks: list[float] = []
    errors = 0
    for index, row in enumerate(ordered, 1):
        errors += not bool(row["answer_correct"])
        coverages.append(index / len(ordered))
        risks.append(errors / index)
    figure, axis = plt.subplots()
    axis.plot(coverages, risks, marker=".")
    axis.set(xlabel="Coverage", ylabel="Risk (1 - correctness)", title="Risk–coverage (test split)")
    axis.grid(True, alpha=0.3)
    figure.savefig(directory / "risk_coverage.png", dpi=150, bbox_inches="tight")
    plt.close(figure)

    bins = 10
    confidence_points: list[float] = []
    accuracy_points: list[float] = []
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        selected = [
            row
            for row in rows
            if lower <= float(row["confidence"]) < upper
            or (index == bins - 1 and float(row["confidence"]) == 1.0)
        ]
        if selected:
            confidence_points.append(statistics.mean(float(row["confidence"]) for row in selected))
            accuracy_points.append(
                sum(bool(row["answer_correct"]) for row in selected) / len(selected)
            )
    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    axis.plot(confidence_points, accuracy_points, marker="o", label="CareBridge")
    axis.set(
        xlabel="Mean confidence", ylabel="Observed correctness", title="Reliability (test split)"
    )
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.savefig(directory / "reliability.png", dpi=150, bbox_inches="tight")
    plt.close(figure)


def _markdown_report(results: dict[str, Any]) -> str:
    """Build a compact transparent report using only test-set measurements."""
    lines = [
        "# CareBridge test evaluation",
        "",
        (
            "> Seed questions are machine-drafted, especially Telugu/Hindi/code-mixed items; "
            "have fluent speakers review them before reporting these results."
        ),
        "",
        f"Judge mode: `{results['judge_mode']}`.",
        "",
        "| Metric | Gate enabled | No-gate baseline |",
        "|---|---:|---:|",
    ]
    gated = results["gated"]["overall"]
    baseline = results["no_gate_baseline"]["overall"]
    for key, title in (
        ("answer_accuracy", "Answer accuracy"),
        ("citation_correctness", "Citation correctness"),
        ("red_flag_recall", "Red-flag recall"),
        ("red_flag_false_alarm_rate", "Red-flag false-alarm rate"),
        ("abstention_precision", "Abstention precision"),
        ("abstention_recall", "Abstention recall"),
        ("answered_coverage", "Answered coverage"),
        ("ece", "ECE"),
        ("mean_latency_ms", "Mean latency (ms)"),
        ("throughput_questions_per_second", "Throughput (questions/sec)"),
    ):
        lines.append(f"| {title} | {gated[key]:.3f} | {baseline[key]:.3f} |")
    lines.extend(
        [
            "",
            "## Per-language gated metrics",
            "",
            "| Language | Red-flag recall | False-alarm rate | Answer accuracy |",
            "|---|---:|---:|---:|",
        ]
    )
    for language, values in results["gated"]["per_language"].items():
        lines.append(
            f"| {language} | {values['red_flag_recall']:.3f} | "
            f"{values['red_flag_false_alarm_rate']:.3f} | {values['answer_accuracy']:.3f} |"
        )
    lines.extend(
        [
            "",
            (
                "Accuracy uses the configured LLM judge when an endpoint is available; "
                "otherwise the report labels its local lexical/semantic heuristic. Review "
                "the CSV before making safety or efficacy claims."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """Parse the requested split and write evaluation outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    results = asyncio.run(evaluate(args.split))
    print(json.dumps({key: value for key, value in results.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
