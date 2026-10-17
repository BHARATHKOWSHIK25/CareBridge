"""Measure configured endpoint latency and completion throughput.

Results are saved to eval/benchmarks/ with a timestamped filename.
The script requires RUN_MODE=live (or an explicit LLM_BASE_URL) so accidental
runs against the local fallback are prevented.
"""

import argparse
import asyncio
import json
import logging
import time
from datetime import UTC, datetime

from app.config import ROOT_DIR, get_settings
from app.db import init_db, save_benchmark
from app.llm import complete_json, last_call_usage
from app.metrics import metrics
from app.schemas import BenchmarkResponse

logger = logging.getLogger(__name__)

PROMPTS = [
    "Return JSON with a concise answer: What does this synthetic demo measure?",
    "Return JSON with a concise answer: Why should an unsupported patient question escalate?",
    "Return JSON with a concise answer: What does document isolation protect?",
    "Return JSON with a concise answer: Name one safety benefit of grounding scores.",
    "Return JSON with a concise answer: What is the purpose of the confidence gate?",
]

BENCHMARKS_DIR = ROOT_DIR / "eval" / "benchmarks"


async def run_benchmark(warmup: int = 1, prompts: list[str] | None = None) -> dict[str, object]:
    """Send a fixed prompt set to the configured model and persist real measurements.

    Args:
        warmup: Number of warm-up calls to discard before measuring.
        prompts: Override the default prompt list (used in tests).
    """
    settings = get_settings()
    if not settings.llm_base_url:
        raise RuntimeError(
            "LLM_BASE_URL is not configured. "
            "Set LLM_BASE_URL and LLM_MODEL in .env before running the benchmark."
        )
    init_db()
    active_prompts = prompts or PROMPTS

    # Warm-up (discarded)
    for i in range(warmup):
        try:
            await complete_json(
                "Benchmark warm-up call. Return JSON with field answer='ok'.",
                {"warmup": i},
                BenchmarkResponse,
                temperature=0.0,
            )
            logger.info("Warm-up call %d complete.", i + 1)
        except Exception as exc:
            logger.warning("Warm-up call %d failed: %s", i + 1, exc)

    measurements = []
    for prompt in active_prompts:
        started = time.perf_counter()
        try:
            await complete_json(
                "Benchmark only. Return a brief answer in the required JSON schema.",
                {"prompt": prompt},
                BenchmarkResponse,
                temperature=0.0,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000
            usage = last_call_usage()
            measurements.append(
                {
                    "prompt_prefix": prompt[:60],
                    "latency_ms": elapsed_ms,
                    "completion_tokens": usage["completion_tokens"],
                    "tokens_per_second": usage["tokens_per_second"],
                }
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000
            logger.error("Benchmark prompt failed: %s", exc)
            measurements.append(
                {
                    "prompt_prefix": prompt[:60],
                    "latency_ms": elapsed_ms,
                    "completion_tokens": 0,
                    "tokens_per_second": 0,
                    "error": str(exc),
                }
            )

    successful = [m for m in measurements if "error" not in m]
    if not successful:
        raise RuntimeError("All benchmark prompts failed. Check LLM endpoint configuration.")

    average_latency = sum(float(row["latency_ms"]) for row in successful) / len(successful)
    token_count = sum(float(row["completion_tokens"]) for row in successful)
    duration_s = sum(float(row["latency_ms"]) for row in successful) / 1000
    throughput = token_count / duration_s if duration_s else 0.0
    p50_latency = sorted(float(m["latency_ms"]) for m in successful)[len(successful) // 2]
    p95_latency = sorted(float(m["latency_ms"]) for m in successful)[
        min(len(successful) - 1, int(0.95 * len(successful)))
    ]

    save_benchmark(average_latency, throughput)
    metrics.set_tokens_per_second(throughput)

    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    result: dict[str, object] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "model": settings.llm_model,
        "base_url": settings.llm_base_url,
        "amd_served": settings.amd_served,
        "run_mode": settings.run_mode,
        "warmup_calls": warmup,
        "prompts_attempted": len(measurements),
        "prompts_succeeded": len(successful),
        "measurements": measurements,
        "summary": {
            "average_latency_ms": average_latency,
            "p50_latency_ms": p50_latency,
            "p95_latency_ms": p95_latency,
            "completion_tokens_per_second": throughput,
        },
        "source": "OpenAI-compatible endpoint usage metadata and wall-clock timings",
    }
    filename = f"benchmark_{timestamp}.json"
    out_path = BENCHMARKS_DIR / filename
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Benchmark saved to %s", out_path)
    return result


def main() -> None:
    """Run the endpoint benchmark and print the measured summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warm-up calls to discard before measuring (default: 1).",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = asyncio.run(run_benchmark(warmup=args.warmup))
    # Print only the summary section for readability.
    print(json.dumps({"model": result["model"], "summary": result["summary"]}, indent=2))
    print(f"\nFull results: {BENCHMARKS_DIR / 'benchmark_*.json'}")


if __name__ == "__main__":
    main()
