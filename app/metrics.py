"""In-memory request counters and latency measurements."""

import threading
import time
from collections import Counter, deque
from typing import Any

from app.config import get_settings
from app.db import latest_benchmark


class Metrics:
    """Thread-safe process metrics for the live demo panel."""

    def __init__(self) -> None:
        """Initialize empty counters and bounded latency history."""
        self._lock = threading.Lock()
        self._counts: Counter[str] = Counter()
        self._latencies: deque[float] = deque(maxlen=1000)
        self._started = time.monotonic()
        self._tokens_per_second = 0.0

    def record(self, status: str, latency_ms: float) -> None:
        """Record one completed chat outcome and its elapsed time."""
        with self._lock:
            self._counts[status] += 1
            self._latencies.append(latency_ms)

    def set_tokens_per_second(self, value: float) -> None:
        """Store the latest measured generation throughput."""
        with self._lock:
            self._tokens_per_second = max(0.0, value)

    def snapshot(self) -> dict[str, Any]:
        """Return aggregate counters, latency percentiles, and model labels."""
        with self._lock:
            values = sorted(self._latencies)
            counts = dict(self._counts)
            average = sum(values) / len(values) if values else 0.0
            p95 = values[min(len(values) - 1, int(0.95 * len(values)))] if values else 0.0
            throughput = self._tokens_per_second
        benchmark = latest_benchmark()
        if benchmark:
            throughput = benchmark["tokens_per_second"]
        settings = get_settings()
        elapsed = max(1.0, time.monotonic() - self._started)
        return {
            "model": settings.llm_model,
            "served_on_amd": settings.amd_served,
            "device": (
                "remote AMD GPU (configured)"
                if settings.amd_served
                else "remote endpoint; hardware not asserted"
            ),
            "backend": settings.backend,
            "tokens_per_second": throughput,
            "avg_latency_ms": average,
            "p95_latency_ms": p95,
            "counts": {
                "answered": counts.get("answered", 0),
                "escalated": counts.get("escalated", 0),
                "red_flag": counts.get("red_flag", 0),
            },
            "requests_per_second": sum(counts.values()) / elapsed,
        }


metrics = Metrics()
