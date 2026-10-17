"""Single adapter for remote OpenAI-compatible model calls and local demo mode."""

import asyncio
import json
import logging
import random
import threading
import time
from collections.abc import Callable
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from app.config import get_settings

logger = logging.getLogger(__name__)
_usage_lock = threading.Lock()
_last_usage: dict[str, float] = {
    "completion_tokens": 0.0,
    "latency_ms": 0.0,
    "tokens_per_second": 0.0,
}

# ---------------------------------------------------------------------------
# Circuit breaker (process-wide, thread-safe)
# ---------------------------------------------------------------------------
_cb_lock = threading.Lock()
_cb_failure_count: int = 0
_cb_opened_at: float | None = None  # monotonic time when the breaker opened


def _cb_state() -> str:
    """Return 'closed', 'open', or 'half-open'."""
    settings = get_settings()
    with _cb_lock:
        if _cb_opened_at is None:
            return "closed"
        elapsed = time.monotonic() - _cb_opened_at
        if elapsed >= settings.llm_circuit_breaker_cooldown:
            return "half-open"
        return "open"


def _cb_record_success() -> None:
    global _cb_failure_count, _cb_opened_at
    with _cb_lock:
        _cb_failure_count = 0
        _cb_opened_at = None


def _cb_record_failure() -> None:
    global _cb_failure_count, _cb_opened_at
    settings = get_settings()
    with _cb_lock:
        _cb_failure_count += 1
        if _cb_failure_count >= settings.llm_circuit_breaker_threshold and _cb_opened_at is None:
            _cb_opened_at = time.monotonic()
            logger.error(
                "LLM circuit breaker opened after %d consecutive failures.",
                _cb_failure_count,
            )


def circuit_breaker_state() -> str:
    """Expose circuit-breaker state for /api/about and tests."""
    return _cb_state()


def reset_circuit_breaker() -> None:
    """Reset the breaker; used in tests and at startup after a successful probe."""
    global _cb_failure_count, _cb_opened_at
    with _cb_lock:
        _cb_failure_count = 0
        _cb_opened_at = None


# ---------------------------------------------------------------------------


class LLMError(RuntimeError):
    """Raised when the configured language-model endpoint cannot serve a request."""


LocalHandler = Callable[[dict[str, Any]], dict[str, Any]]


def last_call_usage() -> dict[str, float]:
    """Return measured token and latency data for the most recent remote call."""
    with _usage_lock:
        return _last_usage.copy()


async def complete_json(
    task: str,
    payload: dict[str, Any],
    response_model: type[BaseModel],
    *,
    temperature: float = 0.0,
    local_handler: LocalHandler | None = None,
    stage_timeout: float | None = None,
) -> BaseModel:
    """Call the configured chat-completions endpoint and validate its JSON response.

    In demo mode (RUN_MODE=demo) and when no endpoint is configured, an explicitly
    supplied deterministic local handler keeps the synthetic demo usable without
    network access.

    In live mode (RUN_MODE=live), local handlers are NEVER invoked — a missing or
    failing endpoint raises LLMError so the pipeline escalates correctly.

    stage_timeout overrides llm_timeout_seconds for a single call (0 = no override).
    """
    settings = get_settings()

    # ---- Demo mode: local fallback ----
    if not settings.llm_base_url:
        if settings.is_live:
            raise LLMError(
                "RUN_MODE=live but LLM_BASE_URL is not configured. "
                "Set LLM_BASE_URL or switch to RUN_MODE=demo."
            )
        if local_handler is None:
            raise LLMError("No LLM_BASE_URL is configured and no local fallback is available.")
        try:
            return response_model.model_validate(local_handler(payload))
        except (ValidationError, ValueError, TypeError) as exc:
            raise LLMError(f"Local {task} fallback returned invalid data.") from exc

    # ---- Live mode: circuit breaker check ----
    state = _cb_state()
    if state == "open":
        raise LLMError("LLM circuit breaker is open; endpoint appears unavailable.")

    effective_timeout = (
        stage_timeout
        if stage_timeout and stage_timeout > 0
        else settings.llm_timeout_seconds
    )

    client = AsyncOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key or "carebridge-demo-key",
        timeout=effective_timeout,
        max_retries=0,
    )
    system_prompt = (
        "You are a component of CareBridge, a safety-first discharge-document assistant. "
        "Never diagnose, recommend treatment, or change medication instructions. "
        "Use only the supplied synthetic document context. Return one JSON object matching "
        f"this schema: {json.dumps(response_model.model_json_schema())}. Task: {task}"
    )
    user_content = json.dumps(payload, ensure_ascii=False)
    last_error: Exception | None = None
    network_retries = 0
    parse_retries = 0
    try:
        while network_retries <= settings.llm_max_retries:
            started = time.perf_counter()
            try:
                response = await client.chat.completions.create(
                    model=settings.llm_model,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                )
                elapsed_ms = (time.perf_counter() - started) * 1000
                completion_tokens = (
                    float(response.usage.completion_tokens) if response.usage else 0.0
                )
                with _usage_lock:
                    _last_usage.update(
                        completion_tokens=completion_tokens,
                        latency_ms=elapsed_ms,
                        tokens_per_second=(
                            completion_tokens / (elapsed_ms / 1000) if elapsed_ms else 0.0
                        ),
                    )
                content = response.choices[0].message.content or ""
                try:
                    data = json.loads(content)
                    result = response_model.model_validate(data)
                    _cb_record_success()
                    return result
                except (json.JSONDecodeError, ValidationError) as exc:
                    last_error = exc
                    if parse_retries >= 1:
                        _cb_record_failure()
                        raise LLMError(f"LLM task {task!r} returned invalid JSON twice.") from exc
                    parse_retries += 1
                    logger.warning("LLM returned invalid JSON for task %s; retrying.", task)
            except Exception as exc:
                if isinstance(exc, LLMError):
                    raise
                last_error = exc
                _cb_record_failure()
                if network_retries == settings.llm_max_retries:
                    break
                await asyncio.sleep(min(0.25 * (2**network_retries) + random.random() * 0.1, 2))
                network_retries += 1
    finally:
        await client.close()
    raise LLMError(f"LLM task {task!r} failed after configured retries.") from last_error
