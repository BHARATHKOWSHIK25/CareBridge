"""FastAPI application exposing CareBridge patient and nurse workflows."""

import hmac
import logging
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles

from app.config import ROOT_DIR, get_settings
from app.db import init_db
from app.handoff import get_ticket, list_tickets, resolve_ticket
from app.llm import LLMError, circuit_breaker_state, complete_json, reset_circuit_breaker
from app.metrics import metrics
from app.pipeline import get_patients, handle_message
from app.redflag import load_rules
from app.retrieval import build_index, set_index
from app.schemas import BenchmarkResponse, ChatRequest, ChatResult, ResolveRequest

logger = logging.getLogger(__name__)
security = HTTPBasic(auto_error=False)
_request_times: dict[str, deque[float]] = defaultdict(deque)
_rate_limit_lock = threading.Lock()

# Set by the startup probe; True = endpoint answered a real call.
_llm_probe_ok: bool = False


async def _probe_llm() -> bool:
    """Send a minimal call to the configured endpoint; update circuit breaker on success."""
    settings = get_settings()
    llm_base_url = getattr(settings, "llm_base_url", None)
    if not llm_base_url:
        return False
    try:
        await complete_json(
            "Startup probe. Return JSON with field answer='ok'.",
            {"probe": True},
            BenchmarkResponse,
            temperature=0.0,
        )
        reset_circuit_breaker()
        model_name = getattr(settings, "llm_model", "unknown")
        logger.info("LLM startup probe succeeded (model=%s).", model_name)
        return True
    except LLMError as exc:
        logger.warning("LLM startup probe failed: %s", exc)
        return False


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Initialize persistence, safety rules, and the in-memory document index."""
    global _llm_probe_ok
    settings = get_settings()

    is_live = getattr(settings, "is_live", False) or getattr(settings, "run_mode", "demo") == "live"
    llm_base_url = getattr(settings, "llm_base_url", None)

    # Fail fast in live mode if no endpoint or nurse password is configured.
    if is_live and not llm_base_url:
        raise RuntimeError(
            "RUN_MODE=live requires LLM_BASE_URL to be set. "
            "Either set the endpoint or switch to RUN_MODE=demo."
        )
    if is_live and not getattr(settings, "nurse_password", None):
        raise RuntimeError(
            "RUN_MODE=live requires NURSE_PASSWORD to protect nurse routes. "
            "Set NURSE_PASSWORD in .env or switch to RUN_MODE=demo."
        )

    init_db()
    load_rules()
    set_index(build_index())
    get_patients()

    # Best-effort probe; does not block startup but records readiness.
    _llm_probe_ok = await _probe_llm()
    if is_live and not _llm_probe_ok:
        logger.warning(
            "RUN_MODE=live but LLM startup probe failed. "
            "Chat requests will escalate until the endpoint is reachable."
        )
    yield


app = FastAPI(title="CareBridge", version="1.0.0", lifespan=lifespan)
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

ui_dir = ROOT_DIR / "ui"
app.mount("/static", StaticFiles(directory=ui_dir), name="static")


def _nurse_auth(
    credentials: Annotated[HTTPBasicCredentials | None, Depends(security)],
) -> None:
    """Protect nurse routes; refuse access in live mode if NURSE_PASSWORD is not set."""
    curr_settings = get_settings()
    password = getattr(curr_settings, "nurse_password", None)
    is_live = (
        getattr(curr_settings, "is_live", False)
        or getattr(curr_settings, "run_mode", "demo") == "live"
    )
    if is_live and not password:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Nurse routes are unavailable in live mode without NURSE_PASSWORD.",
        )
    if not password:
        return
    if (
        credentials is None
        or not hmac.compare_digest(credentials.username.encode("utf-8"), b"nurse")
        or not hmac.compare_digest(credentials.password.encode("utf-8"), password.encode("utf-8"))
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Nurse authentication required.",
            headers={"WWW-Authenticate": "Basic"},
        )


def _check_rate_limit(request: Request) -> None:
    """Apply a per-client in-memory sliding-window rate limit."""
    now = time.monotonic()
    key = request.client.host if request.client else "unknown"
    with _rate_limit_lock:
        queue = _request_times[key]
        while queue and now - queue[0] >= 60:
            queue.popleft()
        if len(queue) >= get_settings().requests_per_minute:
            raise HTTPException(status_code=429, detail="Chat rate limit exceeded; retry shortly.")
        queue.append(now)


@app.get("/health")
async def health() -> dict[str, str]:
    """Report service liveness and readiness."""
    return {"status": "ok", "service": "CareBridge"}


@app.get("/api/about")
async def about() -> dict[str, object]:
    """Return run-mode, model identity, and AMD serving status.

    The AMD label is only shown when RUN_MODE=live and the startup probe succeeded.
    """
    s = get_settings()
    is_live = getattr(s, "is_live", False) or getattr(s, "run_mode", "demo") == "live"
    live_and_ready = is_live and _llm_probe_ok
    amd_served = getattr(s, "amd_served", False)
    return {
        "run_mode": getattr(s, "run_mode", "demo"),
        "model": getattr(s, "llm_model", "Qwen2.5-7B-Instruct"),
        "llm_base_url_configured": bool(getattr(s, "llm_base_url", None)),
        "llm_probe_ok": _llm_probe_ok,
        "amd_served": amd_served if live_and_ready else False,
        "amd_label_active": live_and_ready and amd_served,
        "circuit_breaker": circuit_breaker_state(),
        "version": "1.0.0",
    }


@app.get("/api/patients")
async def patients() -> list[dict[str, str]]:
    """Return synthetic patient selector data without exposing document text."""
    return list(get_patients().values())


@app.post("/api/chat")
async def chat(
    body: ChatRequest,
    _: None = Depends(_check_rate_limit),
) -> ChatResult:
    """Handle one patient question through the safety pipeline."""
    try:
        return await handle_message(body.patient_id, body.message, body.session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Unknown demo patient.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Chat request failed.")
        raise HTTPException(
            status_code=500, detail="The request could not be processed safely."
        ) from exc


@app.get("/api/tickets", dependencies=[Depends(_nurse_auth)])
async def tickets(status: str | None = None) -> list[dict[str, object]]:
    """List nurse tickets, optionally filtered by status=open or resolved."""
    selected = status
    if selected not in {None, "open", "resolved"}:
        raise HTTPException(status_code=422, detail="status must be open or resolved.")
    return list_tickets(selected)


@app.get("/api/tickets/{ticket_id}", dependencies=[Depends(_nurse_auth)])
async def ticket_detail(ticket_id: str) -> dict[str, object]:
    """Return complete nurse context for one ticket."""
    ticket = get_ticket(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="Ticket not found.")
    return ticket


@app.post("/api/tickets/{ticket_id}/resolve", dependencies=[Depends(_nurse_auth)])
async def ticket_resolve(ticket_id: str, body: ResolveRequest) -> dict[str, object]:
    """Resolve a ticket and store the nurse reply."""
    ticket = resolve_ticket(ticket_id, body.nurse_reply)
    if ticket is None:
        raise HTTPException(status_code=404, detail="Ticket not found.")
    return ticket


@app.get("/api/metrics")
async def metrics_endpoint() -> dict[str, object]:
    """Return live throughput, model, and request outcome metrics."""
    return metrics.snapshot()


@app.get("/")
async def patient_page():
    """Serve the patient chat page."""
    return FileResponse(ui_dir / "index.html")


@app.get("/nurse")
async def nurse_page(_: None = Depends(_nurse_auth)):
    """Serve the nurse ticket queue."""
    return FileResponse(ui_dir / "nurse.html")
