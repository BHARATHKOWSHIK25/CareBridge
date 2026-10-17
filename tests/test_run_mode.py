"""Tests for RUN_MODE=live|demo, /api/about, and circuit breaker behavior."""

import pytest
from fastapi.testclient import TestClient

from app import main
from app.config import Settings
from app.llm import LLMError, circuit_breaker_state, complete_json, reset_circuit_breaker
from app.schemas import BenchmarkResponse


def test_demo_mode_chat_includes_run_mode():
    """In demo mode, every chat response carries run_mode='demo'."""
    with TestClient(main.app) as client:
        response = client.post(
            "/api/chat",
            json={
                "patient_id": "demo-appendectomy",
                "message": "Can I shower?",
                "session_id": "test-session",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["run_mode"] == "demo"
        assert data["status"] == "answered"


def test_about_endpoint_reports_status():
    """GET /api/about reports current run_mode, circuit breaker, and probe status."""
    with TestClient(main.app) as client:
        response = client.get("/api/about")
        assert response.status_code == 200
        data = response.json()
        assert "run_mode" in data
        assert "model" in data
        assert "llm_probe_ok" in data
        assert "amd_label_active" in data
        assert "circuit_breaker" in data
        # In demo mode, amd_label_active must be False
        assert data["amd_label_active"] is False


def test_live_mode_without_base_url_fails_startup(monkeypatch):
    """In live mode, startup fails with clear error if LLM_BASE_URL is missing."""
    monkeypatch.setattr(
        main,
        "get_settings",
        lambda: Settings(
            run_mode="live",
            llm_base_url=None,
            nurse_password="test-nurse-password",
        ),
    )
    with pytest.raises(RuntimeError, match="RUN_MODE=live requires LLM_BASE_URL"):
        with TestClient(main.app):
            pass


def test_live_mode_without_nurse_password_fails_startup(monkeypatch):
    """In live mode, startup fails if NURSE_PASSWORD is missing to protect nurse routes."""
    monkeypatch.setattr(
        main,
        "get_settings",
        lambda: Settings(
            run_mode="live",
            llm_base_url="http://fake-llm:8000/v1",
            nurse_password=None,
        ),
    )
    with pytest.raises(RuntimeError, match="RUN_MODE=live requires NURSE_PASSWORD"):
        with TestClient(main.app):
            pass


@pytest.mark.asyncio
async def test_live_mode_refuses_local_fallback(monkeypatch):
    """In live mode, complete_json raises LLMError rather than falling back to local extractor."""
    monkeypatch.setattr(
        "app.llm.get_settings",
        lambda: Settings(
            run_mode="live",
            llm_base_url=None,
        ),
    )
    with pytest.raises(LLMError, match="RUN_MODE=live but LLM_BASE_URL is not configured"):
        await complete_json(
            "test_task",
            {"message": "Can I shower?"},
            BenchmarkResponse,
            local_handler=lambda p: {"answer": "synthetic answer"},
        )


def test_circuit_breaker_trips_and_resets(monkeypatch):
    """Circuit breaker transitions to open after consecutive failures."""
    reset_circuit_breaker()
    assert circuit_breaker_state() == "closed"

    from app.llm import _cb_record_failure

    monkeypatch.setattr(
        "app.llm.get_settings",
        lambda: Settings(llm_circuit_breaker_threshold=3, llm_circuit_breaker_cooldown=30.0),
    )

    _cb_record_failure()
    assert circuit_breaker_state() == "closed"
    _cb_record_failure()
    assert circuit_breaker_state() == "closed"
    _cb_record_failure()
    assert circuit_breaker_state() == "open"

    # Reset recovers
    reset_circuit_breaker()
    assert circuit_breaker_state() == "closed"
