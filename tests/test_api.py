"""FastAPI route, rate limit, and nurse authentication tests."""

from types import SimpleNamespace

from fastapi.testclient import TestClient

from app import main


def test_api_routes_and_ticket_lifecycle():
    """Health, patient, chat, nurse queue, resolution, metrics, and pages work."""
    main._request_times.clear()
    with TestClient(main.app) as client:
        assert client.get("/health").status_code == 200
        assert len(client.get("/api/patients").json()) == 3
        assert client.get("/").status_code == 200
        assert client.get("/nurse").status_code == 200
        chat = client.post(
            "/api/chat",
            json={
                "patient_id": "demo-appendectomy",
                "message": "When can I restart my tablets?",
                "session_id": "api-session",
            },
        )
        assert chat.status_code == 200
        result = chat.json()
        assert result["status"] == "escalated"
        ticket_id = result["ticket_id"]
        assert client.get("/api/tickets?status=open").json()
        assert client.get(f"/api/tickets/{ticket_id}").status_code == 200
        resolved = client.post(
            f"/api/tickets/{ticket_id}/resolve", json={"nurse_reply": "Please call your clinic."}
        )
        assert resolved.status_code == 200
        assert resolved.json()["status"] == "resolved"
        assert client.get("/api/tickets?status=resolved").json()
        assert client.get("/api/metrics").status_code == 200


def test_chat_validation_and_unknown_patient():
    """Oversized, empty, or unknown-patient requests are rejected."""
    with TestClient(main.app) as client:
        assert client.post("/api/chat", json={"patient_id": "x"}).status_code == 422
        response = client.post(
            "/api/chat",
            json={"patient_id": "unknown", "message": "Hello?", "session_id": "x"},
        )
        assert response.status_code == 404
        assert (
            client.post(
                "/api/chat",
                json={"patient_id": "demo-knee", "message": "x" * 1001, "session_id": "x"},
            ).status_code
            == 422
        )


def test_rate_limit_and_nurse_auth(monkeypatch):
    """The sliding-window limiter and optional Basic auth guard are enforced."""
    monkeypatch.setattr(
        main,
        "get_settings",
        lambda: SimpleNamespace(requests_per_minute=1, nurse_password="secret"),
    )
    main._request_times.clear()
    with TestClient(main.app) as client:
        payload = {
            "patient_id": "demo-knee",
            "message": "When do I restart my tablets?",
            "session_id": "rate-test",
        }
        assert client.post("/api/chat", json=payload).status_code == 200
        assert client.post("/api/chat", json=payload).status_code == 429
        assert client.get("/api/tickets").status_code == 401
        assert client.get("/api/tickets", auth=("nurse", "secret")).status_code == 200
