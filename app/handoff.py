"""Nurse ticket creation and resolution."""

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select

from app.db import SessionLocal, Ticket, ticket_to_dict
from app.evolus import get_backend


def create_local_ticket(payload: dict[str, Any]) -> str:
    """Write a ticket to SQLite and return its UUID."""
    ticket_id = str(uuid.uuid4())
    ticket = Ticket(
        id=ticket_id,
        created_at=datetime.now(UTC),
        priority=str(payload["priority"]),
        patient_id=str(payload["patient_id"]),
        language=str(payload["language"]),
        original_message=str(payload["original_message"]),
        english_message=str(payload["english_message"]),
        retrieved_chunks=payload.get("retrieved_chunks", []),
        draft_answer=payload.get("draft_answer"),
        confidence=float(payload.get("confidence", 0.0)),
        breakdown=payload.get("breakdown", {}),
        reason=str(payload["reason"]),
        status="open",
        nurse_reply=None,
    )
    with SessionLocal() as session:
        session.add(ticket)
        session.commit()
    return ticket_id


def create_ticket(payload: dict[str, Any]) -> str:
    """Create a ticket through the selected integration with local fallback."""
    return get_backend().create_ticket_workflow(payload)


def list_tickets(status: str | None = None) -> list[dict[str, Any]]:
    """List tickets filtered to open/resolved, ordered newest first."""
    with SessionLocal() as session:
        query = select(Ticket).order_by(Ticket.created_at.desc())
        if status is not None:
            query = query.where(Ticket.status == status)
        return [ticket_to_dict(ticket) for ticket in session.scalars(query).all()]


def get_ticket(ticket_id: str) -> dict[str, Any] | None:
    """Return one nurse ticket or None when it does not exist."""
    with SessionLocal() as session:
        ticket = session.get(Ticket, ticket_id)
        return ticket_to_dict(ticket) if ticket else None


def resolve_ticket(ticket_id: str, nurse_reply: str) -> dict[str, Any] | None:
    """Mark a ticket resolved with the nurse's non-empty reply."""
    with SessionLocal() as session:
        ticket = session.get(Ticket, ticket_id)
        if ticket is None:
            return None
        ticket.status = "resolved"
        ticket.nurse_reply = nurse_reply.strip()
        session.commit()
        session.refresh(ticket)
        return ticket_to_dict(ticket)
