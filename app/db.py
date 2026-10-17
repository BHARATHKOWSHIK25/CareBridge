"""SQLite persistence for nurse handoff tickets and benchmark measurements."""

import logging
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import get_settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""


class Ticket(Base):
    """A nurse review record with bilingual context and resolution history."""

    __tablename__ = "tickets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    priority: Mapped[str] = mapped_column(String(16), nullable=False)
    patient_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    language: Mapped[str] = mapped_column(String(8), nullable=False)
    original_message: Mapped[str] = mapped_column(Text, nullable=False)
    english_message: Mapped[str] = mapped_column(Text, nullable=False)
    retrieved_chunks: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False)
    draft_answer: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    breakdown: Mapped[dict[str, float]] = mapped_column(JSON, nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    nurse_reply: Mapped[str | None] = mapped_column(Text)


class Benchmark(Base):
    """Persisted endpoint benchmark measurements for the metrics panel."""

    __tablename__ = "benchmarks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    tokens_per_second: Mapped[float] = mapped_column(Float, nullable=False)


def _build_engine(url: str):
    """Create a SQLAlchemy engine, including isolated in-memory test support."""
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    if url in {"sqlite://", "sqlite:///:memory:"}:
        return create_engine(url, connect_args=connect_args, poolclass=StaticPool, future=True)
    return create_engine(url, connect_args=connect_args, future=True)


engine = _build_engine(get_settings().database_url)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db() -> None:
    """Create tables and the parent directory for the configured SQLite database."""
    if engine.url.database and engine.url.database != ":memory:":
        from pathlib import Path

        Path(engine.url.database).parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)


def ticket_to_dict(ticket: Ticket) -> dict[str, Any]:
    """Serialize a ticket without exposing SQLAlchemy internals."""
    created_at = ticket.created_at
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    return {
        "id": ticket.id,
        "created_at": created_at.isoformat(),
        "priority": ticket.priority,
        "patient_id": ticket.patient_id,
        "language": ticket.language,
        "original_message": ticket.original_message,
        "english_message": ticket.english_message,
        "retrieved_chunks": ticket.retrieved_chunks,
        "draft_answer": ticket.draft_answer,
        "confidence": ticket.confidence,
        "breakdown": ticket.breakdown,
        "reason": ticket.reason,
        "status": ticket.status,
        "nurse_reply": ticket.nurse_reply,
    }


def save_benchmark(latency_ms: float, tokens_per_second: float) -> None:
    """Persist one measured remote-model benchmark."""
    with SessionLocal() as session:
        session.add(
            Benchmark(
                created_at=datetime.now(UTC),
                latency_ms=latency_ms,
                tokens_per_second=tokens_per_second,
            )
        )
        session.commit()


def latest_benchmark() -> dict[str, float] | None:
    """Return the most recently persisted benchmark, when present."""
    with SessionLocal() as session:
        row = session.scalar(select(Benchmark).order_by(Benchmark.id.desc()).limit(1))
        if row is None:
            return None
        return {
            "latency_ms": row.latency_ms,
            "tokens_per_second": row.tokens_per_second,
        }
