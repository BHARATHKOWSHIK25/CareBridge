"""Request and response models for the API and safety pipeline."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """A patient message attached to a known synthetic patient."""

    patient_id: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=1000)
    session_id: str = Field(min_length=1, max_length=100)


class Citation(BaseModel):
    """A retrieved source passage shown with an answer."""

    chunk_id: str
    section: str
    source_text: str
    translated_snippet: str | None = None


class ChatResult(BaseModel):
    """Typed outcome returned by the end-to-end chat pipeline."""

    status: Literal["answered", "escalated", "red_flag"]
    run_mode: Literal["live", "demo"] = "demo"
    language: str
    answer: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = 0.0
    breakdown: dict[str, float] = Field(default_factory=dict)
    ticket_id: str | None = None
    latency_ms: float = 0.0
    timings: dict[str, float] = Field(default_factory=dict)


class ResolveRequest(BaseModel):
    """A nurse's response when resolving a ticket."""

    nurse_reply: str = Field(min_length=1, max_length=4000)


class GeneratedAnswer(BaseModel):
    """Strict structured response from the answer generator."""

    answerable: bool
    answer: str
    citations: list[str]
    reasoning_note: str


class GroundingResult(BaseModel):
    """Structured support-judge result."""

    supported_fraction: float = Field(ge=0, le=1)
    unsupported_claims: list[str] = Field(default_factory=list)


class RedFlagClassification(BaseModel):
    """Structured red-flag classifier result."""

    red_flag: bool
    category: str
    confidence: float = Field(ge=0, le=1)


class TicketView(BaseModel):
    """Ticket payload exposed to the nurse UI."""

    id: str
    created_at: str
    priority: str
    patient_id: str
    language: str
    original_message: str
    english_message: str
    retrieved_chunks: list[dict[str, Any]]
    draft_answer: str | None
    confidence: float
    breakdown: dict[str, float]
    reason: str
    status: str
    nurse_reply: str | None


class BenchmarkResponse(BaseModel):
    """Structured response used by the startup probe and benchmark script."""

    answer: str

