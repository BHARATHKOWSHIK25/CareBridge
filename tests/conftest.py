"""Shared offline test configuration and persistence fixtures."""

import os

os.environ.setdefault("EMBEDDINGS_ENABLED", "false")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("GATE_ENABLED", "true")
os.environ["LLM_BASE_URL"] = ""

import pytest
from sqlalchemy import delete

from app.db import SessionLocal, Ticket, init_db


@pytest.fixture(autouse=True)
def clean_database():
    """Create the schema and isolate ticket state for each test."""
    init_db()
    with SessionLocal() as session:
        session.execute(delete(Ticket))
        session.commit()
    yield
