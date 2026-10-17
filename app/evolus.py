"""Local and Evolus knowledge/ticket adapter boundary."""

import logging
from typing import Any, Protocol

import httpx

from app.config import get_settings
from app.retrieval import retrieve

logger = logging.getLogger(__name__)

# FILL FROM EVOLUS DOCS: endpoint paths and request/response schemas are intentionally unknown.
EVOLUS_SEARCH_KNOWLEDGE_PATH = ""
# FILL FROM EVOLUS DOCS: do not send a ticket payload until its schema is confirmed.
EVOLUS_CREATE_TICKET_PATH = ""


class Backend(Protocol):
    """Operations required by the CareBridge handoff pipeline."""

    def search_knowledge(self, doc_id: str, query: str) -> list[dict[str, object]]:
        """Search for document-scoped passages."""

    def create_ticket_workflow(self, payload: dict[str, Any]) -> str:
        """Create a nurse ticket and return its local tracking identifier."""


class LocalBackend:
    """Fully working local document search and SQLite ticket persistence."""

    def search_knowledge(self, doc_id: str, query: str) -> list[dict[str, object]]:
        """Return passages only from the requested discharge document."""
        return retrieve(query, doc_id, k=4)

    def create_ticket_workflow(self, payload: dict[str, Any]) -> str:
        """Persist a ticket locally and return its identifier."""
        from app.handoff import create_local_ticket

        return create_local_ticket(payload)


class EvolusBackend:
    """Evolus adapter boundary that safely falls back until vendor docs are supplied."""

    def __init__(self, local: Backend | None = None) -> None:
        """Read endpoint credentials and retain the functional local backend."""
        settings = get_settings()
        self.base_url = settings.evolus_base_url
        self.api_key = settings.evolus_api_key
        self.workspace_id = settings.evolus_workspace_id
        self.local = local or LocalBackend()

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        """Make an HTTP request only after documented path and schema are supplied."""
        if not all((self.base_url, self.api_key, self.workspace_id, path)):
            raise RuntimeError("Evolus settings or documented endpoint paths are incomplete.")
        raise NotImplementedError(
            "Evolus request and response schemas must be implemented from vendor documentation."
        )

    def search_knowledge(self, doc_id: str, query: str) -> list[dict[str, object]]:
        """Use Evolus knowledge only when its documented path/schema are implemented."""
        try:
            return self._request(
                "GET",
                EVOLUS_SEARCH_KNOWLEDGE_PATH,
                {"doc_id": doc_id, "query": query},
            )
        except (RuntimeError, NotImplementedError, httpx.HTTPError):
            logger.warning("Evolus search unavailable; using local document index.", exc_info=True)
            return self.local.search_knowledge(doc_id, query)

    def create_ticket_workflow(self, payload: dict[str, Any]) -> str:
        """Fall back to SQLite until Evolus's ticket workflow contract is documented."""
        try:
            self._request("POST", EVOLUS_CREATE_TICKET_PATH, payload)
        except (RuntimeError, NotImplementedError, httpx.HTTPError):
            logger.warning("Evolus ticket workflow unavailable; using local SQLite.", exc_info=True)
            return self.local.create_ticket_workflow(payload)
        return self.local.create_ticket_workflow(payload)


def get_backend() -> Backend:
    """Select the configured backend, defaulting to the local implementation."""
    return EvolusBackend() if get_settings().backend.lower() == "evolus" else LocalBackend()
