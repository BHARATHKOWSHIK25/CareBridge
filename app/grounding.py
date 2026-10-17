"""Claim support verification against cited source passages."""

import re
from typing import Any

from app.llm import complete_json
from app.schemas import GroundingResult


def _local_grounding(payload: dict[str, Any]) -> dict[str, Any]:
    """Verify exact local extractive answers against their source text."""
    answer = str(payload.get("answer", "")).strip().casefold()
    cited_text = " ".join(str(chunk.get("text", "")) for chunk in payload.get("chunks", []))
    supported = bool(answer) and answer in cited_text.casefold()
    return {
        "supported_fraction": 1.0 if supported else 0.0,
        "unsupported_claims": [] if supported else [str(payload.get("answer", ""))],
    }


async def check_grounding(
    answer: str,
    cited_chunks: list[dict[str, Any]],
    *,
    stage_timeout: float | None = None,
) -> GroundingResult:
    """Ask the support judge whether answer claims follow from the cited text."""
    if not answer.strip():
        return GroundingResult(supported_fraction=0.0, unsupported_claims=[])
    return await complete_json(
        "Evaluate each factual claim in the answer against only these cited source passages. "
        "Return supported_fraction from 0 to 1 and unsupported_claims.",
        {"answer": answer, "chunks": cited_chunks},
        GroundingResult,
        temperature=0.0,
        local_handler=_local_grounding,
        stage_timeout=stage_timeout,
    )


def embedding_similarity(first: str, second: str) -> float:
    """Compute normalized semantic similarity, with a deterministic lexical fallback."""
    try:
        from sentence_transformers import SentenceTransformer

        from app.config import get_settings

        model = SentenceTransformer(get_settings().embedding_model, device="cpu")
        vectors = model.encode([first, second], normalize_embeddings=True)
        return min(1.0, max(0.0, float((vectors[0] * vectors[1]).sum())))
    except Exception:
        left = set(re.findall(r"[a-z0-9]+", first.casefold()))
        right = set(re.findall(r"[a-z0-9]+", second.casefold()))
        return len(left & right) / max(1, len(left | right))
