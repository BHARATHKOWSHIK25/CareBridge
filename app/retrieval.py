"""Section-aware hybrid retrieval with strict document-level isolation."""

import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from app.config import get_settings

logger = logging.getLogger(__name__)
TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
WORD_TARGET = 200
WORD_OVERLAP = 30
CONTACT_QUERY_TERMS = {"call", "clinic", "contact", "number", "phone", "telephone"}


@dataclass(frozen=True)
class Chunk:
    """One source passage associated with exactly one synthetic document."""

    chunk_id: str
    doc_id: str
    section: str
    text: str


class HybridIndex:
    """In-memory BM25 plus optional sentence-embedding index."""

    def __init__(self, chunks: list[Chunk], *, use_embeddings: bool = True) -> None:
        """Create the in-memory index and attempt to load CPU embeddings."""
        self.chunks = chunks
        self._tokenized = [self._tokens(chunk.text) for chunk in chunks]
        self._bm25 = BM25Okapi(self._tokenized) if chunks else None
        self._model = None
        self._vectors = None
        if use_embeddings and chunks:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(get_settings().embedding_model, device="cpu")
                self._vectors = self._model.encode(
                    [chunk.text for chunk in chunks], normalize_embeddings=True
                )
            except Exception as exc:
                logger.warning(
                    "Sentence embeddings unavailable; using BM25-only retrieval: %s",
                    exc,
                )

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return TOKEN_RE.findall(text.lower())

    def retrieve(self, question_en: str, doc_id: str, k: int = 4) -> list[dict[str, object]]:
        """Return the top passages from the requested document only."""
        if k <= 0:
            return []
        eligible = [i for i, chunk in enumerate(self.chunks) if chunk.doc_id == doc_id]
        if not eligible:
            return []
        query_tokens = self._tokens(question_en)
        if self._bm25 is None:
            return []
        bm25_scores = self._bm25.get_scores(query_tokens)
        contact_intent = bool(set(query_tokens) & CONTACT_QUERY_TERMS)
        cosine_scores: dict[int, float] = {}
        if self._model is not None and self._vectors is not None:
            query_vector = self._model.encode([question_en], normalize_embeddings=True)[0]
            for idx in eligible:
                cosine_scores[idx] = max(0.0, float((self._vectors[idx] * query_vector).sum()))
        raw_bm25 = {idx: max(0.0, float(bm25_scores[idx])) for idx in eligible}
        scored: list[tuple[int, float]] = []
        for idx in eligible:
            bm25 = raw_bm25[idx] / (raw_bm25[idx] + 5.0) if raw_bm25[idx] else 0.0
            cosine = cosine_scores.get(idx, 0.0)
            score = 0.65 * bm25 + 0.35 * cosine if cosine_scores else bm25
            if contact_intent and self.chunks[idx].section.casefold() == "contacts":
                score += 0.5
            scored.append((idx, min(1.0, max(0.0, score))))
        scored.sort(key=lambda item: (-item[1], self.chunks[item[0]].chunk_id))
        return [
            {
                "chunk_id": self.chunks[idx].chunk_id,
                "doc_id": self.chunks[idx].doc_id,
                "section": self.chunks[idx].section,
                "text": self.chunks[idx].text,
                "score": score,
            }
            for idx, score in scored[:k]
        ]


def _load_document(path: Path) -> tuple[str, str]:
    """Read a document and return its metadata ID and markdown body."""
    text = path.read_text(encoding="utf-8")
    metadata = re.search(r"(?ms)^---\s*(.*?)\s*---", text)
    if metadata is None:
        raise ValueError(f"Document {path} is missing its metadata header.")
    doc_match = re.search(r"(?m)^doc_id:\s*([A-Za-z0-9_-]+)\s*$", metadata.group(1))
    if doc_match is None:
        raise ValueError(f"Document {path} is missing doc_id metadata.")
    return doc_match.group(1), text[metadata.end() :].strip()


def chunk_document(path: Path, target_words: int = WORD_TARGET) -> list[Chunk]:
    """Split a markdown document by heading, then into overlapping word windows."""
    doc_id, body = _load_document(path)
    sections: list[tuple[str, str]] = []
    section_name = "Introduction"
    section_lines: list[str] = []
    for line in body.splitlines():
        heading = re.match(r"^#{1,3}\s+(.+)$", line.strip())
        if heading:
            if section_lines:
                sections.append((section_name, "\n".join(section_lines).strip()))
            section_name = heading.group(1).strip()
            section_lines = []
        else:
            section_lines.append(line)
    if section_lines:
        sections.append((section_name, "\n".join(section_lines).strip()))
    chunks: list[Chunk] = []
    for section, text in sections:
        words = text.split()
        if not words:
            continue
        start = 0
        while start < len(words):
            end = min(len(words), start + target_words)
            excerpt = " ".join(words[start:end])
            chunk_id = f"{doc_id}-{len(chunks) + 1:03d}"
            chunks.append(Chunk(chunk_id, doc_id, section, excerpt))
            if end == len(words):
                break
            start = max(start + 1, end - WORD_OVERLAP)
    return chunks


def build_index(docs_dir: Path | None = None, *, use_embeddings: bool | None = None) -> HybridIndex:
    """Build the in-memory retrieval index from all markdown discharge sheets."""
    settings = get_settings()
    document_dir = docs_dir or settings.docs_dir
    files = sorted(document_dir.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"No discharge documents found in {document_dir}.")
    chunks = [chunk for path in files for chunk in chunk_document(path)]
    return HybridIndex(
        chunks,
        use_embeddings=settings.embeddings_enabled if use_embeddings is None else use_embeddings,
    )


_index: HybridIndex | None = None
_index_lock = threading.Lock()


def set_index(index: HybridIndex | None) -> None:
    """Set or clear the process-wide index, primarily for startup and tests."""
    global _index
    with _index_lock:
        _index = index


def get_index() -> HybridIndex:
    """Return the active index, building it lazily if startup has not done so."""
    global _index
    if _index is None:
        with _index_lock:
            if _index is None:
                _index = build_index()
    return _index


def retrieve(question_en: str, doc_id: str, k: int = 4) -> list[dict[str, object]]:
    """Retrieve passages for one document without exposing other documents."""
    return get_index().retrieve(question_en, doc_id, k)
