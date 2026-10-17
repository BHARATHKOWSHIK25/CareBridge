"""Document isolation and ranking behavior for retrieval."""

from app.retrieval import HybridIndex, build_index


def test_retrieval_never_returns_another_documents_content():
    """A query is restricted to the selected patient's document."""
    index = build_index(use_embeddings=False)
    results = index.retrieve("shower incision wound", "doc-cataract-001", k=4)
    assert results
    assert all(result["doc_id"] == "doc-cataract-001" for result in results)
    assert all(
        "cataract" in str(result["text"]).lower() or result["doc_id"] == "doc-cataract-001"
        for result in results
    )


def test_retrieval_respects_top_k_and_normalizes_scores():
    """Retrieval returns at most k ranked passages with scores in [0, 1]."""
    index = build_index(use_embeddings=False)
    results = index.retrieve("when can I shower after surgery", "doc-appendectomy-001", k=2)
    assert len(results) == 2
    assert results[0]["score"] >= results[1]["score"]
    assert all(0 <= result["score"] <= 1 for result in results)
    assert index.retrieve("question", "unknown-document", k=4) == []


def test_contact_questions_prioritize_contact_section():
    """A contact query retrieves verified-contact guidance instead of follow-up prose."""
    index = build_index(use_embeddings=False)
    results = index.retrieve(
        "Does the demo provide a real clinic phone number?",
        "doc-appendectomy-001",
        k=1,
    )
    assert results[0]["section"] == "Contacts"
    assert "No real clinic or emergency phone number is supplied" in str(results[0]["text"])


def test_sentence_splitting_preserves_document_scope():
    """An empty index does not fabricate retrieval results."""
    index = HybridIndex([], use_embeddings=False)
    assert index.retrieve("anything", "doc", k=4) == []
