import pytest

from app.chunking import Chunk
from app.embeddings import DeterministicEmbeddingProvider
from app.hybrid_retrieval import HybridRetriever
from app.retrieval import LexicalRetriever
from app.storage import ChunkStore
from app.vector_store import VectorStore


def build_retriever(tmp_path):
    db_path = tmp_path / "hybrid.db"

    chunk_store = ChunkStore(db_path)

    chunks = [
        Chunk(
            chunk_id="doc:1:0",
            document="doc.pdf",
            page_number=1,
            text="Python security engineering",
        ),
        Chunk(
            chunk_id="doc:1:1",
            document="doc.pdf",
            page_number=1,
            text="Smart contract auditing",
        ),
        Chunk(
            chunk_id="doc:1:2",
            document="doc.pdf",
            page_number=1,
            text="Machine learning systems",
        ),
    ]

    chunk_store.add_chunks(chunks)

    lexical = LexicalRetriever(db_path)

    provider = DeterministicEmbeddingProvider(dimensions=8)
    vector_store = VectorStore(
        db_path,
        dimensions=provider.dimensions,
    )

    for chunk in chunks:
        vector_store.add_embedding(
            chunk.chunk_id,
            provider.embed(chunk.text),
        )

    hybrid = HybridRetriever(
        lexical_retriever=lexical,
        vector_store=vector_store,
        embedding_provider=provider,
        chunk_store=chunk_store,
    )

    return hybrid


def test_hybrid_search_returns_full_chunk_metadata(tmp_path):
    retriever = build_retriever(tmp_path)

    results = retriever.search("Python security", limit=3)

    assert results
    assert results[0].chunk_id == "doc:1:0"
    assert results[0].document == "doc.pdf"
    assert results[0].page_number == 1
    assert results[0].text == "Python security engineering"
    assert results[0].score > 0


def test_hybrid_search_merges_lexical_and_semantic_candidates(tmp_path):
    retriever = build_retriever(tmp_path)

    results = retriever.search("security auditing", limit=3)

    result_ids = {result.chunk_id for result in results}

    assert "doc:1:0" in result_ids
    assert "doc:1:1" in result_ids


def test_hybrid_search_respects_limit(tmp_path):
    retriever = build_retriever(tmp_path)

    results = retriever.search("engineering", limit=1)

    assert len(results) == 1


def test_hybrid_search_empty_query_returns_empty(tmp_path):
    retriever = build_retriever(tmp_path)

    assert retriever.search("   ") == []


def test_hybrid_search_rejects_invalid_limit(tmp_path):
    retriever = build_retriever(tmp_path)

    with pytest.raises(ValueError, match="limit"):
        retriever.search("security", limit=0)


def test_hybrid_search_rejects_invalid_rrf_k(tmp_path):
    db_path = tmp_path / "hybrid.db"

    chunk_store = ChunkStore(db_path)
    lexical = LexicalRetriever(db_path)
    provider = DeterministicEmbeddingProvider(dimensions=8)
    vector_store = VectorStore(db_path, dimensions=8)

    with pytest.raises(ValueError, match="rrf_k"):
        HybridRetriever(
            lexical_retriever=lexical,
            vector_store=vector_store,
            embedding_provider=provider,
            chunk_store=chunk_store,
            rrf_k=0,
        )
