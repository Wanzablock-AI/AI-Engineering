from app.chunking import Chunk
from app.embeddings import DeterministicEmbeddingProvider
from app.hybrid_retrieval import HybridRetriever
from app.retrieval import LexicalRetriever
from app.storage import ChunkStore
from app.vector_store import VectorStore

from evaluation.retrieval_quality import (
    EvaluationCase,
    evaluate_retrieval,
)


def build_retriever(tmp_path):
    db_path = tmp_path / "evaluation.db"

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

    return HybridRetriever(
        lexical_retriever=lexical,
        vector_store=vector_store,
        embedding_provider=provider,
        chunk_store=chunk_store,
    )


def test_hybrid_retrieval_quality(tmp_path):
    retriever = build_retriever(tmp_path)

    cases = [
        EvaluationCase(
            query="Python security",
            relevant_chunk_ids=frozenset({"doc:1:0"}),
        ),
        EvaluationCase(
            query="contract auditing",
            relevant_chunk_ids=frozenset({"doc:1:1"}),
        ),
        EvaluationCase(
            query="machine learning",
            relevant_chunk_ids=frozenset({"doc:1:2"}),
        ),
    ]

    metrics = evaluate_retrieval(
        cases,
        lambda query, limit: [
            result.chunk_id
            for result in retriever.search(query, limit=limit)
        ],
    )

    assert 0.0 <= metrics.recall_at_1 <= 1.0
    assert 0.0 <= metrics.recall_at_3 <= 1.0
    assert 0.0 <= metrics.mean_reciprocal_rank <= 1.0

    assert metrics.recall_at_3 == 1.0
