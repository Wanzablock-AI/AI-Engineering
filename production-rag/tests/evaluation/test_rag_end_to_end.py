from app.generation import GroundedAnswerGenerator, GroundedContext
from app.graph import RAGComponents, RAGGraph
from app.hybrid_retrieval import HybridRetriever
from app.reranking import HybridReranker
from app.retrieval import LexicalRetriever
from app.storage import Chunk, ChunkStore
from app.vector_store import VectorStore


class DeterministicEmbeddingProvider:
    dimensions = 3

    def embed(self, text: str) -> list[float]:
        text = text.lower()

        if "validation" in text:
            return [1.0, 0.0, 0.0]

        if "security" in text:
            return [0.0, 1.0, 0.0]

        return [0.0, 0.0, 1.0]


class DeterministicReranker:
    def score(self, query: str, text: str) -> float:
        query_terms = set(query.lower().split())
        text_terms = set(text.lower().split())

        return float(len(query_terms & text_terms))


class DeterministicAnswerProvider:
    def generate(
        self,
        query: str,
        context: list[GroundedContext],
    ) -> str:
        combined = " ".join(item.text for item in context).lower()

        if "validation" in combined and "security" in combined:
            return (
                "The documents recommend input validation as part of "
                "security processing."
            )

        return "No grounded answer was available."


def build_graph(tmp_path):
    db_path = tmp_path / "rag.db"

    chunk_store = ChunkStore(db_path)
    lexical_retriever = LexicalRetriever(db_path)
    vector_store = VectorStore(db_path, dimensions=3)
    embedding_provider = DeterministicEmbeddingProvider()

    chunks = [
        Chunk(
            chunk_id="security:1:0",
            document="security.pdf",
            page_number=1,
            text=(
                "Input validation should happen before processing "
                "untrusted input."
            ),
        ),
        Chunk(
            chunk_id="security:2:0",
            document="security.pdf",
            page_number=2,
            text=(
                "Security testing should cover untrusted input paths."
            ),
        ),
    ]

    chunk_store.add_chunks(chunks)
    lexical_retriever.add_chunks(chunks)

    for chunk in chunks:
        vector_store.add_embedding(
            chunk.chunk_id,
            embedding_provider.embed(chunk.text),
        )

    retriever = HybridRetriever(
        lexical_retriever=lexical_retriever,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        chunk_store=chunk_store,
    )

    reranker = HybridReranker(DeterministicReranker())
    generator = GroundedAnswerGenerator(
        DeterministicAnswerProvider()
    )

    return RAGGraph(
        RAGComponents(
            retriever=retriever,
            reranker=reranker,
            generator=generator,
        )
    )


def test_rag_pipeline_runs_end_to_end(tmp_path):
    graph = build_graph(tmp_path)

    result = graph.invoke(
        "How should untrusted input be handled for security?"
    )

    assert "input validation" in result.answer.lower()
    assert "security" in result.answer.lower()

    citations = {
        (citation.document, citation.page_number)
        for citation in result.citations
    }

    assert citations == {
        ("security.pdf", 1),
        ("security.pdf", 2),
    }
