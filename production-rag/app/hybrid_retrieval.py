from dataclasses import dataclass

from app.embeddings import EmbeddingProvider
from app.retrieval import LexicalRetriever
from app.storage import ChunkStore
from app.vector_store import VectorStore


@dataclass(frozen=True)
class HybridSearchResult:
    chunk_id: str
    document: str
    page_number: int
    text: str
    score: float


class HybridRetriever:
    def __init__(
        self,
        lexical_retriever: LexicalRetriever,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        chunk_store: ChunkStore,
        rrf_k: int = 60,
    ):
        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero")

        self.lexical_retriever = lexical_retriever
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.chunk_store = chunk_store
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        limit: int = 5,
    ) -> list[HybridSearchResult]:
        if not query.strip():
            return []

        if limit <= 0:
            raise ValueError("limit must be greater than zero")

        lexical_results = self.lexical_retriever.search(
            query,
            limit=limit,
        )

        query_vector = self.embedding_provider.embed(query)

        vector_results = self.vector_store.search(
            query_vector,
            limit=limit,
        )

        scores: dict[str, float] = {}

        for rank, result in enumerate(lexical_results, start=1):
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + (
                1.0 / (self.rrf_k + rank)
            )

        for rank, result in enumerate(vector_results, start=1):
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + (
                1.0 / (self.rrf_k + rank)
            )

        ranked_ids = sorted(
            scores,
            key=lambda chunk_id: scores[chunk_id],
            reverse=True,
        )

        results: list[HybridSearchResult] = []

        for chunk_id in ranked_ids[:limit]:
            chunk = self.chunk_store.get_chunk(chunk_id)

            if chunk is None:
                continue

            results.append(
                HybridSearchResult(
                    chunk_id=chunk.chunk_id,
                    document=chunk.document,
                    page_number=chunk.page_number,
                    text=chunk.text,
                    score=scores[chunk_id],
                )
            )

        return results
