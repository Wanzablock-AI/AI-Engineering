from dataclasses import dataclass
from typing import Protocol

from sentence_transformers import CrossEncoder

from app.hybrid_retrieval import HybridSearchResult


@dataclass(frozen=True)
class RerankedResult:
    chunk_id: str
    document: str
    page_number: int
    text: str
    score: float


class Reranker(Protocol):
    def score(self, query: str, text: str) -> float:
        ...


class CrossEncoderReranker:
    """Cross-encoder reranker using a local Sentence Transformers model."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.model = CrossEncoder(model_name)

    def score(self, query: str, text: str) -> float:
        if not query.strip():
            raise ValueError("query cannot be empty")

        if not text.strip():
            raise ValueError("text cannot be empty")

        score = self.model.predict([(query, text)])[0]

        return float(score)


class HybridReranker:
    """Reranks hybrid-retrieval candidates with a supplied scorer."""

    def __init__(
        self,
        reranker: Reranker,
        min_score: float | None = None,
    ):
        self.reranker = reranker
        self.min_score = min_score

    def rerank(
        self,
        query: str,
        candidates: list[HybridSearchResult],
        limit: int = 5,
    ) -> list[RerankedResult]:
        if not query.strip():
            return []

        if limit <= 0:
            raise ValueError("limit must be greater than zero")

        scored: list[RerankedResult] = []

        for candidate in candidates:
            score = self.reranker.score(
                query,
                candidate.text,
            )

            scored.append(
                RerankedResult(
                    chunk_id=candidate.chunk_id,
                    document=candidate.document,
                    page_number=candidate.page_number,
                    text=candidate.text,
                    score=score,
                )
            )

        scored.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        if self.min_score is not None:
            scored = [
                result
                for result in scored
                if result.score >= self.min_score
            ]

        return scored[:limit]
