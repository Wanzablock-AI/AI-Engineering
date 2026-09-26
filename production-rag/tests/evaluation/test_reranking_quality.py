from app.hybrid_retrieval import HybridSearchResult
from app.reranking import HybridReranker

from evaluation.retrieval_quality import (
    EvaluationCase,
    evaluate_retrieval,
)


class EvaluationReranker:
    def score(self, query: str, text: str) -> float:
        text = text.lower()

        if query == "security":
            if "security" in text:
                return 0.95
            if "contract" in text:
                return 0.70

        if query == "contract":
            if "contract security" in text:
                return 0.95
            if "contract" in text:
                return 0.70

        return 0.20


def build_candidates():
    return [
        HybridSearchResult(
            chunk_id="doc:1:0",
            document="doc.pdf",
            page_number=1,
            text="Machine learning systems",
            score=0.03,
        ),
        HybridSearchResult(
            chunk_id="doc:1:1",
            document="doc.pdf",
            page_number=2,
            text="Smart contract security",
            score=0.02,
        ),
        HybridSearchResult(
            chunk_id="doc:1:2",
            document="doc.pdf",
            page_number=3,
            text="Contract architecture",
            score=0.01,
        ),
    ]


def test_reranking_quality():
    reranker = HybridReranker(EvaluationReranker())

    cases = [
        EvaluationCase(
            query="security",
            relevant_chunk_ids=frozenset({"doc:1:1"}),
        ),
        EvaluationCase(
            query="contract",
            relevant_chunk_ids=frozenset({"doc:1:1"}),
        ),
    ]

    candidates = build_candidates()

    metrics = evaluate_retrieval(
        cases,
        lambda query, limit: [
            result.chunk_id
            for result in reranker.rerank(
                query,
                candidates,
                limit=limit,
            )
        ],
    )

    assert metrics.recall_at_1 == 1.0
    assert metrics.recall_at_3 == 1.0
    assert metrics.mean_reciprocal_rank == 1.0
