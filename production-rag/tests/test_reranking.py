import pytest

from app.hybrid_retrieval import HybridSearchResult
from app.reranking import HybridReranker


class FakeReranker:
    def score(self, query: str, text: str) -> float:
        if "security" in text.lower():
            return 0.95

        if "contract" in text.lower():
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


def test_reranker_orders_candidates_by_cross_encoder_score():
    reranker = HybridReranker(FakeReranker())

    results = reranker.rerank(
        "security",
        build_candidates(),
        limit=3,
    )

    assert results[0].chunk_id == "doc:1:1"
    assert results[0].score == 0.95
    assert results[1].chunk_id == "doc:1:2"
    assert results[2].chunk_id == "doc:1:0"


def test_reranker_preserves_metadata():
    reranker = HybridReranker(FakeReranker())

    results = reranker.rerank(
        "security",
        build_candidates(),
        limit=1,
    )

    assert results[0].document == "doc.pdf"
    assert results[0].page_number == 2
    assert results[0].text == "Smart contract security"


def test_reranker_respects_limit():
    reranker = HybridReranker(FakeReranker())

    results = reranker.rerank(
        "security",
        build_candidates(),
        limit=2,
    )

    assert len(results) == 2


def test_reranker_empty_query_returns_empty():
    reranker = HybridReranker(FakeReranker())

    assert reranker.rerank("   ", build_candidates()) == []


def test_reranker_rejects_invalid_limit():
    reranker = HybridReranker(FakeReranker())

    with pytest.raises(ValueError, match="limit"):
        reranker.rerank(
            "security",
            build_candidates(),
            limit=0,
        )


def test_reranker_empty_candidate_list():
    reranker = HybridReranker(FakeReranker())

    assert reranker.rerank("security", []) == []
