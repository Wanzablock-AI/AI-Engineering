import pytest

from evaluation.retrieval_quality import (
    EvaluationCase,
    evaluate_retrieval,
)


def test_evaluate_retrieval_calculates_recall_and_mrr():
    cases = [
        EvaluationCase(
            query="security",
            relevant_chunk_ids=frozenset({"a"}),
        ),
        EvaluationCase(
            query="contracts",
            relevant_chunk_ids=frozenset({"b"}),
        ),
        EvaluationCase(
            query="python",
            relevant_chunk_ids=frozenset({"c"}),
        ),
    ]

    results = {
        "security": ["a", "x", "y"],
        "contracts": ["x", "b", "y"],
        "python": ["x", "y", "c"],
    }

    metrics = evaluate_retrieval(
        cases,
        lambda query, limit: results[query][:limit],
    )

    assert metrics.recall_at_1 == pytest.approx(1 / 3)
    assert metrics.recall_at_3 == pytest.approx(1.0)
    assert metrics.mean_reciprocal_rank == pytest.approx(
        (1.0 + 0.5 + 1 / 3) / 3
    )


def test_evaluate_retrieval_rejects_empty_cases():
    with pytest.raises(
        ValueError,
        match="evaluation cases cannot be empty",
    ):
        evaluate_retrieval([], lambda query, limit: [])


def test_evaluate_retrieval_rejects_blank_query():
    cases = [
        EvaluationCase(
            query="   ",
            relevant_chunk_ids=frozenset({"a"}),
        )
    ]

    with pytest.raises(
        ValueError,
        match="evaluation query cannot be blank",
    ):
        evaluate_retrieval(cases, lambda query, limit: [])


def test_evaluate_retrieval_rejects_missing_relevant_chunks():
    cases = [
        EvaluationCase(
            query="security",
            relevant_chunk_ids=frozenset(),
        )
    ]

    with pytest.raises(
        ValueError,
        match="relevant chunk IDs",
    ):
        evaluate_retrieval(cases, lambda query, limit: [])
