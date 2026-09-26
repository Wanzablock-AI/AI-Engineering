import pytest

from app.generation import Citation, GroundedAnswer

from evaluation.generation_quality import (
    GenerationEvaluationCase,
    evaluate_generation,
)


def test_evaluate_generation_calculates_metrics():
    cases = [
        GenerationEvaluationCase(
            query="security",
            required_answer_terms=frozenset(
                {"validation", "untrusted input"}
            ),
            expected_citations=frozenset(
                {("security.pdf", 2)}
            ),
        ),
        GenerationEvaluationCase(
            query="testing",
            required_answer_terms=frozenset(
                {"security testing"}
            ),
            expected_citations=frozenset(
                {("security.pdf", 4)}
            ),
        ),
    ]

    answers = {
        "security": GroundedAnswer(
            answer=(
                "Validation is important when handling "
                "untrusted input."
            ),
            citations=[
                Citation(
                    document="security.pdf",
                    page_number=2,
                )
            ],
        ),
        "testing": GroundedAnswer(
            answer="Security testing should cover untrusted paths.",
            citations=[
                Citation(
                    document="security.pdf",
                    page_number=4,
                )
            ],
        ),
    }

    metrics = evaluate_generation(
        cases,
        lambda query: answers[query],
    )

    assert metrics.answer_relevance == pytest.approx(1.0)
    assert metrics.citation_coverage == pytest.approx(1.0)
    assert metrics.grounded_answer_rate == pytest.approx(1.0)


def test_evaluate_generation_detects_missing_citation():
    cases = [
        GenerationEvaluationCase(
            query="security",
            required_answer_terms=frozenset({"validation"}),
            expected_citations=frozenset(
                {("security.pdf", 2)}
            ),
        )
    ]

    result = GroundedAnswer(
        answer="Validation is important.",
        citations=[],
    )

    metrics = evaluate_generation(
        cases,
        lambda query: result,
    )

    assert metrics.answer_relevance == 1.0
    assert metrics.citation_coverage == 0.0
    assert metrics.grounded_answer_rate == 0.0


def test_evaluate_generation_detects_missing_answer_term():
    cases = [
        GenerationEvaluationCase(
            query="security",
            required_answer_terms=frozenset(
                {"validation", "untrusted input"}
            ),
            expected_citations=frozenset(
                {("security.pdf", 2)}
            ),
        )
    ]

    result = GroundedAnswer(
        answer="Validation is important.",
        citations=[
            Citation(
                document="security.pdf",
                page_number=2,
            )
        ],
    )

    metrics = evaluate_generation(
        cases,
        lambda query: result,
    )

    assert metrics.answer_relevance == 0.5
    assert metrics.citation_coverage == 1.0
    assert metrics.grounded_answer_rate == 0.0


def test_evaluate_generation_rejects_empty_cases():
    with pytest.raises(
        ValueError,
        match="evaluation cases cannot be empty",
    ):
        evaluate_generation([], lambda query: None)


def test_evaluate_generation_rejects_blank_query():
    cases = [
        GenerationEvaluationCase(
            query="   ",
            required_answer_terms=frozenset({"security"}),
            expected_citations=frozenset(
                {("security.pdf", 2)}
            ),
        )
    ]

    with pytest.raises(
        ValueError,
        match="evaluation query cannot be blank",
    ):
        evaluate_generation(cases, lambda query: None)


def test_evaluate_generation_rejects_missing_answer_terms():
    cases = [
        GenerationEvaluationCase(
            query="security",
            required_answer_terms=frozenset(),
            expected_citations=frozenset(
                {("security.pdf", 2)}
            ),
        )
    ]

    with pytest.raises(
        ValueError,
        match="required answer terms",
    ):
        evaluate_generation(
            cases,
            lambda query: None,
        )


def test_evaluate_generation_rejects_missing_citations():
    cases = [
        GenerationEvaluationCase(
            query="security",
            required_answer_terms=frozenset({"security"}),
            expected_citations=frozenset(),
        )
    ]

    with pytest.raises(
        ValueError,
        match="expected citations",
    ):
        evaluate_generation(
            cases,
            lambda query: None,
        )
