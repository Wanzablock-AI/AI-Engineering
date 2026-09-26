from app.generation import GroundedAnswerGenerator, GroundedContext
from app.reranking import RerankedResult

from evaluation.generation_quality import (
    GenerationEvaluationCase,
    evaluate_generation,
)


class EvaluationAnswerProvider:
    def generate(
        self,
        query: str,
        context: list[GroundedContext],
    ) -> str:
        if "untrusted input" in query.lower():
            return (
                "Python systems should validate untrusted input "
                "before processing it."
            )

        return "Security testing should cover untrusted input paths."


def build_candidates():
    return [
        RerankedResult(
            chunk_id="doc:1:0",
            document="security.pdf",
            page_number=2,
            text=(
                "Input validation is important for secure Python systems."
            ),
            score=0.95,
        ),
        RerankedResult(
            chunk_id="doc:1:1",
            document="security.pdf",
            page_number=2,
            text=(
                "Validation should occur before processing "
                "untrusted input."
            ),
            score=0.90,
        ),
        RerankedResult(
            chunk_id="doc:1:2",
            document="security.pdf",
            page_number=4,
            text="Security testing should cover untrusted input paths.",
            score=0.80,
        ),
    ]


def test_grounded_answer_quality():
    generator = GroundedAnswerGenerator(
        EvaluationAnswerProvider()
    )

    cases = [
        GenerationEvaluationCase(
            query="How should Python systems handle untrusted input?",
            required_answer_terms=frozenset(
                {"validate", "untrusted input"}
            ),
            expected_citations=frozenset(
                {("security.pdf", 2)}
            ),
        ),
        GenerationEvaluationCase(
            query="What should security testing cover?",
            required_answer_terms=frozenset(
                {"security testing", "untrusted input"}
            ),
            expected_citations=frozenset(
                {
                    ("security.pdf", 2),
                    ("security.pdf", 4),
                }
            ),
        ),
    ]

    candidates = build_candidates()

    metrics = evaluate_generation(
        cases,
        lambda query: generator.generate(
            query,
            candidates,
        ),
    )

    assert metrics.answer_relevance == 1.0
    assert metrics.citation_coverage == 1.0
    assert metrics.grounded_answer_rate == 1.0
