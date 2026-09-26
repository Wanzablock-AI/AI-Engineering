from dataclasses import dataclass
from typing import Sequence

from app.generation import Citation, GroundedAnswer


@dataclass(frozen=True)
class GenerationEvaluationCase:
    query: str
    required_answer_terms: frozenset[str]
    expected_citations: frozenset[tuple[str, int]]


@dataclass(frozen=True)
class GenerationMetrics:
    answer_relevance: float
    citation_coverage: float
    grounded_answer_rate: float


def evaluate_generation(
    cases: Sequence[GenerationEvaluationCase],
    generate,
) -> GenerationMetrics:
    if not cases:
        raise ValueError("evaluation cases cannot be empty")

    relevance_scores: list[float] = []
    citation_scores: list[float] = []
    grounded_scores: list[float] = []

    for case in cases:
        if not case.query.strip():
            raise ValueError("evaluation query cannot be blank")

        if not case.required_answer_terms:
            raise ValueError(
                "evaluation case must contain required answer terms"
            )

        if not case.expected_citations:
            raise ValueError(
                "evaluation case must contain expected citations"
            )

        result = generate(case.query)

        answer = result.answer.lower()

        matching_terms = sum(
            term.lower() in answer
            for term in case.required_answer_terms
        )

        relevance_scores.append(
            matching_terms / len(case.required_answer_terms)
        )

        actual_citations = {
            (citation.document, citation.page_number)
            for citation in result.citations
        }

        matching_citations = (
            case.expected_citations & actual_citations
        )

        citation_scores.append(
            len(matching_citations) / len(case.expected_citations)
        )

        grounded_scores.append(
            1.0
            if case.expected_citations.issubset(actual_citations)
            and matching_terms == len(case.required_answer_terms)
            else 0.0
        )

    total = len(cases)

    return GenerationMetrics(
        answer_relevance=sum(relevance_scores) / total,
        citation_coverage=sum(citation_scores) / total,
        grounded_answer_rate=sum(grounded_scores) / total,
    )
