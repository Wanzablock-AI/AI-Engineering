import pytest

from app.generation import (
    GroundedAnswerGenerator,
    GroundedContext,
)
from app.reranking import RerankedResult


class FakeAnswerProvider:
    def __init__(self):
        self.last_query = None
        self.last_context = None

    def generate(
        self,
        query: str,
        context: list[GroundedContext],
    ) -> str:
        self.last_query = query
        self.last_context = context

        return "Python security requires careful input validation."


def build_candidates():
    return [
        RerankedResult(
            chunk_id="doc:1:0",
            document="security.pdf",
            page_number=2,
            text="Input validation is important for secure Python systems.",
            score=0.95,
        ),
        RerankedResult(
            chunk_id="doc:1:1",
            document="security.pdf",
            page_number=2,
            text="Validation should occur before processing untrusted input.",
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


def test_generation_returns_grounded_answer():
    provider = FakeAnswerProvider()
    generator = GroundedAnswerGenerator(provider)

    result = generator.generate(
        "How should Python systems handle untrusted input?",
        build_candidates(),
    )

    assert result.answer == (
        "Python security requires careful input validation."
    )

    assert provider.last_query == (
        "How should Python systems handle untrusted input?"
    )

    assert len(provider.last_context) == 3


def test_generation_builds_unique_page_citations():
    provider = FakeAnswerProvider()
    generator = GroundedAnswerGenerator(provider)

    result = generator.generate(
        "How should Python systems handle untrusted input?",
        build_candidates(),
    )

    assert result.citations == [
        type(result.citations[0])(
            document="security.pdf",
            page_number=2,
        ),
        type(result.citations[0])(
            document="security.pdf",
            page_number=4,
        ),
    ]


def test_generation_preserves_chunk_context():
    provider = FakeAnswerProvider()
    generator = GroundedAnswerGenerator(provider)

    generator.generate(
        "What does the document say?",
        build_candidates(),
    )

    first = provider.last_context[0]

    assert first.chunk_id == "doc:1:0"
    assert first.document == "security.pdf"
    assert first.page_number == 2
    assert "Input validation" in first.text


def test_generation_with_no_candidates_returns_grounding_message():
    provider = FakeAnswerProvider()
    generator = GroundedAnswerGenerator(provider)

    result = generator.generate(
        "What does the document say?",
        [],
    )

    assert result.answer == (
        "I could not find relevant information in the provided documents."
    )
    assert result.citations == []
    assert provider.last_query is None
    assert provider.last_context is None


def test_generation_rejects_empty_query():
    provider = FakeAnswerProvider()
    generator = GroundedAnswerGenerator(provider)

    with pytest.raises(ValueError, match="query"):
        generator.generate(
            "   ",
            build_candidates(),
        )
