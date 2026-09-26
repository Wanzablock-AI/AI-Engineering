from dataclasses import dataclass
from typing import Protocol

from app.reranking import RerankedResult


@dataclass(frozen=True)
class Citation:
    document: str
    page_number: int


@dataclass(frozen=True)
class GroundedContext:
    chunk_id: str
    document: str
    page_number: int
    text: str


@dataclass(frozen=True)
class GroundedAnswer:
    answer: str
    citations: list[Citation]


class AnswerProvider(Protocol):
    def generate(
        self,
        query: str,
        context: list[GroundedContext],
    ) -> str:
        ...


class GroundedAnswerGenerator:
    """Builds answers strictly from retrieved document context."""

    def __init__(self, provider: AnswerProvider):
        self.provider = provider

    def generate(
        self,
        query: str,
        candidates: list[RerankedResult],
    ) -> GroundedAnswer:
        if not query.strip():
            raise ValueError("query cannot be empty")

        context = [
            GroundedContext(
                chunk_id=candidate.chunk_id,
                document=candidate.document,
                page_number=candidate.page_number,
                text=candidate.text,
            )
            for candidate in candidates
        ]

        if not context:
            return GroundedAnswer(
                answer="I could not find relevant information in the provided documents.",
                citations=[],
            )

        answer = self.provider.generate(
            query,
            context,
        ).strip()

        citations = self._build_citations(context)

        return GroundedAnswer(
            answer=answer,
            citations=citations,
        )

    @staticmethod
    def _build_citations(
        context: list[GroundedContext],
    ) -> list[Citation]:
        seen: set[tuple[str, int]] = set()
        citations: list[Citation] = []

        for item in context:
            key = (item.document, item.page_number)

            if key in seen:
                continue

            seen.add(key)

            citations.append(
                Citation(
                    document=item.document,
                    page_number=item.page_number,
                )
            )

        return citations
