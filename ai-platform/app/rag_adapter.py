from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class RAGCitation:
    document: str
    page_number: int


@dataclass(frozen=True)
class RAGResult:
    answer: str
    citations: list[RAGCitation]


class RAGAdapter:
    def __init__(self, base_url: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def query(self, request: str) -> RAGResult:
        response = httpx.post(
            f"{self.base_url}/query",
            json={"query": request},
            timeout=self.timeout,
        )
        response.raise_for_status()

        payload = response.json()

        return RAGResult(
            answer=payload["answer"],
            citations=[
                RAGCitation(
                    document=citation["document"],
                    page_number=citation["page_number"],
                )
                for citation in payload["citations"]
            ],
        )
