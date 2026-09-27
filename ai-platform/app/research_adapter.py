from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class ResearchResult:
    task_id: str
    question: str
    answer: str
    success: bool


@dataclass(frozen=True)
class ResearchReport:
    request: str
    results: list[ResearchResult]
    synthesis: str


class ResearchAdapter:
    def __init__(self, base_url: str, timeout: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def research(
        self,
        request: str,
        questions: list[str],
    ) -> ResearchReport:
        response = httpx.post(
            f"{self.base_url}/research",
            json={
                "request": request,
                "questions": questions,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()

        payload = response.json()

        return ResearchReport(
            request=payload["request"],
            results=[
                ResearchResult(
                    task_id=result["task_id"],
                    question=result["question"],
                    answer=result["answer"],
                    success=result["success"],
                )
                for result in payload["results"]
            ],
            synthesis=payload["synthesis"],
        )
