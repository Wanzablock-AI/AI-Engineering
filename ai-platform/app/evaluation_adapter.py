from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class EvaluationResult:
    passed: bool
    score: float
    reason: str


class EvaluationAdapter:
    def __init__(self, base_url: str, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def evaluate(
        self,
        request: str,
        response: str,
        model: str,
        provider: str,
        latency_ms: float,
    ) -> EvaluationResult:
        result = httpx.post(
            f"{self.base_url}/evaluate",
            json={
                "request": request,
                "response": response,
                "model": model,
                "provider": provider,
                "latency_ms": latency_ms,
            },
            timeout=self.timeout,
        )

        result.raise_for_status()

        payload = result.json()
        evaluation = payload["evaluation"]

        return EvaluationResult(
            passed=evaluation["passed"],
            score=evaluation["score"],
            reason=evaluation["reason"],
        )
