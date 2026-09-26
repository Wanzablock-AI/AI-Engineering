from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationResult:
    passed: bool
    score: float
    reason: str


class Evaluator:
    def evaluate(self, request: str, response: str) -> EvaluationResult:
        if not isinstance(request, str):
            raise ValueError("Request must be a string")

        if not isinstance(response, str):
            raise ValueError("Response must be a string")

        if not response.strip():
            return EvaluationResult(
                passed=False,
                score=0.0,
                reason="empty_response",
            )

        return EvaluationResult(
            passed=True,
            score=1.0,
            reason="response_present",
        )
