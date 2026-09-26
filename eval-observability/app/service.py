from dataclasses import dataclass

from app.evaluator import EvaluationResult, Evaluator
from app.observability import ObservabilityEvent, ObservabilityRecorder


@dataclass(frozen=True)
class EvaluationServiceResult:
    evaluation: EvaluationResult
    observability: ObservabilityEvent


class EvaluationService:
    def __init__(self, evaluator=None, observability=None, store=None):
        self.evaluator = evaluator or Evaluator()
        self.observability = observability or ObservabilityRecorder()
        self.store = store

    def process(
        self,
        request: str,
        response: str,
        model: str,
        provider: str,
        latency_ms: float,
    ) -> EvaluationServiceResult:
        evaluation = self.evaluator.evaluate(
            request=request,
            response=response,
        )

        event = self.observability.record(
            request=request,
            model=model,
            provider=provider,
            latency_ms=latency_ms,
            success=evaluation.passed,
        )

        if self.store is not None:
            self.store.save(
                request=event.request,
                model=event.model,
                provider=event.provider,
                latency_ms=event.latency_ms,
                success=event.success,
            )

        return EvaluationServiceResult(
            evaluation=evaluation,
            observability=event,
        )
