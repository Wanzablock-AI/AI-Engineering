from dataclasses import dataclass


@dataclass(frozen=True)
class ObservabilityEvent:
    request: str
    model: str
    provider: str
    latency_ms: float
    success: bool


class ObservabilityRecorder:
    def __init__(self):
        self.events = []

    def record(
        self,
        request: str,
        model: str,
        provider: str,
        latency_ms: float,
        success: bool,
    ) -> ObservabilityEvent:
        event = ObservabilityEvent(
            request=request,
            model=model,
            provider=provider,
            latency_ms=latency_ms,
            success=success,
        )
        self.events.append(event)
        return event
