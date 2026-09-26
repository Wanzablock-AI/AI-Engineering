from dataclasses import dataclass


@dataclass(frozen=True)
class AuditEvent:
    allowed: bool
    risk_level: str
    reason: str


class AuditLogger:
    def __init__(self):
        self.events = []

    def record(self, decision):
        event = AuditEvent(
            allowed=decision.allowed,
            risk_level=decision.risk_level,
            reason=decision.reason,
        )
        self.events.append(event)
        return event
