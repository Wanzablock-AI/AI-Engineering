from dataclasses import dataclass

from app.security import SecurityMiddleware


@dataclass(frozen=True)
class GatewayResult:
    allowed: bool
    risk_level: str
    reason: str
    answer: str | None = None


class SecurityGateway:
    def __init__(self, provider, security=None):
        self.provider = provider
        self.security = security or SecurityMiddleware()

    def handle(self, request: str) -> GatewayResult:
        decision = self.security.check(request)

        if not decision.allowed:
            return GatewayResult(
                allowed=False,
                risk_level=decision.risk_level,
                reason=decision.reason,
                answer=None,
            )

        answer = self.provider.generate(request)

        return GatewayResult(
            allowed=True,
            risk_level=decision.risk_level,
            reason=decision.reason,
            answer=answer,
        )
