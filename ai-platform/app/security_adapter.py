from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class SecurityResult:
    allowed: bool
    risk_level: str
    reason: str


class SecurityAdapter:
    def __init__(self, base_url: str, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def check(self, request: str) -> SecurityResult:
        response = httpx.post(
            f"{self.base_url}/check",
            json={"request": request},
            timeout=self.timeout,
        )
        response.raise_for_status()

        payload = response.json()

        return SecurityResult(
            allowed=payload["allowed"],
            risk_level=payload["risk_level"],
            reason=payload["reason"],
        )
