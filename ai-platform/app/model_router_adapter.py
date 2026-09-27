from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class ModelRouterResult:
    model: str
    answer: str


class ModelRouterAdapter:
    def __init__(self, base_url: str, timeout: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(self, request: str) -> ModelRouterResult:
        response = httpx.post(
            f"{self.base_url}/generate",
            json={"request": request},
            timeout=self.timeout,
        )
        response.raise_for_status()

        payload = response.json()

        return ModelRouterResult(
            model=payload["model"],
            answer=payload["answer"],
        )
