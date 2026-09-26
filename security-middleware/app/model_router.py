import httpx


class ModelRouterClient:
    def __init__(self, base_url: str, client=None):
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.Client(timeout=120.0)

    def generate(self, request: str) -> str:
        response = self.client.post(
            f"{self.base_url}/generate",
            json={"request": request},
        )
        response.raise_for_status()

        payload = response.json()
        return payload["answer"]
