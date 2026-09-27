import httpx


class OllamaProvider:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        client=None,
    ):
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.Client(timeout=120.0)

    def generate(self, model: str, prompt: str) -> str:
        response = self.client.post(
            f"{self.base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
            },
        )

        response.raise_for_status()

        payload = response.json()

        return payload["response"]
