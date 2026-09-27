from app.ollama import OllamaProvider
from app.router import ModelRouter


class FakeResponse:
    def raise_for_status(self):
        pass

    def json(self):
        return {"response": "generated answer"}


class FakeClient:
    def __init__(self):
        self.calls = []

    def post(self, url, json):
        self.calls.append((url, json))
        return FakeResponse()


def test_router_selects_model_for_generation():
    client = FakeClient()

    router = ModelRouter()
    provider = OllamaProvider(
        base_url="http://ollama:11434",
        client=client,
    )

    decision = router.route_request("Write a Python function to add two numbers")

    result = provider.generate(
        model=decision.model,
        prompt="Write a Python function to add two numbers",
    )

    assert decision.model == "qwen2.5-coder:3b"
    assert result == "generated answer"

    assert client.calls == [
        (
            "http://ollama:11434/api/generate",
            {
                "model": "qwen2.5-coder:3b",
                "prompt": "Write a Python function to add two numbers",
                "stream": False,
            },
        )
    ]
