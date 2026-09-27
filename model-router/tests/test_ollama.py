import pytest

from app.ollama import OllamaProvider


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def post(self, url, json):
        self.calls.append((url, json))
        return self.response


def test_generate_sends_model_and_prompt():
    client = FakeClient(
        FakeResponse({"response": "hello from model"})
    )

    provider = OllamaProvider(
        base_url="http://ollama:11434",
        client=client,
    )

    result = provider.generate(
        model="llama3.2:3b",
        prompt="Say hello",
    )

    assert result == "hello from model"
    assert client.calls == [
        (
            "http://ollama:11434/api/generate",
            {
                "model": "llama3.2:3b",
                "prompt": "Say hello",
                "stream": False,
            },
        )
    ]


def test_generate_raises_on_http_error():
    client = FakeClient(
        FakeResponse({}, status_code=500)
    )

    provider = OllamaProvider(
        base_url="http://ollama:11434",
        client=client,
    )

    with pytest.raises(RuntimeError):
        provider.generate(
            model="llama3.2:3b",
            prompt="Say hello",
        )


def test_provider_uses_settings_url(monkeypatch):
    monkeypatch.setenv(
        "OLLAMA_BASE_URL",
        "http://100.118.104.245:11434/",
    )

    from app.config import Settings

    client = FakeClient(
        FakeResponse({"response": "hello"})
    )

    provider = OllamaProvider(
        client=client,
        base_url=Settings().ollama_base_url,
    )

    provider.generate(
        model="llama3.2:3b",
        prompt="Say hello",
    )

    assert client.calls[0][0] == (
        "http://100.118.104.245:11434/api/generate"
    )
