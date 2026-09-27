import pytest

from app.provider import OllamaResearchProvider


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def post(self, url, json):
        self.calls.append((url, json))
        return self.response


def test_provider_returns_research_answer():
    client = FakeClient(
        FakeResponse(
            {
                "response": "RAG retrieves relevant context before generation.",
            }
        )
    )

    provider = OllamaResearchProvider(
        base_url="http://ollama:11434",
        model="granite4.1:3b",
        client=client,
    )

    answer = provider.research("What is RAG?")

    assert answer == "RAG retrieves relevant context before generation."
    assert client.calls == [
        (
            "http://ollama:11434/api/generate",
            {
                "model": "granite4.1:3b",
                "prompt": "What is RAG?",
                "stream": False,
            },
        )
    ]


def test_provider_rejects_empty_question():
    client = FakeClient(FakeResponse({"response": "unused"}))

    provider = OllamaResearchProvider(
        base_url="http://ollama:11434",
        model="granite4.1:3b",
        client=client,
    )

    with pytest.raises(ValueError, match="Question cannot be empty"):
        provider.research("")


def test_provider_rejects_missing_response():
    client = FakeClient(FakeResponse({}))

    provider = OllamaResearchProvider(
        base_url="http://ollama:11434",
        model="granite4.1:3b",
        client=client,
    )

    with pytest.raises(RuntimeError, match="Invalid provider response"):
        provider.research("What is RAG?")


def test_provider_accepts_custom_timeout():
    client = FakeClient(FakeResponse({"response": "answer"}))

    provider = OllamaResearchProvider(
        base_url="http://ollama:11434",
        model="granite4.1:3b",
        timeout=300.0,
        client=client,
    )

    assert provider.timeout == 300.0
