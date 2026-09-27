import pytest

from app.contracts import ResearchResult
from app.provider import OllamaSynthesisProvider


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


def make_result(task_id: str, answer: str) -> ResearchResult:
    return ResearchResult(
        task_id=task_id,
        question=f"Question for {task_id}",
        answer=answer,
        success=True,
    )


def test_synthesis_provider_returns_synthesis():
    client = FakeClient(
        FakeResponse(
            {
                "response": "RAG retrieves context while fine-tuning changes model behavior.",
            }
        )
    )

    provider = OllamaSynthesisProvider(
        base_url="http://ollama:11434",
        model="granite4.1:3b",
        client=client,
    )

    results = [
        make_result("task-1", "RAG retrieves context."),
        make_result("task-2", "Fine-tuning changes model behavior."),
    ]

    synthesis = provider.synthesize(
        request="Compare RAG and fine-tuning.",
        results=results,
    )

    assert synthesis == (
        "RAG retrieves context while fine-tuning changes model behavior."
    )

    assert len(client.calls) == 1

    url, payload = client.calls[0]

    assert url == "http://ollama:11434/api/generate"
    assert payload["model"] == "granite4.1:3b"
    assert payload["stream"] is False
    assert "Compare RAG and fine-tuning." in payload["prompt"]
    assert "RAG retrieves context." in payload["prompt"]
    assert "Fine-tuning changes model behavior." in payload["prompt"]


def test_synthesis_provider_rejects_empty_request():
    client = FakeClient(FakeResponse({"response": "unused"}))

    provider = OllamaSynthesisProvider(
        base_url="http://ollama:11434",
        model="granite4.1:3b",
        client=client,
    )

    with pytest.raises(ValueError, match="Request cannot be empty"):
        provider.synthesize("", [make_result("task-1", "Answer")])


def test_synthesis_provider_rejects_empty_results():
    client = FakeClient(FakeResponse({"response": "unused"}))

    provider = OllamaSynthesisProvider(
        base_url="http://ollama:11434",
        model="granite4.1:3b",
        client=client,
    )

    with pytest.raises(ValueError, match="At least one research result"):
        provider.synthesize("Explain RAG.", [])


def test_synthesis_provider_rejects_missing_response():
    client = FakeClient(FakeResponse({}))

    provider = OllamaSynthesisProvider(
        base_url="http://ollama:11434",
        model="granite4.1:3b",
        client=client,
    )

    with pytest.raises(RuntimeError, match="Invalid provider response"):
        provider.synthesize(
            "Explain RAG.",
            [make_result("task-1", "Answer")],
        )
