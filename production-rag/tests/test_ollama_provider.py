import httpx
import pytest

from app.generation import GroundedContext
from app.ollama_provider import OllamaAnswerProvider


def build_context():
    return [
        GroundedContext(
            chunk_id="security:1:0",
            document="security.pdf",
            page_number=2,
            text="Input validation is important for secure systems.",
        )
    ]


def test_provider_builds_grounded_prompt():
    prompt = OllamaAnswerProvider._build_prompt(
        "Why is validation important?",
        build_context(),
    )

    assert "Why is validation important?" in prompt
    assert "security.pdf" in prompt
    assert "Page: 2" in prompt
    assert "Input validation is important" in prompt
    assert "ONLY the supplied document context" in prompt


def test_provider_sends_request(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout

        return httpx.Response(
            200,
            json={"response": "Validation reduces security risk."},
        )

    monkeypatch.setattr(httpx, "post", fake_post)

    provider = OllamaAnswerProvider(
        base_url="http://ollama.test",
        model="test-model",
        timeout=30.0,
    )

    result = provider.generate(
        "Why is validation important?",
        build_context(),
    )

    assert result == "Validation reduces security risk."
    assert captured["url"] == "http://ollama.test/api/generate"
    assert captured["json"]["model"] == "test-model"
    assert captured["json"]["stream"] is False
    assert captured["timeout"] == 30.0


def test_provider_rejects_empty_query():
    provider = OllamaAnswerProvider()

    with pytest.raises(ValueError, match="query"):
        provider.generate("   ", build_context())


def test_provider_rejects_empty_context():
    provider = OllamaAnswerProvider()

    with pytest.raises(ValueError, match="context"):
        provider.generate("What does it say?", [])


def test_provider_handles_http_failure(monkeypatch):
    def fake_post(url, json, timeout):
        raise httpx.ConnectError("connection failed")

    monkeypatch.setattr(httpx, "post", fake_post)

    provider = OllamaAnswerProvider()

    with pytest.raises(RuntimeError, match="Ollama request failed"):
        provider.generate(
            "What does it say?",
            build_context(),
        )


def test_provider_handles_invalid_json(monkeypatch):
    def fake_post(url, json, timeout):
        return httpx.Response(
            200,
            content=b"not-json",
        )

    monkeypatch.setattr(httpx, "post", fake_post)

    provider = OllamaAnswerProvider()

    with pytest.raises(RuntimeError, match="invalid JSON"):
        provider.generate(
            "What does it say?",
            build_context(),
        )


def test_provider_handles_missing_response(monkeypatch):
    def fake_post(url, json, timeout):
        return httpx.Response(
            200,
            json={"response": ""},
        )

    monkeypatch.setattr(httpx, "post", fake_post)

    provider = OllamaAnswerProvider()

    with pytest.raises(RuntimeError, match="generated text"):
        provider.generate(
            "What does it say?",
            build_context(),
        )


def test_provider_uses_local_ollama_default(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)

    provider = OllamaAnswerProvider()

    assert provider.base_url == "http://localhost:11434"


def test_provider_reads_ollama_base_url_from_environment(monkeypatch):
    monkeypatch.setenv(
        "OLLAMA_BASE_URL",
        "http://ollama.internal:11434",
    )

    provider = OllamaAnswerProvider()

    assert provider.base_url == "http://ollama.internal:11434"


def test_provider_reads_ollama_model_from_environment(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "test-model")

    provider = OllamaAnswerProvider()

    assert provider.model == "test-model"
