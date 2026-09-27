from app.evaluation_adapter import EvaluationAdapter


def fake_evaluation_response():
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "evaluation": {
                    "passed": True,
                    "score": 1.0,
                    "reason": "good_response",
                },
                "observability": {
                    "request": "Explain RAG.",
                    "model": "llama3.2:3b",
                    "provider": "model-router",
                    "latency_ms": 125.5,
                    "success": True,
                },
            }

    return FakeResponse()


def test_evaluation_adapter_processes_result(monkeypatch):
    adapter = EvaluationAdapter(
        base_url="http://eval-observability:8000"
    )

    monkeypatch.setattr(
        "app.evaluation_adapter.httpx.post",
        lambda url, json, timeout: fake_evaluation_response(),
    )

    result = adapter.evaluate(
        request="Explain RAG.",
        response="RAG retrieves relevant context before generation.",
        model="llama3.2:3b",
        provider="model-router",
        latency_ms=125.5,
    )

    assert result.passed is True
    assert result.score == 1.0
    assert result.reason == "good_response"


def test_evaluation_adapter_calls_evaluate_endpoint(monkeypatch):
    adapter = EvaluationAdapter(
        base_url="http://eval-observability:8000"
    )

    calls = {}

    def fake_post(url, json, timeout):
        calls["url"] = url
        calls["json"] = json
        calls["timeout"] = timeout
        return fake_evaluation_response()

    monkeypatch.setattr(
        "app.evaluation_adapter.httpx.post",
        fake_post,
    )

    result = adapter.evaluate(
        request="Explain RAG.",
        response="RAG retrieves context.",
        model="llama3.2:3b",
        provider="model-router",
        latency_ms=125.5,
    )

    assert result.passed is True
    assert calls["url"] == "http://eval-observability:8000/evaluate"
    assert calls["json"] == {
        "request": "Explain RAG.",
        "response": "RAG retrieves context.",
        "model": "llama3.2:3b",
        "provider": "model-router",
        "latency_ms": 125.5,
    }
    assert calls["timeout"] > 0
