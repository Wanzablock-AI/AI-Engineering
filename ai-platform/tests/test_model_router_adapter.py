from app.model_router_adapter import ModelRouterAdapter


def fake_router_response(model="granite4.1:3b", answer="Generated answer."):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "model": model,
                "answer": answer,
            }

    return FakeResponse()


def test_model_router_adapter_generates_answer(monkeypatch):
    adapter = ModelRouterAdapter(
        base_url="http://model-router:8000"
    )

    monkeypatch.setattr(
        "app.model_router_adapter.httpx.post",
        lambda url, json, timeout: fake_router_response(),
    )

    result = adapter.generate("Explain RAG.")

    assert result.model == "granite4.1:3b"
    assert result.answer == "Generated answer."


def test_model_router_adapter_calls_generate_endpoint(monkeypatch):
    adapter = ModelRouterAdapter(
        base_url="http://model-router:8000"
    )

    calls = {}

    def fake_post(url, json, timeout):
        calls["url"] = url
        calls["json"] = json
        calls["timeout"] = timeout
        return fake_router_response()

    monkeypatch.setattr(
        "app.model_router_adapter.httpx.post",
        fake_post,
    )

    result = adapter.generate("Explain RAG.")

    assert result.answer == "Generated answer."
    assert calls["url"] == "http://model-router:8000/generate"
    assert calls["json"] == {"request": "Explain RAG."}
    assert calls["timeout"] > 0


def test_model_router_adapter_preserves_model(monkeypatch):
    adapter = ModelRouterAdapter(
        base_url="http://model-router:8000"
    )

    monkeypatch.setattr(
        "app.model_router_adapter.httpx.post",
        lambda url, json, timeout: fake_router_response(
            model="qwen2.5-coder:3b",
            answer="Code answer.",
        ),
    )

    result = adapter.generate("Write Python code.")

    assert result.model == "qwen2.5-coder:3b"
    assert result.answer == "Code answer."
