from fastapi.testclient import TestClient

from app.api import app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_models():
    response = client.get("/models")

    assert response.status_code == 200

    data = response.json()

    assert data == {
        "coding": "qwen2.5-coder:3b",
        "rag": "granite4.1:3b",
        "lightweight": "llama3.2:1b",
        "general": "llama3.2:3b",
    }


def test_route():
    response = client.post(
        "/route",
        json={"request": "Write a Python function"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "model": "qwen2.5-coder:3b",
        "reason": "coding",
    }


def test_route_rejects_empty_request():
    response = client.post(
        "/route",
        json={"request": ""},
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Request cannot be empty"
    }


def test_route_rejects_missing_request():
    response = client.post(
        "/route",
        json={},
    )

    assert response.status_code == 422


def test_generate_routes_and_generates(monkeypatch):
    calls = []

    def fake_generate(model, prompt):
        calls.append((model, prompt))
        return "generated answer"

    monkeypatch.setattr(
        "app.api.provider.generate",
        fake_generate,
    )

    response = client.post(
        "/generate",
        json={"request": "Write a Python function"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "model": "qwen2.5-coder:3b",
        "answer": "generated answer",
    }

    assert calls == [
        (
            "qwen2.5-coder:3b",
            "Write a Python function",
        )
    ]


def test_generate_returns_502_when_provider_fails(monkeypatch):
    def fake_generate(model, prompt):
        raise RuntimeError("Ollama unavailable")

    monkeypatch.setattr(
        "app.api.provider.generate",
        fake_generate,
    )

    response = client.post(
        "/generate",
        json={"request": "Write a Python function"},
    )

    assert response.status_code == 502
    assert response.json() == {
        "detail": "Model provider unavailable"
    }


def test_route_accepts_explicit_model():
    response = client.post(
        "/route",
        json={
            "request": "Explain distributed systems.",
            "model": "granite4.1:3b",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "model": "granite4.1:3b",
        "reason": "explicit",
    }


def test_generate_uses_explicit_model(monkeypatch):
    calls = []

    def fake_generate(model, prompt):
        calls.append((model, prompt))
        return "generated answer"

    monkeypatch.setattr(
        "app.api.provider.generate",
        fake_generate,
    )

    response = client.post(
        "/generate",
        json={
            "request": "Explain distributed systems.",
            "model": "granite4.1:3b",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "model": "granite4.1:3b",
        "answer": "generated answer",
    }

    assert calls == [
        (
            "granite4.1:3b",
            "Explain distributed systems.",
        )
    ]


def test_unknown_explicit_model_is_rejected():
    response = client.post(
        "/route",
        json={
            "request": "Explain distributed systems.",
            "model": "unknown-model",
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Unsupported model: unknown-model"
    }
