from fastapi.testclient import TestClient

from app.gateway import app


def test_internal_error_returns_unified_error_response(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "app.gateway.security_adapter.check",
        fail,
    )

    test_client = TestClient(
        app,
        raise_server_exceptions=False,
    )

    response = test_client.post(
        "/v1/generate",
        json={
            "request": "Explain RAG.",
            "model": "auto",
        },
    )

    assert response.status_code == 500

    payload = response.json()

    assert payload["code"] == "INTERNAL_ERROR"
    assert payload["message"] == "Internal server error."
    assert payload["request_id"]
    assert response.headers["X-Request-ID"] == payload["request_id"]
