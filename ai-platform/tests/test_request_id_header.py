from fastapi.testclient import TestClient

from app.gateway import app


client = TestClient(app)


def test_generate_returns_request_id_header():
    response = client.post(
        "/v1/generate",
        json={
            "request": "Explain RAG.",
            "model": "auto",
        },
    )

    assert response.status_code == 200

    body_request_id = response.json()["request_id"]
    header_request_id = response.headers.get("X-Request-ID")

    assert header_request_id
    assert header_request_id == body_request_id


def test_research_returns_request_id_header():
    response = client.post(
        "/v1/research",
        json={
            "request": "Explain RAG.",
            "questions": [
                "What is RAG?",
            ],
        },
    )

    assert response.status_code == 200

    body_request_id = response.json()["request_id"]
    header_request_id = response.headers.get("X-Request-ID")

    assert header_request_id
    assert header_request_id == body_request_id


def test_validation_error_returns_unified_error_response():
    response = client.post(
        "/v1/generate",
        json={
            "request": "",
            "model": "auto",
        },
    )

    assert response.status_code == 422

    payload = response.json()

    assert payload["code"] == "VALIDATION_ERROR"
    assert payload["message"] == "Request validation failed."
    assert payload["request_id"]
    assert response.headers["X-Request-ID"] == payload["request_id"]


def test_supplied_request_id_is_preserved():
    request_id = "req-test-supplied-123"

    response = client.post(
        "/v1/generate",
        headers={"X-Request-ID": request_id},
        json={
            "request": "Explain RAG.",
            "model": "auto",
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["request_id"] == request_id
    assert response.headers["X-Request-ID"] == request_id


def test_supplied_request_id_is_preserved_on_validation_error():
    request_id = "req-validation-456"

    response = client.post(
        "/v1/generate",
        headers={"X-Request-ID": request_id},
        json={
            "request": "",
            "model": "auto",
        },
    )

    assert response.status_code == 422

    payload = response.json()

    assert payload["code"] == "VALIDATION_ERROR"
    assert payload["request_id"] == request_id
    assert response.headers["X-Request-ID"] == request_id
