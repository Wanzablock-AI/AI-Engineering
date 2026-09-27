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
