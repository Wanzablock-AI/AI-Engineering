from fastapi.testclient import TestClient

from app.api import app
from app.storage import SQLiteEventStore


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_evaluate_returns_evaluation_and_observability():
    response = client.post(
        "/evaluate",
        json={
            "request": "Explain Python lists",
            "response": "Python lists are ordered collections.",
            "model": "qwen2.5-coder:3b",
            "provider": "ollama",
            "latency_ms": 125.5,
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["evaluation"]["passed"] is True
    assert payload["evaluation"]["score"] == 1.0
    assert payload["evaluation"]["reason"] == "response_present"

    assert payload["observability"]["model"] == "qwen2.5-coder:3b"
    assert payload["observability"]["provider"] == "ollama"
    assert payload["observability"]["latency_ms"] == 125.5
    assert payload["observability"]["success"] is True


def test_evaluate_empty_response_fails():
    response = client.post(
        "/evaluate",
        json={
            "request": "Explain Python lists",
            "response": "",
            "model": "granite4.1:3b",
            "provider": "ollama",
            "latency_ms": 200.0,
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["evaluation"]["passed"] is False
    assert payload["evaluation"]["score"] == 0.0
    assert payload["evaluation"]["reason"] == "empty_response"
    assert payload["observability"]["success"] is False


def test_events_returns_persisted_events():
    response = client.post(
        "/evaluate",
        json={
            "request": "Test event",
            "response": "Answer",
            "model": "llama3.2:1b",
            "provider": "ollama",
            "latency_ms": 50.0,
        },
    )

    assert response.status_code == 200

    response = client.get("/events")

    assert response.status_code == 200

    events = response.json()["events"]

    assert len(events) >= 1
    assert events[-1]["request"] == "Test event"


def test_evaluate_rejects_negative_latency():
    response = client.post(
        "/evaluate",
        json={
            "request": "Test",
            "response": "Answer",
            "model": "llama3.2:1b",
            "provider": "ollama",
            "latency_ms": -1.0,
        },
    )

    assert response.status_code == 422


def test_evaluate_rejects_empty_model():
    response = client.post(
        "/evaluate",
        json={
            "request": "Test",
            "response": "Answer",
            "model": "",
            "provider": "ollama",
            "latency_ms": 50.0,
        },
    )

    assert response.status_code == 422


def test_evaluate_rejects_empty_provider():
    response = client.post(
        "/evaluate",
        json={
            "request": "Test",
            "response": "Answer",
            "model": "llama3.2:1b",
            "provider": "",
            "latency_ms": 50.0,
        },
    )

    assert response.status_code == 422


def test_metrics_returns_event_summary():
    response = client.get("/events")
    assert response.status_code == 200

    events = response.json()["events"]

    response = client.get("/metrics")
    assert response.status_code == 200

    metrics = response.json()

    assert metrics["total_events"] == len(events)
    assert metrics["successful_events"] == sum(
        1 for event in events if event["success"]
    )
    assert metrics["failed_events"] == sum(
        1 for event in events if not event["success"]
    )
    assert 0.0 <= metrics["success_rate"] <= 1.0
    assert metrics["average_latency_ms"] >= 0.0
