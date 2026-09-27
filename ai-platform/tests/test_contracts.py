import pytest

from app.contracts import (
    ErrorResponse,
    GenerateRequest,
    GenerateResponse,
    ResearchRequest,
    ResearchResponse,
)


def test_generate_request_accepts_valid_payload():
    payload = GenerateRequest(
        request="Explain RAG.",
        model="auto",
    )

    assert payload.request == "Explain RAG."
    assert payload.model == "auto"


def test_generate_request_rejects_empty_request():
    with pytest.raises(ValueError):
        GenerateRequest(
            request="",
            model="auto",
        )


def test_research_request_accepts_valid_payload():
    payload = ResearchRequest(
        request="Explain vector databases in RAG.",
        questions=[
            "What is a vector database?",
            "Why is it useful for RAG?",
        ],
    )

    assert payload.request == "Explain vector databases in RAG."
    assert len(payload.questions) == 2


def test_research_request_rejects_empty_questions():
    with pytest.raises(ValueError):
        ResearchRequest(
            request="Explain RAG.",
            questions=[],
        )


def test_generate_response_contains_request_metadata():
    response = GenerateResponse(
        request_id="req-123",
        request="Explain RAG.",
        answer="RAG combines retrieval with generation.",
        model="granite4.1:3b",
        provider="ollama",
        latency_ms=1200.5,
    )

    assert response.request_id == "req-123"
    assert response.model == "granite4.1:3b"
    assert response.provider == "ollama"
    assert response.latency_ms == 1200.5


def test_research_response_contains_report():
    response = ResearchResponse(
        request_id="req-456",
        request="Explain RAG.",
        synthesis="RAG retrieves relevant information before generation.",
        results=[],
    )

    assert response.request_id == "req-456"
    assert response.synthesis.startswith("RAG")


def test_error_response_contains_request_id():
    response = ErrorResponse(
        code="SERVICE_UNAVAILABLE",
        message="Model service unavailable.",
        request_id="req-789",
    )

    assert response.code == "SERVICE_UNAVAILABLE"
    assert response.request_id == "req-789"
