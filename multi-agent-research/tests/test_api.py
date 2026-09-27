from fastapi.testclient import TestClient

from app.api import app
from app.contracts import ResearchReport, ResearchResult


class FakeOrchestrator:
    def run(
        self,
        request: str,
        questions: list[str],
    ) -> ResearchReport:
        result = ResearchResult(
            task_id="task-1",
            question=questions[0],
            answer="Test research answer.",
            success=True,
        )

        return ResearchReport(
            request=request,
            results=(result,),
            synthesis="Test synthesized answer.",
        )


app.state.orchestrator = FakeOrchestrator()

client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_research_returns_report():
    response = client.post(
        "/research",
        json={
            "request": "Explain RAG.",
            "questions": [
                "What is RAG?",
            ],
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["request"] == "Explain RAG."
    assert payload["synthesis"] == "Test synthesized answer."
    assert len(payload["results"]) == 1
    assert payload["results"][0]["task_id"] == "task-1"
    assert payload["results"][0]["success"] is True


def test_research_rejects_empty_request():
    response = client.post(
        "/research",
        json={
            "request": "",
            "questions": ["What is RAG?"],
        },
    )

    assert response.status_code == 422


def test_research_rejects_empty_questions():
    response = client.post(
        "/research",
        json={
            "request": "Explain RAG.",
            "questions": [],
        },
    )

    assert response.status_code == 422
