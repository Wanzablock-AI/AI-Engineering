from fastapi.testclient import TestClient

from app.api import create_app
from app.generation import Citation, GroundedAnswer


class FakeGraph:
    def invoke(self, query: str) -> GroundedAnswer:
        return GroundedAnswer(
            answer=f"Grounded answer for: {query}",
            citations=[
                Citation(
                    document="security-guide.pdf",
                    page_number=1,
                ),
            ],
        )


def test_health():
    client = TestClient(create_app(FakeGraph()))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_returns_grounded_answer_and_citations():
    client = TestClient(create_app(FakeGraph()))

    response = client.post(
        "/query",
        json={"query": "What does input validation do?"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Grounded answer for: What does input validation do?",
        "citations": [
            {
                "document": "security-guide.pdf",
                "page_number": 1,
            }
        ],
    }


def test_query_rejects_empty_query():
    client = TestClient(create_app(FakeGraph()))

    response = client.post(
        "/query",
        json={"query": ""},
    )

    assert response.status_code == 422


def test_query_rejects_blank_query():
    client = TestClient(create_app(FakeGraph()))

    response = client.post(
        "/query",
        json={"query": "   "},
    )

    assert response.status_code == 422
