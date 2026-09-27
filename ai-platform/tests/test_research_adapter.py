import httpx
import pytest

from app.research_adapter import ResearchAdapter


def test_research_adapter_parses_report(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "request": "Explain RAG.",
                "results": [
                    {
                        "task_id": "task-1",
                        "question": "What is RAG?",
                        "answer": "Retrieval-augmented generation.",
                        "success": True,
                    }
                ],
                "synthesis": "RAG combines retrieval with generation.",
            }

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(httpx, "post", fake_post)

    result = ResearchAdapter(
        "http://research:8000",
        timeout=45.0,
    ).research(
        request="Explain RAG.",
        questions=["What is RAG?"],
    )

    assert captured["url"] == "http://research:8000/research"
    assert captured["json"] == {
        "request": "Explain RAG.",
        "questions": ["What is RAG?"],
    }
    assert captured["timeout"] == 45.0
    assert result.synthesis == "RAG combines retrieval with generation."
    assert result.results[0].task_id == "task-1"
    assert result.results[0].success is True


def test_research_adapter_propagates_http_error(monkeypatch):
    def fake_post(*args, **kwargs):
        request = httpx.Request("POST", "http://research:8000/research")
        response = httpx.Response(500, request=request)
        return httpx.MockTransport(lambda _: response).handle_request(request)

    monkeypatch.setattr(httpx, "post", fake_post)

    adapter = ResearchAdapter("http://research:8000")

    with pytest.raises(httpx.HTTPStatusError):
        adapter.research(
            request="Explain RAG.",
            questions=["What is RAG?"],
        )
