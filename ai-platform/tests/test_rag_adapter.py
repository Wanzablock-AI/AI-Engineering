import httpx
import pytest

from app.rag_adapter import RAGAdapter, RAGCitation


def test_rag_adapter_query(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "answer": "Input validation checks untrusted input.",
                "citations": [
                    {
                        "document": "docker-rag-test.pdf",
                        "page_number": 1,
                    }
                ],
            }

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(httpx, "post", fake_post)

    adapter = RAGAdapter("http://production-rag:8000")

    result = adapter.query("What does input validation do?")

    assert result.answer == "Input validation checks untrusted input."
    assert result.citations == [
        RAGCitation(
            document="docker-rag-test.pdf",
            page_number=1,
        )
    ]
    assert captured["url"] == "http://production-rag:8000/query"
    assert captured["json"] == {
        "query": "What does input validation do?"
    }
    assert captured["timeout"] == 120.0


def test_rag_adapter_propagates_http_error(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            raise httpx.HTTPStatusError(
                "service unavailable",
                request=httpx.Request(
                    "POST",
                    "http://production-rag:8000/query",
                ),
                response=httpx.Response(
                    503,
                    request=httpx.Request(
                        "POST",
                        "http://production-rag:8000/query",
                    ),
                ),
            )

    monkeypatch.setattr(
        httpx,
        "post",
        lambda *args, **kwargs: FakeResponse(),
    )

    adapter = RAGAdapter("http://production-rag:8000")

    with pytest.raises(httpx.HTTPStatusError):
        adapter.query("test")
