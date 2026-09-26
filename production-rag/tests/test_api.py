import pymupdf

from fastapi.testclient import TestClient

from app.api import create_app
from app.generation import Citation, GroundedAnswer
from app.ingestion import IngestionResult


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


class FakeIngestor:
    def __init__(self):
        self.received_path = None

    def ingest(self, pdf_path):
        self.received_path = pdf_path

        return IngestionResult(
            document="security.pdf",
            pages=2,
            chunks=2,
            embeddings=2,
        )


def test_ingest_returns_ingestion_result(tmp_path):
    pdf_path = tmp_path / "security.pdf"

    document = pymupdf.open()
    document.new_page()
    document.new_page()
    document.save(pdf_path)
    document.close()

    ingestor = FakeIngestor()
    client = TestClient(create_app(FakeGraph(), ingestor))

    with pdf_path.open("rb") as file:
        response = client.post(
            "/ingest",
            files={
                "file": (
                    "security.pdf",
                    file,
                    "application/pdf",
                )
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "document": "security.pdf",
        "pages": 2,
        "chunks": 2,
        "embeddings": 2,
    }

    assert ingestor.received_path is not None


def test_ingest_rejects_non_pdf():
    ingestor = FakeIngestor()
    client = TestClient(create_app(FakeGraph(), ingestor))

    response = client.post(
        "/ingest",
        files={
            "file": (
                "security.txt",
                b"not a pdf",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Only PDF files are supported"
    }


def test_query_rejects_oversized_query():
    client = TestClient(create_app(FakeGraph()))

    response = client.post(
        "/query",
        json={"query": "x" * 4001},
    )

    assert response.status_code == 422


def test_ingest_rejects_non_pdf_content():
    ingestor = FakeIngestor()
    client = TestClient(create_app(FakeGraph(), ingestor))

    response = client.post(
        "/ingest",
        files={
            "file": (
                "security.pdf",
                b"this is not a PDF",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Uploaded file is not a valid PDF"
    }


def test_ingest_rejects_oversized_pdf():
    ingestor = FakeIngestor()
    client = TestClient(create_app(FakeGraph(), ingestor))

    oversized_pdf = b"%PDF-" + b"x" * (10 * 1024 * 1024)

    response = client.post(
        "/ingest",
        files={
            "file": (
                "security.pdf",
                oversized_pdf,
                "application/pdf",
            )
        },
    )

    assert response.status_code == 413
    assert response.json() == {
        "detail": "PDF upload exceeds the 10 MiB limit"
    }

    assert ingestor.received_path is None


def test_ingest_rejects_invalid_pdf_without_initializing_embedding_model(
    monkeypatch,
):
    initialized = False

    class FailingEmbeddingProvider:
        def __init__(self):
            nonlocal initialized
            initialized = True
            raise AssertionError(
                "embedding model should not initialize for rejected upload"
            )

    monkeypatch.setattr(
        "app.api.SentenceTransformerEmbeddingProvider",
        FailingEmbeddingProvider,
    )

    client = TestClient(create_app(FakeGraph()))

    response = client.post(
        "/ingest",
        files={
            "file": (
                "security.pdf",
                b"not a PDF",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Uploaded file is not a valid PDF"
    }
    assert initialized is False
