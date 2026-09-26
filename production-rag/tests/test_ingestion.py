import pymupdf

from app.embeddings import DeterministicEmbeddingProvider
from app.ingestion import DocumentIngestor
from app.storage import ChunkStore
from app.vector_store import VectorStore


def create_test_pdf(path):
    document = pymupdf.open()

    page = document.new_page()
    page.insert_text(
        (72, 72),
        "Input validation protects application boundaries.",
    )

    page = document.new_page()
    page.insert_text(
        (72, 72),
        "Authentication verifies user identity.",
    )

    document.save(path)
    document.close()


def test_ingest_persists_chunks_and_embeddings(tmp_path):
    pdf_path = tmp_path / "security.pdf"
    db_path = tmp_path / "test.db"

    create_test_pdf(pdf_path)

    chunk_store = ChunkStore(db_path)

    embedding_provider = DeterministicEmbeddingProvider(dimensions=8)

    vector_store = VectorStore(
        db_path,
        dimensions=embedding_provider.dimensions,
    )

    ingestor = DocumentIngestor(
        chunk_store=chunk_store,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
    )

    result = ingestor.ingest(pdf_path)

    assert result.document == "security.pdf"
    assert result.pages == 2
    assert result.chunks == 2
    assert result.embeddings == 2

    assert chunk_store.count() == 2
    assert vector_store.count() == 2

    chunk = chunk_store.get_chunk("security.pdf:1:0")

    assert chunk is not None
    assert "Input validation" in chunk.text

    embedding = vector_store.get_embedding("security.pdf:1:0")

    assert embedding is not None
    assert len(embedding) == 8


def test_ingest_skips_empty_pages(tmp_path):
    pdf_path = tmp_path / "mixed.pdf"
    db_path = tmp_path / "test.db"

    document = pymupdf.open()

    page = document.new_page()
    page.insert_text((72, 72), "Useful content.")

    document.new_page()

    document.save(pdf_path)
    document.close()

    chunk_store = ChunkStore(db_path)

    embedding_provider = DeterministicEmbeddingProvider(dimensions=8)

    vector_store = VectorStore(
        db_path,
        dimensions=embedding_provider.dimensions,
    )

    ingestor = DocumentIngestor(
        chunk_store=chunk_store,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
    )

    result = ingestor.ingest(pdf_path)

    assert result.pages == 2
    assert result.chunks == 1
    assert result.embeddings == 1
    assert chunk_store.count() == 1
    assert vector_store.count() == 1
