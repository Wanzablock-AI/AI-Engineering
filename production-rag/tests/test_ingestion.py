import pymupdf

from app.embeddings import DeterministicEmbeddingProvider
from app.ingestion import DocumentIngestor
from app.storage import ChunkStore
from app.retrieval import LexicalRetriever
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


def test_ingest_replaces_existing_document(tmp_path):
    pdf_path = tmp_path / "security.pdf"
    db_path = tmp_path / "test.db"

    document = pymupdf.open()

    page = document.new_page()
    page.insert_text((72, 72), "First document page.")

    page = document.new_page()
    page.insert_text((72, 72), "Second document page.")

    page = document.new_page()
    page.insert_text((72, 72), "Third document page.")

    document.save(pdf_path)
    document.close()

    chunk_store = ChunkStore(db_path)
    embedding_provider = DeterministicEmbeddingProvider(dimensions=8)
    vector_store = VectorStore(
        db_path,
        dimensions=embedding_provider.dimensions,
    )

    lexical_retriever = LexicalRetriever(db_path)

    ingestor = DocumentIngestor(
        chunk_store=chunk_store,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        lexical_retriever=lexical_retriever,
    )

    first_result = ingestor.ingest(pdf_path)

    assert len(lexical_retriever.search("First")) == 1
    assert len(lexical_retriever.search("Third")) == 1

    assert first_result.pages == 3
    assert first_result.chunks == 3
    assert first_result.embeddings == 3

    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Replacement document page one.")

    page = document.new_page()
    page.insert_text((72, 72), "Replacement document page two.")

    document.save(pdf_path)
    document.close()

    second_result = ingestor.ingest(pdf_path)

    assert second_result.document == "security.pdf"
    assert second_result.pages == 2
    assert second_result.chunks == 2
    assert second_result.embeddings == 2

    assert chunk_store.count() == 2
    assert vector_store.count() == 2

    assert chunk_store.get_chunk("security.pdf:3:0") is None
    assert vector_store.get_embedding("security.pdf:3:0") is None

    replacement = chunk_store.get_chunk("security.pdf:1:0")

    assert replacement is not None
    assert "Replacement document" in replacement.text

    assert lexical_retriever.search("Third") == []

    replacement_results = lexical_retriever.search("Replacement")

    assert len(replacement_results) == 2
    assert all(
        result.document == "security.pdf"
        for result in replacement_results
    )


def test_ingest_rejects_pdf_without_text_before_replacing_existing_document(
    tmp_path,
):
    pdf_path = tmp_path / "security.pdf"
    db_path = tmp_path / "test.db"

    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Existing indexed document.")
    document.save(pdf_path)
    document.close()

    chunk_store = ChunkStore(db_path)
    embedding_provider = DeterministicEmbeddingProvider(dimensions=8)
    vector_store = VectorStore(
        db_path,
        dimensions=embedding_provider.dimensions,
    )
    lexical_retriever = LexicalRetriever(db_path)

    ingestor = DocumentIngestor(
        chunk_store=chunk_store,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        lexical_retriever=lexical_retriever,
    )

    ingestor.ingest(pdf_path)

    document = pymupdf.open()
    document.new_page()
    document.save(pdf_path)
    document.close()

    import pytest

    with pytest.raises(ValueError, match="No text chunks"):
        ingestor.ingest(pdf_path)

    assert chunk_store.count() == 1
    assert vector_store.count() == 1
    assert lexical_retriever.search("Existing") != []


class FailingEmbeddingProvider:
    dimensions = 8

    def __init__(self):
        self.calls = 0

    def embed(self, text):
        self.calls += 1

        if self.calls == 2:
            raise RuntimeError("embedding failed")

        return [0.1] * self.dimensions


def test_ingest_preserves_existing_document_when_embedding_fails(
    tmp_path,
):
    pdf_path = tmp_path / "security.pdf"
    db_path = tmp_path / "test.db"

    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Original indexed document.")
    document.save(pdf_path)
    document.close()

    chunk_store = ChunkStore(db_path)
    initial_provider = DeterministicEmbeddingProvider(dimensions=8)
    vector_store = VectorStore(
        db_path,
        dimensions=initial_provider.dimensions,
    )
    lexical_retriever = LexicalRetriever(db_path)

    initial_ingestor = DocumentIngestor(
        chunk_store=chunk_store,
        vector_store=vector_store,
        embedding_provider=initial_provider,
        lexical_retriever=lexical_retriever,
    )

    initial_ingestor.ingest(pdf_path)

    document = pymupdf.open()

    page = document.new_page()
    page.insert_text((72, 72), "Replacement page one.")

    page = document.new_page()
    page.insert_text((72, 72), "Replacement page two.")

    document.save(pdf_path)
    document.close()

    failing_ingestor = DocumentIngestor(
        chunk_store=chunk_store,
        vector_store=vector_store,
        embedding_provider=FailingEmbeddingProvider(),
        lexical_retriever=lexical_retriever,
    )

    import pytest

    with pytest.raises(RuntimeError, match="embedding failed"):
        failing_ingestor.ingest(pdf_path)

    assert chunk_store.count() == 1
    assert vector_store.count() == 1

    original = chunk_store.get_chunk("security.pdf:1:0")

    assert original is not None
    assert "Original indexed document" in original.text

    assert lexical_retriever.search("Original") != []
    assert lexical_retriever.search("Replacement") == []
