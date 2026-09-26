from dataclasses import dataclass
from pathlib import Path

from app.chunking import Chunk, chunk_pages
from app.embeddings import EmbeddingProvider
from app.pdf_ingestion import extract_pages
from app.retrieval import LexicalRetriever
from app.storage import ChunkStore
from app.vector_store import VectorStore


@dataclass(frozen=True)
class IngestionResult:
    document: str
    pages: int
    chunks: int
    embeddings: int


class DocumentIngestor:
    """Ingests PDF documents into chunk and vector storage."""

    def __init__(
        self,
        chunk_store: ChunkStore,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        lexical_retriever: LexicalRetriever | None = None,
        chunk_size: int = 500,
        overlap: int = 100,
    ):
        self.chunk_store = chunk_store
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.lexical_retriever = lexical_retriever
        self.chunk_size = chunk_size
        self.overlap = overlap

    def ingest(self, pdf_path: str | Path) -> IngestionResult:
        path = Path(pdf_path)

        pages = extract_pages(path)

        chunks = chunk_pages(
            pages,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
        )

        self.chunk_store.delete_document(path.name)
        self.vector_store.delete_document(path.name)

        if self.lexical_retriever is not None:
            self.lexical_retriever.delete_document(path.name)

        self.chunk_store.add_chunks(chunks)

        if self.lexical_retriever is not None:
            self.lexical_retriever.add_chunks(chunks)

        for chunk in chunks:
            vector = self.embedding_provider.embed(chunk.text)

            self.vector_store.add_embedding(
                chunk.chunk_id,
                vector,
            )

        return IngestionResult(
            document=path.name,
            pages=len(pages),
            chunks=len(chunks),
            embeddings=len(chunks),
        )
