import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field, field_validator

from app.embeddings import SentenceTransformerEmbeddingProvider
from app.generation import GroundedAnswer
from app.graph import RAGGraph
from app.ingestion import DocumentIngestor, IngestionResult
from app.rag_service import build_rag_graph
from app.retrieval import LexicalRetriever
from app.storage import ChunkStore, DEFAULT_DB_PATH
from app.vector_store import VectorStore


MAX_QUERY_LENGTH = 4_000
MAX_UPLOAD_SIZE = 10 * 1024 * 1024
UPLOAD_CHUNK_SIZE = 1024 * 1024


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_QUERY_LENGTH)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query cannot be blank")

        return value


class CitationResponse(BaseModel):
    document: str
    page_number: int


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]


class IngestionResponse(BaseModel):
    document: str
    pages: int
    chunks: int
    embeddings: int


def create_app(
    graph: RAGGraph | None = None,
    ingestor: DocumentIngestor | None = None,
) -> FastAPI:
    app = FastAPI(
        title="Production RAG API",
        version="0.1.0",
    )

    rag_graph = graph
    document_ingestor = ingestor

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/query", response_model=QueryResponse)
    def query(request: QueryRequest) -> QueryResponse:
        nonlocal rag_graph

        if rag_graph is None:
            rag_graph = build_rag_graph()

        result: GroundedAnswer = rag_graph.invoke(request.query)

        return QueryResponse(
            answer=result.answer,
            citations=[
                CitationResponse(
                    document=citation.document,
                    page_number=citation.page_number,
                )
                for citation in result.citations
            ],
        )

    @app.post("/ingest", response_model=IngestionResponse)
    def ingest(file: UploadFile = File(...)) -> IngestionResponse:
        nonlocal document_ingestor

        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="A PDF filename is required",
            )

        if Path(file.filename).suffix.lower() != ".pdf":
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported",
            )

        safe_filename = Path(file.filename).name

        if not safe_filename:
            raise HTTPException(
                status_code=400,
                detail="A valid PDF filename is required",
            )

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory) / safe_filename

            total_size = 0

            with temporary_path.open("wb") as temporary_file:
                while True:
                    chunk = file.file.read(UPLOAD_CHUNK_SIZE)

                    if not chunk:
                        break

                    total_size += len(chunk)

                    if total_size > MAX_UPLOAD_SIZE:
                        raise HTTPException(
                            status_code=413,
                            detail="PDF upload exceeds the 10 MiB limit",
                        )

                    temporary_file.write(chunk)

            with temporary_path.open("rb") as uploaded_file:
                if uploaded_file.read(5) != b"%PDF-":
                    raise HTTPException(
                        status_code=400,
                        detail="Uploaded file is not a valid PDF",
                    )

            if document_ingestor is None:
                embedding_provider = SentenceTransformerEmbeddingProvider()

                vector_store = VectorStore(
                    DEFAULT_DB_PATH,
                    dimensions=embedding_provider.dimensions,
                )

                lexical_retriever = LexicalRetriever(DEFAULT_DB_PATH)

                document_ingestor = DocumentIngestor(
                    chunk_store=ChunkStore(DEFAULT_DB_PATH),
                    vector_store=vector_store,
                    embedding_provider=embedding_provider,
                    lexical_retriever=lexical_retriever,
                )

            result: IngestionResult = document_ingestor.ingest(
                temporary_path
            )

        return IngestionResponse(
            document=result.document,
            pages=result.pages,
            chunks=result.chunks,
            embeddings=result.embeddings,
        )

    return app


app = create_app()
