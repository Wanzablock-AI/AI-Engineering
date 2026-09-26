from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

from app.generation import GroundedAnswer
from app.graph import RAGGraph
from app.rag_service import build_rag_graph


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)

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


def create_app(graph: RAGGraph | None = None) -> FastAPI:
    app = FastAPI(
        title="Production RAG API",
        version="0.1.0",
    )

    rag_graph = graph

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

    return app


app = create_app()
