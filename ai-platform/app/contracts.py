from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    request: str = Field(min_length=1)
    model: str = Field(min_length=1)


class RAGRequest(BaseModel):
    request: str = Field(min_length=1)


class RAGCitationResponse(BaseModel):
    document: str
    page_number: int


class RAGResponse(BaseModel):
    request_id: str = Field(min_length=1)
    request: str
    answer: str
    citations: list[RAGCitationResponse]


class ResearchRequest(BaseModel):
    request: str = Field(min_length=1)
    questions: list[str] = Field(min_length=1)


class GenerateResponse(BaseModel):
    request_id: str = Field(min_length=1)
    request: str
    answer: str
    model: str
    provider: str
    latency_ms: float = Field(ge=0)


class ResearchResponse(BaseModel):
    request_id: str = Field(min_length=1)
    request: str
    synthesis: str
    results: list


class ErrorResponse(BaseModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
