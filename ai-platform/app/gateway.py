from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse

from app.contracts import (
    GenerateRequest,
    GenerateResponse,
    RAGRequest,
    RAGResponse,
    RAGCitationResponse,
    ResearchRequest,
    ResearchResponse,
    ResearchResultResponse,
)
from app.request_id import generate_request_id
from app.security_adapter import SecurityAdapter
from app.model_router_adapter import ModelRouterAdapter
from app.evaluation_adapter import EvaluationAdapter
from app.rag_adapter import RAGAdapter
from app.research_adapter import ResearchAdapter


app = FastAPI(title="AI Engineering Platform")

security_adapter = SecurityAdapter("http://security-middleware:8000")
model_router_adapter = ModelRouterAdapter("http://model-router:8000")
evaluation_adapter = EvaluationAdapter("http://eval-observability:8000")
rag_adapter = RAGAdapter("http://production-rag:8000")
research_adapter = ResearchAdapter("http://multi-agent-research:8000")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/generate", response_model=GenerateResponse)
def generate(payload: GenerateRequest, response: Response):
    request_id = generate_request_id()
    response.headers["X-Request-ID"] = request_id

    security = security_adapter.check(payload.request)

    if not security.allowed:
        return JSONResponse(
            status_code=403,
            content={
                "detail": {
                    "code": "SECURITY_BLOCKED",
                    "reason": security.reason,
                }
            },
        )

    import time

    started = time.perf_counter()
    result = model_router_adapter.generate(payload.request)
    latency_ms = (time.perf_counter() - started) * 1000

    evaluation_adapter.evaluate(
        request=payload.request,
        response=result.answer,
        model=result.model,
        provider="model-router",
        latency_ms=latency_ms,
    )

    return GenerateResponse(
        request_id=request_id,
        request=payload.request,
        answer=result.answer,
        model=result.model,
        provider="model-router",
        latency_ms=latency_ms,
    )


@app.post("/v1/rag/query", response_model=RAGResponse)
def rag_query(payload: RAGRequest, response: Response):
    request_id = generate_request_id()
    response.headers["X-Request-ID"] = request_id

    security = security_adapter.check(payload.request)

    if not security.allowed:
        return JSONResponse(
            status_code=403,
            content={
                "detail": {
                    "code": "SECURITY_BLOCKED",
                    "reason": security.reason,
                }
            },
        )

    result = rag_adapter.query(payload.request)

    return RAGResponse(
        request_id=request_id,
        request=payload.request,
        answer=result.answer,
        citations=[
            RAGCitationResponse(
                document=citation.document,
                page_number=citation.page_number,
            )
            for citation in result.citations
        ],
    )


@app.post("/v1/research", response_model=ResearchResponse)
def research(payload: ResearchRequest, response: Response):
    request_id = generate_request_id()
    response.headers["X-Request-ID"] = request_id

    security = security_adapter.check(payload.request)

    if not security.allowed:
        return JSONResponse(
            status_code=403,
            content={
                "detail": {
                    "code": "SECURITY_BLOCKED",
                    "reason": security.reason,
                }
            },
        )

    result = research_adapter.research(
        request=payload.request,
        questions=payload.questions,
    )

    return ResearchResponse(
        request_id=request_id,
        request=result.request,
        synthesis=result.synthesis,
        results=[
            ResearchResultResponse(
                task_id=item.task_id,
                question=item.question,
                answer=item.answer,
                success=item.success,
            )
            for item in result.results
        ],
    )
