from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi import HTTPException
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
from app.errors import (
    REQUEST_ID_HEADER,
    get_request_id,
    validation_exception_handler,
    http_exception_handler,
    unhandled_exception_handler,
)
from app.security_adapter import SecurityAdapter
from app.model_router_adapter import ModelRouterAdapter
from app.evaluation_adapter import EvaluationAdapter
from app.rag_adapter import RAGAdapter
from app.research_adapter import ResearchAdapter


app = FastAPI(title="AI Engineering Platform")

app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)

@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get(REQUEST_ID_HEADER) or generate_request_id()
    request.state.request_id = request_id

    response = await call_next(request)
    response.headers[REQUEST_ID_HEADER] = request_id
    return response

security_adapter = SecurityAdapter("http://security-middleware:8000")
model_router_adapter = ModelRouterAdapter("http://model-router:8000")
evaluation_adapter = EvaluationAdapter("http://eval-observability:8000")
rag_adapter = RAGAdapter("http://production-rag:8000")
research_adapter = ResearchAdapter("http://multi-agent-research:8000")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/generate", response_model=GenerateResponse)
def generate(payload: GenerateRequest, response: Response, request: Request):
    request_id = get_request_id(request)

    security = security_adapter.check(payload.request)

    if not security.allowed:
        return JSONResponse(
            status_code=403,
            content={
                "code": "SECURITY_BLOCKED",
                "message": security.reason,
                "request_id": request_id,
            },
            headers={"X-Request-ID": request_id},
        )

    import time

    started = time.perf_counter()
    result = model_router_adapter.generate(
        payload.request,
        model=payload.model,
    )
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
def rag_query(payload: RAGRequest, response: Response, request: Request):
    request_id = get_request_id(request)

    security = security_adapter.check(payload.request)

    if not security.allowed:
        return JSONResponse(
            status_code=403,
            content={
                "code": "SECURITY_BLOCKED",
                "message": security.reason,
                "request_id": request_id,
            },
            headers={"X-Request-ID": request_id},
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
def research(payload: ResearchRequest, response: Response, request: Request):
    request_id = get_request_id(request)

    security = security_adapter.check(payload.request)

    if not security.allowed:
        return JSONResponse(
            status_code=403,
            content={
                "code": "SECURITY_BLOCKED",
                "message": security.reason,
                "request_id": request_id,
            },
            headers={"X-Request-ID": request_id},
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
