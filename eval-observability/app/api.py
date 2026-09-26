import os

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.service import EvaluationService
from app.storage import SQLiteEventStore


app = FastAPI(title="Evaluation + Observability")

db_path = os.getenv("EVAL_DB_PATH", "eval-observability.db")

store = SQLiteEventStore(db_path)
service = EvaluationService(store=store)


class EvaluationRequest(BaseModel):
    request: str
    response: str
    model: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    latency_ms: float = Field(ge=0)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/evaluate")
def evaluate(payload: EvaluationRequest):
    result = service.process(
        request=payload.request,
        response=payload.response,
        model=payload.model,
        provider=payload.provider,
        latency_ms=payload.latency_ms,
    )

    return {
        "evaluation": {
            "passed": result.evaluation.passed,
            "score": result.evaluation.score,
            "reason": result.evaluation.reason,
        },
        "observability": {
            "request": result.observability.request,
            "model": result.observability.model,
            "provider": result.observability.provider,
            "latency_ms": result.observability.latency_ms,
            "success": result.observability.success,
        },
    }


@app.get("/events")
def events():
    return {
        "events": store.list_events(),
    }


@app.get("/metrics")
def metrics():
    return store.metrics()
