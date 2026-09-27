from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.contracts import ResearchReport
from app.orchestrator import ResearchOrchestrator


app = FastAPI(title="Multi-Agent Research")


class ResearchRequest(BaseModel):
    request: str = Field(min_length=1)
    questions: list[str] = Field(min_length=1)


def get_orchestrator() -> ResearchOrchestrator:
    orchestrator = getattr(app.state, "orchestrator", None)

    if orchestrator is None:
        raise RuntimeError("Research orchestrator is not configured")

    return orchestrator


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/research", response_model=ResearchReport)
def research(payload: ResearchRequest):
    orchestrator = get_orchestrator()

    return orchestrator.run(
        request=payload.request,
        questions=payload.questions,
    )
