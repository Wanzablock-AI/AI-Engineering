from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.config import Settings
from app.ollama import OllamaProvider
from app.router import ModelRouter


app = FastAPI(title="Model Router")

router = ModelRouter()
settings = Settings()
provider = OllamaProvider(base_url=settings.ollama_base_url)


class RouteRequest(BaseModel):
    request: str
    model: str | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def models():
    return router.ROUTES


@app.post("/route")
def route(request: RouteRequest):
    try:
        decision = router.route_request(
            request.request,
            model=request.model,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    return {
        "model": decision.model,
        "reason": decision.reason,
    }


@app.post("/generate")
def generate(request: RouteRequest):
    try:
        decision = router.route_request(
            request.request,
            model=request.model,
        )
        try:
            answer = provider.generate(
                model=decision.model,
                prompt=request.request,
            )
        except RuntimeError as exc:
            raise HTTPException(
                status_code=502,
                detail="Model provider unavailable",
            ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    return {
        "model": decision.model,
        "answer": answer,
    }
