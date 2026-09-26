from fastapi import FastAPI
from pydantic import BaseModel

from app.config import Settings
from app.gateway import SecurityGateway
from app.model_router import ModelRouterClient
from app.security import SecurityMiddleware


app = FastAPI(title="Security Middleware")

settings = Settings()
security = SecurityMiddleware()

provider = ModelRouterClient(
    base_url=settings.model_router_base_url,
)

gateway = SecurityGateway(
    provider=provider,
    security=security,
)


class SecurityRequest(BaseModel):
    request: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/check")
def check(request: SecurityRequest):
    decision = security.check(request.request)

    return {
        "allowed": decision.allowed,
        "risk_level": decision.risk_level,
        "reason": decision.reason,
    }


@app.post("/generate")
def generate(request: SecurityRequest):
    result = gateway.handle(request.request)

    return {
        "allowed": result.allowed,
        "risk_level": result.risk_level,
        "reason": result.reason,
        "answer": result.answer,
    }
