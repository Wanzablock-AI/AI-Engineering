import pytest


class FakeAllowedSecurityResult:
    allowed = True
    risk_level = "low"
    reason = "allowed"


class FakeAllowedSecurityAdapter:
    def check(self, request):
        return FakeAllowedSecurityResult()


class FakeModelRouterResult:
    model = "llama3.2:3b"
    answer = "Gateway test answer."


class FakeModelRouterAdapter:
    def generate(self, request):
        return FakeModelRouterResult()


@pytest.fixture(autouse=True)
def mock_security_adapter(monkeypatch):
    monkeypatch.setattr(
        "app.gateway.security_adapter",
        FakeAllowedSecurityAdapter(),
    )


@pytest.fixture(autouse=True)
def mock_model_router_adapter(monkeypatch):
    monkeypatch.setattr(
        "app.gateway.model_router_adapter",
        FakeModelRouterAdapter(),
    )


class FakeEvaluationResult:
    passed = True
    score = 1.0
    reason = "response_present"


class FakeEvaluationAdapter:
    def evaluate(self, request, response, model, provider, latency_ms):
        return FakeEvaluationResult()


@pytest.fixture(autouse=True)
def mock_evaluation_adapter(monkeypatch):
    monkeypatch.setattr(
        "app.gateway.evaluation_adapter",
        FakeEvaluationAdapter(),
    )


class FakeRAGCitation:
    document = "test.pdf"
    page_number = 2


class FakeRAGResult:
    answer = "RAG gateway test answer."
    citations = [FakeRAGCitation()]


class FakeRAGAdapter:
    def query(self, request):
        return FakeRAGResult()


@pytest.fixture(autouse=True)
def mock_rag_adapter(monkeypatch):
    monkeypatch.setattr(
        "app.gateway.rag_adapter",
        FakeRAGAdapter(),
    )
