from app.service import EvaluationService


def test_service_evaluates_and_records_request():
    service = EvaluationService()

    result = service.process(
        request="Explain Python lists",
        response="Python lists are ordered collections.",
        model="qwen2.5-coder:3b",
        provider="ollama",
        latency_ms=125.5,
    )

    assert result.evaluation.passed is True
    assert result.evaluation.score == 1.0
    assert result.evaluation.reason == "response_present"

    assert result.observability.request == "Explain Python lists"
    assert result.observability.model == "qwen2.5-coder:3b"
    assert result.observability.provider == "ollama"
    assert result.observability.latency_ms == 125.5
    assert result.observability.success is True


def test_service_records_failed_evaluation():
    service = EvaluationService()

    result = service.process(
        request="Explain Python lists",
        response="",
        model="granite4.1:3b",
        provider="ollama",
        latency_ms=200.0,
    )

    assert result.evaluation.passed is False
    assert result.evaluation.score == 0.0
    assert result.evaluation.reason == "empty_response"

    assert result.observability.success is False


def test_service_keeps_observability_history():
    service = EvaluationService()

    service.process(
        request="First",
        response="Answer",
        model="llama3.2:1b",
        provider="ollama",
        latency_ms=50.0,
    )

    service.process(
        request="Second",
        response="",
        model="llama3.2:3b",
        provider="ollama",
        latency_ms=100.0,
    )

    assert len(service.observability.events) == 2


class FakeStore:
    def __init__(self):
        self.saved = []

    def save(
        self,
        request,
        model,
        provider,
        latency_ms,
        success,
    ):
        self.saved.append(
            {
                "request": request,
                "model": model,
                "provider": provider,
                "latency_ms": latency_ms,
                "success": success,
            }
        )


def test_service_persists_observability_event():
    store = FakeStore()

    service = EvaluationService(store=store)

    service.process(
        request="Explain Python lists",
        response="Python lists are ordered collections.",
        model="qwen2.5-coder:3b",
        provider="ollama",
        latency_ms=125.5,
    )

    assert len(store.saved) == 1
    assert store.saved[0]["request"] == "Explain Python lists"
    assert store.saved[0]["model"] == "qwen2.5-coder:3b"
    assert store.saved[0]["success"] is True
