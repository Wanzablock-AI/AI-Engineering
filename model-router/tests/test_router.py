import pytest

from app.router import ModelRouter


def test_routes_coding_requests():
    router = ModelRouter()

    decision = router.route("coding")

    assert decision.model == "qwen2.5-coder:3b"
    assert decision.reason == "coding"


def test_routes_rag_requests():
    router = ModelRouter()

    decision = router.route("rag")

    assert decision.model == "granite4.1:3b"
    assert decision.reason == "rag"


def test_routes_lightweight_requests():
    router = ModelRouter()

    decision = router.route("lightweight")

    assert decision.model == "llama3.2:1b"
    assert decision.reason == "lightweight"


def test_routes_general_requests():
    router = ModelRouter()

    decision = router.route("general")

    assert decision.model == "llama3.2:3b"
    assert decision.reason == "general"


def test_unsupported_task_is_rejected():
    router = ModelRouter()

    with pytest.raises(ValueError):
        router.route("unknown")


def test_task_names_are_normalized():
    router = ModelRouter()

    assert router.route(" CODING ").model == "qwen2.5-coder:3b"
    assert router.route("RAG").model == "granite4.1:3b"


def test_empty_task_is_rejected():
    router = ModelRouter()

    with pytest.raises(ValueError):
        router.route("")


def test_none_task_is_rejected():
    router = ModelRouter()

    with pytest.raises((ValueError, AttributeError, TypeError)):
        router.route(None)


def test_classifies_coding_request():
    router = ModelRouter()

    decision = router.route_request("Write a Python function that parses JSON")

    assert decision.model == "qwen2.5-coder:3b"
    assert decision.reason == "coding"


def test_classifies_rag_request():
    router = ModelRouter()

    decision = router.route_request(
        "According to the documents, what is the database backup policy?"
    )

    assert decision.model == "granite4.1:3b"
    assert decision.reason == "rag"


def test_classifies_lightweight_request():
    router = ModelRouter()

    decision = router.route_request("What time is it?")

    assert decision.model == "llama3.2:1b"
    assert decision.reason == "lightweight"


def test_classifies_general_request():
    router = ModelRouter()

    decision = router.route_request(
        "Explain why distributed systems use consensus algorithms."
    )

    assert decision.model == "llama3.2:3b"
    assert decision.reason == "general"


def test_general_technical_question_is_not_coding():
    router = ModelRouter()

    decision = router.route_request(
        "Why do distributed systems use consensus algorithms?"
    )

    assert decision.reason == "general"


def test_coding_request_takes_precedence_over_document_reference():
    router = ModelRouter()

    decision = router.route_request(
        "According to the documents, write Python code to parse the results."
    )

    assert decision.reason == "coding"


def test_empty_request_is_rejected():
    router = ModelRouter()

    with pytest.raises(ValueError):
        router.route_request("")


def test_non_string_request_is_rejected():
    router = ModelRouter()

    with pytest.raises(ValueError):
        router.route_request(None)


def test_document_question_routes_to_rag():
    router = ModelRouter()

    decision = router.route_request(
        "What does the provided file say about security?"
    )

    assert decision.reason == "rag"
