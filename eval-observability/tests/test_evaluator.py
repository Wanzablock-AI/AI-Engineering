import pytest

from app.evaluator import EvaluationResult, Evaluator


def test_evaluator_accepts_non_empty_response():
    evaluator = Evaluator()

    result = evaluator.evaluate(
        request="Explain Python lists",
        response="Python lists are ordered collections.",
    )

    assert isinstance(result, EvaluationResult)
    assert result.passed is True
    assert result.score == 1.0
    assert result.reason == "response_present"


def test_evaluator_rejects_empty_response():
    evaluator = Evaluator()

    result = evaluator.evaluate(
        request="Explain Python lists",
        response="",
    )

    assert result.passed is False
    assert result.score == 0.0
    assert result.reason == "empty_response"


def test_evaluator_rejects_whitespace_response():
    evaluator = Evaluator()

    result = evaluator.evaluate(
        request="Explain Python lists",
        response="   ",
    )

    assert result.passed is False
    assert result.score == 0.0
    assert result.reason == "empty_response"


def test_evaluator_requires_string_request():
    evaluator = Evaluator()

    with pytest.raises(ValueError, match="Request must be a string"):
        evaluator.evaluate(
            request=None,
            response="Valid response",
        )


def test_evaluator_requires_string_response():
    evaluator = Evaluator()

    with pytest.raises(ValueError, match="Response must be a string"):
        evaluator.evaluate(
            request="Explain Python lists",
            response=None,
        )
