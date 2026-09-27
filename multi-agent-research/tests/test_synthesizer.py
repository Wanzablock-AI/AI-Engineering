import pytest

from app.contracts import ResearchReport, ResearchResult
from app.synthesizer import ResearchSynthesizer


class FakeSynthesisProvider:
    def synthesize(self, request: str, results: list[ResearchResult]) -> str:
        answers = " | ".join(result.answer for result in results)
        return f"Synthesis for '{request}': {answers}"


def make_result(task_id: str, answer: str) -> ResearchResult:
    return ResearchResult(
        task_id=task_id,
        question=f"Question for {task_id}",
        answer=answer,
        success=True,
    )


def test_synthesizer_returns_report():
    synthesizer = ResearchSynthesizer(
        provider=FakeSynthesisProvider()
    )

    results = [
        make_result("task-1", "RAG retrieves relevant context."),
        make_result("task-2", "Fine-tuning updates model behavior."),
    ]

    report = synthesizer.synthesize(
        request="Compare RAG and fine-tuning.",
        results=results,
    )

    assert isinstance(report, ResearchReport)
    assert report.request == "Compare RAG and fine-tuning."
    assert report.results == tuple(results)
    assert "RAG retrieves relevant context." in report.synthesis
    assert "Fine-tuning updates model behavior." in report.synthesis


def test_synthesizer_rejects_empty_request():
    synthesizer = ResearchSynthesizer(
        provider=FakeSynthesisProvider()
    )

    with pytest.raises(ValueError, match="Request cannot be empty"):
        synthesizer.synthesize(
            request="",
            results=[make_result("task-1", "Answer")],
        )


def test_synthesizer_rejects_empty_results():
    synthesizer = ResearchSynthesizer(
        provider=FakeSynthesisProvider()
    )

    with pytest.raises(ValueError, match="At least one research result"):
        synthesizer.synthesize(
            request="Explain RAG.",
            results=[],
        )


def test_synthesizer_rejects_invalid_results():
    synthesizer = ResearchSynthesizer(
        provider=FakeSynthesisProvider()
    )

    with pytest.raises(ValueError, match="Results must contain ResearchResult"):
        synthesizer.synthesize(
            request="Explain RAG.",
            results=["not a result"],
        )


def test_synthesizer_preserves_result_order():
    synthesizer = ResearchSynthesizer(
        provider=FakeSynthesisProvider()
    )

    results = [
        make_result("task-2", "Second"),
        make_result("task-1", "First"),
    ]

    report = synthesizer.synthesize(
        request="Test ordering.",
        results=results,
    )

    assert [result.task_id for result in report.results] == [
        "task-2",
        "task-1",
    ]
