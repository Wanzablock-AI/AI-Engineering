import pytest

from app.contracts import ResearchReport, ResearchResult, ResearchTask


def test_research_task_is_immutable():
    task = ResearchTask(
        task_id="task-1",
        question="What is retrieval augmented generation?",
    )

    assert task.task_id == "task-1"
    assert task.question == "What is retrieval augmented generation?"

    with pytest.raises(AttributeError):
        task.question = "changed"


def test_research_result_contains_task_outcome():
    result = ResearchResult(
        task_id="task-1",
        question="What is RAG?",
        answer="RAG combines retrieval with generation.",
        success=True,
    )

    assert result.task_id == "task-1"
    assert result.success is True
    assert result.answer


def test_research_report_contains_results():
    result = ResearchResult(
        task_id="task-1",
        question="What is RAG?",
        answer="RAG combines retrieval with generation.",
        success=True,
    )

    report = ResearchReport(
        request="Explain RAG.",
        results=(result,),
        synthesis="RAG combines retrieval with generation.",
    )

    assert report.request == "Explain RAG."
    assert report.results == (result,)
    assert report.synthesis
