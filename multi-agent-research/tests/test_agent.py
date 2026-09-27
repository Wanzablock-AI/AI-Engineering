import pytest

from app.agent import ResearchAgent
from app.contracts import ResearchResult, ResearchTask


class FakeResearchProvider:
    def research(self, question: str) -> str:
        return f"Research answer for: {question}"


class FailingResearchProvider:
    def research(self, question: str) -> str:
        raise RuntimeError("provider unavailable")


def test_agent_returns_research_result():
    agent = ResearchAgent(provider=FakeResearchProvider())

    task = ResearchTask(
        task_id="task-1",
        question="What is RAG?",
    )

    result = agent.run(task)

    assert isinstance(result, ResearchResult)
    assert result.task_id == "task-1"
    assert result.question == "What is RAG?"
    assert result.answer == "Research answer for: What is RAG?"
    assert result.success is True


def test_agent_rejects_invalid_task():
    agent = ResearchAgent(provider=FakeResearchProvider())

    with pytest.raises(ValueError, match="Task must be a ResearchTask"):
        agent.run("not a task")


def test_agent_handles_provider_failure():
    agent = ResearchAgent(provider=FailingResearchProvider())

    task = ResearchTask(
        task_id="task-2",
        question="What is multi-agent research?",
    )

    result = agent.run(task)

    assert result.task_id == "task-2"
    assert result.question == "What is multi-agent research?"
    assert result.answer == ""
    assert result.success is False
