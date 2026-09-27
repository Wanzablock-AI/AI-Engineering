import pytest

from app.planner import ResearchPlanner


def test_planner_creates_tasks_from_questions():
    planner = ResearchPlanner()

    tasks = planner.plan(
        "Compare RAG and fine-tuning.",
        questions=[
            "What is RAG?",
            "What is fine-tuning?",
            "What are the differences?",
        ],
    )

    assert len(tasks) == 3
    assert tasks[0].task_id == "task-1"
    assert tasks[1].task_id == "task-2"
    assert tasks[2].task_id == "task-3"
    assert tasks[0].question == "What is RAG?"
    assert tasks[2].question == "What are the differences?"


def test_planner_rejects_empty_request():
    planner = ResearchPlanner()

    with pytest.raises(ValueError, match="Request cannot be empty"):
        planner.plan("", questions=["What is RAG?"])


def test_planner_rejects_empty_question_list():
    planner = ResearchPlanner()

    with pytest.raises(ValueError, match="At least one research question"):
        planner.plan("Explain RAG.", questions=[])


def test_planner_rejects_empty_questions():
    planner = ResearchPlanner()

    with pytest.raises(ValueError, match="Research question cannot be empty"):
        planner.plan(
            "Explain RAG.",
            questions=["What is RAG?", "  "],
        )


def test_planner_does_not_modify_question_order():
    planner = ResearchPlanner()

    questions = [
        "First question",
        "Second question",
        "Third question",
    ]

    tasks = planner.plan("Research this.", questions=questions)

    assert [task.question for task in tasks] == questions
