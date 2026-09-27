from app.contracts import ResearchReport, ResearchResult, ResearchTask
from app.orchestrator import ResearchOrchestrator


class FakePlanner:
    def plan(self, request: str, questions: list[str]) -> list[ResearchTask]:
        return [
            ResearchTask(task_id="task-1", question=questions[0]),
            ResearchTask(task_id="task-2", question=questions[1]),
        ]


class FakeAgent:
    def __init__(self):
        self.tasks = []

    def run(self, task: ResearchTask) -> ResearchResult:
        self.tasks.append(task)

        return ResearchResult(
            task_id=task.task_id,
            question=task.question,
            answer=f"Answer: {task.question}",
            success=True,
        )


class FakeSynthesizer:
    def __init__(self):
        self.requests = []
        self.results = []

    def synthesize(
        self,
        request: str,
        results: list[ResearchResult],
    ) -> ResearchReport:
        self.requests.append(request)
        self.results.append(results)

        return ResearchReport(
            request=request,
            results=tuple(results),
            synthesis="Final synthesized answer.",
        )


def test_orchestrator_runs_research_pipeline():
    planner = FakePlanner()
    agent = FakeAgent()
    synthesizer = FakeSynthesizer()

    orchestrator = ResearchOrchestrator(
        planner=planner,
        agent=agent,
        synthesizer=synthesizer,
    )

    report = orchestrator.run(
        request="Compare RAG and fine-tuning.",
        questions=[
            "What is RAG?",
            "What is fine-tuning?",
        ],
    )

    assert isinstance(report, ResearchReport)
    assert report.request == "Compare RAG and fine-tuning."
    assert len(report.results) == 2
    assert report.synthesis == "Final synthesized answer."


def test_orchestrator_passes_tasks_to_agent():
    planner = FakePlanner()
    agent = FakeAgent()
    synthesizer = FakeSynthesizer()

    orchestrator = ResearchOrchestrator(
        planner=planner,
        agent=agent,
        synthesizer=synthesizer,
    )

    orchestrator.run(
        request="Research AI.",
        questions=[
            "Question one",
            "Question two",
        ],
    )

    assert [task.task_id for task in agent.tasks] == [
        "task-1",
        "task-2",
    ]


def test_orchestrator_passes_results_to_synthesizer():
    planner = FakePlanner()
    agent = FakeAgent()
    synthesizer = FakeSynthesizer()

    orchestrator = ResearchOrchestrator(
        planner=planner,
        agent=agent,
        synthesizer=synthesizer,
    )

    orchestrator.run(
        request="Research AI.",
        questions=[
            "Question one",
            "Question two",
        ],
    )

    assert synthesizer.requests == ["Research AI."]
    assert len(synthesizer.results) == 1
    assert len(synthesizer.results[0]) == 2
