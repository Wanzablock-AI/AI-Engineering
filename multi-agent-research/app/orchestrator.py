from app.contracts import ResearchReport
from app.planner import ResearchPlanner


class ResearchOrchestrator:
    def __init__(self, planner, agent, synthesizer):
        self.planner = planner
        self.agent = agent
        self.synthesizer = synthesizer

    def run(
        self,
        request: str,
        questions: list[str],
    ) -> ResearchReport:
        tasks = self.planner.plan(
            request=request,
            questions=questions,
        )

        results = [
            self.agent.run(task)
            for task in tasks
        ]

        return self.synthesizer.synthesize(
            request=request,
            results=results,
        )
