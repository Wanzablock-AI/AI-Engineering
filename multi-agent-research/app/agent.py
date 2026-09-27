from app.contracts import ResearchResult, ResearchTask


class ResearchAgent:
    def __init__(self, provider):
        self.provider = provider

    def run(self, task: ResearchTask) -> ResearchResult:
        if not isinstance(task, ResearchTask):
            raise ValueError("Task must be a ResearchTask")

        try:
            answer = self.provider.research(task.question)
        except Exception:
            return ResearchResult(
                task_id=task.task_id,
                question=task.question,
                answer="",
                success=False,
            )

        return ResearchResult(
            task_id=task.task_id,
            question=task.question,
            answer=answer,
            success=True,
        )
