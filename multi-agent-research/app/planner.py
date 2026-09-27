from app.contracts import ResearchTask


class ResearchPlanner:
    def plan(self, request: str, questions: list[str]) -> list[ResearchTask]:
        if not isinstance(request, str):
            raise ValueError("Request must be a string")

        if not request.strip():
            raise ValueError("Request cannot be empty")

        if not questions:
            raise ValueError("At least one research question is required")

        tasks = []

        for index, question in enumerate(questions, start=1):
            if not isinstance(question, str):
                raise ValueError("Research question must be a string")

            if not question.strip():
                raise ValueError("Research question cannot be empty")

            tasks.append(
                ResearchTask(
                    task_id=f"task-{index}",
                    question=question,
                )
            )

        return tasks
