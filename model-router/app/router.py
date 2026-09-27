from dataclasses import dataclass


@dataclass(frozen=True)
class RouteDecision:
    model: str
    reason: str


class ModelRouter:
    ROUTES = {
        "coding": "qwen2.5-coder:3b",
        "rag": "granite4.1:3b",
        "lightweight": "llama3.2:1b",
        "general": "llama3.2:3b",
    }

    def route(self, task: str) -> RouteDecision:
        task = task.strip().lower()

        if not task:
            raise ValueError("Task type cannot be empty")

        if task not in self.ROUTES:
            raise ValueError(f"Unsupported task type: {task}")

        return RouteDecision(
            model=self.ROUTES[task],
            reason=task,
        )

    def route_request(
        self,
        request: str,
        model: str | None = None,
    ) -> RouteDecision:
        if not isinstance(request, str):
            raise ValueError("Request must be a string")

        text = request.strip().lower()

        if not text:
            raise ValueError("Request cannot be empty")

        if model is not None:
            model = model.strip()

            if not model:
                raise ValueError("Model cannot be empty")

            if model not in self.ROUTES.values():
                raise ValueError(f"Unsupported model: {model}")

            return RouteDecision(
                model=model,
                reason="explicit",
            )

        coding_keywords = (
            "python",
            "code",
            "coding",
            "program",
            "function",
            "script",
            "debug",
            "bug",
            "sql",
            "javascript",
            "typescript",
        )

        rag_keywords = (
            "according to the documents",
            "according to the document",
            "document",
            "documents",
            "provided files",
            "provided file",
            "according to the files",
            "database backup policy",
            "what does the report say",
        )

        lightweight_keywords = (
            "what time is it",
            "current time",
            "hello",
            "hi",
            "hey",
        )

        if any(keyword in text for keyword in coding_keywords):
            return self.route("coding")

        if any(keyword in text for keyword in rag_keywords):
            return self.route("rag")

        if any(keyword in text for keyword in lightweight_keywords):
            return self.route("lightweight")

        return self.route("general")
