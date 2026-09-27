import httpx

from app.contracts import ResearchResult


class OllamaResearchProvider:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "granite4.1:3b",
        timeout: float = 300.0,
        client=None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.client = client

    def research(self, question: str) -> str:
        if not isinstance(question, str):
            raise ValueError("Question must be a string")

        if not question.strip():
            raise ValueError("Question cannot be empty")

        payload = {
            "model": self.model,
            "prompt": question,
            "stream": False,
        }

        if self.client is not None:
            response = self.client.post(
                f"{self.base_url}/api/generate",
                json=payload,
            )
        else:
            response = httpx.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )

        response.raise_for_status()

        payload = response.json()

        answer = payload.get("response")

        if not isinstance(answer, str):
            raise RuntimeError("Invalid provider response")

        return answer


class OllamaSynthesisProvider:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "granite4.1:3b",
        timeout: float = 300.0,
        client=None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.client = client

    def synthesize(
        self,
        request: str,
        results: list[ResearchResult],
    ) -> str:
        if not isinstance(request, str):
            raise ValueError("Request must be a string")

        if not request.strip():
            raise ValueError("Request cannot be empty")

        if not results:
            raise ValueError("At least one research result is required")

        if not all(isinstance(result, ResearchResult) for result in results):
            raise ValueError(
                "Results must contain ResearchResult objects"
            )

        research_text = "\n\n".join(
            f"Task {result.task_id}:\n"
            f"Question: {result.question}\n"
            f"Answer: {result.answer}"
            for result in results
        )

        prompt = (
            "Synthesize the following research results into a concise, "
            "accurate answer to the original request.\n\n"
            f"Original request:\n{request}\n\n"
            f"Research results:\n{research_text}"
        )

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }

        if self.client is not None:
            response = self.client.post(
                f"{self.base_url}/api/generate",
                json=payload,
            )
        else:
            response = httpx.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )

        response.raise_for_status()

        payload = response.json()

        answer = payload.get("response")

        if not isinstance(answer, str):
            raise RuntimeError("Invalid provider response")

        return answer
