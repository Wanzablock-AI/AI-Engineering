import os
from typing import Any

import httpx

from app.generation import GroundedContext


class OllamaAnswerProvider:
    """Answer provider backed by a remote Ollama HTTP API."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
    ):
        self.base_url = (
            base_url
            or os.getenv("OLLAMA_BASE_URL", "http://100.118.104.245:11434")
        ).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "granite4.1:3b")
        self.timeout = timeout

    def generate(
        self,
        query: str,
        context: list[GroundedContext],
    ) -> str:
        if not query.strip():
            raise ValueError("query cannot be empty")

        if not context:
            raise ValueError("context cannot be empty")

        prompt = self._build_prompt(query, context)

        try:
            response = httpx.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Ollama request failed: {exc}"
            ) from exc

        if response.status_code >= 400:
            raise RuntimeError(
                f"Ollama request failed with HTTP {response.status_code}"
            )

        try:
            payload: dict[str, Any] = response.json()
        except ValueError as exc:
            raise RuntimeError(
                "Ollama returned invalid JSON"
            ) from exc

        answer = payload.get("response")

        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError(
                "Ollama response did not contain generated text"
            )

        return answer.strip()

    @staticmethod
    def _build_prompt(
        query: str,
        context: list[GroundedContext],
    ) -> str:
        context_blocks = []

        for index, item in enumerate(context, start=1):
            context_blocks.append(
                (
                    f"[Source {index}]\n"
                    f"Document: {item.document}\n"
                    f"Page: {item.page_number}\n"
                    f"Content:\n{item.text}"
                )
            )

        sources = "\n\n".join(context_blocks)

        return (
            "You are a document-grounded question answering system.\n\n"
            "Answer the user's question using ONLY the supplied document "
            "context.\n"
            "Do not use outside knowledge.\n"
            "Do not invent facts that are not supported by the context.\n"
            "If the context does not contain enough information to answer "
            "the question, say so clearly.\n"
            "Keep the answer concise and factual.\n\n"
            f"DOCUMENT CONTEXT:\n{sources}\n\n"
            f"USER QUESTION:\n{query}\n\n"
            "ANSWER:"
        )
