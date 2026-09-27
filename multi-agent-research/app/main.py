import os

from app.agent import ResearchAgent
from app.api import app
from app.orchestrator import ResearchOrchestrator
from app.planner import ResearchPlanner
from app.provider import OllamaResearchProvider, OllamaSynthesisProvider
from app.synthesizer import ResearchSynthesizer


def build_orchestrator() -> ResearchOrchestrator:
    base_url = os.getenv(
        "OLLAMA_BASE_URL",
        "http://127.0.0.1:11434",
    )
    model = os.getenv(
        "OLLAMA_MODEL",
        "granite4.1:3b",
    )
    timeout = float(
        os.getenv(
            "OLLAMA_TIMEOUT",
            "300.0",
        )
    )

    research_provider = OllamaResearchProvider(
        base_url=base_url,
        model=model,
        timeout=timeout,
    )

    synthesis_provider = OllamaSynthesisProvider(
        base_url=base_url,
        model=model,
        timeout=timeout,
    )

    return ResearchOrchestrator(
        planner=ResearchPlanner(),
        agent=ResearchAgent(provider=research_provider),
        synthesizer=ResearchSynthesizer(provider=synthesis_provider),
    )


app.state.orchestrator = build_orchestrator()
