from dataclasses import dataclass


@dataclass(frozen=True)
class ResearchTask:
    task_id: str
    question: str


@dataclass(frozen=True)
class ResearchResult:
    task_id: str
    question: str
    answer: str
    success: bool


@dataclass(frozen=True)
class ResearchReport:
    request: str
    results: tuple[ResearchResult, ...]
    synthesis: str
