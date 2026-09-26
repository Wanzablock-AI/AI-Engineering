from dataclasses import dataclass
from typing import Callable, Sequence


@dataclass(frozen=True)
class EvaluationCase:
    query: str
    relevant_chunk_ids: frozenset[str]


@dataclass(frozen=True)
class RetrievalMetrics:
    recall_at_1: float
    recall_at_3: float
    mean_reciprocal_rank: float


def evaluate_retrieval(
    cases: Sequence[EvaluationCase],
    retrieve: Callable[[str, int], Sequence[str]],
) -> RetrievalMetrics:
    if not cases:
        raise ValueError("evaluation cases cannot be empty")

    recall_at_1_hits = 0
    recall_at_3_hits = 0
    reciprocal_ranks: list[float] = []

    for case in cases:
        if not case.query.strip():
            raise ValueError("evaluation query cannot be blank")

        if not case.relevant_chunk_ids:
            raise ValueError(
                "evaluation case must contain relevant chunk IDs"
            )

        results = list(retrieve(case.query, 3))

        if results and results[0] in case.relevant_chunk_ids:
            recall_at_1_hits += 1

        first_relevant_rank = None

        for rank, chunk_id in enumerate(results, start=1):
            if chunk_id in case.relevant_chunk_ids:
                first_relevant_rank = rank
                break

        if first_relevant_rank is not None:
            recall_at_3_hits += 1
            reciprocal_ranks.append(1.0 / first_relevant_rank)
        else:
            reciprocal_ranks.append(0.0)

    total = len(cases)

    return RetrievalMetrics(
        recall_at_1=recall_at_1_hits / total,
        recall_at_3=recall_at_3_hits / total,
        mean_reciprocal_rank=sum(reciprocal_ranks) / total,
    )
