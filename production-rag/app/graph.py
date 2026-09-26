from dataclasses import dataclass
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.generation import GroundedAnswer, GroundedAnswerGenerator
from app.hybrid_retrieval import HybridRetriever
from app.reranking import HybridReranker, RerankedResult


@dataclass(frozen=True)
class RAGComponents:
    retriever: HybridRetriever
    reranker: HybridReranker
    generator: GroundedAnswerGenerator


class RAGState(TypedDict, total=False):
    query: str
    retrieved: list
    reranked: list[RerankedResult]
    result: GroundedAnswer


class RAGGraph:
    """LangGraph orchestration for retrieval, reranking, and generation."""

    def __init__(self, components: RAGComponents):
        self.components = components
        self.graph = self._build()

    def _build(self):
        workflow = StateGraph(RAGState)

        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("rerank", self._rerank)
        workflow.add_node("generate", self._generate)

        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "rerank")
        workflow.add_edge("rerank", "generate")
        workflow.add_edge("generate", END)

        return workflow.compile()

    def _retrieve(self, state: RAGState) -> dict:
        query = state["query"]

        return {
            "retrieved": self.components.retriever.search(
                query,
                limit=5,
            )
        }

    def _rerank(self, state: RAGState) -> dict:
        query = state["query"]
        candidates = state.get("retrieved", [])

        return {
            "reranked": self.components.reranker.rerank(
                query,
                candidates,
                limit=5,
            )
        }

    def _generate(self, state: RAGState) -> dict:
        query = state["query"]
        candidates = state.get("reranked", [])

        return {
            "result": self.components.generator.generate(
                query,
                candidates,
            )
        }

    def invoke(self, query: str) -> GroundedAnswer:
        if not query.strip():
            raise ValueError("query cannot be empty")

        state = self.graph.invoke({"query": query})

        return state["result"]
