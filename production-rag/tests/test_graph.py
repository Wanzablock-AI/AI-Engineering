from app.generation import GroundedAnswerGenerator
from app.graph import RAGComponents, RAGGraph
from app.hybrid_retrieval import HybridSearchResult
from app.reranking import HybridReranker


class FakeRetriever:
    def __init__(self):
        self.last_query = None

    def search(self, query: str, limit: int = 5):
        self.last_query = query

        return [
            HybridSearchResult(
                chunk_id="doc:1:0",
                document="security.pdf",
                page_number=2,
                text="Input validation is important.",
                score=0.9,
            ),
            HybridSearchResult(
                chunk_id="doc:1:1",
                document="security.pdf",
                page_number=4,
                text="Security testing should cover input paths.",
                score=0.8,
            ),
        ][:limit]


class FakeReranker:
    def __init__(self):
        self.last_query = None
        self.last_candidates = None

    def score(self, query: str, text: str) -> float:
        return 1.0 if "validation" in text else 0.5


class FakeAnswerProvider:
    def __init__(self):
        self.last_query = None
        self.last_context = None

    def generate(self, query, context):
        self.last_query = query
        self.last_context = context

        return "The document emphasizes input validation and security testing."


def build_graph():
    retriever = FakeRetriever()
    reranker = HybridReranker(FakeReranker())
    provider = FakeAnswerProvider()
    generator = GroundedAnswerGenerator(provider)

    graph = RAGGraph(
        RAGComponents(
            retriever=retriever,
            reranker=reranker,
            generator=generator,
        )
    )

    return graph, retriever, provider


def test_graph_runs_retrieve_rerank_generate():
    graph, retriever, provider = build_graph()

    result = graph.invoke(
        "How should input be handled?"
    )

    assert result.answer == (
        "The document emphasizes input validation and security testing."
    )
    assert retriever.last_query == "How should input be handled?"
    assert provider.last_query == "How should input be handled?"
    assert len(provider.last_context) == 2


def test_graph_preserves_citations():
    graph, _, _ = build_graph()

    result = graph.invoke(
        "How should input be handled?"
    )

    assert result.citations[0].document == "security.pdf"
    assert result.citations[0].page_number == 2
    assert result.citations[1].page_number == 4


def test_graph_rejects_empty_query():
    graph, _, _ = build_graph()

    try:
        graph.invoke("   ")
    except ValueError as exc:
        assert str(exc) == "query cannot be empty"
    else:
        raise AssertionError("Expected ValueError")
