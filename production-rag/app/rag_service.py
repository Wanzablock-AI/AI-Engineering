from pathlib import Path

from app.embeddings import SentenceTransformerEmbeddingProvider
from app.generation import GroundedAnswerGenerator
from app.graph import RAGComponents, RAGGraph
from app.hybrid_retrieval import HybridRetriever
from app.ollama_provider import OllamaAnswerProvider
from app.reranking import CrossEncoderReranker, HybridReranker
from app.retrieval import LexicalRetriever
from app.storage import ChunkStore, DEFAULT_DB_PATH
from app.vector_store import VectorStore


def build_rag_graph(
    db_path: str | Path = DEFAULT_DB_PATH,
) -> RAGGraph:
    db_path = Path(db_path)

    chunk_store = ChunkStore(db_path)
    lexical_retriever = LexicalRetriever(db_path)

    embedding_provider = SentenceTransformerEmbeddingProvider()

    vector_store = VectorStore(
        db_path,
        dimensions=embedding_provider.dimensions,
    )

    retriever = HybridRetriever(
        lexical_retriever=lexical_retriever,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        chunk_store=chunk_store,
    )

    reranker = HybridReranker(
        CrossEncoderReranker()
    )

    provider = OllamaAnswerProvider()
    generator = GroundedAnswerGenerator(provider)

    return RAGGraph(
        RAGComponents(
            retriever=retriever,
            reranker=reranker,
            generator=generator,
        )
    )
