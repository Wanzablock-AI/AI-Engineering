from app.chunking import Chunk
from app.retrieval import LexicalRetriever
from app.storage import ChunkStore


def test_lexical_retrieval_finds_relevant_chunks(tmp_path):
    db_path = tmp_path / "test.db"

    store = ChunkStore(db_path)

    store.add_chunks(
        [
            Chunk(
                chunk_id="security.pdf:1:0",
                document="security.pdf",
                page_number=1,
                text="Prompt injection attacks manipulate AI systems.",
            ),
            Chunk(
                chunk_id="python.pdf:2:0",
                document="python.pdf",
                page_number=2,
                text="Python is a general purpose programming language.",
            ),
        ]
    )

    retriever = LexicalRetriever(db_path)

    results = retriever.search("prompt injection")

    assert len(results) == 1
    assert results[0].chunk_id == "security.pdf:1:0"
    assert results[0].document == "security.pdf"
    assert results[0].page_number == 1
    assert "Prompt injection" in results[0].text


def test_lexical_retrieval_handles_empty_query(tmp_path):
    db_path = tmp_path / "test.db"

    store = ChunkStore(db_path)
    store.add_chunks(
        [
            Chunk(
                chunk_id="test.pdf:1:0",
                document="test.pdf",
                page_number=1,
                text="Some document text.",
            )
        ]
    )

    retriever = LexicalRetriever(db_path)

    assert retriever.search("") == []


def test_lexical_retrieval_rejects_invalid_limit(tmp_path):
    db_path = tmp_path / "test.db"

    store = ChunkStore(db_path)
    retriever = LexicalRetriever(db_path)

    try:
        retriever.search("test", limit=0)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")
