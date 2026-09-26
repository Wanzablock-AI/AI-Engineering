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


def test_lexical_retrieval_handles_natural_language_punctuation(tmp_path):
    db_path = tmp_path / "test.db"

    store = ChunkStore(db_path)

    store.add_chunks(
        [
            Chunk(
                chunk_id="security.pdf:1:0",
                document="security.pdf",
                page_number=1,
                text="Input validation checks incoming data before processing.",
            ),
        ]
    )

    retriever = LexicalRetriever(db_path)

    results = retriever.search("What does input validation do?")

    assert len(results) == 1
    assert results[0].chunk_id == "security.pdf:1:0"




def test_lexical_retriever_deletes_document(tmp_path):
    db_path = tmp_path / "test.db"

    store = ChunkStore(db_path)

    store.add_chunks(
        [
            Chunk(
                chunk_id="security.pdf:1:0",
                document="security.pdf",
                page_number=1,
                text="Zebracode security content.",
            ),
            Chunk(
                chunk_id="other.pdf:1:0",
                document="other.pdf",
                page_number=1,
                text="Other document content.",
            ),
        ]
    )

    retriever = LexicalRetriever(db_path)

    assert len(retriever.search("Zebracode")) == 1

    retriever.delete_document("security.pdf")

    assert retriever.search("Zebracode") == []

    results = retriever.search("other")

    assert len(results) == 1
    assert results[0].document == "other.pdf"

def test_lexical_retriever_replaces_existing_chunk_without_duplicates(tmp_path):
    db_path = tmp_path / "test.db"

    store = ChunkStore(db_path)

    original = Chunk(
        chunk_id="security.pdf:1:0",
        document="security.pdf",
        page_number=1,
        text="Original security content.",
    )

    replacement = Chunk(
        chunk_id="security.pdf:1:0",
        document="security.pdf",
        page_number=1,
        text="Replacement security content.",
    )

    store.add_chunks([original])

    retriever = LexicalRetriever(db_path)

    retriever.add_chunks([replacement])

    assert retriever.search("Original") == []

    results = retriever.search("Replacement")

    assert len(results) == 1
    assert results[0].chunk_id == "security.pdf:1:0"


def test_lexical_retriever_rebuilds_stale_index_from_chunk_store(tmp_path):
    db_path = tmp_path / "test.db"

    store = ChunkStore(db_path)

    store.add_chunks(
        [
            Chunk(
                chunk_id="security.pdf:1:0",
                document="security.pdf",
                page_number=1,
                text="Current security content.",
            )
        ]
    )

    retriever = LexicalRetriever(db_path)

    with retriever._connect() as connection:
        connection.execute(
            """
            INSERT INTO chunks_fts (
                chunk_id,
                document,
                page_number,
                text
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                "stale.pdf:1:0",
                "stale.pdf",
                1,
                "Stale content that no longer exists.",
            ),
        )
        connection.commit()

    retriever = LexicalRetriever(db_path)

    assert retriever.search("Stale") == []

    results = retriever.search("Current")

    assert len(results) == 1
    assert results[0].chunk_id == "security.pdf:1:0"
