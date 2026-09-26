from app.chunking import Chunk
from app.storage import ChunkStore


def test_chunk_store_persists_chunks(tmp_path):
    db_path = tmp_path / "test.db"

    store = ChunkStore(db_path)

    chunks = [
        Chunk(
            chunk_id="test.pdf:1:0",
            document="test.pdf",
            page_number=1,
            text="First chunk.",
        ),
        Chunk(
            chunk_id="test.pdf:2:0",
            document="test.pdf",
            page_number=2,
            text="Second chunk.",
        ),
    ]

    store.add_chunks(chunks)

    assert store.count() == 2

    chunk = store.get_chunk("test.pdf:2:0")

    assert chunk is not None
    assert chunk.document == "test.pdf"
    assert chunk.page_number == 2
    assert chunk.text == "Second chunk."


def test_chunk_store_returns_none_for_missing_chunk(tmp_path):
    store = ChunkStore(tmp_path / "test.db")

    assert store.get_chunk("missing") is None
