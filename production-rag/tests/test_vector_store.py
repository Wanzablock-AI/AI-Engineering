from app.vector_store import VectorStore


def test_vector_store_persists_embedding(tmp_path):
    db_path = tmp_path / "test.db"

    store = VectorStore(db_path, dimensions=4)

    vector = [0.1, 0.2, 0.3, 0.4]

    store.add_embedding("security.pdf:1:0", vector)

    assert store.get_embedding("security.pdf:1:0") == vector
    assert store.count() == 1


def test_vector_store_returns_none_for_missing_embedding(tmp_path):
    db_path = tmp_path / "test.db"

    store = VectorStore(db_path, dimensions=4)

    assert store.get_embedding("missing") is None


def test_vector_store_rejects_wrong_dimensions(tmp_path):
    db_path = tmp_path / "test.db"

    store = VectorStore(db_path, dimensions=4)

    try:
        store.add_embedding(
            "security.pdf:1:0",
            [0.1, 0.2],
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")


def test_vector_store_rejects_empty_chunk_id(tmp_path):
    db_path = tmp_path / "test.db"

    store = VectorStore(db_path, dimensions=4)

    try:
        store.add_embedding("", [0.1, 0.2, 0.3, 0.4])
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")


def test_vector_store_replaces_existing_embedding(tmp_path):
    db_path = tmp_path / "test.db"

    store = VectorStore(db_path, dimensions=4)

    store.add_embedding(
        "security.pdf:1:0",
        [0.1, 0.2, 0.3, 0.4],
    )

    store.add_embedding(
        "security.pdf:1:0",
        [0.5, 0.6, 0.7, 0.8],
    )

    assert store.get_embedding("security.pdf:1:0") == [
        0.5,
        0.6,
        0.7,
        0.8,
    ]

    assert store.count() == 1
