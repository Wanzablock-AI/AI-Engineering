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


def test_vector_store_search_returns_nearest_vectors(tmp_path):
    db_path = tmp_path / "test.db"

    store = VectorStore(db_path, dimensions=3)

    store.add_embedding("similar", [1.0, 0.0, 0.0])
    store.add_embedding("less-similar", [0.0, 1.0, 0.0])
    store.add_embedding("opposite", [-1.0, 0.0, 0.0])

    results = store.search([0.9, 0.1, 0.0], limit=3)

    assert [result.chunk_id for result in results] == [
        "similar",
        "less-similar",
        "opposite",
    ]

    assert results[0].score > results[1].score
    assert results[1].score > results[2].score


def test_vector_store_search_respects_limit(tmp_path):
    db_path = tmp_path / "test.db"

    store = VectorStore(db_path, dimensions=2)

    store.add_embedding("one", [1.0, 0.0])
    store.add_embedding("two", [0.9, 0.1])
    store.add_embedding("three", [0.8, 0.2])

    results = store.search([1.0, 0.0], limit=2)

    assert len(results) == 2


def test_vector_store_search_empty_store(tmp_path):
    db_path = tmp_path / "test.db"

    store = VectorStore(db_path, dimensions=2)

    results = store.search([1.0, 0.0])

    assert results == []


def test_vector_store_search_rejects_wrong_dimensions(tmp_path):
    db_path = tmp_path / "test.db"

    store = VectorStore(db_path, dimensions=2)

    try:
        store.search([1.0, 0.0, 0.0])
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")


def test_vector_store_search_rejects_zero_vector(tmp_path):
    db_path = tmp_path / "test.db"

    store = VectorStore(db_path, dimensions=2)

    try:
        store.search([0.0, 0.0])
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")


def test_vector_store_search_rejects_invalid_limit(tmp_path):
    db_path = tmp_path / "test.db"

    store = VectorStore(db_path, dimensions=2)

    try:
        store.search([1.0, 0.0], limit=0)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")
