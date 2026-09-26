from app.storage import SQLiteEventStore


def test_store_saves_event(tmp_path):
    store = SQLiteEventStore(tmp_path / "events.db")

    store.save(
        request="Explain Python lists",
        model="qwen2.5-coder:3b",
        provider="ollama",
        latency_ms=125.5,
        success=True,
    )

    events = store.list_events()

    assert len(events) == 1
    assert events[0]["request"] == "Explain Python lists"
    assert events[0]["model"] == "qwen2.5-coder:3b"
    assert events[0]["provider"] == "ollama"
    assert events[0]["latency_ms"] == 125.5
    assert events[0]["success"] is True


def test_store_persists_across_instances(tmp_path):
    db_path = tmp_path / "events.db"

    first = SQLiteEventStore(db_path)

    first.save(
        request="First request",
        model="llama3.2:1b",
        provider="ollama",
        latency_ms=50.0,
        success=True,
    )

    second = SQLiteEventStore(db_path)

    events = second.list_events()

    assert len(events) == 1
    assert events[0]["request"] == "First request"


def test_store_returns_empty_list_when_empty(tmp_path):
    store = SQLiteEventStore(tmp_path / "events.db")

    assert store.list_events() == []
