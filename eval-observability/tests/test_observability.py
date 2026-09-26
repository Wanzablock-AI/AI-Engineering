from app.observability import ObservabilityEvent, ObservabilityRecorder


def test_recorder_stores_event():
    recorder = ObservabilityRecorder()

    event = recorder.record(
        request="Explain Python lists",
        model="qwen2.5-coder:3b",
        provider="ollama",
        latency_ms=125.5,
        success=True,
    )

    assert isinstance(event, ObservabilityEvent)
    assert event.request == "Explain Python lists"
    assert event.model == "qwen2.5-coder:3b"
    assert event.provider == "ollama"
    assert event.latency_ms == 125.5
    assert event.success is True


def test_recorder_keeps_multiple_events():
    recorder = ObservabilityRecorder()

    recorder.record(
        request="First",
        model="llama3.2:1b",
        provider="ollama",
        latency_ms=100.0,
        success=True,
    )

    recorder.record(
        request="Second",
        model="granite4.1:3b",
        provider="ollama",
        latency_ms=200.0,
        success=False,
    )

    assert len(recorder.events) == 2


def test_event_is_immutable():
    recorder = ObservabilityRecorder()

    event = recorder.record(
        request="Test",
        model="llama3.2:1b",
        provider="ollama",
        latency_ms=50.0,
        success=True,
    )

    try:
        event.success = False
    except AttributeError:
        pass
    else:
        raise AssertionError("ObservabilityEvent should be immutable")
