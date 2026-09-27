from app.config import Settings


def test_default_ollama_url(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)

    settings = Settings()

    assert settings.ollama_base_url == "http://127.0.0.1:11434"


def test_ollama_url_from_environment(monkeypatch):
    monkeypatch.setenv(
        "OLLAMA_BASE_URL",
        "http://100.118.104.245:11434/",
    )

    settings = Settings()

    assert settings.ollama_base_url == "http://100.118.104.245:11434"
