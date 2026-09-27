import os

import pytest

from app.config import Settings
from app.ollama import OllamaProvider


@pytest.mark.live
def test_live_ollama_generation():
    if not os.getenv("OLLAMA_BASE_URL"):
        pytest.skip("OLLAMA_BASE_URL is not configured")

    settings = Settings()
    provider = OllamaProvider(
        base_url=settings.ollama_base_url,
    )

    result = provider.generate(
        model="qwen2.5-coder:3b",
        prompt="Return only the word: hello",
    )

    assert "hello" in result.lower()
