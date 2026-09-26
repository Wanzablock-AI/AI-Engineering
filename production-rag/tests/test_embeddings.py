from app.embeddings import DeterministicEmbeddingProvider


def test_embedding_provider_returns_expected_dimensions():
    provider = DeterministicEmbeddingProvider(dimensions=8)

    vector = provider.embed("prompt injection")

    assert len(vector) == 8


def test_embedding_provider_is_deterministic():
    provider = DeterministicEmbeddingProvider(dimensions=8)

    first = provider.embed("prompt injection")
    second = provider.embed("prompt injection")

    assert first == second


def test_embedding_provider_rejects_empty_text():
    provider = DeterministicEmbeddingProvider()

    try:
        provider.embed("")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")


def test_embedding_provider_rejects_invalid_dimensions():
    try:
        DeterministicEmbeddingProvider(dimensions=0)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")


def test_sentence_transformer_provider_returns_384_dimensions():
    from app.embeddings import SentenceTransformerEmbeddingProvider

    provider = SentenceTransformerEmbeddingProvider()

    vector = provider.embed("prompt injection")

    assert len(vector) == 384
    assert all(isinstance(value, float) for value in vector)


def test_sentence_transformer_provider_rejects_empty_text():
    from app.embeddings import SentenceTransformerEmbeddingProvider

    provider = SentenceTransformerEmbeddingProvider()

    try:
        provider.embed("")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")
