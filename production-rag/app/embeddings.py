from dataclasses import dataclass
from typing import Protocol

from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class Embedding:
    chunk_id: str
    vector: list[float]


class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> list[float]:
        ...


class DeterministicEmbeddingProvider:
    """Small deterministic provider used for tests."""

    def __init__(self, dimensions: int = 8):
        if dimensions <= 0:
            raise ValueError("dimensions must be greater than zero")

        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("text cannot be empty")

        values = [0.0] * self.dimensions

        for index, character in enumerate(text.encode("utf-8")):
            values[index % self.dimensions] += float(character)

        magnitude = sum(value * value for value in values) ** 0.5

        if magnitude == 0:
            return values

        return [value / magnitude for value in values]


class SentenceTransformerEmbeddingProvider:
    """Semantic embeddings using a local Sentence Transformers model."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.model = SentenceTransformer(model_name)

        self.dimensions = self.model.get_embedding_dimension()

        if self.dimensions is None:
            raise ValueError("Unable to determine embedding dimensions")

    def embed(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("text cannot be empty")

        vector = self.model.encode(
            text,
            normalize_embeddings=True,
        )

        return [float(value) for value in vector]
