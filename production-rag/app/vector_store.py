import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VectorSearchResult:
    chunk_id: str
    score: float


class VectorStore:
    def __init__(
        self,
        db_path: str | Path,
        dimensions: int,
    ):
        self.db_path = Path(db_path)

        if dimensions <= 0:
            raise ValueError("dimensions must be greater than zero")

        self.dimensions = dimensions
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    chunk_id TEXT PRIMARY KEY,
                    vector TEXT NOT NULL
                )
                """
            )

            connection.commit()

    def add_embedding(
        self,
        chunk_id: str,
        vector: list[float],
    ) -> None:
        if not chunk_id.strip():
            raise ValueError("chunk_id cannot be empty")

        if len(vector) != self.dimensions:
            raise ValueError(
                f"Expected vector with {self.dimensions} dimensions"
            )

        if not all(math.isfinite(value) for value in vector):
            raise ValueError(
                "vector values must be finite"
            )

        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO embeddings (
                    chunk_id,
                    vector
                )
                VALUES (?, ?)
                """,
                (
                    chunk_id,
                    json.dumps(vector),
                ),
            )

            connection.commit()

    def delete_document(self, document: str) -> None:
        if not document.strip():
            raise ValueError("document cannot be empty")

        escaped_document = (
            document
            .replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_")
        )
        prefix = f"{escaped_document}:%"

        with self._connect() as connection:
            connection.execute(
                """
                DELETE FROM embeddings
                WHERE chunk_id LIKE ? ESCAPE '\\'
                """,
                (prefix,),
            )

            connection.commit()

    def get_embedding(
        self,
        chunk_id: str,
    ) -> list[float] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT vector
                FROM embeddings
                WHERE chunk_id = ?
                """,
                (chunk_id,),
            ).fetchone()

        if row is None:
            return None

        vector = json.loads(row["vector"])

        if len(vector) != self.dimensions:
            raise ValueError(
                f"Stored vector has {len(vector)} dimensions, "
                f"expected {self.dimensions}"
            )

        return [float(value) for value in vector]

    def search(
        self,
        query_vector: list[float],
        limit: int = 5,
    ) -> list[VectorSearchResult]:
        if len(query_vector) != self.dimensions:
            raise ValueError(
                f"Expected query vector with {self.dimensions} dimensions"
            )

        if limit <= 0:
            raise ValueError("limit must be greater than zero")

        query_magnitude = math.sqrt(
            sum(value * value for value in query_vector)
        )

        if query_magnitude == 0:
            raise ValueError("query vector cannot be zero")

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT chunk_id, vector
                FROM embeddings
                """
            ).fetchall()

        results: list[VectorSearchResult] = []

        for row in rows:
            vector = json.loads(row["vector"])

            if len(vector) != self.dimensions:
                raise ValueError(
                    f"Stored vector has {len(vector)} dimensions, "
                    f"expected {self.dimensions}"
                )

            vector = [float(value) for value in vector]

            magnitude = math.sqrt(
                sum(value * value for value in vector)
            )

            if magnitude == 0:
                continue

            dot_product = sum(
                query_value * vector_value
                for query_value, vector_value in zip(
                    query_vector,
                    vector,
                )
            )

            score = dot_product / (
                query_magnitude * magnitude
            )

            results.append(
                VectorSearchResult(
                    chunk_id=row["chunk_id"],
                    score=score,
                )
            )

        results.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        return results[:limit]

    def count(self) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM embeddings"
            ).fetchone()

        return int(row["count"])
