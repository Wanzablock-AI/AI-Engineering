import json
import sqlite3
from pathlib import Path


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

    def count(self) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM embeddings"
            ).fetchone()

        return int(row["count"])
