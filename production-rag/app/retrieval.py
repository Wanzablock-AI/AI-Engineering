from dataclasses import dataclass
from pathlib import Path
import sqlite3


@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    document: str
    page_number: int
    text: str
    score: float


class LexicalRetriever:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)

        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Database not found: {self.db_path}"
            )

        self._initialize_fts()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_fts(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
                USING fts5(
                    chunk_id UNINDEXED,
                    document UNINDEXED,
                    page_number UNINDEXED,
                    text
                )
                """
            )

            connection.execute(
                """
                INSERT INTO chunks_fts (
                    chunk_id,
                    document,
                    page_number,
                    text
                )
                SELECT
                    c.chunk_id,
                    c.document,
                    c.page_number,
                    c.text
                FROM chunks AS c
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM chunks_fts AS f
                    WHERE f.chunk_id = c.chunk_id
                )
                """
            )

            connection.commit()

    def search(
        self,
        query: str,
        limit: int = 5,
    ) -> list[SearchResult]:
        if not query.strip():
            return []

        if limit <= 0:
            raise ValueError("limit must be greater than zero")

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    chunk_id,
                    document,
                    page_number,
                    text,
                    bm25(chunks_fts) AS score
                FROM chunks_fts
                WHERE chunks_fts MATCH ?
                ORDER BY score
                LIMIT ?
                """,
                (query, limit),
            ).fetchall()

        return [
            SearchResult(
                chunk_id=row["chunk_id"],
                document=row["document"],
                page_number=row["page_number"],
                text=row["text"],
                score=float(row["score"]),
            )
            for row in rows
        ]
