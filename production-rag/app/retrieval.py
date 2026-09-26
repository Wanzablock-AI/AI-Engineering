from dataclasses import dataclass

from app.chunking import Chunk
from pathlib import Path
import re
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
                "DELETE FROM chunks_fts"
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
                    chunk_id,
                    document,
                    page_number,
                    text
                FROM chunks
                """
            )

            connection.commit()

    @staticmethod
    def _build_fts_query(query: str) -> str:
        terms = re.findall(r"\w+", query, flags=re.UNICODE)

        return " OR ".join(
            f'"{term.replace(chr(34), chr(34) * 2)}"'
            for term in terms
        )

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return

        with self._connect() as connection:
            connection.executemany(
                """
                DELETE FROM chunks_fts
                WHERE chunk_id = ?
                """,
                [
                    (chunk.chunk_id,)
                    for chunk in chunks
                ],
            )

            connection.executemany(
                """
                INSERT INTO chunks_fts (
                    chunk_id,
                    document,
                    page_number,
                    text
                )
                VALUES (?, ?, ?, ?)
                """,
                [
                    (
                        chunk.chunk_id,
                        chunk.document,
                        chunk.page_number,
                        chunk.text,
                    )
                    for chunk in chunks
                ],
            )

            connection.commit()

    def delete_document(self, document: str) -> None:
        if not document.strip():
            raise ValueError("document cannot be empty")

        with self._connect() as connection:
            connection.execute(
                """
                DELETE FROM chunks_fts
                WHERE document = ?
                """,
                (document,),
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

        fts_query = self._build_fts_query(query)

        if not fts_query:
            return []

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
                (fts_query, limit),
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
