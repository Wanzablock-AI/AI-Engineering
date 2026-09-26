import sqlite3
from pathlib import Path

from app.chunking import Chunk


DEFAULT_DB_PATH = Path("data/production-rag.db")


class ChunkStore:
    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
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
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    text TEXT NOT NULL
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chunks_document
                ON chunks(document)
                """
            )

            connection.commit()

    def add_chunks(self, chunks: list[Chunk]) -> None:
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO chunks (
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
                DELETE FROM chunks
                WHERE document = ?
                """,
                (document,),
            )

            connection.commit()

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT chunk_id, document, page_number, text
                FROM chunks
                WHERE chunk_id = ?
                """,
                (chunk_id,),
            ).fetchone()

        if row is None:
            return None

        return Chunk(
            chunk_id=row["chunk_id"],
            document=row["document"],
            page_number=row["page_number"],
            text=row["text"],
        )

    def count(self) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM chunks"
            ).fetchone()

        return int(row["count"])
