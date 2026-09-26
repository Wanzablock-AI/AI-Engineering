import sqlite3
from pathlib import Path


class SQLiteEventStore:
    def __init__(self, path):
        self.path = Path(path)

        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request TEXT NOT NULL,
                    model TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    latency_ms REAL NOT NULL,
                    success INTEGER NOT NULL
                )
                """
            )
            connection.commit()

    def save(
        self,
        request: str,
        model: str,
        provider: str,
        latency_ms: float,
        success: bool,
    ):
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                INSERT INTO events (
                    request,
                    model,
                    provider,
                    latency_ms,
                    success
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    request,
                    model,
                    provider,
                    latency_ms,
                    int(success),
                ),
            )
            connection.commit()

    def metrics(self):
        with sqlite3.connect(self.path) as connection:
            row = connection.execute(
                """
                SELECT
                    COUNT(*) AS total_events,
                    COALESCE(SUM(success), 0) AS successful_events,
                    COALESCE(AVG(latency_ms), 0.0) AS average_latency_ms
                FROM events
                """
            ).fetchone()

        total_events = row[0]
        successful_events = row[1]
        failed_events = total_events - successful_events

        success_rate = (
            successful_events / total_events
            if total_events
            else 0.0
        )

        return {
            "total_events": total_events,
            "successful_events": successful_events,
            "failed_events": failed_events,
            "success_rate": success_rate,
            "average_latency_ms": row[2],
        }

    def list_events(self):
        with sqlite3.connect(self.path) as connection:
            connection.row_factory = sqlite3.Row

            rows = connection.execute(
                """
                SELECT request, model, provider, latency_ms, success
                FROM events
                ORDER BY id
                """
            ).fetchall()

        return [
            {
                "request": row["request"],
                "model": row["model"],
                "provider": row["provider"],
                "latency_ms": row["latency_ms"],
                "success": bool(row["success"]),
            }
            for row in rows
        ]
