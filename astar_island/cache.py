from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class SqliteJsonCache:
    def __init__(self, db_path: str) -> None:
        self.db_path = str(Path(db_path))
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_entries (
                    cache_key TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def get(self, cache_key: str) -> Any | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM cache_entries WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def set(self, cache_key: str, payload: Any) -> None:
        payload_json = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO cache_entries (cache_key, payload_json)
                VALUES (?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET payload_json = excluded.payload_json
                """,
                (cache_key, payload_json),
            )
            conn.commit()

