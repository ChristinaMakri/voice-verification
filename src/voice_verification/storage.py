"""Voiceprint persistence. Stores only encrypted embeddings, never audio."""

import threading
from collections import defaultdict
from collections.abc import Sequence
from typing import Protocol

from psycopg_pool import ConnectionPool

SCHEMA = """
CREATE TABLE IF NOT EXISTS voiceprints (
    id            BIGSERIAL PRIMARY KEY,
    user_id       TEXT        NOT NULL,
    model_id      TEXT        NOT NULL,
    embedding_enc BYTEA       NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS voiceprints_user_model_idx ON voiceprints (user_id, model_id);
"""


class VoiceprintStore(Protocol):
    def add_many(self, user_id: str, model_id: str, blobs: Sequence[bytes]) -> None: ...
    def list(self, user_id: str, model_id: str) -> list[bytes]: ...
    def count(self, user_id: str, model_id: str) -> int: ...
    def delete(self, user_id: str) -> int:
        """Delete every voiceprint of the user (all models). Returns rows removed."""
        ...


class InMemoryStore:
    """For tests and local experiments."""

    def __init__(self) -> None:
        self._rows: dict[tuple[str, str], list[bytes]] = defaultdict(list)
        self._lock = threading.Lock()

    def add_many(self, user_id: str, model_id: str, blobs: Sequence[bytes]) -> None:
        with self._lock:
            self._rows[(user_id, model_id)].extend(blobs)

    def list(self, user_id: str, model_id: str) -> list[bytes]:
        with self._lock:
            return list(self._rows.get((user_id, model_id), []))

    def count(self, user_id: str, model_id: str) -> int:
        return len(self.list(user_id, model_id))

    def delete(self, user_id: str) -> int:
        with self._lock:
            keys = [k for k in self._rows if k[0] == user_id]
            return sum(len(self._rows.pop(k)) for k in keys)


class PostgresStore:
    def __init__(self, database_url: str, min_size: int = 1, max_size: int = 5):
        self._pool = ConnectionPool(database_url, min_size=min_size, max_size=max_size, open=True)
        with self._pool.connection() as conn:
            conn.execute(SCHEMA)

    def close(self) -> None:
        self._pool.close()

    def add_many(self, user_id: str, model_id: str, blobs: Sequence[bytes]) -> None:
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO voiceprints (user_id, model_id, embedding_enc) VALUES (%s, %s, %s)",
                [(user_id, model_id, b) for b in blobs],
            )

    def list(self, user_id: str, model_id: str) -> list[bytes]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT embedding_enc FROM voiceprints"
                " WHERE user_id = %s AND model_id = %s ORDER BY id",
                (user_id, model_id),
            ).fetchall()
        return [bytes(r[0]) for r in rows]

    def count(self, user_id: str, model_id: str) -> int:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT count(*) FROM voiceprints WHERE user_id = %s AND model_id = %s",
                (user_id, model_id),
            ).fetchone()
        return int(row[0])

    def delete(self, user_id: str) -> int:
        with self._pool.connection() as conn:
            return conn.execute("DELETE FROM voiceprints WHERE user_id = %s", (user_id,)).rowcount
