"""SQLite-backed memory_store — history + recent messages buffer.

Ported nearly verbatim from ``mem0/memory/storage.py`` (``SQLiteManager``)
with two simplifications:

- No legacy schema migration path (LiteMem starts fresh).
- ``history`` table uses the V3 schema only.

Schema:
    history(id, memory_id, old_memory, new_memory, event,
            created_at, updated_at, is_deleted, actor_id, role)

    messages(id, session_scope, role, content, name, created_at)

``messages`` keeps only the most recent ``recent_messages_limit`` rows per
``session_scope`` (default 10) — same eviction behavior as mem0, used to
seed the "Last k Messages" section of the additive extraction prompt.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MemoryStore:
    """Sync SQLite manager for history + recent messages."""

    def __init__(self, db_path: str = ":memory:", recent_messages_limit: int = 10):
        self.db_path = db_path
        self.recent_messages_limit = recent_messages_limit
        if db_path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self.connection = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._create_tables()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _create_tables(self) -> None:
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                self.connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS history (
                        id           TEXT PRIMARY KEY,
                        memory_id    TEXT,
                        old_memory   TEXT,
                        new_memory   TEXT,
                        event        TEXT,
                        created_at   DATETIME,
                        updated_at   DATETIME,
                        is_deleted   INTEGER,
                        actor_id     TEXT,
                        role         TEXT
                    )
                    """
                )
                self.connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS messages (
                        id            TEXT PRIMARY KEY,
                        session_scope TEXT,
                        role          TEXT,
                        content       TEXT,
                        name          TEXT,
                        created_at    DATETIME
                    )
                    """
                )
                self.connection.execute(
                    "CREATE INDEX IF NOT EXISTS idx_history_memory_id ON history(memory_id)"
                )
                self.connection.execute(
                    "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_scope, created_at)"
                )
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to create memory_store tables: {e}")
                raise

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def add_history(
        self,
        memory_id: str,
        old_memory: Optional[str],
        new_memory: Optional[str],
        event: str,
        *,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
        is_deleted: int = 0,
        actor_id: Optional[str] = None,
        role: Optional[str] = None,
    ) -> None:
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                self.connection.execute(
                    """
                    INSERT INTO history (
                        id, memory_id, old_memory, new_memory, event,
                        created_at, updated_at, is_deleted, actor_id, role
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(uuid.uuid4()),
                        memory_id,
                        old_memory,
                        new_memory,
                        event,
                        created_at,
                        updated_at,
                        is_deleted,
                        actor_id,
                        role,
                    ),
                )
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to add history record: {e}")
                raise

    def batch_add_history(self, records: List[Dict[str, Any]]) -> None:
        if not records:
            return
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                self.connection.executemany(
                    """
                    INSERT INTO history (
                        id, memory_id, old_memory, new_memory, event,
                        created_at, updated_at, is_deleted, actor_id, role
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            str(uuid.uuid4()),
                            r.get("memory_id"),
                            r.get("old_memory"),
                            r.get("new_memory"),
                            r.get("event"),
                            r.get("created_at"),
                            r.get("updated_at"),
                            r.get("is_deleted", 0),
                            r.get("actor_id"),
                            r.get("role"),
                        )
                        for r in records
                    ],
                )
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to batch add history: {e}")
                raise

    def get_history(self, memory_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            cur = self.connection.execute(
                """
                SELECT id, memory_id, old_memory, new_memory, event,
                       created_at, updated_at, is_deleted, actor_id, role
                FROM history
                WHERE memory_id = ?
                ORDER BY created_at ASC, DATETIME(updated_at) ASC
                """,
                (memory_id,),
            )
            rows = cur.fetchall()
        return [
            {
                "id": r[0], "memory_id": r[1], "old_memory": r[2], "new_memory": r[3],
                "event": r[4], "created_at": r[5], "updated_at": r[6],
                "is_deleted": bool(r[7]), "actor_id": r[8], "role": r[9],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Recent messages buffer
    # ------------------------------------------------------------------

    def save_messages(self, messages: List[Dict[str, Any]], session_scope: str) -> None:
        if not messages:
            return
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                now = datetime.now(timezone.utc).isoformat()
                for m in messages:
                    self.connection.execute(
                        """
                        INSERT INTO messages (id, session_scope, role, content, name, created_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            str(uuid.uuid4()),
                            session_scope,
                            m.get("role"),
                            m.get("content"),
                            m.get("name"),
                            now,
                        ),
                    )
                # Evict everything beyond the most-recent N for this session.
                self.connection.execute(
                    """
                    DELETE FROM messages WHERE session_scope = ? AND id NOT IN (
                        SELECT id FROM (
                            SELECT id FROM messages
                            WHERE session_scope = ?
                            ORDER BY created_at DESC
                            LIMIT ?
                        )
                    )
                    """,
                    (session_scope, session_scope, self.recent_messages_limit),
                )
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to save messages: {e}")
                raise

    def get_last_messages(self, session_scope: str, limit: int = 10) -> List[Dict[str, Any]]:
        with self._lock:
            cur = self.connection.execute(
                """
                SELECT role, content, name, created_at FROM (
                    SELECT role, content, name, created_at
                    FROM messages
                    WHERE session_scope = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                ) ORDER BY created_at ASC
                """,
                (session_scope, limit),
            )
            rows = cur.fetchall()
        return [
            {"role": r[0], "content": r[1], "name": r[2], "created_at": r[3]}
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def reset(self) -> None:
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                self.connection.execute("DROP TABLE IF EXISTS history")
                self.connection.execute("DROP TABLE IF EXISTS messages")
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to reset memory_store: {e}")
                raise
        self._create_tables()

    def close(self) -> None:
        if self.connection is not None:
            try:
                self.connection.close()
            finally:
                self.connection = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def build_session_scope(filters: Dict[str, Any]) -> str:
    """Deterministic session scope string for the messages table.

    Same scheme mem0 uses: ``key=value`` pairs joined by ``&``, sorted by key.
    """
    parts = []
    for key in sorted(["user_id", "agent_id", "run_id"]):
        val = filters.get(key)
        if val:
            parts.append(f"{key}={val}")
    return "&".join(parts)
