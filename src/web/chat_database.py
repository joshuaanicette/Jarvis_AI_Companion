from __future__ import annotations

import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


class ChatDatabase:
    """Small, thread-safe SQLite store for browser conversations."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="microseconds")

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(
            self.database_path,
            timeout=10.0,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")

        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL DEFAULT 'New chat',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
            content TEXT NOT NULL,
            category TEXT,
            model TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (conversation_id)
                REFERENCES conversations(id)
                ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_messages_conversation
            ON messages(conversation_id, id);

        CREATE INDEX IF NOT EXISTS idx_conversations_updated
            ON conversations(updated_at DESC);
        """

        with self._lock, self._connection() as connection:
            connection.executescript(schema)
            connection.execute("PRAGMA journal_mode = WAL")

    @staticmethod
    def _title_from_message(message: str, max_length: int = 52) -> str:
        title = " ".join(str(message).split()).strip()
        if not title:
            return "New chat"
        if len(title) <= max_length:
            return title
        return title[: max_length - 1].rstrip() + "…"

    def create_conversation(
        self,
        title: str = "New chat",
        conversation_id: str | None = None,
    ) -> dict[str, object]:
        identifier = conversation_id or str(uuid.uuid4())
        clean_title = self._title_from_message(title)
        now = self._now()

        with self._lock, self._connection() as connection:
            connection.execute(
                """
                INSERT INTO conversations (id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (identifier, clean_title, now, now),
            )

        conversation = self.get_conversation(identifier)
        if conversation is None:
            raise RuntimeError("The conversation could not be created.")
        return conversation

    def ensure_conversation(self, conversation_id: str) -> dict[str, object]:
        now = self._now()
        with self._lock, self._connection() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO conversations
                    (id, title, created_at, updated_at)
                VALUES (?, 'New chat', ?, ?)
                """,
                (conversation_id, now, now),
            )

        conversation = self.get_conversation(conversation_id)
        if conversation is None:
            raise RuntimeError("The conversation could not be loaded.")
        return conversation

    def list_conversations(self, limit: int = 100) -> list[dict[str, object]]:
        safe_limit = max(1, min(int(limit), 250))
        query = """
            SELECT
                c.id,
                c.title,
                c.created_at,
                c.updated_at,
                COUNT(m.id) AS message_count
            FROM conversations AS c
            LEFT JOIN messages AS m ON m.conversation_id = c.id
            GROUP BY c.id
            ORDER BY c.updated_at DESC, c.rowid DESC
            LIMIT ?
        """

        with self._lock, self._connection() as connection:
            rows = connection.execute(query, (safe_limit,)).fetchall()
        return [dict(row) for row in rows]

    def get_conversation(self, conversation_id: str) -> dict[str, object] | None:
        query = """
            SELECT
                c.id,
                c.title,
                c.created_at,
                c.updated_at,
                COUNT(m.id) AS message_count
            FROM conversations AS c
            LEFT JOIN messages AS m ON m.conversation_id = c.id
            WHERE c.id = ?
            GROUP BY c.id
        """

        with self._lock, self._connection() as connection:
            row = connection.execute(query, (conversation_id,)).fetchone()
        return dict(row) if row is not None else None

    def get_messages(
        self,
        conversation_id: str,
        limit: int | None = None,
    ) -> list[dict[str, object]]:
        parameters: tuple[object, ...]

        if limit is None:
            query = """
                SELECT id, conversation_id, role, content, category, model, created_at
                FROM messages
                WHERE conversation_id = ?
                ORDER BY id ASC
            """
            parameters = (conversation_id,)
        else:
            safe_limit = max(1, min(int(limit), 500))
            query = """
                SELECT id, conversation_id, role, content, category, model, created_at
                FROM (
                    SELECT id, conversation_id, role, content, category, model, created_at
                    FROM messages
                    WHERE conversation_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                )
                ORDER BY id ASC
            """
            parameters = (conversation_id, safe_limit)

        with self._lock, self._connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [dict(row) for row in rows]

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        category: str | None = None,
        model: str | None = None,
    ) -> dict[str, object]:
        if role not in {"user", "assistant", "system"}:
            raise ValueError(f"Unsupported message role: {role}")

        clean_content = str(content).strip()
        if not clean_content:
            raise ValueError("Message content cannot be empty.")

        self.ensure_conversation(conversation_id)
        now = self._now()

        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO messages
                    (conversation_id, role, content, category, model, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    role,
                    clean_content,
                    category,
                    model,
                    now,
                ),
            )

            if role == "user":
                connection.execute(
                    """
                    UPDATE conversations
                    SET title = CASE
                            WHEN title IN ('New chat', 'New conversation')
                            THEN ?
                            ELSE title
                        END,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        self._title_from_message(clean_content),
                        now,
                        conversation_id,
                    ),
                )
            else:
                connection.execute(
                    """
                    UPDATE conversations
                    SET updated_at = ?
                    WHERE id = ?
                    """,
                    (now, conversation_id),
                )

            row = connection.execute(
                """
                SELECT id, conversation_id, role, content, category, model, created_at
                FROM messages
                WHERE id = ?
                """,
                (cursor.lastrowid,),
            ).fetchone()

        if row is None:
            raise RuntimeError("The message could not be saved.")
        return dict(row)

    def rename_conversation(
        self,
        conversation_id: str,
        title: str,
    ) -> dict[str, object] | None:
        clean_title = self._title_from_message(title)
        now = self._now()

        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE conversations
                SET title = ?, updated_at = ?
                WHERE id = ?
                """,
                (clean_title, now, conversation_id),
            )

        if cursor.rowcount == 0:
            return None
        return self.get_conversation(conversation_id)

    def delete_conversation(self, conversation_id: str) -> bool:
        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM conversations WHERE id = ?",
                (conversation_id,),
            )
        return cursor.rowcount > 0