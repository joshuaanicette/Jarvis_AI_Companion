from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


@dataclass(slots=True)
class TaskRecord:
    id: int
    title: str
    status: str
    priority: str
    due_at: str | None
    created_at: str


@dataclass(slots=True)
class ReminderRecord:
    id: int
    message: str
    remind_at: str
    delivered: bool
    created_at: str


class ProductivityManager:
    """Persistent local task and reminder storage for Jarvis."""

    def __init__(
        self,
        database_path: str | Path = (
            "data/productivity/productivity.db"
        ),
        poll_interval_seconds: float = 5.0,
    ):
        self.database_path = Path(
            database_path
        )

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.poll_interval_seconds = max(
            1.0,
            float(
                poll_interval_seconds
            ),
        )

        self._stop_event = threading.Event()

        self._thread: (
            threading.Thread
            | None
        ) = None

        self._callback: (
            Callable[[str], None]
            | None
        ) = None

        self._lock = threading.RLock()

        self._initialize_database()

    def _connect(
        self,
    ) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path,
            timeout=10.0,
        )

        connection.row_factory = (
            sqlite3.Row
        )

        return connection

    def _initialize_database(
        self,
    ) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'open',
                    priority TEXT NOT NULL DEFAULT 'normal',
                    due_at TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS reminders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message TEXT NOT NULL,
                    remind_at TEXT NOT NULL,
                    delivered INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );
                """
            )

    @staticmethod
    def _now_iso(
        ) -> str:
        return (
            datetime.now(
                timezone.utc
            )
            .astimezone()
            .isoformat(
                timespec="seconds"
            )
        )

    @staticmethod
    def _normalize_datetime(
        value: datetime | str | None,
    ) -> str | None:
        if value is None:
            return None

        if isinstance(
            value,
            datetime,
        ):
            if value.tzinfo is None:
                value = value.astimezone()

            return value.isoformat(
                timespec="seconds"
            )

        parsed = datetime.fromisoformat(
            str(value)
        )

        if parsed.tzinfo is None:
            parsed = parsed.astimezone()

        return parsed.isoformat(
            timespec="seconds"
        )

    def add_task(
        self,
        title: str,
        priority: str = "normal",
        due_at: datetime | str | None = None,
    ) -> int:
        cleaned = title.strip()

        if not cleaned:
            raise ValueError(
                "Task title cannot be empty."
            )

        normalized_priority = (
            priority
            .strip()
            .lower()
        )

        if normalized_priority not in {
            "low",
            "normal",
            "high",
        }:
            normalized_priority = "normal"

        with (
            self._lock,
            self._connect()
        ) as connection:
            cursor = connection.execute(
                """
                INSERT INTO tasks (
                    title,
                    status,
                    priority,
                    due_at,
                    created_at
                )
                VALUES (?, 'open', ?, ?, ?)
                """,
                (
                    cleaned,
                    normalized_priority,
                    self._normalize_datetime(
                        due_at
                    ),
                    self._now_iso(),
                ),
            )

            return int(
                cursor.lastrowid
            )

    def list_tasks(
        self,
        include_completed: bool = False,
        limit: int = 20,
    ) -> list[TaskRecord]:
        query = """
            SELECT
                id,
                title,
                status,
                priority,
                due_at,
                created_at
            FROM tasks
        """

        parameters: list[
            object
        ] = []

        if not include_completed:
            query += (
                " WHERE status != 'completed'"
            )

        query += (
            " ORDER BY id DESC LIMIT ?"
        )

        parameters.append(
            max(
                1,
                int(limit),
            )
        )

        with (
            self._lock,
            self._connect()
        ) as connection:
            rows = connection.execute(
                query,
                parameters,
            ).fetchall()

        return [
            TaskRecord(
                id=int(
                    row["id"]
                ),
                title=str(
                    row["title"]
                ),
                status=str(
                    row["status"]
                ),
                priority=str(
                    row["priority"]
                ),
                due_at=row["due_at"],
                created_at=str(
                    row["created_at"]
                ),
            )
            for row in rows
        ]

    def complete_task(
        self,
        task_id: int,
    ) -> bool:
        with (
            self._lock,
            self._connect()
        ) as connection:
            cursor = connection.execute(
                """
                UPDATE tasks
                SET status = 'completed'
                WHERE id = ?
                  AND status != 'completed'
                """,
                (
                    int(task_id),
                ),
            )

            return cursor.rowcount > 0

    def delete_task(
        self,
        task_id: int,
    ) -> bool:
        with (
            self._lock,
            self._connect()
        ) as connection:
            cursor = connection.execute(
                """
                DELETE FROM tasks
                WHERE id = ?
                """,
                (
                    int(task_id),
                ),
            )

            return cursor.rowcount > 0

    def add_reminder(
        self,
        message: str,
        remind_at: datetime | str,
    ) -> int:
        cleaned = message.strip()

        if not cleaned:
            raise ValueError(
                (
                    "Reminder message "
                    "cannot be empty."
                )
            )

        normalized_time = (
            self._normalize_datetime(
                remind_at
            )
        )

        if normalized_time is None:
            raise ValueError(
                (
                    "Reminder time "
                    "is required."
                )
            )

        with (
            self._lock,
            self._connect()
        ) as connection:
            cursor = connection.execute(
                """
                INSERT INTO reminders (
                    message,
                    remind_at,
                    delivered,
                    created_at
                )
                VALUES (?, ?, 0, ?)
                """,
                (
                    cleaned,
                    normalized_time,
                    self._now_iso(),
                ),
            )

            return int(
                cursor.lastrowid
            )

    def get_reminder(
        self,
        reminder_id: int,
    ) -> ReminderRecord | None:
        with (
            self._lock,
            self._connect()
        ) as connection:
            row = connection.execute(
                """
                SELECT
                    id,
                    message,
                    remind_at,
                    delivered,
                    created_at
                FROM reminders
                WHERE id = ?
                """,
                (
                    int(
                        reminder_id
                    ),
                ),
            ).fetchone()

        if row is None:
            return None

        return ReminderRecord(
            id=int(
                row["id"]
            ),
            message=str(
                row["message"]
            ),
            remind_at=str(
                row["remind_at"]
            ),
            delivered=bool(
                row["delivered"]
            ),
            created_at=str(
                row["created_at"]
            ),
        )

    def list_reminders(
        self,
        include_delivered: bool = False,
        limit: int = 20,
    ) -> list[ReminderRecord]:
        query = """
            SELECT
                id,
                message,
                remind_at,
                delivered,
                created_at
            FROM reminders
        """

        parameters: list[
            object
        ] = []

        if not include_delivered:
            query += (
                " WHERE delivered = 0"
            )

        query += (
            " ORDER BY remind_at ASC LIMIT ?"
        )

        parameters.append(
            max(
                1,
                int(limit),
            )
        )

        with (
            self._lock,
            self._connect()
        ) as connection:
            rows = connection.execute(
                query,
                parameters,
            ).fetchall()

        return [
            ReminderRecord(
                id=int(
                    row["id"]
                ),
                message=str(
                    row["message"]
                ),
                remind_at=str(
                    row["remind_at"]
                ),
                delivered=bool(
                    row["delivered"]
                ),
                created_at=str(
                    row["created_at"]
                ),
            )
            for row in rows
        ]

    def update_reminder(
        self,
        reminder_id: int,
        message: str | None = None,
        remind_at: datetime | str | None = None,
    ) -> bool:
        reminder = self.get_reminder(
            reminder_id
        )

        if reminder is None:
            return False

        new_message = reminder.message
        new_time = reminder.remind_at

        if message is not None:
            cleaned_message = (
                str(message).strip()
            )

            if not cleaned_message:
                raise ValueError(
                    (
                        "Reminder message "
                        "cannot be empty."
                    )
                )

            new_message = cleaned_message

        if remind_at is not None:
            normalized_time = (
                self._normalize_datetime(
                    remind_at
                )
            )

            if normalized_time is None:
                raise ValueError(
                    (
                        "Reminder time "
                        "is required."
                    )
                )

            new_time = normalized_time

        with (
            self._lock,
            self._connect()
        ) as connection:
            cursor = connection.execute(
                """
                UPDATE reminders
                SET
                    message = ?,
                    remind_at = ?,
                    delivered = 0
                WHERE id = ?
                """,
                (
                    new_message,
                    new_time,
                    int(
                        reminder_id
                    ),
                ),
            )

            return cursor.rowcount > 0

    def delete_reminder(
        self,
        reminder_id: int,
    ) -> bool:
        with (
            self._lock,
            self._connect()
        ) as connection:
            cursor = connection.execute(
                """
                DELETE FROM reminders
                WHERE id = ?
                """,
                (
                    int(
                        reminder_id
                    ),
                ),
            )

            return cursor.rowcount > 0

    def get_due_reminders(
        self,
    ) -> list[ReminderRecord]:
        now = (
            datetime.now()
            .astimezone()
            .isoformat(
                timespec="seconds"
            )
        )

        with (
            self._lock,
            self._connect()
        ) as connection:
            rows = connection.execute(
                """
                SELECT
                    id,
                    message,
                    remind_at,
                    delivered,
                    created_at
                FROM reminders
                WHERE delivered = 0
                  AND remind_at <= ?
                ORDER BY remind_at ASC
                """,
                (
                    now,
                ),
            ).fetchall()

            reminder_ids = [
                int(
                    row["id"]
                )
                for row in rows
            ]

            if reminder_ids:
                placeholders = ",".join(
                    "?"
                    for _ in reminder_ids
                )

                connection.execute(
                    f"""
                    UPDATE reminders
                    SET delivered = 1
                    WHERE id IN ({placeholders})
                    """,
                    reminder_ids,
                )

        return [
            ReminderRecord(
                id=int(
                    row["id"]
                ),
                message=str(
                    row["message"]
                ),
                remind_at=str(
                    row["remind_at"]
                ),
                delivered=True,
                created_at=str(
                    row["created_at"]
                ),
            )
            for row in rows
        ]

    def start(
        self,
        callback: Callable[
            [str],
            None,
        ],
    ) -> bool:
        if (
            self._thread is not None
            and self._thread.is_alive()
        ):
            return False

        self._callback = callback
        self._stop_event.clear()

        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name=(
                "JarvisReminderMonitor"
            ),
        )

        self._thread.start()

        return True

    def stop(
        self,
    ) -> None:
        self._stop_event.set()

        if (
            self._thread is not None
            and self._thread.is_alive()
            and (
                self._thread
                is not threading.current_thread()
            )
        ):
            self._thread.join(
                timeout=2.0
            )

        self._thread = None

    def _run_loop(
        self,
    ) -> None:
        while not self._stop_event.is_set():
            try:
                due = (
                    self.get_due_reminders()
                )

                if self._callback is not None:
                    for reminder in due:
                        self._callback(
                            (
                                "Reminder: "
                                f"{reminder.message}"
                            )
                        )

            except Exception:
                pass

            self._stop_event.wait(
                self.poll_interval_seconds
            )