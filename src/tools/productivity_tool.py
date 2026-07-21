from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any

from src.tools.tool import Tool


class ProductivityTool(Tool):
    """Natural-language task and reminder tool."""

    def __init__(
        self,
        manager,
    ):
        self.manager = manager

    @property
    def name(
        self,
    ):
        return "productivity"

    def can_handle(
        self,
        text: str,
    ) -> bool:
        normalized = self._normalize(
            text
        )

        phrases = (
            "remind me",
            "set a reminder",
            "my reminders",
            "list reminders",
            "show reminders",
            "change reminder",
            "update reminder",
            "edit reminder",
            "rename reminder",
            "reschedule reminder",
            "delete reminder",
            "remove reminder",
            "add task",
            "create task",
            "my tasks",
            "list tasks",
            "show tasks",
            "complete task",
            "finish task",
            "delete task",
            "remove task",
        )

        return any(
            phrase in normalized
            for phrase in phrases
        )

    def execute(
        self,
        text: str = "",
    ) -> str:
        normalized = self._normalize(
            text
        )

        if (
            "list reminders" in normalized
            or "show reminders" in normalized
            or "my reminders" in normalized
        ):
            return self._list_reminders()

        reminder_id = self._extract_id(
            normalized,
            (
                "delete reminder",
                "remove reminder",
            ),
        )

        if reminder_id is not None:
            if self.manager.delete_reminder(
                reminder_id
            ):
                return (
                    "I deleted reminder "
                    f"{reminder_id}."
                )

            return (
                "I could not find reminder "
                f"{reminder_id}."
            )

        reminder_id = self._extract_id(
            normalized,
            (
                "change reminder",
                "update reminder",
                "edit reminder",
                "rename reminder",
                "reschedule reminder",
            ),
        )

        if reminder_id is not None:
            return self._update_reminder(
                text=text,
                reminder_id=reminder_id,
            )

        if (
            "list tasks" in normalized
            or "show tasks" in normalized
            or "my tasks" in normalized
        ):
            return self._list_tasks()

        task_id = self._extract_id(
            normalized,
            (
                "complete task",
                "finish task",
            ),
        )

        if task_id is not None:
            if self.manager.complete_task(
                task_id
            ):
                return (
                    f"Task {task_id} "
                    "is complete."
                )

            return (
                "I could not find an open "
                f"task numbered {task_id}."
            )

        task_id = self._extract_id(
            normalized,
            (
                "delete task",
                "remove task",
            ),
        )

        if task_id is not None:
            if self.manager.delete_task(
                task_id
            ):
                return (
                    "I deleted task "
                    f"{task_id}."
                )

            return (
                "I could not find task "
                f"{task_id}."
            )

        if (
            "remind me" in normalized
            or (
                "set a reminder"
                in normalized
            )
        ):
            return self._create_reminder(
                text
            )

        if (
            "add task" in normalized
            or "create task" in normalized
        ):
            return self._create_task(
                text
            )

        return (
            "I could not understand that "
            "task or reminder request."
        )

    def run(
        self,
        text: str = "",
    ) -> str:
        return self.execute(
            text
        )

    def _create_task(
        self,
        text: str,
    ) -> str:
        cleaned = re.sub(
            (
                r"^\s*"
                r"(jay[,\s]+)?"
                r"(add|create)\s+"
                r"(a\s+)?task\s*"
                r"(to|for)?\s*"
            ),
            "",
            text,
            flags=re.IGNORECASE,
        ).strip(
            " ."
        )

        if not cleaned:
            return (
                "What task should I add?"
            )

        priority = "normal"
        lowered = cleaned.lower()

        if (
            " high priority"
            in lowered
        ):
            priority = "high"

            cleaned = re.sub(
                r"\s+high priority\s*$",
                "",
                cleaned,
                flags=re.IGNORECASE,
            )

        elif (
            " low priority"
            in lowered
        ):
            priority = "low"

            cleaned = re.sub(
                r"\s+low priority\s*$",
                "",
                cleaned,
                flags=re.IGNORECASE,
            )

        task_id = (
            self.manager.add_task(
                title=cleaned,
                priority=priority,
            )
        )

        return (
            f"I added task {task_id}: "
            f"{cleaned}. Priority is "
            f"{priority}."
        )

    def _create_reminder(
        self,
        text: str,
    ) -> str:
        normalized = self._normalize(
            text
        )

        remind_at = (
            self._parse_reminder_time(
                normalized
            )
        )

        if remind_at is None:
            return (
                "I need a reminder time. "
                "Try saying, 'remind me to "
                "test the camera tomorrow "
                "at 3 PM.'"
            )

        message = re.sub(
            (
                r"^\s*"
                r"(jay[,\s]+)?"
                r"(remind me|set a reminder)"
                r"\s*(to|for)?\s*"
            ),
            "",
            text,
            flags=re.IGNORECASE,
        )

        message = self._remove_time_phrase(
            message
        )

        if not message:
            return (
                "What should I remind "
                "you about?"
            )

        reminder_id = (
            self.manager.add_reminder(
                message=message,
                remind_at=remind_at,
            )
        )

        friendly_time = (
            self._format_reminder_time(
                remind_at
            )
        )

        return (
            "I created reminder "
            f"{reminder_id} for "
            f"{friendly_time}: "
            f"{message}."
        )

    def _update_reminder(
        self,
        text: str,
        reminder_id: int,
    ) -> str:
        existing = (
            self.manager.get_reminder(
                reminder_id
            )
        )

        if existing is None:
            return (
                "I could not find reminder "
                f"{reminder_id}."
            )

        normalized = self._normalize(
            text
        )

        new_time = (
            self._parse_reminder_time(
                normalized
            )
        )

        command_match = re.match(
            (
                r"^\s*"
                r"(jay[,\s]+)?"
                r"(change|update|edit|rename|reschedule)"
                r"\s+reminder\s+"
                rf"{reminder_id}\b"
                r"\s*(to|for)?\s*"
            ),
            text,
            flags=re.IGNORECASE,
        )

        if command_match is None:
            return (
                "I could not understand "
                "that reminder update."
            )

        command = (
            command_match
            .group(2)
            .lower()
        )

        new_message = text[
            command_match.end():
        ]

        new_message = (
            self._remove_time_phrase(
                new_message
            )
        )

        if command == "reschedule":
            new_message = ""

        if (
            not new_message
            and new_time is None
        ):
            return (
                "Tell me the new message, "
                "time, or both. For example: "
                "change reminder "
                f"{reminder_id} to finish "
                "the report tomorrow at "
                "6 PM."
            )

        try:
            changed = (
                self.manager
                .update_reminder(
                    reminder_id=(
                        reminder_id
                    ),
                    message=(
                        new_message
                        if new_message
                        else None
                    ),
                    remind_at=new_time,
                )
            )

        except ValueError as error:
            return str(
                error
            )

        if not changed:
            return (
                "I could not update reminder "
                f"{reminder_id}."
            )

        updated = (
            self.manager.get_reminder(
                reminder_id
            )
        )

        if updated is None:
            return (
                "Reminder "
                f"{reminder_id} was updated."
            )

        friendly_time = (
            self._format_reminder_time(
                datetime.fromisoformat(
                    updated.remind_at
                )
            )
        )

        return (
            "I updated reminder "
            f"{reminder_id}: "
            f"{updated.message}, scheduled "
            f"for {friendly_time}."
        )

    @staticmethod
    def _parse_reminder_time(
        text: str,
    ) -> datetime | None:
        match = re.search(
            (
                r"\b"
                r"(today|tomorrow)?"
                r"\s*at\s*"
                r"(\d{1,2})"
                r"(?::(\d{2}))?"
                r"\s*(am|pm)\b"
            ),
            text,
            flags=re.IGNORECASE,
        )

        if match is None:
            return None

        day_word = (
            match.group(1)
            or "today"
        ).lower()

        hour = int(
            match.group(2)
        )

        minute = int(
            match.group(3)
            or 0
        )

        meridiem = (
            match.group(4)
            .lower()
        )

        if (
            not 1 <= hour <= 12
            or not 0 <= minute <= 59
        ):
            return None

        if hour == 12:
            hour = 0

        if meridiem == "pm":
            hour += 12

        now = (
            datetime.now()
            .astimezone()
        )

        target = now.replace(
            hour=hour,
            minute=minute,
            second=0,
            microsecond=0,
        )

        if day_word == "tomorrow":
            target += timedelta(
                days=1
            )

        elif target <= now:
            target += timedelta(
                days=1
            )

        return target

    @staticmethod
    def _remove_time_phrase(
        text: str,
    ) -> str:
        cleaned = re.sub(
            (
                r"\s+"
                r"(today|tomorrow)"
                r"\s+at\s+"
                r"\d{1,2}"
                r"(?::\d{2})?"
                r"\s*(am|pm)"
                r"\s*$"
            ),
            "",
            text,
            flags=re.IGNORECASE,
        )

        cleaned = re.sub(
            (
                r"\s+at\s+"
                r"\d{1,2}"
                r"(?::\d{2})?"
                r"\s*(am|pm)"
                r"\s*$"
            ),
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        return cleaned.strip(
            " ."
        )

    @staticmethod
    def _format_reminder_time(
        value: datetime,
    ) -> str:
        return value.strftime(
            "%A, %B %d at %I:%M %p"
        ).replace(
            " 0",
            " ",
        )

    def _list_tasks(
        self,
    ) -> str:
        tasks = (
            self.manager.list_tasks(
                include_completed=False,
                limit=10,
            )
        )

        if not tasks:
            return (
                "You do not have any "
                "open tasks."
            )

        parts = [
            (
                f"Task {task.id}: "
                f"{task.title}, "
                f"{task.priority} priority"
            )
            for task in reversed(
                tasks
            )
        ]

        return (
            "Your open tasks are: "
            + "; ".join(parts)
            + "."
        )

    def _list_reminders(
        self,
    ) -> str:
        reminders = (
            self.manager.list_reminders(
                include_delivered=False,
                limit=10,
            )
        )

        if not reminders:
            return (
                "You do not have any "
                "pending reminders."
            )

        parts = []

        for reminder in reminders:
            when = (
                datetime.fromisoformat(
                    reminder.remind_at
                )
                .strftime(
                    "%A at %I:%M %p"
                )
                .replace(
                    " 0",
                    " ",
                )
            )

            parts.append(
                (
                    "Reminder "
                    f"{reminder.id}: "
                    f"{reminder.message}, "
                    f"{when}"
                )
            )

        return (
            "Your pending reminders are: "
            + "; ".join(parts)
            + ". You can change, "
            "reschedule, rename, or delete "
            "a reminder using its number."
        )

    @staticmethod
    def _extract_id(
        text: str,
        prefixes: tuple[
            str,
            ...,
        ],
    ) -> int | None:
        for prefix in prefixes:
            match = re.search(
                (
                    rf"\b{re.escape(prefix)}"
                    r"\s+(\d+)\b"
                ),
                text,
            )

            if match is not None:
                return int(
                    match.group(1)
                )

        return None

    @staticmethod
    def _normalize(
        text: Any,
    ) -> str:
        return re.sub(
            r"\s+",
            " ",
            str(text)
            .lower()
            .strip(),
        )