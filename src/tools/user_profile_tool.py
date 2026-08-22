from __future__ import annotations

from src.tools.tool import Tool


class UserProfileTool(Tool):
    """Let the user inspect and selectively forget Jarvis profile data."""

    STATUS_COMMANDS = {
        "profile status",
        "show my profile",
        "show profile",
        "what do you know about me",
        "what have you learned about me",
        "personal database status",
    }
    FORGET_PREFIXES = (
        "profile forget:",
        "forget profile:",
    )

    def __init__(
        self,
        profile_database,
        memory_manager=None,
    ) -> None:
        self.profile_database = profile_database
        self.memory_manager = memory_manager

    @property
    def name(self) -> str:
        return "user_profile"

    def can_handle(self, text: str) -> bool:
        normalized = str(text).casefold().strip()
        return (
            normalized in self.STATUS_COMMANDS
            or any(
                normalized.startswith(prefix)
                for prefix in self.FORGET_PREFIXES
            )
        )

    def execute(self, text: str = "") -> str:
        normalized = str(text).casefold().strip()
        if normalized in self.STATUS_COMMANDS:
            return self.profile_database.get_profile_summary()

        for prefix in self.FORGET_PREFIXES:
            if not normalized.startswith(prefix):
                continue

            query = str(text)[len(prefix):].strip()
            if not query:
                return (
                    "Name what to forget, for example: "
                    "profile forget: robotics"
                )

            removed = self.profile_database.forget_knowledge(query)
            if self.memory_manager is not None:
                removed += int(
                    self.memory_manager.forget_memory(query)
                )

            if removed:
                return (
                    f"I removed {removed} matching saved profile "
                    f"entr{'y' if removed == 1 else 'ies'} for {query}."
                )
            return f"I did not find saved profile data matching {query}."

        return (
            "Say 'profile status' to review saved learning or "
            "'profile forget: <topic>' to remove it."
        )

    def run(self, text: str = "") -> str:
        return self.execute(text)
