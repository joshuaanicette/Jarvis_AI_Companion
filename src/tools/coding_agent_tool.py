from __future__ import annotations

import re

from src.tools.tool import Tool


class CodingAgentTool(Tool):
    name = "coding_agent"

    def __init__(self, coding_agent) -> None:
        self.coding_agent = coding_agent

    def can_handle(self, text: str) -> bool:
        normalized = str(text).casefold().strip()
        return (
            normalized.startswith("approve coding proposal ")
            or normalized in {"list project files", "show project files"}
        )

    def run(self, text: str) -> str:
        approved = self.coding_agent.approve_from_text(text)
        if approved:
            return f"Approved and applied coding proposal {approved}."

        normalized = str(text).casefold().strip()
        if normalized in {"list project files", "show project files"}:
            files = self.coding_agent.list_files()
            return "Project files:\n" + "\n".join(f"- {path}" for path in files)

        return (
            "I can inspect files and stage a proposal, but I will not edit files "
            "until you approve a specific proposal ID."
        )
