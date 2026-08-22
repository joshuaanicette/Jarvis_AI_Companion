from __future__ import annotations

from src.tools.tool import Tool


class CodingAgentTool(Tool):
    @property
    def name(self) -> str:
        return "coding_agent"

    TASK_PREFIXES = ("coding task:", "code task:", "propose code change:")
    ANALYZE_PREFIXES = ("coding analyze:", "analyze code:", "decipher code:")
    SUGGEST_PREFIXES = ("coding suggest:", "suggest feature:", "plan feature:")
    ADD_FILE_PREFIX = "coding add file:"

    def __init__(self, coding_agent, llm, model: str = "qwen2.5:3b") -> None:
        self.coding_agent = coding_agent
        self.llm = llm
        self.model = model

    def can_handle(self, text: str) -> bool:
        normalized = str(text).casefold().strip()
        return (
            normalized.startswith("approve coding proposal ")
            or normalized in {"list project files", "show project files", "coding help"}
            or normalized.startswith(self.ADD_FILE_PREFIX)
            or any(normalized.startswith(prefix) for prefix in (
                *self.TASK_PREFIXES,
                *self.ANALYZE_PREFIXES,
                *self.SUGGEST_PREFIXES,
            ))
        )

    def execute(self, text: str = "") -> str:
        try:
            approved = self.coding_agent.approve_from_text(text)
            if approved:
                return f"Approved and applied coding proposal {approved}."

            normalized = str(text).casefold().strip()
            if normalized in {"list project files", "show project files"}:
                files = self.coding_agent.list_files()
                return "Project files:\n" + "\n".join(f"- {path}" for path in files)

            if normalized == "coding help":
                return (
                    "Coding commands:\n"
                    "- coding analyze: explain src/file.py and how it supports <feature>\n"
                    "- coding suggest: I want <feature>\n"
                    "- coding add file: src/new_file.py | <required functionality>\n"
                    "- coding task: update src/file.py to <change>\n"
                    "- approve coding proposal <proposal-id>"
                )

            if normalized.startswith(self.ADD_FILE_PREFIX):
                payload = str(text)[len(self.ADD_FILE_PREFIX):].strip()
                path, separator, description = payload.partition("|")
                if not separator or not path.strip() or not description.strip():
                    return "Use: coding add file: src/new_file.py | describe the required functionality"
                proposal = self.coding_agent.propose_new_file(
                    path.strip(), description.strip(), self.llm, self.model
                )
                return self.coding_agent.preview(proposal.proposal_id)

            for prefix in self.ANALYZE_PREFIXES:
                if normalized.startswith(prefix):
                    request = str(text)[len(prefix):].strip()
                    return self.coding_agent.analyze_from_prompt(request, self.llm, self.model)

            for prefix in self.SUGGEST_PREFIXES:
                if normalized.startswith(prefix):
                    request = str(text)[len(prefix):].strip()
                    return self.coding_agent.suggest_from_prompt(request, self.llm, self.model)

            for prefix in self.TASK_PREFIXES:
                if normalized.startswith(prefix):
                    task = str(text)[len(prefix):].strip()
                    proposal = self.coding_agent.propose_from_prompt(task, self.llm, self.model)
                    return self.coding_agent.preview(proposal.proposal_id)

        except (KeyError, ValueError) as error:
            return f"Coding request needs adjustment: {error}"
        except Exception as error:
            error_text = str(error)
            if "timed out" in error_text.casefold() or "read timeout" in error_text.casefold():
                return (
                    "The local Qwen coding model took too long to analyze the "
                    "selected code. Try one smaller file at a time, shorten the "
                    "request, or warm the model with 'ollama run qwen2.5:3b' "
                    f"before trying again. Details: {error_text}"
                )
            return f"Coding assistant could not complete that request: {error_text}"

        return "Say 'coding help' to see the available coding commands."

    def run(self, text: str = "") -> str:
        """Compatibility alias used by ToolRouter."""
        return self.execute(text)
