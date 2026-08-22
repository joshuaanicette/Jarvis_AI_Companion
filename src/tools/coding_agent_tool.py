from __future__ import annotations

from src.tools.tool import Tool


class CodingAgentTool(Tool):
    name = "coding_agent"

    TASK_PREFIXES = (
        "coding task:",
        "code task:",
        "propose code change:",
    )

    def __init__(self, coding_agent, llm, model: str = "qwen2.5:3b") -> None:
        self.coding_agent = coding_agent
        self.llm = llm
        self.model = model

    def can_handle(self, text: str) -> bool:
        normalized = str(text).casefold().strip()
        return (
            normalized.startswith("approve coding proposal ")
            or normalized in {"list project files", "show project files"}
            or any(normalized.startswith(prefix) for prefix in self.TASK_PREFIXES)
        )

    def run(self, text: str) -> str:
        approved = self.coding_agent.approve_from_text(text)
        if approved:
            return f"Approved and applied coding proposal {approved}."

        normalized = str(text).casefold().strip()
        if normalized in {"list project files", "show project files"}:
            files = self.coding_agent.list_files()
            return "Project files:\n" + "\n".join(f"- {path}" for path in files)

        for prefix in self.TASK_PREFIXES:
            if normalized.startswith(prefix):
                task = str(text)[len(prefix):].strip()
                if not task:
                    return "Describe the coding task and name the file or files to inspect."
                proposal = self.coding_agent.propose_from_prompt(
                    task=task,
                    llm=self.llm,
                    model=self.model,
                )
                return self.coding_agent.preview(proposal.proposal_id)

        return (
            "I can inspect files and stage a proposal, but I will not edit files "
            "until you approve a specific proposal ID."
        )
