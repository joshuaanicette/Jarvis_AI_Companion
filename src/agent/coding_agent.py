from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.agent.approval import ApprovalGate
from src.ai.memory import MemoryManager
from src.ai.memory_retriever import MemoryRetriever
from src.ai.model_router import ModelDecision, ModelRouter
from src.ai.ollama_llm import OllamaLLM
from src.tools.coding_tools import CodingTools, ToolResult


@dataclass(slots=True)
class CodingAgentResult:
    response: str
    model: str
    reason: str
    memory_context_used: bool


class CodingAgent:
    """Local-first coding assistant for software and embedded projects.

    The first version intentionally separates reasoning from execution:
    - the model may inspect project files and retrieved memory,
    - build/test commands are bounded and allow-listed,
    - writes require explicit caller approval.
    """

    SYSTEM_PROMPT = """You are Jarvis, a local coding and embedded-systems assistant.
Focus on C, C++, Python, Linux, firmware, FPGA workflows, debugging, architecture,
and software engineering. Prefer evidence from the current repository and retrieved
memory over guesses. For embedded work, distinguish hardware facts, toolchain facts,
and code assumptions. Never claim a command succeeded unless tool output says it did.
When proposing a code change, explain the reason and expected verification step.
"""

    def __init__(
        self,
        project_root: str | Path,
        *,
        memory_manager: MemoryManager | None = None,
        model_router: ModelRouter | None = None,
        llm: OllamaLLM | None = None,
        approval_gate: ApprovalGate | None = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.memory_manager = memory_manager or MemoryManager()
        self.memory_retriever = MemoryRetriever(self.memory_manager)
        self.model_router = model_router or ModelRouter()
        self.llm = llm or OllamaLLM()
        self.approval_gate = approval_gate or ApprovalGate()
        self.tools = CodingTools(
            self.project_root,
            approval_gate=self.approval_gate,
        )

    def ask(
        self,
        prompt: str,
        *,
        category: str = "programming",
        include_project_index: bool = True,
    ) -> CodingAgentResult:
        decision = self.model_router.select_model(prompt, category=category)
        memory_context = self.memory_retriever.get_context(prompt, limit=5)

        project_context = ""
        if include_project_index:
            listing = self.tools.list_files(limit=120)
            if listing.ok and listing.output:
                project_context = (
                    "Repository file index:\n"
                    + listing.output
                )

        context_parts = [
            part
            for part in (memory_context, project_context)
            if part
        ]

        user_prompt = prompt
        if context_parts:
            user_prompt = (
                "\n\n".join(context_parts)
                + "\n\nUser task:\n"
                + prompt
            )

        response = self.llm.generate(
            prompt=user_prompt,
            model=decision.model,
            system=self.SYSTEM_PROMPT,
        )

        return CodingAgentResult(
            response=response,
            model=decision.model,
            reason=decision.reason,
            memory_context_used=bool(memory_context),
        )

    def inspect_file(self, relative_path: str) -> ToolResult:
        return self.tools.read_file(relative_path)

    def search_code(self, query: str) -> ToolResult:
        return self.tools.search_text(query)

    def run_build(self, command: list[str]) -> ToolResult:
        return self.tools.run_command(command)

    def propose_file_change(
        self,
        relative_path: str,
        request: str,
    ) -> CodingAgentResult:
        current = self.tools.read_file(relative_path)
        if not current.ok:
            return CodingAgentResult(
                response=current.output,
                model="none",
                reason="Target file could not be read.",
                memory_context_used=False,
            )

        prompt = f"""Modify the file below conceptually, but do not claim it has been written.
Return the complete proposed replacement file followed by a short verification plan.

Path: {relative_path}
Requested change: {request}

Current file:
---BEGIN FILE---
{current.output}
---END FILE---
"""
        return self.ask(
            prompt,
            category="programming",
            include_project_index=False,
        )

    def apply_file_change(
        self,
        relative_path: str,
        content: str,
        *,
        approved: bool = False,
    ) -> ToolResult:
        return self.tools.write_file(
            relative_path,
            content,
            approved=approved,
        )

    def remember_solution(self, note: str) -> None:
        self.memory_manager.save_confirmed_memory(
            note=note,
            category="software",
            importance=0.85,
            confidence=1.0,
            source="coding_agent_verified_solution",
        )

    def route_model(self, prompt: str, category: str = "programming") -> ModelDecision:
        return self.model_router.select_model(prompt, category)
