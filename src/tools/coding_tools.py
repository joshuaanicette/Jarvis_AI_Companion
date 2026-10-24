from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.agent.approval import ApprovalGate, RiskLevel


@dataclass(slots=True)
class ToolResult:
    ok: bool
    output: str
    returncode: int = 0


class CodingTools:
    """Bounded local tools for repository inspection and build/test workflows."""

    DEFAULT_ALLOWED_COMMANDS = {
        "git", "python", "python3", "pytest", "make", "cmake", "ctest",
        "gcc", "g++", "clang", "clang++", "ninja", "pio", "platformio",
    }

    def __init__(
        self,
        project_root: str | Path,
        approval_gate: ApprovalGate | None = None,
        timeout: int = 120,
    ) -> None:
        self.root = Path(project_root).resolve()
        self.approval_gate = approval_gate or ApprovalGate()
        self.timeout = timeout

    def _safe_path(self, relative_path: str | Path) -> Path:
        candidate = (self.root / relative_path).resolve()
        if self.root != candidate and self.root not in candidate.parents:
            raise ValueError("Path escapes the configured project root.")
        return candidate

    def read_file(self, relative_path: str, max_chars: int = 20000) -> ToolResult:
        path = self._safe_path(relative_path)
        if not path.is_file():
            return ToolResult(False, f"File not found: {relative_path}", 1)
        text = path.read_text(encoding="utf-8", errors="replace")
        return ToolResult(True, text[:max_chars])

    def list_files(self, pattern: str = "*", limit: int = 200) -> ToolResult:
        matches = []
        for path in self.root.rglob(pattern):
            if any(part in {".git", ".venv", "venv", "__pycache__", "node_modules"} for part in path.parts):
                continue
            if path.is_file():
                matches.append(str(path.relative_to(self.root)))
            if len(matches) >= limit:
                break
        return ToolResult(True, "\n".join(matches))

    def search_text(self, query: str, suffixes: Iterable[str] | None = None, limit: int = 100) -> ToolResult:
        suffix_set = set(suffixes or [".c", ".cc", ".cpp", ".h", ".hpp", ".py", ".md", ".txt", ".json", ".yaml", ".yml"])
        hits = []
        lowered = query.casefold()
        for path in self.root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in suffix_set:
                continue
            if any(part in {".git", ".venv", "venv", "__pycache__", "node_modules"} for part in path.parts):
                continue
            try:
                for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                    if lowered in line.casefold():
                        hits.append(f"{path.relative_to(self.root)}:{number}: {line.strip()}")
                        if len(hits) >= limit:
                            return ToolResult(True, "\n".join(hits))
            except OSError:
                continue
        return ToolResult(True, "\n".join(hits))

    def run_command(self, command: list[str], risk: RiskLevel = RiskLevel.BUILD, approved: bool = False) -> ToolResult:
        if not command:
            return ToolResult(False, "Empty command.", 2)
        if Path(command[0]).name not in self.DEFAULT_ALLOWED_COMMANDS:
            return ToolResult(False, f"Command not allow-listed: {command[0]}", 126)
        decision = self.approval_gate.check(risk=risk, approved=approved)
        if not decision.allowed:
            return ToolResult(False, decision.reason, 126)
        try:
            completed = subprocess.run(
                command,
                cwd=self.root,
                text=True,
                capture_output=True,
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            return ToolResult(False, f"Command timed out after {self.timeout}s: {error}", 124)
        output = (completed.stdout + "\n" + completed.stderr).strip()
        return ToolResult(completed.returncode == 0, output, completed.returncode)

    def write_file(self, relative_path: str, content: str, approved: bool = False) -> ToolResult:
        decision = self.approval_gate.check(RiskLevel.WRITE, approved=approved)
        if not decision.allowed:
            return ToolResult(False, decision.reason, 126)
        path = self._safe_path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return ToolResult(True, f"Wrote {relative_path}")
