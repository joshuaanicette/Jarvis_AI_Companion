from __future__ import annotations

import difflib
import json
import os
import re
import secrets
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(slots=True)
class FileChange:
    path: str
    original: str
    replacement: str


@dataclass(slots=True)
class ChangeProposal:
    proposal_id: str
    summary: str
    changes: list[FileChange]
    approved: bool = False
    applied: bool = False


class ApprovalRequiredError(RuntimeError):
    pass


class CodingAgent:
    """Inspects code, recommends changes, and applies only approved proposals."""

    BLOCKED_PARTS = {".git", ".env", "__pycache__", ".ssh", "secrets"}
    ALLOWED_SUFFIXES = {
        ".py", ".js", ".ts", ".tsx", ".jsx", ".json", ".yaml", ".yml",
        ".toml", ".md", ".html", ".css", ".sh", ".txt",
    }
    MAX_FILE_CHARS = 30_000

    def __init__(
        self,
        workspace: str | Path = ".",
        proposal_path: str | Path = "data/coding/proposals.json",
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.proposal_path = Path(proposal_path)
        self.proposal_path.parent.mkdir(parents=True, exist_ok=True)
        self._proposals: dict[str, ChangeProposal] = {}
        self._load()

    def _resolve(self, relative_path: str) -> Path:
        candidate = (self.workspace / relative_path).resolve()
        try:
            candidate.relative_to(self.workspace)
        except ValueError as error:
            raise ValueError("Path escapes the coding workspace.") from error
        if any(part in self.BLOCKED_PARTS for part in candidate.parts):
            raise ValueError("That path is protected.")
        if candidate.suffix and candidate.suffix.casefold() not in self.ALLOWED_SUFFIXES:
            raise ValueError(f"Unsupported file type: {candidate.suffix}")
        return candidate

    @classmethod
    def _paths_from_text(cls, text: str) -> set[str]:
        return set(
            re.findall(
                r"(?:^|\s)([A-Za-z0-9_./-]+\.(?:py|js|ts|tsx|jsx|json|ya?ml|toml|md|html|css|sh|txt))",
                str(text),
            )
        )

    def read_file(self, relative_path: str) -> str:
        return self._resolve(relative_path).read_text(encoding="utf-8")

    def list_files(self, limit: int = 200) -> list[str]:
        files: list[str] = []
        for path in self.workspace.rglob("*"):
            if not path.is_file() or any(part in self.BLOCKED_PARTS for part in path.parts):
                continue
            if path.suffix.casefold() in self.ALLOWED_SUFFIXES:
                files.append(str(path.relative_to(self.workspace)))
            if len(files) >= limit:
                break
        return sorted(files)

    def _sources_for(self, paths: set[str], allow_missing: bool) -> list[str]:
        sources: list[str] = []
        for relative_path in sorted(paths):
            path = self._resolve(relative_path)
            if not path.exists() and not allow_missing:
                raise ValueError(f"File not found: {relative_path}")
            content = path.read_text(encoding="utf-8") if path.exists() else ""
            if len(content) > self.MAX_FILE_CHARS:
                raise ValueError(f"{relative_path} is too large for a safe local request.")
            sources.append(f"FILE: {relative_path}\n\`\`\`\n{content}\n\`\`\`")
        return sources

    @staticmethod
    def _generate(llm, prompt: str, model: str) -> str:
        return str(llm.generate(prompt=prompt, model=model)).strip()

    def analyze_from_prompt(self, request: str, llm, model: str = "qwen2.5:3b") -> str:
        """Explain named files and relate them to the requested functionality."""
        paths = self._paths_from_text(request)
        if not paths:
            raise ValueError("Name the file or files Jarvis should analyze.")
        sources = self._sources_for(paths, allow_missing=False)
        prompt = f"""
You are Jarvis's local code analyst. Explain the supplied code accurately.
Do not claim that code was changed or tests were run.

The user wants this functionality:
{request}

For each file, explain its role, main control flow, inputs/outputs, dependencies,
risks or likely bugs, and whether it supports the requested functionality.
Then recommend the smallest practical next implementation steps. Be specific
about which files should change or be added.

SOURCE FILES:
{chr(10).join(sources)}
""".strip()
        return self._generate(llm, prompt, model)

    def suggest_from_prompt(self, request: str, llm, model: str = "qwen2.5:3b") -> str:
        """Suggest a feature design without reading or altering source contents."""
        project_index = "\n".join(f"- {path}" for path in self.list_files(limit=160))
        prompt = f"""
You are Jarvis's local software architect. The user wants this functionality:
{request}

Based only on this project file index, recommend a practical feature design.
State: likely files to inspect first, new files that may be useful, data flow,
tools or dependencies, security/safety considerations, and an incremental
implementation plan. If the existing index is insufficient, name the exact
files the user should ask Jarvis to analyze next. Do not claim that you read
source code, changed files, or ran tests.

PROJECT FILE INDEX:
{project_index}
""".strip()
        return self._generate(llm, prompt, model)

    def propose(self, summary: str, replacements: dict[str, str]) -> ChangeProposal:
        changes: list[FileChange] = []
        for relative_path, replacement in replacements.items():
            path = self._resolve(relative_path)
            original = path.read_text(encoding="utf-8") if path.exists() else ""
            if original != replacement:
                changes.append(FileChange(relative_path, original, replacement))
        if not changes:
            raise ValueError("The proposal does not change any files.")
        proposal = ChangeProposal(
            proposal_id=secrets.token_hex(4),
            summary=str(summary).strip() or "Proposed code changes",
            changes=changes,
        )
        self._proposals[proposal.proposal_id] = proposal
        self._save()
        return proposal

    def propose_from_prompt(self, task: str, llm, model: str = "qwen2.5:3b") -> ChangeProposal:
        """Generate a complete-file proposal; existing files remain unchanged until approved."""
        mentioned = self._paths_from_text(task)
        if not mentioned:
            raise ValueError("Name each project file that may be created or changed.")
        sources = self._sources_for(mentioned, allow_missing=True)
        prompt = f"""
You are Jarvis's local coding planner. Prepare a proposed change but do not
claim files were edited or tests were run. Return JSON only:
{{
  "summary": "short description",
  "changes": [
    {{"path": "relative/project/file.py", "content": "complete replacement file"}}
  ]
}}
Every changed file must be explicitly included below. A missing file may be
created. Preserve unrelated behavior and return complete replacement content,
never partial snippets.

TASK:
{task}

CURRENT FILES:
{chr(10).join(sources)}
""".strip()
        raw = self._generate(llm, prompt, model)
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise ValueError("The coding model did not return a JSON proposal.")
        payload = json.loads(match.group(0))
        replacements: dict[str, str] = {}
        for item in payload.get("changes", []):
            path = str(item.get("path", "")).strip()
            if path not in mentioned:
                raise ValueError(f"The model attempted an unrequested file: {path}")
            replacements[path] = str(item.get("content", ""))
        return self.propose(str(payload.get("summary", task)), replacements)

    def propose_new_file(self, path: str, description: str, llm, model: str = "qwen2.5:3b") -> ChangeProposal:
        destination = self._resolve(path)
        if destination.exists():
            raise ValueError(f"{path} already exists; use a coding task to modify it.")
        task = f"Create new file {path}. Required functionality: {description}"
        return self.propose_from_prompt(task, llm, model)

    def preview(self, proposal_id: str) -> str:
        proposal = self._require(proposal_id)
        sections = [f"Proposal {proposal.proposal_id}: {proposal.summary}"]
        for change in proposal.changes:
            diff = difflib.unified_diff(
                change.original.splitlines(),
                change.replacement.splitlines(),
                fromfile=f"a/{change.path}",
                tofile=f"b/{change.path}",
                lineterm="",
            )
            sections.append("\n".join(diff))
        sections.append(f"To apply, explicitly say: approve coding proposal {proposal.proposal_id}")
        return "\n\n".join(sections)

    def approve_from_text(self, text: str) -> str | None:
        match = re.fullmatch(
            r"\s*approve coding proposal\s+([a-f0-9]{8})\s*",
            str(text),
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        proposal_id = match.group(1).casefold()
        self.apply(proposal_id, approved=True)
        return proposal_id

    def apply(self, proposal_id: str, approved: bool = False) -> None:
        proposal = self._require(proposal_id)
        if proposal.applied:
            return
        if not approved:
            raise ApprovalRequiredError(f"Proposal {proposal_id} has not received explicit approval.")

        staged: list[tuple[Path, Path]] = []
        try:
            for change in proposal.changes:
                destination = self._resolve(change.path)
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_name(destination.name + f".{proposal_id}.tmp")
                temporary.write_text(change.replacement, encoding="utf-8")
                staged.append((temporary, destination))
            for temporary, destination in staged:
                os.replace(temporary, destination)
        finally:
            for temporary, _ in staged:
                if temporary.exists():
                    temporary.unlink()

        proposal.approved = True
        proposal.applied = True
        self._save()

    def _require(self, proposal_id: str) -> ChangeProposal:
        proposal = self._proposals.get(str(proposal_id).casefold())
        if proposal is None:
            raise KeyError(f"Unknown coding proposal: {proposal_id}")
        return proposal

    def _save(self) -> None:
        payload = {key: asdict(value) for key, value in self._proposals.items()}
        temporary = self.proposal_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(self.proposal_path)

    def _load(self) -> None:
        if not self.proposal_path.exists():
            return
        try:
            payload = json.loads(self.proposal_path.read_text(encoding="utf-8"))
            for key, item in payload.items():
                changes = [FileChange(**change) for change in item.get("changes", [])]
                self._proposals[key] = ChangeProposal(
                    proposal_id=item["proposal_id"],
                    summary=item["summary"],
                    changes=changes,
                    approved=bool(item.get("approved", False)),
                    applied=bool(item.get("applied", False)),
                )
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            self._proposals = {}
