from __future__ import annotations

import difflib
import json
import os
import re
import secrets
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


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
    """Stages code edits and applies them only after explicit user approval."""

    BLOCKED_PARTS = {".git", ".env", "__pycache__", ".ssh", "secrets"}
    ALLOWED_SUFFIXES = {
        ".py", ".js", ".ts", ".tsx", ".jsx", ".json", ".yaml", ".yml",
        ".toml", ".md", ".html", ".css", ".sh", ".txt",
    }

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

    def read_file(self, relative_path: str) -> str:
        path = self._resolve(relative_path)
        return path.read_text(encoding="utf-8")

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
        sections.append(
            f'To apply, explicitly say: approve coding proposal {proposal.proposal_id}'
        )
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
            raise ApprovalRequiredError(
                f"Proposal {proposal_id} has not received explicit approval."
            )

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
