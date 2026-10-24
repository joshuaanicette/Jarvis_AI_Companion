from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RiskLevel(str, Enum):
    READ_ONLY = "read_only"
    BUILD = "build"
    WRITE = "write"
    DESTRUCTIVE = "destructive"


@dataclass(slots=True)
class ApprovalDecision:
    allowed: bool
    reason: str
    risk: RiskLevel


class ApprovalGate:
    """Central policy for agent actions.

    Read-only inspection and bounded build/test commands may run automatically.
    Any file mutation, Git mutation, package install, flash, or destructive command
    requires explicit approval from the caller.
    """

    AUTO_ALLOWED = {RiskLevel.READ_ONLY, RiskLevel.BUILD}

    def check(self, risk: RiskLevel, approved: bool = False) -> ApprovalDecision:
        if risk in self.AUTO_ALLOWED:
            return ApprovalDecision(True, "Action is non-mutating or bounded build/test work.", risk)
        if approved:
            return ApprovalDecision(True, "Explicit user approval supplied.", risk)
        return ApprovalDecision(
            False,
            "This action changes files, Git state, packages, hardware, or system state and requires explicit approval.",
            risk,
        )
