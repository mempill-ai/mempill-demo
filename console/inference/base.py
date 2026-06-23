"""console.inference.base — shared types for parsed commands."""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Optional


class CommandKind(str, enum.Enum):
    INGEST = "INGEST"
    RECALL = "RECALL"
    RECALL_REENTRY = "RECALL_REENTRY"
    # Inspector slash-commands
    MEMORY = "MEMORY"
    HISTORY = "HISTORY"
    WHY = "WHY"
    CONTESTED = "CONTESTED"
    AUDIT = "AUDIT"
    RECONCILE = "RECONCILE"
    HELP = "HELP"
    QUIT = "QUIT"
    RESET = "RESET"
    UNKNOWN = "UNKNOWN"


@dataclass
class ParsedCommand:
    kind: CommandKind
    subject: Optional[str] = None
    predicate: Optional[str] = None
    value: Optional[str] = None
    since: Optional[str] = None
    until: Optional[str] = None
    conf: float = 0.9
    source_claim_ref: Optional[str] = None
    audit_limit: int = 10
    raw: str = ""
    error: Optional[str] = None
    # Extra fields populated by LLM path
    extra: dict[str, Any] = field(default_factory=dict)
