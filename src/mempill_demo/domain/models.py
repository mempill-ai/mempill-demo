"""
mempill_demo.domain.models — pure domain types for the mempill console demo.

No imports of mempill, rich, or anthropic allowed in this module.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Optional


# ── Command kinds ─────────────────────────────────────────────────────────────

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


# ── Parsed command ────────────────────────────────────────────────────────────

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
    extra: dict[str, Any] = field(default_factory=dict)


# ── Memory domain types ───────────────────────────────────────────────────────

@dataclass
class AlternativeView:
    value: object
    conf: Optional[float]
    vt_start: str
    vt_end: str
    claim_ref: str


@dataclass
class BeliefView:
    subject: str
    predicate: str
    value: object
    status: str               # "Committed", "Contested", "Superseded", etc.
    conf: Optional[float]
    vt_start: str
    vt_end: str
    provenance: str           # abbreviated: USER/EXT/RECALL/LLM
    claim_ref: str
    corroboration: int
    alternatives: list[AlternativeView]


@dataclass
class ClaimMeta:
    """Metadata recorded at ingest time — audit entries lack these fields."""
    subject: str
    predicate: str
    value: str
    provenance: Any           # raw provenance object (adapter-held); abbrev via _prov_abbr
    valid_time: Optional[dict]
    conf: float
    disposition: str
    claim_ref: str


@dataclass
class ReconcileOutcome:
    claim_ref: str
    disposition: str


@dataclass
class AuditEntry:
    claim_ref: str
    event_kind: str
    disposition: str
    recorded_at: str
    rationale: str


@dataclass
class SessionStats:
    n_ingests: int = 0
    n_queries: int = 0
    n_contested: int = 0
    n_superseded: int = 0


@dataclass
class AgentResponse:
    text: str
    kind: CommandKind
    n_ingests_delta: int = 0
    n_contested_delta: int = 0
    n_superseded_delta: int = 0


# ── Utility ───────────────────────────────────────────────────────────────────

def _prov_abbr(prov: Any) -> str:
    """Convert a raw provenance object to a short label: USER/EXT/RECALL/LLM."""
    if isinstance(prov, dict):
        t = prov.get("type", "")
        k = prov.get("kind", "")
        if t == "External":
            return "USER" if k == "UserAsserted" else "EXT"
        if t == "RecallReEntry":
            return "RECALL"
        if t == "ModelDerived":
            return "LLM"
    if isinstance(prov, str):
        if "UserAsserted" in prov:
            return "USER"
        if "ExternalFirstHand" in prov or "External" in prov:
            return "EXT"
        if "RecallReEntry" in prov or "Recall" in prov:
            return "RECALL"
        if "ModelDerived" in prov or "Model" in prov:
            return "LLM"
    return str(prov)[:6] if prov else ""


def _conf_str(c: Any) -> str:
    if c is None:
        return "N/A"
    try:
        return f"{float(c):.2f}"
    except (TypeError, ValueError):
        return str(c)
