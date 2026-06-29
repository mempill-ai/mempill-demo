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
    # Point-in-time recall modifiers (bi-temporal axes)
    valid_at: Optional[str] = None        # ISO-8601 UTC — valid-time axis
    as_of_tx_time: Optional[str] = None   # ISO-8601 UTC — transaction-time axis


# ── Memory domain types ───────────────────────────────────────────────────────

@dataclass
class AlternativeView:
    value: object
    conf: Optional[float]
    vt_start: str
    vt_end: str
    claim_ref: str
    # Honest granularity display strings (None = use raw vt_start/vt_end)
    vt_start_display: Optional[str] = None
    vt_end_display: Optional[str] = None


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
    # Honest granularity display strings (None = use raw vt_start/vt_end)
    vt_start_display: Optional[str] = None
    vt_end_display: Optional[str] = None


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
class TimelineEntry:
    """A single entry in the ordered belief timeline for a (subject, predicate) pair."""
    value: str
    valid_from: Optional[str]    # RFC3339 or None
    valid_until: Optional[str]   # RFC3339 or None ("open" when None)
    status: str                  # "Current" | "Superseded"
    claim_ref: str


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


def _granularity_display(raw_date: Optional[str], granularity: Optional[str]) -> Optional[str]:
    """Render a date string at the given granularity level.

    Granularity values: "year" → YYYY, "month" → YYYY-MM, "day"/"instant" → YYYY-MM-DD.
    Falls back to the raw date string (first 10 chars) if granularity is None or unknown.
    Returns None if raw_date is None.
    """
    if raw_date is None:
        return None
    # Take the date portion (first 10 chars) as base
    date_part = raw_date[:10] if len(raw_date) >= 10 else raw_date
    if granularity == "year" and len(date_part) >= 4:
        return date_part[:4]
    if granularity == "month" and len(date_part) >= 7:
        return date_part[:7]
    # "day", "instant", or unknown — use full date
    return date_part


def _conf_str(c: Any) -> str:
    if c is None:
        return "N/A"
    try:
        return f"{float(c):.2f}"
    except (TypeError, ValueError):
        return str(c)
