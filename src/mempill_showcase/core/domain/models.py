"""
mempill_showcase.core.domain.models — pure domain types.

No imports of mempill, rich, anthropic, or any framework allowed here.
Lifted and trimmed from mempill_demo.domain.models; only the types needed
by the showcase core are retained.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Optional


# ── Claim write result ────────────────────────────────────────────────────────

@dataclass
class WriteReceipt:
    """Result returned by MemoryStore.write_claim()."""
    claim_ref: str
    disposition: str          # Disposition variant string (CommittedCheap, Contested, etc.)
    contested_with: list[str] = field(default_factory=list)

    def is_contested(self) -> bool:
        return self.disposition in ("Contested", "Conflict", "QueuedForAdjudication")


# ── Recall / query result ─────────────────────────────────────────────────────

@dataclass
class AlternativeView:
    """A competing belief candidate (used when Contested/Conflict)."""
    value: object
    conf: Optional[float]
    vt_start: str
    vt_end: str
    claim_ref: str
    vt_start_display: Optional[str] = None
    vt_end_display: Optional[str] = None


@dataclass
class BeliefView:
    """A resolved (or contested) belief for a subject/predicate pair."""
    subject: str
    predicate: str
    value: object                # None when Contested/NoBelief
    status: str                  # "Resolved" | "Contested" | "Conflict" | "TimingUncertain" | "NoBelief"
    conf: Optional[float]
    vt_start: str
    vt_end: str
    provenance: str              # abbreviated: USER/EXT/RECALL/LLM
    claim_ref: str
    corroboration: int
    alternatives: list[AlternativeView]
    vt_start_display: Optional[str] = None
    vt_end_display: Optional[str] = None

    def is_contested(self) -> bool:
        return self.status in ("Contested", "Conflict")

    def is_resolved(self) -> bool:
        return self.status == "Resolved"


# ── Audit ─────────────────────────────────────────────────────────────────────

@dataclass
class AuditEntry:
    claim_ref: str
    event_kind: str
    disposition: str
    recorded_at: str
    rationale: str


# ── Claim input (write path) ──────────────────────────────────────────────────

@dataclass
class ClaimInput:
    """
    Caller-constructed claim for writing to a MemoryStore.

    valid_from:  partial date string (YYYY / YYYY-MM / YYYY-MM-DD / RFC3339)
                 or None (unknown start).
    valid_until: same formats or None (open-ended).
    provenance:  raw provenance dict from ProvenanceLabel factory, or None
                 (adapter defaults to external_user_asserted).
    """
    subject: str
    predicate: str
    value: Any
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None
    confidence: float = 1.0
    provenance: Optional[dict] = None
    cardinality: str = "Functional"
    criticality: str = "Medium"
    derived_from: list[str] = field(default_factory=list)


# ── Utility helpers ───────────────────────────────────────────────────────────

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
