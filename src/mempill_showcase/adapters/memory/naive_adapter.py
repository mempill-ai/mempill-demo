"""
mempill_showcase.adapters.memory.naive_adapter — last-write-wins in-memory store.

Implements MemoryStore ONLY (intentionally NOT BiTemporalMemoryStore).

This adapter demonstrates the contrast with the mempill adapter:
  - No valid-time semantics (all writes overwrite the current value)
  - No provenance tracking (only the latest value is kept)
  - No Contested disposition (overwrites silently)
  - No bi-temporal queries (query_at / as_of_tx_time unavailable)
  - Recency recall: always returns the most recently written value

The absence of query_at is the architectural proof that bi-temporal capability
is NOT available here — callers must isinstance-check for BiTemporalMemoryStore
if they need it, or simply try/except AttributeError on query_at.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from mempill_showcase.core.domain.models import (
    AuditEntry,
    BeliefView,
    ClaimInput,
    WriteReceipt,
)


@dataclass
class _StoredEntry:
    value: Any
    # Minimal metadata kept for recency ordering
    write_order: int


class NaiveAdapter:
    """Last-write-wins in-memory MemoryStore (no valid-time, no provenance, no Contested).

    Thread safety: not guaranteed. Demo use only.
    """

    def __init__(self) -> None:
        # {agent_id: {(subject, predicate): _StoredEntry}}
        self._store: dict[str, dict[tuple[str, str], _StoredEntry]] = {}
        self._write_counter: int = 0
        # Minimal audit log (write events only, no supersession/oracle events)
        self._audit_log: list[dict] = []

    def _agent_store(self, agent_id: str) -> dict[tuple[str, str], _StoredEntry]:
        return self._store.setdefault(agent_id, {})

    # ── Write path ────────────────────────────────────────────────────────────

    def write_claim(self, agent_id: str, claim: ClaimInput) -> WriteReceipt:
        """Last-write-wins: silently overwrites any prior value.

        Disposition is always "CommittedCheap" — no conflict detection,
        no Contested, no provenance firewall.
        """
        self._write_counter += 1
        key = (claim.subject, claim.predicate)
        self._agent_store(agent_id)[key] = _StoredEntry(
            value=claim.value,
            write_order=self._write_counter,
        )
        # Fake claim_ref so callers that store it don't crash
        import uuid
        ref = str(uuid.uuid4())

        self._audit_log.append({
            "agent_id": agent_id,
            "claim_ref": ref,
            "subject": claim.subject,
            "predicate": claim.predicate,
            "value": claim.value,
            "event_kind": "Ingested",
            "disposition": "CommittedCheap",
            "recorded_at": "",   # no real tx-time
            "rationale": "naive last-write-wins",
        })

        return WriteReceipt(
            claim_ref=ref,
            disposition="CommittedCheap",
            contested_with=[],
        )

    # ── Read path — current belief ────────────────────────────────────────────

    def recall(self, agent_id: str, subject: str, predicate: str) -> BeliefView:
        """Return the most recently written value, or NoBelief if absent."""
        entry = self._agent_store(agent_id).get((subject, predicate))
        if entry is None:
            return BeliefView(
                subject=subject,
                predicate=predicate,
                value=None,
                status="NoBelief",
                conf=None,
                vt_start="",
                vt_end="",
                provenance="",
                claim_ref="",
                corroboration=0,
                alternatives=[],
            )
        return BeliefView(
            subject=subject,
            predicate=predicate,
            value=entry.value,
            status="Resolved",
            conf=None,
            vt_start="",
            vt_end="",
            provenance="",
            claim_ref="",
            corroboration=0,
            alternatives=[],
        )

    # ── Audit ─────────────────────────────────────────────────────────────────

    def audit(self, agent_id: str, limit: int = 50) -> list[AuditEntry]:
        """Return recent audit entries (write events only — no bi-temporal metadata)."""
        relevant = [e for e in self._audit_log if e["agent_id"] == agent_id]
        return [
            AuditEntry(
                claim_ref=e["claim_ref"],
                event_kind=e["event_kind"],
                disposition=e["disposition"],
                recorded_at=e["recorded_at"],
                rationale=e["rationale"],
            )
            for e in relevant[-limit:]
        ]

    # NOTE: query_at() is intentionally NOT implemented here.
    # NaiveAdapter is NOT a BiTemporalMemoryStore.
    # isinstance(adapter, BiTemporalMemoryStore) → False
    # adapter.query_at(...) → AttributeError
    # This is the designed contrast seam.
