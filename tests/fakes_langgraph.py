"""
tests/fakes_langgraph.py — LangGraph-specific test fakes.

LGFakeMemoryStore is a MemoryStore Protocol implementation for LangGraph tests.
It tracks recall_calls and ingested commands for assertion.

Intentionally separate from tests/fakes.py to avoid modifying existing test helpers.
"""
from __future__ import annotations

from typing import Optional

from mempill_demo.domain.models import (
    AuditEntry,
    BeliefView,
    ClaimMeta,
    ParsedCommand,
    ReconcileOutcome,
    TimelineEntry,
)


class LGFakeMemoryStore:
    """
    Minimal MemoryStore + OracleMemoryStore implementation for LangGraph tests.

    - recall() returns the pre-configured belief (or an UNKNOWN belief if none set).
    - reconcile() can be configured to return a post-reconcile belief via
      reconcile_result_belief (simulates valid-time auto-resolution).
    - ingest() records commands in self.ingested without side-effects.
    - list_pending() returns self.pending_items (configurable for adjudication tests).
    - submit() records verdicts in self.submitted_verdicts.
    - All other Protocol methods return safe empty values.
    - recall_calls tracks how many times recall() was invoked.
    - reconcile_calls tracks how many times reconcile() was invoked.
    """

    def __init__(
        self,
        belief: Optional[BeliefView] = None,
        reconcile_result_belief: Optional[BeliefView] = None,
        pending_items: Optional[list[dict]] = None,
        timeline_entries: Optional[list[TimelineEntry]] = None,
    ) -> None:
        self._belief = belief or BeliefView(
            subject="",
            predicate="",
            value=None,
            status="UNKNOWN",
            conf=None,
            vt_start="",
            vt_end="",
            provenance="",
            claim_ref="",
            corroboration=0,
            alternatives=[],
        )
        # If set, recall() returns this belief AFTER reconcile() has been called once.
        # Simulates valid-time resolution (Contested → Resolved/CommittedCheap).
        self._reconcile_result_belief = reconcile_result_belief
        self._pending_items: list[dict] = list(pending_items or [])
        self._timeline_entries: list[TimelineEntry] = list(timeline_entries or [])
        self.ingested: list[ParsedCommand] = []
        self.recall_calls: int = 0
        self.reconcile_calls: int = 0
        self.submitted_verdicts: list[tuple[str, str]] = []  # (handle_id, verdict)

    def recall(self, subject: str, predicate: str) -> BeliefView:
        self.recall_calls += 1
        # After reconcile() has been called, switch to the reconcile_result_belief.
        source = (
            self._reconcile_result_belief
            if (self._reconcile_result_belief and self.reconcile_calls > 0)
            else self._belief
        )
        return BeliefView(
            subject=subject,
            predicate=predicate,
            value=source.value,
            status=source.status,
            conf=source.conf,
            vt_start=source.vt_start,
            vt_end=source.vt_end,
            provenance=source.provenance,
            claim_ref=source.claim_ref,
            corroboration=source.corroboration,
            alternatives=source.alternatives,
        )

    def ingest(self, cmd: ParsedCommand) -> ClaimMeta:
        self.ingested.append(cmd)
        return ClaimMeta(
            subject=cmd.subject or "",
            predicate=cmd.predicate or "",
            value=cmd.value or "",
            provenance="fake",
            valid_time=None,
            conf=cmd.conf,
            disposition="CommittedCheap",
            claim_ref="fake-ref-lg",
        )

    def reconcile(self, subject: str, predicate: str) -> list[ReconcileOutcome]:
        self.reconcile_calls += 1
        return []

    def list_pending(self) -> list[dict]:
        return list(self._pending_items)

    def submit(self, handle_id: str, verdict: str) -> dict:
        self.submitted_verdicts.append((handle_id, verdict))
        # Remove from pending list to simulate successful adjudication
        self._pending_items = [
            p for p in self._pending_items if p.get("handle_id") != handle_id
        ]
        return {"disposition": "Committed", "claim_ref": "fake-adj-ref"}

    def timeline_history(self, subject: str, predicate: str) -> list[TimelineEntry]:
        return list(self._timeline_entries)

    def history(self, subject: str, predicate: str) -> tuple[list[ClaimMeta], list[AuditEntry]]:
        return [], []

    def audit(self, limit: int) -> list[AuditEntry]:
        return []

    def beliefs(self) -> list[BeliefView]:
        return []

    def registry_snapshot(self) -> dict[str, ClaimMeta]:
        return {}
