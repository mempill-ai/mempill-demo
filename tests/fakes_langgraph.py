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
)


class LGFakeMemoryStore:
    """
    Minimal MemoryStore implementation for LangGraph integration tests.

    - recall() returns the pre-configured belief (or an UNKNOWN belief if none set).
    - ingest() records commands in self.ingested without side-effects.
    - All other Protocol methods return safe empty values.
    - recall_calls tracks how many times recall() was invoked.
    """

    def __init__(self, belief: Optional[BeliefView] = None) -> None:
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
        self.ingested: list[ParsedCommand] = []
        self.recall_calls: int = 0

    def recall(self, subject: str, predicate: str) -> BeliefView:
        self.recall_calls += 1
        # Return the pre-configured belief with the queried subject/predicate
        return BeliefView(
            subject=subject,
            predicate=predicate,
            value=self._belief.value,
            status=self._belief.status,
            conf=self._belief.conf,
            vt_start=self._belief.vt_start,
            vt_end=self._belief.vt_end,
            provenance=self._belief.provenance,
            claim_ref=self._belief.claim_ref,
            corroboration=self._belief.corroboration,
            alternatives=self._belief.alternatives,
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
        return []

    def history(self, subject: str, predicate: str) -> tuple[list[ClaimMeta], list[AuditEntry]]:
        return [], []

    def audit(self, limit: int) -> list[AuditEntry]:
        return []

    def beliefs(self) -> list[BeliefView]:
        return []

    def registry_snapshot(self) -> dict[str, ClaimMeta]:
        return {}
