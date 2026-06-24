"""mempill_demo.ports.memory — MemoryStore Protocol + OracleMemoryStore sub-protocol."""
from __future__ import annotations

from typing import Protocol

from mempill_demo.domain.models import (
    AuditEntry,
    BeliefView,
    ClaimMeta,
    ParsedCommand,
    ReconcileOutcome,
)


class MemoryStore(Protocol):
    # Write path
    def ingest(self, cmd: ParsedCommand) -> ClaimMeta: ...

    # Read paths (return domain types, not raw dicts)
    def recall(self, subject: str, predicate: str) -> BeliefView: ...
    def reconcile(self, subject: str, predicate: str) -> list[ReconcileOutcome]: ...
    def history(self, subject: str, predicate: str) -> tuple[list[ClaimMeta], list[AuditEntry]]: ...
    def audit(self, limit: int) -> list[AuditEntry]: ...
    def beliefs(self) -> list[BeliefView]: ...
    def registry_snapshot(self) -> dict[str, ClaimMeta]: ...


class OracleMemoryStore(MemoryStore, Protocol):
    """
    Sub-protocol for oracle-wired stores (decision H.3).

    Extends MemoryStore with the two oracle-specific operations needed by /review.
    The base MemoryStore remains portable (no oracle dependency).
    """

    def list_pending(self) -> list[dict]: ...
    """Return pending adjudication requests for this store's agent_id."""

    def submit(self, handle_id: str, verdict: str) -> dict: ...
    """
    Submit a verdict for a pending adjudication.

    verdict: "Affirm" (challenger wins) | "Deny" (incumbent wins) | "Unknown" (stays Contested)
    Returns the raw engine submit_adjudication response dict.
    """
