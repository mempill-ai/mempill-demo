"""mempill_demo.ports.memory — MemoryStore Protocol."""
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
