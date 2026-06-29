"""
mempill_showcase.core.ports.memory — MemoryStore Protocol + BiTemporalMemoryStore sub-protocol.

No mempill imports here. Only domain types from core.domain.models.

MemoryStore:
  - write_claim(agent_id, claim) → WriteReceipt
  - recall(agent_id, subject, predicate) → BeliefView
  - audit(agent_id, limit) → list[AuditEntry]

BiTemporalMemoryStore(MemoryStore):
  - query_at(agent_id, subject, predicate, valid_at, as_of_tx_time) → BeliefView
    (bi-temporal point-in-time query — only mempill adapter implements this)
"""
from __future__ import annotations

from typing import Optional, Protocol

from mempill_showcase.core.domain.models import AuditEntry, BeliefView, ClaimInput, WriteReceipt


class MemoryStore(Protocol):
    """Base port: write claims + current recall + audit log."""

    def write_claim(self, agent_id: str, claim: ClaimInput) -> WriteReceipt:
        """Write a claim with provenance and world-time valid_from/until.

        Never re-ingest duplicates (adapter responsibility).
        Never default valid_from to now — leave None if not supplied.
        """
        ...

    def recall(self, agent_id: str, subject: str, predicate: str) -> BeliefView:
        """Return the current belief for (agent_id, subject, predicate).

        Returns a BeliefView with status=NoBelief if no claim exists.
        Callers must check .is_contested() before trusting .value.
        """
        ...

    def audit(self, agent_id: str, limit: int = 50) -> list[AuditEntry]:
        """Return the most recent *limit* audit entries for agent_id."""
        ...


class BiTemporalMemoryStore(MemoryStore, Protocol):
    """Sub-protocol for bi-temporal stores (only the mempill adapter implements this).

    The naive adapter implements MemoryStore only — it intentionally does NOT
    implement BiTemporalMemoryStore, proving the contrast seam between the two.
    """

    def query_at(
        self,
        agent_id: str,
        subject: str,
        predicate: str,
        valid_at: Optional[str] = None,
        as_of_tx_time: Optional[str] = None,
    ) -> BeliefView:
        """Bi-temporal point-in-time query on two independent axes.

        valid_at:       ISO-8601 UTC — "what was true in the world at this instant?"
                        None = current valid-time (open-ended / most recent).
        as_of_tx_time:  ISO-8601 UTC — "what did the system know at this tx instant?"
                        None = latest tx-time (default, i.e. use all ingested claims).

        Both axes are independent and can be combined for full bi-temporal semantics.
        Returns BeliefView with status=NoBelief if no claim matches the constraints.
        """
        ...
