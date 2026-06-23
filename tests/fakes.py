"""
tests/fakes.py — FakeMemoryStore implementing the MemoryStore Protocol.

No mempill wheel required — pure in-memory Python. Used by test_domain_rules.py.
"""
from __future__ import annotations

from mempill_demo.domain.models import (
    AlternativeView,
    AuditEntry,
    BeliefView,
    ClaimMeta,
    ParsedCommand,
    ReconcileOutcome,
    CommandKind,
    _prov_abbr,
)


class FakeMemoryStore:
    """
    In-memory MemoryStore for unit tests.

    Behaviour:
    - ingest() stores claims; if a second Functional claim arrives for the same
      subject/predicate, both become "Contested".
    - recall() returns the current BeliefView (Contested if conflict, Committed otherwise).
    - reconcile() resolves by keeping the most recent claim; marks the other Superseded.
    - history() returns registry entries + fake audit.
    - audit() returns the fake audit ledger.
    - beliefs() returns all unique subject/predicate beliefs.
    - registry_snapshot() returns the claim registry.
    """

    def __init__(self) -> None:
        self._claims: list[ClaimMeta] = []
        self._audit: list[AuditEntry] = []
        self._counter = 0

    def _next_ref(self) -> str:
        self._counter += 1
        return f"fake-ref-{self._counter:04d}-" + "0" * 24

    def ingest(self, cmd: ParsedCommand) -> ClaimMeta:
        ref = self._next_ref()
        prov_str = _prov_abbr(
            "Recall" if cmd.kind == CommandKind.RECALL_REENTRY else "External"
        )
        conf = 0.7 if cmd.kind == CommandKind.RECALL_REENTRY else cmd.conf

        # Check for existing Functional claim on same subject/predicate
        existing = [
            c for c in self._claims
            if c.subject == cmd.subject
            and c.predicate == cmd.predicate
            and c.disposition not in ("Superseded", "Invalidated")
        ]

        if cmd.kind == CommandKind.RECALL_REENTRY:
            # Recall re-entry: never creates a conflict; disposition is Superseded
            disp = "Superseded"
        elif existing:
            # Mark all existing as Contested, new one too
            for e in existing:
                if e.disposition not in ("Contested",):
                    e.disposition = "Contested"
            disp = "Contested"
        else:
            disp = "CommittedCheap"

        meta = ClaimMeta(
            subject=cmd.subject,
            predicate=cmd.predicate,
            value=cmd.value,
            provenance=prov_str,
            valid_time={"start": cmd.since} if cmd.since else None,
            conf=conf,
            disposition=disp,
            claim_ref=ref,
        )
        self._claims.append(meta)
        self._audit.append(AuditEntry(
            claim_ref=ref,
            event_kind="ClaimIngested",
            disposition=disp,
            recorded_at="2024-01-01T00:00:00",
            rationale="",
        ))
        return meta

    def recall(self, subject: str, predicate: str) -> BeliefView:
        active = [
            c for c in self._claims
            if c.subject == subject
            and c.predicate == predicate
            and c.disposition not in ("Superseded", "Invalidated")
        ]
        if not active:
            return BeliefView(
                subject=subject, predicate=predicate, value=None,
                status="UNKNOWN", conf=None, vt_start="", vt_end="",
                provenance="", claim_ref="", corroboration=0, alternatives=[],
            )

        primary = active[-1]
        status = primary.disposition
        # Map disposition to standard status strings
        if status == "CommittedCheap":
            status = "Committed"
        elif status == "Superseded":
            status = "Superseded"

        alts = [
            AlternativeView(
                value=c.value,
                conf=c.conf,
                vt_start="",
                vt_end="open",
                claim_ref=c.claim_ref,
            )
            for c in active[:-1]
        ] if len(active) > 1 else []

        return BeliefView(
            subject=subject,
            predicate=predicate,
            value=primary.value,
            status=status,
            conf=primary.conf,
            vt_start="",
            vt_end="open",
            provenance=primary.provenance,
            claim_ref=primary.claim_ref,
            corroboration=0,
            alternatives=alts,
        )

    def reconcile(self, subject: str, predicate: str) -> list[ReconcileOutcome]:
        active = [
            c for c in self._claims
            if c.subject == subject
            and c.predicate == predicate
            and c.disposition not in ("Superseded", "Invalidated")
        ]
        if len(active) <= 1:
            return []
        # Keep the last one, supersede the rest
        winner = active[-1]
        winner.disposition = "CommittedCheap"
        outcomes = []
        for loser in active[:-1]:
            loser.disposition = "Superseded"
            self._audit.append(AuditEntry(
                claim_ref=loser.claim_ref,
                event_kind="ValidityAsserted",
                disposition="Superseded",
                recorded_at="2024-01-01T00:00:01",
                rationale="reconciled",
            ))
            outcomes.append(ReconcileOutcome(claim_ref=loser.claim_ref, disposition="Superseded"))
        outcomes.append(ReconcileOutcome(claim_ref=winner.claim_ref, disposition="CommittedCheap"))
        return outcomes

    def history(
        self, subject: str, predicate: str
    ) -> tuple[list[ClaimMeta], list[AuditEntry]]:
        metas = [c for c in self._claims if c.subject == subject and c.predicate == predicate]
        refs = {m.claim_ref for m in metas}
        entries = [e for e in self._audit if e.claim_ref in refs]
        return metas, entries

    def audit(self, limit: int) -> list[AuditEntry]:
        return self._audit[-limit:] if limit < len(self._audit) else list(self._audit)

    def beliefs(self) -> list[BeliefView]:
        seen: set[tuple[str, str]] = set()
        result = []
        for c in self._claims:
            key = (c.subject, c.predicate)
            if key in seen:
                continue
            seen.add(key)
            result.append(self.recall(c.subject, c.predicate))
        return result

    def registry_snapshot(self) -> dict[str, ClaimMeta]:
        return {c.claim_ref: c for c in self._claims}
