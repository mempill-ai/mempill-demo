"""
mempill_showcase.adapters.memory.mempill_adapter — THE ONLY module that imports mempill.

Implements BiTemporalMemoryStore (which extends MemoryStore).

Write path:
  - Uses ingest_claim() directly (not remember()) so we can forward
    start_granularity/end_granularity inferred from the raw date string.
    This preserves honest display strings ("2025-02" not "2025-02-01").
  - Provenance is caller-supplied via ClaimInput.provenance dict.
  - valid_from=None → valid_time with only valid_time_confidence=0.0 (no date bound).
  - Never re-ingests duplicates (de-dup is caller responsibility via ClaimInput tracking).
  - Never defaults valid_from to now.

Read path:
  - recall(): uses raw query_memory (no valid_at) → current belief.
  - query_at(): raw query_memory with valid_at and/or as_of_tx_time → bi-temporal.
  - Both extract valid_from_display/valid_until_display from the raw response
    (pre-rendered by the engine at the recorded granularity precision).

Lifted from mempill_demo.adapters.memory_mempill, stripped of:
  - Console-specific logging (replaced with module logger at DEBUG only)
  - Session registry (not needed; claim_ref is returned in WriteReceipt)
  - SessionStats counter
  - _run_scenario() console method
  - Timeline/history/reconcile/beliefs/registry_snapshot methods (W2+ scope)
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import mempill
from mempill import ProvenanceLabel, UnparsableDateError
from mempill.ergonomic import _to_rfc3339

from mempill_showcase.core.domain.models import (
    AlternativeView,
    AuditEntry,
    BeliefView,
    ClaimInput,
    WriteReceipt,
    _prov_abbr,
)

log = logging.getLogger(__name__)


# ── Granularity helpers ───────────────────────────────────────────────────────

def _infer_granularity(date_str: Optional[str]) -> Optional[str]:
    """Infer date granularity from the raw caller-supplied string.

    MUST be called BEFORE _to_rfc3339 expansion so the original precision is visible.
    Returns "year" | "month" | "day" | "instant" | None.
    """
    if not date_str:
        return None
    s = date_str.strip()
    if "T" in s:
        return "instant"
    import re
    m = re.match(r"^(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?$", s)
    if not m:
        return None
    if m.group(3):
        return "day"
    if m.group(2):
        return "month"
    return "year"


# ── Response → domain type helpers ────────────────────────────────────────────

def _extract_belief_view(
    subject: str,
    predicate: str,
    raw: dict,
) -> BeliefView:
    """Map a raw query_memory response dict to a BeliefView domain object."""
    belief_raw = raw.get("belief", {})
    status = belief_raw.get("status", "UNKNOWN")
    primary_raw = belief_raw.get("primary") or {}

    candidates_raw: list[dict] = []
    if status in ("Contested", "Conflict"):
        if primary_raw:
            candidates_raw.append(primary_raw)
        candidates_raw.extend(b for b in (belief_raw.get("alternatives") or []) if b)
    else:
        candidates_raw = belief_raw.get("alternatives") or []

    alternatives = [
        AlternativeView(
            value=(c.get("fact") or {}).get("value"),
            conf=(c.get("confidence") or {}).get("value_confidence"),
            vt_start=(c.get("valid_time") or {}).get("start") or "",
            vt_end=(c.get("valid_time") or {}).get("end") or "open",
            claim_ref=str(c.get("claim_ref") or ""),
            vt_start_display=c.get("valid_from_display"),
            vt_end_display=c.get("valid_until_display"),
        )
        for c in candidates_raw
    ]

    if not primary_raw or status in ("NoBelief", "Contested", "Conflict", "TimingUncertain"):
        contested_value = None
        if status in ("Contested", "Conflict") and primary_raw:
            contested_value = (primary_raw.get("fact") or {}).get("value")
        return BeliefView(
            subject=subject,
            predicate=predicate,
            value=contested_value,
            status=status,
            conf=None,
            vt_start="",
            vt_end="",
            provenance="",
            claim_ref="",
            corroboration=0,
            alternatives=alternatives,
        )

    fact = primary_raw.get("fact") or {}
    vt = primary_raw.get("valid_time") or {}
    conf_raw = primary_raw.get("confidence") or {}
    currency = primary_raw.get("currency_signal") or {}
    return BeliefView(
        subject=subject,
        predicate=predicate,
        value=fact.get("value"),
        status=status,
        conf=conf_raw.get("value_confidence"),
        vt_start=vt.get("start") or "",
        vt_end=vt.get("end") or "open",
        provenance=_prov_abbr(primary_raw.get("provenance")),
        claim_ref=str(primary_raw.get("claim_ref") or ""),
        corroboration=currency.get("corroboration_count", 0),
        alternatives=alternatives,
        vt_start_display=primary_raw.get("valid_from_display"),
        vt_end_display=primary_raw.get("valid_until_display"),
    )


# ── Adapter ───────────────────────────────────────────────────────────────────

class MempillAdapter:
    """BiTemporalMemoryStore backed by a real mempill Engine.

    One instance per session. agent_id is supplied per-call (supports multiple users
    sharing one engine instance, though the showcase uses a single agent_id).
    """

    def __init__(self, engine: mempill.Engine) -> None:
        self._engine = engine

    # ── Write path ────────────────────────────────────────────────────────────

    def write_claim(self, agent_id: str, claim: ClaimInput) -> WriteReceipt:
        """Granularity-aware write via ingest_claim (NOT remember()).

        Preserves honest display strings: "2025-02" for Month, "2023" for Year.
        Caller must resolve subject/predicate to canonical keys before calling.
        """
        # Resolve provenance: caller can pass a ProvenanceLabel dict or None
        prov = claim.provenance if claim.provenance is not None else ProvenanceLabel.external_user_asserted()

        since_raw = claim.valid_from or None
        until_raw = claim.valid_until or None

        # Infer granularity BEFORE RFC3339 expansion
        start_gran = _infer_granularity(since_raw)
        end_gran = _infer_granularity(until_raw)

        # Expand partial dates to RFC3339 for storage
        try:
            since_rfc = _to_rfc3339(since_raw) if since_raw else None
        except UnparsableDateError:
            log.debug("unparseable valid_from %r — storing without date bound", since_raw)
            since_rfc = None
            start_gran = None

        try:
            until_rfc = _to_rfc3339(until_raw) if until_raw else None
        except UnparsableDateError:
            log.debug("unparseable valid_until %r — ignoring", until_raw)
            until_rfc = None
            end_gran = None

        has_date = bool(since_rfc or until_rfc)
        vtc = claim.confidence if has_date else 0.0

        valid_time: dict[str, Any] = {"valid_time_confidence": vtc}
        if since_rfc:
            valid_time["start"] = since_rfc
            if start_gran:
                valid_time["start_granularity"] = start_gran
        if until_rfc:
            valid_time["end"] = until_rfc
            if end_gran:
                valid_time["end_granularity"] = end_gran

        request: dict[str, Any] = {
            "agent_id": agent_id,
            "subject": claim.subject,
            "predicate": claim.predicate,
            "value": claim.value,
            "provenance": prov,
            "cardinality": claim.cardinality,
            "valid_time": valid_time,
            "confidence": {
                "value_confidence": claim.confidence,
                "valid_time_confidence": vtc,
            },
            "criticality": claim.criticality,
            "derived_from": claim.derived_from,
        }

        log.debug(
            "ingest_claim agent=%s subject=%s predicate=%s value=%r valid_from=%s",
            agent_id, claim.subject, claim.predicate, claim.value, since_raw,
        )
        resp = self._engine.ingest_claim(request)
        disp = str(resp["disposition"])
        ref = resp["claim_ref"]
        contested = resp.get("contested_with") or []
        log.debug("ingest result disposition=%s claim_ref=%s", disp, ref)

        return WriteReceipt(
            claim_ref=ref,
            disposition=disp,
            contested_with=contested,
        )

    # ── Read path — current belief ────────────────────────────────────────────

    def recall(self, agent_id: str, subject: str, predicate: str) -> BeliefView:
        """Return the current belief (no valid_at pinning)."""
        log.debug("query_memory current agent=%s subject=%s predicate=%s", agent_id, subject, predicate)
        raw = self._engine.query_memory({
            "agent_id": agent_id,
            "subject": subject,
            "predicate": predicate,
        })
        return _extract_belief_view(subject, predicate, raw)

    # ── Read path — bi-temporal ───────────────────────────────────────────────

    def query_at(
        self,
        agent_id: str,
        subject: str,
        predicate: str,
        valid_at: Optional[str] = None,
        as_of_tx_time: Optional[str] = None,
    ) -> BeliefView:
        """Bi-temporal point-in-time query (two independent axes).

        valid_at=None → current valid-time (most recent open claim).
        as_of_tx_time=None → latest recorded facts (all ingested claims visible).
        """
        log.debug(
            "query_memory bitemoral agent=%s subject=%s predicate=%s valid_at=%s as_of=%s",
            agent_id, subject, predicate, valid_at, as_of_tx_time,
        )
        req: dict = {"agent_id": agent_id, "subject": subject, "predicate": predicate}
        if valid_at is not None:
            req["valid_at"] = valid_at
        if as_of_tx_time is not None:
            req["as_of_tx_time"] = as_of_tx_time

        raw = self._engine.query_memory(req)
        return _extract_belief_view(subject, predicate, raw)

    # ── Reconcile ─────────────────────────────────────────────────────────────

    def reconcile(
        self,
        agent_id: str,
        subject_lines: list[list[str]],
        max_passes: int = 3,
    ) -> dict:
        """Run the engine's reconcile loop for (subject, predicate) pairs.

        Folds non-overlapping claim windows into CommittedCheap without oracle
        involvement. Returns the raw engine response from the last pass.

        Exposed so nodes and tools do NOT need to reach into adapter._engine.
        """
        req = {
            "agent_id": agent_id,
            "subject_lines": subject_lines,
        }
        last_resp: dict = {}
        for _pass in range(max_passes):
            last_resp = self._engine.reconcile(req)
            if last_resp.get("oracle_escalations", 0) == 0:
                break
        log.debug(
            "reconcile agent=%s subject_lines=%s passes=%d result=%s",
            agent_id, subject_lines, _pass + 1, last_resp,
        )
        return last_resp

    # ── Audit ─────────────────────────────────────────────────────────────────

    def audit(self, agent_id: str, limit: int = 50) -> list[AuditEntry]:
        """Return the most recent *limit* audit entries for agent_id."""
        resp = self._engine.query_audit({
            "agent_id": agent_id,
            "claim_ref": None,
            "from_tx_time": None,
            "limit": limit,
        })
        return [
            AuditEntry(
                claim_ref=e.get("claim_ref", ""),
                event_kind=e.get("event_kind", "?"),
                disposition=e.get("disposition", "?"),
                recorded_at=str(e.get("recorded_at", "")),
                rationale=e.get("rationale", ""),
            )
            for e in resp.get("entries", [])
        ]
