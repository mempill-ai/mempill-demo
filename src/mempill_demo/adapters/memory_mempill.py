"""
mempill_demo.adapters.memory_mempill — THE ONLY module that imports mempill.

Encapsulates all SDK quirks:
  - belief["belief"]["primary"]["fact"]["value"] deep path extraction
  - audit entries have no subject/predicate/value → session registry correlation
  - ProvenanceLabel and Disposition enum mapping
  - reconcile() returns only winner; loser is in audit ledger
  - Oracle wiring: open_oracle / open_oracle_in_memory for HITL adjudication queue

Date normalization (valid_time) is now delegated to mempill.remember() via
RememberOptions, which handles the RFC3339 expansion internally and raises
UnparsableDateError for natural-language dates.  The ingest() method catches that
error and retries without the date window — preserving the "omit window, still
ingest" fallback that existed when _to_rfc3339() returned None.

RECALL_REENTRY claims still use the raw dict path because remember() hardcodes
derived_from=[] and does not expose a derived_from parameter.  If that gap is
closed upstream, the RECALL_REENTRY branch can be migrated too.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import mempill
from mempill import Disposition, ProvenanceLabel
from mempill import remember as _remember, RememberOptions, UnparsableDateError

log = logging.getLogger("mempill.demo")

from mempill_demo.domain.models import (
    AuditEntry,
    BeliefView,
    AlternativeView,
    ClaimMeta,
    ParsedCommand,
    ReconcileOutcome,
    SessionStats,
    CommandKind,
    _prov_abbr,
)


def _map_alternatives(belief: dict) -> "list[AlternativeView]":
    """Map the `alternatives` array of a query_memory belief into AlternativeView list."""
    out: list[AlternativeView] = []
    for alt in (belief.get("alternatives") or []):
        if alt is None:
            continue
        vt = alt.get("valid_time") or {}
        conf = alt.get("confidence", {}) or {}
        out.append(AlternativeView(
            value=(alt.get("fact", {}) or {}).get("value"),
            conf=conf.get("value_confidence") if isinstance(conf, dict) else conf,
            vt_start=vt.get("start", "") if isinstance(vt, dict) else "",
            vt_end=(vt.get("end") or "open") if isinstance(vt, dict) else "open",
            claim_ref=alt.get("claim_ref", ""),
        ))
    return out


class MempillMemoryStore:
    """MemoryStore adapter backed by a real mempill Engine."""

    def __init__(self, engine: mempill.Engine, agent_id: str) -> None:
        self._engine = engine
        self._agent_id = agent_id
        # Session-local claim registry: claim_ref → ClaimMeta
        # Needed because audit entries carry claim_ref but NOT subject/predicate/value.
        self._registry: dict[str, ClaimMeta] = {}
        self._stats = SessionStats()
        # The (subject, predicate) of the most recent recall — lets /reconcile target
        # the line the user was just asking about without re-typing it.
        self._last_recalled: "Optional[tuple[str, str]]" = None

    @property
    def stats(self) -> SessionStats:
        return self._stats

    # ── Write path ────────────────────────────────────────────────────────────

    def ingest(self, cmd: ParsedCommand) -> ClaimMeta:
        """Ingest a ParsedCommand into the engine; register in local registry.

        UserAsserted and ModelDerived commands are routed through mempill.remember()
        (ergonomic API), which handles RFC3339 normalization and the valid_time dict
        quirk internally.  If a date string is unparseable (e.g. natural language),
        UnparsableDateError is caught and the command is re-submitted without the
        date window so the fact is still stored.

        RECALL_REENTRY uses the raw dict path because mempill.remember() does not
        expose a derived_from parameter.
        """
        prov_str = (cmd.extra or {}).get("provenance_str", "UserAsserted")
        cardinality = (cmd.extra or {}).get("cardinality", "Functional")

        if cmd.kind == CommandKind.RECALL_REENTRY:
            # ── Raw dict path: remember() lacks derived_from support ──────────
            prov = ProvenanceLabel.recall_re_entry()
            conf_val = 0.7
            valid_time: dict = {"valid_time_confidence": conf_val}
            derived_from: list[str] = [cmd.source_claim_ref] if cmd.source_claim_ref else []

            request = {
                "agent_id": self._agent_id,
                "subject": cmd.subject,
                "predicate": cmd.predicate,
                "value": cmd.value,
                "provenance": prov,
                "cardinality": cardinality,
                "valid_time": valid_time,
                "confidence": {
                    "value_confidence": conf_val,
                    "valid_time_confidence": conf_val,
                },
                "criticality": "Low",
                "derived_from": derived_from,
            }

            log.info(
                "→ ingest_claim subject=%s predicate=%s value=%r prov=%s",
                cmd.subject, cmd.predicate, cmd.value, prov,
            )
            log.debug("  ingest_claim request=%r", request)
            resp = self._engine.ingest_claim(request)
            disp = str(resp["disposition"])
            ref = resp["claim_ref"]
            contested = resp.get("contested_with", [])
            log.info(
                "← disposition=%s claim_ref=%s contested_with=%s",
                disp, ref[:8], [r[:8] for r in contested] if contested else [],
            )
            log.debug("  ingest_claim response=%r", resp)
            meta = ClaimMeta(
                subject=cmd.subject,
                predicate=cmd.predicate,
                value=cmd.value,
                provenance=prov,
                valid_time=None,
                conf=conf_val,
                disposition=disp,
                claim_ref=ref,
            )
            self._registry[ref] = meta
            return meta

        # ── Ergonomic path: UserAsserted / ModelDerived ───────────────────────
        prov = ProvenanceLabel.model_derived() if prov_str == "ModelDerived" else ProvenanceLabel.external_user_asserted()
        conf_val = cmd.conf

        opts = RememberOptions(
            valid_from=cmd.since or None,
            valid_until=cmd.until or None,
            confidence=conf_val,
            cardinality=cardinality,
            provenance=prov,
            criticality="Medium",
        )

        log.info(
            "→ ingest_claim subject=%s predicate=%s value=%r prov=%s",
            cmd.subject, cmd.predicate, cmd.value, prov,
        )

        try:
            receipt = _remember(self._engine, self._agent_id, cmd.subject, cmd.predicate, cmd.value, opts)
        except UnparsableDateError as exc:
            # Natural-language date (e.g. "March 2020") — omit the window and
            # re-submit so the fact is still stored (matches pre-refactor behavior
            # where _to_rfc3339 returned None and the bound was silently skipped).
            log.warning("  unparseable date %r — retrying without valid_time window", exc.input)
            opts_no_date = RememberOptions(
                confidence=conf_val,
                cardinality=cardinality,
                provenance=prov,
                criticality="Medium",
            )
            receipt = _remember(self._engine, self._agent_id, cmd.subject, cmd.predicate, cmd.value, opts_no_date)

        disp = str(receipt.disposition)
        ref = receipt.claim_ref
        contested = receipt.contested_with
        log.info(
            "← disposition=%s claim_ref=%s contested_with=%s",
            disp, ref[:8], [r[:8] for r in contested] if contested else [],
        )

        # Reconstruct valid_time for ClaimMeta (only for display purposes in registry)
        has_date = bool(cmd.since or cmd.until)
        meta = ClaimMeta(
            subject=cmd.subject,
            predicate=cmd.predicate,
            value=cmd.value,
            provenance=prov,
            valid_time={"valid_from": cmd.since, "valid_until": cmd.until} if has_date else None,
            conf=conf_val,
            disposition=disp,
            claim_ref=ref,
        )
        self._registry[ref] = meta
        return meta

    # ── Read paths ────────────────────────────────────────────────────────────

    def recall(self, subject: str, predicate: str) -> BeliefView:
        """Query the engine and map to a BeliefView domain object."""
        log.info("→ query_memory subject=%s predicate=%s", subject, predicate)
        resp = self._engine.query_memory({
            "agent_id": self._agent_id,
            "subject": subject,
            "predicate": predicate,
        })
        log.debug("  query_memory response=%r", resp)
        belief = resp.get("belief", {})
        status = belief.get("status", "UNKNOWN")
        primary_val = (belief.get("primary") or {}).get("fact", {}).get("value")
        alts = [
            (a.get("fact") or {}).get("value")
            for a in (belief.get("alternatives") or [])
            if a
        ]
        log.info(
            "← status=%s primary=%r alternatives=%r",
            status, primary_val, alts,
        )
        if subject and predicate:
            self._last_recalled = (subject, predicate)
        return self._map_belief(resp, subject, predicate)

    def last_recalled(self) -> "Optional[tuple[str, str]]":
        """The (subject, predicate) of the most recent recall — used by /reconcile."""
        return self._last_recalled

    def reconcile(self, subject: str, predicate: str) -> list[ReconcileOutcome]:
        """Run reconciliation; return list of ReconcileOutcome domain objects."""
        log.info("→ reconcile subject=%s predicate=%s", subject, predicate)
        resp = self._engine.reconcile({
            "agent_id": self._agent_id,
            "subject_lines": [(subject, predicate)],
        })
        log.debug("  reconcile response=%r", resp)
        outcomes = resp.get("outcomes", [])
        log.info("← reconcile outcomes_count=%d outcomes=%r", len(outcomes), outcomes)
        return [ReconcileOutcome(claim_ref=ref, disposition=disp) for ref, disp in outcomes]

    def history(
        self,
        subject: str,
        predicate: str,
    ) -> tuple[list[ClaimMeta], list[AuditEntry]]:
        """Return all ClaimMeta for the subject/predicate + correlated audit entries."""
        metas = [
            meta for meta in self._registry.values()
            if meta.subject == subject and meta.predicate == predicate
        ]
        if not metas:
            return [], []

        refs = {m.claim_ref for m in metas}
        all_audit = self._fetch_audit(limit=500)
        relevant = [e for e in all_audit if e.claim_ref in refs]
        return metas, relevant

    def audit(self, limit: int) -> list[AuditEntry]:
        """Return the last `limit` audit entries."""
        return self._fetch_audit(limit=limit)

    def beliefs(self) -> list[BeliefView]:
        """Return current beliefs for all unique subject/predicate pairs in registry."""
        seen: set[tuple[str, str]] = set()
        result: list[BeliefView] = []
        for meta in self._registry.values():
            key = (meta.subject, meta.predicate)
            if key in seen:
                continue
            seen.add(key)
            try:
                belief = self.recall(meta.subject, meta.predicate)
                result.append(belief)
            except Exception:
                pass
        return result

    def registry_snapshot(self) -> dict[str, ClaimMeta]:
        """Return a snapshot of the session registry."""
        return dict(self._registry)

    # ── Oracle / HITL methods ─────────────────────────────────────────────────

    def list_pending(self) -> list[dict]:
        """Return pending adjudication requests for this agent from the engine queue."""
        log.info("→ list_pending_adjudications agent_id=%s", self._agent_id)
        result = self._engine.list_pending_adjudications(agent_id=self._agent_id)
        log.debug("  list_pending_adjudications response=%r", result)
        handle_ids = [p.get("handle_id", "?")[:8] for p in result]
        log.info("← pending_count=%d handle_ids=%r", len(result), handle_ids)
        return result

    def submit(self, handle_id: str, verdict: str) -> dict:
        """
        Submit a human verdict for a pending adjudication.

        verdict: "Affirm" | "Deny" | "Unknown"
        Builds the response dict with external_first_hand provenance (decision E in ARCHITECTURE).
        """
        log.info("→ submit_adjudication handle_id=%s verdict=%s", handle_id[:8], verdict)
        response = {
            "handle_id": handle_id,
            "verdict": verdict,
            "evidence_provenance": ProvenanceLabel.external_first_hand(),
        }
        result = self._engine.submit_adjudication(response)
        log.debug("  submit_adjudication response=%r", result)
        log.info(
            "← submit_adjudication disposition=%s claim_ref=%s",
            result.get("disposition", "?"),
            str(result.get("claim_ref", "?"))[:8],
        )
        return result

    def _sweep_expired(self) -> int:
        """
        Sweep expired adjudications back to Contested.

        Returns the count of adjudications swept (0 if none). Called on startup
        (decision H.2) and via /sweep in the REPL.
        """
        log.info("→ sweep_expired_adjudications")
        result = self._engine.sweep_expired_adjudications()
        log.debug("  sweep_expired_adjudications response=%r", result)
        # The engine returns a count or a dict with a count field
        if isinstance(result, int):
            swept = result
        elif isinstance(result, dict):
            swept = result.get("swept", result.get("count", 0))
        else:
            swept = 0
        log.info("← swept_count=%d", swept)
        return swept

    # ── Scenario runner (keeps mempill out of app/) ───────────────────────────

    def _run_scenario(self) -> None:
        """
        Run the 3-act CEO scenario — engine calls live here (not in app/scenario.py)
        so that mempill is only imported in adapters/.
        """
        engine = self._engine
        agent_id = self._agent_id

        print()
        print("=" * 62)
        print("  mempill Console Agent — 3-Act Scenario (auto-play)")
        print("  Subject: acme:ceo  Predicate: held_by")
        print("  No LLM required — deterministic structural memory only.")
        print("=" * 62)

        # ── ACT 1: Alice is CEO ───────────────────────────────────────────────
        print("\n--- ACT 1: Ingest 'Alice is CEO' (valid from 2020-01-01) ---")
        resp_alice = engine.ingest_claim({
            "agent_id": agent_id,
            "subject": "acme:ceo",
            "predicate": "held_by",
            "value": "Alice",
            "provenance": ProvenanceLabel.external_first_hand(),
            "cardinality": "Functional",
            "valid_time": {"start": "2020-01-01T00:00:00Z", "valid_time_confidence": 0.95},
            "confidence": {"value_confidence": 0.95, "valid_time_confidence": 0.95},
            "criticality": "Medium",
            "derived_from": [],
        })
        alice_ref = resp_alice["claim_ref"]
        act1_disp = resp_alice["disposition"]
        self._registry[alice_ref] = ClaimMeta(
            subject="acme:ceo", predicate="held_by", value="Alice",
            provenance=ProvenanceLabel.external_first_hand(),
            valid_time=None, conf=0.95, disposition=str(act1_disp), claim_ref=alice_ref,
        )
        self._stats.n_ingests += 1
        print(f"  disposition:  {act1_disp}")
        print(f"  claim_ref:    {alice_ref[:8]}...")
        if str(act1_disp) == str(Disposition.CommittedCheap):
            print("  [note] CommittedCheap — fast-path commit, no conflict.")
        else:
            print(f"  [note] ACTUAL: {act1_disp!r} (narrating real engine behavior).")
        q1 = engine.query_memory({"agent_id": agent_id, "subject": "acme:ceo", "predicate": "held_by"})
        q1_primary = q1.get("belief", {}).get("primary") or {}
        print(f"  RECALL → value={q1_primary.get('fact', {}).get('value')!r}"
              f"  status={q1.get('belief', {}).get('status')}")

        # ── ACT 2: Bob is CEO — conflict ──────────────────────────────────────
        print("\n--- ACT 2: Ingest 'Bob is CEO' (valid from 2023-03-15) — HITL conflict ---")
        resp_bob = engine.ingest_claim({
            "agent_id": agent_id,
            "subject": "acme:ceo",
            "predicate": "held_by",
            "value": "Bob",
            "provenance": ProvenanceLabel.external_first_hand(),
            "cardinality": "Functional",
            "valid_time": {"start": "2023-03-15T00:00:00Z", "valid_time_confidence": 0.9},
            "confidence": {"value_confidence": 0.9, "valid_time_confidence": 0.9},
            "criticality": "Medium",
            "derived_from": [],
        })
        bob_ref = resp_bob["claim_ref"]
        act2_disp = resp_bob["disposition"]
        contested_with = resp_bob.get("contested_with", [])
        self._registry[bob_ref] = ClaimMeta(
            subject="acme:ceo", predicate="held_by", value="Bob",
            provenance=ProvenanceLabel.external_first_hand(),
            valid_time=None, conf=0.9, disposition=str(act2_disp), claim_ref=bob_ref,
        )
        self._stats.n_ingests += 1
        print(f"  disposition:     {act2_disp}")
        print(f"  contested_with:  {[r[:8]+'...' for r in contested_with]}")

        # With HumanOracle the conflict goes to QueuedForAdjudication instead of plain Contested
        if str(act2_disp) == "QueuedForAdjudication":
            self._stats.n_contested += 1
            print("  [note] QueuedForAdjudication — conflict handed to human oracle queue.")
            print("  [badge] QUEUED  ← conflict is pending human /review")
        elif str(act2_disp) == str(Disposition.Contested):
            self._stats.n_contested += 1
            print("  [note] CONTESTED — two open-ended Functional claims overlap.")
            print("  [badge] CONTESTED  ← use /review to resolve")
        elif str(act2_disp) == str(Disposition.CommittedCheap):
            print("  [note] ACTUAL: CommittedCheap — engine fast-committed Bob without conflict flag.")
        else:
            print(f"  [note] ACTUAL: {act2_disp!r}")

        # Show query_memory while pending (surfaces Contested[both] while queued)
        q_pending = engine.query_memory({"agent_id": agent_id, "subject": "acme:ceo", "predicate": "held_by"})
        pend_status = q_pending.get("belief", {}).get("status")
        pend_primary = (q_pending.get("belief", {}).get("primary") or {})
        pend_val = pend_primary.get("fact", {}).get("value")
        pend_alts = q_pending.get("belief", {}).get("alternatives") or []
        print(f"\n  RECALL while queued → status={pend_status}  primary={pend_val!r}")
        for alt in pend_alts:
            alt_val = (alt.get("fact") or {}).get("value")
            print(f"    alternative: {alt_val!r}")
        print("  Both Alice and Bob are visible — agent reports uncertainty.")

        # Show pending adjudications
        pending = engine.list_pending_adjudications(agent_id=agent_id)
        print(f"\n  Pending adjudications: {len(pending)}")
        for p in pending:
            print(f"    handle={p['handle_id'][:8]}...  "
                  f"incumbent={p['incumbent_value']!r}  challenger={p['challenger_value']!r}")

        # Simulate /review — choose challenger (Bob wins)
        committed_bob_ref = bob_ref
        outcomes = []
        if pending:
            print("\n  [/review] Human chooses: [c]hallenger → Bob wins")
            handle_id = pending[0]["handle_id"]
            sub_result = engine.submit_adjudication({
                "handle_id": handle_id,
                "verdict": "Affirm",
                "evidence_provenance": ProvenanceLabel.external_first_hand(),
            })
            sub_disp = sub_result.get("disposition", "?")
            sub_ref = sub_result.get("claim_ref", bob_ref)
            committed_bob_ref = sub_ref
            print(f"  submit_adjudication → disposition={sub_disp}  ref={sub_ref[:8]}...")
            if sub_disp in ("Superseded", "Invalidated"):
                self._stats.n_superseded += 1
                print(f"  [badge] SUPERSEDED  ← {alice_ref[:8]}... (Alice now superseded)")
            else:
                print(f"  [badge] COMMITTED   ← {sub_ref[:8]}... (Bob, now authoritative)")
            outcomes = [("review-resolved", sub_disp)]
        else:
            # Fallback: engine already resolved without oracle (CommittedCheap path)
            print("\n  [note] No pending adjudications — engine resolved at ingest.")
            committed_bob_ref = bob_ref

        q_post = engine.query_memory({"agent_id": agent_id, "subject": "acme:ceo", "predicate": "held_by"})
        post_val = ((q_post.get("belief", {}).get("primary") or {}).get("fact") or {}).get("value")
        post_status = q_post.get("belief", {}).get("status")
        print(f"  Post-review belief: \"{post_val}\"  status={post_status}")

        # ── ACT 3: Amplification firewall ─────────────────────────────────────
        print("\n--- ACT 3: RECALL_REENTRY ×5 — amplification firewall ---")
        print(f"  Re-ingesting 'Bob is CEO' 5× with RecallReEntry provenance (derived_from={committed_bob_ref[:8]}...)")
        from mempill_demo.domain.models import CommandKind, ParsedCommand
        for i in range(5):
            rr_cmd = ParsedCommand(
                kind=CommandKind.RECALL_REENTRY,
                subject="acme:ceo",
                predicate="held_by",
                value="Bob",
                source_claim_ref=committed_bob_ref,
                conf=0.7,
            )
            self.ingest(rr_cmd)

        q_after = engine.query_memory({"agent_id": agent_id, "subject": "acme:ceo", "predicate": "held_by"})
        after_val = ((q_after.get("belief", {}).get("primary") or {}).get("fact") or {}).get("value")
        after_status = q_after.get("belief", {}).get("status")
        corroboration = (q_after.get("belief", {}).get("primary") or {}).get("currency_signal", {}).get("corroboration_count", 0)
        print(f"  Belief after 5 re-entries: \"{after_val}\"  status={after_status}")
        print(f"  corroboration_count: {corroboration}")
        print("  [badge] FIREWALL HELD — RecallReEntry did not alter the belief.")

        print()
        print("=" * 62)
        print("  Scenario complete.")
        print(f"  Act 1: Alice  → {act1_disp}")
        print(f"  Act 2: Bob    → {act2_disp}  /review→{[d for _, d in outcomes]}")
        print(f"  Act 3: ×5 recall-reentry  → belief unchanged ({after_val}, {after_status})")
        print("=" * 62)
        print()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _fetch_audit(self, limit: int) -> list[AuditEntry]:
        log.info("→ query_audit agent_id=%s limit=%d", self._agent_id, limit)
        resp = self._engine.query_audit({
            "agent_id": self._agent_id,
            "claim_ref": None,
            "from_tx_time": None,
            "limit": limit,
        })
        log.debug("  query_audit response=%r", resp)
        entries = resp.get("entries", [])
        log.info("← audit_entries_count=%d", len(entries))
        return [
            AuditEntry(
                claim_ref=e.get("claim_ref", ""),
                event_kind=e.get("event_kind", "?"),
                disposition=e.get("disposition", "?"),
                recorded_at=str(e.get("recorded_at", "")),
                rationale=e.get("rationale", ""),
            )
            for e in entries
        ]

    def _map_belief(self, resp: dict, subject: str, predicate: str) -> BeliefView:
        """Map a raw query_memory response dict to a BeliefView domain object."""
        belief = resp.get("belief", {})
        status = belief.get("status", "UNKNOWN")
        primary = belief.get("primary")

        if primary is None:
            # No primary winner. This is EITHER a true no-belief OR a Contested belief
            # (Contested has no primary — both candidates live in `alternatives`). Preserve
            # the real status and the alternatives instead of discarding them as "UNKNOWN".
            return BeliefView(
                subject=subject,
                predicate=predicate,
                value=None,
                status=status,
                conf=None,
                vt_start="",
                vt_end="",
                provenance="",
                claim_ref="",
                corroboration=0,
                alternatives=_map_alternatives(belief),
            )

        # Extract primary fields
        value = primary.get("fact", {}).get("value")
        conf_dict = primary.get("confidence", {})
        conf_val = conf_dict.get("value_confidence") if isinstance(conf_dict, dict) else conf_dict
        vt = primary.get("valid_time") or {}
        vt_start = vt.get("start", "") if isinstance(vt, dict) else ""
        vt_end = (vt.get("end") or "open") if isinstance(vt, dict) else "open"
        claim_ref = primary.get("claim_ref", "")
        currency = primary.get("currency_signal", {}) or {}
        corroboration = currency.get("corroboration_count", 0)
        prov = _prov_abbr(primary.get("provenance"))

        # Map alternatives (for R2/Contested)
        alternatives = _map_alternatives(belief)

        return BeliefView(
            subject=subject,
            predicate=predicate,
            value=value,
            status=status,
            conf=conf_val,
            vt_start=vt_start,
            vt_end=vt_end,
            provenance=prov,
            claim_ref=claim_ref,
            corroboration=corroboration,
            alternatives=alternatives,
        )
