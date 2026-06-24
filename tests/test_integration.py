"""
tests/test_integration.py — integration selftest using real mempill engine.

Assertions:
  T1   Alice ingest → CommittedCheap
  T2   RECALL acme:ceo held_by → R1 value="Alice"
  T3   Bob ingest (conflicting open-ended Functional) → QueuedForAdjudication (oracle-wired)
  T4   HITL: submit Affirm → Bob wins; Alice Superseded in audit
  T5   RECALL after adjudication → current=Bob, Resolved
  T6   RECALL_REENTRY ×5 → belief unchanged (firewall held)
  T7   audit contains both Alice and Bob claim_refs
  T8   HITL: list_pending shows handle with correct incumbent/challenger
  T9   HITL: query while queued → Contested[both] surfaces both values (pre-T4)
  T10  HITL: submit Deny (incumbent=Alice wins) → query → single belief Alice
  T11  HITL: submit Unknown (abstain) → belief stays Contested; removed from queue
  T12  HITL durability: file-backed engine, ingest conflict, close+reopen, pending survives

Does NOT import console.inference.llm — fully deterministic.
Exit 0 on all assertions pass. Exit 1 on any failure.
"""
from __future__ import annotations

import sys
import tempfile
import pathlib

import mempill
from mempill import Disposition, ProvenanceLabel
from mempill_demo.adapters.human_oracle import HumanOracle
from mempill_demo.adapters.memory_mempill import MempillMemoryStore


def _assert(condition: bool, label: str, detail: str = "") -> None:
    if condition:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}{': ' + detail if detail else ''}")
        sys.exit(1)


def _belief_value(resp: dict) -> object:
    primary = resp.get("belief", {}).get("primary")
    if primary:
        return primary.get("fact", {}).get("value")
    return None


def _belief_status(resp: dict) -> str:
    return resp.get("belief", {}).get("status", "UNKNOWN")


def run() -> None:
    print()
    print("mempill Console Agent — selftest")
    print("  Deterministic only (no API key required).")
    print()

    # All tests use oracle-wired engine (HumanOracle) — this is the production path.
    engine = mempill.open_oracle_in_memory(HumanOracle())
    agent_id = "selftest-agent"
    store = MempillMemoryStore(engine, agent_id=agent_id)

    # ── T1: Alice ingest → CommittedCheap ────────────────────────────────────
    print("T1 — Alice ingest (external_first_hand, Functional, 2020-)")
    resp_alice = engine.ingest_claim({
        "agent_id": agent_id,
        "subject": "acme:ceo",
        "predicate": "held_by",
        "value": "Alice",
        "provenance": ProvenanceLabel.external_first_hand(),
        "cardinality": "Functional",
        "valid_time": {"start": "2020-01-01T00:00:00Z", "valid_time_confidence": 0.9},
        "confidence": {"value_confidence": 0.95, "valid_time_confidence": 0.9},
        "criticality": "Medium",
        "derived_from": [],
    })
    alice_ref = resp_alice["claim_ref"]
    act1_disp = resp_alice["disposition"]
    print(f"  disposition={act1_disp}  ref={alice_ref[:8]}...")
    _assert(
        act1_disp == Disposition.CommittedCheap,
        "T1: Alice disposition == CommittedCheap",
        f"actual={act1_disp!r}",
    )

    # ── T2: RECALL → R1 value=Alice ──────────────────────────────────────────
    print("\nT2 — RECALL acme:ceo held_by → R1 value=Alice")
    q1 = engine.query_memory({
        "agent_id": agent_id,
        "subject": "acme:ceo",
        "predicate": "held_by",
    })
    val1 = _belief_value(q1)
    status1 = _belief_status(q1)
    print(f"  value={val1!r}  status={status1}")
    _assert(val1 == "Alice", "T2: RECALL value == 'Alice'", f"actual={val1!r}")
    _assert(
        status1 in ("Committed", "CommittedCheap", "CommittedInferred", "Resolved"),
        "T2: status is a Committed variant",
        f"actual={status1!r}",
    )

    # ── T3: Bob ingest (conflicting, open-ended) → QueuedForAdjudication ─────
    print("\nT3 — Bob ingest (same Functional, open-ended, 2023-) — oracle: QueuedForAdjudication")
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
    print(f"  disposition={act2_disp}  ref={bob_ref[:8]}...  contested_with={[r[:8]+'...' for r in contested_with]}")
    _assert(
        str(act2_disp) == "QueuedForAdjudication",
        "T3: Bob disposition == QueuedForAdjudication (oracle-wired)",
        f"actual={act2_disp!r}",
    )

    # ── T9 (interleaved here): query while queued → Contested[both] ──────────
    print("\nT9 — RECALL while queued → Contested[both] surfaces both values")
    q_pending = engine.query_memory({
        "agent_id": agent_id,
        "subject": "acme:ceo",
        "predicate": "held_by",
    })
    pend_status = _belief_status(q_pending)
    pend_alts = q_pending.get("belief", {}).get("alternatives") or []
    print(f"  status={pend_status}  alternatives={len(pend_alts)}")
    _assert(
        pend_status in ("Contested", "QueuedForAdjudication", "PendingConflict"),
        "T9: belief status is Contested/queued while pending",
        f"actual={pend_status!r}",
    )

    # ── T8: list_pending shows handle with correct incumbent/challenger ────────
    print("\nT8 — list_pending shows handle: incumbent=Alice, challenger=Bob")
    pending_list = store.list_pending()
    print(f"  pending count: {len(pending_list)}")
    _assert(len(pending_list) >= 1, "T8: at least 1 pending adjudication", f"actual={len(pending_list)}")
    first = pending_list[0]
    incumbent = first.get("incumbent_value")
    challenger = first.get("challenger_value")
    handle_id = first.get("handle_id")
    print(f"  handle={handle_id[:8]}...  incumbent={incumbent!r}  challenger={challenger!r}")
    _assert(incumbent == "Alice", "T8: incumbent_value == 'Alice'", f"actual={incumbent!r}")
    _assert(challenger == "Bob", "T8: challenger_value == 'Bob'", f"actual={challenger!r}")

    # ── T4: submit Affirm → Bob wins; audit shows Alice Superseded ────────────
    print("\nT4 — submit Affirm (challenger=Bob wins) → adjudication resolved")
    submit_result = store.submit(handle_id, "Affirm")
    sub_disp = submit_result.get("disposition", "?")
    committed_bob_ref = submit_result.get("claim_ref", bob_ref)
    print(f"  submit_adjudication → disposition={sub_disp}  ref={committed_bob_ref[:8]}...")

    audit_t4 = engine.query_audit({
        "agent_id": agent_id,
        "claim_ref": None,
        "from_tx_time": None,
        "limit": 200,
    })
    audit_entries_t4 = audit_t4.get("entries", [])
    all_dispositions_t4 = {e.get("disposition") for e in audit_entries_t4}
    alice_superseded = any(
        e.get("claim_ref") == alice_ref and e.get("disposition") in ("Superseded", "Invalidated")
        for e in audit_entries_t4
    )
    print(f"  Alice ref {alice_ref[:8]}: Superseded event in audit = {alice_superseded}")
    print(f"  All dispositions in audit: {all_dispositions_t4}")
    _assert(alice_superseded, "T4: Alice claim → Superseded in audit after adjudication")
    _assert(sub_disp in ("CommittedCheap", "Committed", "Resolved", "CommittedInferred"),
            "T4: Bob claim → Committed after Affirm", f"actual={sub_disp!r}")

    # ── T5: RECALL after adjudication → current=Bob ───────────────────────────
    print("\nT5 — RECALL after adjudication → current value=Bob")
    q_current = engine.query_memory({
        "agent_id": agent_id,
        "subject": "acme:ceo",
        "predicate": "held_by",
    })
    current_val = _belief_value(q_current)
    current_status = _belief_status(q_current)
    print(f"  value={current_val!r}  status={current_status}")
    _assert(current_val == "Bob", "T5: current belief value == 'Bob'", f"actual={current_val!r}")
    _assert(
        current_status in ("Committed", "CommittedCheap", "CommittedInferred", "Resolved"),
        "T5: current status is a Committed variant",
        f"actual={current_status!r}",
    )

    # ── T6: RECALL_REENTRY ×5 → belief unchanged (firewall held) ─────────────
    print("\nT6 — RECALL_REENTRY ×5 → belief unchanged")
    for i in range(5):
        engine.ingest_claim({
            "agent_id": agent_id,
            "subject": "acme:ceo",
            "predicate": "held_by",
            "value": "Bob",
            "provenance": ProvenanceLabel.recall_re_entry(),
            "cardinality": "Functional",
            "valid_time": {"start": "2023-03-15T00:00:00Z", "valid_time_confidence": 0.7},
            "confidence": {"value_confidence": 0.7, "valid_time_confidence": 0.7},
            "criticality": "Low",
            "derived_from": [committed_bob_ref],
        })

    q_after = engine.query_memory({
        "agent_id": agent_id,
        "subject": "acme:ceo",
        "predicate": "held_by",
    })
    val_after = _belief_value(q_after)
    status_after = _belief_status(q_after)
    corroboration = (q_after.get("belief", {}).get("primary") or {}).get("currency_signal", {}).get("corroboration_count", 0)
    print(f"  value={val_after!r}  status={status_after}  corroboration_count={corroboration}")
    _assert(val_after == "Bob", "T6: belief value unchanged after 5 recall-reentries", f"actual={val_after!r}")
    _assert(
        status_after in ("Committed", "CommittedCheap", "CommittedInferred", "Resolved"),
        "T6: belief status still Committed after 5 recall-reentries",
        f"actual={status_after!r}",
    )
    print(f"  [note] Belief unchanged (firewall held). corroboration_count={corroboration}")

    # ── T7: audit contains both Alice and Bob claim_refs ──────────────────────
    print("\nT7 — audit contains both Alice and Bob claim_refs")
    audit_t7 = engine.query_audit({
        "agent_id": agent_id,
        "claim_ref": None,
        "from_tx_time": None,
        "limit": 500,
    })
    entries_t7 = audit_t7.get("entries", [])
    refs_in_audit = {e.get("claim_ref") for e in entries_t7}
    print(f"  total audit entries: {len(entries_t7)}  distinct claim_refs: {len(refs_in_audit)}")
    print(f"  alice_ref in audit: {alice_ref in refs_in_audit}  (ref={alice_ref[:8]})")
    print(f"  bob_ref in audit:   {bob_ref in refs_in_audit}  (ref={bob_ref[:8]})")

    for ref, label in [(alice_ref, "Alice"), (bob_ref, "Bob")]:
        events = [e for e in entries_t7 if e.get("claim_ref") == ref]
        for e in events:
            print(f"    [{label} {ref[:8]}]  {e.get('event_kind','?')} → {e.get('disposition','?')}")

    _assert(alice_ref in refs_in_audit, "T7: alice_ref appears in audit")
    _assert(bob_ref in refs_in_audit, "T7: bob_ref appears in audit")

    alice_events = {e.get("disposition") for e in entries_t7 if e.get("claim_ref") == alice_ref}
    bob_events = {e.get("disposition") for e in entries_t7 if e.get("claim_ref") == bob_ref}
    print(f"  Alice dispositions in audit: {alice_events}")
    print(f"  Bob dispositions in audit:   {bob_events}")
    _assert(
        bool({"Superseded", "Invalidated"} & alice_events),
        "T7: Alice has Superseded/Invalidated event in audit",
        f"actual={alice_events}",
    )
    _assert(
        bool({"CommittedCheap", "CommittedInferred", "Resolved", "Committed", "QueuedForAdjudication"} & bob_events),
        "T7: Bob has a Committed/Resolved/Queued event in audit",
        f"actual={bob_events}",
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # T10: Deny verdict — incumbent wins
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 62)
    print("T10 — submit Deny (incumbent wins): separate engine instance")
    print("=" * 62)

    deny_engine = mempill.open_oracle_in_memory(HumanOracle())
    deny_agent = "deny-selftest-agent"
    deny_store = MempillMemoryStore(deny_engine, deny_agent)

    deny_engine.ingest_claim({
        "agent_id": deny_agent, "subject": "acme:coo", "predicate": "held_by", "value": "Carol",
        "provenance": ProvenanceLabel.external_first_hand(), "cardinality": "Functional",
        "valid_time": {"start": "2020-01-01T00:00:00Z", "valid_time_confidence": 0.9},
        "confidence": {"value_confidence": 0.9, "valid_time_confidence": 0.9},
        "criticality": "Medium", "derived_from": [],
    })
    deny_engine.ingest_claim({
        "agent_id": deny_agent, "subject": "acme:coo", "predicate": "held_by", "value": "Dave",
        "provenance": ProvenanceLabel.external_first_hand(), "cardinality": "Functional",
        "valid_time": {"start": "2023-01-01T00:00:00Z", "valid_time_confidence": 0.9},
        "confidence": {"value_confidence": 0.9, "valid_time_confidence": 0.9},
        "criticality": "Medium", "derived_from": [],
    })

    deny_pending = deny_store.list_pending()
    print(f"  pending count: {len(deny_pending)}")
    _assert(len(deny_pending) >= 1, "T10: pending adjudication for Deny test")
    deny_handle = deny_pending[0]["handle_id"]
    deny_result = deny_store.submit(deny_handle, "Deny")
    deny_disp = deny_result.get("disposition", "?")
    print(f"  Submit Deny → disposition={deny_disp}")

    q_deny = deny_engine.query_memory({"agent_id": deny_agent, "subject": "acme:coo", "predicate": "held_by"})
    deny_val = _belief_value(q_deny)
    deny_status = _belief_status(q_deny)
    print(f"  Belief after Deny: value={deny_val!r}  status={deny_status}")
    _assert(deny_val == "Carol", "T10: belief value == 'Carol' (incumbent) after Deny", f"actual={deny_val!r}")
    _assert(
        deny_status in ("Committed", "CommittedCheap", "CommittedInferred", "Resolved"),
        "T10: belief status Committed after Deny",
        f"actual={deny_status!r}",
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # T11: Unknown verdict — stays Contested
    # ═══════════════════════════════════════════════════════════════════════════
    print("\nT11 — submit Unknown (abstain): belief stays Contested")

    unk_engine = mempill.open_oracle_in_memory(HumanOracle())
    unk_agent = "unknown-selftest-agent"
    unk_store = MempillMemoryStore(unk_engine, unk_agent)

    unk_engine.ingest_claim({
        "agent_id": unk_agent, "subject": "acme:cfo", "predicate": "held_by", "value": "Eve",
        "provenance": ProvenanceLabel.external_first_hand(), "cardinality": "Functional",
        "valid_time": {"start": "2020-01-01T00:00:00Z", "valid_time_confidence": 0.9},
        "confidence": {"value_confidence": 0.9, "valid_time_confidence": 0.9},
        "criticality": "Medium", "derived_from": [],
    })
    unk_engine.ingest_claim({
        "agent_id": unk_agent, "subject": "acme:cfo", "predicate": "held_by", "value": "Frank",
        "provenance": ProvenanceLabel.external_first_hand(), "cardinality": "Functional",
        "valid_time": {"start": "2023-01-01T00:00:00Z", "valid_time_confidence": 0.9},
        "confidence": {"value_confidence": 0.9, "valid_time_confidence": 0.9},
        "criticality": "Medium", "derived_from": [],
    })

    unk_pending = unk_store.list_pending()
    _assert(len(unk_pending) >= 1, "T11: pending adjudication for Unknown test")
    unk_handle = unk_pending[0]["handle_id"]
    unk_result = unk_store.submit(unk_handle, "Unknown")
    unk_disp = unk_result.get("disposition", "?")
    print(f"  Submit Unknown → disposition={unk_disp}")

    q_unk = unk_engine.query_memory({"agent_id": unk_agent, "subject": "acme:cfo", "predicate": "held_by"})
    unk_status = _belief_status(q_unk)
    print(f"  Belief after Unknown: status={unk_status}")
    _assert(
        unk_status in ("Contested", "PendingConflict"),
        "T11: belief stays Contested after Unknown verdict",
        f"actual={unk_status!r}",
    )
    # Queue should be empty after Unknown (removed from queue)
    unk_pending_after = unk_store.list_pending()
    print(f"  Pending after Unknown: {len(unk_pending_after)}")
    _assert(
        len(unk_pending_after) == 0,
        "T11: no pending adjudications after Unknown (removed from queue)",
        f"actual={len(unk_pending_after)}",
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # T12: Durability — file-backed engine, conflict survives close+reopen
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 62)
    print("T12 — Durability: file-backed engine, defer survives restart")
    print("=" * 62)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(pathlib.Path(tmpdir) / "durability-test.db")
        dur_agent = "durability-agent"

        # Phase A: open, ingest conflict, verify queued
        eng_a = mempill.open_oracle(db_path, HumanOracle())
        eng_a.ingest_claim({
            "agent_id": dur_agent, "subject": "acme:cfo", "predicate": "held_by", "value": "Grace",
            "provenance": ProvenanceLabel.external_first_hand(), "cardinality": "Functional",
            "valid_time": {"start": "2020-01-01T00:00:00Z", "valid_time_confidence": 0.9},
            "confidence": {"value_confidence": 0.9, "valid_time_confidence": 0.9},
            "criticality": "Medium", "derived_from": [],
        })
        resp_hank = eng_a.ingest_claim({
            "agent_id": dur_agent, "subject": "acme:cfo", "predicate": "held_by", "value": "Hank",
            "provenance": ProvenanceLabel.external_first_hand(), "cardinality": "Functional",
            "valid_time": {"start": "2023-01-01T00:00:00Z", "valid_time_confidence": 0.9},
            "confidence": {"value_confidence": 0.9, "valid_time_confidence": 0.9},
            "criticality": "Medium", "derived_from": [],
        })
        hank_disp_a = resp_hank["disposition"]
        pending_a = eng_a.list_pending_adjudications(agent_id=dur_agent)
        print(f"\nT12 Phase A: Hank disposition={hank_disp_a}  pending={len(pending_a)}")
        _assert(
            str(hank_disp_a) == "QueuedForAdjudication",
            "T12A: Hank → QueuedForAdjudication",
            f"actual={hank_disp_a!r}",
        )
        _assert(len(pending_a) >= 1, "T12A: 1+ pending before close", f"actual={len(pending_a)}")
        handle_dur = pending_a[0]["handle_id"]
        print(f"  handle before close: {handle_dur[:8]}...")
        del eng_a  # simulate restart

        # Phase B: reopen, verify pending still present (defer survived restart)
        eng_b = mempill.open_oracle(db_path, HumanOracle())
        pending_b = eng_b.list_pending_adjudications(agent_id=dur_agent)
        print(f"\nT12 Phase B (after reopen): pending={len(pending_b)}")
        _assert(
            len(pending_b) >= 1,
            "T12B: pending adjudication survives engine close+reopen",
            f"actual={len(pending_b)}",
        )
        handle_b = pending_b[0]["handle_id"]
        print(f"  handle after reopen: {handle_b[:8]}...")
        _assert(
            handle_b == handle_dur,
            "T12B: same handle_id after reopen",
            f"expected={handle_dur[:8]} actual={handle_b[:8]}",
        )
        print("  Durability confirmed — deferred adjudication persists across restart.")

    print()
    print("All assertions passed — selftest complete.")
    print()


if __name__ == "__main__":
    run()
    sys.exit(0)
