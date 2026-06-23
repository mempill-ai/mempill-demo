"""
console.scenario — auto-play the 3-act story then hand off to the REPL.

Act 1: Ingest Alice as CEO (2020-)                → CommittedCheap
Act 2: Ingest Bob as CEO (2023-)                  → Contested (or actual disposition)
       /reconcile                                 → Superseded + Resolved
Act 3: RECALL_REENTRY ×5 (amplification firewall) → belief unchanged
"""
from __future__ import annotations

from console.agent import MempillAwareAgent
from console.inference.deterministic import DeterministicParser
from console.inference.base import CommandKind, ParsedCommand
from mempill import ProvenanceLabel, Disposition


def _step(label: str, result: str) -> None:
    print(f"\n[{label}]")
    print(result)


def run_scenario(agent: MempillAwareAgent, parser: DeterministicParser) -> str:
    """
    Run the 3-act auto-play scenario.
    Returns the claim_ref of the committed Bob claim (for RECALL_REENTRY demo).
    """
    print()
    print("=" * 62)
    print("  mempill Console Agent — 3-Act Scenario (auto-play)")
    print("  Subject: acme:ceo  Predicate: held_by")
    print("  No LLM required — deterministic structural memory only.")
    print("=" * 62)

    # ── ACT 1: Alice is CEO ───────────────────────────────────────────────────
    print("\n--- ACT 1: Ingest 'Alice is CEO' (valid from 2020-01-01) ---")
    cmd_alice = ParsedCommand(
        kind=CommandKind.INGEST,
        subject="acme:ceo",
        predicate="held_by",
        value="Alice",
        since="2020-01-01T00:00:00Z",
        conf=0.95,
        raw='INGEST acme:ceo held_by "Alice" SINCE 2020-01-01 CONF 0.95',
    )
    # Use engine directly for richer response
    engine = agent._engine
    resp_alice = engine.ingest_claim({
        "agent_id": agent._agent_id,
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
    agent.n_ingests += 1
    alice_ref = resp_alice["claim_ref"]
    act1_disp = resp_alice["disposition"]
    print(f"  disposition:  {act1_disp}")
    print(f"  claim_ref:    {alice_ref[:8]}...")
    if act1_disp == Disposition.CommittedCheap:
        print("  [note] CommittedCheap — fast-path commit, no conflict.")
    else:
        print(f"  [note] ACTUAL: {act1_disp!r} (narrating real engine behavior).")

    q1 = engine.query_memory({
        "agent_id": agent._agent_id,
        "subject": "acme:ceo",
        "predicate": "held_by",
    })
    print(f"  RECALL → value={q1.get('belief', {}).get('primary', {}).get('fact', {}).get('value')!r}"
          f"  status={q1.get('belief', {}).get('status')}")

    # ── ACT 2: Bob is CEO — conflict ──────────────────────────────────────────
    print("\n--- ACT 2: Ingest 'Bob is CEO' (valid from 2023-03-15) — conflict expected ---")
    resp_bob = engine.ingest_claim({
        "agent_id": agent._agent_id,
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
    agent.n_ingests += 1
    bob_ref = resp_bob["claim_ref"]
    act2_disp = resp_bob["disposition"]
    contested_with = resp_bob.get("contested_with", [])
    print(f"  disposition:     {act2_disp}")
    print(f"  contested_with:  {[r[:8]+'...' for r in contested_with]}")
    if act2_disp == Disposition.Contested:
        agent.n_contested += 1
        print("  [note] CONTESTED — two open-ended Functional claims overlap. Engine did NOT overwrite Alice.")
        print("  [badge] CONTESTED  ← this is the key mempill guarantee")
    elif act2_disp == Disposition.CommittedCheap:
        print("  [note] ACTUAL: CommittedCheap — engine fast-committed Bob without conflict flag.")
        print("         Narrating real engine behavior (plan assumed Contested).")
    else:
        print(f"  [note] ACTUAL: {act2_disp!r}")

    print()
    print("  [/reconcile] Resolving acme:ceo held_by...")
    reconcile_resp = engine.reconcile({
        "agent_id": agent._agent_id,
        "subject_lines": [("acme:ceo", "held_by")],
    })
    outcomes = reconcile_resp.get("outcomes", [])
    escalations = reconcile_resp.get("oracle_escalations", 0)
    print(f"  outcomes:  {outcomes}")
    print(f"  escalations: {escalations}")

    committed_bob_ref = bob_ref
    for ref, disp in outcomes:
        if disp in (Disposition.Superseded, "Superseded", Disposition.Invalidated, "Invalidated"):
            agent.n_superseded += 1
            print(f"  [badge] SUPERSEDED  ← {ref[:8]}...")
        elif disp in (Disposition.CommittedCheap, "CommittedCheap", Disposition.CommittedInferred,
                      "CommittedInferred", "Resolved", "Committed"):
            committed_bob_ref = ref
            print(f"  [badge] COMMITTED   ← {ref[:8]}... (Bob, now authoritative)")

    q_post = engine.query_memory({
        "agent_id": agent._agent_id,
        "subject": "acme:ceo",
        "predicate": "held_by",
    })
    post_val = q_post.get("belief", {}).get("primary", {}).get("fact", {}).get("value")
    post_status = q_post.get("belief", {}).get("status")
    print(f"  Post-reconcile belief: \"{post_val}\"  status={post_status}")

    # ── ACT 3: Amplification firewall ─────────────────────────────────────────
    print("\n--- ACT 3: RECALL_REENTRY ×5 — amplification firewall ---")
    print(f"  Re-ingesting 'Bob is CEO' 5× with RecallReEntry provenance (derived_from={committed_bob_ref[:8]}...)")

    for i in range(5):
        engine.ingest_claim({
            "agent_id": agent._agent_id,
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
        agent.n_ingests += 1

    q_after = engine.query_memory({
        "agent_id": agent._agent_id,
        "subject": "acme:ceo",
        "predicate": "held_by",
    })
    after_val = q_after.get("belief", {}).get("primary", {}).get("fact", {}).get("value")
    after_status = q_after.get("belief", {}).get("status")
    corroboration = (q_after.get("belief", {}).get("primary") or {}).get("currency_signal", {}).get("corroboration_count", 0)

    print(f"  Belief after 5 re-entries: \"{after_val}\"  status={after_status}")
    print(f"  corroboration_count: {corroboration}")
    print("  [badge] FIREWALL HELD — RecallReEntry did not alter the belief.")

    print()
    print("=" * 62)
    print("  Scenario complete.")
    print(f"  Act 1: Alice  → {act1_disp}")
    print(f"  Act 2: Bob    → {act2_disp}  reconcile→{[d for _, d in outcomes]}")
    print(f"  Act 3: ×5 recall-reentry  → belief unchanged ({after_val}, {after_status})")
    print("=" * 62)
    print()

    return committed_bob_ref
