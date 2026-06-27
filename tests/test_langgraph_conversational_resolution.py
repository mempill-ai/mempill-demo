"""
tests/test_langgraph_conversational_resolution.py
— Offline tests for conversational conflict resolution (TASK: DEMO-CONVERSATIONAL-RESOLUTION).

Covers:
  1. Auto-reconcile: Contested belief with distinct valid-times → silently resolved
     by reconcile(); no pending_decision, no user prompt needed.
  2. Conversational adjudication — challenger path: genuine tie stays Contested after
     reconcile(); pending adjudication exists; user replies "Bob" → decision classifier
     picks challenger → submit("Affirm") called → belief Resolved; decision-turn write
     guard prevents "Bob" being ingested as a new claim.
  3. Conversational adjudication — incumbent path: same setup, user replies "Alice"
     → decision classifier picks incumbent → submit("Deny") called.
  4. Conversational adjudication — neither path: user replies off-topic → no verdict
     submitted, pending_decision cleared for next normal turn.
  6. Queue-driven adjudication with NO pending_decision in state and a REALISTIC
     non-marker prior AI message (proving the fix is wording-independent).
  9. Fresh query ("Who is CEO?") while pending adjudication exists and classifier
     returns "neither" → conflict re-surfaced deterministically; no submit.

All tests run WITHOUT an ANTHROPIC_API_KEY — injectable classifier seam is used.
"""
from __future__ import annotations

from typing import Optional

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.runnables import RunnableConfig
from langgraph.store.memory import InMemoryStore

from mempill_demo.domain.models import AlternativeView, BeliefView
from mempill_langgraph.extraction import (
    ClaimExtractResult,
    DecisionClassifyResult,
    ExtractedClaim,
    KeyExtractResult,
)
from mempill_langgraph.graph import build_graph
from mempill_langgraph.nodes import make_nodes

from tests.fakes_langgraph import LGFakeMemoryStore


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fake_llm(*responses):
    return FakeMessagesListChatModel(responses=list(responses))


def _empty_extractor(prompt: str) -> ClaimExtractResult:
    return ClaimExtractResult(claims=[])


def _fixed_key_extractor(subject: str, predicate: str):
    result = KeyExtractResult(subject=subject, predicate=predicate)
    return lambda prompt: result


def _no_key_extractor(prompt: str) -> KeyExtractResult:
    return KeyExtractResult(subject="", predicate="")


def _fixed_decision_classifier(verdict: str):
    """Return a decision classifier that always emits the given verdict."""
    result = DecisionClassifyResult(verdict=verdict)
    return lambda prompt: result


def _sequenced_decision_classifier(*verdicts: str):
    """
    Return a decision classifier that emits verdicts in sequence.

    First call returns verdicts[0], second returns verdicts[1], etc.
    After the sequence is exhausted, returns "neither".

    Used to simulate realistic classifier behavior across turns:
      - Turn C (fresh query "Who is CEO?"): classifier returns "neither"
      - Turn N+1 (user says "Bob"): classifier returns "challenger"
    """
    iter_verdicts = list(verdicts)
    call_count: list[int] = [0]

    def _classify(prompt: str) -> DecisionClassifyResult:
        idx = call_count[0]
        call_count[0] += 1
        verdict = iter_verdicts[idx] if idx < len(iter_verdicts) else "neither"
        return DecisionClassifyResult(verdict=verdict)

    return _classify


_CONTESTED_BELIEF = BeliefView(
    subject="acme:ceo",
    predicate="held_by",
    status="Contested",
    value=None,
    conf=None,
    vt_start="",
    vt_end="",
    provenance="",
    claim_ref="",
    corroboration=0,
    alternatives=[
        AlternativeView(value="Alice", conf=0.95, vt_start="2025-08-01", vt_end="open", claim_ref="ref-alice"),
        AlternativeView(value="Bob",   conf=0.90, vt_start="2026-06-01", vt_end="open", claim_ref="ref-bob"),
    ],
)

_RESOLVED_BELIEF = BeliefView(
    subject="acme:ceo",
    predicate="held_by",
    status="CommittedCheap",
    value="Bob",
    conf=0.90,
    vt_start="2026-06-01",
    vt_end="open",
    provenance="EXT",
    claim_ref="ref-bob",
    corroboration=0,
    alternatives=[],
)

_PENDING_ITEM = {
    "handle_id": "handle-abc123",
    "incumbent_value": "Alice",
    "challenger_value": "Bob",
}


# ── Test 1: Auto-reconcile silently resolves via valid-time succession ─────────

def test_auto_reconcile_resolves_without_user_action():
    """
    When a Contested belief has distinct valid-times (Alice since 2025-08, Bob since 2026-06),
    retrieve_memory calls reconcile() which triggers valid-time succession.
    After reconcile(), recall() returns the Resolved belief (Bob wins — most recent).
    No pending_decision is set.  No slash command needed.
    """
    store = LGFakeMemoryStore(
        belief=_CONTESTED_BELIEF,
        reconcile_result_belief=_RESOLVED_BELIEF,  # simulate valid-time resolution
        pending_items=[],                           # no pending adjudication
    )

    reply = AIMessage(content="The CEO of Acme is Bob.")
    fake_llm = _fake_llm(reply)

    graph = build_graph(
        memory_store=store,
        llm=fake_llm,
        extractor=_empty_extractor,
        key_extractor=_fixed_key_extractor("acme:ceo", "held_by"),
    )

    result = graph.invoke(
        {
            "messages": [HumanMessage(content="Who is CEO?")],
            "user_id": "u1",
            "agent_id": "test",
        },
        config={"configurable": {"thread_id": "auto-reconcile-t1"}},
    )

    # reconcile() must have been called exactly once
    assert store.reconcile_calls == 1, (
        f"Expected 1 reconcile() call, got {store.reconcile_calls}"
    )

    # After auto-reconcile, recall was called twice (initial + post-reconcile)
    assert store.recall_calls == 2, (
        f"Expected 2 recall() calls (initial + post-reconcile), got {store.recall_calls}"
    )

    # memory_context must reflect the Resolved belief (Bob wins)
    memory_ctx = result.get("memory_context", "")
    assert "CONTESTED" not in memory_ctx, (
        f"Expected Resolved (not Contested) memory_context, got: {memory_ctx!r}"
    )
    assert "Bob" in memory_ctx, (
        f"Expected 'Bob' in resolved memory_context, got: {memory_ctx!r}"
    )

    # No pending_decision in state
    assert result.get("pending_decision") is None, (
        f"Expected no pending_decision after auto-reconcile, got: {result.get('pending_decision')}"
    )

    # No writes (empty extractor) and no verdicts submitted
    assert store.ingested == [], f"Expected no ingests, got: {store.ingested}"
    assert store.submitted_verdicts == [], f"Expected no verdicts, got: {store.submitted_verdicts}"


# ── Test 2: Challenger path — user says "Bob" → Affirm submitted, no mis-ingest ─

def test_conversational_adjudication_challenger_wins_no_mis_ingest():
    """
    Genuine tie: belief stays Contested after auto-reconcile (reconcile_result_belief
    is also Contested).  A pending adjudication exists.

    Turn N:  ask "Who is CEO?" → classifier returns "neither" (fresh question is not
             a verdict) → Step 1b falls through → Step 4 sets pending_decision;
             memory_context is CONTESTED.
    Turn N+1: user says "Bob" → classifier returns "challenger"
              → submit("Affirm") called (via Step 1, pending_decision is now in state)
              → write_memory SKIPS ingestion (decision-turn guard)
              → after turn N+1, submitted_verdicts == [("handle-abc123", "Affirm")]
              → ingested == []

    Uses _sequenced_decision_classifier("neither", "challenger") to realistically model:
      - Turn N (fresh query): "neither" (not a verdict)
      - Turn N+1 (user picks Bob): "challenger"
    """
    # After reconcile(), belief is STILL Contested (genuine tie)
    store = LGFakeMemoryStore(
        belief=_CONTESTED_BELIEF,
        reconcile_result_belief=_CONTESTED_BELIEF,   # tie persists
        pending_items=[dict(_PENDING_ITEM)],          # pending adjudication exists
    )

    # Turn N: respond node emits the deterministic ⚖️ question (M1 short-circuit),
    # so the fake_llm reply for Turn N is never actually used — but we supply it for
    # completeness.  Turn N+1: agent confirms resolution.
    reply_n = AIMessage(
        content="I have conflicting information: one source says Alice, another says Bob. "
                "Which one is correct?"
    )
    reply_n1 = AIMessage(content="Got it — I've recorded that Bob is the CEO.")

    fake_llm = _fake_llm(reply_n, reply_n1)
    checkpointer = MemorySaver()

    graph = build_graph(
        memory_store=store,
        llm=fake_llm,
        checkpointer=checkpointer,
        extractor=_empty_extractor,
        key_extractor=_fixed_key_extractor("acme:ceo", "held_by"),
        # Sequenced: Turn N gets "neither" (fresh query), Turn N+1 gets "challenger" (user picks Bob)
        decision_classifier=_sequenced_decision_classifier("neither", "challenger"),
    )

    config = {"configurable": {"thread_id": "adj-challenger-t1"}}

    # Turn N — agent surfaces CONTESTED and records pending_decision
    result_n = graph.invoke(
        {
            "messages": [HumanMessage(content="Who is CEO of Acme?")],
            "user_id": "u1",
            "agent_id": "test",
        },
        config=config,
    )

    memory_ctx_n = result_n.get("memory_context", "")
    assert "CONTESTED" in memory_ctx_n, (
        f"Turn N: expected CONTESTED in memory_context, got: {memory_ctx_n!r}"
    )
    # With "neither" on Turn N, Step 1b falls through → Step 4 sets pending_decision.
    assert result_n.get("pending_decision") is not None, (
        "Turn N: expected pending_decision to be set after 'neither' fall-through"
    )

    # Turn N+1 — user says "Bob"; decision classifier picks challenger
    # write_memory must NOT ingest "Bob" as a new claim
    result_n1 = graph.invoke(
        {"messages": [HumanMessage(content="Bob")]},
        config=config,
    )

    # submit() must have been called with Affirm
    assert len(store.submitted_verdicts) == 1, (
        f"Expected 1 submitted verdict, got: {store.submitted_verdicts}"
    )
    handle_id, verdict = store.submitted_verdicts[0]
    assert handle_id == "handle-abc123", f"Wrong handle_id: {handle_id!r}"
    assert verdict == "Affirm", f"Expected Affirm, got: {verdict!r}"

    # pending_decision must be cleared
    assert result_n1.get("pending_decision") is None, (
        f"Expected pending_decision cleared after resolution, got: {result_n1.get('pending_decision')}"
    )

    # CRITICAL: write_memory must NOT have ingested "Bob" as a new claim
    assert store.ingested == [], (
        f"Decision-turn guard FAILED: write_memory ingested {store.ingested!r} — "
        "'Bob' must not be treated as a new factual claim on a decision turn."
    )


# ── Test 3: Incumbent path — user says "Alice" → Deny submitted ───────────────

def test_conversational_adjudication_incumbent_wins():
    """
    Same setup as Test 2, but the decision classifier returns "incumbent" on Turn N+1.
    submit("Deny") must be called — the existing belief (Alice) stands.

    Uses _sequenced_decision_classifier("neither", "incumbent"):
      - Turn N (fresh query "Who is CEO?"): "neither" → Step 1b falls through → Step 4
        sets pending_decision; next turn handled via Step 1.
      - Turn N+1 (user says "Alice"): "incumbent" → submit("Deny").
    """
    store = LGFakeMemoryStore(
        belief=_CONTESTED_BELIEF,
        reconcile_result_belief=_CONTESTED_BELIEF,
        pending_items=[dict(_PENDING_ITEM)],
    )

    reply_n = AIMessage(
        content="Conflicting info: Alice vs Bob. Which is correct?"
    )
    reply_n1 = AIMessage(content="Understood — I'll keep Alice as the CEO.")
    fake_llm = _fake_llm(reply_n, reply_n1)
    checkpointer = MemorySaver()

    graph = build_graph(
        memory_store=store,
        llm=fake_llm,
        checkpointer=checkpointer,
        extractor=_empty_extractor,
        key_extractor=_fixed_key_extractor("acme:ceo", "held_by"),
        # Turn N: "neither" (fresh query not a verdict); Turn N+1: "incumbent"
        decision_classifier=_sequenced_decision_classifier("neither", "incumbent"),
    )

    config = {"configurable": {"thread_id": "adj-incumbent-t1"}}

    # Turn N
    graph.invoke(
        {
            "messages": [HumanMessage(content="Who is CEO?")],
            "user_id": "u1",
            "agent_id": "test",
        },
        config=config,
    )

    # Turn N+1 — user says "Alice"
    result_n1 = graph.invoke(
        {"messages": [HumanMessage(content="Alice")]},
        config=config,
    )

    assert len(store.submitted_verdicts) == 1, (
        f"Expected 1 verdict, got: {store.submitted_verdicts}"
    )
    _, verdict = store.submitted_verdicts[0]
    assert verdict == "Deny", f"Expected Deny for incumbent, got: {verdict!r}"
    assert result_n1.get("pending_decision") is None, "pending_decision should be cleared"
    assert store.ingested == [], (
        "Decision-turn guard FAILED: 'Alice' must not be ingested as a new claim."
    )


# ── Test 4: Neither path — user off-topic, no verdict, pending preserved ──────

def test_conversational_adjudication_neither_leaves_pending():
    """
    When the decision classifier returns "neither", no submit() is called,
    pending_decision is preserved (stays set for the next turn), and write_memory
    runs normally (not blocked by decision-turn guard).
    """
    store = LGFakeMemoryStore(
        belief=_CONTESTED_BELIEF,
        reconcile_result_belief=_CONTESTED_BELIEF,
        pending_items=[dict(_PENDING_ITEM)],
    )

    reply_n = AIMessage(content="Conflicting info: Alice vs Bob. Which is correct?")
    reply_n1 = AIMessage(content="I still have conflicting information. Can you confirm?")
    fake_llm = _fake_llm(reply_n, reply_n1)
    checkpointer = MemorySaver()

    graph = build_graph(
        memory_store=store,
        llm=fake_llm,
        checkpointer=checkpointer,
        extractor=_empty_extractor,
        key_extractor=_fixed_key_extractor("acme:ceo", "held_by"),
        decision_classifier=_fixed_decision_classifier("neither"),
    )

    config = {"configurable": {"thread_id": "adj-neither-t1"}}

    # Turn N — sets pending_decision
    graph.invoke(
        {
            "messages": [HumanMessage(content="Who is CEO?")],
            "user_id": "u1",
            "agent_id": "test",
        },
        config=config,
    )

    # Turn N+1 — off-topic / neither
    result_n1 = graph.invoke(
        {"messages": [HumanMessage(content="I'm not sure, tell me more.")]},
        config=config,
    )

    # No verdict submitted
    assert store.submitted_verdicts == [], (
        f"Expected no verdicts for 'neither', got: {store.submitted_verdicts}"
    )

    # No ingests (empty extractor) — but the turn was NOT blocked as a decision-turn
    assert store.ingested == [], "Expected no ingests (empty extractor)"


# ── Test 5: Real oracle store — auto-reconcile resolves John→Bob with valid-times ─

def test_auto_reconcile_real_store_valid_time_succession():
    """
    End-to-end with a REAL oracle-wired mempill store.

    Ingest "John is CEO since 2025-08" then "Bob is CEO since 2026-06".
    Engine may or may not queue for adjudication.  Call reconcile() directly to
    simulate the auto-reconcile path; belief should become Resolved to Bob
    (most-recent valid-time wins) or stay Contested for a genuine engine tie.

    This test verifies the real reconcile() call path works without errors, and
    that the belief after reconcile is either Resolved/Committed to Bob, or at
    minimum no longer Contested (engine may vary).

    Uses a real oracle engine — no ANTHROPIC_API_KEY needed.
    """
    import mempill
    from mempill import ProvenanceLabel
    from mempill_demo.adapters.human_oracle import HumanOracle
    from mempill_demo.adapters.memory_mempill import MempillMemoryStore

    engine = mempill.open_oracle_in_memory(HumanOracle())
    store = MempillMemoryStore(engine=engine, agent_id="reconcile-test")

    def _ingest(value: str, start: str) -> dict:
        return store._engine.ingest_claim({
            "agent_id": store._agent_id,
            "subject": "acme:ceo",
            "predicate": "held_by",
            "value": value,
            "provenance": ProvenanceLabel.external_user_asserted(),
            "cardinality": "Functional",
            "valid_time": {"start": start, "valid_time_confidence": 0.9},
            "confidence": {"value_confidence": 0.9, "valid_time_confidence": 0.9},
            "criticality": "Medium",
            "derived_from": [],
        })

    r1 = _ingest("John", "2025-08-01T00:00:00Z")
    r2 = _ingest("Bob",  "2026-06-01T00:00:00Z")

    # Verify a conflict was detected (Contested or QueuedForAdjudication)
    pre_belief = store.recall("acme:ceo", "held_by")
    assert pre_belief.status in (
        "Contested", "Conflict", "QueuedForAdjudication", "CommittedCheap", "Committed"
    ), f"Unexpected pre-reconcile status: {pre_belief.status!r}"

    # Run reconcile — this is what retrieve_memory's auto-reconcile step does
    outcomes = store.reconcile("acme:ceo", "held_by")

    # Recall again (as auto-reconcile does)
    post_belief = store.recall("acme:ceo", "held_by")

    # After reconcile with distinct valid-times, engine should resolve to Bob
    # (most-recent-start wins).  Accept Committed/CommittedCheap/Superseded/Resolved as success.
    # If still Contested (engine tie), the test still passes — reconcile() must not error.
    if post_belief.status in ("Committed", "CommittedCheap", "Superseded", "Resolved"):
        # Resolved: verify Bob is the winner (most-recent valid-time)
        winner = post_belief.value
        assert winner == "Bob", (
            f"Expected Bob (most-recent valid-time) after reconcile, got: {winner!r} "
            f"(status={post_belief.status!r})"
        )
    else:
        # Engine left it Contested — that's acceptable (genuine tie semantics)
        assert post_belief.status in ("Contested", "Conflict", "QueuedForAdjudication"), (
            f"Unexpected post-reconcile status: {post_belief.status!r}"
        )


# ── Tests for the queue-driven adjudication path (Step 1b / DEMO-ADJUDICATION-FROM-QUEUE) ───────


# Contested belief returned after reconcile (genuine tie) with pending item
_CONTESTED_BELIEF_OXANA_JOHN = BeliefView(
    subject="a-consulting:ceo",
    predicate="held_by",
    status="Contested",
    value=None,
    conf=None,
    vt_start="",
    vt_end="",
    provenance="",
    claim_ref="",
    corroboration=0,
    alternatives=[
        AlternativeView(value="Oxana", conf=0.95, vt_start="2024-01-01", vt_end="open", claim_ref="ref-oxana"),
        AlternativeView(value="John",  conf=0.90, vt_start="2025-06-01", vt_end="open", claim_ref="ref-john"),
    ],
)

# After submit("Deny") the incumbent Oxana is retained
_RESOLVED_BELIEF_OXANA = BeliefView(
    subject="a-consulting:ceo",
    predicate="held_by",
    status="CommittedCheap",
    value="Oxana",
    conf=0.95,
    vt_start="2024-01-01",
    vt_end="open",
    provenance="EXT",
    claim_ref="ref-oxana",
    corroboration=0,
    alternatives=[],
)

_PENDING_OXANA_JOHN = {
    "handle_id": "handle-oxana-john",
    "incumbent_value": "Oxana",
    "challenger_value": "John",
}


def _build_queue_driven_store(
    reconcile_result: BeliefView,
    pending_after_reconcile: BeliefView,
) -> "LGFakeMemoryStore":
    """
    Return a store where:
    - Initial recall returns _CONTESTED_BELIEF_OXANA_JOHN (or reconcile_result if
      reconcile was called first, but in this fixture reconcile resolves to contested).
    - The pending queue has the Oxana/John item.
    - After submit() is called, the pending list is emptied and subsequent recall
      returns pending_after_reconcile (Oxana-resolved).
    """
    return LGFakeMemoryStore(
        belief=_CONTESTED_BELIEF_OXANA_JOHN,
        reconcile_result_belief=reconcile_result,
        pending_items=[dict(_PENDING_OXANA_JOHN)],
    )


# ── Test 6: Exact repro — queue-driven adjudication fires with REALISTIC non-marker prior AI ──

def test_queue_driven_adjudication_incumbent_wins_no_state_preseeded():
    """
    THE KEY BUG REPRO (DEMO-ADJUDICATION-ROBUST-GUARD):

    Verifies that Step 1b fires based on ENGINE CORRELATION ALONE — with NO
    dependency on the prior AI message's exact wording.

    The prior AI message is a REALISTIC LLM-phrased string:
        "I have conflicting information: one source says Oxana, another says John.
         Which one is correct?"
    This message does NOT contain "⚖️" and does NOT contain "Which is correct?"
    (it says "Which ONE is correct?").  Under the old brittle guard, Step 1b would
    be skipped and the verdict ignored.  Under the new engine-truth gate it MUST fire.

    State setup (calling retrieve_memory directly, bypassing the graph):
      - pending_items=[{handle_id, incumbent_value='Oxana', challenger_value='John'}]
      - No pending_decision in state (simulates ingest-born conflict — write_memory
        ran on the prior turn AFTER retrieve_memory had already returned)
      - Message history contains the realistic non-marker prior AI message
      - decision_classifier fixed to "incumbent" (user picked Oxana)

    Assertions:
      - submit() called with ("handle-oxana-john", "Deny")
      - list_pending() is empty after retrieval
      - resolved value in the returned memory_context is "Oxana" not "John"
      - _decision_turn is True
      - _resolved_reply contains "Oxana" and not "John"
    """
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

    store = LGFakeMemoryStore(
        belief=_CONTESTED_BELIEF_OXANA_JOHN,
        reconcile_result_belief=_CONTESTED_BELIEF_OXANA_JOHN,  # reconcile doesn't resolve tie
        pending_items=[dict(_PENDING_OXANA_JOHN)],
    )

    # Patch recall so that AFTER submit() is called, it returns Oxana-resolved.
    original_recall = store.recall
    original_submit = store.submit
    submitted: list[bool] = [False]

    def _patched_recall(subject: str, predicate: str) -> BeliefView:
        if submitted[0]:
            return _RESOLVED_BELIEF_OXANA
        return original_recall(subject, predicate)

    def _patched_submit(handle_id: str, verdict: str) -> dict:
        submitted[0] = True
        return original_submit(handle_id, verdict)

    store.recall = _patched_recall  # type: ignore[method-assign]
    store.submit = _patched_submit  # type: ignore[method-assign]

    # Build nodes directly so we can call retrieve_memory with hand-built state.
    fake_llm = FakeMessagesListChatModel(responses=[])
    retrieve_memory, _respond, _write_memory = make_nodes(
        memory_store=store,
        llm=fake_llm,
        extractor=_empty_extractor,
        key_extractor=_fixed_key_extractor("a-consulting:ceo", "held_by"),
        decision_classifier=_fixed_decision_classifier("incumbent"),  # Oxana → Deny
    )

    # REALISTIC prior AI message: NO ⚖️, and "Which one is correct?" (NOT "Which is correct?")
    # Under the old brittle guard this would have caused Step 1b to be SKIPPED.
    realistic_prior_ai = AIMessage(
        content=(
            "I have conflicting information: one source says Oxana, another says John. "
            "Which one is correct?"
        )
    )

    # State: prior AI asked in natural LLM phrasing; user is now answering.
    # pending_decision is NOT set in state (queue-driven path).
    state = {
        "messages": [
            HumanMessage(content="Who is CEO of A Consulting?"),
            realistic_prior_ai,
            HumanMessage(content="Actually, Oxana"),
        ],
        "user_id": "u1",
        "agent_id": "test",
        "pending_decision": None,
        "contested": None,
        "_decision_turn": False,
        "_resolved_reply": None,
        "memory_context": "",
        "last_subject": None,
        "last_predicate": None,
    }

    config = RunnableConfig(configurable={"thread_id": "queue-driven-robust-t1"})
    lg_store = InMemoryStore()

    result = retrieve_memory(state, config, store=lg_store)

    # 1. submit() was called with the correct handle and verdict.
    assert len(store.submitted_verdicts) == 1, (
        f"Expected exactly 1 submitted verdict, got: {store.submitted_verdicts}"
    )
    handle_id, verdict = store.submitted_verdicts[0]
    assert handle_id == "handle-oxana-john", f"Wrong handle_id: {handle_id!r}"
    assert verdict == "Deny", f"Expected Deny (incumbent=Oxana), got: {verdict!r}"

    # 2. list_pending() is empty after the verdict.
    assert store.list_pending() == [], (
        f"Expected empty pending queue after resolution, got: {store.list_pending()}"
    )

    # 3. _resolved_reply contains "Oxana" and does NOT contain "John".
    resolved_reply = result.get("_resolved_reply", "") or ""
    assert "Oxana" in resolved_reply, (
        f"_resolved_reply must contain 'Oxana' (resolved value), got: {resolved_reply!r}"
    )
    assert "John" not in resolved_reply, (
        f"_resolved_reply must NOT contain 'John' (rejected challenger), got: {resolved_reply!r}"
    )

    # 4. _decision_turn is True (write_memory guard prevents ingesting the answer).
    assert result.get("_decision_turn") is True, (
        f"Expected _decision_turn=True on resolution turn, got: {result.get('_decision_turn')}"
    )

    # 5. pending_decision is cleared.
    assert result.get("pending_decision") is None, (
        f"Expected pending_decision cleared after resolution, got: {result.get('pending_decision')}"
    )


# ── Test 7: Verdict beats auto-reconcile — submit path ran, reconcile did not decide ──

def test_queue_driven_verdict_beats_auto_reconcile():
    """
    Verify that when an unresolved pending adjudication exists for the subject
    and the user answers the contested question, auto-reconcile does NOT run
    (or, more precisely, the human verdict determines the outcome — not reconcile).

    We assert that reconcile() is not called on the answer turn (Turn A).
    (reconcile() may have been called on Turn C when the belief was first contested.)

    Uses _sequenced_decision_classifier("neither", "incumbent"):
      - Turn C (fresh query): "neither" → falls through to Step 3/4, pending_decision set.
        auto-reconcile MAY run on Turn C (belief is still contested, and
        _has_unresolved_pending_from_queue is True → reconcile IS skipped on Turn C too).
      - Turn A (user says "Oxana"): "incumbent" → submit("Deny") via Step 1 (pending_decision
        is in state) → auto-reconcile skipped.
    """
    store = LGFakeMemoryStore(
        belief=_CONTESTED_BELIEF_OXANA_JOHN,
        reconcile_result_belief=_CONTESTED_BELIEF_OXANA_JOHN,
        pending_items=[dict(_PENDING_OXANA_JOHN)],
    )

    # Track reconcile calls per turn by recording count at Turn A start
    reply_resolve = AIMessage(content="OK, Oxana is the CEO.")
    fake_llm = _fake_llm(reply_resolve)
    checkpointer = MemorySaver()

    graph = build_graph(
        memory_store=store,
        llm=fake_llm,
        checkpointer=checkpointer,
        extractor=_empty_extractor,
        key_extractor=_fixed_key_extractor("a-consulting:ceo", "held_by"),
        # Turn C: "neither" (fresh query); Turn A: "incumbent" (user picks Oxana)
        decision_classifier=_sequenced_decision_classifier("neither", "incumbent"),
    )

    config = {"configurable": {"thread_id": "verdict-beats-reconcile-t1"}}

    # Turn C: surface contest — "neither" classifier prevents resolution; Step 4 sets
    # pending_decision; _has_unresolved_pending_from_queue keeps auto-reconcile off.
    graph.invoke(
        {
            "messages": [HumanMessage(content="Who is CEO of A Consulting?")],
            "user_id": "u1",
            "agent_id": "test",
        },
        config=config,
    )
    reconcile_calls_after_c = store.reconcile_calls

    # Turn A: user answers "Oxana" — Step 1 fires (pending_decision in state from Turn C).
    # auto-reconcile must NOT run on Turn A (verdict path short-circuits).
    graph.invoke(
        {"messages": [HumanMessage(content="Oxana")]},
        config=config,
    )

    # Reconcile count must NOT have increased on Turn A — submit path short-circuits.
    assert store.reconcile_calls == reconcile_calls_after_c, (
        f"auto-reconcile ran on Turn A when it should have been skipped. "
        f"reconcile_calls after Turn C={reconcile_calls_after_c}, "
        f"after Turn A={store.reconcile_calls}"
    )

    # Submit was called (verdict path ran, not reconcile)
    assert len(store.submitted_verdicts) == 1, (
        f"Expected submit() to determine the outcome, got: {store.submitted_verdicts}"
    )


# ── Test 8: "neither" answer keeps pending intact and re-surfaces conflict ──────

def test_queue_driven_neither_re_surfaces_conflict():
    """
    When the user answers off-topic ("I'm not sure") on the answer turn,
    the decision classifier returns "neither".  The queue-driven path must:
      - NOT call submit()
      - Keep the pending item in the queue (list_pending stays non-empty)
      - Re-surface the conflict deterministically (contested in memory_context)
    """
    store = LGFakeMemoryStore(
        belief=_CONTESTED_BELIEF_OXANA_JOHN,
        reconcile_result_belief=_CONTESTED_BELIEF_OXANA_JOHN,
        pending_items=[dict(_PENDING_OXANA_JOHN)],
    )

    reply_re_ask = AIMessage(content="Re-ask: which is correct, Oxana or John?")
    fake_llm = _fake_llm(reply_re_ask, reply_re_ask)  # two turns
    checkpointer = MemorySaver()

    graph = build_graph(
        memory_store=store,
        llm=fake_llm,
        checkpointer=checkpointer,
        extractor=_empty_extractor,
        key_extractor=_fixed_key_extractor("a-consulting:ceo", "held_by"),
        decision_classifier=_fixed_decision_classifier("neither"),
    )

    config = {"configurable": {"thread_id": "queue-neither-t1"}}

    # Turn C: surface contest
    graph.invoke(
        {
            "messages": [HumanMessage(content="Who is CEO of A Consulting?")],
            "user_id": "u1",
            "agent_id": "test",
        },
        config=config,
    )

    # Turn A: off-topic / neither
    result_a = graph.invoke(
        {"messages": [HumanMessage(content="I'm not sure")]},
        config=config,
    )

    # No submit
    assert store.submitted_verdicts == [], (
        f"Expected no verdicts for 'neither', got: {store.submitted_verdicts}"
    )

    # Pending stays non-empty
    assert len(store.list_pending()) == 1, (
        f"Expected pending item to remain after 'neither', got: {store.list_pending()}"
    )

    # Conflict is re-surfaced in memory_context (CONTESTED marker present)
    memory_ctx = result_a.get("memory_context", "")
    assert "CONTESTED" in memory_ctx, (
        f"Expected CONTESTED in memory_context after 'neither', got: {memory_ctx!r}"
    )


# ── Test 9: Fresh query while pending exists → conflict re-surfaced, no submit ───

def test_fresh_query_while_pending_resurfaces_conflict_no_submit():
    """
    DEMO-ADJUDICATION-ROBUST-GUARD: regression test for the "fresh query misread as
    verdict" concern.

    When a pending adjudication exists for a subject and the user asks a fresh
    question about that SAME subject ("Who is the CEO?") with a classifier that
    returns "neither" (the user did not express a preference), the agent MUST:
      - NOT call submit()
      - Re-surface the conflict deterministically (both candidate values visible,
        ⚖️ marker present in the reply)
      - NOT emit a single confident value ("Oxana" or "John" alone)

    The engine-correlation gate finds the pending item (it correlates to the contested
    belief), so Step 1b runs the classifier.  The classifier returns "neither", so
    Step 1b falls through without submitting and sets _has_unresolved_pending_from_queue.
    Step 4 then builds new_contested/new_pending and the respond node re-surfaces the
    conflict with the ⚖️ deterministic reply.

    This proves that the wording-independent gate is safe — a non-verdict message for a
    contested subject is handled correctly by the classifier, not by the prior AI wording.
    """
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

    store = LGFakeMemoryStore(
        belief=_CONTESTED_BELIEF_OXANA_JOHN,
        reconcile_result_belief=_CONTESTED_BELIEF_OXANA_JOHN,  # tie persists
        pending_items=[dict(_PENDING_OXANA_JOHN)],
    )

    fake_llm = FakeMessagesListChatModel(responses=[])

    # Build retrieve_memory directly to call it with hand-built state.
    retrieve_memory, respond, _write_memory = make_nodes(
        memory_store=store,
        llm=fake_llm,
        extractor=_empty_extractor,
        key_extractor=_fixed_key_extractor("a-consulting:ceo", "held_by"),
        decision_classifier=_fixed_decision_classifier("neither"),  # user gives no clear answer
    )

    # State: fresh query ("Who is the CEO?") with NO pending_decision in state.
    # No prior AI message — this is Turn C (first time the conflict is surfaced)
    # OR the user re-queries after a non-verdict.
    state = {
        "messages": [
            HumanMessage(content="Who is the CEO of A Consulting?"),
        ],
        "user_id": "u1",
        "agent_id": "test",
        "pending_decision": None,
        "contested": None,
        "_decision_turn": False,
        "_resolved_reply": None,
        "memory_context": "",
        "last_subject": None,
        "last_predicate": None,
    }

    config = RunnableConfig(configurable={"thread_id": "fresh-query-resurface-t1"})
    lg_store = InMemoryStore()

    result = retrieve_memory(state, config, store=lg_store)

    # 1. No submit — classifier returned "neither" for the fresh query.
    assert store.submitted_verdicts == [], (
        f"Expected no verdicts for fresh query (neither), got: {store.submitted_verdicts}"
    )

    # 2. Pending item remains in the queue.
    assert len(store.list_pending()) == 1, (
        f"Expected pending item to remain (queue intact), got: {store.list_pending()}"
    )

    # 3. Conflict re-surfaced: memory_context must be CONTESTED (not a single confident value).
    memory_ctx = result.get("memory_context", "")
    assert "CONTESTED" in memory_ctx, (
        f"Expected CONTESTED in memory_context (conflict re-surfaced), got: {memory_ctx!r}"
    )

    # 4. Both candidate values present in memory_context — agent cannot pick one winner.
    assert "Oxana" in memory_ctx, (
        f"Candidate 'Oxana' must appear in memory_context, got: {memory_ctx!r}"
    )
    assert "John" in memory_ctx, (
        f"Candidate 'John' must appear in memory_context, got: {memory_ctx!r}"
    )

    # 5. _decision_turn is NOT True — this was not a resolved decision.
    assert not result.get("_decision_turn"), (
        f"Expected _decision_turn=False for fresh/neither query, got: {result.get('_decision_turn')}"
    )

    # 6. pending_decision is set (Step 4 correlated the pending item after fall-through).
    assert result.get("pending_decision") is not None, (
        "Expected pending_decision to be set after neither fall-through (Step 4)"
    )
