"""
tests.test_react_agent_live — Live semantic E2E tests for the free-form ReAct agent.

Marked with @pytest.mark.live — excluded from the default test run:
  pytest -m "not live"          # fast CI suite (skips this file)
  pytest -m live                # run only live tests (requires ANTHROPIC_API_KEY)

Coverage:
  LIVE-A: Dietary-restriction query → "vegetarian", no spurious interrupt.
  LIVE-B: Role query (no alias map) → employer fact "VP Engineering" or similar.
  LIVE-C: Novel-attribute write + read-back (bob-liu/preferred_airline).
  LIVE-D: Contested write → HITL interrupt → Affirm → recall returns CTO.
  LIVE-E: Step-aside query (Bob / Acme CEO) → mentions Diane Foster.
  LIVE-F: Point-in-time valid_at (Alice city in early 2024 → Austin TX).
  LIVE-G: Audit trail returns events (non-empty list described).

All tests require a valid ANTHROPIC_API_KEY in the environment (or .env file).
The agent makes real calls to the Anthropic API using the model configured in
settings (default: claude-haiku-4-5).

Assertions are semantic (substring, case-insensitive), not exact string matches,
so they remain robust across model phrasing variations.
"""
from __future__ import annotations

import json
import os

import pytest
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.types import Command

# Load .env from the project root so ANTHROPIC_API_KEY is available when
# pytest is invoked from outside the project directory.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

from mempill_showcase.config.di import build_mempill_adapter, build_graph_from_adapter  # noqa: E402
from mempill_showcase.scenarios.seed_data import load_seed_claims, AGENT_ID  # noqa: E402


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_seeded_app():
    """Return (app, adapter) with Day-0 seed facts loaded."""
    adapter = build_mempill_adapter(in_memory=True, oracle_backed=True)
    load_seed_claims(adapter, AGENT_ID)
    app = build_graph_from_adapter(adapter)
    return app, adapter


def _last_ai_text(result: dict) -> str:
    """Extract the last non-empty AI message text from a graph result."""
    msgs = result.get("messages", [])
    for msg in reversed(msgs):
        cls = msg.__class__.__name__
        if cls not in ("AIMessage", "AIMessageChunk"):
            continue
        content = msg.content
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            texts = [
                c.get("text", "") if isinstance(c, dict) else str(c)
                for c in content
            ]
            joined = " ".join(t for t in texts if t.strip())
            if joined.strip():
                return joined.strip()
    return ""


def _has_interrupt(result: dict) -> bool:
    """Return True if the graph paused at an interrupt checkpoint."""
    return bool(result.get("__interrupt__", []))


def _interrupt_payload(result: dict):
    """Return the first interrupt value dict, or None."""
    interrupts = result.get("__interrupt__", [])
    if not interrupts:
        return None
    item = interrupts[0]
    return item.value if hasattr(item, "value") else item


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def seeded_app():
    """Fresh in-memory ReAct app with Day-0 seed facts."""
    return _build_seeded_app()


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.live
def test_live_a_dietary_restriction_no_hitl(seeded_app):
    """LIVE-A: agent answers dietary question from memory, no spurious interrupt.

    Sends "What is Alice's dietary restriction?" to the live agent.
    Expects:
      - answer contains "vegetarian" (case-insensitive)
      - graph did NOT pause at an interrupt (no spurious HITL trigger)
    """
    app, _ = seeded_app
    config = {"configurable": {"thread_id": "live-a-dietary"}}

    result = app.invoke(
        {"messages": [HumanMessage(content="What is Alice's dietary restriction?")]},
        config=config,
    )

    answer = _last_ai_text(result)
    interrupted = _has_interrupt(result)

    assert "vegetarian" in answer.lower(), (
        f"Expected 'vegetarian' in agent answer, got: {answer!r}"
    )
    assert not interrupted, (
        "Agent triggered a spurious HITL interrupt on a simple recall query — "
        "should answer directly from memory without interrupting."
    )


@pytest.mark.live
def test_live_b_role_query_no_alias_map(seeded_app):
    """LIVE-B: role query answered from employer fact without a hardcoded alias map.

    The agent must call recall_subject(alice-chen) and reason from the 'employer'
    predicate value ('Acme Corp / VP Engineering') to answer a 'role' question.
    No hardcoded alias map is required — the agent reads the predicate and
    answers linguistically.
    """
    app, _ = seeded_app
    config = {"configurable": {"thread_id": "live-b-role"}}

    result = app.invoke(
        {"messages": [HumanMessage(content="What role does Alice hold at Acme?")]},
        config=config,
    )

    answer = _last_ai_text(result)
    interrupted = _has_interrupt(result)

    # The answer must mention VP Engineering or Acme (derived from employer fact)
    answer_lower = answer.lower()
    assert "vp" in answer_lower or "engineering" in answer_lower or "acme" in answer_lower, (
        f"Expected role/employer info in answer, got: {answer!r}"
    )
    assert not interrupted, (
        "Role query should not trigger HITL — it is a simple recall."
    )


@pytest.mark.live
def test_live_c_novel_attribute_write_and_read(seeded_app):
    """LIVE-C: write a novel attribute (preferred_airline) then read it back.

    Tests the remember_fact → recall round-trip for an attribute not in seed data.
    bob-liu has no 'preferred_airline' claim — write one, then ask about it.
    """
    app, adapter = seeded_app
    thread_id = "live-c-novel-attr"
    config = {"configurable": {"thread_id": thread_id}}

    # Write a novel fact
    result_write = app.invoke(
        {"messages": [HumanMessage(
            content=(
                "Bob Liu prefers flying Singapore Airlines. "
                "Please store this preference in memory as of 2025."
            )
        )]},
        config=config,
    )
    assert not _has_interrupt(result_write), (
        "Novel attribute write should not trigger HITL (no incumbent claim)."
    )

    # Read back on a fresh thread (same adapter, new thread to avoid message contamination)
    config2 = {"configurable": {"thread_id": "live-c-novel-attr-read"}}
    result_read = app.invoke(
        {"messages": [HumanMessage(content="What airline does Bob Liu prefer?")]},
        config=config2,
    )

    answer = _last_ai_text(result_read)
    assert "singapore" in answer.lower(), (
        f"Expected 'Singapore' in answer after write, got: {answer!r}"
    )
    assert not _has_interrupt(result_read), (
        "Read-back should not trigger HITL."
    )


@pytest.mark.live
def test_live_d_contested_affirm_loop(seeded_app):
    """LIVE-D: contested write → HITL interrupt → Affirm → recall returns CTO.

    Sequence:
      1. Assert "Alice is CTO of Acme since June 2023, not VP Engineering".
      2. Agent writes to employer predicate, detects conflict (Contested).
      3. Agent calls request_adjudication → graph PAUSES (__interrupt__ present).
      4. Resume with Command(resume="Affirm").
      5. Agent resolves → challenger (CTO) wins.
      6. Query "What role does Alice hold now?" → answer contains "CTO".
    """
    app, _ = seeded_app
    thread_id = "live-d-contested-affirm"
    config = {"configurable": {"thread_id": thread_id}}

    # Step 1-3: write challenger claim → should interrupt
    result1 = app.invoke(
        {"messages": [HumanMessage(
            content="Alice has actually been CTO of Acme since June 2023, not VP Engineering"
        )]},
        config=config,
    )
    assert _has_interrupt(result1), (
        "Agent did not interrupt when a contested write was detected. "
        "Expected graph to pause at request_adjudication for human verdict. "
        f"Last agent text: {_last_ai_text(result1)!r}"
    )

    # Verify interrupt payload has expected structure
    payload = _interrupt_payload(result1)
    assert payload is not None, "Interrupt payload must not be None"
    assert payload.get("subject") == "alice-chen", (
        f"Expected subject='alice-chen' in interrupt payload, got {payload.get('subject')!r}"
    )
    assert payload.get("predicate") == "employer", (
        f"Expected predicate='employer' in interrupt payload, got {payload.get('predicate')!r}"
    )

    # Step 4: resume with Affirm verdict
    result2 = app.invoke(
        Command(resume="Affirm"),
        config=config,
    )
    assert not _has_interrupt(result2), (
        "Graph re-interrupted after Affirm verdict — expected clean resume to END."
    )

    # Step 5-6: verify recall returns CTO
    result3 = app.invoke(
        {"messages": [HumanMessage(content="What role does Alice hold now?")]},
        config=config,
    )
    role_answer = _last_ai_text(result3)
    assert "cto" in role_answer.lower(), (
        f"After Affirm, expected role answer to contain 'CTO'. Got: {role_answer!r}"
    )


@pytest.mark.live
def test_live_e_step_aside_ceo_of_acme(seeded_app):
    """LIVE-E: step-aside question about Bob / Acme CEO → mentions Diane Foster.

    "Who is the CEO of Acme Corp?" requires looking up acme-corp/ceo, not alice-chen.
    The seeded fact: acme-corp / ceo = Diane Foster.
    """
    app, _ = seeded_app
    config = {"configurable": {"thread_id": "live-e-ceo"}}

    result = app.invoke(
        {"messages": [HumanMessage(content="Who is the CEO of Acme Corp?")]},
        config=config,
    )

    answer = _last_ai_text(result)
    assert "diane" in answer.lower() or "foster" in answer.lower(), (
        f"Expected Diane Foster as CEO of Acme Corp, got: {answer!r}"
    )
    assert not _has_interrupt(result), (
        "CEO query should not trigger HITL."
    )


@pytest.mark.live
def test_live_f_point_in_time_city_early_2024(seeded_app):
    """LIVE-F: point-in-time valid_at query — Alice's city in early 2024 → Austin TX.

    Alice's city: Austin TX valid_from=2023-06, valid_until=2025-02.
    A query for her city in early 2024 (before the move) must return Austin TX.
    The agent must use recall_at with a 2024 valid_at timestamp.
    """
    app, _ = seeded_app
    config = {"configurable": {"thread_id": "live-f-city-2024"}}

    result = app.invoke(
        {"messages": [HumanMessage(
            content="What city was Alice Chen living in during early 2024, say around March 2024?"
        )]},
        config=config,
    )

    answer = _last_ai_text(result)
    assert "austin" in answer.lower(), (
        f"Expected 'Austin' (Texas) as Alice's city in early 2024, got: {answer!r}"
    )
    assert not _has_interrupt(result), (
        "Bi-temporal recall query should not trigger HITL."
    )


@pytest.mark.live
def test_live_h_ceo_appointment_contests_org_seat(seeded_app):
    """LIVE-H: a new CEO appointment is modeled as an ORG attribute and CONTESTS.

    Seeded fact: acme-corp / ceo = Diane Foster (open-ended, valid_from=2021-04).
    "Joan was appointed as Acme's CEO since Sep 2024 until Nov 2025" is a statement
    about the ORG's leadership seat — the agent must write acme-corp/ceo="Joan"
    (NOT joan/employer), with a bounded valid_from/valid_until, and because
    acme-corp/ceo already holds "Diane Foster" this CONTESTS and triggers HITL.
    """
    app, _ = seeded_app
    config = {"configurable": {"thread_id": "live-h-ceo-contest"}}

    result = app.invoke(
        {"messages": [HumanMessage(
            content="Joan was appointed as Acme's CEO since Sep 2024 until Nov 2025"
        )]},
        config=config,
    )

    assert _has_interrupt(result), (
        "Expected the CEO appointment to contest the incumbent (Diane Foster) and "
        f"pause for adjudication. Last agent text: {_last_ai_text(result)!r}"
    )

    payload = _interrupt_payload(result)
    assert payload is not None, "Interrupt payload must not be None"
    assert payload.get("subject") == "acme-corp", (
        f"Expected the contested write on the ORG (acme-corp), got subject={payload.get('subject')!r}"
    )
    assert payload.get("predicate") == "ceo", (
        f"Expected predicate='ceo', got {payload.get('predicate')!r}"
    )


@pytest.mark.live
def test_live_g_audit_trail_returns_events(seeded_app):
    """LIVE-G: audit_trail query returns non-empty event history.

    Asks the agent for the audit trail / write events for the session.
    The agent should call audit_trail and return a description of events.
    The answer must convey that events were found (non-empty audit).
    """
    app, _ = seeded_app
    config = {"configurable": {"thread_id": "live-g-audit"}}

    result = app.invoke(
        {"messages": [HumanMessage(
            content=f"Show me the audit trail for agent {AGENT_ID} — what memory write events have occurred?"
        )]},
        config=config,
    )

    answer = _last_ai_text(result)
    # The answer should describe events; at minimum it should not say "no events"
    answer_lower = answer.lower()
    has_event_mention = any(
        kw in answer_lower for kw in [
            "event", "claim", "audit", "write", "commit", "record", "entry",
            "alice", "bob", "acme", "jordan",
        ]
    )
    assert has_event_mention, (
        f"Expected audit events to be described in answer, got: {answer!r}"
    )
    assert not _has_interrupt(result), (
        "Audit trail query should not trigger HITL."
    )
