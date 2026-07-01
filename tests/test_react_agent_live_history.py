"""
tests.test_react_agent_live_history — Live E2E replay of the query_history bug
report: a "history over time" question about Acme's CEOs must be answered via
query_history's correct, truncated, non-overlapping chronology — NOT the stale
"Diane Foster onwards" narrative the agent previously produced by improvising
from audit_trail / recall_subject.

Marked with @pytest.mark.live — excluded from the default test run:
  pytest -m "not live"          # fast CI suite (skips this file)
  pytest -m live                # run only live tests (requires ANTHROPIC_API_KEY)

Scenario (fresh seeded adapter, real ANTHROPIC_API_KEY calls):
  1. Seed Diane Foster as acme-corp/ceo (open-ended, valid_from=2021-04) — from
     the standard Day-0 seed data.
  2. "Joan was appointed as Acme's CEO since Sep 2024 until Nov 2025"
     → contests Diane → interrupt → resume Command(resume="Affirm") → Joan wins.
  3. "John was appointed as Acme's CEO since Jan 2025 until Dec 2026"
     (overlaps Joan's stated window) → contests Joan → interrupt →
     resume Command(resume="Affirm") → John wins.
  4. "What is the history of Acme's CEOs over time?"
     → agent must call query_history and answer with the CORRECT truncated,
       non-overlapping chronology: Diane ...→2024-09, Joan 2024-09→2025-01,
       John 2025-01→open. Must NOT say Diane holds the seat "onwards"/"since
       2021" as the current/latest state.

Also verifies the duplicate-write guard: asking the agent to reconfirm an
already-current fact twice in a row does not create a second claim.
"""
from __future__ import annotations

import json
import os

import pytest
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.types import Command

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

from mempill_showcase.config.di import build_mempill_adapter, build_graph_from_adapter  # noqa: E402
from mempill_showcase.scenarios.seed_data import load_seed_claims, AGENT_ID  # noqa: E402


def _build_seeded_app():
    adapter = build_mempill_adapter(in_memory=True, oracle_backed=True)
    load_seed_claims(adapter, AGENT_ID)
    app = build_graph_from_adapter(adapter)
    return app, adapter


def _last_ai_text(result: dict) -> str:
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
    return bool(result.get("__interrupt__", []))


@pytest.fixture()
def seeded_app():
    return _build_seeded_app()


@pytest.mark.live
def test_live_history_diane_joan_john_correct_chronology(seeded_app):
    """Full succession replay → query_history → correct truncated timeline."""
    app, adapter = seeded_app
    thread_id = "live-history-ceo-succession"
    config = {"configurable": {"thread_id": thread_id}}

    # Step 1: Joan contests Diane (the Day-0 seeded incumbent).
    r1 = app.invoke(
        {"messages": [HumanMessage(
            content="Joan was appointed as Acme's CEO since Sep 2024 until Nov 2025"
        )]},
        config=config,
    )
    assert _has_interrupt(r1), (
        f"Expected Joan's appointment to contest Diane Foster. Last text: {_last_ai_text(r1)!r}"
    )
    r2 = app.invoke(Command(resume="Affirm"), config=config)
    assert not _has_interrupt(r2), "Graph re-interrupted after Affirm (Joan over Diane)."

    # Step 2: John contests Joan (overlaps her stated window).
    r3 = app.invoke(
        {"messages": [HumanMessage(
            content="John was appointed as Acme's CEO since Jan 2025 until Dec 2026"
        )]},
        config=config,
    )
    assert _has_interrupt(r3), (
        f"Expected John's appointment to contest Joan. Last text: {_last_ai_text(r3)!r}"
    )
    r4 = app.invoke(Command(resume="Affirm"), config=config)
    assert not _has_interrupt(r4), "Graph re-interrupted after Affirm (John over Joan)."

    # Sanity: the engine's own fold is correct before we even ask the agent.
    history_entries = adapter.query_history(AGENT_ID, "acme-corp", "ceo")
    values = [e["value"] for e in history_entries]
    assert "Diane Foster" in values[0] or "Diane" in values[0]
    assert values[-1] == "John"
    assert history_entries[-1]["status"] == "Current"
    assert history_entries[-1]["valid_until"] is None

    # Step 3: ask the history question — must use query_history, not improvise.
    r5 = app.invoke(
        {"messages": [HumanMessage(
            content="What is the history of Acme's CEOs over time?"
        )]},
        config=config,
    )
    answer = _last_ai_text(r5)
    answer_lower = answer.lower()

    assert "john" in answer_lower, f"Expected John (current CEO) in history answer: {answer!r}"
    assert "joan" in answer_lower, f"Expected Joan in history answer: {answer!r}"
    assert "diane" in answer_lower, f"Expected Diane Foster in history answer: {answer!r}"

    # The stale-bug regression check: must NOT claim Diane is the (current/
    # ongoing) CEO "onwards" from 2021 — she must be presented as superseded.
    stale_markers = ["diane foster: ceo from april 2021 onwards", "diane foster is the current ceo"]
    assert not any(m in answer_lower for m in stale_markers), (
        f"Agent reproduced the STALE narrative (Diane as ongoing CEO). Got: {answer!r}"
    )


@pytest.mark.live
def test_live_no_duplicate_claim_on_reconfirmed_ceo_query(seeded_app):
    """Asking the current CEO fact twice in a row must not create a duplicate
    claim — the agent should report the existing state, not re-write it."""
    app, adapter = seeded_app
    config = {"configurable": {"thread_id": "live-no-dup-ceo"}}

    app.invoke(
        {"messages": [HumanMessage(content="Who is the CEO of Acme Corp?")]},
        config=config,
    )
    before = adapter.query_history(AGENT_ID, "acme-corp", "ceo")

    app.invoke(
        {"messages": [HumanMessage(content="Who is the CEO of Acme Corp?")]},
        config=config,
    )
    after = adapter.query_history(AGENT_ID, "acme-corp", "ceo")

    assert len(after) == len(before), (
        f"A read-only repeated question must not create new claims. "
        f"before={len(before)} entries, after={len(after)} entries."
    )
