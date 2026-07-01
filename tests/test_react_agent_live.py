"""
tests.test_react_agent_live — Live regression tests for the free-form ReAct agent.

Marked with @pytest.mark.live — excluded from the default test run:
  pytest -m "not live"          # fast CI suite (skips this file)
  pytest -m live                # run only live tests (requires ANTHROPIC_API_KEY)

Covers:
  LIVE-A: Dietary-restriction query → "vegetarian", no spurious interrupt.
  LIVE-D: Contested write → HITL interrupt → Affirm verdict → recall returns CTO.

Both tests require a valid ANTHROPIC_API_KEY in the environment (or .env file).
The agent makes real calls to the Anthropic API using the model configured in
settings (default: claude-haiku-4-5).
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
