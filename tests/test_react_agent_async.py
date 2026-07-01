"""
tests.test_react_agent_async — Async path regression tests for the ReAct agent.

Marked with @pytest.mark.live — excluded from the default test run:
  pytest -m "not live"     # fast CI suite (skips this file)
  pytest -m live           # run only live tests (requires ANTHROPIC_API_KEY)

These tests drive the agent via app.ainvoke() — the async execution path used by
LangGraph Studio — to ensure that no tool raises NotImplementedError from _arun.

Coverage:
  ASYNC-A: Dietary-restriction query → "vegetarian", no NotImplementedError.
            (Exact failing case: first turn in Studio → recall_subject _arun error.)
  ASYNC-B: Point-in-time recall_at async — Alice's city in early 2024 → Austin TX.
  ASYNC-C: Contested write → HITL interrupt (async ainvoke) → Command(resume="Affirm")
            → recall returns CTO. Confirms interrupt propagates and resumes correctly
            under async execution.

No pytest-asyncio required: async tests are wrapped with asyncio.run() so they run
as ordinary synchronous tests.
"""
from __future__ import annotations

import asyncio
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
def test_async_a_dietary_restriction_no_error(seeded_app):
    """ASYNC-A: ainvoke path — dietary restriction query → 'vegetarian', no NotImplementedError.

    This is the exact failing case: LangGraph Studio calls tools via _arun.
    Before the fix, the first tool call (recall_subject) raised
    NotImplementedError('RecallSubjectTool does not support async').
    After the fix, the tool returns the correct result.

    Uses asyncio.run() to drive the async path without requiring pytest-asyncio.
    """
    app, _ = seeded_app
    config = {"configurable": {"thread_id": "async-a-dietary"}}

    async def run():
        return await app.ainvoke(
            {"messages": [HumanMessage(content="What is Alice Chen's dietary restriction?")]},
            config=config,
        )

    result = asyncio.run(run())

    answer = _last_ai_text(result)
    interrupted = _has_interrupt(result)

    assert "vegetarian" in answer.lower(), (
        f"ASYNC-A: Expected 'vegetarian' in agent answer via ainvoke, got: {answer!r}"
    )
    assert not interrupted, (
        "ASYNC-A: Dietary restriction query triggered a spurious HITL interrupt — "
        "should answer directly from memory."
    )


@pytest.mark.live
def test_async_b_point_in_time_recall_async(seeded_app):
    """ASYNC-B: ainvoke path — recall_at / recall_subject async → Austin TX for early 2024.

    Alice's city: Austin TX valid_from=2023-06, valid_until=2025-02.
    A query for her city around March 2024 must return Austin TX.
    Confirms recall_at _arun works correctly under async execution.
    """
    app, _ = seeded_app
    config = {"configurable": {"thread_id": "async-b-city"}}

    async def run():
        return await app.ainvoke(
            {"messages": [HumanMessage(
                content="What city was Alice Chen living in during early 2024, say around March 2024?"
            )]},
            config=config,
        )

    result = asyncio.run(run())

    answer = _last_ai_text(result)
    assert "austin" in answer.lower(), (
        f"ASYNC-B: Expected 'Austin' in answer via ainvoke, got: {answer!r}"
    )
    assert not _has_interrupt(result), (
        "ASYNC-B: Bi-temporal recall query should not trigger HITL."
    )


@pytest.mark.live
def test_async_c_contested_hitl_loop_async(seeded_app):
    """ASYNC-C: ainvoke → contested write → interrupt → Command(resume='Affirm') → CTO.

    Verifies that request_adjudication_tool._arun:
      1. Calls interrupt() correctly in the async context (graph pauses).
      2. Resumes correctly when Command(resume='Affirm') is passed to ainvoke.
      3. Resolves the conflict so subsequent recall returns 'CTO'.

    Uses a fresh in-memory adapter so the contested write is clean.
    """
    adapter = build_mempill_adapter(in_memory=True, oracle_backed=True)
    load_seed_claims(adapter, AGENT_ID)
    app = build_graph_from_adapter(adapter)
    thread_id = "async-c-hitl"
    config = {"configurable": {"thread_id": thread_id}}

    # Step 1: write challenger claim → should interrupt
    async def step1():
        return await app.ainvoke(
            {"messages": [HumanMessage(
                content="Alice has actually been CTO of Acme since June 2023, not VP Engineering"
            )]},
            config=config,
        )

    result1 = asyncio.run(step1())

    assert _has_interrupt(result1), (
        "ASYNC-C: Agent did not interrupt when a contested write was detected via ainvoke. "
        "Expected graph to pause at request_adjudication._arun for human verdict. "
        f"Last agent text: {_last_ai_text(result1)!r}"
    )

    payload = _interrupt_payload(result1)
    assert payload is not None, "ASYNC-C: Interrupt payload must not be None"
    assert payload.get("subject") == "alice-chen", (
        f"ASYNC-C: Expected subject='alice-chen', got {payload.get('subject')!r}"
    )
    assert payload.get("predicate") == "employer", (
        f"ASYNC-C: Expected predicate='employer', got {payload.get('predicate')!r}"
    )

    # Step 2: resume with Affirm verdict via ainvoke
    async def step2():
        return await app.ainvoke(
            Command(resume="Affirm"),
            config=config,
        )

    result2 = asyncio.run(step2())

    assert not _has_interrupt(result2), (
        "ASYNC-C: Graph re-interrupted after Affirm verdict — expected clean resume to END."
    )

    # Step 3: verify recall returns CTO
    async def step3():
        return await app.ainvoke(
            {"messages": [HumanMessage(content="What role does Alice hold now?")]},
            config=config,
        )

    result3 = asyncio.run(step3())
    role_answer = _last_ai_text(result3)
    assert "cto" in role_answer.lower(), (
        f"ASYNC-C: After Affirm via ainvoke, expected role answer to contain 'CTO'. "
        f"Got: {role_answer!r}"
    )
