"""
mempill_showcase.frameworks.langgraph.studio_graph — LangGraph Studio entry point.

Exposes a module-level `graph` (compiled StateGraph) for use with LangGraph Studio
(`langgraph dev`) and the `langgraph.json` manifest.

Design:
  - Builds an oracle-backed in-memory MempillAdapter ONCE at module level.
  - Calls load_seed_claims() immediately so the store has Day-0 facts on first turn.
  - Defaults agent_id to "jordan-park-001" (the seeded agent) when the user leaves
    that field blank in Studio.
  - If ANTHROPIC_API_KEY is present in the environment (loaded via .env), uses
    LLMSupervisor for natural-language intent routing; otherwise MockSupervisor.
  - The graph is compiled WITHOUT a MemorySaver so LangGraph Studio / `langgraph dev`
    can attach its own checkpointer (Studio rejects graphs with a custom checkpointer).

Usage (LangGraph Studio):
  1. Optionally set ANTHROPIC_API_KEY in .env for best results (natural language routing).
  2. .venv/bin/langgraph dev          # Studio opens at http://127.0.0.1:2024
  3. In Studio, select the "exec_assistant" graph.
  4. Fill ONLY the "User Input" field — agent_id defaults to "jordan-park-001".
  5. The Day-0 seed is already loaded; try e.g.:
       "What's Alice Chen's current city?"   → Austin TX (seeded)
       "Alice moved to New York in February 2025" → succession write

Nodes:
  supervisor  — intent classification + routing (with default agent_id injection)
  crew_a      — UPDATE_CONTACT intake (write to mempill)
  crew_b      — RESEARCH distillation (RAG + mempill distil)
  crew_c      — PREPARE_BRIEFING / RECALL_HISTORY / COMPLIANCE_AUDIT (read-only)
  hitl_node   — Human-in-the-loop gate for Contested writes
"""
from __future__ import annotations

import logging
import os

from langgraph.graph import END, StateGraph

from mempill_showcase.config.bootstrap import bootstrap
from mempill_showcase.config.di import build_mempill_adapter, build_tools
from mempill_showcase.frameworks.langgraph.crew_nodes import (
    make_crew_a_node,
    make_crew_b_node,
    make_crew_c_node,
)
from mempill_showcase.frameworks.langgraph.graph import (
    _crew_a_router,
    _crew_b_router,
    _supervisor_router,
)
from mempill_showcase.frameworks.langgraph.hitl_node import make_hitl_node
from mempill_showcase.frameworks.langgraph.state import ExecAssistantState
from mempill_showcase.frameworks.langgraph.supervisor_node import (
    MockSupervisor,
    make_supervisor_node,
)
from mempill_showcase.scenarios.seed_data import AGENT_ID, load_seed_claims

log = logging.getLogger(__name__)

# ── Default agent_id ──────────────────────────────────────────────────────────
# Matches the agent_id used by load_seed_claims() so seed data is queryable
# without the user having to fill in the agent_id field in Studio.
_DEFAULT_AGENT_ID = AGENT_ID  # "jordan-park-001"


def _build_studio_graph():
    """Build the compiled StateGraph for LangGraph Studio.

    Steps:
      1. Call bootstrap() to load .env (picks up ANTHROPIC_API_KEY if present).
      2. Build oracle-backed in-memory MempillAdapter.
      3. Seed Day-0 claims so the store has real facts from turn 1.
      4. Select LLMSupervisor (if ANTHROPIC_API_KEY present) or MockSupervisor.
      5. Wrap the supervisor node to inject default agent_id when blank.
      6. Compile the graph without a MemorySaver (Studio manages persistence).

    Returns (compiled_graph, adapter, classifier).
    The adapter is module-level — seed writes from one turn are visible in
    subsequent turns within the same `langgraph dev` server process.
    """
    # Step 1: load .env so ANTHROPIC_API_KEY etc. reach os.environ before we inspect them.
    # bootstrap() is idempotent; safe to call at import time here because
    # studio_graph is only imported by `langgraph dev`, not by the test suite.
    bootstrap()

    # Step 2: build the adapter
    adapter = build_mempill_adapter(in_memory=True, oracle_backed=True)

    # Step 3: seed Day-0 facts immediately
    _seed_count = 0
    try:
        refs = load_seed_claims(adapter, agent_id=_DEFAULT_AGENT_ID)
        _seed_count = len(refs)
        log.info(
            "studio_graph: seeded %d Day-0 claims for agent_id=%r",
            _seed_count,
            _DEFAULT_AGENT_ID,
        )
    except Exception as exc:
        log.warning("studio_graph: seed failed (%s) — graph will start with empty store", exc)

    # Step 4: supervisor selection
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if api_key:
        try:
            from mempill_showcase.frameworks.langgraph.supervisor_node import LLMSupervisor
            classifier = LLMSupervisor()
            log.info("studio_graph: ANTHROPIC_API_KEY present — using LLMSupervisor")
        except Exception as exc:
            log.warning(
                "studio_graph: LLMSupervisor init failed (%s) — falling back to MockSupervisor",
                exc,
            )
            classifier = MockSupervisor()
    else:
        classifier = MockSupervisor()
        log.info("studio_graph: no ANTHROPIC_API_KEY — using MockSupervisor (deterministic)")

    # Step 5: build nodes
    tools = build_tools(adapter)

    inner_supervisor_fn = make_supervisor_node(classifier)

    def supervisor_with_default(state: ExecAssistantState) -> dict:
        """Classify intent; inject default agent_id when user leaves it blank."""
        if not state.get("agent_id"):
            state = dict(state)
            state["agent_id"] = _DEFAULT_AGENT_ID
            log.debug("studio_graph: agent_id defaulted to %r", _DEFAULT_AGENT_ID)
        updates = inner_supervisor_fn(state)
        # Ensure agent_id is propagated into the state patch so downstream nodes see it
        if not updates.get("agent_id"):
            updates = {**updates, "agent_id": state["agent_id"]}
        return updates

    supervisor_with_default.__name__ = "supervisor_node"

    crew_a_fn = make_crew_a_node(
        remember_tool=tools.remember_tool,
        date_parser=tools.date_parser,
        adapter=adapter,
        crew=None,
    )
    crew_b_fn = make_crew_b_node(
        remember_tool=tools.remember_tool,
        rag_write_tool=tools.rag_write_tool,
        adapter=adapter,
        crew=None,
    )
    crew_c_fn = make_crew_c_node(
        recall_tool=tools.recall_tool,
        audit_tool=tools.audit_tool,
        crew=None,
    )
    hitl_fn = make_hitl_node(
        adapter=adapter,
        recall_tool=tools.recall_tool,
    )

    # Step 6: assemble and compile the graph (no MemorySaver — Studio injects its own)
    g = StateGraph(ExecAssistantState)
    g.add_node("supervisor", supervisor_with_default)
    g.add_node("crew_a", crew_a_fn)
    g.add_node("crew_b", crew_b_fn)
    g.add_node("crew_c", crew_c_fn)
    g.add_node("hitl_node", hitl_fn)

    g.set_entry_point("supervisor")
    g.add_conditional_edges(
        "supervisor",
        _supervisor_router,
        {"crew_a": "crew_a", "crew_b": "crew_b", "crew_c": "crew_c"},
    )
    g.add_conditional_edges(
        "crew_a",
        _crew_a_router,
        {"hitl": "hitl_node", END: END},
    )
    g.add_conditional_edges(
        "crew_b",
        _crew_b_router,
        {"hitl": "hitl_node", END: END},
    )
    g.add_edge("crew_c", END)
    g.add_edge("hitl_node", END)

    compiled = g.compile(checkpointer=None)
    log.info(
        "studio_graph: compiled ExecAssistant StateGraph "
        "(classifier=%s, seeded=%d claims, default_agent_id=%r, no MemorySaver)",
        type(classifier).__name__,
        _seed_count,
        _DEFAULT_AGENT_ID,
    )
    return compiled, adapter, classifier


# ── Module-level compiled graph ───────────────────────────────────────────────

# Build once at import time.  `langgraph dev` imports this module once and
# serves all Studio turns from the same process — so the adapter persists
# writes across turns automatically (seed in turn 0, recall in turn 1 works).
graph, studio_adapter, studio_classifier = _build_studio_graph()
