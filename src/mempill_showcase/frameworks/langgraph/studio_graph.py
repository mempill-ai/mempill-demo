"""
mempill_showcase.frameworks.langgraph.studio_graph — LangGraph Studio entry point.

Exposes a module-level `graph` (compiled ReAct agent) for use with LangGraph Studio
(`langgraph dev`) and the `langgraph.json` manifest.

Design:
  - Builds an oracle-backed in-memory MempillAdapter ONCE at module level.
  - Calls load_seed_claims() immediately so the store has Day-0 facts on first turn.
  - The graph is compiled WITHOUT a MemorySaver so LangGraph Studio / `langgraph dev`
    can attach its own checkpointer (Studio rejects graphs with a custom checkpointer).
  - Default model: claude-haiku-4-5 (overridable via ANTHROPIC_MODEL env var).
  - ANTHROPIC_API_KEY is required for the ReAct agent to call the LLM.

Usage (LangGraph Studio):
  1. Set ANTHROPIC_API_KEY in .env.
  2. .venv/bin/langgraph dev          # Studio opens at http://127.0.0.1:2024
  3. In Studio, select the graph.
  4. Fill the "messages" field with a HumanMessage — agent_id defaults to "jordan-park-001".
  5. The Day-0 seed is already loaded; try:
       "What is Alice Chen's dietary restriction?"
       "What role does Alice hold?"
       "Alice has actually been CTO of Acme since June 2023, not VP Engineering"
"""
from __future__ import annotations

import logging

from mempill_showcase.config.bootstrap import bootstrap
from mempill_showcase.config.di import build_agent_tools, build_mempill_adapter, _adapter_from_settings
from mempill_showcase.frameworks.langgraph.graph import build_graph
from mempill_showcase.scenarios.seed_data import AGENT_ID, load_seed_claims

log = logging.getLogger(__name__)


def _load_default_agent_id() -> str:
    try:
        from mempill_showcase.config.settings import get_settings
        return get_settings().mempill_agent_id
    except Exception:
        return AGENT_ID


_DEFAULT_AGENT_ID = _load_default_agent_id()


def _build_studio_graph():
    """Build the compiled ReAct agent for LangGraph Studio.

    Steps:
      1. Call bootstrap() to load .env.
      2. Build oracle-backed in-memory MempillAdapter.
      3. Seed Day-0 claims.
      4. Build 7 agent tools.
      5. Compile the graph without MemorySaver (Studio manages persistence).

    Returns (compiled_graph, adapter).
    """
    bootstrap()

    from mempill_showcase.config.settings import get_settings
    _settings = get_settings()

    adapter = _adapter_from_settings(_settings)

    _seed_count = 0
    try:
        refs = load_seed_claims(adapter, agent_id=_DEFAULT_AGENT_ID)
        _seed_count = len(refs)
        if _seed_count:
            log.info(
                "studio_graph: seeded %d Day-0 claims for agent_id=%r",
                _seed_count, _DEFAULT_AGENT_ID,
            )
        else:
            log.info(
                "studio_graph: store already seeded for agent_id=%r — seed skipped",
                _DEFAULT_AGENT_ID,
            )
    except Exception as exc:
        log.warning("studio_graph: seed failed (%s) — starting with existing store state", exc)

    tools = build_agent_tools(adapter)

    # No checkpointer — Studio injects its own persistence layer
    compiled = build_graph(
        adapter=adapter,
        tools=tools,
        checkpointer=None,
        model_name=_settings.anthropic_model,
    )

    log.info(
        "studio_graph: compiled ReAct ExecAssistant "
        "(model=%s, seeded=%d claims, agent_id=%r, no MemorySaver)",
        _settings.anthropic_model,
        _seed_count,
        _DEFAULT_AGENT_ID,
    )
    return compiled, adapter


# ── Module-level compiled graph ───────────────────────────────────────────────

# Built once at import time.  `langgraph dev` imports this module once and
# serves all Studio turns from the same process.
graph, studio_adapter = _build_studio_graph()
