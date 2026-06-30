"""
mempill_showcase.frameworks.langgraph.studio_graph — LangGraph Studio entry point.

Exposes a module-level `graph` (compiled StateGraph) for use with LangGraph Studio
(`langgraph dev`) and the `langgraph.json` manifest.

Design:
  - Uses in-memory ORACLE-backed MempillAdapter + MockSupervisor (deterministic, no API key).
  - The graph is built WITHOUT a hard-coded MemorySaver so LangGraph Studio / `langgraph dev`
    can attach its own checkpointer for the persistent thread store it manages.
  - The module-level `graph` is a compiled StateGraph that Studio discovers via langgraph.json.

Usage (LangGraph Studio):
  1. pip install "langgraph-cli[inmem]"
  2. .venv/bin/langgraph dev          # Studio opens at http://localhost:2024
  3. In Studio, select the "exec_assistant" graph.
  4. Send an input, e.g.: {"user_input": "recall alice-chen/city", "agent_id": "demo-agent"}

The graph runs entirely without an API key (MockSupervisor).
Set ANTHROPIC_API_KEY in .env to switch to the live LLM supervisor path automatically
via the LLMSupervisor lazy-load (not wired here by default — swap classifier= for live use).

Nodes:
  supervisor  — intent classification + routing
  crew_a      — UPDATE_CONTACT intake (write to mempill)
  crew_b      — RESEARCH distillation (RAG + mempill distil)
  crew_c      — PREPARE_BRIEFING / RECALL_HISTORY / COMPLIANCE_AUDIT (read-only)
  hitl_node   — Human-in-the-loop gate for Contested writes
"""
from __future__ import annotations

import logging

from mempill_showcase.config.di import build_mempill_adapter, build_tools
from mempill_showcase.frameworks.langgraph.graph import build_graph
from mempill_showcase.frameworks.langgraph.supervisor_node import MockSupervisor

log = logging.getLogger(__name__)


def _build_studio_graph():
    """Build the compiled StateGraph for LangGraph Studio.

    Uses a fresh in-memory oracle-backed MempillAdapter and MockSupervisor so the
    graph is fully deterministic with no API key required.

    NOTE: MemorySaver is intentionally NOT attached here.  LangGraph Studio /
    `langgraph dev` injects its own checkpointer at runtime.  If you need HITL
    interrupt/resume in Studio, Studio's built-in persistence layer handles it.
    """
    adapter = build_mempill_adapter(in_memory=True, oracle_backed=True)
    tools = build_tools(adapter)
    classifier = MockSupervisor()
    # checkpointer=None: do NOT attach a MemorySaver.  LangGraph Studio / `langgraph dev`
    # injects its own persistence layer and rejects graphs with a custom checkpointer.
    compiled = build_graph(adapter=adapter, tools=tools, classifier=classifier, checkpointer=None)
    log.info(
        "studio_graph: compiled ExecAssistant StateGraph (MockSupervisor, oracle-backed, no API key)"
    )
    return compiled


# Module-level compiled graph — discovered by langgraph.json as:
#   "exec_assistant": "./src/mempill_showcase/frameworks/langgraph/studio_graph.py:graph"
graph = _build_studio_graph()
