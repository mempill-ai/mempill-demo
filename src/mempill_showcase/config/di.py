"""
mempill_showcase.config.di — minimal dependency injection factory.

Builds the engine + chosen adapter (mempill or naive) based on settings.
All mempill imports are deferred to this module and the mempill_adapter.

W3 additions:
  build_tools(adapter)  — constructs ShowcaseTools (all W2 tool instances)
  build_langgraph(...)  — constructs the full ExecAssistant StateGraph
"""
from __future__ import annotations

from enum import Enum
from typing import Optional, Union

from mempill_showcase.adapters.memory.naive_adapter import NaiveAdapter


class AdapterMode(str, Enum):
    MEMPILL = "mempill"
    NAIVE = "naive"


def build_mempill_adapter(in_memory: bool = True):
    """Build a MempillAdapter wrapping a real mempill engine.

    in_memory=True  → ephemeral in-memory engine (for tests and demos)
    in_memory=False → raises NotImplementedError (file-backed not wired in W1)
    """
    from mempill import open_in_memory
    from mempill_showcase.adapters.memory.mempill_adapter import MempillAdapter

    if not in_memory:
        raise NotImplementedError("File-backed mempill engine not wired in W1; use in_memory=True")

    engine = open_in_memory()
    return MempillAdapter(engine)


def build_naive_adapter() -> NaiveAdapter:
    """Build a NaiveAdapter (no mempill, no valid-time, no provenance)."""
    return NaiveAdapter()


def build_adapter(
    mode: AdapterMode = AdapterMode.MEMPILL,
    in_memory: bool = True,
) -> Union["MempillAdapter", NaiveAdapter]:  # type: ignore[name-defined]
    """Factory: return the adapter for the requested mode.

    mode=MEMPILL → MempillAdapter (BiTemporalMemoryStore)
    mode=NAIVE   → NaiveAdapter (MemoryStore only)
    """
    if mode == AdapterMode.MEMPILL:
        return build_mempill_adapter(in_memory=in_memory)
    elif mode == AdapterMode.NAIVE:
        return build_naive_adapter()
    else:
        raise ValueError(f"Unknown AdapterMode: {mode!r}")


def build_tools(adapter):
    """Build a ShowcaseTools NamedTuple from a MempillAdapter.

    All W2 tool instances are constructed here and returned as a bundle
    for injection into the graph nodes (W3+).
    """
    from mempill_showcase.frameworks.langgraph.graph import ShowcaseTools
    from mempill_showcase.tools.date_parser_tool import DateParserTool
    from mempill_showcase.tools.mempill_audit_tool import MempillAuditTool
    from mempill_showcase.tools.mempill_recall_tool import MempillRecallTool
    from mempill_showcase.tools.mempill_remember_tool import MempillRememberTool
    from mempill_showcase.tools.rag_read_tool import RAGReadTool
    from mempill_showcase.tools.rag_write_tool import InMemoryRAGStore, RAGWriteTool

    rag_store = InMemoryRAGStore()
    return ShowcaseTools(
        remember_tool=MempillRememberTool(adapter=adapter),
        recall_tool=MempillRecallTool(adapter=adapter),
        audit_tool=MempillAuditTool(adapter=adapter),
        date_parser=DateParserTool(),
        rag_write_tool=RAGWriteTool(store=rag_store),
        rag_read_tool=RAGReadTool(store=rag_store),
    )


def build_langgraph(
    in_memory: bool = True,
    classifier=None,
):
    """Build the full ExecAssistant LangGraph app.

    Returns (app, adapter) so callers can seed data or inspect state after the graph.
    classifier: SupervisorClassifier implementation; defaults to MockSupervisor.
    """
    from mempill_showcase.frameworks.langgraph.graph import build_graph

    adapter = build_mempill_adapter(in_memory=in_memory)
    tools = build_tools(adapter)
    app = build_graph(adapter=adapter, tools=tools, classifier=classifier)
    return app, adapter
