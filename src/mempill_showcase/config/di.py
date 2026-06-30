"""
mempill_showcase.config.di — minimal dependency injection factory.

Builds the engine + chosen adapter (mempill or naive) based on settings.
All mempill imports are deferred to this module and the mempill_adapter.

W3 additions:
  build_tools(adapter)  — constructs ShowcaseTools (all W2 tool instances)
  build_langgraph(...)  — constructs the full ExecAssistant StateGraph

W4 additions:
  build_langgraph(..., use_crewai=False, llm=None)
    use_crewai=False (default): deterministic shell path (no API key, CI-safe)
    use_crewai=True:            CrewAI crews injected into the graph nodes
    llm:                        LiteLLM model string or crewai.LLM instance;
                                forwarded to build_crews() when use_crewai=True

W6 additions:
  build_app(use_crewai=False, classifier=None, adapter=None)
    Single-call factory for the compiled runnable graph.
    Returns (app, adapter) — the same contract as build_langgraph().
    - classifier=None: defaults to MockSupervisor (no API key required).
    - classifier=LLMSupervisor(...): real-LLM path (requires ANTHROPIC_API_KEY).
    - adapter=None: creates a fresh in-memory MempillAdapter internally.
    - adapter=<existing>: reuse a pre-seeded adapter (tests / CLI).

Environment variables (W6):
  ANTHROPIC_API_KEY  — required for LLMSupervisor; absent = MockSupervisor.
  ANTHROPIC_MODEL    — Anthropic model for LLMSupervisor (default: claude-3-5-haiku-20241022).
  LANGSMITH_API_KEY  — enables LangSmith tracing (optional; absent = no-op).
  LANGSMITH_TRACING  — set to "true" to force-enable tracing.
  LANGSMITH_PROJECT  — LangSmith project name (default: "mempill-showcase").
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
    use_crewai: bool = False,
    llm=None,
):
    """Build the full ExecAssistant LangGraph app.

    Returns (app, adapter) so callers can seed data or inspect state after the graph.

    Args:
        in_memory:   Use in-memory mempill engine (True for tests/demos).
        classifier:  SupervisorClassifier implementation; defaults to MockSupervisor.
        use_crewai:  False (default) → deterministic shell path (no API key, CI-safe).
                     True → CrewAI crews are injected into crew_a/b/c nodes.
                     The shell heuristics remain as a fallback if kickoff() fails.
        llm:         LiteLLM model string (e.g. "anthropic/claude-3-5-sonnet-20241022")
                     or a crewai.LLM instance.  Forwarded to build_crews().
                     Ignored when use_crewai=False.
    """
    from mempill_showcase.frameworks.langgraph.graph import build_graph

    adapter = build_mempill_adapter(in_memory=in_memory)
    tools = build_tools(adapter)

    crews = None
    if use_crewai:
        from mempill_showcase.frameworks.crewai.crews import build_crews
        crews = build_crews(adapter=adapter, tools=tools, llm=llm)

    app = build_graph(adapter=adapter, tools=tools, classifier=classifier, crews=crews)
    return app, adapter


def build_app(
    use_crewai: bool = False,
    classifier=None,
    adapter=None,
    llm=None,
):
    """W6 single-call factory: assemble the runnable ExecAssistant graph.

    This is the canonical entry point for both deterministic (test/demo) and
    live (API key) graph assembly.

    Args:
        use_crewai:  False (default) → deterministic shell path (no API key, CI-safe).
                     True → inject CrewAI crews into crew_a/b/c nodes.
        classifier:  SupervisorClassifier implementation.
                     None (default) → MockSupervisor (deterministic, no API key).
                     Pass LLMSupervisor(...) for the live path.
                     The function NEVER auto-selects LLMSupervisor — the caller
                     must explicitly opt in to avoid surprise API calls.
        adapter:     Pre-built MempillAdapter to reuse (e.g. a pre-seeded adapter
                     from a test or CLI). When None, a fresh in-memory adapter
                     is created internally.
        llm:         LiteLLM model string or crewai.LLM instance; forwarded to
                     build_crews() when use_crewai=True. Ignored otherwise.

    Returns:
        (app, adapter) — compiled LangGraph app + the MempillAdapter instance.
        The caller can use adapter to seed data, inspect state, or run
        bi-temporal queries after the graph executes.

    Example — deterministic (no API key):
        app, adapter = build_app()
        result = app.invoke({"user_input": "...", "agent_id": "..."}, config)

    Example — live LLM supervisor:
        from mempill_showcase.frameworks.langgraph.supervisor_node import LLMSupervisor
        app, adapter = build_app(classifier=LLMSupervisor())
        result = app.invoke({"user_input": "...", "agent_id": "..."}, config)
    """
    import os
    from mempill_showcase.frameworks.langgraph.graph import build_graph

    if adapter is None:
        adapter = build_mempill_adapter(in_memory=True)

    tools = build_tools(adapter)

    crews = None
    if use_crewai:
        from mempill_showcase.frameworks.crewai.crews import build_crews
        crews = build_crews(adapter=adapter, tools=tools, llm=llm)

    app = build_graph(adapter=adapter, tools=tools, classifier=classifier, crews=crews)
    return app, adapter
