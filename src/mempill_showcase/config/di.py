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

W7 additions:
  build_mempill_adapter(in_memory=True, oracle_backed=True)
    oracle_backed=True (NEW DEFAULT for showcase):
      Opens engine with open_oracle_in_memory(HumanOracle()) so that genuine
      conflicting Functional writes return QueuedForAdjudication (not bare Contested)
      and sit in the pending queue until submit_adjudication() resolves them.
      The adapter gains list_pending_adjudications() and submit_adjudication().
    oracle_backed=False:
      Falls back to open_in_memory() (non-oracle engine, W1-W6 behaviour).
      Genuine conflicts still return Contested; oracle methods raise AttributeError.
    Tests that need predictable CommittedCheap-only behaviour may pass
    oracle_backed=False explicitly.

W9 additions:
  build_app_from_settings(settings)
    Settings-driven factory that selects the adapter based on NAIVE_MODE.
    settings.naive_mode=False (default) → MempillAdapter (bi-temporal, oracle-backed)
    settings.naive_mode=True            → NaiveAdapter (last-write-wins, no bi-temporal)
    Returns (app, adapter) — same contract as build_app().

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


def build_mempill_adapter(in_memory: bool = True, oracle_backed: bool = True):
    """Build a MempillAdapter wrapping a real mempill engine.

    Args:
        in_memory:     True → ephemeral in-memory engine (for tests and demos).
                       False → raises NotImplementedError (file-backed not wired).
        oracle_backed: True (default) → open_oracle_in_memory(HumanOracle()) so that
                       genuine conflicting Functional writes return QueuedForAdjudication
                       and queue for human adjudication via list_pending_adjudications /
                       submit_adjudication.
                       False → open_in_memory() (non-oracle, W1-W6 behaviour).
                       Genuine conflicts return Contested; oracle methods raise AttributeError.
    """
    import mempill
    from mempill_showcase.adapters.memory.mempill_adapter import MempillAdapter
    from mempill_showcase.core.ports.oracle import HumanOracle

    if not in_memory:
        raise NotImplementedError("File-backed mempill engine not wired; use in_memory=True")

    if oracle_backed:
        oracle = HumanOracle()
        engine = mempill.open_oracle_in_memory(oracle)
    else:
        engine = mempill.open_in_memory()

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


# ── W9: build_app_from_settings (NAIVE_MODE toggle) ──────────────────────────

def build_app_from_settings(settings=None):
    """W9 settings-driven factory: select adapter based on NAIVE_MODE.

    Args:
        settings: a Settings instance (or compatible object with .naive_mode).
                  If None, loads from environment via get_settings().

    Returns:
        (app, adapter) — compiled LangGraph app + the chosen adapter.

    Adapter selection:
        settings.naive_mode=False (default) → MempillAdapter (bi-temporal, oracle-backed)
        settings.naive_mode=True            → NaiveAdapter (last-write-wins, no bi-temporal)

    Example — default (mempill):
        app, adapter = build_app_from_settings()
        # adapter is MempillAdapter

    Example — naive mode via env (NAIVE_MODE=true):
        import os; os.environ["NAIVE_MODE"] = "true"
        app, adapter = build_app_from_settings()
        # adapter is NaiveAdapter — watch it misbehave

    Example — explicit settings:
        from mempill_showcase.config.settings import Settings
        app, adapter = build_app_from_settings(Settings(naive_mode=True))
    """
    if settings is None:
        from mempill_showcase.config.settings import get_settings
        settings = get_settings()

    # Wire LangSmith tracing from settings (no-op without a key)
    from mempill_showcase.observability import configure_tracing_from_settings
    configure_tracing_from_settings(settings)

    if settings.naive_mode:
        # Naive mode: return the NaiveAdapter without a LangGraph app.
        # The NaiveAdapter is intentionally NOT a BiTemporalMemoryStore, so the
        # mempill-specific LangChain tools (MempillRememberTool etc.) cannot be
        # built against it. Callers in naive mode should interact with the adapter
        # directly (write_claim / recall / audit) or via naive_baseline.py.
        # We return (None, adapter) so callers can still inspect the adapter.
        adapter = build_naive_adapter()
        return None, adapter
    else:
        adapter = build_mempill_adapter(in_memory=True, oracle_backed=True)
        return build_app(adapter=adapter)


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
