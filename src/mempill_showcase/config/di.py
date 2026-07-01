"""
mempill_showcase.config.di — minimal dependency injection factory.

Builds the engine + chosen adapter (mempill or naive) based on settings.
All mempill imports are deferred to this module and the mempill_adapter.

Public API:
  build_mempill_adapter(in_memory, oracle_backed, db_path)
    Build a MempillAdapter wrapping a real mempill engine.

  build_naive_adapter()
    Build a NaiveAdapter (no mempill, no valid-time, no provenance).

  build_agent_tools(adapter)
    Construct ShowcaseTools (all 7 ReAct agent tool instances).

  build_graph(adapter, tools, checkpointer, model_name)
    Build + compile the ReAct ExecAssistant agent.

  build_app(adapter=None, model_name=None)
    Single-call factory: (app, adapter).

  build_app_from_settings(settings)
    Settings-driven factory: selects adapter based on NAIVE_MODE.
    naive_mode=False → MempillAdapter → (app, adapter)
    naive_mode=True  → NaiveAdapter  → (None, adapter)

NAIVE_MODE:
  When NAIVE_MODE=true, build_app_from_settings returns (None, NaiveAdapter).
  NaiveAdapter does not have a LangGraph app.

Environment variables:
  ANTHROPIC_API_KEY  — required for the ReAct LLM; absent = no LLM calls.
  ANTHROPIC_MODEL    — Anthropic model string (default: claude-haiku-4-5).
  LANGSMITH_API_KEY  — enables LangSmith tracing (optional; absent = no-op).
  LANGSMITH_TRACING  — set to "true" to force-enable tracing.
  LANGSMITH_PROJECT  — LangSmith project name (default: "mempill-showcase").
  MEMPILL_DB_PATH    — optional persistent SQLite engine path.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional, Union

from mempill_showcase.adapters.memory.naive_adapter import NaiveAdapter

# Sentinel used as the default for checkpointer parameters so that an explicit
# None can be distinguished from "caller did not pass anything" (which should
# default to MemorySaver).  Mirrors the same sentinel in frameworks/langgraph/graph.py.
_SENTINEL = object()


def _adapter_from_settings(settings=None):
    """Return a MempillAdapter configured from settings (db_path, oracle_backed)."""
    if settings is None:
        from mempill_showcase.config.settings import get_settings
        settings = get_settings()
    return build_mempill_adapter(
        in_memory=True,
        oracle_backed=True,
        db_path=settings.mempill_db_path or None,
    )


class AdapterMode(str, Enum):
    MEMPILL = "mempill"
    NAIVE = "naive"


def build_mempill_adapter(
    in_memory: bool = True,
    oracle_backed: bool = True,
    db_path: Optional[str] = None,
):
    """Build a MempillAdapter wrapping a real mempill engine.

    Args:
        in_memory:     True → ephemeral in-memory engine (tests and demos).
                       Ignored when db_path is supplied.
        oracle_backed: True (default) → open_oracle_in_memory(HumanOracle()) so that
                       genuine conflicting Functional writes return QueuedForAdjudication
                       and queue for human adjudication via list_pending_adjudications /
                       submit_adjudication.
                       False → open_in_memory() (non-oracle).
                       Always True when db_path is supplied.
        db_path:       Optional filesystem path for a persistent SQLite-backed engine.
                       When set, opens via mempill.open_oracle(path, HumanOracle()).
    """
    import mempill
    from mempill_showcase.adapters.memory.mempill_adapter import MempillAdapter
    from mempill_showcase.core.ports.oracle import HumanOracle

    if db_path:
        import pathlib
        pathlib.Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        oracle = HumanOracle()
        engine = mempill.open_oracle(str(db_path), oracle)
        return MempillAdapter(engine)

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
    """Factory: return the adapter for the requested mode."""
    if mode == AdapterMode.MEMPILL:
        return build_mempill_adapter(in_memory=in_memory)
    elif mode == AdapterMode.NAIVE:
        return build_naive_adapter()
    else:
        raise ValueError(f"Unknown AdapterMode: {mode!r}")


def build_agent_tools(adapter):
    """Build a ShowcaseTools NamedTuple with the 7 ReAct agent tools.

    All tool instances are bound to the provided MempillAdapter and returned
    as a bundle for injection into build_graph().
    """
    from mempill_showcase.frameworks.langgraph.graph import ShowcaseTools
    from mempill_showcase.tools.audit_trail_tool import AuditTrailTool
    from mempill_showcase.tools.get_contested_tool import GetContestedTool
    from mempill_showcase.tools.list_pending_adjudications_tool import ListPendingAdjudicationsTool
    from mempill_showcase.tools.mempill_recall_tool import MempillRecallTool
    from mempill_showcase.tools.query_history_tool import QueryHistoryTool
    from mempill_showcase.tools.recall_as_of_tool import RecallAsOfTool
    from mempill_showcase.tools.recall_at_tool import RecallAtTool
    from mempill_showcase.tools.recall_subject_tool import RecallSubjectTool
    from mempill_showcase.tools.remember_fact_tool import RememberFactTool
    from mempill_showcase.tools.request_adjudication_tool import RequestAdjudicationTool
    from mempill_showcase.tools.resolve_adjudication_tool import ResolveAdjudicationTool

    recall_tool = MempillRecallTool(adapter=adapter)

    return ShowcaseTools(
        recall_subject_tool=RecallSubjectTool(adapter=adapter),
        recall_at_tool=RecallAtTool(adapter=adapter),
        recall_as_of_tool=RecallAsOfTool(adapter=adapter),
        remember_fact_tool=RememberFactTool(adapter=adapter),
        get_contested_tool=GetContestedTool(adapter=adapter),
        request_adjudication_tool=RequestAdjudicationTool(
            adapter=adapter,
            recall_tool=recall_tool,
        ),
        audit_trail_tool=AuditTrailTool(adapter=adapter),
        list_pending_adjudications_tool=ListPendingAdjudicationsTool(adapter=adapter),
        resolve_adjudication_tool=ResolveAdjudicationTool(adapter=adapter),
        query_history_tool=QueryHistoryTool(adapter=adapter),
    )


# Backward-compat alias: old code that calls build_tools() still works.
# build_tools was the W3 ShowcaseTools constructor (remember_tool, recall_tool, etc.).
# It is now replaced by build_agent_tools() but is kept to avoid import errors in
# any surviving callers (e.g. test fixtures that have not yet been migrated).
def build_tools(adapter):
    """Backward-compat alias for build_agent_tools(). Prefer build_agent_tools()."""
    return build_agent_tools(adapter)


def build_graph_from_adapter(adapter, checkpointer=_SENTINEL, model_name=None):
    """Build + compile the ReAct ExecAssistant agent from an adapter.

    Returns the compiled app.

    Checkpointer semantics (consistent with build_graph):
      checkpointer omitted (default) → MemorySaver() is used automatically.
      checkpointer=None              → compile without a checkpointer (Studio path).
      checkpointer=<instance>        → use the supplied checkpointer as-is.
    """
    from mempill_showcase.frameworks.langgraph.graph import build_graph
    from mempill_showcase.frameworks.langgraph.graph import _SENTINEL as _GRAPH_SENTINEL

    tools = build_agent_tools(adapter)
    # Translate the local sentinel to graph.py's sentinel so build_graph's
    # `if checkpointer is _SENTINEL` guard triggers correctly.
    cp = _GRAPH_SENTINEL if checkpointer is _SENTINEL else checkpointer
    return build_graph(adapter=adapter, tools=tools, checkpointer=cp, model_name=model_name)


def build_app(
    adapter=None,
    model_name: Optional[str] = None,
    # Legacy params accepted but ignored — kept for call-site compat
    use_crewai: bool = False,
    classifier=None,
    llm=None,
):
    """Single-call factory: assemble the runnable ExecAssistant ReAct agent.

    Args:
        adapter:     Pre-built MempillAdapter to reuse (or None → fresh in-memory).
        model_name:  Anthropic model string. Defaults to ANTHROPIC_MODEL env var or
                     settings.anthropic_model or "claude-haiku-4-5".
        use_crewai:  Ignored (CrewAI deleted). Accepted for call-site compat.
        classifier:  Ignored (no classifier in ReAct topology). Accepted for compat.
        llm:         Ignored. Accepted for call-site compat.

    Returns:
        (app, adapter) — compiled LangGraph app + the MempillAdapter instance.
    """
    if adapter is None:
        adapter = build_mempill_adapter(in_memory=True)

    if model_name is None:
        try:
            from mempill_showcase.config.settings import get_settings
            model_name = get_settings().anthropic_model
        except Exception:
            model_name = "claude-haiku-4-5"

    app = build_graph_from_adapter(adapter, model_name=model_name)
    return app, adapter


def build_app_from_settings(settings=None):
    """Settings-driven factory: select adapter based on NAIVE_MODE.

    Args:
        settings: a Settings instance (or compatible object with .naive_mode).
                  If None, loads from environment via get_settings().

    Returns:
        (app, adapter) — compiled ReAct agent + the chosen adapter.
        When naive_mode=True, app is None (NaiveAdapter has no LangGraph app).
    """
    if settings is None:
        from mempill_showcase.config.settings import get_settings
        settings = get_settings()

    from mempill_showcase.observability import configure_tracing_from_settings
    configure_tracing_from_settings(settings)

    if settings.naive_mode:
        adapter = build_naive_adapter()
        return None, adapter
    else:
        adapter = _adapter_from_settings(settings)
        return build_app(adapter=adapter, model_name=settings.anthropic_model)


# ── Legacy build_langgraph (backward-compat) ─────────────────────────────────

def build_langgraph(
    in_memory: bool = True,
    classifier=None,
    use_crewai: bool = False,
    llm=None,
):
    """Backward-compat shim for build_langgraph().  Delegates to build_app()."""
    adapter = build_mempill_adapter(in_memory=in_memory)
    return build_app(adapter=adapter)
