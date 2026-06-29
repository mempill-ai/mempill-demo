"""
mempill_showcase.frameworks.langgraph.graph — StateGraph assembly + build factory.

Graph topology:
  supervisor → (conditional) → crew_a | crew_b | crew_c
  crew_a     → (conditional) → hitl_node (if pending_contested) | END
  crew_b     → (conditional) → hitl_node (if pending_contested) | END
  crew_c     → END
  hitl_node  → END

Checkpointer:
  MemorySaver is attached so interrupt() / Command(resume=...) state persists
  across invocations on the same thread_id.

Factory:
  `build_graph(adapter, tools, classifier)` is the public entry point.
  - `adapter`    — MempillAdapter (the single mempill boundary)
  - `tools`      — ShowcaseTools NamedTuple (all W2 tool instances)
  - `classifier` — SupervisorClassifier impl (default: MockSupervisor)

DI integration:
  config/di.py calls build_graph(...) to wire the full graph in a single call.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from mempill_showcase.frameworks.langgraph.crew_nodes import (
    make_crew_a_node,
    make_crew_b_node,
    make_crew_c_node,
)
from mempill_showcase.frameworks.langgraph.hitl_node import make_hitl_node
from mempill_showcase.frameworks.langgraph.state import ExecAssistantState
from mempill_showcase.frameworks.langgraph.supervisor_node import (
    MockSupervisor,
    SupervisorClassifier,
    make_supervisor_node,
)

if TYPE_CHECKING:
    from mempill_showcase.adapters.memory.mempill_adapter import MempillAdapter
    from mempill_showcase.tools.date_parser_tool import DateParserTool
    from mempill_showcase.tools.mempill_audit_tool import MempillAuditTool
    from mempill_showcase.tools.mempill_recall_tool import MempillRecallTool
    from mempill_showcase.tools.mempill_remember_tool import MempillRememberTool
    from mempill_showcase.tools.rag_read_tool import RAGReadTool
    from mempill_showcase.tools.rag_write_tool import RAGWriteTool

log = logging.getLogger(__name__)


class ShowcaseTools(NamedTuple):
    """All W2 tool instances needed by the graph nodes."""
    remember_tool: "MempillRememberTool"
    recall_tool: "MempillRecallTool"
    audit_tool: "MempillAuditTool"
    date_parser: "DateParserTool"
    rag_write_tool: "RAGWriteTool"
    rag_read_tool: "RAGReadTool"


# ── Routing helpers ───────────────────────────────────────────────────────────

def _supervisor_router(state: ExecAssistantState) -> str:
    """Edge from supervisor: route to the correct crew node."""
    route = state.get("route", "crew_c")
    log.debug("supervisor_router: route=%s", route)
    return route


def _crew_a_router(state: ExecAssistantState) -> str:
    """Edge from crew_a: go to hitl if contested, else end."""
    pending = state.get("pending_contested")
    if pending:
        log.debug("crew_a_router: Contested detected → hitl_node")
        return "hitl"
    return END


def _crew_b_router(state: ExecAssistantState) -> str:
    """Edge from crew_b: go to hitl if contested, else end."""
    pending = state.get("pending_contested")
    if pending:
        log.debug("crew_b_router: Contested detected → hitl_node")
        return "hitl"
    return END


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph(
    adapter: "MempillAdapter",
    tools: ShowcaseTools,
    classifier: Optional[SupervisorClassifier] = None,
) -> "CompiledGraph":  # type: ignore[type-arg]
    """Build and compile the ExecAssistant StateGraph.

    Returns a compiled LangGraph app with MemorySaver checkpointer.
    The app supports interrupt() / Command(resume=...) for HITL flows.

    Args:
        adapter:    MempillAdapter — the single mempill boundary.
        tools:      ShowcaseTools NamedTuple with all W2 tool instances.
        classifier: SupervisorClassifier impl. Defaults to MockSupervisor()
                    (deterministic, no API key). Swap for LLMSupervisor in W6.
    """
    if classifier is None:
        classifier = MockSupervisor()

    # ── Build node functions ──────────────────────────────────────────────────
    supervisor_fn = make_supervisor_node(classifier)

    crew_a_fn = make_crew_a_node(
        remember_tool=tools.remember_tool,
        date_parser=tools.date_parser,
        adapter=adapter,
    )
    crew_b_fn = make_crew_b_node(
        remember_tool=tools.remember_tool,
        rag_write_tool=tools.rag_write_tool,
        adapter=adapter,
    )
    crew_c_fn = make_crew_c_node(
        recall_tool=tools.recall_tool,
        audit_tool=tools.audit_tool,
    )
    hitl_fn = make_hitl_node(
        adapter=adapter,
        recall_tool=tools.recall_tool,
    )

    # ── Assemble StateGraph ───────────────────────────────────────────────────
    g = StateGraph(ExecAssistantState)

    g.add_node("supervisor", supervisor_fn)
    g.add_node("crew_a", crew_a_fn)
    g.add_node("crew_b", crew_b_fn)
    g.add_node("crew_c", crew_c_fn)
    g.add_node("hitl_node", hitl_fn)

    # Entry point
    g.set_entry_point("supervisor")

    # Supervisor → crew (conditional on intent/route)
    g.add_conditional_edges(
        "supervisor",
        _supervisor_router,
        {
            "crew_a": "crew_a",
            "crew_b": "crew_b",
            "crew_c": "crew_c",
        },
    )

    # Crew A → hitl or end (conditional on pending_contested)
    g.add_conditional_edges(
        "crew_a",
        _crew_a_router,
        {
            "hitl": "hitl_node",
            END: END,
        },
    )

    # Crew B → hitl or end
    g.add_conditional_edges(
        "crew_b",
        _crew_b_router,
        {
            "hitl": "hitl_node",
            END: END,
        },
    )

    # Crew C always ends (read-only)
    g.add_edge("crew_c", END)

    # HITL → end (after resolution)
    g.add_edge("hitl_node", END)

    # ── Compile with MemorySaver checkpointer ─────────────────────────────────
    checkpointer = MemorySaver()
    app = g.compile(checkpointer=checkpointer)

    log.info("build_graph: ExecAssistant StateGraph compiled (classifier=%s)", type(classifier).__name__)
    return app
