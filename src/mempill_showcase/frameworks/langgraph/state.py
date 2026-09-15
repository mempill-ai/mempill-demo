"""
mempill_showcase.frameworks.langgraph.state — ExecAssistantState TypedDict.

State for the ReAct agent graph.  The agent is a create_react_agent node; the
only fields the graph infrastructure needs beyond the built-in ``messages`` list
are:

  Core input
    user_input    — raw user turn text (still accepted for backward-compat wrappers)
    agent_id      — mempill session owner (e.g. "jordan-park-001")

  HITL (set by request_adjudication_tool after interrupt/resume)
    hitl_verdict         — canonical verdict after human resumes: "Affirm" | "Deny" | "Abstain"
    hitl_resolved_belief — JSON string of the post-resolution recall (or None)

  Session output
    output_text   — last assistant message text (convenience alias; populated by
                    the graph wrapper from the final AIMessage)
    error         — non-None if an unrecoverable error occurred

Removed (old classifier fields, not needed by ReAct):
  intent, route, recall_result, write_result, audit_result, briefing_text,
  pending_contested.

IntentLabel is retained as a re-export so existing imports from other modules
that reference it do not break during the transition period.
"""
from __future__ import annotations

from typing import Annotated, Any, Optional
from typing_extensions import TypedDict

from mempill_showcase.frameworks.langgraph.router_state import _safe_add_messages


class IntentLabel:
    """Retained for backward-compatibility only.  Not used by the ReAct agent."""
    UPDATE_CONTACT    = "UPDATE_CONTACT"
    RESEARCH          = "RESEARCH"
    PREPARE_BRIEFING  = "PREPARE_BRIEFING"
    RECALL_HISTORY    = "RECALL_HISTORY"
    COMPLIANCE_AUDIT  = "COMPLIANCE_AUDIT"

    ALL = frozenset([
        UPDATE_CONTACT,
        RESEARCH,
        PREPARE_BRIEFING,
        RECALL_HISTORY,
        COMPLIANCE_AUDIT,
    ])


class ExecAssistantState(TypedDict, total=False):
    # ── Core input ────────────────────────────────────────────────────────────
    user_input: str
    agent_id: str

    # ── LangGraph ReAct built-in: message list ────────────────────────────────
    # Annotated[..., _safe_add_messages] (TASK-33 item 3): mirrors
    # RouterState.messages' hardened accumulation reducer (router_state.py) —
    # merge/dedup-by-id across turns instead of last-write-wins, without
    # crashing on non-coercible junk items. NOTE: create_react_agent
    # (frameworks/langgraph/graph.py:build_graph) builds its own internal
    # agent graph with LangGraph's own built-in add_messages-based state
    # schema — it does NOT consume ExecAssistantState as a state_schema
    # today, so this annotation is currently inert for that path and exists
    # for (a) hitl_node.py's type hint and (b) any future caller that DOES
    # compile a graph against ExecAssistantState directly.
    messages: Annotated[list[Any], _safe_add_messages]

    # ── HITL (populated after interrupt/resume inside request_adjudication) ───
    hitl_verdict: Optional[str]           # "Affirm" | "Deny" | "Abstain"
    hitl_resolved_belief: Optional[str]   # JSON of post-resolution recall

    # ── Session output ────────────────────────────────────────────────────────
    output_text: Optional[str]
    error: Optional[str]
