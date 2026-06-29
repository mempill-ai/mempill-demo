"""
mempill_showcase.frameworks.langgraph.state — ExecAssistantState TypedDict.

Single state object threaded through every node in the LangGraph StateGraph.
Designed so each node reads what it needs and writes only its slice.

Fields:
  Core input
    user_input    — raw user turn text
    agent_id      — mempill session owner (e.g. "jordan-park-001")

  Routing
    intent        — classified intent string (see IntentLabel)
    route         — next node target after supervisor ("crew_a" | "crew_b" | "crew_c" | "hitl")

  Crew payloads
    recall_result — JSON string returned by the last MempillRecallTool invocation
    write_result  — JSON string returned by the last MempillRememberTool invocation
    audit_result  — JSON string returned by the last MempillAuditTool invocation
    briefing_text — human-readable output from Crew C (scheduling / briefing)

  HITL
    pending_contested — dict describing the contested write that triggered HITL:
        {
          "subject":      "alice-chen",
          "predicate":    "employer",
          "incumbent":    {"value": ..., "valid_from_display": ..., "claim_ref": ...},
          "challenger":   {"value": ..., "valid_from_display": ..., "claim_ref": ...},
          "claim_refs":   [<contested claim refs>],
        }
        None → no pending HITL gate.
    hitl_verdict  — verdict returned by the human via Command(resume=...):
        "Affirm" | "Deny" | "Abstain"
        None before the HITL node resolves.
    hitl_resolved_belief — JSON string of the post-resolution recall (or None).

  Session output
    output_text   — final human-facing output (set by the last active node)
    error         — non-None if a node encountered an unrecoverable error
"""
from __future__ import annotations

from typing import Any, Optional
from typing_extensions import TypedDict


class IntentLabel:
    """Canonical intent strings — used by supervisor and routing edges."""
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

    # ── Routing ───────────────────────────────────────────────────────────────
    intent: str               # one of IntentLabel constants
    route: str                # "crew_a" | "crew_b" | "crew_c" | "hitl" | "end"

    # ── Crew payloads ─────────────────────────────────────────────────────────
    recall_result: Optional[str]   # JSON from MempillRecallTool
    write_result: Optional[str]    # JSON from MempillRememberTool
    audit_result: Optional[str]    # JSON from MempillAuditTool
    briefing_text: Optional[str]   # human-readable output from Crew C

    # ── HITL ──────────────────────────────────────────────────────────────────
    pending_contested: Optional[dict[str, Any]]  # see docstring; None = no gate
    hitl_verdict: Optional[str]                  # "Affirm" | "Deny" | "Abstain"
    hitl_resolved_belief: Optional[str]          # JSON of post-resolution recall

    # ── Session output ────────────────────────────────────────────────────────
    output_text: Optional[str]
    error: Optional[str]
