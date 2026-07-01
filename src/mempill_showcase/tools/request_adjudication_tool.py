"""
mempill_showcase.tools.request_adjudication_tool — RequestAdjudicationTool

The 7th agent tool: Human-In-The-Loop adjudication trigger for the free-form
ReAct agent.

When the agent detects a contested write (remember_fact returns is_contested=True),
it:
  1. Calls get_contested to surface incumbent/challenger details.
  2. Calls this tool to pause execution and present the conflict to the human.

This tool calls LangGraph interrupt(payload) which pauses the graph.  On
Command(resume=<verdict>) the graph resumes inside the tool's _run() body.

Verdict normalisation:
  Accepts the same forgiving set as hitl_node._normalize_verdict:
    - Exact: Affirm / Deny / Abstain (case-insensitive)
    - Synonyms: yes/accept/challenger → Affirm; no/reject/incumbent → Deny;
      defer/skip → Abstain
    - Pasted candidate value → Affirm (challenger) or Deny (incumbent)

After normalisation the tool:
  - Submits the adjudication via adapter.submit_adjudication() (oracle path).
  - Falls back to adapter.reconcile() for non-oracle engines.
  - Recalls the resolved belief (non-null value required; falls back to
    adjudication outcome when recall returns TimingUncertain).
  - Returns a JSON string describing the resolution so the agent can
    compose the final answer.

The shared _normalize_verdict and _resolve_via_oracle_or_reconcile helpers
are re-used verbatim from hitl_node to avoid behaviour regression.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional, Type

from langchain_core.tools import BaseTool
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from mempill_showcase.adapters.memory.mempill_adapter import MempillAdapter
from mempill_showcase.frameworks.langgraph.hitl_node import (
    _normalize_verdict,
    _resolve_via_oracle_or_reconcile,
)
from mempill_showcase.tools.mempill_recall_tool import MempillRecallTool

log = logging.getLogger(__name__)


class RequestAdjudicationInput(BaseModel):
    agent_id: str = Field(description="Agent/user session ID (e.g. 'jordan-park-001')")
    subject: str = Field(description="Entity key of the contested fact (e.g. 'alice-chen')")
    predicate: str = Field(description="Predicate key of the contested fact (e.g. 'employer')")
    reason: str = Field(
        description=(
            "Human-readable explanation of why adjudication is needed. "
            "Should describe both the incumbent and challenger values and "
            "why they conflict. The human will see this in the interrupt payload."
        ),
    )
    incumbent_value: Optional[str] = Field(
        default=None,
        description="Current incumbent claim value (from get_contested output).",
    )
    challenger_value: Optional[str] = Field(
        default=None,
        description="Challenger claim value being disputed (from get_contested output).",
    )
    claim_refs: Optional[list[str]] = Field(
        default=None,
        description="List of claim ref UUIDs involved in the conflict (from get_contested).",
    )


class RequestAdjudicationTool(BaseTool):
    """Pause the agent and request human adjudication for a contested fact.

    Call this tool when remember_fact returns is_contested=True and you have
    retrieved the incumbent/challenger details via get_contested.

    The tool:
      1. Calls LangGraph interrupt(payload) — the graph PAUSES here.
         The interrupt payload is presented to the human showing the conflict.
      2. On Command(resume=<verdict>), normalises the verdict
         (forgiving: Affirm/Deny/Abstain + synonyms + pasted candidate value).
      3. Submits the adjudication via the oracle queue.
      4. Recalls the resolved belief to confirm.
      5. Returns a JSON string with the resolution so the agent can answer.

    Verdict semantics:
      Affirm  — challenger is correct; incumbent is superseded.
      Deny    — incumbent is correct; challenger is rejected.
      Abstain — defer; both remain Contested.
    """

    name: str = "request_adjudication"
    description: str = (
        "Pause and request human adjudication for a contested memory write. "
        "Call this ONLY when remember_fact returns is_contested=True and you "
        "have called get_contested to surface the competing values. "
        "Supply agent_id, subject, predicate, reason (explain the conflict), "
        "and optionally incumbent_value, challenger_value, claim_refs. "
        "The graph will pause until the human responds. After the human replies "
        "the tool resolves the conflict and returns the winning value as JSON."
    )
    args_schema: Type[BaseModel] = RequestAdjudicationInput

    model_config = {"arbitrary_types_allowed": True}

    adapter: MempillAdapter
    recall_tool: MempillRecallTool

    def _run(
        self,
        agent_id: str,
        subject: str,
        predicate: str,
        reason: str,
        incumbent_value: Optional[str] = None,
        challenger_value: Optional[str] = None,
        claim_refs: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> str:
        # Build interrupt payload — shown to the human
        interrupt_payload: dict[str, Any] = {
            "question": (
                f"Conflict on {subject}/{predicate}:\n"
                f"  Incumbent:  {incumbent_value!r}\n"
                f"  Challenger: {challenger_value!r}\n"
                f"Reason: {reason}\n"
                "Which is correct? Reply: 'Affirm' (challenger wins), "
                "'Deny' (incumbent wins), or 'Abstain' (defer)."
            ),
            "subject": subject,
            "predicate": predicate,
            "incumbent": {"value": incumbent_value},
            "challenger": {"value": challenger_value},
            "claim_refs": claim_refs or [],
            "reason": reason,
        }

        log.info(
            "RequestAdjudicationTool: interrupting for %s/%s — awaiting human verdict",
            subject, predicate,
        )

        # ── GRAPH PAUSES HERE ────────────────────────────────────────────────────
        raw_verdict: str = interrupt(interrupt_payload)
        # ── GRAPH RESUMES HERE with verdict from Command(resume=...) ─────────────

        log.info(
            "RequestAdjudicationTool: resumed raw_verdict=%r for %s/%s",
            raw_verdict, subject, predicate,
        )

        verdict = _normalize_verdict(
            raw_verdict,
            challenger_value=challenger_value,
            incumbent_value=incumbent_value,
        )
        log.info(
            "RequestAdjudicationTool: normalized verdict=%r (raw=%r) for %s/%s",
            verdict, raw_verdict, subject, predicate,
        )

        if verdict not in ("Affirm", "Deny", "Abstain"):
            result = {
                "verdict": raw_verdict,
                "normalized": verdict,
                "status": "invalid_verdict",
                "message": (
                    f"Unrecognised verdict {raw_verdict!r}. "
                    "Reply Affirm, Deny, or Abstain."
                ),
            }
            return json.dumps(result)

        if verdict == "Abstain":
            result = {
                "verdict": "Abstain",
                "status": "deferred",
                "subject": subject,
                "predicate": predicate,
                "message": (
                    f"Adjudication deferred for {subject}/{predicate}. "
                    "Both claims remain Contested."
                ),
            }
            return json.dumps(result)

        # Affirm or Deny — submit and recall
        winning_value, disposition = _resolve_via_oracle_or_reconcile(
            self.adapter, agent_id, subject, predicate, verdict, claim_refs or [],
        )

        resolved_value: Optional[str] = None
        resolved_status: Optional[str] = None
        try:
            raw_recall = self.recall_tool.invoke({
                "agent_id": agent_id,
                "subject": subject,
                "predicate": predicate,
            })
            rb = json.loads(raw_recall)
            resolved_value = rb.get("value")
            resolved_status = rb.get("status")
        except Exception as exc:
            log.warning("RequestAdjudicationTool: post-resolution recall failed: %s", exc)

        # Fallback to adjudication outcome when recall is TimingUncertain or null
        if not resolved_value and winning_value:
            resolved_value = winning_value
            resolved_status = "Resolved" if disposition == "CommittedCheap" else "Adjudicated"

        winning_label = "challenger" if verdict == "Affirm" else "incumbent"
        result = {
            "verdict": verdict,
            "status": "resolved",
            "subject": subject,
            "predicate": predicate,
            "winning_value": resolved_value,
            "winning_label": winning_label,
            "resolved_status": resolved_status,
            "disposition": disposition,
            "message": (
                f"Conflict on {subject}/{predicate} resolved by {verdict}: "
                f"{winning_label} wins → value is now {resolved_value!r}."
            ),
        }
        log.info(
            "RequestAdjudicationTool: resolved %s/%s verdict=%s value=%r",
            subject, predicate, verdict, resolved_value,
        )
        return json.dumps(result)

    async def _arun(
        self,
        agent_id: str,
        subject: str,
        predicate: str,
        reason: str,
        incumbent_value: Optional[str] = None,
        challenger_value: Optional[str] = None,
        claim_refs: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> str:
        # interrupt() reads LangGraph context via a contextvar that is NOT safe to
        # transfer to a thread executor.  We therefore re-implement the body here
        # in the async coroutine so that interrupt() runs on the event-loop thread
        # where the contextvar is live — identical logic to _run.
        interrupt_payload: dict[str, Any] = {
            "question": (
                f"Conflict on {subject}/{predicate}:\n"
                f"  Incumbent:  {incumbent_value!r}\n"
                f"  Challenger: {challenger_value!r}\n"
                f"Reason: {reason}\n"
                "Which is correct? Reply: 'Affirm' (challenger wins), "
                "'Deny' (incumbent wins), or 'Abstain' (defer)."
            ),
            "subject": subject,
            "predicate": predicate,
            "incumbent": {"value": incumbent_value},
            "challenger": {"value": challenger_value},
            "claim_refs": claim_refs or [],
            "reason": reason,
        }

        log.info(
            "RequestAdjudicationTool (_arun): interrupting for %s/%s — awaiting human verdict",
            subject, predicate,
        )

        # ── GRAPH PAUSES HERE ────────────────────────────────────────────────────
        raw_verdict: str = interrupt(interrupt_payload)
        # ── GRAPH RESUMES HERE with verdict from Command(resume=...) ─────────────

        log.info(
            "RequestAdjudicationTool (_arun): resumed raw_verdict=%r for %s/%s",
            raw_verdict, subject, predicate,
        )

        verdict = _normalize_verdict(
            raw_verdict,
            challenger_value=challenger_value,
            incumbent_value=incumbent_value,
        )
        log.info(
            "RequestAdjudicationTool (_arun): normalized verdict=%r (raw=%r) for %s/%s",
            verdict, raw_verdict, subject, predicate,
        )

        if verdict not in ("Affirm", "Deny", "Abstain"):
            result = {
                "verdict": raw_verdict,
                "normalized": verdict,
                "status": "invalid_verdict",
                "message": (
                    f"Unrecognised verdict {raw_verdict!r}. "
                    "Reply Affirm, Deny, or Abstain."
                ),
            }
            return json.dumps(result)

        if verdict == "Abstain":
            result = {
                "verdict": "Abstain",
                "status": "deferred",
                "subject": subject,
                "predicate": predicate,
                "message": (
                    f"Adjudication deferred for {subject}/{predicate}. "
                    "Both claims remain Contested."
                ),
            }
            return json.dumps(result)

        # Affirm or Deny — submit and recall
        winning_value, disposition = _resolve_via_oracle_or_reconcile(
            self.adapter, agent_id, subject, predicate, verdict, claim_refs or [],
        )

        resolved_value: Optional[str] = None
        resolved_status: Optional[str] = None
        try:
            raw_recall = self.recall_tool.invoke({
                "agent_id": agent_id,
                "subject": subject,
                "predicate": predicate,
            })
            rb = json.loads(raw_recall)
            resolved_value = rb.get("value")
            resolved_status = rb.get("status")
        except Exception as exc:
            log.warning("RequestAdjudicationTool (_arun): post-resolution recall failed: %s", exc)

        # Fallback to adjudication outcome when recall is TimingUncertain or null
        if not resolved_value and winning_value:
            resolved_value = winning_value
            resolved_status = "Resolved" if disposition == "CommittedCheap" else "Adjudicated"

        winning_label = "challenger" if verdict == "Affirm" else "incumbent"
        result = {
            "verdict": verdict,
            "status": "resolved",
            "subject": subject,
            "predicate": predicate,
            "winning_value": resolved_value,
            "winning_label": winning_label,
            "resolved_status": resolved_status,
            "disposition": disposition,
            "message": (
                f"Conflict on {subject}/{predicate} resolved by {verdict}: "
                f"{winning_label} wins → value is now {resolved_value!r}."
            ),
        }
        log.info(
            "RequestAdjudicationTool (_arun): resolved %s/%s verdict=%s value=%r",
            subject, predicate, verdict, resolved_value,
        )
        return json.dumps(result)
