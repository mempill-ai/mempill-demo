"""
mempill_showcase.frameworks.langgraph.hitl_node — disposition-driven HITL interrupt gate.

Design (Approved Decision 4):
  - The HITL gate is triggered when a MempillRememberTool write returns Contested.
  - Crew A detects Contested in the write result, sets `pending_contested` in state,
    and the graph conditional edge routes to this node.
  - `hitl_node` calls LangGraph `interrupt(payload)` presenting the two candidates
    with their provenance. The graph PAUSES here; LangGraph persists state.
  - On `Command(resume=<verdict>)` the graph resumes inside `hitl_node`.
    The node receives the verdict from `interrupt()` return value.
  - The node calls `adapter.reconcile(...)` or submits the adjudication through
    the mempill engine via the oracle (list_pending_adjudications / submit_adjudication)
    to resolve the Contested claim.
  - After resolution, the node recalls the resolved belief and stores it in
    `hitl_resolved_belief`. `pending_contested` is cleared (set to None).

Verdict semantics:
  "Affirm"  — challenger is correct; incumbent is superseded.
  "Deny"    — incumbent is correct; challenger is rejected.
  "Abstain" — leave both as Contested; do not write a resolution.
              (rare; only when the human genuinely cannot decide now)

The node does NOT use any LLM. The verdict comes from the human resume payload.
"""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from langgraph.types import interrupt

from mempill_showcase.frameworks.langgraph.state import ExecAssistantState

if TYPE_CHECKING:
    from mempill_showcase.adapters.memory.mempill_adapter import MempillAdapter
    from mempill_showcase.tools.mempill_recall_tool import MempillRecallTool

log = logging.getLogger(__name__)


def make_hitl_node(adapter: "MempillAdapter", recall_tool: "MempillRecallTool"):
    """Factory: return a HITL LangGraph node bound to *adapter* and *recall_tool*.

    The node:
      1. Builds a human-readable interrupt payload from `pending_contested`.
      2. Calls interrupt() — graph pauses; LangGraph delivers payload to the caller.
      3. Receives verdict from Command(resume=<verdict>) return value.
      4. Resolves via mempill oracle (submit_adjudication) or reconcile.
      5. Recalls the resolved belief and writes `hitl_resolved_belief`.
      6. Clears `pending_contested`, sets `hitl_verdict`, updates `output_text`.
    """

    def hitl_node(state: ExecAssistantState) -> dict:
        agent_id = state.get("agent_id", "jordan-park-001")
        contested = state.get("pending_contested") or {}

        subject   = contested.get("subject", "unknown")
        predicate = contested.get("predicate", "unknown")
        incumbent = contested.get("incumbent") or {}
        challenger = contested.get("challenger") or {}
        claim_refs = contested.get("claim_refs") or []

        # Build the interrupt payload — presented to the human.
        interrupt_payload: dict[str, Any] = {
            "question": (
                f"Conflict on {subject}/{predicate}:\n"
                f"  Incumbent:  {incumbent.get('value')!r} "
                f"(valid from {incumbent.get('valid_from_display')}, "
                f"source: {incumbent.get('provenance', 'unknown')})\n"
                f"  Challenger: {challenger.get('value')!r} "
                f"(valid from {challenger.get('valid_from_display')}, "
                f"source: {challenger.get('provenance', 'unknown')})\n"
                "Which is correct? Reply: 'Affirm' (challenger wins), "
                "'Deny' (incumbent wins), or 'Abstain' (defer)."
            ),
            "subject": subject,
            "predicate": predicate,
            "incumbent": incumbent,
            "challenger": challenger,
            "claim_refs": claim_refs,
        }

        log.info("hitl_node: interrupting for %s/%s — awaiting human verdict", subject, predicate)

        # --- GRAPH PAUSES HERE ---
        verdict: str = interrupt(interrupt_payload)
        # --- GRAPH RESUMES HERE with verdict from Command(resume=...) ---

        log.info("hitl_node: resumed with verdict=%r for %s/%s", verdict, subject, predicate)

        resolved_belief_json: str | None = None

        if verdict in ("Affirm", "Deny"):
            # Attempt oracle submission.  The pending adjudication must exist
            # in the engine for this to succeed.  We try to find it by subject/predicate.
            try:
                pending = adapter._engine.list_pending_adjudications({
                    "agent_id": agent_id,
                })
                adjudications = pending.get("adjudications") or []
                # Find the entry for our subject/predicate.
                handle_id = None
                for entry in adjudications:
                    e_subject = entry.get("subject") or entry.get("fact", {}).get("subject")
                    e_predicate = entry.get("predicate") or entry.get("fact", {}).get("predicate")
                    if e_subject == subject and e_predicate == predicate:
                        handle_id = entry.get("handle_id") or entry.get("id")
                        break

                if handle_id:
                    import mempill
                    evidence_prov = mempill.ProvenanceLabel.external_user_asserted()
                    adapter._engine.submit_adjudication({
                        "handle_id": handle_id,
                        "verdict": verdict,
                        "evidence_provenance": evidence_prov,
                    })
                    log.info(
                        "hitl_node: submitted adjudication handle=%s verdict=%s",
                        handle_id, verdict,
                    )
                else:
                    # No oracle handle — fall back to reconcile for non-overlapping windows
                    log.info(
                        "hitl_node: no oracle handle found for %s/%s; falling back to reconcile",
                        subject, predicate,
                    )
                    reconcile_req = {
                        "agent_id": agent_id,
                        "subject_lines": [[subject, predicate]],
                    }
                    for _ in range(3):
                        resp = adapter._engine.reconcile(reconcile_req)
                        if resp.get("oracle_escalations", 0) == 0:
                            break

            except Exception as exc:
                log.warning("hitl_node: adjudication/reconcile failed: %s", exc)

            # Recall the resolved belief to confirm
            try:
                resolved_belief_json = recall_tool.invoke({
                    "agent_id": agent_id,
                    "subject": subject,
                    "predicate": predicate,
                })
            except Exception as exc:
                log.warning("hitl_node: post-resolution recall failed: %s", exc)

        elif verdict == "Abstain":
            log.info("hitl_node: human abstained — leaving %s/%s as Contested", subject, predicate)
        else:
            log.warning("hitl_node: unrecognised verdict %r — treating as Abstain", verdict)

        output = f"HITL resolved {subject}/{predicate}: verdict={verdict}"
        if resolved_belief_json:
            try:
                rb = json.loads(resolved_belief_json)
                output += f" → belief={rb.get('value')!r} status={rb.get('status')}"
            except Exception:
                pass

        return {
            "hitl_verdict": verdict,
            "hitl_resolved_belief": resolved_belief_json,
            "pending_contested": None,
            "output_text": output,
        }

    hitl_node.__name__ = "hitl_node"
    return hitl_node
