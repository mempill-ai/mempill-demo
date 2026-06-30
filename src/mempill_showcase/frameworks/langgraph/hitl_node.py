"""
mempill_showcase.frameworks.langgraph.hitl_node — disposition-driven HITL interrupt gate.

Design (W7 — real oracle):
  - The HITL gate is triggered when a MempillRememberTool write returns Contested /
    QueuedForAdjudication (with an oracle-backed engine, the disposition is
    QueuedForAdjudication; the adapter's is_contested flag is set in both cases).
  - Crew A detects is_contested in the write result, sets `pending_contested` in state,
    and the graph conditional edge routes to this node.
  - `hitl_node` calls LangGraph `interrupt(payload)` presenting the two candidates
    with their provenance.  The graph PAUSES here; LangGraph persists state.
  - On `Command(resume=<verdict>)` the graph resumes inside `hitl_node`.
    The node receives the verdict from `interrupt()` return value.
  - With an oracle-backed engine:
      1. adapter.list_pending_adjudications(agent_id) → finds the handle_id for
         the subject/predicate conflict in the oracle queue.
      2. adapter.submit_adjudication(agent_id, handle_id, verdict) → engine resolves:
           Affirm → challenger CommittedCheap, incumbent Superseded.
           Deny   → challenger Superseded, incumbent CommittedCheap.
      3. Post-resolution recall via recall_tool confirms Resolved status.
  - With a non-oracle engine (open_in_memory), falls back to reconcile().
  - After resolution the node recalls the resolved belief, stores it in
    `hitl_resolved_belief`, and clears `pending_contested`.
  - The W6 observability `mempill.contested` span is emitted before the interrupt.

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
from mempill_showcase.observability import emit_contested_span

if TYPE_CHECKING:
    from mempill_showcase.adapters.memory.mempill_adapter import MempillAdapter
    from mempill_showcase.tools.mempill_recall_tool import MempillRecallTool

log = logging.getLogger(__name__)


def make_hitl_node(adapter: "MempillAdapter", recall_tool: "MempillRecallTool"):
    """Factory: return a HITL LangGraph node bound to *adapter* and *recall_tool*.

    The node:
      1. Emits the mempill.contested observability span (W6).
      2. Builds a human-readable interrupt payload from `pending_contested`.
      3. Calls interrupt() — graph pauses; LangGraph delivers payload to the caller.
      4. Receives verdict from Command(resume=<verdict>) return value.
      5. Resolves via the REAL mempill oracle:
           adapter.list_pending_adjudications() → find handle_id
           adapter.submit_adjudication(handle_id, verdict) → engine resolves
         Falls back to adapter.reconcile() when the engine has no oracle queue.
      6. Recalls the resolved belief via recall_tool and writes `hitl_resolved_belief`.
      7. Clears `pending_contested`, sets `hitl_verdict`, updates `output_text`.
    """

    def hitl_node(state: ExecAssistantState) -> dict:
        agent_id = state.get("agent_id", "jordan-park-001")
        contested = state.get("pending_contested") or {}

        subject    = contested.get("subject", "unknown")
        predicate  = contested.get("predicate", "unknown")
        incumbent  = contested.get("incumbent") or {}
        challenger = contested.get("challenger") or {}
        claim_refs = contested.get("claim_refs") or []

        # ── W6 observability: emit contested span before interrupt ────────────
        try:
            emit_contested_span(
                subject=subject,
                predicate=predicate,
                incumbent=incumbent,
                challenger=challenger,
                agent_id=agent_id,
                extra={"claim_refs": claim_refs, "source": "hitl_node"},
            )
        except Exception as _obs_exc:
            log.debug("hitl_node: contested span suppressed: %s", _obs_exc)

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
            _resolve_via_oracle_or_reconcile(
                adapter, agent_id, subject, predicate, verdict, claim_refs
            )

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
            log.info(
                "hitl_node: human abstained — leaving %s/%s as Contested", subject, predicate
            )
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


def _resolve_via_oracle_or_reconcile(
    adapter: "MempillAdapter",
    agent_id: str,
    subject: str,
    predicate: str,
    verdict: str,
    claim_refs: list,
) -> None:
    """Resolve a contested claim using the oracle queue or fall back to reconcile.

    Oracle path (preferred — open_oracle_in_memory engines):
      1. adapter.list_pending_adjudications(agent_id) → list of pending entries.
      2. Find the entry whose subject/predicate matches.
      3. adapter.submit_adjudication(agent_id, handle_id, verdict).

    Fallback (non-oracle engines):
      adapter.reconcile(agent_id, [[subject, predicate]]).

    Both paths are idempotent if the conflict has already been resolved.
    """
    # Try the oracle path first
    oracle_available = hasattr(adapter._engine, "list_pending_adjudications")
    if oracle_available:
        try:
            pending = adapter.list_pending_adjudications(agent_id)
            # Collect ALL handles for this subject/predicate (multiple writes can queue
            # multiple entries when a conflict is written more than once before resolution)
            matching_handles: list[str] = []
            for entry in pending:
                if entry.get("subject") == subject and entry.get("predicate") == predicate:
                    h = entry.get("handle_id")
                    if h:
                        matching_handles.append(h)

            if matching_handles:
                for handle_id in matching_handles:
                    try:
                        result = adapter.submit_adjudication(agent_id, handle_id, verdict)
                        log.info(
                            "hitl_node: oracle submit handle=%s verdict=%s → disposition=%s",
                            handle_id[:8], verdict, result.get("disposition"),
                        )
                    except Exception as sub_exc:
                        log.warning(
                            "hitl_node: oracle submit handle=%s failed: %s",
                            handle_id[:8], sub_exc,
                        )
                return  # Oracle path complete — done
            else:
                log.info(
                    "hitl_node: oracle queue has no pending entry for %s/%s "
                    "(may already be resolved); falling through to reconcile",
                    subject, predicate,
                )
        except Exception as exc:
            log.warning(
                "hitl_node: oracle submit failed for %s/%s: %s — falling back to reconcile",
                subject, predicate, exc,
            )

    # Fallback: reconcile for non-oracle engines or when oracle entry not found
    try:
        for _ in range(3):
            resp = adapter.reconcile(agent_id, [[subject, predicate]])
            if resp.get("oracle_escalations", 0) == 0:
                break
        log.info("hitl_node: reconcile fallback complete for %s/%s", subject, predicate)
    except Exception as exc:
        log.warning("hitl_node: reconcile fallback failed for %s/%s: %s", subject, predicate, exc)
