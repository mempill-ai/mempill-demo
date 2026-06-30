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
        try:
            from mempill_showcase.config.settings import get_settings
            _default_agent_id = get_settings().mempill_agent_id
        except Exception:
            _default_agent_id = "jordan-park-001"
        agent_id = state.get("agent_id", _default_agent_id)
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
            # _resolve_via_oracle returns (winning_value, disposition) from the
            # adjudication outcome — reliable even when recall returns TimingUncertain.
            adjudication_winning_value, adjudication_disposition = _resolve_via_oracle_or_reconcile(
                adapter, agent_id, subject, predicate, verdict, claim_refs
            )

            # Recall the resolved belief to confirm — preferred source when non-null.
            post_recall_value: str | None = None
            post_recall_status: str | None = None
            try:
                raw_recall = recall_tool.invoke({
                    "agent_id": agent_id,
                    "subject": subject,
                    "predicate": predicate,
                })
                rb = json.loads(raw_recall)
                post_recall_value = rb.get("value")
                post_recall_status = rb.get("status")
                # Use the recall result as the primary source ONLY when it has a real value
                if post_recall_value:
                    resolved_belief_json = raw_recall
                    log.info(
                        "hitl_node: post-resolution recall → value=%r status=%s",
                        post_recall_value, post_recall_status,
                    )
            except Exception as exc:
                log.warning("hitl_node: post-resolution recall failed: %s", exc)

            # Fallback: if recall returned null/TimingUncertain, build resolved belief
            # from the adjudication outcome (the winning value is always known here).
            if not post_recall_value and adjudication_winning_value:
                fallback_belief = {
                    "subject": subject,
                    "predicate": predicate,
                    "value": adjudication_winning_value,
                    "status": "Resolved" if adjudication_disposition == "CommittedCheap" else "Adjudicated",
                    "disposition": adjudication_disposition,
                    "source": "adjudication_outcome",
                    "note": (
                        "Post-resolution recall returned TimingUncertain (undated conflict). "
                        f"Winner determined from adjudication: verdict={verdict}."
                    ),
                }
                resolved_belief_json = json.dumps(fallback_belief)
                log.info(
                    "hitl_node: recall TimingUncertain — using adjudication outcome: "
                    "value=%r disposition=%s verdict=%s",
                    adjudication_winning_value, adjudication_disposition, verdict,
                )

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
                resolved_val = rb.get("value")
                resolved_status = rb.get("status")
                output += f" → resolved: {subject}/{predicate} = {resolved_val!r} (verdict={verdict})"
                if rb.get("source") == "adjudication_outcome":
                    output += " [from adjudication outcome — temporal window ambiguous]"
                else:
                    output += f" status={resolved_status}"
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
) -> tuple[str | None, str | None]:
    """Resolve a contested claim using the oracle queue or fall back to reconcile.

    Returns:
        (winning_value, disposition) where winning_value is the value of the winning
        claim (challenger for Affirm, incumbent for Deny) and disposition is the
        engine disposition string (e.g. "CommittedCheap").  Both may be None if the
        oracle path was not taken or the entry was not found.

    Oracle path (preferred — open_oracle_in_memory engines):
      1. adapter.list_pending_adjudications(agent_id) → list of pending entries.
      2. Find the entry whose subject/predicate matches.
         From the entry: capture incumbent_value and challenger_value.
      3. adapter.submit_adjudication(agent_id, handle_id, verdict) → disposition.
      4. Derive winning_value from verdict:
           Affirm → challenger wins → winning_value = challenger_value
           Deny   → incumbent wins → winning_value = incumbent_value

    Fallback (non-oracle engines):
      adapter.reconcile(agent_id, [[subject, predicate]]).
      winning_value is not recoverable in this path → returns (None, None).

    Both paths are idempotent if the conflict has already been resolved.
    """
    # Try the oracle path first
    oracle_available = hasattr(adapter._engine, "list_pending_adjudications")
    if oracle_available:
        try:
            pending = adapter.list_pending_adjudications(agent_id)
            # Collect ALL handles for this subject/predicate (multiple writes can queue
            # multiple entries when a conflict is written more than once before resolution)
            matching_entries: list[dict] = []
            for entry in pending:
                if entry.get("subject") == subject and entry.get("predicate") == predicate:
                    h = entry.get("handle_id")
                    if h:
                        matching_entries.append(entry)

            if matching_entries:
                # Use the FIRST matching entry to determine the winning value.
                # challenger_value/incumbent_value are direct fields in the pending entry.
                first_entry = matching_entries[0]
                incumbent_value: str | None = first_entry.get("incumbent_value")
                challenger_value: str | None = first_entry.get("challenger_value")
                winning_value: str | None = challenger_value if verdict == "Affirm" else incumbent_value

                last_disposition: str | None = None
                for entry in matching_entries:
                    handle_id = entry["handle_id"]
                    try:
                        result = adapter.submit_adjudication(agent_id, handle_id, verdict)
                        last_disposition = result.get("disposition")
                        log.info(
                            "hitl_node: oracle submit handle=%s verdict=%s → disposition=%s "
                            "incumbent=%r challenger=%r winning=%r",
                            handle_id[:8], verdict, last_disposition,
                            incumbent_value, challenger_value, winning_value,
                        )
                    except Exception as sub_exc:
                        log.warning(
                            "hitl_node: oracle submit handle=%s failed: %s",
                            handle_id[:8], sub_exc,
                        )
                return winning_value, last_disposition  # Oracle path complete
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

    # Fallback: reconcile for non-oracle engines or when oracle entry not found.
    # winning_value is not recoverable in this path.
    try:
        for _ in range(3):
            resp = adapter.reconcile(agent_id, [[subject, predicate]])
            if resp.get("oracle_escalations", 0) == 0:
                break
        log.info("hitl_node: reconcile fallback complete for %s/%s", subject, predicate)
    except Exception as exc:
        log.warning("hitl_node: reconcile fallback failed for %s/%s: %s", subject, predicate, exc)

    return None, None
