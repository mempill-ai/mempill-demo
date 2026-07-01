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
      1. adapter.list_pending_adjudications(agent_id) → finds ALL pending handles
         for the subject/predicate conflict in the oracle queue (there may be
         more than one stacked row — see RESEARCH_STALE_ADJUDICATIONS.md).
      2. QUEUE-COLLAPSE: determine the winning value from the human verdict on
         the decided row, then Affirm the row whose challenger == winner and
         Deny every OTHER pending row on that subject-line. This collapses the
         whole stack to a SINGLE live belief instead of leaving orphaned pending
         rows or producing multiple live winners.
      3. Post-resolution sweep asserts zero pending rows remain for the
         subject-line; any residual is defensively Denied.
      4. Post-resolution recall via recall_tool confirms Resolved status.
  - With a non-oracle engine (open_in_memory), falls back to reconcile().
  - Stale rows the CURRENT interrupt does not know about (never surfaced by any
    live interrupt) can be listed via the list_pending_adjudications tool and
    resolved directly via the resolve_adjudication tool, which applies the same
    queue-collapse policy.
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
        raw_verdict: str = interrupt(interrupt_payload)
        # --- GRAPH RESUMES HERE with verdict from Command(resume=...) ---

        log.info("hitl_node: resumed with raw verdict=%r for %s/%s", raw_verdict, subject, predicate)

        # ── Normalize / map the resume value to a canonical verdict ─────────
        # Accepts: exact keywords (Affirm/Deny/Abstain), synonyms, and pasted
        # candidate values (challenger_value → Affirm, incumbent_value → Deny).
        verdict = _normalize_verdict(
            raw_verdict,
            challenger_value=challenger.get("value"),
            incumbent_value=incumbent.get("value"),
        )
        log.info(
            "hitl_node: normalized verdict=%r (raw=%r) for %s/%s",
            verdict, raw_verdict, subject, predicate,
        )

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
        elif verdict == "_invalid_":
            log.warning(
                "hitl_node: unrecognised verdict raw=%r for %s/%s — not resolving",
                raw_verdict, subject, predicate,
            )
        # (no else — all canonical verdicts handled above)

        # ── Build output_text and decide whether to clear pending_contested ──
        if verdict in ("Affirm", "Deny"):
            # Genuine resolution: format the resolved-belief summary.
            output = f"HITL resolved {subject}/{predicate}:"
            if resolved_belief_json:
                try:
                    rb = json.loads(resolved_belief_json)
                    resolved_val = rb.get("value")
                    resolved_status = rb.get("status")
                    winning_label = (
                        "challenger" if verdict == "Affirm" else "incumbent"
                    )
                    output += (
                        f" {resolved_val!r} ({winning_label} wins, verdict={verdict})"
                    )
                    if rb.get("source") == "adjudication_outcome":
                        output += " [from adjudication outcome — temporal window ambiguous]"
                    else:
                        output += f" status={resolved_status}"
                except Exception:
                    output += f" verdict={verdict}"
            else:
                output += f" verdict={verdict}"
            return {
                "hitl_verdict": verdict,
                "hitl_resolved_belief": resolved_belief_json,
                "pending_contested": None,  # cleared — genuinely resolved
                "output_text": output,
            }

        elif verdict == "Abstain":
            output = (
                f"HITL deferred {subject}/{predicate} — still Contested (Abstain). "
                "Reply 'Affirm' (challenger wins), 'Deny' (incumbent wins), or 'Abstain' to defer."
            )
            return {
                "hitl_verdict": verdict,
                "hitl_resolved_belief": None,
                "pending_contested": contested,  # keep — not resolved
                "output_text": output,
            }

        else:  # verdict == "_invalid_"
            output = (
                f"Invalid verdict {raw_verdict!r}. "
                "Reply 'Affirm' (challenger wins), 'Deny' (incumbent wins), or 'Abstain'."
            )
            return {
                "hitl_verdict": raw_verdict,  # preserve raw for transparency
                "hitl_resolved_belief": None,
                "pending_contested": contested,  # keep — not resolved
                "output_text": output,
            }

    hitl_node.__name__ = "hitl_node"
    return hitl_node


def _normalize_verdict(
    raw: str,
    *,
    challenger_value: str | None,
    incumbent_value: str | None,
) -> str:
    """Map a raw human resume string to a canonical verdict.

    Mapping priority (first match wins):
      1. Exact keyword (case-insensitive, stripped): affirm / deny / abstain.
      2. Synonym: yes/accept/challenger → Affirm; no/reject/incumbent → Deny;
         defer/skip → Abstain.
      3. Pasted candidate value (case/space-insensitive):
           challenger_value → Affirm; incumbent_value → Deny.
      4. Unrecognized → returns sentinel "_invalid_".

    Args:
        raw: The raw string the human typed / pasted.
        challenger_value: The challenger claim value from the interrupt payload.
        incumbent_value: The incumbent claim value from the interrupt payload.

    Returns:
        "Affirm" | "Deny" | "Abstain" | "_invalid_"
    """
    normed = raw.strip().lower()
    # Remove surrounding quotes that Studio may inject
    normed = normed.strip("\"'")

    # 1. Exact keyword match
    if normed == "affirm":
        return "Affirm"
    if normed == "deny":
        return "Deny"
    if normed == "abstain":
        return "Abstain"

    # 2. Synonyms
    _affirm_synonyms = {"yes", "accept", "challenger", "approve", "confirm", "correct"}
    _deny_synonyms = {"no", "reject", "incumbent", "decline", "wrong", "incorrect"}
    _abstain_synonyms = {"defer", "skip", "later", "unsure", "unknown", "pass"}

    if normed in _affirm_synonyms:
        return "Affirm"
    if normed in _deny_synonyms:
        return "Deny"
    if normed in _abstain_synonyms:
        return "Abstain"

    # 3. Pasted candidate value — compare after collapsing whitespace
    def _canon(s: str | None) -> str:
        if s is None:
            return ""
        return " ".join(s.lower().split())

    normed_collapsed = " ".join(normed.split())
    if challenger_value and normed_collapsed == _canon(challenger_value):
        return "Affirm"
    if incumbent_value and normed_collapsed == _canon(incumbent_value):
        return "Deny"

    # 4. Unrecognized
    return "_invalid_"


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
        engine disposition string (e.g. "CommittedCheap") for the FIRST-decided
        handle.  Both may be None if the oracle path was not taken or the entry
        was not found.

    Oracle path (preferred — open_oracle_in_memory engines):
      1. adapter.list_pending_adjudications(agent_id) → list of pending entries.
      2. Find ALL entries whose subject/predicate matches (multiple writes can
         queue multiple pending rows when a conflict is written more than once
         before resolution — every row is frozen against the SAME original
         incumbent, see tasks/20-multiagent-showcase/RESEARCH_STALE_ADJUDICATIONS.md).
      3. QUEUE-COLLAPSE POLICY (fixes stale/orphaned pending rows):
         a. Determine the WINNER from the human verdict on the row that matches
            the current interrupt's claim_refs (or the first matching row if
            claim_refs don't disambiguate):
              Affirm → winner = that row's challenger_value
              Deny   → winner = that row's incumbent_value
         b. For EVERY pending row on this (agent_id, subject, predicate):
              - the row whose challenger_value == winner is Affirmed
              - every OTHER row is Denied (its stale challenger is superseded)
            This collapses the whole fan of `(incumbent, X)` rows to a SINGLE
            live belief instead of producing multiple live winners or leaving
            orphaned pending rows (see RESEARCH doc Q2/Q3).
      4. Post-resolution sweep: re-list pending for this subject/predicate and
         Deny any residual rows (defensive; should normally be zero after step 3).

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
            matching_entries: list[dict] = [
                entry for entry in pending
                if entry.get("subject") == subject
                and entry.get("predicate") == predicate
                and entry.get("handle_id")
            ]

            if matching_entries:
                # Prefer the entry whose challenger claim_ref matches the current
                # interrupt's claim_refs (the row this verdict was actually meant
                # for); fall back to the first matching entry when claim_refs are
                # absent or don't match anything (e.g. legacy callers).
                decided_entry = matching_entries[0]
                if claim_refs:
                    for entry in matching_entries:
                        payload = entry.get("request_payload") or {}
                        challenger_ref = (payload.get("challenger") or {}).get("claim_ref")
                        if challenger_ref and challenger_ref in claim_refs:
                            decided_entry = entry
                            break

                incumbent_value: str | None = decided_entry.get("incumbent_value")
                challenger_value: str | None = decided_entry.get("challenger_value")
                winning_value: str | None = (
                    challenger_value if verdict == "Affirm" else incumbent_value
                )

                first_disposition: str | None = None
                for entry in matching_entries:
                    handle_id = entry["handle_id"]
                    entry_challenger = entry.get("challenger_value")
                    # Queue-collapse: Affirm the row whose challenger IS the
                    # winner; Deny every other row on this subject-line so the
                    # belief converges to exactly one live claim.
                    row_verdict = "Affirm" if entry_challenger == winning_value else "Deny"
                    try:
                        result = adapter.submit_adjudication(agent_id, handle_id, row_verdict)
                        disposition = result.get("disposition")
                        if first_disposition is None:
                            first_disposition = disposition
                        log.info(
                            "hitl_node: oracle submit handle=%s row_verdict=%s → disposition=%s "
                            "incumbent=%r challenger=%r winning=%r",
                            handle_id[:8], row_verdict, disposition,
                            incumbent_value, entry_challenger, winning_value,
                        )
                    except Exception as sub_exc:
                        log.warning(
                            "hitl_node: oracle submit handle=%s failed: %s",
                            handle_id[:8], sub_exc,
                        )

                # Post-resolution sweep: assert zero pending rows remain for this
                # subject-line; Deny any residual (defensive — should be a no-op).
                try:
                    residual = adapter.list_pending_adjudications(agent_id)
                    for entry in residual:
                        if entry.get("subject") == subject and entry.get("predicate") == predicate:
                            residual_handle = entry.get("handle_id")
                            adapter.submit_adjudication(agent_id, residual_handle, "Deny")
                            log.warning(
                                "hitl_node: residual pending row swept (Deny) handle=%s for %s/%s",
                                residual_handle[:8] if residual_handle else residual_handle,
                                subject, predicate,
                            )
                except Exception as sweep_exc:
                    log.warning(
                        "hitl_node: post-resolution sweep failed for %s/%s: %s",
                        subject, predicate, sweep_exc,
                    )

                return winning_value, first_disposition  # Oracle path complete
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
