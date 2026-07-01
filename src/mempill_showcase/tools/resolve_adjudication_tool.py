"""
mempill_showcase.tools.resolve_adjudication_tool — ResolveAdjudicationTool

Resolves a SPECIFIC pending adjudication by handle_id — the missing piece that
lets the agent/human clear a stale row without needing a live interrupt.

Queue-collapse policy (mirrors hitl_node._resolve_via_oracle_or_reconcile):
  Resolving one handle with Affirm/Deny determines a WINNER for the whole
  (agent_id, subject, predicate) subject-line — not just the one handle. To
  avoid leaving orphaned pending rows that would later re-open the same
  subject-line as permanently Contested (two Affirms == two live winners), this
  tool:
    1. Submits the requested verdict for the target handle.
    2. Derives the winning value (challenger for Affirm, incumbent for Deny).
    3. Sweeps ALL OTHER pending handles for the same (subject, predicate):
       any whose challenger_value == winner is Affirmed; every other one is
       Denied. This collapses the whole queue to a single live belief.
    4. Re-lists pending for the subject-line and asserts zero remain; any
       residual rows (should not normally happen) are Denied as a final sweep.

Abstain is NOT supported here — Abstain means "leave pending", so there is
nothing to resolve; callers wanting to defer should simply not call this tool.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from mempill_showcase.adapters.memory.mempill_adapter import MempillAdapter
from mempill_showcase.frameworks.langgraph.hitl_node import _normalize_verdict
from mempill_showcase.observability import traceable_mempill

log = logging.getLogger(__name__)


class ResolveAdjudicationInput(BaseModel):
    agent_id: str = Field(
        default="jordan-park-001",
        description="Agent/user session ID (e.g. 'jordan-park-001'). Defaults to 'jordan-park-001'.",
    )
    handle_id: str = Field(
        description="The pending adjudication handle_id to resolve (from list_pending_adjudications).",
    )
    verdict: str = Field(
        description=(
            "Human verdict: 'Affirm' (challenger wins) | 'Deny' (incumbent wins). "
            "Also accepts synonyms (yes/no/accept/reject/...) or a pasted candidate "
            "value matching the handle's challenger_value or incumbent_value."
        ),
    )


class ResolveAdjudicationTool(BaseTool):
    """Resolve a SPECIFIC pending adjudication by handle_id, collapsing the queue.

    Use this to clear a stale pending adjudication that was never surfaced by a
    live HITL interrupt — e.g. a second or third contested write on the same
    subject/predicate written before the first was resolved. Call
    list_pending_adjudications first to find the handle_id.

    Resolving one handle determines the WINNING value for the whole
    (subject, predicate) subject-line. This tool then sweeps every OTHER
    pending handle on that same subject-line: Affirms the one whose challenger
    matches the winner, Denies all the rest — so the subject-line collapses to
    a single live belief instead of leaving orphaned pending rows.

    Returns JSON with the outcome: disposition, winning_value, and how many
    other pending rows were swept.
    """

    name: str = "resolve_adjudication"
    description: str = (
        "Resolve a SPECIFIC pending adjudication by its handle_id — including "
        "stale ones not tied to the current turn's interrupt. Supply agent_id "
        "(default 'jordan-park-001'), handle_id (from list_pending_adjudications), "
        "and verdict ('Affirm' = challenger wins, 'Deny' = incumbent wins; synonyms "
        "and pasted candidate values also accepted). Automatically collapses any "
        "OTHER pending rows for the same subject/predicate so the belief converges "
        "to a single winner with zero stale pending rows left behind. Returns JSON "
        "with disposition, winning_value, and swept_count."
    )
    args_schema: Type[BaseModel] = ResolveAdjudicationInput

    model_config = {"arbitrary_types_allowed": True}

    adapter: MempillAdapter

    @traceable_mempill(name="mempill.resolve_adjudication")
    def _run(
        self,
        handle_id: str,
        verdict: str,
        agent_id: str = "jordan-park-001",
        **kwargs: Any,
    ) -> str:
        log.debug(
            "ResolveAdjudicationTool: agent=%s handle=%s raw_verdict=%r",
            agent_id, handle_id[:8] if handle_id else handle_id, verdict,
        )

        try:
            pending = self.adapter.list_pending_adjudications(agent_id)
        except AttributeError as exc:
            log.warning("ResolveAdjudicationTool: oracle queue unavailable: %s", exc)
            return json.dumps({
                "status": "unavailable",
                "message": "Engine is not oracle-backed — no pending adjudication queue exists.",
            })

        target: Optional[dict] = next(
            (e for e in pending if e.get("handle_id") == handle_id), None
        )
        if target is None:
            result = {
                "status": "not_found",
                "handle_id": handle_id,
                "message": (
                    f"No pending adjudication with handle_id={handle_id!r} "
                    "(it may already be resolved, or belongs to a different agent_id)."
                ),
            }
            log.warning("ResolveAdjudicationTool: handle not found: %s", handle_id)
            return json.dumps(result)

        subject = target.get("subject")
        predicate = target.get("predicate")
        incumbent_value = target.get("incumbent_value")
        challenger_value = target.get("challenger_value")

        normalized = _normalize_verdict(
            verdict,
            challenger_value=challenger_value,
            incumbent_value=incumbent_value,
        )
        if normalized not in ("Affirm", "Deny"):
            result = {
                "status": "invalid_verdict",
                "verdict": verdict,
                "normalized": normalized,
                "message": (
                    f"Unrecognised or unsupported verdict {verdict!r} for resolve_adjudication. "
                    "Use 'Affirm' (challenger wins) or 'Deny' (incumbent wins). "
                    "'Abstain' is not applicable here — it means leave the row pending."
                ),
            }
            return json.dumps(result)

        winning_value = challenger_value if normalized == "Affirm" else incumbent_value

        # 1. Resolve the target handle.
        target_result = self.adapter.submit_adjudication(agent_id, handle_id, normalized)
        log.info(
            "ResolveAdjudicationTool: target handle=%s verdict=%s -> disposition=%s",
            handle_id[:8], normalized, target_result.get("disposition"),
        )

        # 2. Queue-collapse sweep: resolve every OTHER pending row on the same
        #    (subject, predicate) so the subject-line converges to one winner.
        swept: list[dict] = []
        try:
            remaining = self.adapter.list_pending_adjudications(agent_id)
        except AttributeError:
            remaining = []

        for entry in remaining:
            if entry.get("handle_id") == handle_id:
                continue
            if entry.get("subject") != subject or entry.get("predicate") != predicate:
                continue
            other_handle = entry.get("handle_id")
            other_challenger = entry.get("challenger_value")
            sweep_verdict = "Affirm" if other_challenger == winning_value else "Deny"
            try:
                sweep_result = self.adapter.submit_adjudication(agent_id, other_handle, sweep_verdict)
                swept.append({
                    "handle_id": other_handle,
                    "challenger_value": other_challenger,
                    "verdict": sweep_verdict,
                    "disposition": sweep_result.get("disposition"),
                })
                log.info(
                    "ResolveAdjudicationTool: swept handle=%s challenger=%r verdict=%s -> disposition=%s",
                    other_handle[:8], other_challenger, sweep_verdict, sweep_result.get("disposition"),
                )
            except Exception as exc:
                log.warning(
                    "ResolveAdjudicationTool: sweep failed for handle=%s: %s",
                    other_handle[:8], exc,
                )

        # 3. Post-sweep assertion: confirm zero pending remain for this subject-line.
        #    Deny any residual rows (should not normally happen).
        residual_swept: list[str] = []
        try:
            final_pending = self.adapter.list_pending_adjudications(agent_id)
        except AttributeError:
            final_pending = []

        residual = [
            e for e in final_pending
            if e.get("subject") == subject and e.get("predicate") == predicate
        ]
        for entry in residual:
            residual_handle = entry.get("handle_id")
            try:
                self.adapter.submit_adjudication(agent_id, residual_handle, "Deny")
                residual_swept.append(residual_handle)
                log.warning(
                    "ResolveAdjudicationTool: residual pending row swept (Deny) handle=%s",
                    residual_handle[:8],
                )
            except Exception as exc:
                log.warning(
                    "ResolveAdjudicationTool: residual sweep failed for handle=%s: %s",
                    residual_handle[:8], exc,
                )

        winning_label = "challenger" if normalized == "Affirm" else "incumbent"
        result = {
            "status": "resolved",
            "handle_id": handle_id,
            "subject": subject,
            "predicate": predicate,
            "verdict": normalized,
            "disposition": target_result.get("disposition"),
            "winning_value": winning_value,
            "winning_label": winning_label,
            "swept_count": len(swept),
            "swept": swept,
            "residual_swept_count": len(residual_swept),
            "message": (
                f"Resolved {subject}/{predicate} handle {handle_id[:8]} by {normalized}: "
                f"{winning_label} wins → value is now {winning_value!r}. "
                f"Collapsed {len(swept)} other pending row(s) on the same subject-line."
            ),
        }
        log.info(
            "ResolveAdjudicationTool: %s/%s resolved verdict=%s winning=%r swept=%d residual=%d",
            subject, predicate, normalized, winning_value, len(swept), len(residual_swept),
        )
        return json.dumps(result, default=str)

    async def _arun(
        self,
        handle_id: str,
        verdict: str,
        agent_id: str = "jordan-park-001",
        **kwargs: Any,
    ) -> str:
        return self._run(handle_id=handle_id, verdict=verdict, agent_id=agent_id, **kwargs)
