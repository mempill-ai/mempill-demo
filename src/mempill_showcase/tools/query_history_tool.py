"""
mempill_showcase.tools.query_history_tool — QueryHistoryTool

LangChain BaseTool wrapping MempillAdapter.query_history.

"What is the history of Acme's CEOs over time?" / "who held X before?" /
"succession of <role>" questions must be answered from THIS tool, not by
narrating raw audit_trail/recall_subject claim data. The engine's own fold is
chronologically ordered; each entry's valid_until reflects the engine's
effective window for that claim (its own stated end, or the start of the next
entry in the fold if that comes first) — it is NOT evidence that "a later
adjudication shortened" the claim. Overlapping claims are a genuine conflict,
not a resolved succession: consult query_memory (recall_subject/recall_at) for
the current conflict/Contested status of a (subject, predicate) line before
asserting which claim "won". Reconstructing the timeline by hand from
individually-stated claim dates instead of this tool's fold can still yield a
stale/incoherent narrative, since it ignores the engine's chronological
ordering and truncation.

Returns a JSON dict with:
  subject, predicate, entries: [{claim_ref, value, valid_from, valid_until,
  valid_from_display, valid_until_display, status, provenance, value_confidence}]

valid_from_display/valid_until_display are the engine's own honest, granularity-aware
render of valid_from/valid_until (e.g. "2025-12" for a month-granular fact, never a
fabricated day) — natively returned by mempill 0.4.0's query_history (engine PR #67).
Raw valid_from/valid_until are KEPT alongside the display strings (needed for
ordering/precision when granularity=instant) — the display fields are additions, not
replacements.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from mempill_showcase.adapters.memory.mempill_adapter import MempillAdapter
from mempill_showcase.core.domain.normalise import normalise_key
from mempill_showcase.observability import traceable_mempill

log = logging.getLogger(__name__)


class QueryHistoryInput(BaseModel):
    agent_id: str = Field(description="Agent/user session ID (e.g. 'jordan-park-001')")
    subject: str = Field(description="Entity key (e.g. 'acme-corp')")
    predicate: str = Field(description="Predicate key (e.g. 'ceo')")


class QueryHistoryTool(BaseTool):
    """Return the canonical, chronologically-folded history for (subject, predicate).

    Use this tool for ANY "history / succession / over time / who held X before"
    question about a specific (subject, predicate) pair — e.g. "what is the
    history of Acme's CEOs over time?", "who held the CTO role before Alice?".

    The returned entries follow the engine's own chronological fold: each
    entry's valid_until reflects the engine's effective window for that claim
    (its own stated end, or an adjacent entry's start where that comes first)
    — this is NOT evidence that a later adjudication shortened the claim, and
    overlapping windows on the same line are a genuine, possibly-unresolved
    conflict rather than an already-settled succession. For the current
    conflict/Contested status of a (subject, predicate) line, consult
    recall_subject/recall_at (query_memory), not this tool's status field
    alone. Report the entries' dates AS RETURNED — do NOT reconstruct the
    timeline from audit_trail or recall_subject's originally-stated claim
    dates, and do NOT re-derive valid_until from what a claim first said.

    Returns JSON with subject, predicate, and an entries list; each entry
    contains claim_ref, value, valid_from, valid_until, valid_from_display,
    valid_until_display, status ("Current"/"Superseded"), provenance,
    value_confidence.

    valid_from_display/valid_until_display are pre-rendered at the fact's
    ACTUAL recorded precision (e.g. "2025-12" for a month-granular date,
    "2023" for year). ALWAYS use these *_display strings verbatim when
    reporting dates to the user — NEVER expand them into, or otherwise state,
    a specific day the user/claim did not provide. These fields may be null
    for entries whose granularity could not be determined (older data); in
    that case fall back to reporting at month precision (YYYY-MM, derived by
    truncating valid_from/valid_until) rather than guessing a day.
    """

    name: str = "query_history"
    description: str = (
        "Return the CORRECT, chronologically-ordered, non-overlapping history "
        "timeline for a (subject, predicate) pair — e.g. 'history of Acme's CEOs "
        "over time', 'who held the CTO role before Alice?', succession questions. "
        "Supply agent_id, subject, predicate. The result is the engine's own "
        "chronological fold: each entry's valid_until reflects the engine's "
        "effective window for that claim, not necessarily what the claim "
        "originally stated — this does NOT mean a later adjudication 'shortened' "
        "it. Overlapping entries are a genuine conflict; consult "
        "recall_subject/recall_at for current Contested status. Report the "
        "entries exactly as returned — do NOT reconstruct history by hand from "
        "audit_trail or recall_subject, and do NOT narrate each claim's "
        "originally-stated dates. Returns JSON with "
        "subject, predicate, and an entries list (claim_ref, value, valid_from, "
        "valid_until, valid_from_display, valid_until_display, status, "
        "provenance, value_confidence). ALWAYS report dates using the "
        "*_display strings verbatim (e.g. '2025-12') — NEVER expand a month- or "
        "year-granular date into a fabricated specific day."
    )
    args_schema: Type[BaseModel] = QueryHistoryInput

    model_config = {"arbitrary_types_allowed": True}

    adapter: MempillAdapter

    @traceable_mempill(name="mempill.query_history")
    def _run(
        self,
        agent_id: str,
        subject: str,
        predicate: str,
        **kwargs: Any,
    ) -> str:
        subject = normalise_key(subject)
        predicate = normalise_key(predicate)

        log.debug(
            "QueryHistoryTool: agent=%s subject=%s predicate=%s",
            agent_id, subject, predicate,
        )

        raw_entries = self.adapter.query_history(agent_id, subject, predicate)

        entries = [
            {
                "claim_ref": e.get("claim_ref"),
                "value": e.get("value"),
                "valid_from": e.get("valid_from"),
                "valid_until": e.get("valid_until"),
                "valid_from_display": e.get("valid_from_display"),
                "valid_until_display": e.get("valid_until_display"),
                "status": e.get("status"),
                "provenance": e.get("provenance"),
                "value_confidence": e.get("value_confidence"),
            }
            for e in (raw_entries or [])
        ]

        result = {
            "subject": subject,
            "predicate": predicate,
            "entries": entries,
        }
        log.debug("QueryHistoryTool: %d entries for %s/%s", len(entries), subject, predicate)
        return json.dumps(result, default=str)

    async def _arun(
        self,
        agent_id: str,
        subject: str,
        predicate: str,
        **kwargs: Any,
    ) -> str:
        return self._run(
            agent_id=agent_id,
            subject=subject,
            predicate=predicate,
            **kwargs,
        )
