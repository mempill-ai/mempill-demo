"""
mempill_showcase.tools.query_history_tool — QueryHistoryTool

LangChain BaseTool wrapping MempillAdapter.query_history.

"What is the history of Acme's CEOs over time?" / "who held X before?" /
"succession of <role>" questions must be answered from THIS tool, not by
narrating raw audit_trail/recall_subject claim data. The engine's own fold is
already chronologically ordered, truncated, and non-overlapping — a later
adjudication may have shortened an earlier entry's valid_until below what that
claim originally stated (e.g. Joan's stated end of 2025-11 truncated to 2025-01
because John was later Affirmed over the overlapping tail). Reconstructing the
timeline by hand from individually-stated claim dates reproduces the truncated-
away overlap and yields a stale/incoherent narrative.

Returns a JSON dict with:
  subject, predicate, entries: [{claim_ref, value, valid_from, valid_until,
  status, provenance, value_confidence}]
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

    The returned entries are ALREADY the correct, non-overlapping chronology:
    each entry's valid_from/valid_until/status reflects the engine's canonical
    fold, which may TRUNCATE an entry's originally-stated end date when a later
    adjudication resolved an overlap in favor of a successor. Report the
    entries' dates and status AS RETURNED — do NOT reconstruct the timeline
    from audit_trail or recall_subject's originally-stated claim dates, and do
    NOT re-derive valid_until from what a claim first said.

    Returns JSON with subject, predicate, and an entries list; each entry
    contains claim_ref, value, valid_from, valid_until, status
    ("Current"/"Superseded"), provenance, value_confidence.
    """

    name: str = "query_history"
    description: str = (
        "Return the CORRECT, chronologically-ordered, non-overlapping history "
        "timeline for a (subject, predicate) pair — e.g. 'history of Acme's CEOs "
        "over time', 'who held the CTO role before Alice?', succession questions. "
        "Supply agent_id, subject, predicate. The result is the engine's own "
        "adjudicated fold: each entry's valid_from/valid_until/status is "
        "authoritative and may already be truncated relative to what a claim "
        "originally stated. Report the entries exactly as returned — do NOT "
        "reconstruct history by hand from audit_trail or recall_subject, and do "
        "NOT narrate each claim's originally-stated dates. Returns JSON with "
        "subject, predicate, and an entries list (claim_ref, value, valid_from, "
        "valid_until, status, provenance, value_confidence)."
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
