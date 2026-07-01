"""
mempill_showcase.tools.recall_at_tool — RecallAtTool

LangChain BaseTool for valid-time (world-time) point-in-time queries.

"What was Alice's city on 2024-06-01?"
  → call recall_at(subject="alice-chen", predicate="city", valid_at="2024-06-01T00:00:00Z")

This tool pins ONLY the valid-time axis (what was true in the world at that date).
Transaction-time is always the latest (all ingested claims visible). For a
transaction-time-only query see RecallAsOfTool; for full bi-temporal see
MempillRecallTool with both axes.

Returns a JSON dict with:
  predicate, value, status, is_contested, valid_from_display,
  valid_until_display, provenance, claim_ref, conf.
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


class RecallAtInput(BaseModel):
    agent_id: str = Field(description="Agent/user session ID (e.g. 'jordan-park-001')")
    subject: str = Field(description="Entity key (e.g. 'alice-chen')")
    predicate: str = Field(description="Predicate key (e.g. 'city', 'employer')")
    valid_at: str = Field(
        description=(
            "ISO-8601 UTC instant — 'what was true in the world at this moment?' "
            "Example: '2024-06-01T00:00:00Z'"
        ),
    )


class RecallAtTool(BaseTool):
    """Point-in-time recall pinned on the valid-time (world-time) axis.

    Use this when the question asks what was true in the world at a specific date:
      "What city did Alice live in on June 1st 2024?"
      "What was Bob's employer in 2022?"

    Calls adapter.query_at(valid_at=...) — transaction-time is always the latest.
    For transaction-time as-of queries (what did the system know at time T?)
    use RecallAsOfTool instead.

    Returns JSON with status, value, valid_from_display, valid_until_display,
    is_contested, provenance, and claim_ref.
    Callers MUST check is_contested before using value in any action output.
    """

    name: str = "recall_at"
    description: str = (
        "Recall what a fact was at a specific point in world-time (valid-time axis). "
        "Use when the question asks 'what was true on <date>?' or 'as of <date>?'. "
        "Supply agent_id, subject, predicate, and valid_at (ISO-8601 UTC instant). "
        "Returns JSON with status, value, valid_from_display, valid_until_display, "
        "is_contested, provenance, claim_ref."
    )
    args_schema: Type[BaseModel] = RecallAtInput

    model_config = {"arbitrary_types_allowed": True}

    adapter: MempillAdapter

    @traceable_mempill(name="mempill.recall_at")
    def _run(
        self,
        agent_id: str,
        subject: str,
        predicate: str,
        valid_at: str,
        **kwargs: Any,
    ) -> str:
        subject = normalise_key(subject)
        predicate = normalise_key(predicate)

        log.debug(
            "RecallAtTool: agent=%s subject=%s predicate=%s valid_at=%s",
            agent_id, subject, predicate, valid_at,
        )

        belief = self.adapter.query_at(
            agent_id, subject, predicate, valid_at=valid_at,
        )

        result = {
            "subject": belief.subject,
            "predicate": belief.predicate,
            "value": belief.value,
            "status": belief.status,
            "is_contested": belief.is_contested(),
            "conf": belief.conf,
            "valid_from_display": belief.vt_start_display,
            "valid_until_display": belief.vt_end_display,
            "provenance": belief.provenance,
            "claim_ref": belief.claim_ref,
        }
        log.debug("RecallAtTool result: %s", result)
        return json.dumps(result, default=str)

    async def _arun(
        self,
        agent_id: str,
        subject: str,
        predicate: str,
        valid_at: str,
        **kwargs: Any,
    ) -> str:
        return self._run(
            agent_id=agent_id,
            subject=subject,
            predicate=predicate,
            valid_at=valid_at,
            **kwargs,
        )
