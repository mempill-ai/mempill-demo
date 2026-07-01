"""
mempill_showcase.tools.recall_as_of_tool — RecallAsOfTool

LangChain BaseTool for transaction-time as-of queries.

"What did the system believe about Alice's employer as of yesterday?"
  → call recall_as_of(subject="alice-chen", predicate="employer",
                      as_of_tx_time="2026-06-29T00:00:00Z")

This tool pins ONLY the transaction-time axis (what did the system know at that
moment in time, regardless of what was actually true in the world). Valid-time is
always the current (most recent open claim per predicate).

For valid-time (world-time) point-in-time queries see RecallAtTool.
For full bi-temporal (both axes) see MempillRecallTool with both axes supplied.

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


class RecallAsOfInput(BaseModel):
    agent_id: str = Field(description="Agent/user session ID (e.g. 'jordan-park-001')")
    subject: str = Field(description="Entity key (e.g. 'alice-chen')")
    predicate: str = Field(description="Predicate key (e.g. 'city', 'employer')")
    as_of_tx_time: str = Field(
        description=(
            "ISO-8601 UTC instant — 'what did the system know at this moment?' "
            "Example: '2026-06-29T00:00:00Z'"
        ),
    )


class RecallAsOfTool(BaseTool):
    """Transaction-time as-of recall — what did the system believe at time T?

    Use this when the question is about the system's recorded knowledge at a past
    transaction time (NOT about what was true in the world):
      "What did we have on record for Alice's city last week?"
      "What did the system know about Bob's employer as of 2025-12-31?"

    Calls adapter.query_at(as_of_tx_time=...) — valid-time is always the current
    (most recent open claim). For world-time point-in-time queries (what was actually
    true on a given date?) use RecallAtTool instead.

    Returns JSON with status, value, valid_from_display, valid_until_display,
    is_contested, provenance, and claim_ref.
    Callers MUST check is_contested before using value in any action output.
    """

    name: str = "recall_as_of"
    description: str = (
        "Recall what the system believed about a fact at a specific transaction-time. "
        "Use when the question asks 'what did the system know as of <date>?' "
        "or 'what was recorded before <date>?'. "
        "Supply agent_id, subject, predicate, and as_of_tx_time (ISO-8601 UTC instant). "
        "Returns JSON with status, value, valid_from_display, valid_until_display, "
        "is_contested, provenance, claim_ref."
    )
    args_schema: Type[BaseModel] = RecallAsOfInput

    model_config = {"arbitrary_types_allowed": True}

    adapter: MempillAdapter

    @traceable_mempill(name="mempill.recall_as_of")
    def _run(
        self,
        agent_id: str,
        subject: str,
        predicate: str,
        as_of_tx_time: str,
        **kwargs: Any,
    ) -> str:
        subject = normalise_key(subject)
        predicate = normalise_key(predicate)

        log.debug(
            "RecallAsOfTool: agent=%s subject=%s predicate=%s as_of=%s",
            agent_id, subject, predicate, as_of_tx_time,
        )

        belief = self.adapter.query_at(
            agent_id, subject, predicate, as_of_tx_time=as_of_tx_time,
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
        log.debug("RecallAsOfTool result: %s", result)
        return json.dumps(result, default=str)

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("RecallAsOfTool does not support async")
