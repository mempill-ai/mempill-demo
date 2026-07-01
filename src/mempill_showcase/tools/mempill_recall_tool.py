"""
mempill_showcase.tools.mempill_recall_tool — MempillRecallTool

LangChain BaseTool wrapping the MempillAdapter recall + bi-temporal query_at path.

Modes:
  - current recall (no valid_at, no as_of_tx_time): returns the open-ended belief.
  - bi-temporal query_at: pins valid_at (world-time axis) and/or as_of_tx_time
    (transaction-time axis) — both axes are independent.

Returns a structured JSON result including:
  - status: "Resolved" | "Contested" | "TimingUncertain" | "NoBelief"
  - value: the resolved value, or None if Contested/NoBelief
  - valid_from_display: honest pre-rendered date at original granularity
    (e.g. "2025-02" for Month, "2023" for Year — from the engine)
  - valid_until_display: same for the end bound
  - is_contested: bool — callers must check this before acting on value
  - alternatives: list of competing candidates (when Contested/Conflict)
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from mempill_showcase.adapters.memory.mempill_adapter import MempillAdapter
from mempill_showcase.observability import emit_contested_span, traceable_mempill

log = logging.getLogger(__name__)


class RecallInput(BaseModel):
    agent_id: str = Field(description="Agent/user session ID (e.g. 'jordan-park-001')")
    subject: str = Field(description="Canonical entity key (e.g. 'alice-chen')")
    predicate: str = Field(description="Canonical predicate key (e.g. 'city')")
    valid_at: Optional[str] = Field(
        default=None,
        description=(
            "ISO-8601 UTC instant for valid-time axis — 'what was true in the world at this point?' "
            "None = current valid-time (most recent open claim)."
        ),
    )
    as_of_tx_time: Optional[str] = Field(
        default=None,
        description=(
            "ISO-8601 UTC instant for transaction-time axis — 'what did the system know then?' "
            "None = use all ingested claims (latest tx-time)."
        ),
    )


class MempillRecallTool(BaseTool):
    """Recall a belief from the mempill memory engine.

    Supports both current recall and bi-temporal point-in-time queries:
      - current recall: omit valid_at and as_of_tx_time.
      - valid-time query: supply valid_at to ask "what was true on that date?"
      - transaction-time query: supply as_of_tx_time to ask "what did we know then?"
      - full bi-temporal: supply both axes independently.

    Returns a JSON dict with status, value, valid_from_display, and is_contested.
    Callers MUST check is_contested before using value for any action output.
    """

    name: str = "mempill_recall"
    description: str = (
        "Recall a fact from the mempill memory engine. "
        "Supply canonical subject + predicate. Optionally pin valid_at (world-time) "
        "and/or as_of_tx_time (transaction-time) for historical queries. "
        "Returns JSON with status, value, valid_from_display, is_contested, and alternatives."
    )
    args_schema: Type[BaseModel] = RecallInput

    model_config = {"arbitrary_types_allowed": True}

    adapter: MempillAdapter

    @traceable_mempill(name="mempill.recall")
    def _run(
        self,
        agent_id: str,
        subject: str,
        predicate: str,
        valid_at: Optional[str] = None,
        as_of_tx_time: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        log.debug(
            "MempillRecallTool: agent=%s subject=%s predicate=%s valid_at=%s as_of=%s",
            agent_id, subject, predicate, valid_at, as_of_tx_time,
        )

        if valid_at is not None or as_of_tx_time is not None:
            belief = self.adapter.query_at(
                agent_id, subject, predicate,
                valid_at=valid_at,
                as_of_tx_time=as_of_tx_time,
            )
        else:
            belief = self.adapter.recall(agent_id, subject, predicate)

        alternatives = [
            {
                "value": alt.value,
                "conf": alt.conf,
                "valid_from_display": alt.vt_start_display,
                "valid_until_display": alt.vt_end_display,
                "claim_ref": alt.claim_ref,
            }
            for alt in belief.alternatives
        ]

        result = {
            "subject": belief.subject,
            "predicate": belief.predicate,
            "value": belief.value,
            "status": belief.status,
            "is_contested": belief.is_contested(),
            "is_resolved": belief.is_resolved(),
            "conf": belief.conf,
            "valid_from_display": belief.vt_start_display,
            "valid_until_display": belief.vt_end_display,
            "provenance": belief.provenance,
            "claim_ref": belief.claim_ref,
            "alternatives": alternatives,
        }
        log.debug("MempillRecallTool result: %s", result)

        # ── LangSmith: emit dedicated mempill.contested span on contested recall ──
        if belief.is_contested():
            try:
                alts = belief.alternatives or []
                inc = alts[0].__dict__ if alts else {}
                chal = alts[1].__dict__ if len(alts) > 1 else {}
                emit_contested_span(
                    subject=subject,
                    predicate=predicate,
                    incumbent={"value": inc.get("value"), "valid_from_display": inc.get("vt_start_display"), "claim_ref": inc.get("claim_ref")},
                    challenger={"value": chal.get("value"), "valid_from_display": chal.get("vt_start_display"), "claim_ref": chal.get("claim_ref")},
                    agent_id=agent_id,
                    extra={"valid_at": valid_at, "as_of_tx_time": as_of_tx_time},
                )
            except Exception as _exc:
                log.debug("MempillRecallTool: contested span error suppressed: %s", _exc)

        return json.dumps(result, default=str)

    async def _arun(
        self,
        agent_id: str,
        subject: str,
        predicate: str,
        valid_at: Optional[str] = None,
        as_of_tx_time: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        return self._run(
            agent_id=agent_id,
            subject=subject,
            predicate=predicate,
            valid_at=valid_at,
            as_of_tx_time=as_of_tx_time,
            **kwargs,
        )
