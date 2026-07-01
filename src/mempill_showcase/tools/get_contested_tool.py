"""
mempill_showcase.tools.get_contested_tool — GetContestedTool

LangChain BaseTool that retrieves competing/alternative beliefs for a
(subject, predicate) pair using the existing adapter recall path.

When mempill returns Contested or Conflict, the BeliefView.alternatives list
contains the competing candidates (incumbent + challenger). This tool surfaces
those candidates in a structured form so the agent can explain the conflict and
decide whether to request human adjudication.

Uses the existing adapter.recall() method — no new engine calls required.

Returns a JSON dict with:
  - status: "Contested" | "Conflict" | "Resolved" | "NoBelief" | other
  - incumbent: {value, valid_from_display, valid_until_display, claim_ref} or None
  - challenger: {value, valid_from_display, valid_until_display, claim_ref} or None
  - alternatives: full list of competing candidates (when more than two)
  - is_contested: bool
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


class GetContestedInput(BaseModel):
    agent_id: str = Field(description="Agent/user session ID (e.g. 'jordan-park-001')")
    subject: str = Field(description="Entity key (e.g. 'alice-chen')")
    predicate: str = Field(description="Predicate key (e.g. 'employer', 'city')")


class GetContestedTool(BaseTool):
    """Retrieve competing/alternative beliefs for a contested subject/predicate.

    Use this tool when recall indicates a Contested or Conflict status, or when
    you need to surface the competing values before requesting human adjudication.

    Calls adapter.recall() and extracts the alternatives list from the BeliefView.
    When status is Contested/Conflict, returns incumbent (alternatives[0]) and
    challenger (alternatives[1]) in a structured form. When Resolved or NoBelief,
    returns an empty alternatives list with the current status.

    Returns JSON with status, is_contested, incumbent, challenger, and full alternatives.
    """

    name: str = "get_contested"
    description: str = (
        "Retrieve competing beliefs for a contested subject/predicate pair. "
        "Use when recall returns is_contested=true, or before requesting human adjudication. "
        "Supply agent_id, subject, and predicate. "
        "Returns JSON with status, is_contested, incumbent (first candidate), "
        "challenger (second candidate), and a full alternatives list."
    )
    args_schema: Type[BaseModel] = GetContestedInput

    model_config = {"arbitrary_types_allowed": True}

    adapter: MempillAdapter

    @traceable_mempill(name="mempill.get_contested")
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
            "GetContestedTool: agent=%s subject=%s predicate=%s",
            agent_id, subject, predicate,
        )

        belief = self.adapter.recall(agent_id, subject, predicate)

        alts = [
            {
                "value": a.value,
                "valid_from_display": a.vt_start_display,
                "valid_until_display": a.vt_end_display,
                "claim_ref": a.claim_ref,
                "conf": a.conf,
            }
            for a in (belief.alternatives or [])
        ]

        incumbent = alts[0] if len(alts) > 0 else None
        challenger = alts[1] if len(alts) > 1 else None

        result = {
            "subject": belief.subject,
            "predicate": belief.predicate,
            "status": belief.status,
            "is_contested": belief.is_contested(),
            "incumbent": incumbent,
            "challenger": challenger,
            "alternatives": alts,
        }
        log.debug("GetContestedTool result: status=%s alts=%d", belief.status, len(alts))
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
