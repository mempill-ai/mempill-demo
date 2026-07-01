"""
mempill_showcase.tools.list_pending_adjudications_tool — ListPendingAdjudicationsTool

Thin wrapper over adapter.list_pending_adjudications() that surfaces the FULL
oracle queue to the agent/human — including stale rows that predate the current
turn's interrupt. Without this tool there is no agent-facing way to see pending
adjudications that never triggered a live HITL interrupt (e.g. a second/third
conflict written before a human replied to the first).

Returns a JSON dict with a lean list of ALL pending handles:
  [{handle_id, subject, predicate, incumbent_value, challenger_value,
    queued_at, status}]

Use resolve_adjudication(agent_id, handle_id, verdict) to clear a specific one.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from mempill_showcase.adapters.memory.mempill_adapter import MempillAdapter
from mempill_showcase.observability import traceable_mempill

log = logging.getLogger(__name__)


class ListPendingAdjudicationsInput(BaseModel):
    agent_id: str = Field(
        default="jordan-park-001",
        description="Agent/user session ID (e.g. 'jordan-park-001'). Defaults to 'jordan-park-001'.",
    )


class ListPendingAdjudicationsTool(BaseTool):
    """List ALL pending adjudications in the oracle queue, including stale ones.

    Use this to see outstanding conflicts awaiting a human decision — including
    rows that never triggered a live interrupt (e.g. a second or third contested
    write on the same subject/predicate before the first was resolved).

    Returns JSON: {"agent_id": ..., "pending_count": N, "pending": [
      {handle_id, subject, predicate, incumbent_value, challenger_value,
       queued_at, status}, ...
    ]}

    Raises no error when the engine is not oracle-backed — instead returns an
    empty list with a "note" explaining the queue is unavailable, so the agent
    can degrade gracefully rather than crash.
    """

    name: str = "list_pending_adjudications"
    description: str = (
        "List ALL pending adjudications in the oracle queue for an agent — "
        "including stale ones that never triggered a live interrupt. "
        "Supply optional agent_id (default 'jordan-park-001'). "
        "Returns JSON with pending_count and a pending list, each entry containing "
        "handle_id, subject, predicate, incumbent_value, challenger_value, "
        "queued_at, and status. Use resolve_adjudication(handle_id, verdict) to "
        "clear a specific entry."
    )
    args_schema: Type[BaseModel] = ListPendingAdjudicationsInput

    model_config = {"arbitrary_types_allowed": True}

    adapter: MempillAdapter

    @traceable_mempill(name="mempill.list_pending_adjudications")
    def _run(
        self,
        agent_id: str = "jordan-park-001",
        **kwargs: Any,
    ) -> str:
        log.debug("ListPendingAdjudicationsTool: agent=%s", agent_id)

        try:
            pending = self.adapter.list_pending_adjudications(agent_id)
        except AttributeError as exc:
            log.warning("ListPendingAdjudicationsTool: oracle queue unavailable: %s", exc)
            return json.dumps({
                "agent_id": agent_id,
                "pending_count": 0,
                "pending": [],
                "note": "Engine is not oracle-backed — no pending adjudication queue exists.",
            })

        lean = [
            {
                "handle_id": entry.get("handle_id"),
                "subject": entry.get("subject"),
                "predicate": entry.get("predicate"),
                "incumbent_value": entry.get("incumbent_value"),
                "challenger_value": entry.get("challenger_value"),
                "queued_at": entry.get("queued_at"),
                "status": entry.get("status"),
            }
            for entry in pending
        ]

        result = {
            "agent_id": agent_id,
            "pending_count": len(lean),
            "pending": lean,
        }
        log.debug("ListPendingAdjudicationsTool: returned %d pending entries", len(lean))
        return json.dumps(result, default=str)

    async def _arun(
        self,
        agent_id: str = "jordan-park-001",
        **kwargs: Any,
    ) -> str:
        return self._run(agent_id=agent_id, **kwargs)
