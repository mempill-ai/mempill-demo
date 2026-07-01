"""
mempill_showcase.tools.mempill_audit_tool — MempillAuditTool

LangChain BaseTool wrapping the MempillAdapter audit ledger.

Returns the chronological audit ledger for an agent_id — every claim write,
succession, Contested event, and oracle resolution, with timestamps and rationale.
Used by Crew C (Scheduling & Briefing) for compliance queries (T-08).
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from mempill_showcase.adapters.memory.mempill_adapter import MempillAdapter
from mempill_showcase.observability import traceable_mempill

log = logging.getLogger(__name__)


class AuditInput(BaseModel):
    agent_id: str = Field(description="Agent/user session ID (e.g. 'jordan-park-001')")
    limit: int = Field(default=50, description="Maximum number of audit entries to return")
    claim_ref: Optional[str] = Field(
        default=None,
        description="Optional UUID to filter audit entries to a specific claim",
    )


class MempillAuditTool(BaseTool):
    """Return the mempill audit ledger for an agent.

    Provides the chronological log of every claim ingested, superseded,
    contested, or resolved — with recorded_at timestamps, dispositions,
    and rationale notes. Supports compliance queries (T-08: full belief-state
    reconstruction for a past moment) when combined with bi-temporal recall.

    Returns a JSON dict with an 'entries' list. Each entry contains:
      claim_ref, event_kind, disposition, recorded_at, rationale.
    """

    name: str = "mempill_audit"
    description: str = (
        "Return the mempill audit ledger for an agent session. "
        "Supply agent_id and optional limit / claim_ref filter. "
        "Returns JSON with an 'entries' list (claim_ref, event_kind, disposition, "
        "recorded_at, rationale)."
    )
    args_schema: Type[BaseModel] = AuditInput

    model_config = {"arbitrary_types_allowed": True}

    adapter: MempillAdapter

    @traceable_mempill(name="mempill.audit")
    def _run(
        self,
        agent_id: str,
        limit: int = 50,
        claim_ref: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        log.debug(
            "MempillAuditTool: agent=%s limit=%d claim_ref=%s",
            agent_id, limit, claim_ref,
        )

        entries = self.adapter.audit(agent_id, limit=limit)

        # Optionally filter to a single claim_ref
        if claim_ref:
            entries = [e for e in entries if e.claim_ref == claim_ref]

        result = {
            "agent_id": agent_id,
            "entry_count": len(entries),
            "entries": [
                {
                    "claim_ref": e.claim_ref,
                    "event_kind": e.event_kind,
                    "disposition": e.disposition,
                    "recorded_at": e.recorded_at,
                    "rationale": e.rationale,
                }
                for e in entries
            ],
        }
        log.debug("MempillAuditTool returned %d entries", len(entries))
        return json.dumps(result, default=str)

    async def _arun(
        self,
        agent_id: str,
        limit: int = 50,
        claim_ref: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        return self._run(
            agent_id=agent_id,
            limit=limit,
            claim_ref=claim_ref,
            **kwargs,
        )
