"""
mempill_showcase.tools.audit_trail_tool — AuditTrailTool

LangChain BaseTool wrapping the MempillAdapter audit ledger — thin wrapper
over the same adapter.audit() path as the existing MempillAuditTool.

The existing MempillAuditTool requires an agent_id positional arg. This tool
makes agent_id optional with a sensible default so the ReAct agent can call it
with just a limit, matching the simpler free-form tool contract defined in
ARCHITECTURE_PROPOSAL.md Tool 7.

Both tools co-exist; MempillAuditTool is retained untouched for the legacy graph.

Returns a JSON dict with:
  - agent_id: str
  - entry_count: int
  - entries: list of {claim_ref, event_kind, disposition, recorded_at, rationale}
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

# Hard cap on the number of audit entries a single call can return. A natural-
# language request for a "full audit trail" must never be able to trigger an
# unbounded fetch against a ledger that may hold billions of records.
_MAX_LIMIT = 200


class AuditTrailInput(BaseModel):
    agent_id: str = Field(
        default="demo-agent",
        description="Agent/user session ID (e.g. 'jordan-park-001'). Defaults to 'demo-agent'.",
    )
    limit: int = Field(
        default=50,
        description=(
            f"Maximum number of audit entries to return (most recent first). "
            f"Hard-capped at {_MAX_LIMIT} regardless of the requested value."
        ),
    )
    claim_ref: Optional[str] = Field(
        default=None,
        description="Optional UUID to filter audit entries to a specific claim.",
    )
    from_tx_time: Optional[str] = Field(
        default=None,
        description=(
            "Optional ISO-8601 UTC lower bound for pagination — only entries "
            "recorded at or after this transaction time are returned. Use this "
            "to page through a trail larger than the hard cap."
        ),
    )


class AuditTrailTool(BaseTool):
    """Return the mempill audit ledger for an agent session.

    Provides the chronological log of every claim ingested, superseded,
    contested, or resolved — with recorded_at timestamps, dispositions,
    and rationale notes.

    Use this for compliance queries ("show me all write events"),
    debugging ("what happened to claim X?"), or demonstrating mempill's
    full audit capability.

    Returns JSON with an 'entries' list. Each entry contains:
      claim_ref, event_kind, disposition, recorded_at, rationale.

    This tool reuses adapter.audit() — same logic as MempillAuditTool,
    with an optional agent_id defaulting to 'demo-agent' for convenience.

    limit is hard-capped at _MAX_LIMIT: a request exceeding the cap is silently
    clamped and the result carries a 'note' explaining the clamp, so a natural-
    language "show me the full audit trail" can never trigger an unbounded
    fetch. Use from_tx_time to page through a trail larger than the cap.
    """

    name: str = "audit_trail"
    description: str = (
        "Return the mempill audit ledger — the chronological log of every claim "
        "write, succession, contested event, and oracle resolution. "
        "Supply optional agent_id (default 'demo-agent'), optional limit (default 50, "
        f"hard-capped at {_MAX_LIMIT}), optional claim_ref filter, and optional "
        "from_tx_time (ISO-8601 UTC) to page past the cap. "
        "Returns JSON with entry_count and an entries list (claim_ref, event_kind, "
        "disposition, recorded_at, rationale); includes a 'note' if the requested "
        "limit was clamped."
    )
    args_schema: Type[BaseModel] = AuditTrailInput

    model_config = {"arbitrary_types_allowed": True}

    adapter: MempillAdapter

    @traceable_mempill(name="mempill.audit_trail")
    def _run(
        self,
        agent_id: str = "demo-agent",
        limit: int = 50,
        claim_ref: Optional[str] = None,
        from_tx_time: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        clamped = limit > _MAX_LIMIT
        effective_limit = min(limit, _MAX_LIMIT)

        log.debug(
            "AuditTrailTool: agent=%s limit=%d effective_limit=%d claim_ref=%s from_tx_time=%s",
            agent_id, limit, effective_limit, claim_ref, from_tx_time,
        )

        entries = self.adapter.audit(agent_id, limit=effective_limit, from_tx_time=from_tx_time)

        # Optionally filter to a single claim_ref
        if claim_ref:
            entries = [e for e in entries if e.claim_ref == claim_ref]

        result: dict[str, Any] = {
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
        if clamped:
            result["note"] = (
                f"Requested limit {limit} exceeds the hard cap of {_MAX_LIMIT}; "
                f"showing the {_MAX_LIMIT} most recent of the trail. "
                "Narrow with from_tx_time or claim_ref."
            )
        log.debug("AuditTrailTool returned %d entries (clamped=%s)", len(entries), clamped)
        return json.dumps(result, default=str)

    async def _arun(
        self,
        agent_id: str = "demo-agent",
        limit: int = 50,
        claim_ref: Optional[str] = None,
        from_tx_time: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        return self._run(
            agent_id=agent_id,
            limit=limit,
            claim_ref=claim_ref,
            from_tx_time=from_tx_time,
            **kwargs,
        )
