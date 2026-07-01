"""
mempill_showcase.tools.recall_subject_tool — RecallSubjectTool

LangChain BaseTool wrapping MempillAdapter.query_subject.

Returns ALL known predicate/value pairs for a subject in one call — the
cornerstone of the free-form ReAct agent answering "what role does Alice hold?"
or "tell me everything about Alice" without a predicate alias map.

The LLM reads the returned predicate names (e.g. "employer", "city") and
resolves them linguistically in its final answer. No canonical_keys lookup
is required.

Returns a JSON dict with:
  - subject: str
  - fact_count: int
  - facts: list of {predicate, value, status, valid_from_display,
           valid_until_display, provenance, claim_ref, conf, is_contested}
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from mempill_showcase.adapters.memory.mempill_adapter import MempillAdapter
from mempill_showcase.core.domain.normalise import normalise_key
from mempill_showcase.observability import traceable_mempill

log = logging.getLogger(__name__)


class RecallSubjectInput(BaseModel):
    agent_id: str = Field(description="Agent/user session ID (e.g. 'jordan-park-001')")
    subject: str = Field(
        description=(
            "Entity key to recall all facts about (e.g. 'alice-chen', 'bob-liu'). "
            "Apply soft normalisation: strip, lowercase, spaces→hyphens."
        ),
    )
    valid_at: Optional[str] = Field(
        default=None,
        description=(
            "ISO-8601 UTC instant for valid-time axis — 'what was true in the world at this point?' "
            "None = current valid-time (most recent open claim per predicate)."
        ),
    )
    as_of_tx_time: Optional[str] = Field(
        default=None,
        description=(
            "ISO-8601 UTC instant for transaction-time axis — 'what did the system know then?' "
            "None = use all ingested claims (latest tx-time)."
        ),
    )


class RecallSubjectTool(BaseTool):
    """Return ALL stored facts for a subject in a single call.

    Use this tool when the question is open-ended: "what do you know about Alice?",
    "what role does Alice hold?", "tell me everything about Bob Liu".

    The tool returns one entry per stored predicate (employer, city, dietary_restriction,
    etc.) with bi-temporal metadata. The LLM reads the predicate names from the result
    and resolves them linguistically — no alias map needed.

    Optional valid_at / as_of_tx_time pin both bi-temporal axes independently (same
    semantics as recall_at / recall_as_of but across ALL predicates at once).

    Returns JSON with subject, fact_count, and a facts list.
    Each fact contains: predicate, value, status, is_contested,
    valid_from_display, valid_until_display, provenance, claim_ref, conf.
    """

    name: str = "recall_subject"
    description: str = (
        "Return ALL stored facts for a subject (entity). "
        "Call this when you need to know anything about a person or entity — "
        "e.g. 'what role does Alice hold?', 'tell me everything about Bob'. "
        "Supply agent_id and subject. Optionally pin valid_at (world-time) "
        "and/or as_of_tx_time (transaction-time) for historical queries. "
        "Returns JSON with a facts list; each fact contains predicate, value, "
        "status (Resolved/Contested/NoBelief), is_contested, valid_from_display, "
        "valid_until_display, provenance, claim_ref, conf. "
        "Read predicate names from the result to answer questions linguistically — "
        "no predicate map is needed."
    )
    args_schema: Type[BaseModel] = RecallSubjectInput

    model_config = {"arbitrary_types_allowed": True}

    adapter: MempillAdapter

    @traceable_mempill(name="mempill.recall_subject")
    def _run(
        self,
        agent_id: str,
        subject: str,
        valid_at: Optional[str] = None,
        as_of_tx_time: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        subject = normalise_key(subject)

        log.debug(
            "RecallSubjectTool: agent=%s subject=%s valid_at=%s as_of=%s",
            agent_id, subject, valid_at, as_of_tx_time,
        )

        raw_facts = self.adapter.query_subject(
            agent_id, subject,
            valid_at=valid_at,
            as_of_tx_time=as_of_tx_time,
        )

        facts = [
            {
                "predicate": f.get("predicate"),
                "value": f.get("value"),
                "status": f.get("status"),
                "is_contested": f.get("status") in ("Contested", "Conflict"),
                "valid_from_display": f.get("valid_from_display"),
                "valid_until_display": f.get("valid_until_display"),
                "provenance": f.get("provenance"),
                "claim_ref": f.get("claim_ref"),
                "conf": f.get("conf"),
            }
            for f in (raw_facts or [])
        ]

        result = {
            "subject": subject,
            "fact_count": len(facts),
            "facts": facts,
        }
        log.debug("RecallSubjectTool: %d facts for subject=%s", len(facts), subject)
        return json.dumps(result, default=str)

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("RecallSubjectTool does not support async")
