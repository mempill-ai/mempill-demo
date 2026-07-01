"""
mempill_showcase.tools.remember_fact_tool — RememberFactTool

LangChain BaseTool wrapping the MempillAdapter write path WITHOUT a canonical-key
guard. This is the free-form replacement for MempillRememberTool — it accepts ANY
predicate the LLM chooses and normalises it via soft rules only:

  Soft normalisation (applied before storage):
    subject:   strip → lowercase → spaces→hyphens  (e.g. "Alice Chen" → "alice-chen")
    predicate: strip → lowercase → spaces→hyphens  (e.g. "Favorite Color" → "favorite-color")

The succession encapsulation (recall-then-close pattern) from MempillRememberTool is
preserved exactly: if an open-ended incumbent exists for the same (subject, predicate)
and the new valid_from is after the incumbent's, the tool closes the old window and
reconciles — returning CommittedCheap. Genuine overlapping conflicts return Contested.

MempillRememberTool (canonical-key guard) is left untouched in the tools/ directory;
this file is a NEW tool. Both co-exist until the graph wave deletes the old one.

Returns a JSON dict with: claim_ref, disposition, is_contested.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from mempill import ProvenanceLabel
from mempill_showcase.adapters.memory.mempill_adapter import MempillAdapter
from mempill_showcase.core.domain.models import ClaimInput
from mempill_showcase.core.domain.dates import is_later
from mempill_showcase.core.domain.normalise import normalise_key
from mempill_showcase.observability import traceable_mempill

log = logging.getLogger(__name__)


def _build_provenance(channel: str) -> dict:
    """Map a friendly channel string to a ProvenanceLabel dict."""
    mapping = {
        "userasserted":      ProvenanceLabel.external_user_asserted(),
        "externalfirsthand": ProvenanceLabel.external_first_hand(),
        "modelderived":      ProvenanceLabel.model_derived(),
        "recallreentry":     ProvenanceLabel.recall_re_entry(),
    }
    return mapping.get(channel.strip().lower(), ProvenanceLabel.external_user_asserted())


class RememberFactInput(BaseModel):
    agent_id: str = Field(description="Agent/user session ID (e.g. 'jordan-park-001')")
    subject: str = Field(
        description=(
            "Entity key — any free-form name is accepted; "
            "soft normalisation strips, lowercases, and replaces spaces with hyphens. "
            "Example: 'Alice Chen' → 'alice-chen'."
        ),
    )
    predicate: str = Field(
        description=(
            "Predicate key — any free-form label is accepted; "
            "soft normalisation strips, lowercases, and replaces spaces with hyphens. "
            "Examples: 'city', 'employer', 'favorite-color', 'dietary-restriction'."
        ),
    )
    value: str = Field(description="Claim value (string or JSON-serialisable)")
    valid_from: Optional[str] = Field(
        default=None,
        description="World-time start (YYYY / YYYY-MM / YYYY-MM-DD / RFC3339). None = unknown.",
    )
    valid_until: Optional[str] = Field(
        default=None,
        description=(
            "World-time end (YYYY / YYYY-MM / YYYY-MM-DD / RFC3339); "
            "None = open-ended. Supply this for a bounded interval, e.g. a "
            "fixed-term appointment or a known end date."
        ),
    )
    provenance: Optional[str] = Field(
        default="UserAsserted",
        description=(
            "Provenance channel: 'UserAsserted' | 'ExternalFirstHand' | "
            "'ModelDerived' | 'RecallReEntry'."
        ),
    )
    confidence: float = Field(default=1.0, description="Value confidence [0.0, 1.0]")


class RememberFactTool(BaseTool):
    """Write a free-form fact claim to mempill with automatic succession encapsulation.

    Accepts ANY subject and predicate — no canonical-key vocabulary check.
    Soft normalisation is applied: subject and predicate are lowercased and
    spaces are replaced with hyphens before storage.

    Succession rule (preserved from MempillRememberTool):
      If an open-ended incumbent exists for the same (subject, predicate), the new
      claim is ALSO open-ended (no valid_until), and the new valid_from is AFTER the
      incumbent's start, the tool closes the old window (recall-then-close pattern)
      and reconciles — returning CommittedCheap. A BOUNDED challenger (valid_until
      present) against an open-ended incumbent is left alone here: it overlaps by
      the engine's own non-overlap rule and returns Contested, requiring HITL
      resolution — this is how competing bounded appointments (e.g. two people
      both claiming the same org role) surface as genuine conflicts instead of
      being silently superseded.

    Returns JSON with claim_ref, disposition (CommittedCheap or Contested), and
    is_contested. Callers must check is_contested before trusting the write.
    """

    name: str = "remember_fact"
    description: str = (
        "Write a new fact claim to the mempill memory engine. "
        "Accepts any free-form subject and predicate — no vocabulary check. "
        "Supply agent_id, subject, predicate, value, optional valid_from date, "
        "optionally valid_until for a bounded interval, e.g. a fixed-term appointment "
        "(None = open-ended), optional provenance channel, and optional confidence. "
        "Handles succession automatically: if an earlier open claim exists for the "
        "same (subject, predicate), the tool closes the prior window before writing the "
        "new claim. Returns JSON with claim_ref, disposition, and is_contested."
    )
    args_schema: Type[BaseModel] = RememberFactInput

    model_config = {"arbitrary_types_allowed": True}

    adapter: MempillAdapter

    @traceable_mempill(name="mempill.remember_fact")
    def _run(
        self,
        agent_id: str,
        subject: str,
        predicate: str,
        value: str,
        valid_from: Optional[str] = None,
        valid_until: Optional[str] = None,
        provenance: Optional[str] = "UserAsserted",
        confidence: float = 1.0,
        **kwargs: Any,
    ) -> str:
        subject = normalise_key(subject)
        predicate = normalise_key(predicate)
        prov = _build_provenance(provenance or "UserAsserted")

        log.debug(
            "RememberFactTool: agent=%s subject=%s predicate=%s value=%r valid_from=%s valid_until=%s",
            agent_id, subject, predicate, value, valid_from, valid_until,
        )

        claim = ClaimInput(
            subject=subject,
            predicate=predicate,
            value=value,
            valid_from=valid_from,
            valid_until=valid_until,
            confidence=confidence,
            provenance=prov,
            cardinality="Functional",
            criticality="Medium",
        )

        # ── Succession encapsulation (recall-then-close) ──────────────────────
        # Only auto-close the incumbent when the CHALLENGER is itself open-ended
        # (a true "X replaces Y indefinitely" succession). A bounded challenger
        # (valid_until present) against an open-ended incumbent OVERLAPS by the
        # engine's own non_overlapping rule (open end == infinity, always >
        # any finite start) — that is a genuine conflict, not a clean handoff,
        # and must be left for ingest_claim/reconcile to surface as Contested.
        close_step_performed = False
        if valid_from and not valid_until:
            incumbent = self.adapter.recall(agent_id, subject, predicate)
            if (
                incumbent.status == "Resolved"
                and incumbent.vt_end == "open"
                and incumbent.vt_start
                and incumbent.claim_ref
            ):
                inc_display = incumbent.vt_start_display or ""
                if inc_display and is_later(valid_from, inc_display) and incumbent.value != value:
                    log.debug(
                        "RememberFactTool: closing open incumbent %s valid_from=%s at %s",
                        incumbent.claim_ref, inc_display, valid_from,
                    )
                    close_claim = ClaimInput(
                        subject=subject,
                        predicate=predicate,
                        value=incumbent.value,
                        valid_from=inc_display,
                        valid_until=valid_from,
                        confidence=confidence,
                        provenance=ProvenanceLabel.external_user_asserted(),
                        cardinality="Functional",
                        criticality="Medium",
                    )
                    self.adapter.write_claim(agent_id, close_claim)
                    close_step_performed = True

        receipt = self.adapter.write_claim(agent_id, claim)

        initial_contested = bool(receipt.contested_with)
        should_reconcile = valid_from and (not initial_contested or close_step_performed)

        if should_reconcile:
            try:
                self.adapter.reconcile(
                    agent_id=agent_id,
                    subject_lines=[[subject, predicate]],
                    max_passes=3,
                )
                final_belief = self.adapter.recall(agent_id, subject, predicate)
                final_disposition = (
                    "Contested" if final_belief.is_contested() else "CommittedCheap"
                )
            except Exception as exc:
                log.debug("RememberFactTool: reconcile skipped: %s", exc)
                final_disposition = receipt.disposition
        elif initial_contested and not close_step_performed:
            final_disposition = "Contested"
            log.debug(
                "RememberFactTool: genuine conflict on %s/%s — escalating to HITL",
                subject, predicate,
            )
        else:
            final_disposition = receipt.disposition

        is_contested_final = final_disposition in ("Contested", "Conflict", "QueuedForAdjudication")
        result = {
            "claim_ref": receipt.claim_ref,
            "disposition": final_disposition,
            "is_contested": is_contested_final,
        }
        log.debug("RememberFactTool result: %s", result)
        return json.dumps(result)

    async def _arun(
        self,
        agent_id: str,
        subject: str,
        predicate: str,
        value: str,
        valid_from: Optional[str] = None,
        valid_until: Optional[str] = None,
        provenance: Optional[str] = "UserAsserted",
        confidence: float = 1.0,
        **kwargs: Any,
    ) -> str:
        return self._run(
            agent_id=agent_id,
            subject=subject,
            predicate=predicate,
            value=value,
            valid_from=valid_from,
            valid_until=valid_until,
            provenance=provenance,
            confidence=confidence,
            **kwargs,
        )
