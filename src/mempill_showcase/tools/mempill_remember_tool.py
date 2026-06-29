"""
mempill_showcase.tools.mempill_remember_tool — MempillRememberTool

LangChain BaseTool wrapping the MempillAdapter write path.

SUCCESSION ENCAPSULATION (the recall-then-close pattern):
  When a new value + valid_from is supplied for a FUNCTIONAL (subject, predicate):

  1. We first recall the current belief.
  2. If an OPEN-ENDED incumbent exists (vt_end == "open", status == "Resolved") AND
     the new claim's valid_from is strictly AFTER the incumbent's valid_from:
     a. Write a BOUNDED version of the incumbent (same value, same valid_from,
        valid_until = new valid_from). This tells the engine the old value's
        valid window closed at the new claim's start.
     b. Write the new challenger claim (valid_from = new valid_from, no valid_until).
     c. Call engine.reconcile() to fold non-overlapping windows into CommittedCheap.
        The engine now sees: incumbent-bounded (no overlap with challenger) →
        CommittedCheap; challenger → CommittedCheap (clean succession).
        bi-temporal queries for dates within the incumbent's window return the old
        value correctly. Current recall returns the new value.
  3. If the incumbent's valid_from is EQUAL TO or AFTER the new valid_from,
     the windows genuinely overlap → the tool lets the write proceed without
     forcing resolution and the engine returns Contested. HITL waves (W7) handle
     genuine conflicts. The tool does NOT force-resolve ambiguous conflicts.
  4. If there is no incumbent (NoBelief) or status is already Contested, we
     write the new claim normally without the close-step.

CANONICAL KEYS:
  subject and predicate MUST already be canonical keys from canonical_keys.py.
  The tool asserts this and raises ValueError if either is unknown.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from mempill import ProvenanceLabel
from mempill_showcase.adapters.memory.mempill_adapter import MempillAdapter
from mempill_showcase.core.domain.canonical_keys import (
    all_canonical_entities,
    all_canonical_predicates,
)
from mempill_showcase.core.domain.models import ClaimInput, WriteReceipt

log = logging.getLogger(__name__)


class RememberInput(BaseModel):
    agent_id: str = Field(description="Agent/user session ID (e.g. 'jordan-park-001')")
    subject: str = Field(description="Canonical entity key (e.g. 'alice-chen')")
    predicate: str = Field(description="Canonical predicate key (e.g. 'city')")
    value: str = Field(description="Claim value (string or JSON-serialisable)")
    valid_from: Optional[str] = Field(
        default=None,
        description="World-time start (YYYY / YYYY-MM / YYYY-MM-DD / RFC3339). None = unknown.",
    )
    confidence: float = Field(default=1.0, description="Value confidence [0.0, 1.0]")
    provenance_channel: str = Field(
        default="UserAsserted",
        description=(
            "Provenance channel: 'UserAsserted' | 'ExternalFirstHand' | "
            "'ModelDerived' | 'RecallReEntry'"
        ),
    )
    cardinality: str = Field(default="Functional", description="'Functional' | 'SetValued'")
    criticality: str = Field(default="Medium", description="'Low' | 'Medium' | 'High' | 'Critical'")


def _build_provenance(channel: str) -> dict:
    """Map a friendly channel string to a ProvenanceLabel dict."""
    mapping = {
        "userasserted":     ProvenanceLabel.external_user_asserted(),
        "externalfirsthand": ProvenanceLabel.external_first_hand(),
        "modelderived":     ProvenanceLabel.model_derived(),
        "recallreentry":    ProvenanceLabel.recall_re_entry(),
    }
    return mapping.get(channel.lower(), ProvenanceLabel.external_user_asserted())


class MempillRememberTool(BaseTool):
    """Write a claim to mempill with automatic succession encapsulation.

    Succession rule (recall-then-close pattern):
      If an open-ended incumbent exists for the same (subject, predicate) and the
      new claim's valid_from is AFTER the incumbent's, mempill's engine handles
      succession automatically — returning CommittedCheap and end-bounding the
      prior claim. No manual valid_until is required from the caller.

      If the windows genuinely overlap (ambiguous dates, same valid_from, or
      unclear ordering), the engine returns Contested. HITL waves resolve this;
      the tool does NOT force-resolve genuine conflicts.

    Canonical key enforcement:
      Both subject and predicate must be resolvable canonical keys. Use
      canonical_keys.resolve_entity / resolve_predicate before calling this tool.
    """

    name: str = "mempill_remember"
    description: str = (
        "Write a new fact claim to the mempill memory engine. "
        "Supply canonical entity key (subject), canonical predicate, value, "
        "optional valid_from date, confidence, and provenance_channel. "
        "Returns a JSON dict with claim_ref, disposition (CommittedCheap or Contested), "
        "and contested_with (list of conflicting claim refs if Contested)."
    )
    args_schema: Type[BaseModel] = RememberInput

    # Injected at construction time — not a Pydantic field so the adapter
    # (which contains a mempill engine) is not serialised by Pydantic.
    # Use model_config to allow arbitrary types.
    model_config = {"arbitrary_types_allowed": True}

    adapter: MempillAdapter

    def _run(
        self,
        agent_id: str,
        subject: str,
        predicate: str,
        value: str,
        valid_from: Optional[str] = None,
        confidence: float = 1.0,
        provenance_channel: str = "UserAsserted",
        cardinality: str = "Functional",
        criticality: str = "Medium",
        **kwargs: Any,
    ) -> str:
        # Canonical key guard
        known_entities = all_canonical_entities()
        known_predicates = all_canonical_predicates()
        if subject not in known_entities:
            raise ValueError(
                f"Unknown entity key '{subject}'. "
                f"Resolve via canonical_keys.resolve_entity() first. "
                f"Known: {sorted(known_entities)}"
            )
        if predicate not in known_predicates:
            raise ValueError(
                f"Unknown predicate key '{predicate}'. "
                f"Resolve via canonical_keys.resolve_predicate() first. "
                f"Known: {sorted(known_predicates)}"
            )

        prov = _build_provenance(provenance_channel)
        claim = ClaimInput(
            subject=subject,
            predicate=predicate,
            value=value,
            valid_from=valid_from,
            confidence=confidence,
            provenance=prov,
            cardinality=cardinality,
            criticality=criticality,
        )

        log.debug(
            "MempillRememberTool: agent=%s subject=%s predicate=%s value=%r valid_from=%s channel=%s",
            agent_id, subject, predicate, value, valid_from, provenance_channel,
        )

        # ── Succession encapsulation (recall-then-close) ──────────────────────
        # Only attempt when we have a valid_from (needed to determine temporal order).
        if valid_from and cardinality == "Functional":
            incumbent = self.adapter.recall(agent_id, subject, predicate)
            if (
                incumbent.status == "Resolved"
                and incumbent.vt_end == "open"
                and incumbent.vt_start  # has a known start
                and incumbent.claim_ref  # is a real claim
            ):
                # Compare incumbent start (RFC3339) vs new valid_from (partial date).
                # The adapter's _to_rfc3339 converts partial dates; we use the
                # incumbent's vt_start_display for precision-safe comparison.
                inc_display = incumbent.vt_start_display or ""
                if inc_display and inc_display < valid_from and incumbent.value != value:
                    # Clean succession: incumbent started before new claim.
                    # Write a bounded version of the incumbent to close its window.
                    log.debug(
                        "MempillRememberTool: closing open incumbent %s valid_from=%s at %s",
                        incumbent.claim_ref, inc_display, valid_from,
                    )
                    close_claim = ClaimInput(
                        subject=subject,
                        predicate=predicate,
                        value=incumbent.value,
                        valid_from=inc_display,
                        valid_until=valid_from,  # close at the new claim's start
                        confidence=confidence,
                        provenance=ProvenanceLabel.external_user_asserted(),
                        cardinality=cardinality,
                        criticality=criticality,
                    )
                    self.adapter.write_claim(agent_id, close_claim)

        receipt: WriteReceipt = self.adapter.write_claim(agent_id, claim)

        # Trigger engine succession fold after writing the challenger.
        # For Functional predicates with a new valid_from, reconcile() resolves
        # non-overlapping windows into CommittedCheap without oracle involvement.
        # Multiple reconcile passes may be needed: the first pass Supersedes the
        # open-ended incumbent; the second pass promotes the challenger to CommittedCheap.
        if valid_from and cardinality == "Functional":
            try:
                reconcile_req = {
                    "agent_id": agent_id,
                    "subject_lines": [[subject, predicate]],
                }
                for _pass in range(3):  # up to 3 passes; normal succession needs 2
                    resp = self.adapter._engine.reconcile(reconcile_req)
                    if resp.get("oracle_escalations", 0) == 0:
                        break
                log.debug(
                    "MempillRememberTool: reconcile complete for %s/%s (passes=%d)",
                    subject, predicate, _pass + 1,
                )
                # Re-read the final disposition from engine state
                final_belief = self.adapter.recall(agent_id, subject, predicate)
                final_disposition = (
                    "Contested" if final_belief.is_contested()
                    else "CommittedCheap"
                )
            except Exception as exc:
                log.debug("MempillRememberTool: reconcile skipped: %s", exc)
                final_disposition = receipt.disposition
        else:
            final_disposition = receipt.disposition

        result = {
            "claim_ref": receipt.claim_ref,
            "disposition": final_disposition,
            "contested_with": receipt.contested_with,
            "is_contested": final_disposition in ("Contested", "Conflict", "QueuedForAdjudication"),
        }
        log.debug("MempillRememberTool result: %s", result)
        return json.dumps(result)

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("MempillRememberTool does not support async")
