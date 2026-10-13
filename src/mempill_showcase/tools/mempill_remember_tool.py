"""
mempill_showcase.tools.mempill_remember_tool — MempillRememberTool

LangChain BaseTool wrapping the MempillAdapter write path.

SUCCESSION ENCAPSULATION (the end_fact idiom, TASK-33-W4-DEMO):
  When a new value + valid_from is supplied for a FUNCTIONAL (subject, predicate):

  1. We resolve the live claim(s) on the line via adapter.resolve_live_claim_for_line()
     — the same canonical fold recall()/query_history() use (never a heuristic).
  2. 1 live claim (the common case): if it is OPEN-ENDED (vt_end == "open",
     status == "Resolved") AND the new claim's valid_from is strictly AFTER the
     incumbent's valid_from:
     a. adapter.end_fact() bounds the ACTUAL incumbent claim in place (via
        engine.assert_validity(Bound) — the row itself is never touched, only a
        validity assertion is appended; no duplicate row is ever written).
     b. Write the new challenger claim (valid_from = new valid_from, no valid_until).
        Because the incumbent is no longer live, the write folds directly to
        CommittedCheap — a clean succession, no reconcile() needed to paper over it.
     c. bi-temporal queries for dates within the incumbent's (now-bounded) window
        return the old value correctly. Current recall returns the new value.
  3. 0 live claims: nothing to close — write the new claim normally.
  4. >1 live claims (the line is ALREADY Contested/ambiguous): we never guess which
     claim to close. The new claim is written as another candidate on the line and
     the engine's own Contested/HITL gate surfaces the ambiguity, same as today —
     the tool does NOT force-resolve genuine conflicts.

NOTE (Wave B): The canonical_keys guard (subject/predicate must be in a closed
vocabulary) has been removed. The new open-world design applies soft normalisation
(lowercase + separator) only. The succession logic is unchanged.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from mempill import ProvenanceLabel
from mempill_showcase.adapters.memory.mempill_adapter import MempillAdapter
from mempill_showcase.core.domain.models import ClaimInput, WriteReceipt

from mempill_showcase.observability import emit_contested_span, traceable_mempill

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

    Succession rule (end_fact idiom, TASK-33-W4-DEMO):
      If exactly one open-ended incumbent exists for the same (subject, predicate)
      and the new claim's valid_from is AFTER the incumbent's, the tool bounds the
      incumbent in place via adapter.end_fact() (engine.assert_validity(Bound))
      before writing the challenger — returning CommittedCheap. No manual
      valid_until is required from the caller.

      If the windows genuinely overlap (ambiguous dates, same valid_from, or
      unclear ordering), the engine returns Contested. HITL waves resolve this;
      the tool does NOT force-resolve genuine conflicts.

    Soft normalisation (Wave B):
      Subject and predicate are normalised (lowercase + separator) before storage.
      No closed vocabulary check is applied.
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

    @traceable_mempill(name="mempill.remember")
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
        # Soft normalisation only (canonical_keys guard removed in Wave B)
        # The LLM supplies any subject/predicate; soft rules ensure storage consistency.
        subject = subject.strip().lower().replace(" ", "-")
        predicate = predicate.strip().lower().replace(" ", "_")

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

        # ── Succession encapsulation (end_fact idiom, TASK-33-W4-DEMO) ─────────
        # Only attempt when we have a valid_from (needed to determine temporal order).
        # Track whether a close-step was performed: if yes, the challenger write
        # below folds directly to CommittedCheap (the incumbent is no longer live).
        close_step_performed = False
        ambiguous_incumbent_note: Optional[str] = None

        if valid_from and cardinality == "Functional":
            resolution = self.adapter.resolve_live_claim_for_line(agent_id, subject, predicate)
            live_status = resolution.get("status")

            if live_status == "single":
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
                        # Bound the ACTUAL incumbent claim in place — never a duplicate row.
                        log.debug(
                            "MempillRememberTool: end_fact closing incumbent %s valid_from=%s at %s",
                            incumbent.claim_ref, inc_display, valid_from,
                        )
                        try:
                            self.adapter.end_fact(
                                agent_id, subject, predicate, at=valid_from,
                                provenance=ProvenanceLabel.external_user_asserted(),
                                confidence=confidence,
                            )
                            close_step_performed = True
                        except Exception as exc:
                            log.debug("MempillRememberTool: end_fact close-step skipped: %s", exc)
            elif live_status == "ambiguous":
                # The line is ALREADY Contested (>1 live claim) — never guess which
                # claim to close. Write the challenger as another candidate and let
                # the engine's Contested/HITL gate surface the ambiguity, as today.
                ambiguous_incumbent_note = (
                    f"{resolution.get('live_count')} live claims already exist for "
                    f"{subject}/{predicate}; not auto-closing — write proceeds as a "
                    "new candidate and the line remains Contested pending HITL."
                )
                log.debug("MempillRememberTool: %s", ambiguous_incumbent_note)
            # live_status == "empty": nothing to close; fall through to normal write.

        receipt: WriteReceipt = self.adapter.write_claim(agent_id, claim)

        # Trigger engine succession fold after writing the challenger.
        #
        # With the end_fact idiom the incumbent is already bounded (no longer live)
        # BEFORE the challenger write, so a clean succession folds directly to
        # CommittedCheap with no contested_with — reconcile() here is defensive
        # (harmless no-op in the common case; still useful when close_step_performed
        # is False but the write is otherwise clean, e.g. no prior incumbent).
        #
        # When NOT to reconcile:
        #   contested_with non-empty AND no close-step — this is a genuine
        #   overlapping conflict requiring HITL. Auto-resolving it via reconcile
        #   would silently pick a winner without human confirmation.
        initial_contested = bool(receipt.contested_with)
        should_reconcile = (
            valid_from
            and cardinality == "Functional"
            and (not initial_contested or close_step_performed)
        )

        if should_reconcile:
            try:
                self.adapter.reconcile(
                    agent_id=agent_id,
                    subject_lines=[[subject, predicate]],
                    max_passes=3,
                )
                log.debug(
                    "MempillRememberTool: reconcile complete for %s/%s",
                    subject, predicate,
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
        elif initial_contested and not close_step_performed:
            # Genuine conflict: return Contested without auto-resolution.
            # HITL node is responsible for submitting adjudication.
            final_disposition = "Contested"
            log.debug(
                "MempillRememberTool: skipping reconcile for %s/%s — genuine conflict "
                "(contested_with=%s, no close-step), escalating to HITL",
                subject, predicate, receipt.contested_with,
            )
        else:
            final_disposition = receipt.disposition

        is_contested_final = final_disposition in ("Contested", "Conflict", "QueuedForAdjudication")
        result = {
            "claim_ref": receipt.claim_ref,
            "disposition": final_disposition,
            "contested_with": receipt.contested_with,
            "is_contested": is_contested_final,
        }
        if ambiguous_incumbent_note:
            result["ambiguous_incumbent_note"] = ambiguous_incumbent_note
        log.debug("MempillRememberTool result: %s", result)

        # ── LangSmith: emit dedicated mempill.contested span ─────────────────
        if is_contested_final:
            try:
                incumbent_belief = self.adapter.recall(agent_id, subject, predicate)
                alts = incumbent_belief.alternatives or []
                inc = alts[0].__dict__ if alts else {}
                chal = alts[1].__dict__ if len(alts) > 1 else {}
                emit_contested_span(
                    subject=subject,
                    predicate=predicate,
                    incumbent={"value": inc.get("value"), "valid_from_display": inc.get("vt_start_display"), "claim_ref": inc.get("claim_ref")},
                    challenger={"value": chal.get("value"), "valid_from_display": chal.get("vt_start_display"), "claim_ref": chal.get("claim_ref")},
                    agent_id=agent_id,
                    extra={"new_value": value, "contested_refs": receipt.contested_with},
                )
            except Exception as _exc:
                log.debug("MempillRememberTool: contested span error suppressed: %s", _exc)

        return json.dumps(result)

    async def _arun(
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
        return self._run(
            agent_id=agent_id,
            subject=subject,
            predicate=predicate,
            value=value,
            valid_from=valid_from,
            confidence=confidence,
            provenance_channel=provenance_channel,
            cardinality=cardinality,
            criticality=criticality,
            **kwargs,
        )
