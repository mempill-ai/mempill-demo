"""
mempill_showcase.frameworks.langgraph.crew_nodes — crew node shells (W3/W10).

These are plain Python functions that call W2 tools directly.

Crew A — crew_a_node (Intake & Memory):
  Intent: UPDATE_CONTACT
  Flow:
    1. Use DateParserTool to extract valid_from from the user utterance.
    2. Resolve entity/predicate to canonical keys (canonical_keys.py).
    3. Call MempillRememberTool to write the claim.
    4. If disposition == Contested (is_contested), populate `pending_contested`
       with subject/predicate + candidate details + claim_refs, and set route="hitl".
    5. Otherwise: set route="end", write `write_result`.

Crew B — crew_b_node (Research & Intelligence):
  Intent: RESEARCH
  Flow:
    1. Write the full raw research text to the RAG store (RAGWriteTool) — bulk context.
    2. Distill an atomic claim into mempill (MempillRememberTool) with
       provenance_channel="ExternalFirstHand" and confidence<1.0.
    3. Detect Contested: if is_contested, set `pending_contested` + route="hitl".
    4. Otherwise: set route="end", write `write_result`.

Crew C — crew_c_node (Scheduling & Briefing):
  Intents: PREPARE_BRIEFING | RECALL_HISTORY | COMPLIANCE_AUDIT
  Read-only: NEVER writes to mempill.
  Flow for PREPARE_BRIEFING:
    MempillRecallTool (current, valid_at=None) for each relevant attribute.
  Flow for RECALL_HISTORY:
    MempillRecallTool with valid_at or as_of_tx_time from the request context.
  Flow for COMPLIANCE_AUDIT:
    MempillAuditTool for the full ledger.

LLM extraction (W10):
  LLMExtractor — a single structured Anthropic API call that extracts
  {entity, predicate, value, valid_from} from a free-form sentence.
  It replaces the heuristic shell when ANTHROPIC_API_KEY is present.
  Unlike CrewAI crew kickoff, it uses a single bounded call and returns
  structured data; the Python shell writes to mempill (reliable, no tool-loop).

  make_crew_a_node(..., extractor=LLMExtractor(...)) → LLM extraction path.
  make_crew_a_node(..., extractor=None, crew=None) → deterministic shell.

Shell architecture (W3):
  - No CrewAI agents; no LLM; no autonomous routing within a crew.
  - Entity/predicate are extracted from the state dict via simple keyword heuristics
    or hardcoded fixtures for the scenario.
  - canonical_keys.resolve_entity / resolve_predicate enforce AC-7.
"""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Optional

from mempill_showcase.core.domain.canonical_keys import resolve_entity, resolve_predicate
from mempill_showcase.frameworks.langgraph.state import ExecAssistantState, IntentLabel

if TYPE_CHECKING:
    from mempill_showcase.tools.date_parser_tool import DateParserTool
    from mempill_showcase.tools.mempill_audit_tool import MempillAuditTool
    from mempill_showcase.tools.mempill_recall_tool import MempillRecallTool
    from mempill_showcase.tools.mempill_remember_tool import MempillRememberTool
    from mempill_showcase.tools.rag_read_tool import RAGReadTool
    from mempill_showcase.tools.rag_write_tool import RAGWriteTool

log = logging.getLogger(__name__)


def _get_agent_id_default() -> str:
    """Return the configured default agent_id from Settings (env: MEMPILL_AGENT_ID)."""
    try:
        from mempill_showcase.config.settings import get_settings
        return get_settings().mempill_agent_id
    except Exception:
        return "jordan-park-001"


# Module-level constant read once at import time; callers that need the live
# value per-invocation should call _get_agent_id_default() directly.
AGENT_ID_DEFAULT = _get_agent_id_default()

import re as _re

# Matches "since 2025-02", "since February 2025", "last month" etc.
# Capture group 1: the date portion to pass to DateParserTool.
_DATE_HINT_RE = _re.compile(
    r"\b(?:since|from|starting|as\s+of)\s+"
    r"((?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)"
    r"\s+\d{1,2},?\s+\d{4}"
    r"|(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)"
    r"\s+\d{4}"
    r"|\d{4}-\d{2}-\d{2}"
    r"|\d{4}-\d{2}"
    r"|\d{4})",
    _re.IGNORECASE,
)

# Bare ISO date in sentence (e.g. "2025-02", "2025", "2023-06")
_ISO_DATE_RE = _re.compile(r"\b(\d{4}-\d{2}-\d{2}|\d{4}-\d{2}|\d{4})\b")


def _extract_date_hint(user_input: str) -> Optional[str]:
    """Extract a date string from a sentence for DateParserTool.

    Priority:
      1. "since/from/as of <date>" → return "<date>" (prefix stripped)
      2. Bare ISO date in the sentence
      3. None
    """
    m = _DATE_HINT_RE.search(user_input)
    if m:
        return m.group(1)
    m2 = _ISO_DATE_RE.search(user_input)
    if m2:
        return m2.group(1)
    return None


# ── Internal helpers ──────────────────────────────────────────────────────────

def _extract_entity_predicate_value(user_input: str) -> tuple[str | None, str | None, str | None]:
    """Heuristic shell extraction from user text.

    Returns (canonical_entity, canonical_predicate, value_hint) or (None, None, None).
    W4 CrewAI replaces this with a real Canonicalization Agent + Intake Agent.
    """
    lower = user_input.lower()

    # Entity detection (first match wins)
    entity = None
    for alias in ["alice chen", "alice", "bob liu", "bob", "acme corp", "acme", "jordan"]:
        if alias in lower:
            entity = resolve_entity(alias)
            if entity:
                break

    # Predicate detection
    predicate = None
    if any(k in lower for k in ["city", "relocated", "moved", "location"]):
        predicate = resolve_predicate("city")
    elif any(k in lower for k in ["employer", "role", "title", "promoted", "cto", "ceo", "vp"]):
        predicate = resolve_predicate("employer")
    elif "dietary" in lower or "diet" in lower or "vegetarian" in lower:
        predicate = resolve_predicate("dietary_restriction")
    elif "travel" in lower or "flight" in lower or "seat" in lower:
        predicate = resolve_predicate("travel_preference")
    elif "hotel" in lower:
        predicate = resolve_predicate("preferred_hotel")

    # Value hint: look for city names or explicit "to X"
    value = None
    value_map = {
        "new york": "New York NY",
        "nyc": "New York NY",
        "austin": "Austin TX",
        "san francisco": "San Francisco CA",
        "cto": "Acme Corp / CTO",
        "vp engineering": "Acme Corp / VP Engineering",
        "marcus webb": "Marcus Webb",
        "diane foster": "Diane Foster",
    }
    for kw, val in value_map.items():
        if kw in lower:
            value = val
            break

    return entity, predicate, value


def _build_pending_contested(
    subject: str,
    predicate: str,
    write_json: dict,
    adapter,
    agent_id: str,
) -> dict:
    """Build the pending_contested payload from a Contested write result.

    Recalls both the incumbent and challenger from the adapter to get their
    provenance and valid_from_display for the HITL presentation.
    """
    claim_refs = write_json.get("contested_with") or []
    new_ref = write_json.get("claim_ref", "")
    if new_ref and new_ref not in claim_refs:
        claim_refs = [new_ref] + claim_refs

    # Recall the current (Contested) belief to surface both candidates
    try:
        belief = adapter.recall(agent_id, subject, predicate)
        candidates = belief.alternatives or []
        incumbent = {}
        challenger = {}
        if candidates:
            # First alternative is typically the incumbent (older claim)
            inc = candidates[0]
            incumbent = {
                "value": inc.value,
                "valid_from_display": inc.vt_start_display,
                "claim_ref": inc.claim_ref,
                "provenance": "UserAsserted",
            }
            if len(candidates) > 1:
                chal = candidates[1]
                challenger = {
                    "value": chal.value,
                    "valid_from_display": chal.vt_start_display,
                    "claim_ref": chal.claim_ref,
                    "provenance": "ExternalFirstHand",
                }
        elif belief.value:
            # Contested belief where primary holds the contested value
            incumbent = {
                "value": belief.value,
                "valid_from_display": belief.vt_start_display,
                "claim_ref": belief.claim_ref,
                "provenance": belief.provenance,
            }
    except Exception as exc:
        log.warning("_build_pending_contested: recall failed: %s", exc)
        incumbent = {}
        challenger = {}

    return {
        "subject": subject,
        "predicate": predicate,
        "incumbent": incumbent,
        "challenger": challenger,
        "claim_refs": claim_refs,
    }


# ── LLM extractor (W10 — single structured call, ANTHROPIC_API_KEY required) ──

_EXTRACTOR_SYSTEM = """You are a claim extraction engine for a bi-temporal memory system.

Given a natural-language sentence, extract ONE atomic fact and return it as a JSON object.

Known canonical entity keys:
  alice-chen   → Alice Chen, alice
  bob-liu      → Bob Liu, bob
  acme-corp    → Acme Corp, Acme
  jordan-park  → Jordan Park, jordan

Known canonical predicate keys — choose the BEST match:
  city                 → person's city/location (moved to, relocated, now lives in)
  employer             → person's EMPLOYMENT: their employer company AND job title
                         Use when the SUBJECT is a PERSON and the sentence describes
                         that person's job, role, title, or place of work.
                         (VP, CTO, CEO, Director, Partner, promoted to, now works at, hired as)
                         value format: "Company / Job Title" e.g. "Acme Corp / CTO"
  dietary_restriction  → dietary restriction (vegetarian, vegan, kosher, etc.)
  travel_preference    → travel/flight preference (window seat, aisle, etc.)
  preferred_hotel      → preferred hotel or hotel loyalty programme
  ceo                  → ONLY when stating who is the CEO OF a company where the ENTITY is the company
  cto                  → ONLY when stating who is the CTO OF a company where the ENTITY is the company

DISAMBIGUATION RULE — person's title vs company's officer:
  "Alice is now CTO of Acme"  → subject=ALICE, her JOB changed → entity=alice-chen, predicate=employer, value="Acme Corp / CTO"
  "Acme's new CTO is Alice"   → subject=ACME, its officer changed → entity=acme-corp, predicate=cto, value="Alice Chen"
  The key: if the sentence focuses on what the PERSON is doing, use alice-chen/employer.

Rules:
- Return ONLY a JSON object — no prose, no markdown, no extra text.
- If you cannot confidently extract the entity or predicate, set them to null.
- valid_from: ISO date in YYYY, YYYY-MM, or YYYY-MM-DD format. "now" or no explicit date → null.
- value: a short descriptive string.

JSON schema:
{
  "entity":     "<canonical entity key or null>",
  "predicate":  "<canonical predicate key or null>",
  "value":      "<claim value string>",
  "valid_from": "<YYYY[-MM[-DD]] or null>"
}"""


class LLMExtractor:
    """Single structured-output Anthropic call to extract claim fields from free text.

    Extracts {entity, predicate, value, valid_from} from a natural-language sentence.
    Uses ONE bounded API call with a structured JSON prompt — no tool-loop, no CrewAI.
    The calling code (make_crew_a_node) writes to mempill via the Python remember_tool
    using the extracted fields.

    Falls back to returning all-None on any API or parse error.
    """

    def __init__(self, model_name: Optional[str] = None) -> None:
        if model_name:
            self._model = model_name
        else:
            try:
                from mempill_showcase.config.settings import get_settings
                self._model = get_settings().anthropic_model
            except Exception:
                import os
                self._model = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5")
        log.info("LLMExtractor: initialised with model=%s", self._model)

    def extract(self, sentence: str) -> dict:
        """Extract claim fields from *sentence*.

        Returns a dict with keys: entity, predicate, value, valid_from.
        Any field may be None if the model cannot extract it with confidence.
        """
        import json as _json
        try:
            from langchain_anthropic import ChatAnthropic
            from langchain_core.messages import HumanMessage, SystemMessage
            llm = ChatAnthropic(model=self._model, temperature=0.0)
            messages = [
                SystemMessage(content=_EXTRACTOR_SYSTEM),
                HumanMessage(content=f"Sentence: {sentence}"),
            ]
            response = llm.invoke(messages)
            raw = response.content.strip()
            # Strip markdown code fences if the model wraps in ```json ... ```
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            parsed = _json.loads(raw)
            log.debug("LLMExtractor: %r → %s", sentence[:80], parsed)
            return parsed
        except Exception as exc:
            log.warning("LLMExtractor: extraction failed (%s) — returning null fields", exc)
            return {"entity": None, "predicate": None, "value": None, "valid_from": None}


# ── Crew A node factory ───────────────────────────────────────────────────────

def make_crew_a_node(
    remember_tool: "MempillRememberTool",
    date_parser: "DateParserTool",
    adapter,
    crew=None,
    extractor: Optional["LLMExtractor"] = None,
):
    """Factory: returns the crew_a_node function bound to its tools.

    Priority order:
      1. LLM extraction path (extractor is not None, W10):
         A single structured Anthropic call extracts {entity, predicate, value,
         valid_from}; the Python remember_tool writes to mempill reliably.
         This is the preferred free-form path when ANTHROPIC_API_KEY is present.
      2. CrewAI live path (crew is not None, W4, kept for compatibility):
         Delegates to crew.kickoff(); falls back to shell on error.
      3. Shell path (crew=None, extractor=None):
         Deterministic keyword heuristics — no API key, CI-safe (W3).

    The state contract and routing logic (pending_contested → hitl) are identical
    in all paths — only the extraction mechanism differs.
    """

    def crew_a_node(state: ExecAssistantState) -> dict:
        """Crew A node — intake: extract → remember → detect Contested.

        LLM extraction path (extractor set): single structured LLM call (W10).
        CrewAI live path    (crew set):      delegates to CrewAI kickoff (W4 legacy).
        Shell path          (both None):     deterministic heuristic extraction (W3).
        """
        user_input = state.get("user_input", "")
        agent_id = state.get("agent_id", AGENT_ID_DEFAULT)

        log.info(
            "crew_a_node: input=%r (extractor=%s crew=%s)",
            user_input[:80],
            type(extractor).__name__ if extractor else "none",
            type(crew).__name__ if crew else "none",
        )

        # ── W10 LLM extraction path (single structured call — preferred) ─────
        if extractor is not None:
            extracted = extractor.extract(user_input)
            llm_entity = extracted.get("entity")
            llm_predicate = extracted.get("predicate")
            llm_value = extracted.get("value")
            llm_valid_from = extracted.get("valid_from")

            if llm_entity and llm_predicate and llm_value:
                # LLM extraction succeeded — write via remember_tool (reliable Python path)
                log.info(
                    "crew_a_node [llm-extract]: entity=%s predicate=%s value=%r valid_from=%s",
                    llm_entity, llm_predicate, llm_value, llm_valid_from,
                )
                try:
                    raw = remember_tool.invoke({
                        "agent_id": agent_id,
                        "subject": llm_entity,
                        "predicate": llm_predicate,
                        "value": llm_value,
                        "valid_from": llm_valid_from,
                        "confidence": 1.0,
                        "provenance_channel": "UserAsserted",
                    })
                    write_json = json.loads(raw)
                except Exception as exc:
                    log.error("crew_a_node [llm-extract]: remember_tool failed: %s", exc)
                    return {
                        "error": str(exc),
                        "route": "end",
                        "output_text": f"crew_a [llm]: remember_tool error: {exc}",
                    }
                is_contested = write_json.get("is_contested", False)
                log.info(
                    "crew_a_node [llm-extract]: wrote %s/%s=%r disposition=%s is_contested=%s",
                    llm_entity, llm_predicate, llm_value,
                    write_json.get("disposition"), is_contested,
                )
                if is_contested:
                    pending = _build_pending_contested(
                        llm_entity, llm_predicate, write_json, adapter, agent_id
                    )
                    return {
                        "write_result": raw,
                        "pending_contested": pending,
                        "route": "hitl",
                        "output_text": (
                            f"crew_a [llm]: Contested write for {llm_entity}/{llm_predicate}"
                            " — escalating to HITL"
                        ),
                    }
                return {
                    "write_result": raw,
                    "pending_contested": None,
                    "route": "end",
                    "output_text": (
                        f"crew_a [llm]: wrote {llm_entity}/{llm_predicate}={llm_value!r} "
                        f"(disposition={write_json.get('disposition')} valid_from={llm_valid_from})"
                    ),
                }
            else:
                # LLM could not extract — fall through to shell heuristics
                log.warning(
                    "crew_a_node: LLM extraction returned null fields (entity=%s predicate=%s value=%s)"
                    " — falling back to shell",
                    llm_entity, llm_predicate, llm_value,
                )

        # ── CrewAI live path (W4 — legacy, kept for compatibility) ────────────
        elif crew is not None:
            try:
                result = crew.kickoff(inputs={
                    "user_request": user_input,
                    "agent_id": agent_id,
                })
                raw_output = str(result)
                # Parse disposition from crew output (may be JSON or text)
                import re as _re2
                is_contested = bool(_re2.search(r'"is_contested"\s*:\s*true', raw_output, _re2.IGNORECASE))
                subject, predicate, value = _extract_entity_predicate_value(user_input)
                if is_contested and subject and predicate:
                    # Build pending_contested from the adapter so routing still works
                    pending = _build_pending_contested(subject, predicate, {}, adapter, agent_id)
                    return {
                        "write_result": raw_output,
                        "pending_contested": pending,
                        "route": "hitl",
                        "output_text": f"crew_a [crewai]: Contested write for {subject}/{predicate} — escalating to HITL",
                    }
                return {
                    "write_result": raw_output,
                    "pending_contested": None,
                    "route": "end",
                    "output_text": f"crew_a [crewai]: {raw_output[:200]}",
                }
            except Exception as exc:
                log.error("crew_a_node: CrewAI kickoff failed: %s — falling back to shell", exc)
                # Fall through to shell path on error

        # ── Shell path (W3 deterministic heuristics) ─────────────────────────
        # 1. Parse valid_from via DateParserTool.
        # Extract date hint first (DateParserTool expects a date string, not a sentence).
        valid_from: Optional[str] = None
        date_hint = _extract_date_hint(user_input)
        if date_hint:
            try:
                date_raw = date_parser.invoke({"date_string": date_hint})
                date_result = json.loads(date_raw)
                if "error" not in date_result:
                    valid_from = date_result.get("iso_date")
                else:
                    log.debug("crew_a_node: DateParserTool error for hint %r: %s", date_hint, date_result.get("error"))
            except Exception as exc:
                log.debug("crew_a_node: date parse failed for hint %r: %s", date_hint, exc)
        else:
            log.debug("crew_a_node: no date hint found in input %r", user_input[:60])

        # 2. Heuristic entity/predicate/value extraction (W3 shell)
        subject, predicate, value = _extract_entity_predicate_value(user_input)

        if not subject or not predicate or not value:
            log.warning(
                "crew_a_node: could not extract subject/predicate/value from input; "
                "subject=%s predicate=%s value=%s",
                subject, predicate, value,
            )
            return {
                "write_result": None,
                "route": "end",
                "output_text": (
                    f"crew_a: could not extract entity/predicate/value from: {user_input!r}"
                ),
            }

        # 3. Write claim via MempillRememberTool
        try:
            raw = remember_tool.invoke({
                "agent_id": agent_id,
                "subject": subject,
                "predicate": predicate,
                "value": value,
                "valid_from": valid_from,
                "confidence": 1.0,
                "provenance_channel": "UserAsserted",
            })
            write_json = json.loads(raw)
        except Exception as exc:
            log.error("crew_a_node: remember_tool failed: %s", exc)
            return {"error": str(exc), "route": "end", "output_text": f"crew_a error: {exc}"}

        is_contested = write_json.get("is_contested", False)
        log.info(
            "crew_a_node: wrote %s/%s=%r disposition=%s is_contested=%s",
            subject, predicate, value,
            write_json.get("disposition"), is_contested,
        )

        # 4. Detect Contested → HITL
        if is_contested:
            pending = _build_pending_contested(subject, predicate, write_json, adapter, agent_id)
            return {
                "write_result": raw,
                "pending_contested": pending,
                "route": "hitl",
                "output_text": (
                    f"crew_a: Contested write for {subject}/{predicate} — escalating to HITL"
                ),
            }

        # 5. Clean write
        return {
            "write_result": raw,
            "pending_contested": None,
            "route": "end",
            "output_text": (
                f"crew_a: wrote {subject}/{predicate}={value!r} "
                f"(disposition={write_json.get('disposition')})"
            ),
        }

    crew_a_node.__name__ = "crew_a_node"
    return crew_a_node


# ── Crew B node factory ───────────────────────────────────────────────────────

def make_crew_b_node(
    remember_tool: "MempillRememberTool",
    rag_write_tool: "RAGWriteTool",
    adapter,
    crew=None,
):
    """Factory: returns the crew_b_node function bound to its tools.

    If `crew` is a CrewAI Crew object, the node invokes crew.kickoff(inputs=...)
    instead of the shell heuristics.  Shell path is the no-API-key fallback.
    """

    def crew_b_node(state: ExecAssistantState) -> dict:
        """Crew B node — research: RAG write bulk + distilled claim → detect Contested.

        Live path  (crew is not None): delegates to CrewAI crew.kickoff().
        Shell path (crew is None):     deterministic heuristic extraction (W3).
        """
        user_input = state.get("user_input", "")
        agent_id = state.get("agent_id", AGENT_ID_DEFAULT)

        log.info("crew_b_node: processing research input=%r (crew=%s)", user_input[:80], type(crew).__name__ if crew else "shell")

        # ── CrewAI live path ──────────────────────────────────────────────────
        if crew is not None:
            try:
                result = crew.kickoff(inputs={
                    "user_request": user_input,
                    "agent_id": agent_id,
                })
                raw_output = str(result)
                import re as _re3
                is_contested = bool(_re3.search(r'"is_contested"\s*:\s*true', raw_output, _re3.IGNORECASE))
                subject, predicate, value = _extract_entity_predicate_value(user_input)
                if is_contested and subject and predicate:
                    pending = _build_pending_contested(subject, predicate, {}, adapter, agent_id)
                    return {
                        "write_result": raw_output,
                        "pending_contested": pending,
                        "route": "hitl",
                        "output_text": f"crew_b [crewai]: Contested distilled claim for {subject}/{predicate} — escalating to HITL",
                    }
                return {
                    "write_result": raw_output,
                    "pending_contested": None,
                    "route": "end",
                    "output_text": f"crew_b [crewai]: {raw_output[:200]}",
                }
            except Exception as exc:
                log.error("crew_b_node: CrewAI kickoff failed: %s — falling back to shell", exc)
                # Fall through to shell path on error

        # ── Shell path (W3 deterministic heuristics) ──────────────────────────
        # 1. Write raw research to RAG (bulk context — stays in RAG, not mempill)
        rag_write_tool.invoke({
            "text": user_input,
            "namespace": "research",
        })

        # 2. Heuristic extraction of distilled claim (W3 shell)
        subject, predicate, value = _extract_entity_predicate_value(user_input)

        if not subject or not predicate or not value:
            log.warning(
                "crew_b_node: could not extract distilled claim from input; "
                "subject=%s predicate=%s value=%s",
                subject, predicate, value,
            )
            return {
                "write_result": None,
                "route": "end",
                "output_text": (
                    f"crew_b: wrote research to RAG; no distilled claim extracted from: {user_input!r}"
                ),
            }

        # Infer valid_from from context (research crew uses lower confidence)
        # Look for year mentions in the text — simplistic for W3 shell
        valid_from: Optional[str] = None
        import re
        m = re.search(r"\b(20\d{2}(?:-\d{2})?)\b", user_input)
        if m:
            valid_from = m.group(1)

        # 3. Distil to mempill with ExternalFirstHand provenance
        try:
            raw = remember_tool.invoke({
                "agent_id": agent_id,
                "subject": subject,
                "predicate": predicate,
                "value": value,
                "valid_from": valid_from,
                "confidence": 0.85,
                "provenance_channel": "ExternalFirstHand",
            })
            write_json = json.loads(raw)
        except Exception as exc:
            log.error("crew_b_node: remember_tool failed: %s", exc)
            return {"error": str(exc), "route": "end", "output_text": f"crew_b error: {exc}"}

        is_contested = write_json.get("is_contested", False)
        log.info(
            "crew_b_node: distilled %s/%s=%r disposition=%s is_contested=%s",
            subject, predicate, value,
            write_json.get("disposition"), is_contested,
        )

        # 4. Detect Contested → HITL
        if is_contested:
            pending = _build_pending_contested(subject, predicate, write_json, adapter, agent_id)
            return {
                "write_result": raw,
                "pending_contested": pending,
                "route": "hitl",
                "output_text": (
                    f"crew_b: Contested distilled claim for {subject}/{predicate} — escalating to HITL"
                ),
            }

        return {
            "write_result": raw,
            "pending_contested": None,
            "route": "end",
            "output_text": (
                f"crew_b: distilled {subject}/{predicate}={value!r} to mempill "
                f"(disposition={write_json.get('disposition')})"
            ),
        }

    crew_b_node.__name__ = "crew_b_node"
    return crew_b_node


# ── Crew C node factory ───────────────────────────────────────────────────────

def make_crew_c_node(
    recall_tool: "MempillRecallTool",
    audit_tool: "MempillAuditTool",
    crew=None,
):
    """Factory: returns the crew_c_node function bound to its tools.

    Crew C is READ-ONLY — it NEVER writes to mempill.

    If `crew` is a CrewAI Crew object, the node invokes crew.kickoff(inputs=...)
    instead of the shell heuristics.  Shell path is the no-API-key fallback.
    """

    def crew_c_node(state: ExecAssistantState) -> dict:
        """Crew C node — scheduling/briefing/recall/audit: read-only mempill access.

        Live path  (crew is not None): delegates to CrewAI crew.kickoff().
        Shell path (crew is None):     deterministic heuristic reads (W3).

        Crew C is strictly READ-ONLY — the live path passes no write-capable tools
        to the crew (enforced at crew construction in build_crew_c_agents).
        """
        user_input = state.get("user_input", "")
        agent_id = state.get("agent_id", AGENT_ID_DEFAULT)
        intent = state.get("intent", IntentLabel.RECALL_HISTORY)

        log.info("crew_c_node: intent=%s input=%r (crew=%s)", intent, user_input[:80], type(crew).__name__ if crew else "shell")

        # ── CrewAI live path ──────────────────────────────────────────────────
        if crew is not None:
            try:
                result = crew.kickoff(inputs={
                    "user_request": user_input,
                    "agent_id": agent_id,
                    "intent": str(intent),
                })
                raw_output = str(result)
                return {
                    "recall_result": raw_output,
                    "output_text": f"crew_c [crewai]: {raw_output[:300]}",
                }
            except Exception as exc:
                log.error("crew_c_node: CrewAI kickoff failed: %s — falling back to shell", exc)
                # Fall through to shell path on error

        # ── Shell path (W3 deterministic reads) ───────────────────────────────
        # ── COMPLIANCE_AUDIT path ─────────────────────────────────────────────
        if intent == IntentLabel.COMPLIANCE_AUDIT:
            raw = audit_tool.invoke({"agent_id": agent_id, "limit": 100})
            audit_data = json.loads(raw)
            count = audit_data.get("entry_count", 0)
            return {
                "audit_result": raw,
                "output_text": (
                    f"crew_c [compliance_audit]: {count} audit entries retrieved "
                    f"for agent_id={agent_id}."
                ),
            }

        # ── Extract subject and predicate for recall paths ────────────────────
        subject, predicate, _ = _extract_entity_predicate_value(user_input)

        # Fall back to alice-chen/employer if we can't parse — sensible default for demo
        if not subject:
            subject = "alice-chen"
        if not predicate:
            predicate = "employer"

        # ── RECALL_HISTORY path — bi-temporal query ───────────────────────────
        if intent == IntentLabel.RECALL_HISTORY:
            # Extract valid_at or as_of_tx_time hints from the text (W3 shell heuristic)
            valid_at: Optional[str] = None
            as_of_tx_time: Optional[str] = None

            lower = user_input.lower()
            # "as of tx" / "what did we know" → as_of_tx_time axis
            if any(k in lower for k in ["what did the assistant believe", "when i booked", "as of tx"]):
                # Use a fixed demo timestamp for the Austin booking scenario
                as_of_tx_time = "2025-01-15T00:00:00Z"
            elif any(k in lower for k in ["in q1", "q1 board", "january 1", "jan 1", "what was"]):
                valid_at = "2025-01-01T00:00:00Z"

            raw = recall_tool.invoke({
                "agent_id": agent_id,
                "subject": subject,
                "predicate": predicate,
                "valid_at": valid_at,
                "as_of_tx_time": as_of_tx_time,
            })
            recall_data = json.loads(raw)
            return {
                "recall_result": raw,
                "output_text": (
                    f"crew_c [recall_history]: {subject}/{predicate}={recall_data.get('value')!r} "
                    f"status={recall_data.get('status')} "
                    f"(valid_at={valid_at} as_of_tx_time={as_of_tx_time})"
                ),
            }

        # ── PREPARE_BRIEFING path — current recall for multiple attributes ────
        results = {}
        for pred_key in ("employer", "city", "dietary_restriction"):
            if resolve_predicate(pred_key) is None:
                continue
            raw = recall_tool.invoke({
                "agent_id": agent_id,
                "subject": subject,
                "predicate": pred_key,
            })
            r = json.loads(raw)
            results[pred_key] = {
                "value": r.get("value"),
                "status": r.get("status"),
                "is_contested": r.get("is_contested"),
            }

        omitted = [k for k, v in results.items() if v.get("is_contested")]
        included = {k: v["value"] for k, v in results.items() if not v.get("is_contested")}

        briefing_parts = [f"{k}={v!r}" for k, v in included.items()]
        briefing = f"Briefing for {subject}: " + ", ".join(briefing_parts)
        if omitted:
            briefing += f" [OMITTED (Contested): {', '.join(omitted)}]"

        return {
            "recall_result": json.dumps(results),
            "briefing_text": briefing,
            "output_text": briefing,
        }

    crew_c_node.__name__ = "crew_c_node"
    return crew_c_node
