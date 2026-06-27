"""mempill_langgraph.nodes — retrieve_memory, respond, write_memory graph nodes.

IMPORT BOUNDARY: this module MUST NOT import mempill directly.
It depends only on the MemoryStore Protocol and domain types from mempill_demo.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

from mempill_demo.domain.models import BeliefView, CommandKind, ParsedCommand
from mempill_demo.ports.memory import MemoryStore

from mempill_langgraph.extraction import ClaimExtractResult, DecisionClassifyResult, KeyExtractResult
from mempill_langgraph.prompts import (
    CONTESTED_BLOCK,
    DECISION_CLASSIFY_PROMPT,
    EMPTY_BLOCK,
    EXTRACTION_PROMPT,
    KEY_EXTRACTION_PROMPT,
    MEMORY_SYSTEM_PREFIX,
    NO_BELIEF_BLOCK,
    RESOLVED_BLOCK,
    TIMELINE_BLOCK,
    TIMELINE_ENTRY_LINE,
)
from mempill_langgraph.state import AgentState, ContestedInfo, PendingDecision

log = logging.getLogger("mempill.demo")


# ── Belief formatting (deterministic Python — not LLM) ───────────────────────

def _format_belief(belief: BeliefView) -> str:
    """Map a BeliefView to a deterministic memory-block string for the respond node."""
    if belief.value is None and not belief.alternatives:
        return NO_BELIEF_BLOCK.format(subject=belief.subject, predicate=belief.predicate)

    if belief.status in ("Contested", "Conflict") and (belief.alternatives or belief.value is not None):
        # Contested has no winner. The candidates may be (primary + alternatives) or,
        # when there is no primary, entirely in `alternatives`. Surface the first two.
        candidates = []
        if belief.value is not None:
            candidates.append((belief.value, belief.conf, belief.vt_start, belief.vt_end))
        candidates.extend((a.value, a.conf, a.vt_start, a.vt_end) for a in belief.alternatives)
        (va, ca, sa, ea) = candidates[0]
        (vb, cb, sb, eb) = candidates[1] if len(candidates) > 1 else (None, None, "", "")
        return CONTESTED_BLOCK.format(
            subject=belief.subject,
            predicate=belief.predicate,
            value_a=va, conf_a=ca if ca is not None else "?", start_a=sa or "?", end_a=ea or "open",
            value_b=vb, conf_b=cb if cb is not None else "?", start_b=sb or "?", end_b=eb or "open",
        )

    return RESOLVED_BLOCK.format(
        status=belief.status,
        subject=belief.subject,
        predicate=belief.predicate,
        value=belief.value,
        conf=belief.conf if belief.conf is not None else "?",
        vt_start=belief.vt_start or "?",
        vt_end=belief.vt_end or "open",
        provenance=belief.provenance,
        corroboration=belief.corroboration,
    )


# ── Timeline formatting (deterministic Python — not LLM) ─────────────────────

def _format_timeline(subject: str, predicate: str, entries: list) -> str:
    """Format a list of TimelineEntry objects into a compact TIMELINE_BLOCK string.

    Only called when len(entries) > 1 — single-entry histories add no useful context.
    Entries are ordered oldest→newest (Superseded first, Current last) as returned
    by mempill.history().
    """
    lines = []
    for e in entries:
        vf = e.valid_from or "?"
        vu = e.valid_until or "open"
        lines.append(
            TIMELINE_ENTRY_LINE.format(
                status=e.status,
                value=e.value,
                valid_from=vf,
                valid_until=vu,
            )
        )
    return TIMELINE_BLOCK.format(
        subject=subject,
        predicate=predicate,
        entries="\n".join(lines),
    )


# ── Node factory ─────────────────────────────────────────────────────────────

def make_nodes(
    memory_store: MemoryStore,
    llm: Any,
    extractor: Optional[Callable[[str], ClaimExtractResult]] = None,
    key_extractor: Optional[Callable[[str], KeyExtractResult]] = None,
    decision_classifier: Optional[Callable[[str], DecisionClassifyResult]] = None,
):
    """
    Return the three node functions closed over memory_store and llm.

    Injectable seams (all override the default llm.with_structured_output binding):
      extractor:            str -> ClaimExtractResult  (write_memory)
      key_extractor:        str -> KeyExtractResult    (retrieve_memory key lookup)
      decision_classifier:  str -> DecisionClassifyResult  (conversational adjudication)

    All seams use the same canonical key convention so write↔read always match.
    When a parameter is None the default llm.with_structured_output binding is used
    (production path — requires a real or tool-call-capable model).
    """
    # Build the default extraction chains once at construction time.
    _extractor_llm = extractor or _make_default_extractor(llm)
    _key_extractor_llm = key_extractor or _make_default_key_extractor(llm)

    # Decision classifier is built lazily — only when a pending_decision is actually
    # present in state.  This avoids calling llm.with_structured_output() for models
    # (like FakeMessagesListChatModel) that raise NotImplementedError for that method.
    _explicit_decision_classifier = decision_classifier
    _lazy_classifier: list[Callable[[str], DecisionClassifyResult]] = []  # mutable cell

    def _get_decision_classifier() -> Callable[[str], DecisionClassifyResult]:
        if _explicit_decision_classifier is not None:
            return _explicit_decision_classifier
        if not _lazy_classifier:
            _lazy_classifier.append(_make_default_decision_classifier(llm))
        return _lazy_classifier[0]

    # ── Node A: retrieve_memory ───────────────────────────────────────────────

    def retrieve_memory(state: AgentState, config: RunnableConfig, *, store: BaseStore) -> dict:
        """
        1. If pending_decision is set from a prior Contested turn, classify the
           current user message to pick incumbent / challenger / neither.
           - challenger  → submit_adjudication("Affirm")  (challenger wins)
           - incumbent   → submit_adjudication("Deny")    (incumbent stands)
           - neither     → leave pending; continue with normal recall
           After submitting, clear pending_decision and recall the freshly-resolved belief.
           IMPORTANT: when we process a decision-turn, set a flag so write_memory
           does NOT ingest the user's short answer as a new claim.

        2. Extract (subject, predicate) from the latest human message and recall.

        3. AUTO-RECONCILE (silent, deterministic): if belief.status == "Contested",
           call memory_store.reconcile(subject, predicate) then recall again.
           Valid-time succession (most-recent wins) resolves the line without any
           user action.  Only a genuine tie remains Contested after this step.

        4. CONVERSATIONAL ADJUDICATION (residual tie): if, after auto-reconcile,
           the belief is STILL Contested AND list_pending() has an entry for this
           (subject, predicate), record a PendingDecision in state.  The respond
           node will ask the user which value is correct.

        The `store: BaseStore` parameter is accepted (Pattern-A LangGraph injection)
        but unused — mempill is accessed via the `memory_store` closure.
        """
        # Get the latest human message
        messages = state.get("messages", [])
        latest_human = ""
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "human":
                latest_human = msg.content or ""
                break
            if msg.__class__.__name__ == "HumanMessage":
                latest_human = msg.content or ""
                break

        # ── Step 1: Process any outstanding pending_decision ──────────────────
        # PendingDecision is a TypedDict → dict at runtime; use [] not . access.
        existing_decision: Optional[dict] = state.get("pending_decision")
        if existing_decision is not None and latest_human:
            inc_val = existing_decision["incumbent_value"]
            chal_val = existing_decision["challenger_value"]
            handle_id = existing_decision["handle_id"]
            dec_subject = existing_decision["subject"]
            dec_predicate = existing_decision["predicate"]

            verdict_result = _classify_decision(
                latest_human,
                inc_val,
                chal_val,
                _get_decision_classifier(),
            )
            if verdict_result in ("challenger", "incumbent"):
                # Map to engine API verdicts
                engine_verdict = "Affirm" if verdict_result == "challenger" else "Deny"
                try:
                    memory_store.submit(handle_id, engine_verdict)
                    log.info(
                        "conversational adjudication: %s/%s → %s (verdict=%s)",
                        dec_subject, dec_predicate, verdict_result, engine_verdict,
                    )
                except Exception as exc:
                    log.warning("conversational adjudication submit failed: %s", exc)

                # Recall the freshly-resolved belief
                try:
                    resolved_belief = memory_store.recall(dec_subject, dec_predicate)
                    memory_context = _format_belief(resolved_belief)
                except Exception:
                    memory_context = NO_BELIEF_BLOCK.format(
                        subject=dec_subject, predicate=dec_predicate,
                    )

                # Clear pending_decision; mark this turn as a decision turn so
                # write_memory does not ingest the user's short pick as a new claim.
                return {
                    "memory_context": memory_context,
                    "pending_decision": None,
                    "_decision_turn": True,  # consumed by write_memory guard
                }
            else:
                # User said "neither" or off-topic — leave pending, proceed normally
                log.info(
                    "conversational adjudication: user said 'neither' for %s/%s — staying pending",
                    dec_subject, dec_predicate,
                )

        # ── Step 2: Normal recall ─────────────────────────────────────────────
        conversation_context = _build_conversation_context(messages, latest_human)
        subject, predicate = _extract_canonical_key(latest_human, _key_extractor_llm, conversation_context)

        if not subject or not predicate:
            # Key was not freshly extracted — try to fall back to the last resolved key
            # (handles follow-up / pronoun turns: "were some persons before him?")
            subject = state.get("last_subject") or None
            predicate = state.get("last_predicate") or None

        if not subject or not predicate:
            # No subject from this turn AND no carry-over — genuine greeting / small-talk
            return {"memory_context": EMPTY_BLOCK}

        try:
            belief = memory_store.recall(subject, predicate)
        except Exception:
            return {"memory_context": NO_BELIEF_BLOCK.format(subject=subject, predicate=predicate)}

        # ── Step 3: Auto-reconcile (silent, deterministic) ────────────────────
        if belief.status in ("Contested", "Conflict"):
            try:
                memory_store.reconcile(subject, predicate)
                belief = memory_store.recall(subject, predicate)
                log.info(
                    "auto-reconcile %s/%s → status=%s",
                    subject, predicate, belief.status,
                )
            except Exception as exc:
                log.warning("auto-reconcile failed for %s/%s: %s", subject, predicate, exc)

        # ── Step 3b: Timeline injection ───────────────────────────────────────
        # Fetch the full ordered history for this (subject, predicate) and append a
        # compact TIMELINE block to memory_context when there are >1 entries.
        # This lets the LLM answer "who was before / prior / history" questions without
        # any separate intent classifier — both current and historical are in one context.
        timeline_block = ""
        try:
            timeline_entries = memory_store.timeline_history(subject, predicate)
            if len(timeline_entries) > 1:
                timeline_block = _format_timeline(subject, predicate, timeline_entries)
                log.info(
                    "timeline_injection %s/%s → %d entries",
                    subject, predicate, len(timeline_entries),
                )
        except AttributeError:
            # memory_store does not implement timeline_history (e.g. base FakeMemoryStore)
            pass
        except Exception as exc:
            log.warning("timeline_history failed for %s/%s: %s", subject, predicate, exc)

        # ── Step 4: Conversational adjudication setup (residual tie) ──────────
        new_pending: Optional[PendingDecision] = None
        new_contested: Optional[ContestedInfo] = None

        if belief.status in ("Contested", "Conflict"):
            # Build the candidate list from the same recall data used by _format_belief.
            # This is set UNCONDITIONALLY whenever the belief is Contested/Conflict so
            # that respond can short-circuit to a deterministic reply regardless of
            # whether a pending adjudication item was correlated (closes the M1 gap).
            raw_candidates = []
            if belief.value is not None:
                raw_candidates.append({
                    "value": belief.value,
                    "conf": belief.conf,
                    "vt_start": belief.vt_start or "",
                    "vt_end": belief.vt_end or "",
                })
            for alt in (belief.alternatives or []):
                raw_candidates.append({
                    "value": alt.value,
                    "conf": alt.conf,
                    "vt_start": alt.vt_start or "",
                    "vt_end": alt.vt_end or "",
                })
            new_contested = {
                "subject": subject,
                "predicate": predicate,
                "candidates": raw_candidates,
            }
            log.info(
                "retrieve_memory: contested %s/%s — %d candidate(s)",
                subject, predicate, len(raw_candidates),
            )

            try:
                pending_list = memory_store.list_pending()
            except AttributeError:
                # memory_store does not implement list_pending (base MemoryStore)
                pending_list = []
            except Exception as exc:
                log.warning("list_pending failed: %s", exc)
                pending_list = []

            # Find a pending adjudication for this (subject, predicate)
            for pending_item in pending_list:
                # The engine does not attach subject/predicate to pending items directly;
                # correlate via the incumbent/challenger values present in the belief.
                # A pending item matches when its incumbent_value appears in the belief
                # candidates (belief has no primary when Contested — candidates in alternatives).
                inc_val = pending_item.get("incumbent_value")
                chal_val = pending_item.get("challenger_value")
                handle_id = pending_item.get("handle_id", "")
                if inc_val and chal_val and handle_id:
                    # PendingDecision is a TypedDict (serializable plain dict at runtime)
                    new_pending = {
                        "subject": subject,
                        "predicate": predicate,
                        "handle_id": handle_id,
                        "incumbent_value": str(inc_val),
                        "challenger_value": str(chal_val),
                    }
                    break  # take the first matching pending item

        memory_context = _format_belief(belief)
        if timeline_block:
            memory_context = memory_context + "\n\n" + timeline_block
        result: dict = {
            "memory_context": memory_context,
            "last_subject": subject,
            "last_predicate": predicate,
        }
        if new_pending is not None:
            result["pending_decision"] = new_pending
        if new_contested is not None:
            result["contested"] = new_contested
        return result

    # ── Node B: respond ───────────────────────────────────────────────────────

    def respond(state: AgentState) -> dict:
        """
        Build the AI reply.

        M1 SECURITY SHORT-CIRCUIT: when the recalled belief is Contested/Conflict,
        the reply is ALWAYS built deterministically in Python — the LLM is NEVER
        invoked for a contested turn. Two paths:

          1. pending_decision is set (adjudication item correlated): render from
             incumbent/challenger pair (today's behavior; drives the next-turn verdict).

          2. contested is set but pending_decision is None (no correlated adjudication
             item, or list_pending returned []): render from contested["candidates"]
             listing all values generically — closes the M1 gap where a prompt-injection
             could previously reach the LLM when the value-matching heuristic missed.

        For all other turns, the LLM is invoked normally with the full message
        history + memory context as a SystemMessage prefix.
        """
        pending: Optional[dict] = state.get("pending_decision")
        if pending is not None:
            # Path 1: adjudication-correlated contested turn — render from pending pair.
            subject = pending.get("subject", "?")
            predicate = pending.get("predicate", "?")
            incumbent = pending.get("incumbent_value", "?")
            challenger = pending.get("challenger_value", "?")
            det_text = (
                f'⚖️ "{subject} / {predicate}" is contested — I cannot choose:\n'
                f"   • {incumbent}\n"
                f"   • {challenger}\n"
                "Which is correct?"
            )
            log.info(
                "respond: deterministic contested reply (pending) for %s/%s (LLM bypassed)",
                subject, predicate,
            )
            return {"messages": [AIMessage(content=det_text)]}

        contested: Optional[dict] = state.get("contested")
        if contested is not None:
            # Path 2: contested belief with no correlated adjudication item.
            # Render all candidate values deterministically — LLM is bypassed.
            subject = contested.get("subject", "?")
            predicate = contested.get("predicate", "?")
            candidates: list = contested.get("candidates", [])
            bullet_lines = "\n".join(
                f"   • {c.get('value', '?')}" for c in candidates
            ) if candidates else "   • (no candidates)"
            det_text = (
                f'⚖️ "{subject} / {predicate}" is contested — I cannot choose:\n'
                f"{bullet_lines}\n"
                "Which is correct?"
            )
            log.info(
                "respond: deterministic contested reply (no-pending) for %s/%s "
                "— %d candidate(s) (LLM bypassed)",
                subject, predicate, len(candidates),
            )
            return {"messages": [AIMessage(content=det_text)]}

        memory_context = state.get("memory_context", "")
        system_content = MEMORY_SYSTEM_PREFIX.format(memory_context=memory_context)
        system_msg = SystemMessage(content=system_content)
        messages_to_send = [system_msg] + list(state["messages"])
        ai_reply = llm.invoke(messages_to_send)
        return {"messages": [ai_reply]}

    # ── Node C: write_memory ──────────────────────────────────────────────────

    def write_memory(state: AgentState) -> dict:
        """
        Extract new factual claims from the latest human/user message using structured
        output and ingest each into the MemoryStore with mapped provenance.

        Extraction source is the USER's message (not the AI reply) so that:
        - Facts the user stated are ingested as UserAsserted (External) provenance.
        - The AI restating previously-recalled facts cannot create duplicate ingests.
        - HITL /review fires correctly for conflicting user-stated facts.

        Decision-turn guard: if retrieve_memory set _decision_turn=True, the user's
        message was a short adjudication answer ("Bob", "the first one", "incumbent").
        Ingesting it as a factual claim would create a spurious belief.  Skip the
        entire extraction step for that turn.

        Returns {} (no state mutation) — this is a side-effect-only node.
        """
        # Decision-turn guard: message was "Bob" / "the first one" etc. — not a claim.
        if state.get("_decision_turn"):
            log.debug("write_memory: skipping — decision-turn guard active")
            return {}

        messages = state.get("messages", [])
        last_human_content = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_human_content = msg.content or ""
                break
            if msg.__class__.__name__ == "HumanMessage":
                last_human_content = msg.content or ""
                break

        if not last_human_content:
            return {}

        # Recall re-entry firewall: if the user message merely restates the primary
        # value from memory_context, treat as RECALL_REENTRY, not INGEST.
        memory_context = state.get("memory_context", "")

        # Build conversation context for the claim extractor (same approach as retrieve_memory).
        write_conversation_context = _build_conversation_context(messages, last_human_content)

        try:
            result: ClaimExtractResult = _extractor_llm(
                EXTRACTION_PROMPT.format(
                    context=write_conversation_context,
                    user_message=last_human_content,
                )
            )
        except Exception as exc:
            # Never crash write_memory — graceful no-op on extraction failure
            log.warning("write_memory: claim extraction failed: %s", exc)
            return {}

        if not result.claims:
            return {}

        user_id = state.get("user_id", "unknown")
        agent_id = state.get("agent_id", "langgraph-agent")

        for claim in result.claims:
            # Recall re-entry detection: if the (subject, predicate, value) triple
            # matches what was just recalled this turn, skip to avoid amplification.
            if _is_recall_reentry(claim.subject, claim.predicate, str(claim.value), memory_context):
                continue

            kind = CommandKind.INGEST
            # Claims extracted from the user's own message are always UserAsserted —
            # the user is the source of truth for what they stated directly.
            # This maps to prov={'type':'External','kind':'UserAsserted'} in the engine,
            # ensuring conflicts trigger QueuedForAdjudication (HITL /review).
            provenance_str = "UserAsserted"

            cmd = ParsedCommand(
                kind=kind,
                subject=claim.subject,
                predicate=claim.predicate,
                value=claim.value,
                conf=claim.conf,
                since=claim.since,
                until=claim.until,
                extra={
                    "provenance_str": provenance_str,
                    "cardinality": "Functional",
                },
            )
            try:
                memory_store.ingest(cmd)
            except Exception as exc:
                # Don't crash the conversation on a failed write — but surface it.
                log.warning(
                    "write_memory: ingest failed for %s/%s=%r: %s",
                    claim.subject, claim.predicate, claim.value, exc,
                )

        return {}

    return retrieve_memory, respond, write_memory


def _make_default_decision_classifier(llm: Any) -> Callable[[str], DecisionClassifyResult]:
    """Bind llm.with_structured_output(DecisionClassifyResult) for production use."""
    classifier_llm = llm.with_structured_output(DecisionClassifyResult)

    def _classify(prompt: str) -> DecisionClassifyResult:
        return classifier_llm.invoke(prompt)

    return _classify


def _classify_decision(
    user_message: str,
    incumbent: str,
    challenger: str,
    classifier_fn: Callable[[str], DecisionClassifyResult],
) -> str:
    """
    Use the injected classifier_fn to determine whether the user picked the
    incumbent, the challenger, or neither.  Returns one of: "challenger",
    "incumbent", "neither".  Defaults to "neither" on any failure.
    """
    if not user_message or not user_message.strip():
        return "neither"
    try:
        prompt = DECISION_CLASSIFY_PROMPT.format(
            incumbent=incumbent,
            challenger=challenger,
            user_message=user_message,
        )
        result: DecisionClassifyResult = classifier_fn(prompt)
        verdict = (result.verdict or "neither").strip().lower()
        if verdict in ("challenger", "incumbent"):
            return verdict
        return "neither"
    except Exception as exc:
        log.warning("decision classifier failed: %s", exc)
        return "neither"


def _make_default_extractor(llm: Any) -> Callable[[str], ClaimExtractResult]:
    """Bind llm.with_structured_output(ClaimExtractResult) for production use."""
    extractor_llm = llm.with_structured_output(ClaimExtractResult)

    def _extract(prompt: str) -> ClaimExtractResult:
        return extractor_llm.invoke(prompt)

    return _extract


def _make_default_key_extractor(llm: Any) -> Callable[[str], KeyExtractResult]:
    """Bind llm.with_structured_output(KeyExtractResult) for production use."""
    key_extractor_llm = llm.with_structured_output(KeyExtractResult)

    def _extract_key(prompt: str) -> KeyExtractResult:
        return key_extractor_llm.invoke(prompt)

    return _extract_key


def _build_conversation_context(
    messages: list,
    current_message: str,
    max_messages: int = 6,
    max_chars_per_message: int = 200,
) -> str:
    """
    Build a short recent-conversation transcript from state messages, excluding
    the current message being extracted.  Used to give both extractors enough
    context to compose canonical keys across turns (e.g. "Who is a CEO?" + "the
    company is Acme" → acme:ceo).

    Returns an empty string when there is no prior context to include.
    """
    context_lines: list[str] = []
    prior_messages = [
        m for m in messages
        if (m.content if hasattr(m, "content") else "") != current_message
    ]
    # Take up to the last `max_messages` prior messages
    for msg in prior_messages[-max_messages:]:
        content = (msg.content or "") if hasattr(msg, "content") else ""
        # Truncate long messages
        if len(content) > max_chars_per_message:
            content = content[:max_chars_per_message] + "..."
        msg_type = getattr(msg, "type", None) or msg.__class__.__name__
        if msg_type == "human" or msg_type == "HumanMessage":
            context_lines.append(f"User: {content}")
        elif msg_type == "ai" or msg_type == "AIMessage":
            context_lines.append(f"Assistant: {content}")
    return "\n".join(context_lines)


def _extract_canonical_key(
    text: str,
    key_extractor_fn: Callable[[str], KeyExtractResult],
    context: str = "",
) -> tuple[Optional[str], Optional[str]]:
    """
    Use the injected key_extractor_fn to derive the canonical (subject, predicate)
    for a user question.  Returns (None, None) when no key is identifiable
    (greeting, small-talk, or extractor failure).

    The `context` string is a recent-conversation transcript inserted into the
    prompt so the extractor can compose cross-turn canonical keys (e.g. a prior
    turn asked about a CEO role; the current turn supplies the organization).
    The seam callable still receives a single `str` prompt — no signature change.
    """
    if not text or not text.strip():
        return None, None
    try:
        result: KeyExtractResult = key_extractor_fn(
            KEY_EXTRACTION_PROMPT.format(context=context, user_question=text)
        )
        subject = (result.subject or "").strip().lower() or None
        predicate = (result.predicate or "").strip().lower() or None
        return subject, predicate
    except Exception as exc:
        log.warning("retrieve_memory: key extraction failed: %s", exc)
        return None, None


def _is_recall_reentry(subject: str, predicate: str, value: str, memory_context: str) -> bool:
    """
    Detect if a claim is a re-statement of what was just recalled this turn.
    Simple string-match heuristic: checks if the value appears in the RESOLVED_BLOCK
    for this (subject, predicate) pair in the current memory_context.
    """
    if not memory_context or not value:
        return False
    # Look for the subject+predicate line in memory_context then check value proximity
    context_lower = memory_context.lower()
    if subject.lower() in context_lower and predicate.lower() in context_lower:
        if value.lower() in context_lower:
            return True
    return False
