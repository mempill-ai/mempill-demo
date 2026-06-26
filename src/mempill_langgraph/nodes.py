"""mempill_langgraph.nodes — retrieve_memory, respond, write_memory graph nodes.

IMPORT BOUNDARY: this module MUST NOT import mempill directly.
It depends only on the MemoryStore Protocol and domain types from mempill_demo.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Callable, Optional

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

from mempill_demo.domain.models import BeliefView, CommandKind, ParsedCommand
from mempill_demo.ports.memory import MemoryStore

from mempill_langgraph.extraction import ClaimExtractResult
from mempill_langgraph.prompts import (
    CONTESTED_BLOCK,
    EMPTY_BLOCK,
    EXTRACTION_PROMPT,
    MEMORY_SYSTEM_PREFIX,
    NO_BELIEF_BLOCK,
    RESOLVED_BLOCK,
)
from mempill_langgraph.state import AgentState

log = logging.getLogger("mempill.demo")

# ── Subject/predicate extraction heuristics ──────────────────────────────────

# Patterns: (regex, subject_group, predicate_literal)
_SUBJECT_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"who\s+is\s+(?:the\s+)?ceo\s+of\s+(\w+)", re.I), r"\1:ceo", "held_by"),
    (re.compile(r"(?:ceo|chief\s+executive)\s+of\s+(\w+)", re.I), r"\1:ceo", "held_by"),
    (re.compile(r"where\s+(?:does|did)\s+(\w+)\s+live", re.I), r"\1", "lives_in"),
    (re.compile(r"where\s+is\s+(\w+)\s+(?:located|based)", re.I), r"\1", "location"),
    (re.compile(r"what\s+is\s+(\w+)['’]?s?\s+role", re.I), r"\1", "role"),
    (re.compile(r"(\w+)\s+(?:is|are)\s+the\s+ceo\s+of\s+(\w+)", re.I), r"\2:ceo", "held_by"),
    (re.compile(r"(\w+)['’]?s?\s+role\s+(?:at|in)\s+(\w+)", re.I), r"\1", "role"),
]


def _extract_subject_predicate(text: str) -> tuple[Optional[str], Optional[str]]:
    """Heuristic keyword-pattern extraction — no LLM call, no json.loads."""
    for pattern, subj_template, pred in _SUBJECT_PATTERNS:
        m = pattern.search(text)
        if m:
            # Replace back-references in template using match groups
            subject = pattern.sub(subj_template, text[m.start():m.end()]).strip()
            # Re-apply simpler group substitution
            try:
                groups = m.groups()
                s = subj_template
                for i, g in enumerate(groups, start=1):
                    s = s.replace(f"\\{i}", (g or "").lower())
                subject = s
            except Exception:
                subject = (m.group(1) or "").lower()
            return subject.lower(), pred
    return None, None


# ── Belief formatting (deterministic Python — not LLM) ───────────────────────

def _format_belief(belief: BeliefView) -> str:
    """Map a BeliefView to a deterministic memory-block string for the respond node."""
    if belief.value is None and not belief.alternatives:
        return NO_BELIEF_BLOCK.format(subject=belief.subject, predicate=belief.predicate)

    if belief.status in ("Contested",) and belief.alternatives:
        alt = belief.alternatives[0]
        return CONTESTED_BLOCK.format(
            subject=belief.subject,
            predicate=belief.predicate,
            value_a=belief.value,
            conf_a=belief.conf if belief.conf is not None else "?",
            start_a=belief.vt_start or "?",
            end_a=belief.vt_end or "open",
            value_b=alt.value,
            conf_b=alt.conf if alt.conf is not None else "?",
            start_b=alt.vt_start or "?",
            end_b=alt.vt_end or "open",
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


# ── Node factory ─────────────────────────────────────────────────────────────

def make_nodes(
    memory_store: MemoryStore,
    llm: Any,
    extractor: Optional[Callable[[str], ClaimExtractResult]] = None,
):
    """
    Return the three node functions closed over memory_store and llm.

    The optional `extractor` parameter accepts a callable (str -> ClaimExtractResult)
    to override the default llm.with_structured_output binding. This is used in
    offline tests where FakeMessagesListChatModel cannot cleanly drive tool-calls
    via with_structured_output in all cases.

    If extractor is None, the default llm.with_structured_output(ClaimExtractResult)
    binding is used (production path — requires a real or tool-call-capable model).
    """
    # Build the default extraction chain once at construction time.
    _extractor_llm = extractor or _make_default_extractor(llm)

    # ── Node A: retrieve_memory ───────────────────────────────────────────────

    def retrieve_memory(state: AgentState, config: RunnableConfig, *, store: BaseStore) -> dict:
        """
        Determine (subject, predicate) from the latest human message via heuristics,
        call MemoryStore.recall via the PORT, format BeliefView into memory_context.

        The `store: Any` parameter (LangGraph's store injection via Pattern A) is
        accepted in the signature but intentionally unused — mempill is accessed
        via the `memory_store` closure, not LangGraph's InMemoryStore.
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

        subject, predicate = _extract_subject_predicate(latest_human)

        if subject is None or predicate is None:
            # No identifiable subject — greeting or small-talk
            return {"memory_context": EMPTY_BLOCK}

        try:
            belief = memory_store.recall(subject, predicate)
            memory_context = _format_belief(belief)
        except Exception:
            # Graceful degradation — don't crash on recall failure
            memory_context = NO_BELIEF_BLOCK.format(subject=subject, predicate=predicate)

        return {"memory_context": memory_context}

    # ── Node B: respond ───────────────────────────────────────────────────────

    def respond(state: AgentState) -> dict:
        """Call the LLM with full message history + memory context as SystemMessage prefix."""
        memory_context = state.get("memory_context", "")
        system_content = MEMORY_SYSTEM_PREFIX.format(memory_context=memory_context)
        system_msg = SystemMessage(content=system_content)
        messages_to_send = [system_msg] + list(state["messages"])
        ai_reply = llm.invoke(messages_to_send)
        return {"messages": [ai_reply]}

    # ── Node C: write_memory ──────────────────────────────────────────────────

    def write_memory(state: AgentState) -> dict:
        """
        Extract new factual claims from the latest AI reply using structured output
        and ingest each into the MemoryStore with mapped provenance.

        Returns {} (no state mutation) — this is a side-effect-only node.
        """
        messages = state.get("messages", [])
        last_ai_content = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                last_ai_content = msg.content or ""
                break
            if msg.__class__.__name__ == "AIMessage":
                last_ai_content = msg.content or ""
                break

        if not last_ai_content:
            return {}

        # Recall re-entry firewall: if the AI reply merely restates the primary
        # value from memory_context, treat as RECALL_REENTRY, not INGEST.
        memory_context = state.get("memory_context", "")

        try:
            result: ClaimExtractResult = _extractor_llm(
                EXTRACTION_PROMPT.format(assistant_message=last_ai_content)
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
            provenance_str = "UserAsserted" if claim.is_user_asserted else "ModelDerived"

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


def _make_default_extractor(llm: Any) -> Callable[[str], ClaimExtractResult]:
    """Bind llm.with_structured_output for production use."""
    extractor_llm = llm.with_structured_output(ClaimExtractResult)

    def _extract(prompt: str) -> ClaimExtractResult:
        return extractor_llm.invoke(prompt)

    return _extract


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
