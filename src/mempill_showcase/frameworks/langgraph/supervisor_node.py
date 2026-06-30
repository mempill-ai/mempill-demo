"""
mempill_showcase.frameworks.langgraph.supervisor_node — intent classification + routing.

Architecture:
  - Defines a `SupervisorClassifier` Protocol — any implementation works (mock, LLM, etc.).
  - `MockSupervisor` is a deterministic keyword/fixture classifier; NO API key, NO model call.
    It is the default implementation used in tests and the W3 graph.
  - `LLMSupervisor` (W6) calls a real Anthropic model via langchain_anthropic.ChatAnthropic.
    It is OPTIONAL: only instantiated when an ANTHROPIC_API_KEY is present in the environment.
    Never call it without a key — the constructor will raise or the API call will fail.
  - `supervisor_node` is the LangGraph node function: classifies intent and sets `route`.

Intent → Route mapping:
  UPDATE_CONTACT   → crew_a (intake crew: parse + write to mempill)
  RESEARCH         → crew_b (research crew: RAG + distilled claims)
  PREPARE_BRIEFING → crew_c (briefing crew: recall + compose output)
  RECALL_HISTORY   → crew_c (same crew, bi-temporal recall path)
  COMPLIANCE_AUDIT → crew_c (audit path within crew_c)

W6 seam:
  Replace MockSupervisor with an LLMSupervisor that calls a real model.
  The node function (`supervisor_node`) and the protocol do not change.
  Swap the classifier at graph build time via `build_graph(classifier=LLMSupervisor(...))`.

Environment variables (LLMSupervisor):
  ANTHROPIC_API_KEY  — required when using LLMSupervisor (raises at call time if absent).
  ANTHROPIC_MODEL    — optional; default is "claude-haiku-4-5".
"""
from __future__ import annotations

import logging
import os
from typing import Optional, Protocol, runtime_checkable

from mempill_showcase.frameworks.langgraph.state import ExecAssistantState, IntentLabel

log = logging.getLogger(__name__)


# ── Classifier protocol ───────────────────────────────────────────────────────

@runtime_checkable
class SupervisorClassifier(Protocol):
    """Classify a user-turn string into an IntentLabel constant.

    Implementations must be deterministic given the same input (for MockSupervisor)
    or best-effort (for LLM classifiers in W6+).
    """

    def classify(self, user_input: str) -> str:
        """Return one of the IntentLabel constants."""
        ...


# ── Mock (deterministic keyword) classifier ───────────────────────────────────

# Ordered: earlier entries win.
_KEYWORD_RULES: list[tuple[list[str], str]] = [
    # UPDATE_CONTACT — user is reporting a new fact about a contact
    (
        [
            "relocated", "moved to", "just told me", "now lives", "updated",
            "changed her", "changed his", "new address", "new city", "new role",
            "promoted to", "now works", "left ", "joined ", "hired ",
        ],
        IntentLabel.UPDATE_CONTACT,
    ),
    # COMPLIANCE_AUDIT — audit / compliance / exact belief queries
    (
        [
            "audit", "compliance", "exact belief", "show me the belief",
            "what did the assistant believe", "regulatory", "ledger",
        ],
        IntentLabel.COMPLIANCE_AUDIT,
    ),
    # RECALL_HISTORY — historical / point-in-time queries
    (
        [
            "what was", "what were", "in q1", "in q2", "in q3", "in q4",
            "on january", "on february", "on march", "on april", "on may",
            "on june", "on july", "on august", "on september", "on october",
            "on november", "on december",
            "when i booked", "at the time", "as of", "back in",
            "history of", "previous role", "prior city", "last known",
        ],
        IntentLabel.RECALL_HISTORY,
    ),
    # RESEARCH — external information lookup
    (
        [
            "who is the", "who's the", "find out", "look up", "research",
            "i read somewhere", "i heard", "news about", "new cto", "new ceo",
            "according to", "article", "linkedin",
        ],
        IntentLabel.RESEARCH,
    ),
    # PREPARE_BRIEFING — scheduling / briefing / communication drafting
    (
        [
            "prepare", "briefing", "brief", "draft", "schedule", "book",
            "arrange", "set up", "dinner", "lunch", "meeting", "email",
            "send", "include her", "include his",
        ],
        IntentLabel.PREPARE_BRIEFING,
    ),
]

# Fixture overrides: exact user_input string (lowercased) → intent.
# Used in tests to guarantee deterministic routing without keyword fragility.
_FIXTURE_OVERRIDES: dict[str, str] = {
    "intent:update_contact":   IntentLabel.UPDATE_CONTACT,
    "intent:research":         IntentLabel.RESEARCH,
    "intent:prepare_briefing": IntentLabel.PREPARE_BRIEFING,
    "intent:recall_history":   IntentLabel.RECALL_HISTORY,
    "intent:compliance_audit": IntentLabel.COMPLIANCE_AUDIT,
}

_INTENT_TO_ROUTE: dict[str, str] = {
    IntentLabel.UPDATE_CONTACT:   "crew_a",
    IntentLabel.RESEARCH:         "crew_b",
    IntentLabel.PREPARE_BRIEFING: "crew_c",
    IntentLabel.RECALL_HISTORY:   "crew_c",
    IntentLabel.COMPLIANCE_AUDIT: "crew_c",
}


class MockSupervisor:
    """Deterministic keyword + fixture classifier — no LLM, no API key.

    Classification order:
      1. Exact fixture override (for tests: pass "intent:recall_history" etc.)
      2. Keyword scan (first match wins, ordered by rule priority above)
      3. Default: RECALL_HISTORY (safe fallback — read-only)
    """

    def classify(self, user_input: str) -> str:
        lower = user_input.strip().lower()

        # 1. Fixture override
        if lower in _FIXTURE_OVERRIDES:
            return _FIXTURE_OVERRIDES[lower]

        # 2. Keyword scan
        for keywords, intent in _KEYWORD_RULES:
            for kw in keywords:
                if kw in lower:
                    log.debug("MockSupervisor: matched keyword %r → intent=%s", kw, intent)
                    return intent

        # 3. Default
        log.debug("MockSupervisor: no keyword match, defaulting to RECALL_HISTORY")
        return IntentLabel.RECALL_HISTORY


# ── LLM classifier (W6 — OPTIONAL, requires ANTHROPIC_API_KEY) ───────────────

_LLM_SYSTEM_PROMPT = """You are an intent classifier for an executive assistant system.
Classify the user's message into exactly one of these intents:
- UPDATE_CONTACT: user is reporting a NEW or UPDATED fact about a contact (city, role, employer, dietary restriction, travel preference, etc.). Key signal: the user is telling you something changed ("Alice moved to ...", "Bob is now ...", "Alice just told me she's ...").
- RECALL_HISTORY: user is asking a QUESTION about a known attribute of a known contact — this is the DEFAULT for any attribute question. Use this for: "What is X's <attr>?", "Who is X's employer?", "Where does X live?", "What does X eat?", "What was X's <attr> in <year>?", "What was X doing in Q1?", point-in-time queries, or any question that can be answered from memory about a known entity. If the question is about an attribute of a known contact (Alice, Bob, Acme, Jordan), ALWAYS use RECALL_HISTORY — never RESEARCH.
- RESEARCH: ONLY when the user explicitly asks to gather NEW EXTERNAL information about an entity — e.g. "Research <entity>", "Look up background on <entity>", "Find out about <company>", "What's in the news about <entity>", "I heard something about <entity> — can you look it up?". A plain attribute question ("What is Alice's dietary restriction?") is NEVER research — use RECALL_HISTORY instead.
- PREPARE_BRIEFING: user wants a briefing, summary, or draft for a meeting/email/event (e.g. "Brief me on Alice", "Prepare a meeting summary", "Draft an email to Bob").
- COMPLIANCE_AUDIT: user wants an audit trail, compliance check, or belief-state history (e.g. "Show me the audit ledger", "What did the system believe on date X?").

CRITICAL DISAMBIGUATION — RECALL_HISTORY vs RESEARCH:
  "What is Alice's dietary restriction?"  → RECALL_HISTORY  (attribute question about known contact)
  "Who is Alice's employer?"              → RECALL_HISTORY  (attribute question about known contact)
  "Where does Alice live?"                → RECALL_HISTORY  (attribute question about known contact)
  "What was Alice's city in 2024?"        → RECALL_HISTORY  (historical attribute query)
  "Research Acme Corp"                    → RESEARCH        (explicit external research request)
  "Look up background on Bob"             → RESEARCH        (explicit external lookup request)
  "Find out about Acme's new product"     → RESEARCH        (explicit find-out request)

When in doubt between RECALL_HISTORY and RESEARCH, ALWAYS choose RECALL_HISTORY.

Reply with ONLY the intent label (one of the five above), nothing else.
"""

_LLM_VALID_LABELS = frozenset(IntentLabel.ALL)


class LLMSupervisor:
    """Real-LLM intent classifier using langchain_anthropic.ChatAnthropic.

    This classifier is OPTIONAL and should only be instantiated when ANTHROPIC_API_KEY
    is present. The default graph path always uses MockSupervisor so no API key is
    required for tests or CI.

    Construction:
        model_name — Anthropic model string (default: env ANTHROPIC_MODEL or
                     "claude-haiku-4-5" as a cost-effective classifier).
        temperature — generation temperature (default 0.0 for determinism).

    Falls back to RECALL_HISTORY on any API error to avoid crashing the graph.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
    ) -> None:
        from langchain_anthropic import ChatAnthropic

        if model_name:
            resolved_model = model_name
        else:
            try:
                from mempill_showcase.config.settings import get_settings
                resolved_model = get_settings().anthropic_model
            except Exception:
                resolved_model = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5")
        self._llm = ChatAnthropic(model=resolved_model, temperature=temperature)
        self._model_name = resolved_model
        log.info("LLMSupervisor: initialised with model=%s", resolved_model)

    def classify(self, user_input: str) -> str:
        """Call the Anthropic model to classify *user_input* into an IntentLabel.

        Returns one of the five IntentLabel constants.
        Falls back to RECALL_HISTORY on any exception (safe default — read-only).
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        try:
            messages = [
                SystemMessage(content=_LLM_SYSTEM_PROMPT),
                HumanMessage(content=user_input),
            ]
            response = self._llm.invoke(messages)
            raw = response.content.strip().upper()
            # Normalise: take first word/token in case the model adds punctuation
            token = raw.split()[0].rstrip(".,;:") if raw else ""
            if token in _LLM_VALID_LABELS:
                log.debug("LLMSupervisor: classified %r → %s", user_input[:80], token)
                return token
            # Partial match fallback — model said e.g. "UPDATE" instead of "UPDATE_CONTACT"
            for label in _LLM_VALID_LABELS:
                if label.startswith(token) or token in label:
                    log.debug(
                        "LLMSupervisor: partial match %r → %s (raw=%r)",
                        user_input[:80], label, raw,
                    )
                    return label
            log.warning(
                "LLMSupervisor: unrecognised label %r from model; defaulting to RECALL_HISTORY",
                raw,
            )
            return IntentLabel.RECALL_HISTORY
        except Exception as exc:
            log.warning(
                "LLMSupervisor: API call failed (%s); defaulting to RECALL_HISTORY",
                exc,
            )
            return IntentLabel.RECALL_HISTORY


# ── LangGraph node ────────────────────────────────────────────────────────────

def make_supervisor_node(classifier: SupervisorClassifier):
    """Factory: returns a LangGraph node function bound to *classifier*.

    Usage:
      node_fn = make_supervisor_node(MockSupervisor())
      graph.add_node("supervisor", node_fn)

    The returned function updates: intent, route.
    """
    def supervisor_node(state: ExecAssistantState) -> dict:
        user_input = state.get("user_input", "")
        intent = classifier.classify(user_input)
        route = _INTENT_TO_ROUTE.get(intent, "crew_c")
        log.debug(
            "supervisor_node: input=%r → intent=%s route=%s",
            user_input[:80], intent, route,
        )
        return {"intent": intent, "route": route}

    supervisor_node.__name__ = "supervisor_node"
    return supervisor_node
