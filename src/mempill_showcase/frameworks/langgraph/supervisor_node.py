"""
mempill_showcase.frameworks.langgraph.supervisor_node — intent classification + routing.

Architecture:
  - Defines a `SupervisorClassifier` Protocol — any implementation works (mock, LLM, etc.).
  - `MockSupervisor` is a deterministic keyword/fixture classifier; NO API key, NO model call.
    It is the default implementation used in tests and the W3 graph.
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
"""
from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

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
