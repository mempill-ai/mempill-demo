"""
mempill_showcase.frameworks.langgraph.router_graph — dual-agent router graph.

Topology (ARCHITECTURE.md §2):
  START -> route_query -> (conditional) -> {people_ops_subgraph | org_registry_subgraph} -> END

`route_query` is a structured-output classifier node (RouteDecision) run against
a cheap/fast model (default claude-haiku-4-5, overridable via ANTHROPIC_MODEL).
Ambiguity policy: default-route to "people_ops" (approved design, ARCHITECTURE.md
§2) — jordan-park and his people are the higher-frequency query surface in the
existing demo script.

agent_id injection (state-injection, confirms ARCHITECTURE.md §2 C2):
`route_query` sets `agent_id` in the returned state dict based on the chosen
route ("people-ops-001" or "org-registry-001") BEFORE the subgraph runs. Each
subgraph's ReAct agent is the current create_react_agent topology, unchanged —
its system prompt states the correct default agent_id per-instance (see
graph.py:build_system_prompt), and the LLM continues to pass agent_id as an
ordinary tool-call argument exactly as today. No tool code changes.

NO resume-guard: SPIKE_RESULTS.md Smoke 3 confirmed (via an A/B control test
with the guard explicitly disabled) that LangGraph's native Pregel resume
mechanism resumes from the interrupted checkpoint at the task/subgraph level,
not from START, regardless of whether route_query has a resume-guard. Adding
one would be unnecessary dead code — intentionally omitted per SPIKE_RESULTS.md
recommendation.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Literal, Optional

from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from mempill_showcase.frameworks.langgraph.agents_config import (
    ORG_REGISTRY_SPEC,
    PEOPLE_OPS_SPEC,
)
from mempill_showcase.frameworks.langgraph.router_state import RouterState

log = logging.getLogger(__name__)

# ── Classifier output schema ──────────────────────────────────────────────────


class RouteDecision(BaseModel):
    """Structured-output schema for the route_query classifier node."""
    agent: Literal["people_ops", "org_registry"]
    rationale: str


# Ambiguity default: people_ops (ARCHITECTURE.md §2 — approved design).
_DEFAULT_ROUTE: Literal["people_ops", "org_registry"] = "people_ops"

_AGENT_ID_BY_ROUTE = {
    "people_ops": PEOPLE_OPS_SPEC.agent_id,
    "org_registry": ORG_REGISTRY_SPEC.agent_id,
}

_CLASSIFIER_PROMPT_TEMPLATE = """\
You are a routing classifier for a bi-temporal memory assistant split into two \
domain-specific agents. Decide which agent should handle the user's message.

- people_ops: handles PERSON facts — {people_known_entities}.
  Route here for questions/statements naming a PERSON (e.g. "Alice", "Bob",
  "Jordan") or their employer, city, dietary restriction, travel preference,
  preferred hotel, etc.
- org_registry: handles ORGANISATION/ROLE facts — {org_known_entities}.
  Route here for questions/statements naming an ORG or a leadership SEAT
  (e.g. "Acme", "CEO", "who leads Acme", "Acme's headquarters").

If the message is ambiguous or does not clearly name a person or an org/role,
default to people_ops.

User message: {user_message}
"""


def _last_human_text(messages: list[Any]) -> str:
    """Extract the text of the most recent message for the classifier prompt."""
    if not messages:
        return ""
    last = messages[-1]
    # Support both LangChain message objects (.content) and plain dicts.
    content = getattr(last, "content", None)
    if content is None and isinstance(last, dict):
        content = last.get("content")
    if content is None:
        content = str(last)
    return str(content)


def _build_classifier(model_name: Optional[str] = None):
    """Build the structured-output classifier model.

    Defaults to ANTHROPIC_MODEL env var or "claude-haiku-4-5", matching
    graph.py's model-selection convention.
    """
    if model_name is None:
        model_name = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5")
    model = ChatAnthropic(model=model_name)  # type: ignore[call-arg]
    return model.with_structured_output(RouteDecision)


def make_route_query_node(model_name: Optional[str] = None):
    """Build the route_query node function bound to a configured classifier.

    Returns a callable(state: RouterState) -> dict suitable for add_node.
    On any classifier failure (including missing API key), falls back to the
    default route rather than raising, so the graph degrades gracefully.
    """
    classifier = _build_classifier(model_name)

    def route_query(state: RouterState) -> dict:
        messages = state.get("messages", [])
        user_text = _last_human_text(messages)
        prompt = _CLASSIFIER_PROMPT_TEMPLATE.format(
            people_known_entities=PEOPLE_OPS_SPEC.known_entities,
            org_known_entities=ORG_REGISTRY_SPEC.known_entities,
            user_message=user_text,
        )

        try:
            decision = classifier.invoke(prompt)
            route = decision.agent
            rationale = decision.rationale
        except Exception as exc:
            log.warning(
                "route_query: classifier failed (%s) — defaulting to %r",
                exc,
                _DEFAULT_ROUTE,
            )
            route = _DEFAULT_ROUTE
            rationale = f"classifier error, defaulted: {exc}"

        agent_id = _AGENT_ID_BY_ROUTE.get(route, _AGENT_ID_BY_ROUTE[_DEFAULT_ROUTE])

        log.info(
            "route_query: route=%s agent_id=%s rationale=%s",
            route, agent_id, rationale,
        )

        return {
            "route": route,
            "route_rationale": rationale,
            "agent_id": agent_id,
        }

    return route_query


def _select_route(state: RouterState) -> str:
    """Conditional edge function: dispatch based on state['route'] set by route_query."""
    return state.get("route", _DEFAULT_ROUTE)


def build_router_graph(
    people_ops_subgraph: Any,
    org_registry_subgraph: Any,
    model_name: Optional[str] = None,
    checkpointer: Any = None,
) -> Any:
    """Build + compile the top-level dual-agent router graph.

    Topology: START -> route_query -> (conditional) -> {people_ops | org_registry} -> END.

    Args:
        people_ops_subgraph:   compiled ReAct subgraph for agent_id "people-ops-001".
        org_registry_subgraph: compiled ReAct subgraph for agent_id "org-registry-001".
        model_name:            Anthropic model string for the classifier. Defaults
                                to ANTHROPIC_MODEL env var or "claude-haiku-4-5".
        checkpointer:          Checkpointer instance or None (default). Pass None
                                for the LangGraph Studio path (Studio injects its
                                own persistence layer, mirroring build_graph's
                                checkpointer=None convention for Studio).

    Returns:
        A compiled StateGraph (CompiledGraph).
    """
    graph = StateGraph(RouterState)

    graph.add_node("route_query", make_route_query_node(model_name=model_name))
    graph.add_node("people_ops_subgraph", people_ops_subgraph)
    graph.add_node("org_registry_subgraph", org_registry_subgraph)

    graph.add_edge(START, "route_query")
    graph.add_conditional_edges(
        "route_query",
        _select_route,
        {
            "people_ops": "people_ops_subgraph",
            "org_registry": "org_registry_subgraph",
        },
    )
    graph.add_edge("people_ops_subgraph", END)
    graph.add_edge("org_registry_subgraph", END)

    compiled = graph.compile(checkpointer=checkpointer)

    log.info(
        "build_router_graph: compiled dual-agent router (model=%s, checkpointer=%s)",
        model_name or os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5"),
        type(checkpointer).__name__ if checkpointer else "None",
    )
    return compiled
