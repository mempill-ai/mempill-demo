"""
mempill_showcase.frameworks.langgraph.router_state — RouterState TypedDict.

State for the top-level dual-agent router graph (ARCHITECTURE.md §2).  The
router graph dispatches to one of two compiled ReAct subgraphs
(people_ops_subgraph / org_registry_subgraph) based on `route_query`'s
classification.  `messages` and `agent_id` are shared keys with each
subgraph's own `ExecAssistantState` — LangGraph maps parent state to subgraph
state on entry by matching keys, so no adapter node is required.

Fields:
  messages         — shared HumanMessage/AIMessage list, passed through to
                      whichever subgraph is dispatched to.
  agent_id         — injected by route_query BEFORE dispatch (state-injection,
                      confirms ARCHITECTURE.md §2 C2): "people-ops-001" or
                      "org-registry-001". Existing tools take agent_id as an
                      ordinary runtime call parameter — zero tool changes.
  route            — the classifier's decision: "people_ops" | "org_registry".
                      Read by the conditional edge function to select the
                      next node.
  route_rationale  — free-text rationale from the classifier, logged/surfaced
                      for demo transparency (not used for control flow).

RouterInputState (TASK-31-W5): a narrower TypedDict exposing ONLY `messages`,
passed as StateGraph(..., input_schema=RouterInputState) in router_graph.py.
Without this, LangGraph Studio's input form for the compiled graph reflects
the FULL RouterState (messages + agent_id + route + route_rationale) as raw
JSON-array-of-fields input, rather than the friendly "+ Message" widget Studio
renders for a schema whose only field is `messages`. This caused a real user
error: Studio's raw-array input coerced a comma-separated value into an int
("2025") which then crashed LangChain message coercion. Restricting the
graph's declared input_schema to {messages} fixes the Studio UI without
touching internal state flow (agent_id/route/route_rationale are still set by
route_query as before; they are simply not part of the graph's INPUT contract).
"""
from __future__ import annotations

from typing import Any, Literal
from typing_extensions import TypedDict


class RouterState(TypedDict, total=False):
    messages: list[Any]
    agent_id: str
    route: Literal["people_ops", "org_registry"]
    route_rationale: str


class RouterInputState(TypedDict, total=False):
    """Narrow input schema for the router graph — Studio input form shows
    only Messages (see module docstring)."""
    messages: list[Any]
