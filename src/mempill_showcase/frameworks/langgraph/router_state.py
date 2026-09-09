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

Multi-turn `messages` reducer (TASK-33 item 3): `messages` carries
`Annotated[list, _safe_add_messages]` — a hardened wrapper around LangGraph's
built-in `add_messages` reducer.

  WHY a reducer at all: without one, a plain `list[Any]` channel is
  last-write-wins — every `app.invoke(..., config={"thread_id": X})` call on
  an EXISTING thread fully OVERWRITES the checkpointed `messages` value with
  just the new call's input, discarding all prior turns before route_query
  (or any node) ever runs. Empirically this broke real multi-turn follow-ups
  end-to-end (e.g. "What about her city?" after "What is Alice Chen's dietary
  restriction?" — the agent lost the referent and asked the user to
  clarify who "her" was, instead of answering from turn 1's context).

  WHY NOT the plain `add_messages` reducer: `add_messages` raises
  `NotImplementedError`/`ValueError` on items it cannot coerce to a
  BaseMessage (e.g. a raw int, float, bool, None, or a dict missing
  role/content) — and that coercion runs at the CHANNEL-MERGE step, when the
  graph's raw `invoke()` input is applied, BEFORE route_query's own
  `_normalize_messages` hardening (TASK-33 item 1) ever gets a chance to run.
  Using the bare `add_messages` reducer here would therefore REINTRODUCE the
  exact Studio int-crash regression item 1 fixes — just one step earlier.

  `_safe_add_messages` closes that gap: it first tries the real
  `add_messages`; on failure it re-attempts item-by-item, silently dropping
  (with a warning) whichever items don't coerce, and merges the rest. This
  keeps genuine cross-turn history (route_query's `_normalize_messages` then
  applies its own stricter "usable content" business rule on top, e.g.
  dropping a coercible-but-blank `""`) while remaining immune to the same
  primitive-junk crash class as the router's input hardening.
"""
from __future__ import annotations

import logging
from typing import Annotated, Any, Literal
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages

log = logging.getLogger(__name__)


def _safe_add_messages(left: list[Any], right: Any) -> list[Any]:
    """Hardened wrapper around `langgraph.graph.message.add_messages`.

    Delegates to the real `add_messages` reducer (dedup/merge-by-id,
    chronological accumulation across turns). On a coercion failure (a raw
    primitive or malformed dict that `add_messages` cannot turn into a
    BaseMessage), filters the offending item(s) out — logging a warning with
    ONLY counts/types, never raw item values (they may carry user content /
    PII) — and retries, so a single bad item degrades gracefully instead of
    crashing the channel merge for the whole thread.

    Filters BOTH *left* (the existing accumulated history) and *right* (this
    update), not just *right*: a checkpoint written before this reducer
    existed — or corrupted by any other path — could already carry
    non-coercible junk in *left*. Filtering only *right* would leave
    `add_messages(left, safe_right)` raising on that pre-existing junk forever,
    permanently blocking resume of that thread.
    """
    try:
        return add_messages(left, right)
    except Exception as exc:
        safe_left, dropped_left = _filter_coercible(left)
        safe_right, dropped_right = _filter_coercible(right)
        log.warning(
            "_safe_add_messages: channel-merge coercion failed (%s) — "
            "dropped %d non-coercible item(s) (%d from existing history "
            "types=%s, %d from this update types=%s)",
            type(exc).__name__,
            len(dropped_left) + len(dropped_right),
            len(dropped_left), [type(d).__name__ for d in dropped_left],
            len(dropped_right), [type(d).__name__ for d in dropped_right],
        )
        return add_messages(safe_left, safe_right)


def _filter_coercible(items: Any) -> tuple[list[Any], list[Any]]:
    """Split *items* into (coercible, non_coercible) by probing each item
    through `add_messages` individually. Used by `_safe_add_messages` to
    recover from a whole-batch coercion failure on either side of the merge."""
    items_list = items if isinstance(items, list) else ([items] if items else [])
    safe: list[Any] = []
    dropped: list[Any] = []
    for item in items_list:
        try:
            add_messages([], [item])  # probe: does this item coerce alone?
            safe.append(item)
        except Exception:
            dropped.append(item)
    return safe, dropped


class RouterState(TypedDict, total=False):
    messages: Annotated[list[Any], _safe_add_messages]
    agent_id: str
    route: Literal["people_ops", "org_registry"]
    route_rationale: str


class RouterInputState(TypedDict, total=False):
    """Narrow input schema for the router graph — Studio input form shows
    only Messages (see module docstring). The reducer that actually governs
    channel-merge behavior lives on `RouterState.messages` (the graph's
    `state_schema`); LangGraph matches this input schema to that channel by
    key name, so no separate reducer annotation is needed here."""
    messages: list[Any]
