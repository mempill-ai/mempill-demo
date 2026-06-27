"""mempill_langgraph.state — AgentState definition."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import MessagesState


class PendingDecision(TypedDict, total=True):
    """
    Holds the state of a pending conversational adjudication.

    Stored as a TypedDict so LangGraph's MemorySaver checkpointer can serialize it
    via msgpack/JSON without a custom encoder.

    Set by retrieve_memory when a belief is still Contested after auto-reconcile
    AND a pending adjudication exists for the (subject, predicate).  Cleared
    (set to None) by retrieve_memory once the user's next-turn message is classified
    and the verdict is submitted (or the user said "neither").
    """
    subject: str
    predicate: str
    handle_id: str
    incumbent_value: str
    challenger_value: str


class ContestedInfo(TypedDict, total=False):
    """
    Structured snapshot of a contested belief for deterministic rendering.

    Set by retrieve_memory whenever belief.status is Contested or Conflict,
    regardless of whether a pending adjudication was correlated.  The respond
    node uses this to build a deterministic reply that lists all candidate
    values — the LLM is NOT invoked on any contested turn.

    Fields:
      subject    — belief subject (canonical key component)
      predicate  — belief predicate (canonical key component)
      candidates — list of candidate dicts, each with keys:
                     value, conf, vt_start, vt_end
    """
    subject: str
    predicate: str
    candidates: List[Dict[str, Any]]


class AgentState(MessagesState):
    """
    Extends MessagesState with mempill-specific fields.

    MessagesState provides:
        messages: Annotated[list[BaseMessage], add_messages]
    The add_messages reducer appends (never replaces) on each node return.
    """
    memory_context: str                         # formatted belief block; set by retrieve_memory, read by respond
    user_id: str                                # stable cross-turn identity for MemoryStore namespacing
    agent_id: str                               # mempill agent identity; passed through to the adapter
    pending_decision: Optional[PendingDecision] # set when awaiting user's pick between contested values
    contested: Optional[ContestedInfo]          # set on ANY contested turn; drives LLM bypass in respond
    _decision_turn: bool                        # True when the user's message was a decision answer (write guard)
    _resolved_reply: Optional[str]              # deterministic resolution text from queue-driven adjudication; respond uses it directly
    last_subject: Optional[str]                 # canonical subject from the most-recent resolved key (for follow-up turns)
    last_predicate: Optional[str]               # canonical predicate from the most-recent resolved key (for follow-up turns)
