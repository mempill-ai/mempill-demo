"""mempill_langgraph.state — AgentState definition."""
from __future__ import annotations

from typing import Optional, TypedDict

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
    _decision_turn: bool                        # True when the user's message was a decision answer (write guard)
    last_subject: Optional[str]                 # canonical subject from the most-recent resolved key (for follow-up turns)
    last_predicate: Optional[str]               # canonical predicate from the most-recent resolved key (for follow-up turns)
