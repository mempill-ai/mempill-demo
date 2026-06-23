"""mempill_langgraph.state — AgentState definition."""
from __future__ import annotations

from langgraph.graph import MessagesState


class AgentState(MessagesState):
    """
    Extends MessagesState with mempill-specific fields.

    MessagesState provides:
        messages: Annotated[list[BaseMessage], add_messages]
    The add_messages reducer appends (never replaces) on each node return.
    """
    memory_context: str   # formatted belief block; set by retrieve_memory, read by respond
    user_id: str          # stable cross-turn identity for MemoryStore namespacing
    agent_id: str         # mempill agent identity; passed through to the adapter
