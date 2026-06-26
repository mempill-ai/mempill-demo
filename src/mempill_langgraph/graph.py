"""mempill_langgraph.graph — build_graph factory.

IMPORT BOUNDARY: this module MUST NOT import mempill directly.
It receives a MemoryStore instance via parameter from the composition root (__main__.py).
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore

from mempill_demo.ports.memory import MemoryStore
from mempill_langgraph.extraction import ClaimExtractResult, DecisionClassifyResult, KeyExtractResult
from mempill_langgraph.nodes import make_nodes
from mempill_langgraph.state import AgentState


def build_graph(
    memory_store: MemoryStore,
    llm: Any,
    checkpointer: Optional[Any] = None,
    extractor: Optional[Callable[[str], ClaimExtractResult]] = None,
    key_extractor: Optional[Callable[[str], KeyExtractResult]] = None,
    decision_classifier: Optional[Callable[[str], DecisionClassifyResult]] = None,
):
    """
    Construct and compile the mempill LangGraph conversational agent.

    Graph topology (unconditional edges):
        START -> retrieve_memory -> respond -> write_memory -> END

    Args:
        memory_store: MemoryStore Protocol implementation (e.g. MempillMemoryStore).
                      Injected here — never imported from mempill directly.
        llm: Any LangChain-compatible chat model (ChatAnthropic, FakeMessagesListChatModel, etc.).
        checkpointer: Optional LangGraph checkpointer for multi-turn history.
                      Defaults to a new MemorySaver() if not provided.
        extractor: Optional callable (str -> ClaimExtractResult) to override the
                   default llm.with_structured_output binding for write_memory.
                   Used in offline tests.
        key_extractor: Optional callable (str -> KeyExtractResult) to override the
                       default LLM-backed canonical key extractor for retrieve_memory.
                       Uses the SAME canonical key convention as extractor.
                       Used in offline tests.
        decision_classifier: Optional callable (str -> DecisionClassifyResult) to
                             override the default LLM-backed decision classifier used
                             for conversational adjudication.  Used in offline tests.

    Returns:
        Compiled StateGraph ready for invocation.
    """
    retrieve_memory, respond, write_memory = make_nodes(
        memory_store=memory_store,
        llm=llm,
        extractor=extractor,
        key_extractor=key_extractor,
        decision_classifier=decision_classifier,
    )

    builder = StateGraph(AgentState)
    builder.add_node("retrieve_memory", retrieve_memory)
    builder.add_node("respond", respond)
    builder.add_node("write_memory", write_memory)

    builder.add_edge(START, "retrieve_memory")
    builder.add_edge("retrieve_memory", "respond")
    builder.add_edge("respond", "write_memory")
    builder.add_edge("write_memory", END)

    # Use an InMemoryStore for Pattern-A store injection (retrieve_memory accepts
    # store: BaseStore in its signature per Pattern-A; it uses the mempill adapter
    # via closure, not this LangGraph store).
    lg_store = InMemoryStore()

    # Default to MemorySaver for multi-turn history if no checkpointer supplied
    cp = checkpointer if checkpointer is not None else MemorySaver()

    return builder.compile(checkpointer=cp, store=lg_store)
