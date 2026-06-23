"""Interactive REPL: python -m mempill_langgraph

COMPOSITION ROOT — THE ONLY module in mempill_langgraph that imports mempill.
Constructs MempillMemoryStore + ChatAnthropic, builds the graph, runs the REPL.
"""
from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

load_dotenv()

if not os.environ.get("ANTHROPIC_API_KEY"):
    print("ERROR: ANTHROPIC_API_KEY not set. Add it to .env or export it.")
    sys.exit(1)

# All mempill imports after the API key guard
import mempill  # noqa: E402  (composition root — the only allowed mempill import)
from langchain_anthropic import ChatAnthropic  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402

from mempill_demo.adapters.memory_mempill import MempillMemoryStore  # noqa: E402
from mempill_langgraph.graph import build_graph  # noqa: E402

# ── Build graph ───────────────────────────────────────────────────────────────

_AGENT_ID = "langgraph-demo"
_USER_ID = "user-001"
_MODEL = os.environ.get("MEMPILL_MODEL", "claude-sonnet-4-6")

engine = mempill.Engine()
memory_store = MempillMemoryStore(engine=engine, agent_id=_AGENT_ID)
llm = ChatAnthropic(model=_MODEL, temperature=0.0, max_tokens=1024)
graph = build_graph(memory_store=memory_store, llm=llm)

# ── REPL ──────────────────────────────────────────────────────────────────────

thread_id = f"{_USER_ID}-session-1"
config = {"configurable": {"thread_id": thread_id}}

print()
print("mempill LangGraph Agent")
print(f"Model: {_MODEL}  |  User: {_USER_ID}  |  Thread: {thread_id}")
print("Type 'quit', 'exit', or 'q' to exit.")
print()

_first_turn = True

while True:
    try:
        user_input = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        break

    if not user_input:
        continue

    if user_input.lower() in ("quit", "exit", "q"):
        break

    invoke_payload: dict = {"messages": [HumanMessage(content=user_input)]}
    # Supply user_id and agent_id on the first turn only; checkpointer restores them thereafter.
    if _first_turn:
        invoke_payload["user_id"] = _USER_ID
        invoke_payload["agent_id"] = _AGENT_ID
        _first_turn = False

    try:
        result = graph.invoke(invoke_payload, config=config)
    except Exception as exc:
        print(f"\n[ERROR] Graph invocation failed: {exc}")
        continue

    last_ai = result["messages"][-1]
    print(f"\nAgent: {last_ai.content}\n")
