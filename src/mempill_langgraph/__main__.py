"""Interactive REPL: python -m mempill_langgraph

COMPOSITION ROOT — THE ONLY module in mempill_langgraph that imports mempill.
Constructs MempillMemoryStore + ChatAnthropic, builds the graph, runs the REPL.

Wave 3 additions (ARCHITECTURE §D):
  - Oracle-wired engine via mempill.open_oracle(db, HumanOracle())
  - /review REPL command: calls run_review(memory_store, presenter) before graph.invoke
  - /sweep REPL command: calls memory_store._sweep_expired() on demand
  - Conflict banner: stdout-only after each turn (NOT injected into LLM context)
  - Sweep on startup: quiet unless ≥1 adjudication was reverted
"""
from __future__ import annotations

import logging
import os
import pathlib
import sys

from dotenv import load_dotenv

load_dotenv()

# ── Verbose logging (MEMPILL_VERBOSE or --verbose arg) ───────────────────────
# Parse --verbose before the API-key guard so logging is active from the start.
# We do a minimal manual parse here to avoid full argparse setup at module level.
_verbosity: int = 0
_argv = sys.argv[1:]
_verbosity += _argv.count("--verbose") + _argv.count("-v")
if _verbosity == 0:
    _env_val = os.environ.get("MEMPILL_VERBOSE", "").strip().lower()
    if _env_val in ("1", "true", "yes"):
        _verbosity = 1
    elif _env_val == "2":
        _verbosity = 2

if _verbosity > 0:
    _log_level = logging.DEBUG if _verbosity >= 2 else logging.INFO
    logging.basicConfig(
        level=_log_level,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    # Enable LangChain call-level debug output when available
    try:
        from langchain.globals import set_debug  # type: ignore[import-untyped]
        set_debug(True)
    except ImportError:
        pass  # langchain.globals unavailable — degrade gracefully

if not os.environ.get("ANTHROPIC_API_KEY"):
    print("ERROR: ANTHROPIC_API_KEY not set. Add it to .env or export it.")
    sys.exit(1)

# All mempill imports after the API key guard
import mempill  # noqa: E402  (composition root — the only allowed mempill import)
from langchain_anthropic import ChatAnthropic  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402

from mempill_demo.adapters.human_oracle import HumanOracle  # noqa: E402
from mempill_demo.adapters.memory_mempill import MempillMemoryStore  # noqa: E402
from mempill_demo.adapters.presenter_rich import RichPresenter  # noqa: E402
from mempill_demo.app.review import run_review  # noqa: E402
from mempill_langgraph.graph import build_graph  # noqa: E402

# ── Build graph ───────────────────────────────────────────────────────────────

_AGENT_ID = "langgraph-demo"
_USER_ID = "user-001"
_MODEL = os.environ.get("MEMPILL_MODEL", "claude-sonnet-4-6")

_db_path = pathlib.Path(os.environ.get("MEMPILL_DB_PATH", ".mempill/langgraph.db"))
_db_path.parent.mkdir(parents=True, exist_ok=True)

# Oracle-wired engine (§D.1): conflicts route to QueuedForAdjudication
engine = mempill.open_oracle(str(_db_path), HumanOracle())
memory_store = MempillMemoryStore(engine=engine, agent_id=_AGENT_ID)
llm = ChatAnthropic(model=_MODEL, temperature=0.0, max_tokens=1024)
graph = build_graph(memory_store=memory_store, llm=llm)
presenter = RichPresenter()

# ── Startup sweep (§D — decision H.2) ────────────────────────────────────────
# Sweep expired adjudications back to Contested; print only if ≥1 reverted.
_swept = memory_store._sweep_expired()
if _swept:
    print(f"[startup] Swept {_swept} expired adjudication(s) back to Contested.")

# ── REPL ──────────────────────────────────────────────────────────────────────

thread_id = f"{_USER_ID}-session-1"
config = {"configurable": {"thread_id": thread_id}}

print()
print("mempill LangGraph Agent")
print(f"Model: {_MODEL}  |  User: {_USER_ID}  |  Thread: {thread_id}")
print("Type 'quit', 'exit', or 'q' to exit.  Type '/review' to resolve conflicts.")
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

    # ── /review command (§D.1 — REPL-level, not graph-level) ─────────────────
    if user_input.lower() == "/review":
        run_review(memory_store, presenter)
        continue

    # ── /sweep command (on-demand sweep) ─────────────────────────────────────
    if user_input.lower() == "/sweep":
        n = memory_store._sweep_expired()
        if n:
            print(f"[/sweep] Swept {n} expired adjudication(s) back to Contested.")
        else:
            print("[/sweep] No expired adjudications to sweep.")
        continue

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

    # ── Conflict banner (§D.4 — stdout only, NOT in LLM context) ─────────────
    # Check after the turn so the user sees it before typing the next message.
    _pending = memory_store.list_pending()
    if _pending:
        n = len(_pending)
        print(f"[!] {n} unresolved conflict(s) — type /review to resolve.")
