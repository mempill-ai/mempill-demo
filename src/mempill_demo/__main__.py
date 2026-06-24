"""
mempill_demo/__main__.py — composition root.

Usage:
  python -m mempill_demo                     REPL (deterministic grammar)
  python -m mempill_demo --llm               REPL with LLM extraction (requires ANTHROPIC_API_KEY)
  python -m mempill_demo --scenario          auto-play 3-act story then REPL
  python -m mempill_demo --selftest          run assertion suite (no API key) then exit
  python -m mempill_demo --reset             delete the persistent DB and exit
  python -m mempill_demo --db <path>         custom DB path
  python -m mempill_demo --agent <agent_id>  custom agent_id
"""
from __future__ import annotations

import argparse
import pathlib
import sys

from dotenv import load_dotenv

load_dotenv()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m mempill_demo",
        description="mempill Interactive Console Agent",
    )
    p.add_argument(
        "--db",
        default=".mempill/console.db",
        metavar="PATH",
        help="File-backed DB path (default: .mempill/console.db)",
    )
    p.add_argument(
        "--agent",
        default="console-user",
        metavar="AGENT_ID",
        help="Agent ID (default: console-user)",
    )
    p.add_argument(
        "--llm",
        action="store_true",
        help="Enable LLM-backed NL parsing (requires ANTHROPIC_API_KEY)",
    )
    p.add_argument(
        "--scenario",
        action="store_true",
        help="Auto-play the 3-act demonstration scenario then hand off to REPL",
    )
    p.add_argument(
        "--selftest",
        action="store_true",
        help="Run the deterministic assertion suite and exit (no API key required)",
    )
    p.add_argument(
        "--reset",
        action="store_true",
        help="Delete the persistent DB file and exit",
    )
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # ── --selftest: run assertions, exit ─────────────────────────────────────
    if args.selftest:
        # Add project root to sys.path so tests/ is importable.
        # __file__ is src/mempill_demo/__main__.py → go up 3 levels to repo root.
        project_root = pathlib.Path(__file__).resolve().parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from tests.test_integration import run as run_selftest
        run_selftest()
        sys.exit(0)

    # ── --llm: guard before anything else ────────────────────────────────────
    if args.llm:
        from mempill_demo.adapters.inference_llm import guard
        guard()  # exits with code 1 if key missing

    # ── Resolve DB path and optional --reset ──────────────────────────────────
    db_path = pathlib.Path(args.db)

    if args.reset:
        if db_path.exists():
            db_path.unlink()
            print(f"Deleted: {db_path}")
        else:
            print(f"DB not found: {db_path} (nothing to delete)")
        sys.exit(0)

    # ── Open the engine (oracle-wired for HITL) ───────────────────────────────
    import mempill
    from mempill_demo.adapters.human_oracle import HumanOracle

    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = mempill.open_oracle(str(db_path), HumanOracle())

    # ── Build adapters ────────────────────────────────────────────────────────
    from mempill_demo.adapters.memory_mempill import MempillMemoryStore
    from mempill_demo.adapters.inference_deterministic import DeterministicParser
    from mempill_demo.adapters.presenter_rich import RichPresenter
    from mempill_demo.domain.models import SessionStats

    store = MempillMemoryStore(engine, agent_id=args.agent)

    # ── Sweep expired adjudications on startup (decision H.2, quiet if none) ─
    swept = store._sweep_expired()
    if swept:
        print(f"[startup] Swept {swept} expired adjudication(s) back to Contested.")
    presenter = RichPresenter()
    det_parser = DeterministicParser()

    if args.llm:
        from mempill_demo.adapters.inference_llm import LLMParser
        cmd_parser = LLMParser()
    else:
        cmd_parser = det_parser

    stats = SessionStats()

    # ── --scenario: auto-play then fall into REPL ─────────────────────────────
    if args.scenario:
        from mempill_demo.app.scenario import run_scenario
        run_scenario(store, presenter)

    # ── REPL ──────────────────────────────────────────────────────────────────
    from mempill_demo.app.session import run_repl
    run_repl(
        store=store,
        presenter=presenter,
        cmd_parser=cmd_parser,
        det_parser=det_parser,
        stats=stats,
        llm_mode=args.llm,
    )


if __name__ == "__main__":
    main()
