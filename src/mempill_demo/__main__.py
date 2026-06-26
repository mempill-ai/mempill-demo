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
import logging
import os
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
    p.add_argument(
        "--verbose",
        action="count",
        default=0,
        help=(
            "Enable verbose logging of engine calls to stderr. "
            "Pass once for INFO (summaries); pass twice for DEBUG (raw payloads). "
            "Also honoured via env MEMPILL_VERBOSE=1 (INFO) or MEMPILL_VERBOSE=2 (DEBUG)."
        ),
    )
    return p


def _configure_logging(verbosity: int) -> None:
    """Configure logging based on verbosity level.

    verbosity=0 → silent (no basicConfig call; logging stays at WARNING default).
    verbosity=1 → INFO  (engine call summaries).
    verbosity=2+ → DEBUG (raw payloads as well).
    Also reads MEMPILL_VERBOSE env var (1 → INFO, 2 → DEBUG) if CLI count is 0.
    """
    if verbosity == 0:
        env_val = os.environ.get("MEMPILL_VERBOSE", "").strip().lower()
        if env_val in ("1", "true", "yes"):
            verbosity = 1
        elif env_val == "2":
            verbosity = 2

    if verbosity == 0:
        return  # remain silent — no logging config

    level = logging.DEBUG if verbosity >= 2 else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # ── --verbose / MEMPILL_VERBOSE: configure logging before anything else ──
    _configure_logging(args.verbose)

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

    def _remove_db(path: pathlib.Path) -> list[str]:
        """Delete a SQLite DB plus its -wal/-shm sidecars. Leftover WAL/SHM
        sidecars otherwise cause a 'disk I/O error' on the next open."""
        removed = []
        for p in (path, path.with_name(path.name + "-wal"),
                  path.with_name(path.name + "-shm")):
            if p.exists():
                p.unlink()
                removed.append(p.name)
        return removed

    if args.reset:
        removed = _remove_db(db_path)
        print(f"Deleted: {', '.join(removed)}" if removed
              else f"DB not found: {db_path} (nothing to delete)")
        sys.exit(0)

    # --scenario is an auto-play demonstration: start from a clean DB so it is
    # reproducible and never accumulates state (duplicate claims/pendings) across runs.
    if args.scenario:
        _remove_db(db_path)

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
