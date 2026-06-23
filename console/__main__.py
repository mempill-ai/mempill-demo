"""
console/__main__.py — mempill Interactive Console Agent entry point.

Usage:
  python -m console                     REPL (deterministic grammar)
  python -m console --llm               REPL with LLM extraction (requires ANTHROPIC_API_KEY)
  python -m console --scenario          auto-play 3-act story then REPL
  python -m console --selftest          run assertion suite (no API key) then exit
  python -m console --reset             delete the persistent DB and exit
  python -m console --db <path>         custom DB path
  python -m console --agent <agent_id>  custom agent_id
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m console",
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

    # ── --selftest: run assertions, exit ──────────────────────────────────────
    if args.selftest:
        from console.selftest import run as run_selftest
        run_selftest()
        sys.exit(0)

    # ── --llm: guard before anything else ─────────────────────────────────────
    if args.llm:
        from console.inference.llm import guard
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

    # ── Open the engine ───────────────────────────────────────────────────────
    import mempill

    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = mempill.open(str(db_path))

    # ── Build agent + parser ──────────────────────────────────────────────────
    from console.agent import MempillAwareAgent
    from console.inference.deterministic import DeterministicParser
    from console.inference.base import CommandKind

    agent = MempillAwareAgent(engine, agent_id=args.agent)

    if args.llm:
        from console.inference.llm import LLMParser
        cmd_parser = LLMParser()
    else:
        cmd_parser = DeterministicParser()

    det_parser = DeterministicParser()  # always available for slash commands

    # ── Show startup audit (last 5) ───────────────────────────────────────────
    _print_startup_audit(engine, args.agent)

    # ── --scenario: auto-play then fall into REPL ─────────────────────────────
    if args.scenario:
        from console.scenario import run_scenario
        run_scenario(agent, det_parser)

    # ── REPL ──────────────────────────────────────────────────────────────────
    _repl(agent, cmd_parser, det_parser, engine, args.agent, args.llm)


def _print_startup_audit(engine: object, agent_id: str) -> None:
    """Print last 5 audit entries on startup (persistence awareness)."""
    try:
        audit = engine.query_audit({
            "agent_id": agent_id,
            "claim_ref": None,
            "from_tx_time": None,
            "limit": 5,
        })
        entries = audit.get("entries", [])
        if entries:
            print()
            print(f"[Loaded DB: {len(entries)} recent audit entries]")
            for e in entries[-5:]:
                ref = (e.get("claim_ref") or e.get("id", ""))[:8]
                kind = e.get("event_kind", "?")
                disp = e.get("disposition", "?")
                ts = str(e.get("recorded_at", ""))[:10]
                print(f"  {ts}  {ref}  {kind}  [{disp}]")
    except Exception:
        pass


def _repl(
    agent,
    cmd_parser,
    det_parser,
    engine,
    agent_id: str,
    llm_mode: bool,
) -> None:
    from console.inference.base import CommandKind

    mode_label = "[LLM mode]" if llm_mode else "[deterministic]"
    print()
    print(f"mempill Console Agent  {mode_label}")
    print("Type /help for commands, /quit to exit.")
    print()

    while True:
        try:
            raw = input("mempill> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            continue

        # Slash commands always use deterministic parser
        if raw.startswith("/"):
            cmd = det_parser.parse(raw)
        else:
            cmd = cmd_parser.parse(raw)

        if cmd.kind == CommandKind.QUIT:
            print("Goodbye.")
            break

        if cmd.kind == CommandKind.RESET:
            print("[Use --reset flag on startup to delete the DB.]")
            continue

        result = agent.handle(cmd)
        print(result)
        print()

        # After each successful INGEST/RECALL, auto-render the panel
        if cmd.kind in (CommandKind.INGEST, CommandKind.RECALL,
                        CommandKind.RECONCILE, CommandKind.RECALL_REENTRY):
            _render_panel(engine, agent, agent_id)


def _render_panel(engine, agent, agent_id: str) -> None:
    """Render the rich memory panel after write/query operations."""
    try:
        audit = engine.query_audit({
            "agent_id": agent_id,
            "claim_ref": None,
            "from_tx_time": None,
            "limit": 50,
        })

        # Build current_beliefs by querying unique subject/predicate pairs from registry
        current_beliefs = []
        seen: set[tuple[str, str]] = set()
        for ref, meta in agent._registry.items():
            key = (meta.subject, meta.predicate)
            if key in seen:
                continue
            seen.add(key)
            try:
                q = engine.query_memory({
                    "agent_id": agent_id,
                    "subject": meta.subject,
                    "predicate": meta.predicate,
                })
                current_beliefs.append(q)
            except Exception:
                pass

        from console.panel import render_panel
        render_panel(
            audit_entries=audit.get("entries", []),
            n_ingests=agent.n_ingests,
            n_queries=agent.n_queries,
            n_contested=agent.n_contested,
            n_superseded=agent.n_superseded,
            session_registry=agent._registry,
            current_beliefs=current_beliefs,
        )
    except Exception as exc:
        # Panel is best-effort — never crash the REPL
        print(f"[panel error: {exc}]")


if __name__ == "__main__":
    main()
