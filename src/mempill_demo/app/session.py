"""
mempill_demo.app.session — REPL loop and panel rendering.

run_repl() is the main entry-point for the interactive session.
"""
from __future__ import annotations

from mempill_demo.domain.agent import handle_command
from mempill_demo.domain.models import CommandKind, SessionStats
from mempill_demo.ports.inference import InferenceSource
from mempill_demo.ports.memory import MemoryStore
from mempill_demo.ports.presenter import Presenter


def run_repl(
    store: MemoryStore,
    presenter: Presenter,
    cmd_parser: InferenceSource,
    det_parser: InferenceSource,
    stats: SessionStats,
    llm_mode: bool = False,
) -> None:
    """Run the interactive REPL until /quit or EOF."""
    # Show startup audit (persistence awareness)
    _print_startup_audit(store, presenter)

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

        # /review — human-in-the-loop adjudication
        if raw.lower() == "/review":
            from mempill_demo.app.review import run_review
            run_review(store, presenter)
            continue

        # /sweep — expire timed-out adjudications
        if raw.lower() == "/sweep":
            _sweep(store)
            continue

        response = handle_command(cmd, store, stats)
        presenter.render(response)

        # After each successful write/query operation, auto-render the panel
        if cmd.kind in (CommandKind.INGEST, CommandKind.RECALL,
                        CommandKind.RECONCILE, CommandKind.RECALL_REENTRY):
            _render_panel(store, presenter, stats)


def _print_startup_audit(store: MemoryStore, presenter: Presenter) -> None:
    """Show the last 5 audit entries on startup for cross-session persistence awareness."""
    try:
        entries = store.audit(5)
        presenter.render_startup_audit(entries)
    except Exception:
        pass


def _sweep(store: MemoryStore) -> None:
    """Call sweep_expired_adjudications() on oracle-wired stores."""
    sweep_fn = getattr(store, "_sweep_expired", None)
    if sweep_fn is None:
        print("[/sweep] This store does not support oracle sweep.")
        return
    n = sweep_fn()
    if n:
        print(f"[/sweep] Swept {n} expired adjudication(s) back to Contested.")
    else:
        print("[/sweep] No expired adjudications found.")


def _render_panel(
    store: MemoryStore,
    presenter: Presenter,
    stats: SessionStats,
) -> None:
    """Render the rich memory panel — best-effort, never crashes the REPL."""
    try:
        audit_entries = store.audit(50)
        beliefs = store.beliefs()
        registry = store.registry_snapshot()
        presenter.render_panel(audit_entries, beliefs, registry, stats)
    except Exception as exc:
        print(f"[panel error: {exc}]")
