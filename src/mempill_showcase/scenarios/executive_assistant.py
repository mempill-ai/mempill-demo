"""
mempill_showcase.scenarios.executive_assistant — 6-beat ReAct agent scenario.

Drives the Executive Assistant scenario through the free-form ReAct agent
(7 tools: recall_subject, recall_at, recall_as_of, remember_fact,
get_contested, request_adjudication, audit_trail) using real mempill
bi-temporal writes and a real Anthropic LLM call.

DESIGN DECISIONS:
  - Uses the ReAct agent (create_react_agent) via build_app_from_settings /
    build_graph_from_adapter with a MemorySaver thread config.
  - The 6 beats are driven by natural-language HumanMessage turns; the agent
    selects tools and forms answers autonomously.
  - The contested beat sends "Alice has actually been CTO of Acme since June 2023,
    not VP Engineering" which forces valid_from=2023-06 — a same-period overlap
    with the seeded VP Engineering (also 2023-06) → genuine Contested /
    QueuedForAdjudication → HITL interrupt.
  - HITL is handled via Command(resume='Affirm') on the paused thread.
  - TX-time capture: we grab the Austin city claim's recorded_at from the audit
    log for use in the compliance_replay (AC-4).

Beats:
  B-01  recall alice-chen facts free-form  (ask-anything)
  B-02  point-in-time question  (city in early 2024 → Austin TX)
  B-03  new fact write  (alice-chen/travel_preference)
  B-04  contested update  (CTO of Acme since June 2023) → HITL interrupt → Affirm
  B-05  confirm CTO resolution  (recall alice-chen/employer → CTO)
  B-06  compliance/audit query  (audit_trail)

Public interface:
  run_scenario(adapter, rag_store=None, agent_id=None, tx_separation_delay=0.0) -> ScenarioTrace
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from mempill_showcase.config.di import build_graph_from_adapter
from mempill_showcase.scenarios.seed_data import AGENT_ID, load_seed_claims

if TYPE_CHECKING:
    from mempill_showcase.adapters.memory.mempill_adapter import MempillAdapter

log = logging.getLogger(__name__)


def all_canonical_entities() -> frozenset:
    """Inline replacement for deleted canonical_keys.all_canonical_entities()."""
    return frozenset({"alice-chen", "bob-liu", "acme-corp", "jordan-park"})


# ── Beat result dataclass ─────────────────────────────────────────────────────


@dataclass
class BeatResult:
    """Result of a single scenario beat."""
    beat_id: str
    description: str
    mempill_op: str
    value: Optional[str]
    status: Optional[str]
    disposition: Optional[str]
    is_contested: bool = False
    graph_state: dict = field(default_factory=dict)
    tx_time_captured: Optional[str] = None
    extra: dict = field(default_factory=dict)


@dataclass
class ScenarioTrace:
    """Full trace of all 6 beats. Designed for test assertions and CLI display."""
    beats: list[BeatResult] = field(default_factory=list)
    # Captured tx timestamps for AC-4 verification
    tx_before_nyc_write: Optional[str] = None
    tx_after_nyc_write: Optional[str] = None
    # Audit entries from B-06
    audit_entries: list[dict] = field(default_factory=list)
    # Subjects written across all beats
    subjects_written: set[str] = field(default_factory=set)
    # RAG doc count (kept for compat; not used in ReAct path)
    rag_doc_count_after_t03: int = 0
    mempill_write_count_t03: int = 0

    def beat(self, beat_id: str) -> Optional[BeatResult]:
        for b in self.beats:
            if b.beat_id == beat_id:
                return b
        return None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _last_ai_text(result: dict) -> str:
    """Extract the last non-empty AI message text from a graph result dict."""
    msgs = result.get("messages", [])
    for msg in reversed(msgs):
        cls = msg.__class__.__name__
        if cls not in ("AIMessage", "AIMessageChunk"):
            continue
        content = msg.content
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            texts = [
                c.get("text", "") if isinstance(c, dict) else str(c)
                for c in content
            ]
            joined = " ".join(t for t in texts if t.strip())
            if joined.strip():
                return joined.strip()
    return ""


def _has_interrupt(result: dict) -> bool:
    return bool(result.get("__interrupt__", []))


def _capture_latest_tx_time(adapter: "MempillAdapter", agent_id: str) -> Optional[str]:
    """Capture the real recorded_at of the most recent audit entry."""
    try:
        entries = adapter.audit(agent_id, limit=1)
        if entries:
            return entries[0].recorded_at
    except Exception as exc:
        log.warning("_capture_latest_tx_time: audit failed: %s", exc)
    return None


# ── Main scenario runner ──────────────────────────────────────────────────────

def run_scenario(
    adapter: "MempillAdapter",
    rag_store: Any = None,
    tx_separation_delay: float = 0.0,
    agent_id: Optional[str] = None,
) -> ScenarioTrace:
    """Execute all 6 beats via the ReAct agent and return a ScenarioTrace.

    Args:
        adapter:              A freshly created MempillAdapter (caller must not pre-seed it).
        rag_store:            Ignored (ReAct agent path does not use a shared RAG store).
        tx_separation_delay:  Seconds to sleep between Austin seed and any NYC write to
                              guarantee tx_before_nyc < tx_after_nyc. Default 0.0.
        agent_id:             Explicit agent_id override; falls back to Settings or AGENT_ID.

    Returns:
        ScenarioTrace with per-beat BeatResult entries and captured tx timestamps.
    """
    if agent_id is None:
        try:
            from mempill_showcase.config.settings import get_settings
            agent_id = get_settings().mempill_agent_id
        except Exception:
            agent_id = AGENT_ID

    model_name: Optional[str] = None
    try:
        from mempill_showcase.config.settings import get_settings
        model_name = get_settings().anthropic_model
    except Exception:
        pass

    trace = ScenarioTrace()

    # Build a MemorySaver-backed app for this scenario run
    checkpointer = MemorySaver()
    app = build_graph_from_adapter(adapter, checkpointer=checkpointer, model_name=model_name)

    # ── SEED Day-0 data ───────────────────────────────────────────────────────
    seed_refs = load_seed_claims(adapter, agent_id)
    log.info("Seeded %d Day-0 claims (0 = already present)", len(seed_refs))
    trace.subjects_written.update({"alice-chen", "bob-liu", "acme-corp", "jordan-park"})

    # Capture Austin city claim's tx time for AC-4 / compliance_replay
    try:
        austin_belief = adapter.recall(agent_id, "alice-chen", "city")
        austin_ref = austin_belief.claim_ref
        all_seed_audit = adapter.audit(agent_id, limit=20)
        for aud in all_seed_audit:
            if aud.claim_ref == austin_ref:
                trace.tx_before_nyc_write = aud.recorded_at
                break
        if not trace.tx_before_nyc_write:
            trace.tx_before_nyc_write = _capture_latest_tx_time(adapter, agent_id)
    except Exception as exc:
        log.warning("Could not capture Austin tx time: %s", exc)

    if tx_separation_delay > 0.0:
        import time as _time
        _time.sleep(tx_separation_delay)

    # ── B-01: Free-form recall — ask-anything ─────────────────────────────────
    # Ask about Alice's dietary restriction (seeded: vegetarian)
    cfg_b01 = {"configurable": {"thread_id": f"scenario-b01-{id(adapter)}"}}
    result_b01 = app.invoke(
        {"messages": [HumanMessage(content=f"What is Alice Chen's dietary restriction? (agent_id: {agent_id})")]},
        config=cfg_b01,
    )
    answer_b01 = _last_ai_text(result_b01)
    log.info("B-01 answer: %r", answer_b01)

    trace.beats.append(BeatResult(
        beat_id="B-01",
        description="Ask-anything: Alice's dietary restriction (free-form recall)",
        mempill_op="recall_subject",
        value=answer_b01,
        status="Resolved" if "vegetarian" in answer_b01.lower() else "unknown",
        disposition=None,
        extra={"agent_answer": answer_b01},
    ))

    # ── B-02: Point-in-time question — Alice's city in early 2024 ─────────────
    # Alice's city: Austin TX valid_from=2023-06, valid_until=2025-02
    # → valid_at=2024-03-01 should return Austin TX
    cfg_b02 = {"configurable": {"thread_id": f"scenario-b02-{id(adapter)}"}}
    result_b02 = app.invoke(
        {"messages": [HumanMessage(
            content=f"What was Alice Chen's city in early 2024, say around March 2024? (agent_id: {agent_id})"
        )]},
        config=cfg_b02,
    )
    answer_b02 = _last_ai_text(result_b02)
    log.info("B-02 answer: %r", answer_b02)

    trace.beats.append(BeatResult(
        beat_id="B-02",
        description="Point-in-time: Alice's city in March 2024 (recall_at → Austin TX)",
        mempill_op="recall_at",
        value=answer_b02,
        status="Resolved" if "austin" in answer_b02.lower() else "unknown",
        disposition=None,
        extra={"agent_answer": answer_b02, "valid_at": "2024-03-01T00:00:00Z"},
    ))

    # ── B-03: New fact write — alice-chen/travel_preference ───────────────────
    # travel_preference has no incumbent claim → CommittedCheap (no HITL)
    cfg_b03 = {"configurable": {"thread_id": f"scenario-b03-{id(adapter)}"}}
    result_b03 = app.invoke(
        {"messages": [HumanMessage(
            content=(
                f"Please note that Alice Chen prefers business class travel as of 2025. "
                f"Store this in memory. (agent_id: {agent_id})"
            )
        )]},
        config=cfg_b03,
    )
    answer_b03 = _last_ai_text(result_b03)
    log.info("B-03 answer: %r", answer_b03)

    # Verify via direct adapter recall
    travel_belief = adapter.recall(agent_id, "alice-chen", "travel_preference")
    trace.subjects_written.add("alice-chen")

    trace.beats.append(BeatResult(
        beat_id="B-03",
        description="New fact write: alice-chen/travel_preference = business class (CommittedCheap)",
        mempill_op="remember_fact",
        value=travel_belief.value or answer_b03,
        status=travel_belief.status,
        disposition=None,
        extra={"agent_answer": answer_b03, "direct_recall_value": travel_belief.value},
    ))

    # ── B-04: Contested update → HITL interrupt → Affirm ─────────────────────
    # B-04 always uses a FRESH in-memory oracle-backed adapter so there is always
    # a clean VP Engineering incumbent regardless of what the persistent DB contains.
    # This guarantees the same-period CTO write is always a genuine contest.
    from mempill_showcase.config.di import build_mempill_adapter as _build_mempill_adapter
    from mempill_showcase.scenarios.seed_data import load_seed_claims as _load_seed
    hitl_adapter = _build_mempill_adapter(in_memory=True, oracle_backed=True)
    _load_seed(hitl_adapter, agent_id)
    hitl_checkpointer = MemorySaver()
    hitl_app = build_graph_from_adapter(
        hitl_adapter, checkpointer=hitl_checkpointer, model_name=model_name
    )

    cfg_b04 = {"configurable": {"thread_id": f"scenario-b04-{id(hitl_adapter)}"}}
    result_b04 = hitl_app.invoke(
        {"messages": [HumanMessage(
            content=(
                f"Alice has actually been CTO of Acme since June 2023, not VP Engineering — "
                f"June 2023 is the same start date she had as VP Engineering, so this is a "
                f"correction, not a new period. Please update her employer record. "
                f"(agent_id: {agent_id})"
            )
        )]},
        config=cfg_b04,
    )
    interrupted = _has_interrupt(result_b04)
    answer_b04_pre = _last_ai_text(result_b04)
    log.info("B-04 pre-HITL interrupted=%s answer=%r", interrupted, answer_b04_pre)

    if interrupted:
        # HITL truly fired — resume with Affirm and report honest resolution
        hitl_verdict = "Affirm"
        result_b04_resume = hitl_app.invoke(
            Command(resume=hitl_verdict),
            config=cfg_b04,
        )
        answer_b04_post = _last_ai_text(result_b04_resume)
        log.info("B-04 post-HITL answer: %r", answer_b04_post)

        employer_after_hitl = hitl_adapter.recall(agent_id, "alice-chen", "employer")
        trace.subjects_written.add("alice-chen")

        trace.beats.append(BeatResult(
            beat_id="B-04",
            description="Contested: Alice CTO since June 2023 → HITL interrupt → Affirm → CTO wins",
            mempill_op="contested+hitl",
            value=employer_after_hitl.value or answer_b04_post,
            status=employer_after_hitl.status,
            disposition=None,
            is_contested=True,
            graph_state={
                "interrupted": True,
                "hitl_verdict": hitl_verdict,
                "agent_answer_post_hitl": answer_b04_post,
            },
            extra={
                "hitl_triggered": True,
                "post_hitl_answer": answer_b04_post,
                "direct_recall_employer": employer_after_hitl.value,
                "direct_recall_status": employer_after_hitl.status,
            },
        ))
        log.info(
            "B-04: HITL triggered correctly. employer=%r status=%s",
            employer_after_hitl.value, employer_after_hitl.status,
        )
    else:
        # HITL did NOT fire — report this honestly; do not fake a verdict
        log.warning("B-04: HITL did NOT trigger. Agent answered without interrupting.")
        employer_current = hitl_adapter.recall(agent_id, "alice-chen", "employer")
        trace.subjects_written.add("alice-chen")

        trace.beats.append(BeatResult(
            beat_id="B-04",
            description="Contested: Alice CTO since June 2023 → HITL DID NOT TRIGGER (beat failed)",
            mempill_op="contested+hitl",
            value=employer_current.value or answer_b04_pre,
            status="HITL_NOT_TRIGGERED",
            disposition=None,
            is_contested=False,
            graph_state={
                "interrupted": False,
                "hitl_verdict": None,
                "agent_answer_pre_hitl": answer_b04_pre,
            },
            extra={
                "hitl_triggered": False,
                "agent_answer": answer_b04_pre,
                "direct_recall_employer": employer_current.value,
                "direct_recall_status": employer_current.status,
            },
        ))
        log.warning(
            "B-04: NO HITL. employer=%r status=%s agent_answer=%r",
            employer_current.value, employer_current.status, answer_b04_pre,
        )

    # ── B-05: Confirm CTO resolution ─────────────────────────────────────────
    # Follow-up recall from the hitl_adapter so it reflects the adjudication result.
    cfg_b05 = {"configurable": {"thread_id": f"scenario-b05-{id(hitl_adapter)}"}}
    result_b05 = hitl_app.invoke(
        {"messages": [HumanMessage(
            content=f"What role does Alice Chen currently hold at Acme? (agent_id: {agent_id})"
        )]},
        config=cfg_b05,
    )
    answer_b05 = _last_ai_text(result_b05)
    log.info("B-05 answer: %r", answer_b05)

    trace.beats.append(BeatResult(
        beat_id="B-05",
        description="Confirm HITL resolution: recall Alice's employer → CTO wins",
        mempill_op="recall_subject",
        value=answer_b05,
        status="Resolved" if "cto" in answer_b05.lower() else "unknown",
        disposition=None,
        extra={"agent_answer": answer_b05},
    ))

    # ── B-06: Compliance/audit query ─────────────────────────────────────────
    cfg_b06 = {"configurable": {"thread_id": f"scenario-b06-{id(adapter)}"}}
    result_b06 = app.invoke(
        {"messages": [HumanMessage(
            content=f"Show me the audit trail for agent {agent_id} — what memory write events have occurred?"
        )]},
        config=cfg_b06,
    )
    answer_b06 = _last_ai_text(result_b06)
    log.info("B-06 answer: %r", answer_b06[:200])

    # Also capture raw audit entries for the trace
    raw_audit = adapter.audit(agent_id, limit=100)
    audit_dicts = [
        {
            "claim_ref": e.claim_ref,
            "event_kind": e.event_kind,
            "disposition": e.disposition,
            "recorded_at": e.recorded_at,
            "rationale": e.rationale,
        }
        for e in raw_audit
    ]
    trace.audit_entries = audit_dicts

    trace.beats.append(BeatResult(
        beat_id="B-06",
        description="Compliance audit: audit_trail → write event history",
        mempill_op="audit_trail",
        value=str(len(audit_dicts)),
        status="audit_complete",
        disposition=None,
        extra={
            "agent_answer": answer_b06[:300],
            "raw_audit_count": len(audit_dicts),
        },
    ))
    log.info("B-06: audit_entries=%d", len(audit_dicts))

    return trace


# ── CLI entry ─────────────────────────────────────────────────────────────────

def _reset_db(db_path: str | None, console: object) -> None:
    """Delete db_path and any SQLite sidecar files (-wal, -shm) if db_path is set.

    Guards:
      - Only deletes db_path itself and the two well-known SQLite sidecar suffixes.
      - Never deletes a path outside the configured db_path's parent directory.
      - When db_path is None (in-memory store), logs a no-op message.
    """
    import pathlib

    _print = getattr(console, "print", print)

    if not db_path:
        _print("[dim]--reset-db: in-memory store — nothing to delete.[/dim]")
        return

    base = pathlib.Path(db_path).resolve()
    for suffix in ("", "-wal", "-shm"):
        target = pathlib.Path(str(base) + suffix)
        if target.exists():
            target.unlink()
            _print(f"[yellow]--reset-db: deleted {target}[/yellow]")
        else:
            _print(f"[dim]--reset-db: {target} not found, skipping.[/dim]")
    _print(f"[green]--reset-db: reset DB at {base}[/green]")


def main() -> None:
    """CLI entry point: mempill-showcase console command.

    Runs the 6-beat executive-assistant scenario through the ReAct agent
    and prints a summary. Requires ANTHROPIC_API_KEY in the environment or .env.

    Flags:
      --reset-db   Delete the file-backed DB (and -wal/-shm sidecars) before
                   running so the demo starts from a clean seed. For an in-memory
                   store this is a no-op.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="mempill Executive Assistant Scenario (6-beat ReAct agent)"
    )
    parser.add_argument(
        "--reset-db",
        action="store_true",
        default=False,
        help=(
            "Delete the file-backed DB (MEMPILL_DB_PATH) before running so the "
            "demo starts from a clean seed. For an in-memory store this is a no-op."
        ),
    )
    args = parser.parse_args()

    from mempill_showcase.config.bootstrap import bootstrap
    bootstrap()

    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich import box

    from mempill_showcase.config.di import _adapter_from_settings
    from mempill_showcase.config.settings import get_settings

    console = Console()
    _settings = get_settings()

    if args.reset_db:
        _reset_db(_settings.mempill_db_path, console)

    console.print()
    console.print(Panel(
        "[bold white]mempill Executive Assistant Scenario[/bold white]\n"
        "[dim]6-beat ReAct agent run (requires ANTHROPIC_API_KEY)[/dim]",
        border_style="blue",
    ))

    adapter = _adapter_from_settings(_settings)
    agent_id = _settings.mempill_agent_id
    trace = run_scenario(adapter, agent_id=agent_id)

    table = Table(
        title="[bold]Beat Results[/bold]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        expand=True,
    )
    table.add_column("Beat", width=6, style="bold")
    table.add_column("Description", ratio=2)
    table.add_column("Op", width=20)
    table.add_column("Value / Result", ratio=2)
    table.add_column("Status", width=14)

    for beat in trace.beats:
        value_str = beat.value or "-"
        if len(value_str) > 40:
            value_str = value_str[:37] + "..."
        status_style = "green" if beat.status in ("Resolved", "audit_complete") else "yellow"
        table.add_row(
            beat.beat_id,
            beat.description[:60] + "…" if len(beat.description) > 60 else beat.description,
            beat.mempill_op,
            value_str,
            f"[{status_style}]{beat.status or '-'}[/{status_style}]",
        )

    console.print(table)
    console.print(f"\n[dim]tx_before_nyc_write: {trace.tx_before_nyc_write}[/dim]")
    console.print(f"[dim]audit_entries: {len(trace.audit_entries)}[/dim]")

    # Key narrative output for verification — only report HITL resolution if it truly happened
    b04 = trace.beat("B-04")
    if b04:
        console.print()
        hitl_triggered = b04.extra.get("hitl_triggered", False)
        if hitl_triggered:
            console.print("[bold cyan]HITL resolution (real interrupt occurred):[/bold cyan]")
            console.print(f"  hitl_triggered: [bold green]True[/bold green]")
            console.print(f"  hitl_verdict: [bold green]{b04.graph_state.get('hitl_verdict')}[/bold green]")
            console.print(f"  employer after Affirm: [bold green]{b04.extra.get('direct_recall_employer')}[/bold green]")
            console.print(f"  employer status: [bold green]{b04.extra.get('direct_recall_status')}[/bold green]")
        else:
            console.print("[bold red]HITL NOT TRIGGERED (B-04 beat failed):[/bold red]")
            console.print("  [red]The agent did not call request_adjudication — no interrupt occurred.[/red]")
            console.print(f"  hitl_triggered: [bold red]False[/bold red]")
            console.print(f"  employer (unresolved): [yellow]{b04.extra.get('direct_recall_employer')}[/yellow]")
            console.print(f"  employer status: [yellow]{b04.extra.get('direct_recall_status')}[/yellow]")

    b05 = trace.beat("B-05")
    if b05:
        console.print()
        console.print("[bold cyan]Post-HITL recall (B-05):[/bold cyan]")
        cto_confirmed = "cto" in (b05.value or "").lower()
        color = "green" if cto_confirmed else "yellow"
        console.print(f"  [{color}]{b05.value or '(no answer)'}[/{color}]")

    console.print(
        "\n[bold green]Scenario complete.[/bold green] "
        "Run [cyan]mempill-showcase-compare[/cyan] for naive-vs-mempill contrast, "
        "or [cyan]mempill-showcase-audit[/cyan] for the compliance replay.\n"
    )


# ── Utility kept for compliance_replay compat ─────────────────────────────────

def _find_value_at(entries: list[dict], valid_at_iso: str) -> Optional[str]:
    """Filter query_history entries to find which value was valid at *valid_at_iso*."""
    from datetime import datetime, timezone

    try:
        ts = datetime.fromisoformat(valid_at_iso.replace("Z", "+00:00"))
    except ValueError:
        return None

    for ent in entries:
        vf_str = ent.get("valid_from")
        vu_str = ent.get("valid_until")
        if not vf_str:
            continue
        try:
            vf = datetime.fromisoformat(vf_str.replace("Z", "+00:00"))
            vu = datetime.fromisoformat(vu_str.replace("Z", "+00:00")) if vu_str else None
        except ValueError:
            continue
        if vf <= ts and (vu is None or vu > ts):
            return ent.get("value")
    return None


if __name__ == "__main__":
    main()
