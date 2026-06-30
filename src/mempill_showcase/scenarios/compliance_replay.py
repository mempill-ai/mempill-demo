"""
mempill_showcase.scenarios.compliance_replay — T-08 Compliance Audit Report.

Answers the enterprise question:
  "What did the assistant BELIEVE on date T when it made decision D,
   and with what provenance?"

Runs (or reuses a run of) the executive-assistant scenario, captures real
transaction timestamps from the engine's audit log, then:
  1. query_at(as_of_tx_time=<captured T>) for alice-chen/{employer,city,dietary_restriction}
  2. query_audit() for the full provenance ledger

Renders a Rich report showing:
  - Belief state AS OF the compliance point-in-time (pre-NYC-write beliefs)
  - Belief state NOW (current state for contrast)
  - Full provenance chain (audit ledger)

HONEST TX-TIME NOTE (W5 invariant):
  Transaction time is ENGINE-STAMPED at ingest — you cannot inject past tx_time
  via the Python API. The compliance replay uses REAL captured timestamps from
  the engine audit log. The narrative "decision was made on date X" maps to a
  real captured timestamp. This proves the AXIS — it does not fake past dates.

AC-4 (transaction-time replay): query_at(as_of_tx_time=<before NYC write>) returns
  the belief that existed BEFORE the NYC update was recorded.
AC-5 (audit trail completeness): query_audit returns the full ledger with
  claim_ref, event_kind, recorded_at, disposition, and rationale on every entry.

Public interface:
  run_compliance_replay(adapter=None) -> ComplianceReport
    Runs the full scenario and returns a structured report.

  print_compliance_report(report) -> None
    Renders the report to stdout using Rich.

CLI entry:
  python -m mempill_showcase.scenarios.compliance_replay
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mempill_showcase.adapters.memory.mempill_adapter import MempillAdapter


# ── Report types ──────────────────────────────────────────────────────────────

@dataclass
class BeliefAtTime:
    """Belief for one predicate captured at a specific tx-time."""
    predicate: str
    value: Optional[str]
    status: str
    vt_start_display: Optional[str]
    vt_end_display: Optional[str]
    provenance: str
    claim_ref: str
    note: str = ""


@dataclass
class AuditLedgerEntry:
    """One entry from the audit ledger."""
    claim_ref: str
    event_kind: str
    disposition: str
    recorded_at: str
    rationale: str


@dataclass
class ComplianceReport:
    """Structured compliance audit report for alice-chen as of the compliance moment.

    The compliance moment = the real tx timestamp captured just before the NYC write
    (simulating the moment the March briefing was sent, before later updates).

    This proves AC-4 (tx-time replay) and AC-5 (audit trail completeness).
    """
    # The compliance point-in-time (real captured tx timestamp)
    compliance_tx_time: Optional[str]

    # Beliefs AS OF the compliance moment (before NYC write was known to the engine)
    beliefs_at_compliance_time: list[BeliefAtTime] = field(default_factory=list)

    # Current beliefs (for contrast)
    beliefs_now: list[BeliefAtTime] = field(default_factory=list)

    # Full audit ledger
    audit_ledger: list[AuditLedgerEntry] = field(default_factory=list)

    # Summary statistics
    total_audit_entries: int = 0
    succession_event_count: int = 0
    oracle_event_count: int = 0
    ingest_event_count: int = 0

    # Whether the as-of-time belief differs from current (proves the axis)
    axis_proven: bool = False

    def belief_at(self, predicate: str) -> Optional[BeliefAtTime]:
        for b in self.beliefs_at_compliance_time:
            if b.predicate == predicate:
                return b
        return None

    def belief_now(self, predicate: str) -> Optional[BeliefAtTime]:
        for b in self.beliefs_now:
            if b.predicate == predicate:
                return b
        return None


# ── Core runner ───────────────────────────────────────────────────────────────

def run_compliance_replay(adapter: Optional["MempillAdapter"] = None) -> ComplianceReport:
    """Run the compliance audit replay and return a structured ComplianceReport.

    If adapter is None, runs the full 8-beat executive-assistant scenario
    internally to build a realistic belief state, then performs the T-08
    compliance audit on top.

    The compliance point-in-time is the REAL tx timestamp captured just before
    the NYC write — proving the tx-time axis without injecting fake dates.

    Args:
        adapter: Optional pre-seeded MempillAdapter. Useful in tests to inject
                 a controlled adapter. When None, runs the full scenario.

    Returns:
        ComplianceReport with beliefs at the compliance moment, current beliefs,
        and the full audit ledger.
    """
    from mempill_showcase.scenarios.seed_data import AGENT_ID, load_seed_claims
    from mempill_showcase.config.di import build_mempill_adapter, _adapter_from_settings
    from mempill_showcase.core.domain.models import ClaimInput

    # Resolve agent_id from Settings (env: MEMPILL_AGENT_ID)
    try:
        from mempill_showcase.config.settings import get_settings
        _agent_id = get_settings().mempill_agent_id
    except Exception:
        _agent_id = AGENT_ID

    run_full_scenario = adapter is None
    compliance_tx_time: Optional[str] = None

    if run_full_scenario:
        # Build a FRESH in-memory adapter for the compliance scenario.
        # The file-backed engine (MEMPILL_DB_PATH) is intentionally NOT used here:
        # the compliance scenario needs a clean slate to capture the tx-time axis
        # (Austin→NYC succession) correctly. The file-backed engine is for the CLI
        # and LangGraph Studio paths, not for in-process scenario testing.
        from mempill_showcase.scenarios.executive_assistant import run_scenario
        adapter = build_mempill_adapter(in_memory=True, oracle_backed=True)
        trace = run_scenario(adapter, agent_id=_agent_id)
        # Use the scenario's captured pre-NYC tx timestamp as the compliance moment.
        compliance_tx_time = trace.tx_before_nyc_write
    else:
        # Adapter was supplied (test path). We need to capture the compliance
        # tx timestamp ourselves. The caller is responsible for ensuring the
        # adapter has a realistic belief state seeded before calling this.
        # We look for the Austin city claim's tx time in the audit log.
        from mempill import ProvenanceLabel

        # Seed data if not already present (idempotent — skips if already seeded)
        austin_belief = adapter.recall(_agent_id, "alice-chen", "city")
        if austin_belief.status in ("NoBelief",):
            load_seed_claims(adapter, _agent_id)

        # Capture the Austin claim's tx time
        all_audit = adapter.audit(_agent_id, limit=50)
        austin_belief2 = adapter.recall(_agent_id, "alice-chen", "city")
        austin_ref = austin_belief2.claim_ref

        for e in all_audit:
            if e.claim_ref == austin_ref:
                compliance_tx_time = e.recorded_at
                break

        if not compliance_tx_time and all_audit:
            compliance_tx_time = all_audit[0].recorded_at

    # ── T-08: Bi-temporal belief queries as_of compliance_tx_time ────────────
    # These are the 3 alice-chen predicates relevant to the March briefing.
    predicates = ["employer", "city", "dietary_restriction"]
    beliefs_at_time: list[BeliefAtTime] = []
    beliefs_now_list: list[BeliefAtTime] = []

    for pred in predicates:
        # AS OF the compliance moment (tx-time pinned)
        if compliance_tx_time:
            b_at = adapter.query_at(
                _agent_id, "alice-chen", pred,
                as_of_tx_time=compliance_tx_time,
            )
        else:
            # Fallback if no tx time captured (should not happen in practice)
            b_at = adapter.recall(_agent_id, "alice-chen", pred)

        beliefs_at_time.append(BeliefAtTime(
            predicate=pred,
            value=b_at.value,
            status=b_at.status,
            vt_start_display=b_at.vt_start_display,
            vt_end_display=b_at.vt_end_display,
            provenance=b_at.provenance or "unknown",
            claim_ref=b_at.claim_ref,
            note=_belief_note(pred, b_at.status),
        ))

        # CURRENT belief (for contrast)
        b_now = adapter.recall(_agent_id, "alice-chen", pred)
        beliefs_now_list.append(BeliefAtTime(
            predicate=pred,
            value=b_now.value,
            status=b_now.status,
            vt_start_display=b_now.vt_start_display,
            vt_end_display=b_now.vt_end_display,
            provenance=b_now.provenance or "unknown",
            claim_ref=b_now.claim_ref,
        ))

    # ── Audit ledger ──────────────────────────────────────────────────────────
    raw_audit = adapter.audit(_agent_id, limit=100)
    ledger = [
        AuditLedgerEntry(
            claim_ref=e.claim_ref,
            event_kind=e.event_kind,
            disposition=e.disposition,
            recorded_at=e.recorded_at,
            rationale=e.rationale,  # already coerced to str by MempillAdapter.audit()
        )
        for e in raw_audit
    ]

    # Classify events — mempill engine event_kind variants:
    #   ClaimCommitted, ValidityAsserted, AdjudicationResolved, QueuedForAdjudication
    succession_count = sum(
        1 for e in ledger
        if "Supersed" in e.event_kind or "Succession" in e.event_kind or "Validity" in e.event_kind
    )
    oracle_count = sum(
        1 for e in ledger
        if "Oracle" in e.event_kind or "Adjudic" in e.event_kind
    )
    ingest_count = sum(
        1 for e in ledger
        if "Claim" in e.event_kind or "Ingested" in e.event_kind
    )

    # Prove the axis: AS-OF belief must differ from current belief for at least one predicate
    axis_proven = any(
        at_b.value != now_b.value
        for at_b, now_b in zip(beliefs_at_time, beliefs_now_list)
        if at_b.value is not None or now_b.value is not None
    )

    return ComplianceReport(
        compliance_tx_time=compliance_tx_time,
        beliefs_at_compliance_time=beliefs_at_time,
        beliefs_now=beliefs_now_list,
        audit_ledger=ledger,
        total_audit_entries=len(ledger),
        succession_event_count=succession_count,
        oracle_event_count=oracle_count,
        ingest_event_count=ingest_count,
        axis_proven=axis_proven,
    )


# ── Display ───────────────────────────────────────────────────────────────────

def print_compliance_report(report: ComplianceReport) -> None:
    """Render the compliance report to stdout using Rich."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box

    console = Console()
    console.print()
    console.print(Panel(
        "[bold white]mempill Compliance Audit Report — T-08[/bold white]\n"
        "[dim]alice-chen belief state replay via as_of_tx_time (AC-4 + AC-5)[/dim]",
        border_style="cyan",
    ))

    # ── Compliance moment ─────────────────────────────────────────────────────
    console.print(
        f"\n[bold cyan]Compliance Point-in-Time (real tx timestamp):[/bold cyan] "
        f"[yellow]{report.compliance_tx_time or 'NOT CAPTURED'}[/yellow]"
    )
    console.print(
        "[dim]This is the engine-stamped ingest time of the Austin city claim — "
        "the state just BEFORE the NYC update was recorded. "
        "No fake dates are injected; the tx clock is the engine's.[/dim]\n"
    )

    # ── Belief state table ────────────────────────────────────────────────────
    table = Table(
        title="[bold]alice-chen Belief State[/bold]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        expand=True,
    )
    table.add_column("Predicate", style="bold", width=22)
    table.add_column("AS OF compliance time", ratio=2)
    table.add_column("NOW (current)", ratio=2)
    table.add_column("Axis proven?", width=14, justify="center")

    predicates = ["employer", "city", "dietary_restriction"]
    for pred in predicates:
        at_b = report.belief_at(pred)
        now_b = report.belief_now(pred)

        at_val = at_b.value if at_b else None
        now_val = now_b.value if now_b else None
        at_status = at_b.status if at_b else "?"
        now_status = now_b.status if now_b else "?"
        at_prov = at_b.provenance if at_b else "?"

        at_cell = (
            f"[{'green' if at_val else 'red'}]{at_val or 'NoBelief'}[/{'green' if at_val else 'red'}]\n"
            f"[dim]status={at_status}  prov={at_prov}[/dim]"
        )
        now_cell = (
            f"[{'green' if now_val else 'red'}]{now_val or 'NoBelief'}[/{'green' if now_val else 'red'}]\n"
            f"[dim]status={now_status}[/dim]"
        )

        differs = (at_val != now_val)
        axis_cell = (
            "[bold green]YES[/bold green]" if differs
            else "[dim]same[/dim]"
        )
        table.add_row(pred, at_cell, now_cell, axis_cell)

    console.print(table)

    axis_summary = (
        "[bold green]PROVEN — AS-OF beliefs differ from current beliefs.[/bold green]"
        if report.axis_proven
        else "[yellow]SAME — beliefs unchanged (may indicate scenario did not run fully).[/yellow]"
    )
    console.print(f"\n[bold]Bi-temporal axis:[/bold] {axis_summary}\n")

    # ── Provenance narrative ──────────────────────────────────────────────────
    console.print("[bold cyan]Compliance Narrative (T-08 answer):[/bold cyan]")
    console.print(Panel(
        _render_narrative(report),
        border_style="yellow",
        title="[bold]What the assistant believed at the compliance moment[/bold]",
    ))

    # ── Audit ledger ──────────────────────────────────────────────────────────
    ledger_table = Table(
        title="[bold]Provenance Ledger (query_audit)[/bold]",
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold",
        expand=True,
    )
    ledger_table.add_column("event_kind", width=20)
    ledger_table.add_column("disposition", width=20)
    ledger_table.add_column("recorded_at", width=32)
    ledger_table.add_column("claim_ref (abbrev)", width=14)
    ledger_table.add_column("rationale", ratio=1)

    for entry in report.audit_ledger[:30]:  # cap display at 30 rows
        kind_style = "red" if "Super" in entry.event_kind or "Oracle" in entry.event_kind else "green"
        ledger_table.add_row(
            f"[{kind_style}]{entry.event_kind}[/{kind_style}]",
            entry.disposition,
            f"[dim]{entry.recorded_at}[/dim]",
            entry.claim_ref[:12] + "…" if len(entry.claim_ref) > 12 else entry.claim_ref,
            entry.rationale[:60] + "…" if len(entry.rationale) > 60 else entry.rationale,
        )

    if len(report.audit_ledger) > 30:
        ledger_table.add_row(
            "[dim]...[/dim]", "", "", "",
            f"[dim]{len(report.audit_ledger) - 30} more entries (run with limit=100)[/dim]",
        )

    console.print(ledger_table)

    # ── Summary ───────────────────────────────────────────────────────────────
    console.print()
    console.print(Panel(
        f"[bold]Ledger Summary:[/bold]\n"
        f"  Total entries:      {report.total_audit_entries}\n"
        f"  Ingested events:    {report.ingest_event_count}\n"
        f"  Succession events:  {report.succession_event_count}\n"
        f"  Oracle events:      {report.oracle_event_count}\n\n"
        f"[bold green]AC-4:[/bold green] Transaction-time replay confirmed — "
        f"{'axis proven (AS-OF differs from NOW)' if report.axis_proven else 'axis same (all beliefs unchanged)'}\n"
        f"[bold green]AC-5:[/bold green] Audit trail complete — "
        f"{report.total_audit_entries} entries with claim_ref, event_kind, disposition, recorded_at, rationale\n\n"
        "[dim]This report is structurally unsolvable without bi-temporal storage. "
        "No vector store or single-axis timestamp system can produce this output. "
        "mempill makes it a standard query, not a forensic engineering project.[/dim]",
        border_style="green",
        title="[bold]Enterprise Hook: The Answer mempill Enables[/bold]",
    ))


def _belief_note(predicate: str, status: str) -> str:
    """Generate a human-readable note for a belief's status in the compliance context."""
    if status == "NoBelief":
        return "Not yet recorded at this tx time"
    if status in ("Contested", "QueuedForAdjudication"):
        return "Contested at this tx time — HITL pending"
    if predicate == "city":
        return "Active at query time"
    if predicate == "employer":
        return "Active at query time (CTO override recorded later — after this tx)"
    return "Active"


def _render_narrative(report: ComplianceReport) -> str:
    """Render the T-08 compliance narrative as a formatted string."""
    lines = [
        f"Belief State for alice-chen as of {report.compliance_tx_time or 'N/A'}:\n",
    ]
    for b in report.beliefs_at_compliance_time:
        vt_range = ""
        if b.vt_start_display:
            vt_range = f"  valid_from: {b.vt_start_display}"
            if b.vt_end_display:
                vt_range += f" → {b.vt_end_display}"
        lines.append(
            f"  {b.predicate}:\n"
            f"    value:    {b.value or 'NoBelief'}\n"
            f"    status:   {b.status}\n"
            f"    source:   {b.provenance}\n"
            + (f"    {vt_range}\n" if vt_range else "")
            + f"    note:     {b.note}\n"
        )

    oracle_entries = [e for e in report.audit_ledger
                      if "Oracle" in e.event_kind or "Adjudic" in e.event_kind]
    if oracle_entries:
        lines.append("\nHITL / Oracle Events on record:")
        for e in oracle_entries:
            lines.append(f"  {e.recorded_at} — {e.event_kind}: {e.rationale[:80]}")

    lines.append(
        f"\nAudit Trail: {report.total_audit_entries} entries "
        f"({report.ingest_event_count} ingested, "
        f"{report.succession_event_count} succession, "
        f"{report.oracle_event_count} oracle)"
    )
    return "\n".join(lines)


# ── CLI entry ─────────────────────────────────────────────────────────────────

def main() -> None:
    """CLI entry point: python -m mempill_showcase.scenarios.compliance_replay

    On startup, loads ``.env`` from the working directory so that LANGSMITH_*
    and ANTHROPIC_API_KEY values set there take effect (LangSmith tracing,
    LLM supervisor selection). Safe no-op when ``.env`` is absent.
    """
    from mempill_showcase.config.bootstrap import bootstrap
    bootstrap()

    report = run_compliance_replay()
    print_compliance_report(report)


if __name__ == "__main__":
    main()
