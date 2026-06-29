"""
mempill_demo.adapters.presenter_rich — Rich panel renderer implementing Presenter.

Tables:
  "Current Beliefs":  Subject | Predicate | Value | Status | Valid From | Valid Until | Conf | Prov
  "Recent Ledger":    recorded_at | claim_ref[:8] | event_kind | disposition
Footer: "Session: N ingests | M queries | K contested | J superseded"

Uses Console(force_terminal=False) so output degrades gracefully in non-TTY.
"""
from __future__ import annotations

try:
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text
    from rich import box
    _RICH = True
except ImportError:
    _RICH = False

from mempill_demo.domain.models import (
    AgentResponse,
    AuditEntry,
    BeliefView,
    ClaimMeta,
    SessionStats,
    _conf_str,
    _prov_abbr,
    _granularity_display,
)


# Status badge mapping
_BADGE: dict[str, tuple[str, str]] = {
    "Committed":              ("COMMITTED",  "green"),
    "CommittedCheap":         ("COMMITTED",  "green"),
    "CommittedInferred":      ("COMMITTED",  "green"),
    "Reinstated":             ("COMMITTED",  "green"),
    "Resolved":               ("COMMITTED",  "green"),
    "Contested":              ("CONTESTED",  "yellow"),
    "PendingConflict":        ("CONTESTED",  "yellow"),
    "Superseded":             ("SUPERSEDED", "dim"),
    "Invalidated":            ("SUPERSEDED", "dim"),
    "Queued":                 ("PENDING",    "cyan"),
    "QueuedForAdjudication":  ("PENDING",    "cyan"),
    "PendingReview":          ("PENDING",    "cyan"),
    "PendingLowConfidence":   ("LOW-CONF",   "yellow"),
    "Quarantined":            ("REJECTED",   "red"),
    "Rejected":               ("REJECTED",   "red"),
}


class RichPresenter:
    """Presenter adapter using the Rich library."""

    def __init__(self) -> None:
        self._console = Console(force_terminal=False)

    def render(self, response: AgentResponse) -> None:
        """Print the response text to stdout."""
        print(response.text)
        print()

    def render_panel(
        self,
        audit_entries: list[AuditEntry],
        beliefs: list[BeliefView],
        registry: dict[str, ClaimMeta],
        stats: SessionStats,
    ) -> None:
        """Render the two-table memory panel to stdout."""
        if not _RICH:
            self._console.print("[Memory Panel — rich not available]")
            self._console.print(
                f"Session: {stats.n_ingests} ingests | {stats.n_queries} queries | "
                f"{stats.n_contested} contested | {stats.n_superseded} superseded"
            )
            return

        # ── "Current Beliefs" table ───────────────────────────────────────────
        beliefs_table = Table(
            title="Current Beliefs",
            box=box.SIMPLE,
            show_lines=False,
            expand=False,
        )
        beliefs_table.add_column("Subject",     style="cyan",  no_wrap=True)
        beliefs_table.add_column("Predicate",   style="cyan",  no_wrap=True)
        beliefs_table.add_column("Value",       style="bold white")
        beliefs_table.add_column("Status",      no_wrap=True)
        beliefs_table.add_column("Valid From",  style="dim")
        beliefs_table.add_column("Valid Until", style="dim")
        beliefs_table.add_column("Conf",        style="dim")
        beliefs_table.add_column("Prov",        style="dim")

        if beliefs:
            for b in beliefs:
                status = b.status
                label, color = _BADGE.get(status, (status[:8], "white"))
                cv_str = _conf_str(b.conf) if b.conf is not None else ""
                # Use honest display strings when available (granularity-aware)
                vf_display = b.vt_start_display or (b.vt_start[:10] if b.vt_start else "")
                vu_display = b.vt_end_display or (b.vt_end[:10] if b.vt_end and b.vt_end != "open" else b.vt_end or "open")
                beliefs_table.add_row(
                    b.subject, b.predicate, str(b.value),
                    Text(label, style=color),
                    vf_display, vu_display, cv_str, b.provenance,
                )
        elif registry:
            # Fallback: show registry entries when beliefs not yet fetched
            seen: set[tuple[str, str]] = set()
            for ref, meta in registry.items():
                key = (meta.subject, meta.predicate)
                if key in seen:
                    continue
                seen.add(key)
                disp = meta.disposition
                vt = meta.valid_time or {}
                vt_start = vt.get("start", "") if isinstance(vt, dict) else ""
                vt_end = (vt.get("end") or "open") if isinstance(vt, dict) else "open"
                cv_str = f"{meta.conf:.2f}"
                prov = _prov_abbr(meta.provenance)
                label, color = _BADGE.get(disp, (disp[:8], "white"))
                beliefs_table.add_row(
                    meta.subject, meta.predicate, meta.value,
                    Text(label, style=color), vt_start, vt_end, cv_str, prov,
                )
        else:
            beliefs_table.add_row(
                "(no claims this session)", "", "", Text("", style="dim"), "", "", "", "",
            )

        self._console.print(beliefs_table)

        # ── "Recent Ledger" — last 5 audit entries ────────────────────────────
        recent = audit_entries[-5:] if len(audit_entries) > 5 else audit_entries
        ledger_table = Table(
            title="Recent Ledger",
            box=box.SIMPLE,
            show_lines=False,
            expand=False,
        )
        ledger_table.add_column("recorded_at", style="dim",  no_wrap=True)
        ledger_table.add_column("ref[:8]",     style="dim",  no_wrap=True)
        ledger_table.add_column("event_kind",  style="cyan", no_wrap=True)
        ledger_table.add_column("disposition", no_wrap=True)

        for e in recent:
            ref  = e.claim_ref[:8]
            ts   = e.recorded_at[:19]
            label, color = _BADGE.get(e.disposition, (e.disposition[:8], "white"))
            ledger_table.add_row(ts, ref, e.event_kind, Text(label, style=color))

        self._console.print(ledger_table)

        # ── Footer ────────────────────────────────────────────────────────────
        self._console.print(
            f"Session: [bold]{stats.n_ingests}[/bold] ingests | "
            f"[bold]{stats.n_queries}[/bold] queries | "
            f"[yellow]{stats.n_contested}[/yellow] contested | "
            f"[dim]{stats.n_superseded}[/dim] superseded"
        )

    def render_startup_audit(self, entries: list[AuditEntry]) -> None:
        """Print last audit entries on startup (persistence awareness)."""
        if not entries:
            return
        print()
        print(f"[Loaded DB: {len(entries)} recent audit entries]")
        for e in entries[-5:]:
            ref = e.claim_ref[:8]
            ts = e.recorded_at[:10]
            print(f"  {ts}  {ref}  {e.event_kind}  [{e.disposition}]")
