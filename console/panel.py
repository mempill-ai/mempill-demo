"""
console.panel — rich memory panel renderer.

Tables:
  "Current Beliefs":  Subject | Predicate | Value | Status | Valid From | Valid Until | Conf | Prov
  "Recent Ledger":    recorded_at | claim_ref[:8] | event_kind | disposition
Footer: "Session: N ingests | M queries | K contested | J superseded"

AUDIT SHAPE NOTE: engine.query_audit() entries contain:
  entry_id, agent_id, claim_ref, event_kind, disposition, rationale, recorded_at
  There are NO subject/predicate/value fields. We enrich with session registry data.

Uses Console(force_terminal=False) so output degrades gracefully in non-TTY.
"""
from __future__ import annotations

from typing import Any, Optional

try:
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text
    from rich import box
    _RICH = True
except ImportError:
    _RICH = False


# Status badge mapping (PLANNING.md §3)
_BADGE: dict[str, tuple[str, str]] = {
    "Committed":           ("COMMITTED",  "green"),
    "CommittedCheap":      ("COMMITTED",  "green"),
    "CommittedInferred":   ("COMMITTED",  "green"),
    "Reinstated":          ("COMMITTED",  "green"),
    "Resolved":            ("COMMITTED",  "green"),
    "Contested":           ("CONTESTED",  "yellow"),
    "PendingConflict":     ("CONTESTED",  "yellow"),
    "Superseded":          ("SUPERSEDED", "dim"),
    "Invalidated":         ("SUPERSEDED", "dim"),
    "Queued":              ("PENDING",    "cyan"),
    "QueuedForAdjudication": ("PENDING", "cyan"),
    "PendingReview":       ("PENDING",    "cyan"),
    "PendingLowConfidence": ("LOW-CONF",  "yellow"),
    "Quarantined":         ("REJECTED",   "red"),
    "Rejected":            ("REJECTED",   "red"),
}


def _prov_abbr(prov: Any) -> str:
    if isinstance(prov, dict):
        t = prov.get("type", "")
        k = prov.get("kind", "")
        if t == "External":
            return "USER" if k == "UserAsserted" else "EXT"
        if t == "RecallReEntry":
            return "RECALL"
        if t == "ModelDerived":
            return "LLM"
    if isinstance(prov, str):
        if "UserAsserted" in prov:
            return "USER"
        if "ExternalFirstHand" in prov or "External" in prov:
            return "EXT"
        if "RecallReEntry" in prov or "Recall" in prov:
            return "RECALL"
        if "ModelDerived" in prov or "Model" in prov:
            return "LLM"
    return str(prov)[:6] if prov else ""


def render_panel(
    audit_entries: list[dict],
    n_ingests: int,
    n_queries: int,
    n_contested: int,
    n_superseded: int,
    # Optional: session registry for enriched display
    session_registry: Optional[dict] = None,
    # Optional: list of current belief dicts from query_memory for unique subj/pred pairs
    current_beliefs: Optional[list[dict]] = None,
) -> None:
    """Render the two-table memory panel to stdout."""
    console = Console(force_terminal=False)

    if not _RICH:
        console.print("[Memory Panel — rich not available]")
        console.print(f"Session: {n_ingests} ingests | {n_queries} queries | "
                      f"{n_contested} contested | {n_superseded} superseded")
        return

    # ── Build "Current Beliefs" from session registry ─────────────────────────
    beliefs_table = Table(
        title="Current Beliefs",
        box=box.SIMPLE,
        show_lines=False,
        expand=False,
    )
    beliefs_table.add_column("Subject",    style="cyan",  no_wrap=True)
    beliefs_table.add_column("Predicate",  style="cyan",  no_wrap=True)
    beliefs_table.add_column("Value",      style="bold white")
    beliefs_table.add_column("Status",     no_wrap=True)
    beliefs_table.add_column("Valid From", style="dim")
    beliefs_table.add_column("Valid Until", style="dim")
    beliefs_table.add_column("Conf",       style="dim")
    beliefs_table.add_column("Prov",       style="dim")

    if current_beliefs:
        for b in current_beliefs:
            belief = b.get("belief", {})
            status = belief.get("status", "UNKNOWN")
            primary = belief.get("primary") or {}
            fact = primary.get("fact", {}) or {}
            subj = fact.get("subject", "?")
            pred = fact.get("predicate", "?")
            val  = str(fact.get("value", ""))
            conf_dict = primary.get("confidence", {}) or {}
            cv = conf_dict.get("value_confidence", "")
            cv_str = f"{float(cv):.2f}" if cv != "" else ""
            vt   = primary.get("valid_time") or {}
            vt_start = vt.get("start", "") if isinstance(vt, dict) else ""
            vt_end   = (vt.get("end") or "open") if isinstance(vt, dict) else "open"
            prov = _prov_abbr(primary.get("provenance"))

            label, color = _BADGE.get(status, (status[:8], "white"))
            beliefs_table.add_row(subj, pred, val, Text(label, style=color),
                                  vt_start, vt_end, cv_str, prov)
    elif session_registry:
        # Fallback: show registry entries
        seen: set[tuple[str, str]] = set()
        for ref, meta in session_registry.items():
            key = (meta.subject, meta.predicate)
            if key in seen:
                continue
            seen.add(key)
            disp = meta.disposition
            vt = meta.valid_time or {}
            vt_start = vt.get("start", "") if isinstance(vt, dict) else ""
            vt_end   = (vt.get("end") or "open") if isinstance(vt, dict) else "open"
            cv_str = f"{meta.conf:.2f}"
            prov = _prov_abbr(meta.provenance)
            label, color = _BADGE.get(disp, (disp[:8], "white"))
            beliefs_table.add_row(meta.subject, meta.predicate, meta.value,
                                  Text(label, style=color), vt_start, vt_end, cv_str, prov)
    else:
        beliefs_table.add_row("(no claims this session)", "", "", Text("", style="dim"), "", "", "", "")

    console.print(beliefs_table)

    # ── "Recent Ledger" — last 5 audit entries ────────────────────────────────
    recent = audit_entries[-5:] if len(audit_entries) > 5 else audit_entries

    ledger_table = Table(
        title="Recent Ledger",
        box=box.SIMPLE,
        show_lines=False,
        expand=False,
    )
    ledger_table.add_column("recorded_at",  style="dim",  no_wrap=True)
    ledger_table.add_column("ref[:8]",      style="dim",  no_wrap=True)
    ledger_table.add_column("event_kind",   style="cyan", no_wrap=True)
    ledger_table.add_column("disposition",  no_wrap=True)

    for e in recent:
        ref  = e.get("claim_ref", "")[:8]
        disp = e.get("disposition", "?")
        kind = e.get("event_kind", "?")
        ts   = str(e.get("recorded_at", ""))[:19]

        label, color = _BADGE.get(disp, (disp[:8], "white"))
        ledger_table.add_row(ts, ref, kind, Text(label, style=color))

    console.print(ledger_table)

    # ── Footer ────────────────────────────────────────────────────────────────
    console.print(
        f"Session: [bold]{n_ingests}[/bold] ingests | "
        f"[bold]{n_queries}[/bold] queries | "
        f"[yellow]{n_contested}[/yellow] contested | "
        f"[dim]{n_superseded}[/dim] superseded"
    )
