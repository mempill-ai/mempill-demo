"""
mempill_demo.domain.agent — pure R1-R6 command dispatch.

NO imports of mempill, rich, or anthropic. Receives domain types, returns
AgentResponse. The MemoryStore port is imported by Protocol reference only.
"""
from __future__ import annotations

from typing import Optional

from mempill_demo.domain.models import (
    AgentResponse,
    BeliefView,
    CommandKind,
    ParsedCommand,
    SessionStats,
    _conf_str,
    _prov_abbr,
)
from mempill_demo.ports.memory import MemoryStore


# ── Help text ─────────────────────────────────────────────────────────────────

_HELP_TEXT = """\
mempill Console Agent — command grammar:

  INGEST <subject> <predicate> "<value>" [SINCE <ISO>] [UNTIL <ISO>] [CONF <0-1>]
    Ingest a claim into the memory engine.

  RECALL <subject> <predicate>
    Recall the current belief for a subject/predicate pair.

  RECALL <subject> <predicate> valid=<YYYY-MM-DD>
    Point-in-time recall: which claim was valid at that date?
    (valid-time axis — ignores when beliefs were recorded)

  RECALL <subject> <predicate> tx=<YYYY-MM-DD>
    Historical recall: what did we believe as of that transaction date?
    (transaction-time axis — ignores what was valid then)

  RECALL <subject> <predicate> valid=<YYYY-MM-DD> tx=<YYYY-MM-DD>
    Bi-temporal recall: what was valid at <valid> as we knew it at <tx>?
    Use this to recover superseded beliefs before a reconcile.

  RECALL_REENTRY <subject> <predicate> "<value>" <source_claim_ref>
    Re-ingest a recalled value (tests the amplification firewall).

  /memory           — list all claims ingested this session
  /history <s> <p>  — full audit history for (subject, predicate)
  /why <s> <p>      — disposition timeline for (subject, predicate)
  /contested        — list all currently contested beliefs
  /audit [N]        — last N raw audit entries (default 10)
  /reconcile <s> <p>— run reconciliation for (subject, predicate)
  /help             — this help text
  /quit             — exit the REPL\
"""


# ── Pure dispatch entry-point ─────────────────────────────────────────────────

def handle_command(
    cmd: ParsedCommand,
    store: MemoryStore,
    stats: SessionStats,
) -> AgentResponse:
    """
    Pure dispatch: CommandKind → AgentResponse.
    Stats are updated by mutation (caller owns the SessionStats object).
    """
    if cmd.kind == CommandKind.INGEST:
        return _handle_ingest(cmd, store, stats)
    if cmd.kind == CommandKind.RECALL:
        return _handle_recall(cmd, store, stats)
    if cmd.kind == CommandKind.RECALL_REENTRY:
        return _handle_recall_reentry(cmd, store, stats)
    if cmd.kind == CommandKind.MEMORY:
        return _handle_memory(store)
    if cmd.kind == CommandKind.HISTORY:
        return _handle_history(cmd, store)
    if cmd.kind == CommandKind.WHY:
        return _handle_why(cmd, store)
    if cmd.kind == CommandKind.CONTESTED:
        return _handle_contested(store, stats)
    if cmd.kind == CommandKind.AUDIT:
        return _handle_audit(cmd, store)
    if cmd.kind == CommandKind.RECONCILE:
        return _handle_reconcile(cmd, store, stats)
    if cmd.kind == CommandKind.HELP:
        return AgentResponse(text=_HELP_TEXT, kind=cmd.kind)
    if cmd.kind == CommandKind.QUIT:
        return AgentResponse(text="Goodbye.", kind=cmd.kind)
    if cmd.kind == CommandKind.UNKNOWN:
        return AgentResponse(
            text=f"[error] {cmd.error or 'Unknown command. Type /help.'}",
            kind=cmd.kind,
        )
    return AgentResponse(
        text=f"[error] Unhandled command kind: {cmd.kind}",
        kind=cmd.kind,
    )


# ── INGEST ────────────────────────────────────────────────────────────────────

def _handle_ingest(
    cmd: ParsedCommand,
    store: MemoryStore,
    stats: SessionStats,
) -> AgentResponse:
    meta = store.ingest(cmd)
    stats.n_ingests += 1
    disp = meta.disposition
    ref = meta.claim_ref

    lines = [f"Ingested: {cmd.subject} {cmd.predicate} = \"{cmd.value}\""]
    lines.append(f"  disposition: {disp}")
    lines.append(f"  claim_ref:   {ref[:8]}...")

    n_contested = 0
    n_superseded = 0

    if disp in ("Contested", "PendingConflict"):
        n_contested += 1
        lines.append("  [!] Contested — use /reconcile to resolve.")
    elif disp in ("Superseded", "Invalidated"):
        n_superseded += 1
        lines.append("  [note] Claim was immediately superseded by an existing belief.")

    stats.n_contested += n_contested
    stats.n_superseded += n_superseded

    return AgentResponse(
        text="\n".join(lines),
        kind=cmd.kind,
        n_ingests_delta=1,
        n_contested_delta=n_contested,
        n_superseded_delta=n_superseded,
    )


# ── RECALL ────────────────────────────────────────────────────────────────────

def _handle_recall(
    cmd: ParsedCommand,
    store: MemoryStore,
    stats: SessionStats,
) -> AgentResponse:
    stats.n_queries += 1
    has_temporal = bool(cmd.valid_at or cmd.as_of_tx_time)

    if has_temporal:
        recall_at_fn = getattr(store, "recall_at", None)
        if recall_at_fn is None:
            return AgentResponse(
                text="[error] This store does not support point-in-time recall (valid=/tx=).",
                kind=cmd.kind,
            )
        belief = recall_at_fn(
            cmd.subject, cmd.predicate,
            valid_at=cmd.valid_at,
            as_of_tx_time=cmd.as_of_tx_time,
        )
        text = _apply_agent_rules_temporal(belief, cmd.subject, cmd.predicate, stats,
                                           valid_at=cmd.valid_at, as_of_tx_time=cmd.as_of_tx_time)
    else:
        belief = store.recall(cmd.subject, cmd.predicate)
        text = _apply_agent_rules(belief, cmd.subject, cmd.predicate, stats)

    return AgentResponse(text=text, kind=cmd.kind)


def _apply_agent_rules(
    belief: BeliefView,
    subject: str,
    predicate: str,
    stats: SessionStats,
) -> str:
    """Apply agent rules R1-R5 to a BeliefView."""
    status = belief.status

    # R5 — no belief at all (value is None + status UNKNOWN)
    if belief.value is None and status == "UNKNOWN":
        return (
            f"No memory for {subject} {predicate}.\n"
            "Use INGEST to add a claim."
        )

    value = belief.value
    conf_val = belief.conf
    vt_start = belief.vt_start
    vt_end = belief.vt_end or "open"
    ref = belief.claim_ref[:8] if belief.claim_ref else ""
    corroboration = belief.corroboration

    if status in ("Committed", "CommittedCheap", "CommittedInferred", "Reinstated", "Resolved"):
        # R1
        lines = [f"Memory: {subject} {predicate} = \"{value}\" (conf {_conf_str(conf_val)})"]
        if vt_start or vt_end != "open":
            lines.append(f"  valid: {vt_start or '?'} → {vt_end}")
        lines.append(f"  status: {status}  ref: {ref}...")
        if corroboration:
            lines.append(f"  corroboration_count: {corroboration}")
        return "\n".join(lines)

    if status in ("Contested", "PendingConflict"):
        # R2 — surface both values
        stats.n_contested += 1
        lines = [f"[CONTESTED] {subject} {predicate} has conflicting claims:"]
        lines.append(
            f"  Primary:  \"{value}\" conf={_conf_str(conf_val)}"
            f"  valid: {vt_start or '?'} → {vt_end}  ref={ref}..."
        )
        for alt in belief.alternatives:
            lines.append(
                f"  Conflict: \"{alt.value}\" conf={_conf_str(alt.conf)}"
                f"  valid: {alt.vt_start or '?'} → {alt.vt_end or 'open'}  ref={alt.claim_ref[:8] if alt.claim_ref else ''}..."
            )
        lines.append("  Use /reconcile to resolve. Agent will NOT guess.")
        return "\n".join(lines)

    if status in ("Superseded", "Invalidated"):
        # R3
        stats.n_superseded += 1
        lines = [f"Memory: {subject} {predicate} = \"{value}\" (conf {_conf_str(conf_val)})"]
        lines.append(f"  status: {status} — prior versions exist")
        if vt_start or vt_end != "open":
            lines.append(f"  valid: {vt_start or '?'} → {vt_end}")
        lines.append("  Use /history to see the full timeline.")
        return "\n".join(lines)

    # Default: show whatever the engine says
    lines = [f"Memory: {subject} {predicate} = \"{value}\"  status={status}"]
    if vt_start:
        lines.append(f"  valid: {vt_start} → {vt_end}")
    return "\n".join(lines)


def _apply_agent_rules_temporal(
    belief: BeliefView,
    subject: str,
    predicate: str,
    stats: SessionStats,
    valid_at: "Optional[str]" = None,
    as_of_tx_time: "Optional[str]" = None,
) -> str:
    """Format a point-in-time (bi-temporal) RECALL result for human readability.

    The header explains which axes were queried so the reader understands the context.
    NoBelief, Contested, and committed results all receive appropriate labels.
    """
    # Build context label
    parts: list[str] = []
    if valid_at:
        date_str = valid_at[:10]
        parts.append(f"valid at {date_str}")
    if as_of_tx_time:
        date_str = as_of_tx_time[:10]
        parts.append(f"as we believed at tx {date_str}")
    if not parts:
        parts.append("current")
    context = "(" + ", ".join(parts) + ")"

    status = belief.status
    value = belief.value

    if value is None or status in ("NoBelief", "UNKNOWN"):
        return (
            f"Belief {context}: NoBelief — no claim covers this point in time.\n"
            f"  subject={subject} predicate={predicate}\n"
            "  Tip: try a different date or check /history for the valid-time windows."
        )

    vt_start = belief.vt_start
    vt_end = belief.vt_end or "open"
    ref = belief.claim_ref[:8] if belief.claim_ref else ""
    conf_s = _conf_str(belief.conf) if belief.conf is not None else "N/A"

    if status in ("Committed", "CommittedCheap", "CommittedInferred", "Reinstated", "Resolved"):
        lines = [f"Belief {context}: \"{value}\" (conf {conf_s})"]
        lines.append(f"  subject={subject} predicate={predicate}")
        lines.append(f"  claim valid: {vt_start or '?'} → {vt_end}  ref={ref}...")
        lines.append(f"  status: {status}")
        return "\n".join(lines)

    if status in ("Contested", "PendingConflict"):
        lines = [f"[CONTESTED] {context}: {subject} {predicate} has conflicting claims:"]
        lines.append(f"  Primary:  \"{value}\" conf={conf_s}  valid: {vt_start or '?'} → {vt_end}  ref={ref}...")
        for alt in belief.alternatives:
            lines.append(
                f"  Conflict: \"{alt.value}\" conf={_conf_str(alt.conf)}"
                f"  valid: {alt.vt_start or '?'} → {alt.vt_end or 'open'}"
                f"  ref={alt.claim_ref[:8] if alt.claim_ref else ''}..."
            )
        return "\n".join(lines)

    if status in ("Superseded", "Invalidated"):
        lines = [f"Belief {context}: \"{value}\" (conf {conf_s})  [SUPERSEDED at this tx-time]"]
        lines.append(f"  subject={subject} predicate={predicate}")
        if vt_start or vt_end != "open":
            lines.append(f"  claim valid: {vt_start or '?'} → {vt_end}")
        lines.append("  Use /history to see the full timeline.")
        return "\n".join(lines)

    # Fallback
    lines = [f"Belief {context}: \"{value}\"  status={status}"]
    lines.append(f"  subject={subject} predicate={predicate}")
    return "\n".join(lines)


# ── RECALL_REENTRY ────────────────────────────────────────────────────────────

def _handle_recall_reentry(
    cmd: ParsedCommand,
    store: MemoryStore,
    stats: SessionStats,
) -> AgentResponse:
    """R4 — ingest as recall-re-entry then re-query to show firewall held."""
    meta = store.ingest(cmd)
    stats.n_ingests += 1
    disp = meta.disposition
    new_ref = meta.claim_ref[:8]

    # Re-query to show the current belief
    belief = store.recall(cmd.subject, cmd.predicate)
    corroboration = belief.corroboration

    lines = [
        f"RECALL_REENTRY: {cmd.subject} {cmd.predicate} = \"{cmd.value}\"",
        f"  re-entry disposition: {disp}  ref: {new_ref}...",
        f"  Belief unchanged (firewall held).  corroboration_count={corroboration}",
    ]
    return AgentResponse(text="\n".join(lines), kind=cmd.kind, n_ingests_delta=1)


# ── /memory ───────────────────────────────────────────────────────────────────

def _handle_memory(store: MemoryStore) -> AgentResponse:
    registry = store.registry_snapshot()
    if not registry:
        return AgentResponse(
            text="Memory: (empty this session) — no claims ingested yet.",
            kind=CommandKind.MEMORY,
        )

    lines = [f"Memory: {len(registry)} claim(s) ingested this session:"]
    seen_subj_pred: set[tuple[str, str]] = set()
    for ref, meta in registry.items():
        key = (meta.subject, meta.predicate)
        marker = ""
        if key not in seen_subj_pred:
            seen_subj_pred.add(key)
            try:
                belief = store.recall(meta.subject, meta.predicate)
                cur_val = belief.value
                cur_status = belief.status
                marker = f"  [current belief: \"{cur_val}\" {cur_status}]"
            except Exception:
                pass
        lines.append(
            f"  {ref[:8]}  {meta.subject} {meta.predicate} = \"{meta.value}\""
            f"  [{meta.disposition}]{marker}"
        )
    return AgentResponse(text="\n".join(lines), kind=CommandKind.MEMORY)


# ── /history ──────────────────────────────────────────────────────────────────

def _handle_history(cmd: ParsedCommand, store: MemoryStore) -> AgentResponse:
    """R6 — real audit entries + session registry for subject/predicate."""
    metas, audit_entries = store.history(cmd.subject, cmd.predicate)

    if not metas:
        return AgentResponse(
            text=(
                f"No history for {cmd.subject} {cmd.predicate} in this session.\n"
                "Claims ingested in previous sessions are tracked by claim_ref only."
            ),
            kind=cmd.kind,
        )

    refs_for_line = {m.claim_ref for m in metas}
    lines = [f"History: {cmd.subject} {cmd.predicate}"]
    lines.append(f"  {len(refs_for_line)} distinct claim(s), {len(audit_entries)} audit event(s):")

    # Group audit by claim_ref
    by_ref: dict[str, list] = {}
    for e in audit_entries:
        by_ref.setdefault(e.claim_ref, []).append(e)

    meta_by_ref = {m.claim_ref: m for m in metas}
    for ref in refs_for_line:
        meta = meta_by_ref[ref]
        lines.append(f"\n  claim {ref[:8]}  value=\"{meta.value}\"  prov={_prov_abbr(meta.provenance)}")
        for e in by_ref.get(ref, []):
            lines.append(f"    {e.recorded_at[:19]}  {e.event_kind}  → {e.disposition}")

    return AgentResponse(text="\n".join(lines), kind=cmd.kind)


# ── /why ─────────────────────────────────────────────────────────────────────

def _handle_why(cmd: ParsedCommand, store: MemoryStore) -> AgentResponse:
    """R6 — show disposition transitions from audit entries."""
    metas, audit_entries = store.history(cmd.subject, cmd.predicate)

    if not metas:
        return AgentResponse(
            text=f"No session history for {cmd.subject} {cmd.predicate}.",
            kind=cmd.kind,
        )

    meta_by_ref = {m.claim_ref: m for m in metas}
    lines = [f"Why: {cmd.subject} {cmd.predicate} — disposition timeline ({len(audit_entries)} events):"]
    for e in sorted(audit_entries, key=lambda x: x.recorded_at):
        ref_short = e.claim_ref[:8]
        meta = meta_by_ref.get(e.claim_ref)
        val = meta.value if meta else "?"
        lines.append(
            f"  {e.recorded_at[:19]}  {ref_short}  \"{val}\" "
            f" {e.event_kind} → {e.disposition}"
        )
    return AgentResponse(text="\n".join(lines), kind=cmd.kind)


# ── /contested ────────────────────────────────────────────────────────────────

def _handle_contested(store: MemoryStore, stats: SessionStats) -> AgentResponse:
    all_beliefs = store.beliefs()
    contested = [
        b for b in all_beliefs
        if b.status in ("Contested", "PendingConflict")
    ]
    if not contested:
        return AgentResponse(
            text="No contested beliefs found in the current session.",
            kind=CommandKind.CONTESTED,
        )

    lines = [f"Contested beliefs ({len(contested)}):"]
    for b in contested:
        lines.append(f"  {b.subject} {b.predicate}  — use /reconcile {b.subject} {b.predicate}")
    return AgentResponse(text="\n".join(lines), kind=CommandKind.CONTESTED)


# ── /audit ────────────────────────────────────────────────────────────────────

def _handle_audit(cmd: ParsedCommand, store: MemoryStore) -> AgentResponse:
    entries = store.audit(cmd.audit_limit)
    registry = store.registry_snapshot()
    if not entries:
        return AgentResponse(text="Audit: no entries yet.", kind=cmd.kind)

    lines = [f"Audit: last {len(entries)} entries:"]
    for e in entries:
        ref = e.claim_ref[:8]
        ts = e.recorded_at[:19]
        meta = registry.get(e.claim_ref)
        if meta:
            lines.append(
                f"  {ts}  {ref}  {meta.subject} {meta.predicate}=\"{meta.value}\""
                f"  {e.event_kind}→[{e.disposition}]"
            )
        else:
            lines.append(f"  {ts}  {ref}  {e.event_kind}→[{e.disposition}]")
    return AgentResponse(text="\n".join(lines), kind=cmd.kind)


# ── /reconcile ────────────────────────────────────────────────────────────────

def _handle_reconcile(
    cmd: ParsedCommand,
    store: MemoryStore,
    stats: SessionStats,
) -> AgentResponse:
    outcomes = store.reconcile(cmd.subject, cmd.predicate)
    if not outcomes:
        return AgentResponse(
            text=(
                f"Reconcile {cmd.subject} {cmd.predicate}: no explicit outcomes "
                "(claims may have already self-resolved)."
            ),
            kind=cmd.kind,
        )

    n_superseded = 0
    lines = [f"Reconcile {cmd.subject} {cmd.predicate}:"]
    for o in outcomes:
        if o.disposition in ("Superseded", "Invalidated"):
            n_superseded += 1
        lines.append(f"  {o.claim_ref[:8]}...  → {o.disposition}")
    lines.append("  Re-query with RECALL to see the current belief.")

    stats.n_superseded += n_superseded
    return AgentResponse(
        text="\n".join(lines),
        kind=cmd.kind,
        n_superseded_delta=n_superseded,
    )
