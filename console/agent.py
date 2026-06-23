"""
console.agent — MempillAwareAgent: applies the 6 agent rules.

Rules (from PLANNING.md §4):
  R1 Resolved/Committed → "Memory: <subj> <pred> = <value>"
  R2 Contested → surface BOTH values with timeframes+conf+claim_ref; suggest /reconcile
  R3 Superseded → answer current value + "prior versions exist — /history"
  R4 After RECALL_REENTRY → show returned disposition, re-query, "Belief unchanged (firewall held)"
  R5 No primary/belief → "No memory for <subj> <pred>. Use INGEST."
  R6 /history & /why read query_audit (real entries), not invented

AUDIT SHAPE NOTE:
  engine.query_audit() returns entries with fields:
    entry_id, agent_id, claim_ref, event_kind, disposition, rationale, recorded_at
  There are NO subject/predicate/value fields in audit entries.
  We maintain a local claim registry (claim_ref → ingest metadata) for the session.
  For cross-session history, we use the claim_ref to correlate known refs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import mempill
from mempill import Disposition, ProvenanceLabel

from console.inference.base import CommandKind, ParsedCommand


# ── Helpers (mirrored from demo/temporal_validity.py) ─────────────────────────

def belief_value(belief_dict: dict) -> object:
    primary = belief_dict.get("belief", {}).get("primary")
    if primary:
        return primary.get("fact", {}).get("value")
    return None


def belief_status(belief_dict: dict) -> str:
    return belief_dict.get("belief", {}).get("status", "UNKNOWN")


def _conf_str(c: Any) -> str:
    if c is None:
        return "N/A"
    try:
        return f"{float(c):.2f}"
    except (TypeError, ValueError):
        return str(c)


# ── Claim registry ─────────────────────────────────────────────────────────────

@dataclass
class ClaimMeta:
    """Metadata recorded at ingest time — audit entries lack these fields."""
    subject: str
    predicate: str
    value: str
    provenance: dict
    valid_time: Optional[dict]
    conf: float
    disposition: str
    claim_ref: str


# ── Agent ─────────────────────────────────────────────────────────────────────

class MempillAwareAgent:
    """Executes parsed commands against the mempill engine and returns narrated output."""

    def __init__(self, engine: mempill.Engine, agent_id: str = "console-user") -> None:
        self._engine = engine
        self._agent_id = agent_id
        # Session-local claim registry: claim_ref → ClaimMeta
        self._registry: dict[str, ClaimMeta] = {}
        # Session stats
        self.n_ingests = 0
        self.n_queries = 0
        self.n_contested = 0
        self.n_superseded = 0

    # ── Public entry-point ────────────────────────────────────────────────────

    def handle(self, cmd: ParsedCommand) -> str:
        """Dispatch a ParsedCommand and return a human-readable response string."""
        if cmd.kind == CommandKind.INGEST:
            return self._handle_ingest(cmd)
        if cmd.kind == CommandKind.RECALL:
            return self._handle_recall(cmd)
        if cmd.kind == CommandKind.RECALL_REENTRY:
            return self._handle_recall_reentry(cmd)
        if cmd.kind == CommandKind.MEMORY:
            return self._handle_memory()
        if cmd.kind == CommandKind.HISTORY:
            return self._handle_history(cmd)
        if cmd.kind == CommandKind.WHY:
            return self._handle_why(cmd)
        if cmd.kind == CommandKind.CONTESTED:
            return self._handle_contested()
        if cmd.kind == CommandKind.AUDIT:
            return self._handle_audit(cmd)
        if cmd.kind == CommandKind.RECONCILE:
            return self._handle_reconcile(cmd)
        if cmd.kind == CommandKind.HELP:
            return _HELP_TEXT
        if cmd.kind == CommandKind.QUIT:
            return "Goodbye."
        if cmd.kind == CommandKind.UNKNOWN:
            return f"[error] {cmd.error or 'Unknown command. Type /help.'}"
        return f"[error] Unhandled command kind: {cmd.kind}"

    # ── INGEST ────────────────────────────────────────────────────────────────

    def _handle_ingest(self, cmd: ParsedCommand) -> str:
        # Build provenance — LLM path may carry extra["provenance_str"]
        prov_str = (cmd.extra or {}).get("provenance_str", "UserAsserted")
        if prov_str == "ModelDerived":
            prov = ProvenanceLabel.model_derived()
        else:
            prov = ProvenanceLabel.external_user_asserted()

        cardinality = (cmd.extra or {}).get("cardinality", "Functional")

        # Always include valid_time_confidence (PLANNING.md risk R1)
        valid_time: dict = {"valid_time_confidence": cmd.conf}
        if cmd.since:
            valid_time["start"] = cmd.since
        if cmd.until:
            valid_time["end"] = cmd.until

        request = {
            "agent_id": self._agent_id,
            "subject": cmd.subject,
            "predicate": cmd.predicate,
            "value": cmd.value,
            "provenance": prov,
            "cardinality": cardinality,
            "valid_time": valid_time,
            "confidence": {
                "value_confidence": cmd.conf,
                "valid_time_confidence": cmd.conf,
            },
            "criticality": "Medium",
            "derived_from": [],
        }

        resp = self._engine.ingest_claim(request)
        self.n_ingests += 1
        disp = resp["disposition"]
        ref = resp["claim_ref"]
        contested = resp.get("contested_with", [])

        # Register in local registry
        self._registry[ref] = ClaimMeta(
            subject=cmd.subject,
            predicate=cmd.predicate,
            value=cmd.value,
            provenance=prov,
            valid_time=valid_time if (cmd.since or cmd.until) else None,
            conf=cmd.conf,
            disposition=disp,
            claim_ref=ref,
        )

        lines = [f"Ingested: {cmd.subject} {cmd.predicate} = \"{cmd.value}\""]
        lines.append(f"  disposition: {disp}")
        lines.append(f"  claim_ref:   {ref[:8]}...")

        if disp == Disposition.Contested or (contested and disp != Disposition.CommittedCheap):
            self.n_contested += 1
            lines.append(f"  contested_with: {[r[:8]+'...' for r in contested]}")
            lines.append("  [!] Contested — use /reconcile to resolve.")
        elif disp == Disposition.Superseded:
            self.n_superseded += 1
            lines.append("  [note] Claim was immediately superseded by an existing belief.")

        return "\n".join(lines)

    # ── RECALL ────────────────────────────────────────────────────────────────

    def _handle_recall(self, cmd: ParsedCommand) -> str:
        self.n_queries += 1
        resp = self._engine.query_memory({
            "agent_id": self._agent_id,
            "subject": cmd.subject,
            "predicate": cmd.predicate,
        })
        return self._apply_agent_rules(resp, cmd.subject, cmd.predicate)

    def _apply_agent_rules(self, resp: dict, subject: str, predicate: str) -> str:
        """Apply agent rules R1-R5 to a query_memory response."""
        belief = resp.get("belief", {})
        status = belief.get("status", "UNKNOWN")
        primary = belief.get("primary")

        # R5 — no belief at all
        if primary is None:
            return (
                f"No memory for {subject} {predicate}.\n"
                "Use INGEST to add a claim."
            )

        value = primary.get("fact", {}).get("value")
        conf = primary.get("confidence", {})
        conf_val = conf.get("value_confidence") if isinstance(conf, dict) else conf
        vt = primary.get("valid_time") or {}
        vt_start = vt.get("start", "") if isinstance(vt, dict) else ""
        vt_end = (vt.get("end") or "open") if isinstance(vt, dict) else "open"
        ref = primary.get("claim_ref", "")[:8]

        currency = primary.get("currency_signal", {}) or {}
        corroboration = currency.get("corroboration_count", 0)

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
            self.n_contested += 1
            alternatives = belief.get("alternatives", []) or []
            lines = [f"[CONTESTED] {subject} {predicate} has conflicting claims:"]
            lines.append(f"  Primary:  \"{value}\" conf={_conf_str(conf_val)}"
                         f"  valid: {vt_start or '?'} → {vt_end}  ref={ref}...")
            for alt in alternatives:
                alt_fact = alt.get("fact", {}) or {}
                alt_val = alt_fact.get("value")
                alt_conf = alt.get("confidence", {}) or {}
                alt_cv = alt_conf.get("value_confidence") if isinstance(alt_conf, dict) else alt_conf
                alt_vt = alt.get("valid_time") or {}
                alt_start = alt_vt.get("start", "") if isinstance(alt_vt, dict) else ""
                alt_end = (alt_vt.get("end") or "open") if isinstance(alt_vt, dict) else "open"
                alt_ref = alt.get("claim_ref", "")[:8]
                lines.append(f"  Conflict: \"{alt_val}\" conf={_conf_str(alt_cv)}"
                             f"  valid: {alt_start or '?'} → {alt_end}  ref={alt_ref}...")
            lines.append("  Use /reconcile to resolve. Agent will NOT guess.")
            return "\n".join(lines)

        if status in ("Superseded", "Invalidated"):
            # R3
            self.n_superseded += 1
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

    # ── RECALL_REENTRY ────────────────────────────────────────────────────────

    def _handle_recall_reentry(self, cmd: ParsedCommand) -> str:
        """R4 — ingest as recall-re-entry then re-query to show firewall held."""
        ref = cmd.source_claim_ref or ""
        request = {
            "agent_id": self._agent_id,
            "subject": cmd.subject,
            "predicate": cmd.predicate,
            "value": cmd.value,
            "provenance": ProvenanceLabel.recall_re_entry(),
            "cardinality": "Functional",
            "valid_time": {"valid_time_confidence": 0.7},
            "confidence": {"value_confidence": 0.7, "valid_time_confidence": 0.7},
            "criticality": "Low",
            "derived_from": [ref] if ref else [],
        }
        resp = self._engine.ingest_claim(request)
        self.n_ingests += 1
        disp = resp["disposition"]
        new_ref = resp["claim_ref"][:8]

        # Register
        self._registry[resp["claim_ref"]] = ClaimMeta(
            subject=cmd.subject,
            predicate=cmd.predicate,
            value=cmd.value,
            provenance=ProvenanceLabel.recall_re_entry(),
            valid_time=None,
            conf=0.7,
            disposition=disp,
            claim_ref=resp["claim_ref"],
        )

        # Re-query to show the current belief
        q = self._engine.query_memory({
            "agent_id": self._agent_id,
            "subject": cmd.subject,
            "predicate": cmd.predicate,
        })
        currency = (q.get("belief", {}).get("primary") or {}).get("currency_signal", {}) or {}
        corroboration = currency.get("corroboration_count", 0)

        lines = [
            f"RECALL_REENTRY: {cmd.subject} {cmd.predicate} = \"{cmd.value}\"",
            f"  re-entry disposition: {disp}  ref: {new_ref}...",
            f"  Belief unchanged (firewall held).  corroboration_count={corroboration}",
        ]
        return "\n".join(lines)

    # ── /memory ───────────────────────────────────────────────────────────────

    def _handle_memory(self) -> str:
        """Show unique claims from the session registry + audit."""
        if not self._registry:
            return "Memory: (empty this session) — no claims ingested yet."

        lines = [f"Memory: {len(self._registry)} claim(s) ingested this session:"]
        seen_subj_pred: set[tuple[str, str]] = set()
        for ref, meta in self._registry.items():
            key = (meta.subject, meta.predicate)
            marker = ""
            if key not in seen_subj_pred:
                seen_subj_pred.add(key)
                # Query for current belief
                try:
                    q = self._engine.query_memory({
                        "agent_id": self._agent_id,
                        "subject": meta.subject,
                        "predicate": meta.predicate,
                    })
                    cur_val = belief_value(q)
                    cur_status = belief_status(q)
                    marker = f"  [current belief: \"{cur_val}\" {cur_status}]"
                except Exception:
                    pass
            lines.append(f"  {ref[:8]}  {meta.subject} {meta.predicate} = \"{meta.value}\""
                         f"  [{meta.disposition}]{marker}")
        return "\n".join(lines)

    # ── /history ──────────────────────────────────────────────────────────────

    def _handle_history(self, cmd: ParsedCommand) -> str:
        """R6 — real audit entries + session registry for subject/predicate."""
        # Find claim_refs for this subject/predicate from the registry
        refs_for_line = {
            ref for ref, meta in self._registry.items()
            if meta.subject == cmd.subject and meta.predicate == cmd.predicate
        }

        if not refs_for_line:
            return (
                f"No history for {cmd.subject} {cmd.predicate} in this session.\n"
                "Claims ingested in previous sessions are tracked by claim_ref only."
            )

        # Pull full audit and filter by known refs
        audit = self._engine.query_audit({
            "agent_id": self._agent_id,
            "claim_ref": None,
            "from_tx_time": None,
            "limit": 500,
        })
        entries = audit.get("entries", [])
        # Filter audit entries to matching refs
        relevant = [e for e in entries if e.get("claim_ref") in refs_for_line]

        lines = [f"History: {cmd.subject} {cmd.predicate}"]
        lines.append(f"  {len(refs_for_line)} distinct claim(s), {len(relevant)} audit event(s):")

        # Group by claim_ref
        by_ref: dict[str, list[dict]] = {}
        for e in relevant:
            r = e["claim_ref"]
            by_ref.setdefault(r, []).append(e)

        for ref in refs_for_line:
            meta = self._registry[ref]
            lines.append(f"\n  claim {ref[:8]}  value=\"{meta.value}\"  prov={_prov_abbr(meta.provenance)}")
            for e in by_ref.get(ref, []):
                lines.append(f"    {e.get('recorded_at','')[:19]}  {e.get('event_kind','?')}  → {e.get('disposition','?')}")

        return "\n".join(lines)

    # ── /why ─────────────────────────────────────────────────────────────────

    def _handle_why(self, cmd: ParsedCommand) -> str:
        """R6 — show disposition transitions from audit entries."""
        refs_for_line = {
            ref for ref, meta in self._registry.items()
            if meta.subject == cmd.subject and meta.predicate == cmd.predicate
        }

        if not refs_for_line:
            return f"No session history for {cmd.subject} {cmd.predicate}."

        audit = self._engine.query_audit({
            "agent_id": self._agent_id,
            "claim_ref": None,
            "from_tx_time": None,
            "limit": 500,
        })
        entries = audit.get("entries", [])
        relevant = [e for e in entries if e.get("claim_ref") in refs_for_line]

        lines = [f"Why: {cmd.subject} {cmd.predicate} — disposition timeline ({len(relevant)} events):"]
        for e in sorted(relevant, key=lambda x: x.get("recorded_at", "")):
            ref_short = e.get("claim_ref", "")[:8]
            meta = self._registry.get(e.get("claim_ref", ""))
            val = meta.value if meta else "?"
            lines.append(
                f"  {e.get('recorded_at','')[:19]}  {ref_short}  \"{val}\" "
                f" {e.get('event_kind','?')} → {e.get('disposition','?')}"
            )
        return "\n".join(lines)

    # ── /contested ────────────────────────────────────────────────────────────

    def _handle_contested(self) -> str:
        # Check current beliefs for contested subject/predicate pairs
        contested_pairs: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for ref, meta in self._registry.items():
            key = (meta.subject, meta.predicate)
            if key in seen:
                continue
            seen.add(key)
            try:
                q = self._engine.query_memory({
                    "agent_id": self._agent_id,
                    "subject": meta.subject,
                    "predicate": meta.predicate,
                })
                if belief_status(q) in ("Contested", "PendingConflict"):
                    contested_pairs.append(key)
            except Exception:
                pass

        if not contested_pairs:
            return "No contested beliefs found in the current session."

        lines = [f"Contested beliefs ({len(contested_pairs)}):"]
        for subj, pred in contested_pairs:
            lines.append(f"  {subj} {pred}  — use /reconcile {subj} {pred}")
        return "\n".join(lines)

    # ── /audit ────────────────────────────────────────────────────────────────

    def _handle_audit(self, cmd: ParsedCommand) -> str:
        audit = self._engine.query_audit({
            "agent_id": self._agent_id,
            "claim_ref": None,
            "from_tx_time": None,
            "limit": cmd.audit_limit,
        })
        entries = audit.get("entries", [])
        if not entries:
            return "Audit: no entries yet."
        lines = [f"Audit: last {len(entries)} entries:"]
        for e in entries:
            ref = e.get("claim_ref", "")[:8]
            disp = e.get("disposition", "?")
            kind = e.get("event_kind", "?")
            ts = str(e.get("recorded_at", ""))[:19]
            # Enrich with registry data if available
            meta = self._registry.get(e.get("claim_ref", ""))
            if meta:
                lines.append(f"  {ts}  {ref}  {meta.subject} {meta.predicate}=\"{meta.value}\"  {kind}→[{disp}]")
            else:
                lines.append(f"  {ts}  {ref}  {kind}→[{disp}]")
        return "\n".join(lines)

    # ── /reconcile ────────────────────────────────────────────────────────────

    def _handle_reconcile(self, cmd: ParsedCommand) -> str:
        resp = self._engine.reconcile({
            "agent_id": self._agent_id,
            "subject_lines": [(cmd.subject, cmd.predicate)],
        })
        outcomes = resp.get("outcomes", [])
        escalations = resp.get("oracle_escalations", 0)

        if not outcomes:
            return (
                f"Reconcile {cmd.subject} {cmd.predicate}: no explicit outcomes "
                "(claims may have already self-resolved)."
            )

        lines = [f"Reconcile {cmd.subject} {cmd.predicate}:"]
        for ref, disp in outcomes:
            if disp in (Disposition.Superseded, Disposition.Invalidated, "Superseded", "Invalidated"):
                self.n_superseded += 1
            lines.append(f"  {ref[:8]}...  → {disp}")
        if escalations:
            lines.append(f"  oracle_escalations: {escalations}")
        lines.append("  Re-query with RECALL to see the current belief.")
        return "\n".join(lines)


# ── Prov abbreviation ─────────────────────────────────────────────────────────

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
    return str(prov)[:6]


# ── Help text ─────────────────────────────────────────────────────────────────

_HELP_TEXT = """\
mempill Console Agent — command grammar:

  INGEST <subject> <predicate> "<value>" [SINCE <ISO>] [UNTIL <ISO>] [CONF <0-1>]
    Ingest a claim into the memory engine.

  RECALL <subject> <predicate>
    Recall the current belief for a subject/predicate pair.

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
