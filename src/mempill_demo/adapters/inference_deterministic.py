"""
mempill_demo.adapters.inference_deterministic — keyword-insensitive grammar parser.

Grammar:
  INGEST <subject> <predicate> "<value>" [SINCE <ISO>] [UNTIL <ISO>] [CONF <0-1>]
  RECALL <subject> <predicate>
  RECALL_REENTRY <subject> <predicate> "<value>" <source_claim_ref>
  /memory
  /history <subj> <pred>
  /why <subj> <pred>
  /contested
  /audit [N]
  /reconcile <subj> <pred>
  /help
  /quit
  /reset
"""
from __future__ import annotations

import re
from typing import Optional

from mempill_demo.domain.models import CommandKind, ParsedCommand


class DeterministicParser:
    """Parse a raw user input string into a ParsedCommand."""

    def parse(self, raw: str) -> ParsedCommand:
        stripped = raw.strip()
        if not stripped:
            return ParsedCommand(kind=CommandKind.UNKNOWN, raw=raw, error="Empty input")
        upper = stripped.upper()

        # ── Slash-commands ────────────────────────────────────────────────────
        if upper.startswith("/"):
            return self._parse_slash(stripped, raw)

        # ── Keyword commands ──────────────────────────────────────────────────
        first = stripped.split()[0].upper()

        if first == "INGEST":
            return self._parse_ingest(stripped, raw)
        if first == "RECALL_REENTRY":
            return self._parse_recall_reentry(stripped, raw)
        if first == "RECALL":
            return self._parse_recall(stripped, raw)

        return ParsedCommand(kind=CommandKind.UNKNOWN, raw=raw,
                             error=f"Unknown command '{first}'. Type /help.")

    # ── Slash handlers ────────────────────────────────────────────────────────

    def _parse_slash(self, stripped: str, raw: str) -> ParsedCommand:
        parts = stripped.split(maxsplit=3)
        cmd = parts[0].lower()

        if cmd == "/memory":
            return ParsedCommand(kind=CommandKind.MEMORY, raw=raw)

        if cmd == "/contested":
            return ParsedCommand(kind=CommandKind.CONTESTED, raw=raw)

        if cmd == "/help":
            return ParsedCommand(kind=CommandKind.HELP, raw=raw)

        if cmd in ("/quit", "/q", "/exit"):
            return ParsedCommand(kind=CommandKind.QUIT, raw=raw)

        if cmd == "/reset":
            return ParsedCommand(kind=CommandKind.RESET, raw=raw)

        if cmd == "/history":
            if len(parts) < 3:
                return ParsedCommand(kind=CommandKind.UNKNOWN, raw=raw,
                                     error="/history requires <subject> <predicate>")
            return ParsedCommand(kind=CommandKind.HISTORY, raw=raw,
                                 subject=parts[1], predicate=parts[2])

        if cmd == "/why":
            if len(parts) < 3:
                return ParsedCommand(kind=CommandKind.UNKNOWN, raw=raw,
                                     error="/why requires <subject> <predicate>")
            return ParsedCommand(kind=CommandKind.WHY, raw=raw,
                                 subject=parts[1], predicate=parts[2])

        if cmd == "/audit":
            limit = 10
            if len(parts) >= 2:
                try:
                    limit = int(parts[1])
                except ValueError:
                    pass
            return ParsedCommand(kind=CommandKind.AUDIT, raw=raw, audit_limit=limit)

        if cmd == "/reconcile":
            if len(parts) < 3:
                return ParsedCommand(kind=CommandKind.UNKNOWN, raw=raw,
                                     error="/reconcile requires <subject> <predicate>")
            return ParsedCommand(kind=CommandKind.RECONCILE, raw=raw,
                                 subject=parts[1], predicate=parts[2])

        return ParsedCommand(kind=CommandKind.UNKNOWN, raw=raw,
                             error=f"Unknown command '{cmd}'. Type /help.")

    # ── INGEST parser ─────────────────────────────────────────────────────────

    def _parse_ingest(self, stripped: str, raw: str) -> ParsedCommand:
        m = re.search(r'"([^"]*)"', stripped)
        if not m:
            return ParsedCommand(kind=CommandKind.UNKNOWN, raw=raw,
                                 error='INGEST requires a quoted value: INGEST subj pred "value"')
        value = m.group(1)

        before_quote = stripped[:m.start()].strip()
        tokens = before_quote.split()
        if len(tokens) < 3:
            return ParsedCommand(kind=CommandKind.UNKNOWN, raw=raw,
                                 error='INGEST syntax: INGEST <subject> <predicate> "<value>"')
        subject = tokens[1]
        predicate = tokens[2]

        after_quote = stripped[m.end():].strip()
        since, until, conf = self._parse_modifiers(after_quote)

        return ParsedCommand(
            kind=CommandKind.INGEST,
            subject=subject,
            predicate=predicate,
            value=value,
            since=since,
            until=until,
            conf=conf,
            raw=raw,
        )

    # ── RECALL parser ─────────────────────────────────────────────────────────

    def _parse_recall(self, stripped: str, raw: str) -> ParsedCommand:
        tokens = stripped.split()
        if len(tokens) < 3:
            return ParsedCommand(kind=CommandKind.UNKNOWN, raw=raw,
                                 error="RECALL requires <subject> <predicate>")
        return ParsedCommand(kind=CommandKind.RECALL, raw=raw,
                             subject=tokens[1], predicate=tokens[2])

    # ── RECALL_REENTRY parser ─────────────────────────────────────────────────

    def _parse_recall_reentry(self, stripped: str, raw: str) -> ParsedCommand:
        m = re.search(r'"([^"]*)"', stripped)
        if not m:
            return ParsedCommand(kind=CommandKind.UNKNOWN, raw=raw,
                                 error='RECALL_REENTRY requires a quoted value')
        value = m.group(1)
        before_quote = stripped[:m.start()].strip()
        tokens = before_quote.split()
        if len(tokens) < 3:
            return ParsedCommand(kind=CommandKind.UNKNOWN, raw=raw,
                                 error='RECALL_REENTRY syntax: RECALL_REENTRY <subject> <predicate> "<value>" <ref>')
        subject = tokens[1]
        predicate = tokens[2]

        after_quote = stripped[m.end():].strip()
        parts = after_quote.split()
        source_ref = parts[0] if parts else None

        return ParsedCommand(
            kind=CommandKind.RECALL_REENTRY,
            subject=subject,
            predicate=predicate,
            value=value,
            source_claim_ref=source_ref,
            raw=raw,
        )

    # ── Modifier helper ───────────────────────────────────────────────────────

    def _parse_modifiers(self, text: str) -> tuple[Optional[str], Optional[str], float]:
        since: Optional[str] = None
        until: Optional[str] = None
        conf: float = 0.9

        tokens = text.upper().split()
        original_tokens = text.split()
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok == "SINCE" and i + 1 < len(original_tokens):
                since = self._normalise_iso(original_tokens[i + 1])
                i += 2
            elif tok == "UNTIL" and i + 1 < len(original_tokens):
                until = self._normalise_iso(original_tokens[i + 1])
                i += 2
            elif tok == "CONF" and i + 1 < len(original_tokens):
                try:
                    conf = max(0.0, min(1.0, float(original_tokens[i + 1])))
                except ValueError:
                    pass
                i += 2
            else:
                i += 1

        return since, until, conf

    @staticmethod
    def _normalise_iso(date_str: str) -> str:
        if len(date_str) == 10 and re.match(r"\d{4}-\d{2}-\d{2}$", date_str):
            return date_str + "T00:00:00Z"
        return date_str
