"""
tests/test_granularity_display.py — deterministic tests for partial-date ingest
and honest granularity display (0.3.0 showcase).

Tests verify:
  TG1  Parser: INGEST SINCE 2020-03 UNTIL 2023 → since/until preserved as raw strings
  TG2  Parser: INGEST SINCE 2020-03-15 → day granularity preserved
  TG3  Parser: INGEST SINCE 2020 → year granularity preserved
  TG4  Integration: month-precision SINCE + year-precision UNTIL → display shows "2020-03" / "2023"
  TG5  Integration: day-precision dates → display shows "2023-03-15" (no fabrication)
  TG6  Integration: recall_at with partial-date ingest → honest valid window in result
  TG7  Integration: plain RECALL after partial-date ingest → vt_start_display correct
  TG8  Integration: month date → display does NOT contain fabricated day "2020-03-01"
  TG9  Integration: year date → display does NOT contain fabricated month "2023-01"

All tests use the real in-memory engine (no API key required).
"""
from __future__ import annotations

import pytest

import mempill
from mempill import ProvenanceLabel

from mempill_demo.adapters.inference_deterministic import DeterministicParser
from mempill_demo.adapters.memory_mempill import MempillMemoryStore
from mempill_demo.domain.models import CommandKind, ParsedCommand, SessionStats
from mempill_demo.domain.agent import handle_command


# ── Parser tests (no engine) ─────────────────────────────────────────────────

parser = DeterministicParser()


def test_tg1_parser_partial_dates_preserved():
    """TG1: INGEST SINCE 2020-03 UNTIL 2023 passes raw strings through."""
    cmd = parser.parse('INGEST acme:ceo held_by "Alice" SINCE 2020-03 UNTIL 2023')
    assert cmd.kind == CommandKind.INGEST
    assert cmd.since == "2020-03", f"Expected '2020-03', got {cmd.since!r}"
    assert cmd.until == "2023", f"Expected '2023', got {cmd.until!r}"


def test_tg2_parser_day_precision_preserved():
    """TG2: INGEST SINCE 2020-03-15 → exact date preserved."""
    cmd = parser.parse('INGEST acme:ceo held_by "Bob" SINCE 2023-03-15')
    assert cmd.kind == CommandKind.INGEST
    assert cmd.since == "2023-03-15", f"Expected '2023-03-15', got {cmd.since!r}"


def test_tg3_parser_year_precision_preserved():
    """TG3: INGEST SINCE 2020 → year-only string preserved."""
    cmd = parser.parse('INGEST acme:ceo held_by "Alice" SINCE 2020')
    assert cmd.kind == CommandKind.INGEST
    assert cmd.since == "2020", f"Expected '2020', got {cmd.since!r}"


# ── Integration tests (real engine) ──────────────────────────────────────────

def _make_engine_store() -> tuple[mempill.Engine, MempillMemoryStore]:
    engine = mempill.open_in_memory()
    store = MempillMemoryStore(engine, "tg-agent")
    return engine, store


def test_tg4_month_year_display():
    """TG4: month SINCE + year UNTIL → display strings show "2020-03" and "2023"."""
    _, store = _make_engine_store()
    cmd = ParsedCommand(
        kind=CommandKind.INGEST,
        subject="acme:ceo",
        predicate="held_by",
        value="Alice",
        since="2020-03",
        until="2023",
        conf=0.9,
    )
    store.ingest(cmd)

    belief = store.recall("acme:ceo", "held_by")
    assert belief.value == "Alice", f"Expected Alice, got {belief.value!r}"

    # Honest display must NOT fabricate day or month
    assert belief.vt_start_display is not None, "vt_start_display should be set"
    assert belief.vt_end_display is not None, "vt_end_display should be set"
    assert belief.vt_start_display == "2020-03", (
        f"Expected '2020-03', got {belief.vt_start_display!r}"
    )
    assert belief.vt_end_display == "2023", (
        f"Expected '2023', got {belief.vt_end_display!r}"
    )


def test_tg5_day_precision_display():
    """TG5: day-precision dates → display shows "2023-03-15" exactly."""
    _, store = _make_engine_store()
    cmd = ParsedCommand(
        kind=CommandKind.INGEST,
        subject="acme:ceo",
        predicate="held_by",
        value="Bob",
        since="2023-03-15",
        until=None,
        conf=0.9,
    )
    store.ingest(cmd)

    belief = store.recall("acme:ceo", "held_by")
    assert belief.value == "Bob"
    assert belief.vt_start_display == "2023-03-15", (
        f"Expected '2023-03-15', got {belief.vt_start_display!r}"
    )
    # open-ended: vt_end_display should be None (no end set)
    assert belief.vt_end_display is None, (
        f"Expected None for open-ended, got {belief.vt_end_display!r}"
    )


def test_tg6_recall_at_shows_honest_window():
    """TG6: recall_at with partial-date claim → honest valid window in BeliefView."""
    _, store = _make_engine_store()

    # Alice: 2020-03 → 2023
    alice_cmd = ParsedCommand(
        kind=CommandKind.INGEST,
        subject="acme:ceo",
        predicate="held_by",
        value="Alice",
        since="2020-03",
        until="2023",
        conf=0.9,
    )
    store.ingest(alice_cmd)

    # Bob: 2023-03-15 → open
    bob_cmd = ParsedCommand(
        kind=CommandKind.INGEST,
        subject="acme:ceo",
        predicate="held_by",
        value="Bob",
        since="2023-03-15",
        until=None,
        conf=0.9,
    )
    store.ingest(bob_cmd)

    # Query valid at 2021-06-01 → Alice
    belief = store.recall_at("acme:ceo", "held_by", valid_at="2021-06-01T00:00:00Z")
    assert belief.value == "Alice", f"Expected Alice, got {belief.value!r}"
    assert belief.vt_start_display == "2020-03", (
        f"Expected '2020-03', got {belief.vt_start_display!r}"
    )
    assert belief.vt_end_display == "2023", (
        f"Expected '2023', got {belief.vt_end_display!r}"
    )

    # Query valid at 2025-01-01 → Bob
    belief_bob = store.recall_at("acme:ceo", "held_by", valid_at="2025-01-01T00:00:00Z")
    assert belief_bob.value == "Bob", f"Expected Bob, got {belief_bob.value!r}"
    assert belief_bob.vt_start_display == "2023-03-15", (
        f"Expected '2023-03-15', got {belief_bob.vt_start_display!r}"
    )


def test_tg7_plain_recall_vt_start_display():
    """TG7: plain RECALL after partial-date ingest → vt_start_display matches granularity."""
    _, store = _make_engine_store()
    cmd = ParsedCommand(
        kind=CommandKind.INGEST,
        subject="org:found",
        predicate="founded_in",
        value="Acme Corp",
        since="2010",
        until=None,
        conf=0.95,
    )
    store.ingest(cmd)

    belief = store.recall("org:found", "founded_in")
    assert belief.value == "Acme Corp"
    assert belief.vt_start_display == "2010", (
        f"Expected '2010' (year granularity), got {belief.vt_start_display!r}"
    )


def test_tg8_month_display_no_fabricated_day():
    """TG8: month-precision date → display does NOT contain fabricated day."""
    _, store = _make_engine_store()
    cmd = ParsedCommand(
        kind=CommandKind.INGEST,
        subject="event:start",
        predicate="began",
        value="Project Alpha",
        since="2020-03",
        until=None,
        conf=0.9,
    )
    store.ingest(cmd)

    belief = store.recall("event:start", "began")
    assert belief.vt_start_display is not None
    # Must NOT include fabricated day component
    assert "-01" not in belief.vt_start_display, (
        f"Display should not contain fabricated day, got {belief.vt_start_display!r}"
    )
    assert belief.vt_start_display == "2020-03"


def test_tg9_year_display_no_fabricated_month():
    """TG9: year-precision date → display does NOT contain fabricated month."""
    _, store = _make_engine_store()
    cmd = ParsedCommand(
        kind=CommandKind.INGEST,
        subject="acme:ceo",
        predicate="tenure_end",
        value="Alice",
        since=None,
        until="2023",
        conf=0.9,
    )
    store.ingest(cmd)

    belief = store.recall("acme:ceo", "tenure_end")
    assert belief.vt_end_display is not None
    # Must NOT include fabricated month or day
    assert belief.vt_end_display == "2023", (
        f"Expected '2023', got {belief.vt_end_display!r}"
    )
    assert "-01" not in belief.vt_end_display


# ── REPL command path tests ───────────────────────────────────────────────────

def test_tg10_repl_ingest_partial_date_response():
    """TG10: handle_command INGEST with partial dates → disposition text returned."""
    engine = mempill.open_in_memory()
    store = MempillMemoryStore(engine, "tg10-agent")
    stats = SessionStats()

    cmd = parser.parse('INGEST acme:ceo held_by "Alice" SINCE 2020-03 UNTIL 2023')
    resp = handle_command(cmd, store, stats)
    assert "Alice" in resp.text
    assert "Ingested" in resp.text
    assert stats.n_ingests == 1


def test_tg11_repl_recall_shows_honest_granularity():
    """TG11: RECALL after partial-date ingest → valid window shows "2020-03" not "2020-03-01"."""
    engine = mempill.open_in_memory()
    store = MempillMemoryStore(engine, "tg11-agent")
    stats = SessionStats()

    # Ingest via parser (tests the full INGEST grammar → adapter path)
    ingest_cmd = parser.parse('INGEST acme:ceo held_by "Alice" SINCE 2020-03 UNTIL 2023')
    handle_command(ingest_cmd, store, stats)

    recall_cmd = ParsedCommand(
        kind=CommandKind.RECALL,
        subject="acme:ceo",
        predicate="held_by",
    )
    resp = handle_command(recall_cmd, store, stats)

    assert "Alice" in resp.text
    assert "2020-03" in resp.text, (
        f"Expected '2020-03' in RECALL output, got: {resp.text!r}"
    )
    assert "2020-03-01" not in resp.text, (
        f"Fabricated day found in RECALL output: {resp.text!r}"
    )
    assert "2023" in resp.text
    assert "2023-01" not in resp.text, (
        f"Fabricated month found in RECALL output: {resp.text!r}"
    )
