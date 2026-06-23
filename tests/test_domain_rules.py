"""
tests/test_domain_rules.py — pure unit tests for R1-R6 using FakeMemoryStore.

No mempill wheel required.
"""
from __future__ import annotations

import pytest

from mempill_demo.domain.agent import handle_command
from mempill_demo.domain.models import CommandKind, ParsedCommand, SessionStats
from tests.fakes import FakeMemoryStore


def _ingest(store, subject, predicate, value, conf=0.9, since=None):
    cmd = ParsedCommand(
        kind=CommandKind.INGEST,
        subject=subject, predicate=predicate, value=value,
        conf=conf, since=since,
    )
    return handle_command(cmd, store, SessionStats())


def _recall(store, subject, predicate, stats=None):
    if stats is None:
        stats = SessionStats()
    cmd = ParsedCommand(kind=CommandKind.RECALL, subject=subject, predicate=predicate)
    return handle_command(cmd, store, stats)


def test_r1_committed_returns_value():
    """R1: committed belief → Memory: <subj> <pred> = '<value>'"""
    store = FakeMemoryStore()
    _ingest(store, "alice", "role", "CEO")
    stats = SessionStats()
    resp = _recall(store, "alice", "role", stats)
    assert "CEO" in resp.text
    assert "Memory:" in resp.text
    assert resp.kind == CommandKind.RECALL


def test_r5_no_belief():
    """R5: no memory → 'No memory for ...'"""
    store = FakeMemoryStore()
    resp = _recall(store, "bob", "role")
    assert "No memory" in resp.text


def test_r2_contested_surfaces_both():
    """R2: contested belief → CONTESTED + both values + /reconcile hint"""
    store = FakeMemoryStore()
    stats = SessionStats()
    _ingest(store, "acme", "ceo", "Alice")
    _ingest(store, "acme", "ceo", "Bob")
    resp = _recall(store, "acme", "ceo", stats)
    assert "CONTESTED" in resp.text
    assert "Alice" in resp.text or "Bob" in resp.text
    assert "reconcile" in resp.text.lower()


def test_r3_superseded_shows_history_hint():
    """R3: superseded belief → status shown + /history hint"""
    store = FakeMemoryStore()
    stats = SessionStats()
    # Ingest two, reconcile, then recall — the winner is Committed; loser is Superseded
    _ingest(store, "acme", "ceo", "Alice")
    _ingest(store, "acme", "ceo", "Bob")
    # Manually supersede the first to test R3 directly
    metas, _ = store.history("acme", "ceo")
    metas[0].disposition = "Superseded"
    metas[1].disposition = "CommittedCheap"
    resp = _recall(store, "acme", "ceo", stats)
    # Should surface the current committed belief (Bob)
    assert "Bob" in resp.text


def test_r4_recall_reentry_firewall():
    """R4: RECALL_REENTRY disposition returns firewall message."""
    store = FakeMemoryStore()
    stats = SessionStats()
    _ingest(store, "acme", "ceo", "Alice")
    cmd = ParsedCommand(
        kind=CommandKind.RECALL_REENTRY,
        subject="acme",
        predicate="ceo",
        value="Alice",
        source_claim_ref="fake-ref",
    )
    resp = handle_command(cmd, store, stats)
    assert "RECALL_REENTRY" in resp.text
    assert "firewall" in resp.text.lower()


def test_r6_history_shows_audit():
    """R6: /history returns audit events for known refs."""
    store = FakeMemoryStore()
    stats = SessionStats()
    _ingest(store, "acme", "ceo", "Alice")
    cmd = ParsedCommand(kind=CommandKind.HISTORY, subject="acme", predicate="ceo")
    resp = handle_command(cmd, store, stats)
    assert "History:" in resp.text
    assert "Alice" in resp.text


def test_r6_why_shows_timeline():
    """R6: /why returns disposition timeline."""
    store = FakeMemoryStore()
    stats = SessionStats()
    _ingest(store, "acme", "ceo", "Alice")
    cmd = ParsedCommand(kind=CommandKind.WHY, subject="acme", predicate="ceo")
    resp = handle_command(cmd, store, stats)
    assert "Why:" in resp.text


def test_reconcile_resolves_contested():
    """Reconcile reduces contested pair to a single committed belief."""
    store = FakeMemoryStore()
    stats = SessionStats()
    _ingest(store, "acme", "ceo", "Alice")
    _ingest(store, "acme", "ceo", "Bob")

    cmd = ParsedCommand(kind=CommandKind.RECONCILE, subject="acme", predicate="ceo")
    resp = handle_command(cmd, store, stats)
    assert "Reconcile" in resp.text

    # After reconcile, recall should be non-contested
    recall_resp = _recall(store, "acme", "ceo", stats)
    assert "CONTESTED" not in recall_resp.text


def test_help_command():
    """HELP returns the help text."""
    store = FakeMemoryStore()
    stats = SessionStats()
    cmd = ParsedCommand(kind=CommandKind.HELP)
    resp = handle_command(cmd, store, stats)
    assert "INGEST" in resp.text
    assert "RECALL" in resp.text


def test_memory_command_empty():
    """MEMORY with empty registry returns 'empty' message."""
    store = FakeMemoryStore()
    stats = SessionStats()
    cmd = ParsedCommand(kind=CommandKind.MEMORY)
    resp = handle_command(cmd, store, stats)
    assert "empty" in resp.text.lower()


def test_memory_command_with_claims():
    """MEMORY with claims returns registry summary."""
    store = FakeMemoryStore()
    stats = SessionStats()
    _ingest(store, "x", "y", "z")
    cmd = ParsedCommand(kind=CommandKind.MEMORY)
    resp = handle_command(cmd, store, stats)
    assert "Memory:" in resp.text
    assert "z" in resp.text
