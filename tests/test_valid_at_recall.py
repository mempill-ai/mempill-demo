"""
tests/test_valid_at_recall.py — integration tests for point-in-time (bi-temporal) RECALL.

Tests verify:
  TV1  Parser: RECALL s p valid=2021-06-01 → ParsedCommand.valid_at set
  TV2  Parser: RECALL s p tx=2021-06-01 → ParsedCommand.as_of_tx_time set
  TV3  Parser: RECALL s p valid=2021-06-01 tx=2020-01-01 → both set
  TV4  Parser: RECALL s p (plain) → neither set
  TV5  Integration: clean succession (no reconcile) — valid=2021 → Alice
  TV6  Integration: clean succession — valid=2025 → Bob
  TV7  Integration: clean succession — valid=2019 → NoBelief
  TV8  Integration: after oracle Affirm (Bob wins) — valid=2021 as-of-NOW → Alice still
       (her own valid-time window covers 2021 regardless of a later Affirm; fixed in
       mempill #76 — previously the engine ignored valid_at once narrowed to one live claim)
  TV9  Integration: after oracle Affirm — valid=2021 tx=<before-adjudication> → Alice resurfaces
  TV10 Integration: current (plain) RECALL after succession → Bob

Requires prerelease mempill wheel with valid_at support.
"""
from __future__ import annotations

import datetime
import time

import pytest

import mempill
from mempill import ProvenanceLabel
from mempill_demo.adapters.human_oracle import HumanOracle
from mempill_demo.adapters.memory_mempill import MempillMemoryStore
from mempill_demo.adapters.inference_deterministic import DeterministicParser
from mempill_demo.domain.models import CommandKind, ParsedCommand, SessionStats
from mempill_demo.domain.agent import handle_command


# ── Parser unit tests (no engine needed) ─────────────────────────────────────

parser = DeterministicParser()


def test_tv1_parser_valid_modifier():
    """TV1: valid= token parsed into valid_at as ISO-8601 UTC."""
    cmd = parser.parse("RECALL acme:ceo held_by valid=2021-06-01")
    assert cmd.kind == CommandKind.RECALL
    assert cmd.subject == "acme:ceo"
    assert cmd.predicate == "held_by"
    assert cmd.valid_at == "2021-06-01T00:00:00Z"
    assert cmd.as_of_tx_time is None


def test_tv2_parser_tx_modifier():
    """TV2: tx= token parsed into as_of_tx_time as ISO-8601 UTC."""
    cmd = parser.parse("RECALL acme:ceo held_by tx=2020-01-01")
    assert cmd.kind == CommandKind.RECALL
    assert cmd.valid_at is None
    assert cmd.as_of_tx_time == "2020-01-01T00:00:00Z"


def test_tv3_parser_both_modifiers():
    """TV3: both valid= and tx= parsed correctly."""
    cmd = parser.parse("RECALL acme:ceo held_by valid=2021-06-01 tx=2020-01-01")
    assert cmd.valid_at == "2021-06-01T00:00:00Z"
    assert cmd.as_of_tx_time == "2020-01-01T00:00:00Z"


def test_tv4_parser_plain_recall():
    """TV4: plain RECALL sets neither modifier."""
    cmd = parser.parse("RECALL acme:ceo held_by")
    assert cmd.kind == CommandKind.RECALL
    assert cmd.valid_at is None
    assert cmd.as_of_tx_time is None


# ── Integration tests (real engine) ──────────────────────────────────────────

def _make_clean_succession_engine() -> tuple[mempill.Engine, MempillMemoryStore]:
    """Build an in-memory engine with Alice [2020,2023) + Bob [2023,open)."""
    engine = mempill.open_in_memory()
    agent_id = "tv-test-agent"
    store = MempillMemoryStore(engine, agent_id)

    engine.ingest_claim({
        "agent_id": agent_id, "subject": "acme:ceo", "predicate": "held_by", "value": "Alice",
        "provenance": ProvenanceLabel.external_first_hand(), "cardinality": "Functional",
        "valid_time": {
            "start": "2020-01-01T00:00:00Z",
            "end": "2023-03-15T00:00:00Z",
            "valid_time_confidence": 0.95,
        },
        "confidence": {"value_confidence": 0.95, "valid_time_confidence": 0.95},
        "criticality": "Medium", "derived_from": [],
    })
    engine.ingest_claim({
        "agent_id": agent_id, "subject": "acme:ceo", "predicate": "held_by", "value": "Bob",
        "provenance": ProvenanceLabel.external_first_hand(), "cardinality": "Functional",
        "valid_time": {"start": "2023-03-15T00:00:00Z", "valid_time_confidence": 0.9},
        "confidence": {"value_confidence": 0.9, "valid_time_confidence": 0.9},
        "criticality": "Medium", "derived_from": [],
    })
    return engine, store


def test_tv5_clean_succession_alice():
    """TV5: valid=2021-06-01 in clean succession → Alice."""
    _, store = _make_clean_succession_engine()
    belief = store.recall_at("acme:ceo", "held_by", valid_at="2021-06-01T00:00:00Z")
    assert belief.value == "Alice", f"Expected Alice, got {belief.value!r}"
    assert belief.status not in ("NoBelief", "UNKNOWN", None)


def test_tv6_clean_succession_bob():
    """TV6: valid=2025-01-01 in clean succession → Bob."""
    _, store = _make_clean_succession_engine()
    belief = store.recall_at("acme:ceo", "held_by", valid_at="2025-01-01T00:00:00Z")
    assert belief.value == "Bob", f"Expected Bob, got {belief.value!r}"


def test_tv7_clean_succession_no_belief():
    """TV7: valid=2019-01-01 before all claims → NoBelief."""
    _, store = _make_clean_succession_engine()
    belief = store.recall_at("acme:ceo", "held_by", valid_at="2019-01-01T00:00:00Z")
    assert belief.value is None
    assert belief.status in ("NoBelief", "UNKNOWN")


def test_tv8_after_reconcile_valid_at_2021_still_sees_alice():
    """TV8: after oracle Affirm, valid=2021 as-of-NOW → Alice (her own valid-time
    window covers 2021 regardless of a later Affirm bounding her forward window).
    Fixed in mempill #76: the engine previously ignored the valid_at window once
    disposition narrowing left exactly one live claim, unconditionally returning
    Bob even for dates before his own valid_from (2023-03-15)."""
    engine = mempill.open_oracle_in_memory(HumanOracle())
    agent_id = "tv8-agent"
    store = MempillMemoryStore(engine, agent_id)

    # Ingest Alice open-ended (overlapping with Bob — triggers HITL)
    engine.ingest_claim({
        "agent_id": agent_id, "subject": "acme:ceo", "predicate": "held_by", "value": "Alice",
        "provenance": ProvenanceLabel.external_first_hand(), "cardinality": "Functional",
        "valid_time": {"start": "2020-01-01T00:00:00Z", "valid_time_confidence": 0.95},
        "confidence": {"value_confidence": 0.95, "valid_time_confidence": 0.95},
        "criticality": "Medium", "derived_from": [],
    })
    engine.ingest_claim({
        "agent_id": agent_id, "subject": "acme:ceo", "predicate": "held_by", "value": "Bob",
        "provenance": ProvenanceLabel.external_first_hand(), "cardinality": "Functional",
        "valid_time": {"start": "2023-03-15T00:00:00Z", "valid_time_confidence": 0.9},
        "confidence": {"value_confidence": 0.9, "valid_time_confidence": 0.9},
        "criticality": "Medium", "derived_from": [],
    })

    pending = engine.list_pending_adjudications(agent_id=agent_id)
    assert len(pending) >= 1, "Expected at least 1 pending adjudication"
    handle = pending[0]["handle_id"]
    engine.submit_adjudication({
        "handle_id": handle, "verdict": "Affirm",
        "evidence_provenance": ProvenanceLabel.external_first_hand(),
    })

    # valid=2021 as-of-NOW: Alice is Superseded (as a disposition), but her own
    # valid-time window still covers 2021 — the point-in-time query must honor
    # that window, not just "whichever claim is currently live."
    belief = store.recall_at("acme:ceo", "held_by", valid_at="2021-06-01T00:00:00Z")
    assert belief.value == "Alice", (
        f"TV8: expected Alice (her window covers 2021 even though Bob later won "
        f"the Affirm), got {belief.value!r} status={belief.status}"
    )


def test_tv9_bitemporal_alice_resurfaces():
    """TV9: valid=2021 + tx=<before adjudication> → Alice resurfaces.

    This is the core bi-temporal value prop:
    'what did we believe in 2021 as we knew it before Bob's adjudication?'
    """
    engine = mempill.open_oracle_in_memory(HumanOracle())
    agent_id = "tv9-agent"
    store = MempillMemoryStore(engine, agent_id)

    engine.ingest_claim({
        "agent_id": agent_id, "subject": "acme:ceo", "predicate": "held_by", "value": "Alice",
        "provenance": ProvenanceLabel.external_first_hand(), "cardinality": "Functional",
        "valid_time": {"start": "2020-01-01T00:00:00Z", "valid_time_confidence": 0.95},
        "confidence": {"value_confidence": 0.95, "valid_time_confidence": 0.95},
        "criticality": "Medium", "derived_from": [],
    })

    # Record tx-time BEFORE Bob is ingested
    t_before_bob = datetime.datetime.now(datetime.timezone.utc)
    time.sleep(0.05)

    engine.ingest_claim({
        "agent_id": agent_id, "subject": "acme:ceo", "predicate": "held_by", "value": "Bob",
        "provenance": ProvenanceLabel.external_first_hand(), "cardinality": "Functional",
        "valid_time": {"start": "2023-03-15T00:00:00Z", "valid_time_confidence": 0.9},
        "confidence": {"value_confidence": 0.9, "valid_time_confidence": 0.9},
        "criticality": "Medium", "derived_from": [],
    })

    pending = engine.list_pending_adjudications(agent_id=agent_id)
    assert len(pending) >= 1
    handle = pending[0]["handle_id"]
    time.sleep(0.05)
    engine.submit_adjudication({
        "handle_id": handle, "verdict": "Affirm",
        "evidence_provenance": ProvenanceLabel.external_first_hand(),
    })

    # Bi-temporal query: valid_at=2021, as_of=before Bob was ingested → Alice
    tx_str = t_before_bob.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
    belief = store.recall_at(
        "acme:ceo", "held_by",
        valid_at="2021-06-01T00:00:00Z",
        as_of_tx_time=tx_str,
    )
    assert belief.value == "Alice", (
        f"TV9: expected Alice (as-of before Bob), got {belief.value!r} status={belief.status}"
    )


def test_tv10_plain_recall_after_succession():
    """TV10: plain RECALL after clean succession → Bob (current belief)."""
    _, store = _make_clean_succession_engine()
    belief = store.recall("acme:ceo", "held_by")
    assert belief.value == "Bob", f"Expected Bob, got {belief.value!r}"


# ── REPL command path tests ───────────────────────────────────────────────────

def test_tv11_repl_command_routes_valid_at():
    """TV11: handle_command with valid_at → produces temporal result text."""
    engine = mempill.open_in_memory()
    agent_id = "tv11-agent"
    store = MempillMemoryStore(engine, agent_id)
    stats = SessionStats()

    engine.ingest_claim({
        "agent_id": agent_id, "subject": "acme:ceo", "predicate": "held_by", "value": "Alice",
        "provenance": ProvenanceLabel.external_first_hand(), "cardinality": "Functional",
        "valid_time": {
            "start": "2020-01-01T00:00:00Z",
            "end": "2023-03-15T00:00:00Z",
            "valid_time_confidence": 0.95,
        },
        "confidence": {"value_confidence": 0.95, "valid_time_confidence": 0.95},
        "criticality": "Medium", "derived_from": [],
    })

    cmd = ParsedCommand(
        kind=CommandKind.RECALL,
        subject="acme:ceo",
        predicate="held_by",
        valid_at="2021-06-01T00:00:00Z",
    )
    resp = handle_command(cmd, store, stats)
    assert "Alice" in resp.text
    assert "valid at 2021-06-01" in resp.text


def test_tv12_repl_command_no_belief():
    """TV12: valid= query before any claim → NoBelief text."""
    engine = mempill.open_in_memory()
    agent_id = "tv12-agent"
    store = MempillMemoryStore(engine, agent_id)
    stats = SessionStats()

    # No claims at all
    cmd = ParsedCommand(
        kind=CommandKind.RECALL,
        subject="acme:ceo",
        predicate="held_by",
        valid_at="2019-01-01T00:00:00Z",
    )
    resp = handle_command(cmd, store, stats)
    assert "NoBelief" in resp.text or "no claim" in resp.text.lower()
