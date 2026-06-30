"""
mempill_showcase.scenarios.naive_baseline — drive the 4 money-shot beats against NaiveAdapter.

Mirrors the relevant beats from executive_assistant.py but uses NaiveAdapter exclusively.
Since NaiveAdapter is last-write-wins with no valid-time/Contested/audit provenance,
each beat showcases a specific failure mode.

Public interface:
  run_naive_baseline(adapter: NaiveAdapter) -> NaiveTrace
    Seeds the same Day-0 claims, then drives the 4 contrast beats and returns a NaiveTrace.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from mempill_showcase.adapters.memory.naive_adapter import NaiveAdapter
from mempill_showcase.core.domain.models import ClaimInput
from mempill_showcase.scenarios.seed_data import AGENT_ID


def _seed_naive(adapter: NaiveAdapter, agent_id: str) -> None:
    """Load the same Day-0 claims into the NaiveAdapter (no provenance, no valid_from)."""
    seed = [
        ("alice-chen",  "employer",             "Acme Corp / VP Engineering"),
        ("alice-chen",  "city",                 "Austin TX"),
        ("alice-chen",  "dietary_restriction",  "vegetarian"),
        ("bob-liu",     "employer",             "Meridian Ventures / Partner"),
        ("bob-liu",     "travel_preference",    "window seat, no checked bags"),
        ("acme-corp",   "ceo",                  "Diane Foster"),
        ("jordan-park", "preferred_hotel",      "Marriott Bonvoy Gold"),
    ]
    for subject, predicate, value in seed:
        claim = ClaimInput(subject=subject, predicate=predicate, value=value)
        adapter.write_claim(agent_id, claim)


# ── Beat result ───────────────────────────────────────────────────────────────

@dataclass
class NaiveBeatResult:
    """Result of a single naive contrast beat."""
    beat_id: str
    description: str
    naive_answer: Any            # What naive actually returned
    failure_mode: str            # Short description of the failure
    has_point_in_time: bool = False   # Can it answer "what was X on date D?" → always False
    has_conflict_detection: bool = False  # Did it surface a conflict? → always False
    has_audit_trail: bool = False     # Does it have a real audit trail? → always False
    # Raw write receipt for Contrast 2
    write_receipt: Optional[Any] = None
    extra: dict = field(default_factory=dict)


@dataclass
class NaiveTrace:
    """Full trace of the 4 contrast beats on NaiveAdapter."""
    beats: list[NaiveBeatResult] = field(default_factory=list)

    def beat(self, beat_id: str) -> Optional[NaiveBeatResult]:
        for b in self.beats:
            if b.beat_id == beat_id:
                return b
        return None


# ── Main runner ───────────────────────────────────────────────────────────────

def run_naive_baseline(adapter: Optional[NaiveAdapter] = None) -> NaiveTrace:
    """Drive the 4 money-shot contrast beats through NaiveAdapter.

    Args:
        adapter: Optional pre-built NaiveAdapter. Created fresh if None.

    Returns:
        NaiveTrace documenting what naive actually does at each contrast beat.
    """
    if adapter is None:
        adapter = NaiveAdapter()

    trace = NaiveTrace()
    agent_id = AGENT_ID

    # Seed Day-0 data
    _seed_naive(adapter, agent_id)

    # ── Contrast 1: The Austin Lunch Failure ─────────────────────────────────
    # After the move is recorded (last-write-wins overwrites city to NYC),
    # the old "Austin TX" value is gone with no recovery possible.
    # However if we recall BEFORE the update we get Austin; after the update
    # naive returns NYC (because it's last-write-wins), BUT there is no way
    # to distinguish "is this current?" or recover Austin for history.
    # The failure: no ability to ask "what was the city at time X?" → irrecoverable history.

    # Step 1: recall before update (Austin)
    belief_before = adapter.recall(agent_id, "alice-chen", "city")
    city_before = belief_before.value

    # Step 2: Alice moves to NYC (last-write-wins silently overwrites)
    adapter.write_claim(agent_id, ClaimInput(
        subject="alice-chen", predicate="city", value="New York NY"
    ))

    # Step 3: now try to recover Austin TX history → impossible
    belief_after = adapter.recall(agent_id, "alice-chen", "city")
    city_after = belief_after.value  # NYC (correct for current), but Austin gone

    # Attempt to call query_at → will raise AttributeError (by design)
    has_point_in_time = hasattr(adapter, "query_at")

    trace.beats.append(NaiveBeatResult(
        beat_id="C1",
        description="Stale city: After Alice moves to NYC, can we recover Austin TX for history?",
        naive_answer=city_after,  # Returns NYC (last write wins)
        failure_mode=(
            "History irrecoverable. 'Austin TX' is permanently overwritten. "
            "No point-in-time query exists. query_at → AttributeError."
        ),
        has_point_in_time=has_point_in_time,
        extra={
            "city_before_update": city_before,
            "city_after_update": city_after,
            "austin_recoverable": False,
            "query_at_available": has_point_in_time,
        },
    ))

    # ── Contrast 2: Silent Overwrite of Alice's Title ─────────────────────────
    # A conflicting title arrives (CTO from a news source vs VP Engineering from user).
    # Naive: silently overwrites. No Contested. No HITL. Prior title gone.

    # Step 1: read existing employer (VP Engineering from seed)
    belief_employer_before = adapter.recall(agent_id, "alice-chen", "employer")
    employer_before = belief_employer_before.value

    # Step 2: write conflicting claim (external says CTO) — silently overwrites
    cto_receipt = adapter.write_claim(agent_id, ClaimInput(
        subject="alice-chen", predicate="employer", value="Acme Corp / CTO"
    ))

    # Step 3: read back — VP Engineering is gone, CTO is now "current"
    belief_employer_after = adapter.recall(agent_id, "alice-chen", "employer")
    employer_after = belief_employer_after.value

    is_contested = cto_receipt.is_contested()  # always False for naive

    trace.beats.append(NaiveBeatResult(
        beat_id="C2",
        description="Silent overwrite: conflicting employer (VP Eng vs CTO) — does naive surface a conflict?",
        naive_answer=employer_after,  # CTO (overwrite happened silently)
        failure_mode=(
            "Silent last-write-wins overwrite. 'VP Engineering' permanently lost. "
            "No Contested disposition. No HITL. No provenance. "
            f"receipt.is_contested()={is_contested}."
        ),
        has_conflict_detection=is_contested,
        write_receipt=cto_receipt,
        extra={
            "employer_before_write": employer_before,
            "employer_after_write": employer_after,
            "disposition": cto_receipt.disposition,
            "is_contested": is_contested,
            "vp_engineering_recoverable": False,
        },
    ))

    # ── Contrast 3: Q1 Board Report Query ─────────────────────────────────────
    # Jordan asks "What was Alice's title in Q1 (Jan 2025)?"
    # Naive can only return today's value (CTO from C2 overwrite) — wrong for Q1.
    # There is no valid_at or point-in-time query.

    # Attempt to recall employer with a "valid_at" → naive just returns current
    # (no parameter for valid_at exists in recall(); calling query_at raises AttributeError)
    belief_q1 = adapter.recall(agent_id, "alice-chen", "employer")  # returns CTO (wrong for Q1)
    q1_answer = belief_q1.value

    # Prove query_at doesn't exist
    query_at_error: Optional[str] = None
    try:
        adapter.query_at(agent_id, "alice-chen", "employer", valid_at="2025-01-01T00:00:00Z")  # type: ignore[attr-defined]
    except AttributeError as e:
        query_at_error = str(e)

    trace.beats.append(NaiveBeatResult(
        beat_id="C3",
        description="Q1 query: 'What was Alice's title in Q1 2025?' — naive returns today's value.",
        naive_answer=q1_answer,  # CTO — wrong; should be VP Engineering for Q1
        failure_mode=(
            "No point-in-time query. recall() always returns today's (last-written) value. "
            f"query_at raises AttributeError: {query_at_error!r}. "
            "Correct Q1 answer was 'Acme Corp / VP Engineering'; naive returns current value."
        ),
        has_point_in_time=False,
        extra={
            "q1_answer_from_naive": q1_answer,
            "correct_q1_answer": "Acme Corp / VP Engineering",
            "query_at_error": query_at_error,
        },
    ))

    # ── Contrast 4: Compliance Audit ──────────────────────────────────────────
    # "What did the assistant believe about Alice on March 10?"
    # Naive: only today's facts. No tx-time axis. Audit log has write events
    # but no way to reconstruct the belief state at a past moment.

    # Naive audit only has flat write events — no disposition history, no supersessions,
    # no oracle events, no tx-time queryable entries.
    naive_audit = adapter.audit(agent_id, limit=100)
    audit_entry_count = len(naive_audit)

    # Check for supersession events in audit (will be absent in naive)
    has_supersession_events = any(
        e.event_kind in ("Superseded", "OracleAdjudicated")
        for e in naive_audit
    )

    # Check for tx-time replay capability (it does not exist)
    as_of_tx_time_available = False
    tx_replay_error: Optional[str] = None
    try:
        adapter.query_at(agent_id, "alice-chen", "city", as_of_tx_time="2025-03-10T08:00:00Z")  # type: ignore[attr-defined]
    except AttributeError as e:
        tx_replay_error = str(e)

    trace.beats.append(NaiveBeatResult(
        beat_id="C4",
        description="Compliance audit: 'What did the assistant believe about Alice on March 10?'",
        naive_answer=f"{audit_entry_count} write-event entries, no tx-time replay possible",
        failure_mode=(
            "No transaction-time axis. Cannot reconstruct belief state at a past moment. "
            f"Audit has {audit_entry_count} flat write-event entries only. "
            f"No supersession events: has_supersession_events={has_supersession_events}. "
            f"as_of_tx_time replay raises AttributeError: {tx_replay_error!r}."
        ),
        has_audit_trail=False,  # Naive has write-event log but NOT a bi-temporal audit
        extra={
            "audit_entry_count": audit_entry_count,
            "has_supersession_events": has_supersession_events,
            "as_of_tx_time_available": as_of_tx_time_available,
            "tx_replay_error": tx_replay_error,
            "naive_audit_kinds": list({e.event_kind for e in naive_audit}),
        },
    ))

    return trace
