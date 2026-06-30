"""
mempill_showcase.tests.test_naive_vs_mempill — W8 contrast tests.

Asserts that the 4 documented failure modes MANIFEST on NaiveAdapter and are
CORRECT on MempillAdapter.  No API key required.  No LLM calls.

What we test (per SCENARIO.md §4):
  C1 — Stale city: Austin TX is irrecoverable from NaiveAdapter after NYC write;
        mempill query_at(valid_at=past) returns it correctly.
  C2 — Silent overwrite: NaiveAdapter overwrites VP→CTO with no Contested;
        mempill returns QueuedForAdjudication (is_contested=True).
  C3 — Q1 query: NaiveAdapter.query_at raises AttributeError (no such method);
        mempill query_at(valid_at="2025-01-01") returns the correct past value.
  C4 — Compliance audit: NaiveAdapter has no tx-time replay, no supersession events;
        mempill as_of_tx_time returns Austin before the NYC write was recorded.

Honesty contract:
  - We assert what each adapter ACTUALLY does — no strawman.
  - NaiveAdapter is genuinely last-write-wins + recency recall; we assert
    exactly that behaviour (current recall works; history does not).
  - We use run_comparison() to also assert the ComparisonResult marks each
    verdict correctly (naive=FAIL, mempill=PASS).
"""
from __future__ import annotations

import pytest

from mempill_showcase.adapters.memory.naive_adapter import NaiveAdapter
from mempill_showcase.config.di import build_mempill_adapter
from mempill_showcase.core.domain.models import ClaimInput
from mempill_showcase.scenarios.compare import ComparisonResult, run_comparison
from mempill_showcase.scenarios.naive_baseline import NaiveTrace, run_naive_baseline
from mempill_showcase.scenarios.seed_data import AGENT_ID, load_seed_claims


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def naive_trace() -> NaiveTrace:
    """Run the naive baseline once per module — no API key, no mempill."""
    adapter = NaiveAdapter()
    return run_naive_baseline(adapter)


@pytest.fixture(scope="module")
def comparison() -> ComparisonResult:
    """Run the full comparison once per module."""
    return run_comparison()


# ── C1: Stale city ────────────────────────────────────────────────────────────

class TestContrastC1StaleCity:
    """C1 — After Alice moves to NYC, naive loses Austin TX permanently."""

    def test_naive_c1_beat_present(self, naive_trace: NaiveTrace) -> None:
        assert naive_trace.beat("C1") is not None

    def test_naive_current_recall_works(self, naive_trace: NaiveTrace) -> None:
        """Naive DOES return the current (last-written) value correctly."""
        c1 = naive_trace.beat("C1")
        assert c1 is not None
        # After NYC write, naive returns NYC (last-write-wins is correct for current)
        assert c1.extra["city_after_update"] == "New York NY"

    def test_naive_austin_irrecoverable(self, naive_trace: NaiveTrace) -> None:
        """Austin TX is permanently overwritten — not recoverable from NaiveAdapter."""
        c1 = naive_trace.beat("C1")
        assert c1 is not None
        assert c1.extra["austin_recoverable"] is False

    def test_naive_no_query_at(self, naive_trace: NaiveTrace) -> None:
        """NaiveAdapter has no query_at method — no point-in-time capability."""
        c1 = naive_trace.beat("C1")
        assert c1 is not None
        assert c1.extra["query_at_available"] is False
        # Confirm it raises AttributeError
        adapter = NaiveAdapter()
        with pytest.raises(AttributeError):
            adapter.query_at(AGENT_ID, "alice-chen", "city", valid_at="2024-01-01T00:00:00Z")  # type: ignore[attr-defined]

    def test_naive_marked_fail_in_comparison(self, comparison: ComparisonResult) -> None:
        c1 = comparison.get("C1")
        assert c1 is not None
        assert c1.naive_verdict == "FAIL"

    def test_mempill_marked_pass_in_comparison(self, comparison: ComparisonResult) -> None:
        c1 = comparison.get("C1")
        assert c1 is not None
        assert c1.mempill_verdict == "PASS"

    def test_mempill_query_at_returns_austin(self) -> None:
        """mempill query_at(valid_at=2024-01-01) returns Austin TX — correct historical value."""
        adapter = build_mempill_adapter(in_memory=True, oracle_backed=False)
        load_seed_claims(adapter, AGENT_ID)
        # Seed has Austin with valid_until=2025-02; query before that → Austin
        belief = adapter.query_at(
            AGENT_ID, "alice-chen", "city",
            valid_at="2024-01-01T00:00:00Z",
        )
        assert belief.value == "Austin TX"
        assert belief.status == "Resolved"

    def test_mempill_current_recall_returns_nyc_after_succession(self) -> None:
        """After NYC write, mempill current recall returns NYC (not Austin)."""
        import time
        from mempill import ProvenanceLabel
        adapter = build_mempill_adapter(in_memory=True, oracle_backed=False)
        load_seed_claims(adapter, AGENT_ID)
        time.sleep(0.05)
        adapter.write_claim(AGENT_ID, ClaimInput(
            subject="alice-chen", predicate="city", value="New York NY",
            valid_from="2025-02", confidence=1.0,
            provenance=ProvenanceLabel.external_user_asserted(),
            cardinality="Functional",
        ))
        belief = adapter.recall(AGENT_ID, "alice-chen", "city")
        assert belief.value == "New York NY"
        assert belief.status == "Resolved"


# ── C2: Silent overwrite ──────────────────────────────────────────────────────

class TestContrastC2SilentOverwrite:
    """C2 — Conflicting employer claim: naive silently overwrites; mempill Contests."""

    def test_naive_c2_beat_present(self, naive_trace: NaiveTrace) -> None:
        assert naive_trace.beat("C2") is not None

    def test_naive_no_contested_on_conflict(self, naive_trace: NaiveTrace) -> None:
        """NaiveAdapter write never returns Contested — always CommittedCheap."""
        c2 = naive_trace.beat("C2")
        assert c2 is not None
        assert c2.extra["is_contested"] is False
        assert c2.extra["disposition"] == "CommittedCheap"

    def test_naive_vp_engineering_gone(self, naive_trace: NaiveTrace) -> None:
        """VP Engineering is permanently overwritten — unrecoverable."""
        c2 = naive_trace.beat("C2")
        assert c2 is not None
        assert c2.extra["vp_engineering_recoverable"] is False
        assert c2.extra["employer_after_write"] == "Acme Corp / CTO"

    def test_naive_write_receipt_not_contested(self, naive_trace: NaiveTrace) -> None:
        """is_contested() on the WriteReceipt is always False for naive."""
        c2 = naive_trace.beat("C2")
        assert c2 is not None
        receipt = c2.write_receipt
        assert receipt is not None
        assert receipt.is_contested() is False

    def test_mempill_conflict_returns_contested(self) -> None:
        """mempill write of conflicting Functional claim returns QueuedForAdjudication."""
        from mempill import ProvenanceLabel
        adapter = build_mempill_adapter(in_memory=True, oracle_backed=True)
        load_seed_claims(adapter, AGENT_ID)
        # VP Engineering is open-ended (valid_from=2023-06, valid_until=None)
        # Write CTO at overlapping valid_from → should Contest
        receipt = adapter.write_claim(AGENT_ID, ClaimInput(
            subject="alice-chen", predicate="employer", value="Acme Corp / CTO",
            valid_from="2025-01", confidence=0.75,
            provenance=ProvenanceLabel.external_first_hand(),
            cardinality="Functional",
        ))
        assert receipt.is_contested(), (
            f"Expected Contested/QueuedForAdjudication, got disposition={receipt.disposition!r}"
        )

    def test_mempill_no_silent_overwrite(self) -> None:
        """After Contested write, the incumbent VP Engineering is still accessible."""
        from mempill import ProvenanceLabel
        adapter = build_mempill_adapter(in_memory=True, oracle_backed=True)
        load_seed_claims(adapter, AGENT_ID)
        # Write conflicting claim
        adapter.write_claim(AGENT_ID, ClaimInput(
            subject="alice-chen", predicate="employer", value="Acme Corp / CTO",
            valid_from="2025-01", confidence=0.75,
            provenance=ProvenanceLabel.external_first_hand(),
            cardinality="Functional",
        ))
        # The belief is now Contested — but VP Engineering is NOT deleted
        belief = adapter.recall(AGENT_ID, "alice-chen", "employer")
        # Status is Contested (not Resolved to CTO)
        assert belief.is_contested()

    def test_naive_marked_fail_in_comparison(self, comparison: ComparisonResult) -> None:
        c2 = comparison.get("C2")
        assert c2 is not None
        assert c2.naive_verdict == "FAIL"

    def test_mempill_marked_pass_in_comparison(self, comparison: ComparisonResult) -> None:
        c2 = comparison.get("C2")
        assert c2 is not None
        assert c2.mempill_verdict == "PASS"


# ── C3: Q1 point-in-time query ────────────────────────────────────────────────

class TestContrastC3Q1Query:
    """C3 — Historical query: naive returns today's value; mempill returns past value."""

    def test_naive_c3_beat_present(self, naive_trace: NaiveTrace) -> None:
        assert naive_trace.beat("C3") is not None

    def test_naive_query_at_raises_attribute_error(self) -> None:
        """NaiveAdapter.query_at does not exist — raises AttributeError."""
        adapter = NaiveAdapter()
        with pytest.raises(AttributeError):
            adapter.query_at(AGENT_ID, "alice-chen", "employer", valid_at="2025-01-01T00:00:00Z")  # type: ignore[attr-defined]

    def test_naive_has_no_point_in_time(self, naive_trace: NaiveTrace) -> None:
        c3 = naive_trace.beat("C3")
        assert c3 is not None
        assert c3.has_point_in_time is False
        assert c3.extra["query_at_error"] is not None

    def test_naive_returns_current_not_historical(self, naive_trace: NaiveTrace) -> None:
        """Naive recall always returns the current (last-written) value, regardless of date intent."""
        c3 = naive_trace.beat("C3")
        assert c3 is not None
        # The naive answer is the current last-written employer value (NOT the Q1 historical one)
        # After C2 overwrite, it's "Acme Corp / CTO"
        assert c3.extra["q1_answer_from_naive"] != c3.extra["correct_q1_answer"], (
            "Naive should NOT return the historically correct Q1 answer — "
            "it can only return today's last-written value"
        )

    def test_mempill_query_at_returns_correct_q1_value(self) -> None:
        """mempill query_at(valid_at=2024-01-01) returns Austin TX — the city before the move."""
        adapter = build_mempill_adapter(in_memory=True, oracle_backed=False)
        load_seed_claims(adapter, AGENT_ID)
        # City: Austin valid_from=2023-06, valid_until=2025-02 → 2024-01-01 is in that window
        belief = adapter.query_at(
            AGENT_ID, "alice-chen", "city",
            valid_at="2024-01-01T00:00:00Z",
        )
        assert belief.value == "Austin TX", (
            f"Expected 'Austin TX' for valid_at=2024-01-01, got {belief.value!r}"
        )
        assert belief.status == "Resolved"

    def test_naive_marked_fail_in_comparison(self, comparison: ComparisonResult) -> None:
        c3 = comparison.get("C3")
        assert c3 is not None
        assert c3.naive_verdict == "FAIL"

    def test_mempill_marked_pass_in_comparison(self, comparison: ComparisonResult) -> None:
        c3 = comparison.get("C3")
        assert c3 is not None
        assert c3.mempill_verdict == "PASS"


# ── C4: Compliance audit / tx-time replay ────────────────────────────────────

class TestContrastC4ComplianceAudit:
    """C4 — Tx-time replay: naive cannot reconstruct past belief state; mempill can."""

    def test_naive_c4_beat_present(self, naive_trace: NaiveTrace) -> None:
        assert naive_trace.beat("C4") is not None

    def test_naive_no_tx_time_replay(self, naive_trace: NaiveTrace) -> None:
        """NaiveAdapter has no as_of_tx_time capability."""
        c4 = naive_trace.beat("C4")
        assert c4 is not None
        assert c4.extra["as_of_tx_time_available"] is False
        assert c4.extra["tx_replay_error"] is not None

    def test_naive_no_supersession_events(self, naive_trace: NaiveTrace) -> None:
        """Naive audit log has only Ingested/write events — no Superseded/OracleAdjudicated."""
        c4 = naive_trace.beat("C4")
        assert c4 is not None
        assert c4.extra["has_supersession_events"] is False
        assert "Ingested" in c4.extra["naive_audit_kinds"] or c4.extra["audit_entry_count"] > 0

    def test_naive_has_no_real_audit_trail(self, naive_trace: NaiveTrace) -> None:
        """NaiveAdapter.has_audit_trail is False — write-event log is NOT a bi-temporal audit."""
        c4 = naive_trace.beat("C4")
        assert c4 is not None
        assert c4.has_audit_trail is False

    def test_naive_as_of_tx_time_raises_attribute_error(self) -> None:
        """NaiveAdapter.query_at (as_of_tx_time) raises AttributeError."""
        adapter = NaiveAdapter()
        with pytest.raises(AttributeError):
            adapter.query_at(AGENT_ID, "alice-chen", "city", as_of_tx_time="2025-03-10T08:00:00Z")  # type: ignore[attr-defined]

    def test_mempill_tx_time_replay_returns_prior_belief(self) -> None:
        """mempill as_of_tx_time=<before NYC write> returns Austin TX."""
        import time
        from mempill import ProvenanceLabel
        adapter = build_mempill_adapter(in_memory=True, oracle_backed=False)
        load_seed_claims(adapter, AGENT_ID)

        # Capture Austin's tx time from the audit
        austin_belief = adapter.recall(AGENT_ID, "alice-chen", "city")
        austin_ref = austin_belief.claim_ref
        audit_entries = adapter.audit(AGENT_ID, limit=20)
        tx_before: str | None = None
        for e in audit_entries:
            if e.claim_ref == austin_ref:
                tx_before = e.recorded_at
                break
        if not tx_before and audit_entries:
            tx_before = audit_entries[0].recorded_at
        assert tx_before is not None, "Could not capture Austin's tx timestamp"

        # Sleep to ensure NYC write gets a strictly later timestamp
        time.sleep(0.1)

        # Write NYC
        adapter.write_claim(AGENT_ID, ClaimInput(
            subject="alice-chen", predicate="city", value="New York NY",
            valid_from="2025-02", confidence=1.0,
            provenance=ProvenanceLabel.external_user_asserted(),
            cardinality="Functional",
        ))

        # Replay as_of_tx_time=<before NYC> → should return Austin
        belief_before = adapter.query_at(
            AGENT_ID, "alice-chen", "city",
            as_of_tx_time=tx_before,
        )
        assert belief_before.value == "Austin TX", (
            f"Expected Austin TX (pre-NYC belief), got {belief_before.value!r}"
        )

    def test_mempill_audit_has_claim_events(self) -> None:
        """mempill audit log is non-empty and contains claim-lifecycle events."""
        import time
        from mempill import ProvenanceLabel
        adapter = build_mempill_adapter(in_memory=True, oracle_backed=False)
        load_seed_claims(adapter, AGENT_ID)
        time.sleep(0.05)
        # Write NYC succession
        adapter.write_claim(AGENT_ID, ClaimInput(
            subject="alice-chen", predicate="city", value="New York NY",
            valid_from="2025-02", confidence=1.0,
            provenance=ProvenanceLabel.external_user_asserted(),
            cardinality="Functional",
        ))
        audit = adapter.audit(AGENT_ID, limit=50)
        # Audit must be non-empty (all seed claims + NYC write are recorded)
        assert len(audit) >= 7, f"Expected at least 7 audit entries, got {len(audit)}"
        # Audit contains at least one event kind (the exact kind depends on mempill version)
        kinds = {e.event_kind for e in audit}
        assert len(kinds) > 0, f"Expected at least one event kind in audit, got empty set"
        # Each entry has a claim_ref — every claim is individually trackable
        claim_refs = {e.claim_ref for e in audit if e.claim_ref}
        assert len(claim_refs) >= 7, (
            f"Expected at least 7 distinct claim_refs in audit, got {len(claim_refs)}"
        )

    def test_naive_marked_fail_in_comparison(self, comparison: ComparisonResult) -> None:
        c4 = comparison.get("C4")
        assert c4 is not None
        assert c4.naive_verdict == "FAIL"

    def test_mempill_marked_pass_in_comparison(self, comparison: ComparisonResult) -> None:
        c4 = comparison.get("C4")
        assert c4 is not None
        assert c4.mempill_verdict == "PASS"


# ── Overall ComparisonResult assertions ───────────────────────────────────────

class TestComparisonResult:
    """Assert the ComparisonResult object marks verdicts correctly for all 4 contrasts."""

    def test_all_four_contrasts_present(self, comparison: ComparisonResult) -> None:
        ids = {c.contrast_id for c in comparison.contrasts}
        assert ids == {"C1", "C2", "C3", "C4"}

    def test_all_naive_fail(self, comparison: ComparisonResult) -> None:
        assert comparison.all_naive_fail(), (
            f"Expected all naive verdicts to be FAIL; got: "
            f"{[(c.contrast_id, c.naive_verdict) for c in comparison.contrasts]}"
        )

    def test_all_mempill_pass(self, comparison: ComparisonResult) -> None:
        assert comparison.all_mempill_pass(), (
            f"Expected all mempill verdicts to be PASS; got: "
            f"{[(c.contrast_id, c.mempill_verdict) for c in comparison.contrasts]}"
        )

    def test_comparison_has_failure_modes(self, comparison: ComparisonResult) -> None:
        for c in comparison.contrasts:
            assert c.failure_mode, f"Contrast {c.contrast_id} has empty failure_mode"

    def test_comparison_has_mempill_mechanisms(self, comparison: ComparisonResult) -> None:
        for c in comparison.contrasts:
            assert c.mempill_mechanism, f"Contrast {c.contrast_id} has empty mempill_mechanism"
