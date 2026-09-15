"""
mempill_showcase.tests.test_end_fact_adapter — MempillAdapter.end_fact() /
resolve_live_claim_for_line() unit tests, TASK-33-W4-DEMO.

Uses a FAKE engine (duck-typed, no real mempill Rust engine) so these tests
exercise ONLY the adapter's wiring — request/response translation, provenance
defaulting, and exception passthrough — independent of the real engine's
resolution/persistence logic (that's covered by mempill-python's own
test_end_fact.py in the engine repo).

Covers:
  - resolve_live_claim_for_line(): pure passthrough of the engine's dict.
  - end_fact(): 0 live -> mempill.NotFoundError; 1 live -> bounds and returns
    EndFactResult; >1 live -> mempill.ValidationError (never guesses).
  - end_fact() forwards provenance/confidence/at correctly; defaults provenance
    to External/UserAsserted when the caller supplies none.
  - Neither error case calls engine.assert_validity() — resolution alone
    determines the outcome for 0/>1 live (mempill.ergonomic.end_fact's contract).
"""
from __future__ import annotations

import pytest

import mempill
from mempill_showcase.adapters.memory.mempill_adapter import MempillAdapter
from mempill_showcase.core.domain.models import EndFactResult

AGENT_ID = "fake-agent-001"


class FakeEngine:
    """Duck-typed fake satisfying the subset of the mempill.Engine surface
    mempill.ergonomic.end_fact() calls: resolve_live_claim_for_line() +
    assert_validity()."""

    def __init__(self, resolution: dict, assert_validity_response: dict | None = None):
        self._resolution = resolution
        self._response = assert_validity_response
        self.assert_validity_calls: list[dict] = []
        self.resolve_calls: list[tuple] = []

    def resolve_live_claim_for_line(self, agent_id: str, subject: str, predicate: str) -> dict:
        self.resolve_calls.append((agent_id, subject, predicate))
        return self._resolution

    def assert_validity(self, request: dict) -> dict:
        self.assert_validity_calls.append(request)
        return self._response


# ── resolve_live_claim_for_line() — pure passthrough ────────────────────────

class TestResolveLiveClaimForLinePassthrough:
    def test_passes_through_empty(self) -> None:
        fake = FakeEngine({"status": "empty", "claim_ref": None, "live_count": None})
        adapter = MempillAdapter(fake)
        resolution = adapter.resolve_live_claim_for_line(AGENT_ID, "alice-chen", "city")
        assert resolution == {"status": "empty", "claim_ref": None, "live_count": None}
        assert fake.resolve_calls == [(AGENT_ID, "alice-chen", "city")]

    def test_passes_through_single(self) -> None:
        fake = FakeEngine({"status": "single", "claim_ref": "claim-1", "live_count": None})
        adapter = MempillAdapter(fake)
        resolution = adapter.resolve_live_claim_for_line(AGENT_ID, "alice-chen", "city")
        assert resolution["status"] == "single"
        assert resolution["claim_ref"] == "claim-1"

    def test_passes_through_ambiguous(self) -> None:
        fake = FakeEngine({"status": "ambiguous", "claim_ref": None, "live_count": 3})
        adapter = MempillAdapter(fake)
        resolution = adapter.resolve_live_claim_for_line(AGENT_ID, "alice-chen", "city")
        assert resolution["status"] == "ambiguous"
        assert resolution["live_count"] == 3


# ── end_fact() — 0 live claims ───────────────────────────────────────────────

class TestEndFactEmptyLine:
    def test_raises_not_found_error(self) -> None:
        fake = FakeEngine({"status": "empty", "claim_ref": None, "live_count": None})
        adapter = MempillAdapter(fake)
        with pytest.raises(mempill.NotFoundError):
            adapter.end_fact(AGENT_ID, "alice-chen", "city", at="2024-01")

    def test_never_calls_assert_validity(self) -> None:
        fake = FakeEngine({"status": "empty", "claim_ref": None, "live_count": None})
        adapter = MempillAdapter(fake)
        with pytest.raises(mempill.NotFoundError):
            adapter.end_fact(AGENT_ID, "alice-chen", "city", at="2024-01")
        assert fake.assert_validity_calls == []


# ── end_fact() — >1 live claims (ambiguous — never guess) ──────────────────

class TestEndFactAmbiguousLine:
    def test_raises_validation_error(self) -> None:
        fake = FakeEngine({"status": "ambiguous", "claim_ref": None, "live_count": 2})
        adapter = MempillAdapter(fake)
        with pytest.raises(mempill.ValidationError):
            adapter.end_fact(AGENT_ID, "alice-chen", "city", at="2024-01")

    def test_never_calls_assert_validity(self) -> None:
        fake = FakeEngine({"status": "ambiguous", "claim_ref": None, "live_count": 2})
        adapter = MempillAdapter(fake)
        with pytest.raises(mempill.ValidationError):
            adapter.end_fact(AGENT_ID, "alice-chen", "city", at="2024-01")
        assert fake.assert_validity_calls == []


# ── end_fact() — 1 live claim (the common case) ─────────────────────────────

class TestEndFactSingleLiveClaim:
    def _fake(self) -> FakeEngine:
        return FakeEngine(
            resolution={"status": "single", "claim_ref": "claim-abc", "live_count": None},
            assert_validity_response={
                "claim_ref": "claim-abc",
                "disposition": "Superseded",
                "effective_at": "2024-01-01T00:00:00Z",
                "no_op": False,
            },
        )

    def test_returns_end_fact_result(self) -> None:
        fake = self._fake()
        adapter = MempillAdapter(fake)
        result = adapter.end_fact(AGENT_ID, "alice-chen", "city", at="2024-01")
        assert isinstance(result, EndFactResult)
        assert result.claim_ref == "claim-abc"
        assert result.disposition == "Superseded"
        assert result.effective_at == "2024-01-01T00:00:00Z"
        assert result.no_op is False

    def test_targets_the_resolved_claim_ref(self) -> None:
        fake = self._fake()
        adapter = MempillAdapter(fake)
        adapter.end_fact(AGENT_ID, "alice-chen", "city", at="2024-01")
        assert len(fake.assert_validity_calls) == 1
        request = fake.assert_validity_calls[0]
        assert request["target"] == "claim-abc"
        assert request["agent_id"] == AGENT_ID
        assert request["assertion"]["type"] == "Bound"

    def test_defaults_provenance_to_external_user_asserted(self) -> None:
        fake = self._fake()
        adapter = MempillAdapter(fake)
        adapter.end_fact(AGENT_ID, "alice-chen", "city", at="2024-01")
        request = fake.assert_validity_calls[0]
        assert request["provenance"]["type"] == "External"
        assert request["provenance"]["kind"] == "UserAsserted"

    def test_forwards_custom_provenance_and_confidence(self) -> None:
        fake = self._fake()
        adapter = MempillAdapter(fake)
        custom_prov = {"type": "External", "kind": "ExternalFirstHand"}
        adapter.end_fact(
            AGENT_ID, "alice-chen", "city", at="2024-01",
            provenance=custom_prov, confidence=0.8,
        )
        request = fake.assert_validity_calls[0]
        assert request["provenance"] == custom_prov
        assert request["confidence"]["value_confidence"] == 0.8
        assert request["confidence"]["valid_time_confidence"] == 0.8

    def test_no_op_repeat_surfaces_on_result(self) -> None:
        fake = FakeEngine(
            resolution={"status": "single", "claim_ref": "claim-abc", "live_count": None},
            assert_validity_response={
                "claim_ref": "claim-abc",
                "disposition": "Superseded",
                "effective_at": "2024-01-01T00:00:00Z",
                "no_op": True,
            },
        )
        adapter = MempillAdapter(fake)
        result = adapter.end_fact(AGENT_ID, "alice-chen", "city", at="2024-01")
        assert result.no_op is True
