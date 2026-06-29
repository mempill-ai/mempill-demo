"""
mempill_showcase.tests.test_core_adapter — W1 contract tests.

No API key, no LLM, no external services required.
Uses a real in-memory mempill engine via the MempillAdapter.

Done-when assertions (6):
  1. seed Day-0 claims → recall(alice-chen, city) == "Austin TX" (Resolved)
  2. ingest city="New York NY" valid_from="2025-02" → CommittedCheap
  3. recall(alice-chen, city) == "New York NY" (succession: prior Austin superseded)
  4. query_at(alice-chen, city, valid_at="2025-01-01T00:00:00Z") == "Austin TX" (bi-temporal)
  5. canonical_keys.resolve_entity("Alice Chen") == "alice-chen"
  6. naive adapter: after overwrite, recall returns latest; confirm it is NOT a BiTemporalMemoryStore
     and query_at raises AttributeError (no bi-temporal support)
"""
from __future__ import annotations

import pytest

from mempill import ProvenanceLabel
from mempill_showcase.adapters.memory.mempill_adapter import MempillAdapter
from mempill_showcase.adapters.memory.naive_adapter import NaiveAdapter
from mempill_showcase.config.di import build_mempill_adapter, build_naive_adapter
from mempill_showcase.core.domain.canonical_keys import resolve_entity, resolve_predicate
from mempill_showcase.core.domain.models import ClaimInput
from mempill_showcase.core.ports.memory import BiTemporalMemoryStore
from mempill_showcase.scenarios.seed_data import AGENT_ID, load_seed_claims


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def mempill_adapter() -> MempillAdapter:
    """Fresh in-memory mempill adapter per test."""
    return build_mempill_adapter(in_memory=True)


@pytest.fixture()
def seeded_adapter() -> MempillAdapter:
    """In-memory mempill adapter with Day-0 seed claims loaded."""
    adapter = build_mempill_adapter(in_memory=True)
    load_seed_claims(adapter, agent_id=AGENT_ID)
    return adapter


@pytest.fixture()
def naive_adapter() -> NaiveAdapter:
    """Fresh naive adapter per test."""
    return build_naive_adapter()


# ── Test 1: Seed → recall current city ───────────────────────────────────────

def test_seed_recall_city_resolved(seeded_adapter: MempillAdapter) -> None:
    """After seeding Day-0 claims, alice-chen/city == Austin TX (Resolved)."""
    result = seeded_adapter.recall(AGENT_ID, "alice-chen", "city")
    assert result.status == "Resolved", f"Expected Resolved, got {result.status}"
    assert result.value == "Austin TX", f"Expected 'Austin TX', got {result.value!r}"


# ── Test 2: Ingest succession → CommittedCheap ───────────────────────────────

def test_ingest_city_succession_committed(seeded_adapter: MempillAdapter) -> None:
    """Writing alice-chen/city = 'New York NY' valid_from=2025-02 returns CommittedCheap.

    The prior 'Austin TX' claim (valid_from=2023-06) does not overlap with the
    new 2025-02 start (the engine applies succession / end-bounding).
    """
    claim = ClaimInput(
        subject="alice-chen",
        predicate="city",
        value="New York NY",
        valid_from="2025-02",
        confidence=1.0,
        provenance=ProvenanceLabel.external_user_asserted(),
    )
    receipt = seeded_adapter.write_claim(AGENT_ID, claim)
    assert receipt.disposition == "CommittedCheap", (
        f"Expected CommittedCheap, got {receipt.disposition!r} "
        f"(contested_with={receipt.contested_with})"
    )
    assert receipt.claim_ref, "Expected a non-empty claim_ref UUID"


# ── Test 3: After succession, current recall == NYC ──────────────────────────

def test_succession_current_recall_is_new_value(seeded_adapter: MempillAdapter) -> None:
    """After writing NYC valid_from=2025-02, recall returns NYC (succession applied)."""
    claim = ClaimInput(
        subject="alice-chen",
        predicate="city",
        value="New York NY",
        valid_from="2025-02",
        confidence=1.0,
        provenance=ProvenanceLabel.external_user_asserted(),
    )
    seeded_adapter.write_claim(AGENT_ID, claim)

    result = seeded_adapter.recall(AGENT_ID, "alice-chen", "city")
    assert result.status == "Resolved", f"Expected Resolved, got {result.status}"
    assert result.value == "New York NY", (
        f"Expected 'New York NY' after succession, got {result.value!r}"
    )


# ── Test 4: Bi-temporal query_at (valid_at in the past) → Austin TX ──────────

def test_bitemoral_query_at_past_date_returns_prior_value(seeded_adapter: MempillAdapter) -> None:
    """query_at(valid_at=2025-01-01) returns Austin TX (world-valid on that date).

    Setup:
      Austin TX  valid_from=2023-06  (ingested at Day-0 seed)
      NYC        valid_from=2025-02  (ingested here; supersedes Austin from 2025-02)

    On 2025-01-01, NYC was not yet true → engine must return Austin TX.
    """
    # First ingest the succession so both claims exist
    claim = ClaimInput(
        subject="alice-chen",
        predicate="city",
        value="New York NY",
        valid_from="2025-02",
        confidence=1.0,
        provenance=ProvenanceLabel.external_user_asserted(),
    )
    seeded_adapter.write_claim(AGENT_ID, claim)

    # Now query at a date before the move
    result = seeded_adapter.query_at(
        AGENT_ID,
        "alice-chen",
        "city",
        valid_at="2025-01-01T00:00:00Z",
    )
    assert result.status == "Resolved", (
        f"Expected Resolved for valid_at=2025-01-01, got {result.status}"
    )
    assert result.value == "Austin TX", (
        f"Bi-temporal query should return 'Austin TX' for 2025-01-01, got {result.value!r}"
    )


# ── Test 5: canonical_keys.resolve_entity ────────────────────────────────────

def test_canonical_resolve_entity_alice_chen() -> None:
    """resolve_entity('Alice Chen') == 'alice-chen'."""
    assert resolve_entity("Alice Chen") == "alice-chen"
    # Also verify common variants
    assert resolve_entity("alice") == "alice-chen"
    assert resolve_entity("Alice") == "alice-chen"
    assert resolve_entity("alice-chen") == "alice-chen"


def test_canonical_resolve_entity_other_entities() -> None:
    """All scenario entities resolve correctly."""
    assert resolve_entity("Bob Liu") == "bob-liu"
    assert resolve_entity("Acme Corp") == "acme-corp"
    assert resolve_entity("Jordan Park") == "jordan-park"


def test_canonical_resolve_predicate() -> None:
    """Common predicate aliases resolve correctly."""
    assert resolve_predicate("city") == "city"
    assert resolve_predicate("employer") == "employer"
    assert resolve_predicate("dietary restriction") == "dietary_restriction"


def test_canonical_unknown_returns_none() -> None:
    """Unknown entity returns None (never invents a key)."""
    assert resolve_entity("Completely Unknown Person XYZ") is None
    assert resolve_predicate("invented_field_xyz") is None


# ── Test 6: Naive adapter — overwrite + no bi-temporal ───────────────────────

def test_naive_overwrite_returns_latest(naive_adapter: NaiveAdapter) -> None:
    """Naive adapter: after second write, recall returns the newer value."""
    claim_austin = ClaimInput(
        subject="alice-chen",
        predicate="city",
        value="Austin TX",
        valid_from="2023-06",
    )
    claim_nyc = ClaimInput(
        subject="alice-chen",
        predicate="city",
        value="New York NY",
        valid_from="2025-02",
    )
    naive_adapter.write_claim(AGENT_ID, claim_austin)
    naive_adapter.write_claim(AGENT_ID, claim_nyc)

    result = naive_adapter.recall(AGENT_ID, "alice-chen", "city")
    assert result.status == "Resolved"
    assert result.value == "New York NY", (
        f"Naive adapter should return most-recent write, got {result.value!r}"
    )


def test_naive_is_not_bitemporal(naive_adapter: NaiveAdapter) -> None:
    """NaiveAdapter does NOT implement BiTemporalMemoryStore.

    Verified two ways:
      1. isinstance check fails (structural subtype — NaiveAdapter lacks query_at)
      2. Calling query_at raises AttributeError
    This is the intentional contrast seam between the two adapters.
    """
    # isinstance check: NaiveAdapter intentionally does NOT have query_at()
    # Python's Protocol structural check requires runtime_checkable; we use
    # hasattr as the definitive test instead (consistent with the seam design)
    assert not hasattr(naive_adapter, "query_at"), (
        "NaiveAdapter must NOT have query_at — it is not a BiTemporalMemoryStore"
    )

    # Calling it raises AttributeError
    with pytest.raises(AttributeError):
        _ = naive_adapter.query_at(  # type: ignore[attr-defined]
            AGENT_ID, "alice-chen", "city", valid_at="2025-01-01T00:00:00Z"
        )


def test_mempill_adapter_is_bitemporal(mempill_adapter: MempillAdapter) -> None:
    """MempillAdapter has query_at (is a BiTemporalMemoryStore)."""
    assert hasattr(mempill_adapter, "query_at"), (
        "MempillAdapter must have query_at — it IS a BiTemporalMemoryStore"
    )
