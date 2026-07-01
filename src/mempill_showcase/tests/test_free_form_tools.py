"""
mempill_showcase.tests.test_free_form_tools — Wave-A free-form memory tool tests.

No API key, no LLM, no external services. Uses a real in-memory mempill engine
via MempillAdapter (same pattern as test_core_adapter and test_tools).

Covers:
  A1. adapter.query_subject — thin wrapper over engine.query_subject
        - returns ALL predicates seeded for a subject
        - valid_at narrows to historical window
        - as_of_tx_time filters to known claims at that tx time
        - free-form predicate ("favorite-color") surfaces with recall_subject

  A2. RecallSubjectTool
        - returns all 3 predicates for seeded subject (including free-form)
        - fact_count and facts structure match
        - soft subject normalisation ("Alice Chen" → "alice-chen")
        - empty subject → fact_count == 0

  A3. RecallAtTool
        - returns historical value when valid_at is in incumbent's window
        - returns current value when valid_at is after succession

  A4. RecallAsOfTool
        - returns NoBelief as_of before any write (early tx time)
        - returns current value as_of a far-future tx time

  A5. RememberFactTool (free-form, NO canonical guard)
        - writes a free-form predicate ("favorite-color") without error
        - recall_subject then surfaces it
        - subject + predicate soft normalisation verified
        - succession: write Austin then NYC via RememberFactTool → current == NYC
        - bi-temporal correctness preserved: query_at(Austin window) == Austin

  A6. GetContestedTool
        - Contested belief: incumbent + challenger populated
        - Resolved belief: alternatives empty, is_contested == False
        - NoBelief: is_contested == False

  A7. AuditTrailTool
        - returns entries after writes (entry_count >= n_writes)
        - claim_ref filter works
        - empty on fresh adapter
"""
from __future__ import annotations

import json

import pytest

from mempill import ProvenanceLabel
from mempill_showcase.adapters.memory.mempill_adapter import MempillAdapter
from mempill_showcase.config.di import build_mempill_adapter
from mempill_showcase.core.domain.models import ClaimInput
from mempill_showcase.scenarios.seed_data import AGENT_ID
from mempill_showcase.tools.audit_trail_tool import AuditTrailTool
from mempill_showcase.tools.get_contested_tool import GetContestedTool
from mempill_showcase.tools.recall_as_of_tool import RecallAsOfTool
from mempill_showcase.tools.recall_at_tool import RecallAtTool
from mempill_showcase.tools.recall_subject_tool import RecallSubjectTool
from mempill_showcase.tools.remember_fact_tool import RememberFactTool


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def adapter() -> MempillAdapter:
    """Fresh in-memory mempill adapter per test (oracle-backed for Contested support)."""
    return build_mempill_adapter(in_memory=True, oracle_backed=True)


@pytest.fixture()
def recall_subject_tool(adapter: MempillAdapter) -> RecallSubjectTool:
    return RecallSubjectTool(adapter=adapter)


@pytest.fixture()
def recall_at_tool(adapter: MempillAdapter) -> RecallAtTool:
    return RecallAtTool(adapter=adapter)


@pytest.fixture()
def remember_fact_tool_for_at(adapter: MempillAdapter) -> RememberFactTool:
    """Separate remember_fact_tool sharing the same adapter as recall_at_tool fixture."""
    return RememberFactTool(adapter=adapter)


@pytest.fixture()
def recall_as_of_tool(adapter: MempillAdapter) -> RecallAsOfTool:
    return RecallAsOfTool(adapter=adapter)


@pytest.fixture()
def remember_fact_tool(adapter: MempillAdapter) -> RememberFactTool:
    return RememberFactTool(adapter=adapter)


@pytest.fixture()
def get_contested_tool(adapter: MempillAdapter) -> GetContestedTool:
    return GetContestedTool(adapter=adapter)


@pytest.fixture()
def audit_trail_tool(adapter: MempillAdapter) -> AuditTrailTool:
    return AuditTrailTool(adapter=adapter)


def _seed_claims(adapter: MempillAdapter, agent_id: str = AGENT_ID) -> None:
    """Seed alice-chen with 3 predicates including a free-form one."""
    for subject, predicate, value, valid_from in [
        ("alice-chen", "city",           "Austin TX",                  "2023-06"),
        ("alice-chen", "employer",       "Acme Corp / VP Engineering", "2023-06"),
        ("alice-chen", "favorite-color", "deep-blue",                  "2023-01"),
    ]:
        adapter.write_claim(agent_id, ClaimInput(
            subject=subject,
            predicate=predicate,
            value=value,
            valid_from=valid_from,
            confidence=1.0,
            provenance=ProvenanceLabel.external_user_asserted(),
        ))


# ── A1: adapter.query_subject ─────────────────────────────────────────────────

class TestAdapterQuerySubject:
    """A1 — MempillAdapter.query_subject thin wrapper tests."""

    def test_query_subject_returns_all_predicates(self, adapter: MempillAdapter) -> None:
        """query_subject returns ALL predicates seeded for alice-chen (including free-form)."""
        _seed_claims(adapter)

        result = adapter.query_subject(AGENT_ID, "alice-chen")
        predicates = {f["predicate"] for f in result}

        assert "city" in predicates, f"Expected 'city' in predicates, got {predicates}"
        assert "employer" in predicates, f"Expected 'employer' in predicates, got {predicates}"
        assert "favorite-color" in predicates, (
            f"Expected free-form 'favorite-color' in predicates, got {predicates}"
        )
        assert len(result) == 3, f"Expected 3 facts, got {len(result)}"

    def test_query_subject_valid_at_narrows(self, adapter: MempillAdapter) -> None:
        """query_subject valid_at returns facts valid at that date.

        Uses RememberFactTool (via adapter) to write with proper succession,
        then queries at a date before the move to verify the old value returns.
        """
        # Use RememberFactTool to correctly handle succession close-step
        rft = RememberFactTool(adapter=adapter)
        rft.invoke({
            "agent_id": AGENT_ID,
            "subject": "alice-chen",
            "predicate": "city",
            "value": "Austin TX",
            "valid_from": "2023-06",
        })
        rft.invoke({
            "agent_id": AGENT_ID,
            "subject": "alice-chen",
            "predicate": "city",
            "value": "New York NY",
            "valid_from": "2025-02",
        })

        # Query at 2024-01-01 — still Austin
        result = adapter.query_subject(AGENT_ID, "alice-chen", valid_at="2024-01-01T00:00:00Z")
        city_facts = [f for f in result if f["predicate"] == "city"]
        assert city_facts, "Expected at least one city fact for valid_at=2024-01-01"
        assert city_facts[0]["value"] == "Austin TX", (
            f"Expected 'Austin TX' for valid_at=2024-01-01, got {city_facts[0]['value']!r}"
        )

    def test_query_subject_as_of_tx_time_far_future(self, adapter: MempillAdapter) -> None:
        """query_subject as_of_tx_time far-future returns current facts."""
        _seed_claims(adapter)

        result = adapter.query_subject(AGENT_ID, "alice-chen", as_of_tx_time="2099-01-01T00:00:00Z")
        assert len(result) >= 1, "Far-future as_of_tx_time should return current facts"
        predicates = {f["predicate"] for f in result}
        assert "city" in predicates

    def test_query_subject_free_form_predicate_surfaces(self, adapter: MempillAdapter) -> None:
        """Free-form predicate 'favorite-color' appears in query_subject result."""
        _seed_claims(adapter)

        result = adapter.query_subject(AGENT_ID, "alice-chen")
        color_facts = [f for f in result if f["predicate"] == "favorite-color"]
        assert color_facts, "Expected free-form 'favorite-color' fact in query_subject result"
        assert color_facts[0]["value"] == "deep-blue"
        assert color_facts[0]["status"] == "Resolved"

    def test_query_subject_unknown_returns_empty(self, adapter: MempillAdapter) -> None:
        """Unknown subject returns empty list."""
        result = adapter.query_subject(AGENT_ID, "nobody-xyz-123")
        assert result == [], f"Expected [] for unknown subject, got {result}"


# ── A2: RecallSubjectTool ─────────────────────────────────────────────────────

class TestRecallSubjectTool:
    """A2 — RecallSubjectTool returns all predicates via one call."""

    def test_returns_all_three_predicates(self, recall_subject_tool: RecallSubjectTool, adapter: MempillAdapter) -> None:
        """recall_subject returns all 3 seeded predicates including free-form."""
        _seed_claims(adapter)

        raw = recall_subject_tool.invoke({"agent_id": AGENT_ID, "subject": "alice-chen"})
        result = json.loads(raw)

        assert result["subject"] == "alice-chen"
        assert result["fact_count"] == 3, (
            f"Expected 3 facts, got {result['fact_count']}: {result['facts']}"
        )
        predicates = {f["predicate"] for f in result["facts"]}
        assert "city" in predicates
        assert "employer" in predicates
        assert "favorite-color" in predicates

    def test_fact_structure_contains_required_keys(self, recall_subject_tool: RecallSubjectTool, adapter: MempillAdapter) -> None:
        """Each fact dict contains all required keys."""
        _seed_claims(adapter)

        raw = recall_subject_tool.invoke({"agent_id": AGENT_ID, "subject": "alice-chen"})
        result = json.loads(raw)

        required = {"predicate", "value", "status", "is_contested",
                    "valid_from_display", "valid_until_display", "provenance",
                    "claim_ref", "conf"}
        for fact in result["facts"]:
            missing = required - set(fact.keys())
            assert not missing, f"Fact missing keys {missing}: {fact}"

    def test_soft_subject_normalisation(self, recall_subject_tool: RecallSubjectTool, adapter: MempillAdapter) -> None:
        """'Alice Chen' (raw) is normalised to 'alice-chen' and finds facts."""
        _seed_claims(adapter)

        raw = recall_subject_tool.invoke({"agent_id": AGENT_ID, "subject": "Alice Chen"})
        result = json.loads(raw)

        assert result["subject"] == "alice-chen", (
            f"subject in result should be normalised 'alice-chen', got {result['subject']!r}"
        )
        assert result["fact_count"] == 3

    def test_unknown_subject_returns_empty_facts(self, recall_subject_tool: RecallSubjectTool) -> None:
        """Unknown subject returns fact_count == 0."""
        raw = recall_subject_tool.invoke({"agent_id": AGENT_ID, "subject": "nobody-xyz"})
        result = json.loads(raw)
        assert result["fact_count"] == 0
        assert result["facts"] == []

    def test_free_form_predicate_value_correct(self, recall_subject_tool: RecallSubjectTool, adapter: MempillAdapter) -> None:
        """Free-form 'favorite-color' value is 'deep-blue' and status is Resolved."""
        _seed_claims(adapter)

        raw = recall_subject_tool.invoke({"agent_id": AGENT_ID, "subject": "alice-chen"})
        result = json.loads(raw)

        color_facts = [f for f in result["facts"] if f["predicate"] == "favorite-color"]
        assert color_facts, "Expected 'favorite-color' in facts"
        assert color_facts[0]["value"] == "deep-blue"
        assert color_facts[0]["status"] == "Resolved"
        assert not color_facts[0]["is_contested"]


# ── A3: RecallAtTool ──────────────────────────────────────────────────────────

class TestRecallAtTool:
    """A3 — RecallAtTool: valid-time point-in-time queries."""

    def test_recall_at_returns_historical_value(
        self,
        recall_at_tool: RecallAtTool,
        remember_fact_tool_for_at: RememberFactTool,
    ) -> None:
        """query valid_at=2024-01-01 returns Austin TX (before NYC succession).

        Uses RememberFactTool for writes so the succession close-step is applied
        (same pattern as TestRememberToolSuccession in test_tools.py).
        """
        remember_fact_tool_for_at.invoke({
            "agent_id": AGENT_ID,
            "subject": "alice-chen",
            "predicate": "city",
            "value": "Austin TX",
            "valid_from": "2023-06",
        })
        remember_fact_tool_for_at.invoke({
            "agent_id": AGENT_ID,
            "subject": "alice-chen",
            "predicate": "city",
            "value": "New York NY",
            "valid_from": "2025-02",
        })

        raw = recall_at_tool.invoke({
            "agent_id": AGENT_ID,
            "subject": "alice-chen",
            "predicate": "city",
            "valid_at": "2024-01-01T00:00:00Z",
        })
        result = json.loads(raw)

        assert result["status"] == "Resolved", f"Expected Resolved, got {result['status']!r}"
        assert result["value"] == "Austin TX", (
            f"Expected 'Austin TX' at valid_at=2024-01-01, got {result['value']!r}"
        )
        assert not result["is_contested"]

    def test_recall_at_returns_current_after_succession(
        self,
        recall_at_tool: RecallAtTool,
        remember_fact_tool_for_at: RememberFactTool,
    ) -> None:
        """query valid_at=2026-01-01 returns NYC (post-succession)."""
        remember_fact_tool_for_at.invoke({
            "agent_id": AGENT_ID,
            "subject": "alice-chen",
            "predicate": "city",
            "value": "Austin TX",
            "valid_from": "2023-06",
        })
        remember_fact_tool_for_at.invoke({
            "agent_id": AGENT_ID,
            "subject": "alice-chen",
            "predicate": "city",
            "value": "New York NY",
            "valid_from": "2025-02",
        })

        raw = recall_at_tool.invoke({
            "agent_id": AGENT_ID,
            "subject": "alice-chen",
            "predicate": "city",
            "valid_at": "2026-01-01T00:00:00Z",
        })
        result = json.loads(raw)
        assert result["value"] == "New York NY", (
            f"Expected 'New York NY' at valid_at=2026-01-01, got {result['value']!r}"
        )

    def test_recall_at_result_keys(self, recall_at_tool: RecallAtTool, remember_fact_tool_for_at: RememberFactTool) -> None:
        """RecallAtTool result has all required keys."""
        remember_fact_tool_for_at.invoke({
            "agent_id": AGENT_ID,
            "subject": "alice-chen",
            "predicate": "city",
            "value": "Austin TX",
            "valid_from": "2023-06",
        })
        raw = recall_at_tool.invoke({
            "agent_id": AGENT_ID,
            "subject": "alice-chen",
            "predicate": "city",
            "valid_at": "2023-12-01T00:00:00Z",
        })
        result = json.loads(raw)
        required = {"subject", "predicate", "value", "status", "is_contested",
                    "conf", "valid_from_display", "valid_until_display",
                    "provenance", "claim_ref"}
        missing = required - set(result.keys())
        assert not missing, f"Missing keys: {missing}"


# ── A4: RecallAsOfTool ────────────────────────────────────────────────────────

class TestRecallAsOfTool:
    """A4 — RecallAsOfTool: transaction-time as-of queries."""

    def test_recall_as_of_far_future_returns_current(
        self,
        recall_as_of_tool: RecallAsOfTool,
        adapter: MempillAdapter,
    ) -> None:
        """as_of far-future returns the current belief."""
        _seed_claims(adapter)

        raw = recall_as_of_tool.invoke({
            "agent_id": AGENT_ID,
            "subject": "alice-chen",
            "predicate": "city",
            "as_of_tx_time": "2099-01-01T00:00:00Z",
        })
        result = json.loads(raw)
        assert result["status"] == "Resolved", f"Expected Resolved, got {result['status']!r}"
        assert result["value"] == "Austin TX"

    def test_recall_as_of_before_write_returns_nobelief(
        self,
        recall_as_of_tool: RecallAsOfTool,
    ) -> None:
        """as_of before any writes returns NoBelief (empty adapter)."""
        raw = recall_as_of_tool.invoke({
            "agent_id": AGENT_ID,
            "subject": "alice-chen",
            "predicate": "city",
            "as_of_tx_time": "1970-01-01T00:00:01Z",
        })
        result = json.loads(raw)
        assert result["status"] == "NoBelief", (
            f"Expected NoBelief as_of 1970, got {result['status']!r}"
        )
        assert result["value"] is None

    def test_recall_as_of_result_keys(self, recall_as_of_tool: RecallAsOfTool, adapter: MempillAdapter) -> None:
        """RecallAsOfTool result has all required keys."""
        _seed_claims(adapter)
        raw = recall_as_of_tool.invoke({
            "agent_id": AGENT_ID,
            "subject": "alice-chen",
            "predicate": "employer",
            "as_of_tx_time": "2099-06-01T00:00:00Z",
        })
        result = json.loads(raw)
        required = {"subject", "predicate", "value", "status", "is_contested",
                    "conf", "valid_from_display", "valid_until_display",
                    "provenance", "claim_ref"}
        missing = required - set(result.keys())
        assert not missing, f"Missing keys: {missing}"


# ── A5: RememberFactTool (free-form) ─────────────────────────────────────────

class TestRememberFactTool:
    """A5 — RememberFactTool: free-form writes, normalisation, succession."""

    def test_write_free_form_predicate(
        self,
        remember_fact_tool: RememberFactTool,
    ) -> None:
        """Writing a free-form predicate 'favorite-color' succeeds without error."""
        raw = remember_fact_tool.invoke({
            "agent_id": AGENT_ID,
            "subject": "alice-chen",
            "predicate": "favorite-color",
            "value": "deep-blue",
            "valid_from": "2023-01",
        })
        result = json.loads(raw)
        assert result["disposition"] in ("CommittedCheap", "Contested", "QueuedForAdjudication"), (
            f"Expected a valid disposition, got {result['disposition']!r}"
        )
        assert result["claim_ref"], "claim_ref must be non-empty"

    def test_recall_subject_surfaces_free_form_after_write(
        self,
        remember_fact_tool: RememberFactTool,
        recall_subject_tool: RecallSubjectTool,
    ) -> None:
        """After writing a free-form predicate, recall_subject surfaces it."""
        remember_fact_tool.invoke({
            "agent_id": AGENT_ID,
            "subject": "alice-chen",
            "predicate": "favorite-color",
            "value": "deep-blue",
            "valid_from": "2023-01",
        })
        remember_fact_tool.invoke({
            "agent_id": AGENT_ID,
            "subject": "alice-chen",
            "predicate": "city",
            "value": "Austin TX",
            "valid_from": "2023-06",
        })
        remember_fact_tool.invoke({
            "agent_id": AGENT_ID,
            "subject": "alice-chen",
            "predicate": "employer",
            "value": "Acme Corp",
            "valid_from": "2023-06",
        })

        raw = recall_subject_tool.invoke({"agent_id": AGENT_ID, "subject": "alice-chen"})
        result = json.loads(raw)

        predicates = {f["predicate"] for f in result["facts"]}
        assert "favorite-color" in predicates, (
            f"Free-form 'favorite-color' must appear in recall_subject. Got: {predicates}"
        )

    def test_subject_normalisation_spaces_to_hyphens(
        self,
        remember_fact_tool: RememberFactTool,
        recall_subject_tool: RecallSubjectTool,
    ) -> None:
        """'Alice Chen' subject is normalised to 'alice-chen' and data is retrievable."""
        remember_fact_tool.invoke({
            "agent_id": AGENT_ID,
            "subject": "Alice Chen",  # spaces → hyphens via normalisation
            "predicate": "nickname",
            "value": "Ace",
            "valid_from": "2024-01",
        })

        raw = recall_subject_tool.invoke({"agent_id": AGENT_ID, "subject": "alice-chen"})
        result = json.loads(raw)
        predicates = {f["predicate"] for f in result["facts"]}
        assert "nickname" in predicates, (
            f"Data written under 'Alice Chen' must be retrievable as 'alice-chen'. "
            f"Got predicates: {predicates}"
        )

    def test_predicate_normalisation_spaces_to_hyphens(
        self,
        remember_fact_tool: RememberFactTool,
        recall_subject_tool: RecallSubjectTool,
    ) -> None:
        """'Favorite Color' predicate is normalised to 'favorite-color'."""
        remember_fact_tool.invoke({
            "agent_id": AGENT_ID,
            "subject": "alice-chen",
            "predicate": "Favorite Color",  # spaces → hyphens via normalisation
            "value": "red",
            "valid_from": "2023-01",
        })

        raw = recall_subject_tool.invoke({"agent_id": AGENT_ID, "subject": "alice-chen"})
        result = json.loads(raw)
        predicates = {f["predicate"] for f in result["facts"]}
        assert "favorite-color" in predicates, (
            f"Predicate 'Favorite Color' must be normalised to 'favorite-color'. "
            f"Got predicates: {predicates}"
        )

    def test_succession_current_recall_is_nyc(
        self,
        remember_fact_tool: RememberFactTool,
        recall_subject_tool: RecallSubjectTool,
    ) -> None:
        """After Austin then NYC via RememberFactTool, current recall == NYC."""
        remember_fact_tool.invoke({
            "agent_id": AGENT_ID,
            "subject": "alice-chen",
            "predicate": "city",
            "value": "Austin TX",
            "valid_from": "2023-06",
        })
        remember_fact_tool.invoke({
            "agent_id": AGENT_ID,
            "subject": "alice-chen",
            "predicate": "city",
            "value": "New York NY",
            "valid_from": "2025-02",
        })

        raw = recall_subject_tool.invoke({"agent_id": AGENT_ID, "subject": "alice-chen"})
        result = json.loads(raw)

        city_facts = [f for f in result["facts"] if f["predicate"] == "city"]
        assert city_facts, "Expected city fact after writes"
        assert city_facts[0]["value"] == "New York NY", (
            f"Current recall should be 'New York NY' after succession. "
            f"Got {city_facts[0]['value']!r}"
        )
        assert city_facts[0]["status"] == "Resolved"

    def test_succession_bitemoral_correctness(
        self,
        remember_fact_tool: RememberFactTool,
        recall_at_tool: RecallAtTool,
    ) -> None:
        """After succession, valid_at in Austin window returns Austin TX."""
        remember_fact_tool.invoke({
            "agent_id": AGENT_ID,
            "subject": "alice-chen",
            "predicate": "city",
            "value": "Austin TX",
            "valid_from": "2023-06",
        })
        remember_fact_tool.invoke({
            "agent_id": AGENT_ID,
            "subject": "alice-chen",
            "predicate": "city",
            "value": "New York NY",
            "valid_from": "2025-02",
        })

        raw = recall_at_tool.invoke({
            "agent_id": AGENT_ID,
            "subject": "alice-chen",
            "predicate": "city",
            "valid_at": "2024-06-01T00:00:00Z",
        })
        result = json.loads(raw)
        assert result["value"] == "Austin TX", (
            f"Bi-temporal: valid_at=2024-06 should return 'Austin TX', "
            f"got {result['value']!r}"
        )

    def test_result_keys_present(self, remember_fact_tool: RememberFactTool) -> None:
        """RememberFactTool result always contains claim_ref, disposition, is_contested."""
        raw = remember_fact_tool.invoke({
            "agent_id": AGENT_ID,
            "subject": "bob-liu",
            "predicate": "status",
            "value": "active",
        })
        result = json.loads(raw)
        assert "claim_ref" in result
        assert "disposition" in result
        assert "is_contested" in result


# ── A6: GetContestedTool ─────────────────────────────────────────────────────

class TestGetContestedTool:
    """A6 — GetContestedTool surfaces competing beliefs."""

    def test_contested_belief_has_incumbent_and_challenger(
        self,
        get_contested_tool: GetContestedTool,
        adapter: MempillAdapter,
    ) -> None:
        """After a same-period conflict, get_contested returns incumbent + challenger."""
        # Write VP Engineering (incumbent)
        adapter.write_claim(AGENT_ID, ClaimInput(
            subject="alice-chen",
            predicate="employer",
            value="Acme Corp / VP Engineering",
            valid_from="2023-06",
            confidence=1.0,
            provenance=ProvenanceLabel.external_user_asserted(),
            cardinality="Functional",
        ))
        # Write CTO at same period (genuine conflict → Contested)
        adapter.write_claim(AGENT_ID, ClaimInput(
            subject="alice-chen",
            predicate="employer",
            value="Acme Corp / CTO",
            valid_from="2023-06",  # same period → genuine conflict
            confidence=1.0,
            provenance=ProvenanceLabel.external_user_asserted(),
            cardinality="Functional",
        ))

        raw = get_contested_tool.invoke({
            "agent_id": AGENT_ID,
            "subject": "alice-chen",
            "predicate": "employer",
        })
        result = json.loads(raw)

        assert result["is_contested"], (
            f"Expected is_contested=True for same-period conflict, got status={result['status']!r}"
        )
        # When contested: alternatives should contain at least the two competing values
        assert len(result["alternatives"]) >= 2, (
            f"Expected at least 2 alternatives for contested belief, got {len(result['alternatives'])}"
        )
        alt_values = {a["value"] for a in result["alternatives"]}
        assert "Acme Corp / VP Engineering" in alt_values or "Acme Corp / CTO" in alt_values, (
            f"Expected competing values in alternatives: {alt_values}"
        )

    def test_resolved_belief_not_contested(
        self,
        get_contested_tool: GetContestedTool,
        adapter: MempillAdapter,
    ) -> None:
        """Resolved belief returns is_contested=False and empty incumbent/challenger."""
        adapter.write_claim(AGENT_ID, ClaimInput(
            subject="alice-chen",
            predicate="city",
            value="Austin TX",
            valid_from="2023-06",
            confidence=1.0,
            provenance=ProvenanceLabel.external_user_asserted(),
        ))

        raw = get_contested_tool.invoke({
            "agent_id": AGENT_ID,
            "subject": "alice-chen",
            "predicate": "city",
        })
        result = json.loads(raw)
        assert not result["is_contested"], f"Expected is_contested=False, got {result}"
        assert result["status"] == "Resolved"

    def test_nobelief_not_contested(
        self,
        get_contested_tool: GetContestedTool,
    ) -> None:
        """NoBelief on unknown predicate returns is_contested=False."""
        raw = get_contested_tool.invoke({
            "agent_id": AGENT_ID,
            "subject": "alice-chen",
            "predicate": "unknown-predicate-xyz",
        })
        result = json.loads(raw)
        assert not result["is_contested"]
        assert result["status"] == "NoBelief"
        assert result["incumbent"] is None
        assert result["challenger"] is None

    def test_result_structure_keys(
        self,
        get_contested_tool: GetContestedTool,
        adapter: MempillAdapter,
    ) -> None:
        """GetContestedTool result always has required top-level keys."""
        adapter.write_claim(AGENT_ID, ClaimInput(
            subject="bob-liu",
            predicate="city",
            value="San Francisco",
            confidence=1.0,
            provenance=ProvenanceLabel.external_user_asserted(),
        ))
        raw = get_contested_tool.invoke({
            "agent_id": AGENT_ID,
            "subject": "bob-liu",
            "predicate": "city",
        })
        result = json.loads(raw)
        required = {"subject", "predicate", "status", "is_contested",
                    "incumbent", "challenger", "alternatives"}
        missing = required - set(result.keys())
        assert not missing, f"Missing keys: {missing}"


# ── A7: AuditTrailTool ────────────────────────────────────────────────────────

class TestAuditTrailTool:
    """A7 — AuditTrailTool returns audit ledger."""

    def test_audit_empty_on_fresh_adapter(self, audit_trail_tool: AuditTrailTool) -> None:
        """Fresh adapter has no audit entries."""
        raw = audit_trail_tool.invoke({"agent_id": AGENT_ID, "limit": 50})
        result = json.loads(raw)
        assert result["entry_count"] == 0

    def test_audit_returns_entries_after_writes(
        self,
        audit_trail_tool: AuditTrailTool,
        adapter: MempillAdapter,
    ) -> None:
        """After 3 writes, audit returns at least 3 entries."""
        _seed_claims(adapter)

        raw = audit_trail_tool.invoke({"agent_id": AGENT_ID, "limit": 50})
        result = json.loads(raw)
        assert result["entry_count"] >= 3, (
            f"Expected at least 3 audit entries after 3 writes, got {result['entry_count']}"
        )

    def test_audit_entry_has_required_fields(
        self,
        audit_trail_tool: AuditTrailTool,
        adapter: MempillAdapter,
    ) -> None:
        """Each audit entry has claim_ref, event_kind, disposition, recorded_at, rationale."""
        _seed_claims(adapter)
        raw = audit_trail_tool.invoke({"agent_id": AGENT_ID, "limit": 50})
        result = json.loads(raw)
        required = {"claim_ref", "event_kind", "disposition", "recorded_at", "rationale"}
        for entry in result["entries"]:
            missing = required - set(entry.keys())
            assert not missing, f"Audit entry missing keys {missing}: {entry}"

    def test_audit_claim_ref_filter(
        self,
        audit_trail_tool: AuditTrailTool,
        adapter: MempillAdapter,
    ) -> None:
        """claim_ref filter returns only entries for that specific claim."""
        receipt = adapter.write_claim(AGENT_ID, ClaimInput(
            subject="alice-chen",
            predicate="city",
            value="Austin TX",
            valid_from="2023-06",
            confidence=1.0,
            provenance=ProvenanceLabel.external_user_asserted(),
        ))
        target_ref = receipt.claim_ref

        # Write another claim
        adapter.write_claim(AGENT_ID, ClaimInput(
            subject="bob-liu",
            predicate="city",
            value="San Francisco",
            confidence=1.0,
            provenance=ProvenanceLabel.external_user_asserted(),
        ))

        raw = audit_trail_tool.invoke({
            "agent_id": AGENT_ID,
            "limit": 50,
            "claim_ref": target_ref,
        })
        result = json.loads(raw)
        for entry in result["entries"]:
            assert entry["claim_ref"] == target_ref, (
                f"Filtered audit should only return entries for {target_ref!r}, "
                f"got {entry['claim_ref']!r}"
            )

    def test_audit_result_echoes_agent_id(
        self,
        audit_trail_tool: AuditTrailTool,
    ) -> None:
        """Result dict echoes agent_id."""
        raw = audit_trail_tool.invoke({"agent_id": AGENT_ID})
        result = json.loads(raw)
        assert result["agent_id"] == AGENT_ID
