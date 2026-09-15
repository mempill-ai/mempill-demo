"""
mempill_showcase.tests.test_seed_data — TASK-31 T31-2 seed-split regression tests.

Verifies the seed_data.py split introduced for the dual-agent router
(ARCHITECTURE.md §5):
  PEOPLE_OPS_SEED_CLAIMS  — person-subject facts (alice-chen, bob-liu, jordan-park)
  ORG_REGISTRY_SEED_CLAIMS — acme-corp facts (ceo + NEW hq_location)
  _SEED_CLAIMS            — full legacy list, unchanged behavior (7 facts)

No API key required — pure data/adapter assertions.
"""
from __future__ import annotations

import pytest

from mempill_showcase.config.di import build_mempill_adapter
from mempill_showcase.scenarios.seed_data import (
    AGENT_ID,
    ORG_REGISTRY_SEED_CLAIMS,
    PEOPLE_OPS_SEED_CLAIMS,
    _SEED_CLAIMS,
    load_seed_claims,
)


class TestSeedSplitUnion:
    """The two domain-scoped lists union back to legacy 7 + 1 new, no dup/loss."""

    def test_legacy_list_has_7_facts(self):
        assert len(_SEED_CLAIMS) == 7, f"Legacy _SEED_CLAIMS must have 7 facts, got {len(_SEED_CLAIMS)}"

    def test_people_ops_list_is_person_subjects_only(self):
        subjects = {c[0] for c in PEOPLE_OPS_SEED_CLAIMS}
        assert subjects == {"alice-chen", "bob-liu", "jordan-park"}, (
            f"PEOPLE_OPS_SEED_CLAIMS must only contain person subjects, got {subjects}"
        )

    def test_org_registry_list_is_acme_corp_only(self):
        subjects = {c[0] for c in ORG_REGISTRY_SEED_CLAIMS}
        assert subjects == {"acme-corp"}, (
            f"ORG_REGISTRY_SEED_CLAIMS must only contain acme-corp facts, got {subjects}"
        )

    def test_org_registry_has_new_hq_location_seed(self):
        predicates = {c[1] for c in ORG_REGISTRY_SEED_CLAIMS}
        assert "hq_location" in predicates, (
            "ORG_REGISTRY_SEED_CLAIMS must include the new acme-corp/hq_location seed"
        )
        assert "ceo" in predicates, "ORG_REGISTRY_SEED_CLAIMS must retain the legacy ceo seed"

    def test_org_registry_has_exactly_2_facts(self):
        assert len(ORG_REGISTRY_SEED_CLAIMS) == 2, (
            f"ORG_REGISTRY_SEED_CLAIMS must have 2 facts (ceo + hq_location), "
            f"got {len(ORG_REGISTRY_SEED_CLAIMS)}"
        )

    def test_union_equals_legacy_7_plus_1_new_no_dup_no_loss(self):
        """union(PEOPLE_OPS, ORG_REGISTRY) == legacy 7 facts + exactly 1 new fact."""
        union_keys = {(c[0], c[1]) for c in PEOPLE_OPS_SEED_CLAIMS + ORG_REGISTRY_SEED_CLAIMS}
        legacy_keys = {(c[0], c[1]) for c in _SEED_CLAIMS}

        assert len(union_keys) == len(PEOPLE_OPS_SEED_CLAIMS) + len(ORG_REGISTRY_SEED_CLAIMS), (
            "Union must have no duplicate (subject, predicate) keys across the two lists"
        )
        assert legacy_keys.issubset(union_keys), (
            f"Union must be a superset of the legacy 7 facts. Missing: {legacy_keys - union_keys}"
        )
        new_keys = union_keys - legacy_keys
        assert new_keys == {("acme-corp", "hq_location")}, (
            f"Union must add EXACTLY 1 new fact (acme-corp/hq_location), got new keys: {new_keys}"
        )
        assert len(union_keys) == 8, f"Union must total 8 facts (7 legacy + 1 new), got {len(union_keys)}"

    def test_hq_location_matches_neighboring_seed_shape(self):
        """hq_location tuple shape/granularity matches the neighboring ceo seed's field requirements."""
        hq = next(c for c in ORG_REGISTRY_SEED_CLAIMS if c[1] == "hq_location")
        ceo = next(c for c in ORG_REGISTRY_SEED_CLAIMS if c[1] == "ceo")
        assert len(hq) == len(ceo) == 6, "Seed tuples must be 6-element (subject, predicate, value, valid_from, valid_until, provenance_type)"
        assert hq[3] == "2019-01-01", "hq_location valid_from must be day-precision, matching ceo's granularity"
        assert hq[4] is None, "hq_location must be open-ended (valid_until=None), matching ceo"
        assert hq[5] == "ExternalFirstHand", "hq_location provenance must be ExternalFirstHand, matching ceo"


class TestLoadSeedClaimsBackwardCompat:
    """load_seed_claims(adapter, agent_id) with no `claims` arg is unchanged."""

    def test_default_claims_none_uses_legacy_list(self):
        adapter = build_mempill_adapter(in_memory=True, oracle_backed=True)
        refs = load_seed_claims(adapter, AGENT_ID)
        assert len(refs) == 7, f"Default load_seed_claims must write 7 legacy facts, got {len(refs)}"

        # hq_location must NOT be present when using the legacy default path.
        belief = adapter.recall(AGENT_ID, "acme-corp", "hq_location")
        assert belief.status == "NoBelief", (
            "Legacy default load_seed_claims() must NOT write the new hq_location seed"
        )

    def test_people_ops_claims_writes_6_facts(self):
        adapter = build_mempill_adapter(in_memory=True, oracle_backed=True)
        refs = load_seed_claims(adapter, "people-ops-001", claims=PEOPLE_OPS_SEED_CLAIMS)
        assert len(refs) == 6, f"PEOPLE_OPS_SEED_CLAIMS must write 6 facts, got {len(refs)}"

        belief = adapter.recall("people-ops-001", "acme-corp", "ceo")
        assert belief.status == "NoBelief", "people-ops DB must not contain any acme-corp facts"

    def test_org_registry_claims_writes_2_facts(self):
        adapter = build_mempill_adapter(in_memory=True, oracle_backed=True)
        refs = load_seed_claims(adapter, "org-registry-001", claims=ORG_REGISTRY_SEED_CLAIMS)
        assert len(refs) == 2, f"ORG_REGISTRY_SEED_CLAIMS must write 2 facts, got {len(refs)}"

        belief = adapter.recall("org-registry-001", "acme-corp", "hq_location")
        assert belief.status != "NoBelief"
        assert belief.value == "Austin, TX"

        belief_alice = adapter.recall("org-registry-001", "alice-chen", "employer")
        assert belief_alice.status == "NoBelief", "org-registry DB must not contain any person facts"

    def test_org_registry_idempotent_on_second_call(self):
        """Domain-scoped seeding is idempotent (uses the correct per-domain sentinel)."""
        adapter = build_mempill_adapter(in_memory=True, oracle_backed=True)
        first = load_seed_claims(adapter, "org-registry-001", claims=ORG_REGISTRY_SEED_CLAIMS)
        assert len(first) == 2

        second = load_seed_claims(adapter, "org-registry-001", claims=ORG_REGISTRY_SEED_CLAIMS)
        assert second == [], "Second call must be a no-op (idempotency guard via correct sentinel)"
