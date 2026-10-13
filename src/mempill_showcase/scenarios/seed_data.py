"""
mempill_showcase.scenarios.seed_data — Day-0 seed claims for the showcase agent.

Loads the 7 initial facts from SCENARIO.md into a MemoryStore adapter via write_claim().
Designed to be idempotent: load_seed_claims() checks whether the store already
contains seed data for the given agent_id and skips if present — safe for both
in-memory and file-backed (persistent) engines.

Seed claims (from SCENARIO.md § System Starting State):
  alice-chen  / employer           = Acme Corp / VP Engineering  valid_from=2023-06     UserAsserted
  alice-chen  / city               = Austin TX                    valid_from=2023-06-01  UserAsserted
  alice-chen  / dietary_restriction= vegetarian                   valid_from=2024-01-01  UserAsserted
  bob-liu     / employer           = Meridian Ventures / Partner  valid_from=2022-09-01  UserAsserted
  bob-liu     / travel_preference  = window seat, no checked bags valid_from=2023-03-01  UserAsserted
  acme-corp   / ceo                = Diane Foster                 valid_from=2021-04-01  ExternalFirstHand
  jordan-park / preferred_hotel    = Marriott Bonvoy Gold         valid_from=2024-06-01  UserAsserted

TASK-31 dual-agent router split (ARCHITECTURE.md §5): the legacy 7-fact list
above is now composed from two domain-scoped lists —
  PEOPLE_OPS_SEED_CLAIMS  (alice-chen, bob-liu, jordan-park — 6 facts)
  ORG_REGISTRY_SEED_CLAIMS (acme-corp — 1 legacy fact + 1 NEW seed:
                            acme-corp/hq_location, valid_from=2019-01-01,
                            ExternalFirstHand)
_SEED_CLAIMS (the full legacy list, used by default when load_seed_claims()
is called with claims=None) is now PEOPLE_OPS_SEED_CLAIMS + the ORIGINAL
7th fact only (acme-corp/ceo) — i.e. it does NOT include the new
hq_location seed, preserving exact backward-compatible behavior for existing
single-agent callers/tests. The new hq_location fact is exclusive to
ORG_REGISTRY_SEED_CLAIMS.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mempill_showcase.adapters.memory.mempill_adapter import MempillAdapter

from mempill_showcase.core.domain.models import ClaimInput

log = logging.getLogger(__name__)

# Default agent_id — used as the default parameter value and for the module-level
# constant that legacy callers (tests, studio_graph) import.  In all live paths
# the value comes from Settings.mempill_agent_id (env: MEMPILL_AGENT_ID).
AGENT_ID = "jordan-park-001"

# Canonical "known-seed" sentinel: if this belief already exists the DB has been
# seeded for this agent_id and we must not re-ingest (prevents Contested dupes on
# file-backed engines across restarts).
_SEED_SENTINEL_SUBJECT = "alice-chen"
_SEED_SENTINEL_PREDICATE = "employer"

# ── Seed definitions ──────────────────────────────────────────────────────────
# Each tuple: (subject, predicate, value, valid_from, provenance_type)
# provenance_type: "UserAsserted" | "ExternalFirstHand"

# PEOPLE_OPS_SEED_CLAIMS — person-subject facts (alice-chen, bob-liu, jordan-park).
# Same 6 tuples/values as the legacy list, unchanged.
PEOPLE_OPS_SEED_CLAIMS = [
    # (subject, predicate, value, valid_from, valid_until, provenance_type)
    # alice-chen/city is bounded at valid_until=2025-02 (Austin is the Day-0 belief;
    # valid_until is set to the known move date so the successor NYC claim at 2025-02
    # does not overlap → CommittedCheap succession without oracle conflict).
    ("alice-chen",  "employer",            "Acme Corp / VP Engineering",   "2023-06",     None,       "UserAsserted"),
    ("alice-chen",  "city",                "Austin TX",                    "2023-06-01",  "2025-02",  "UserAsserted"),
    ("alice-chen",  "dietary_restriction", "vegetarian",                   "2024-01-01",  None,       "UserAsserted"),
    ("bob-liu",     "employer",            "Meridian Ventures / Partner",  "2022-09-01",  None,       "UserAsserted"),
    ("bob-liu",     "travel_preference",   "window seat, no checked bags", "2023-03-01",  None,       "UserAsserted"),
    ("jordan-park", "preferred_hotel",     "Marriott Bonvoy Gold",         "2024-06-01",  None,       "UserAsserted"),
]

# ORG_REGISTRY_SEED_CLAIMS — acme-corp facts: the 1 legacy fact (ceo) + ONE new
# seed (hq_location) so the org-registry DB is non-trivially populated on first
# run (ARCHITECTURE.md §5). hq_location is inert w.r.t. existing contested-write
# demo flows (no overlapping predicate with ceo).
ORG_REGISTRY_SEED_CLAIMS = [
    ("acme-corp",   "ceo",                 "Diane Foster",                 "2021-04-01",  None,       "ExternalFirstHand"),
    ("acme-corp",   "hq_location",         "Austin, TX",                   "2019-01-01",  None,       "ExternalFirstHand"),
]

# _SEED_CLAIMS — full legacy list (7 facts), preserved EXACTLY for backward
# compatibility: existing single-agent callers (exec_assistant, tests) that
# call load_seed_claims(adapter, agent_id) with no `claims` arg must see the
# same 7 facts as before this split — NOT the new hq_location seed.
_SEED_CLAIMS = PEOPLE_OPS_SEED_CLAIMS + [
    ("acme-corp",   "ceo",                 "Diane Foster",                 "2021-04-01",  None,       "ExternalFirstHand"),
]


def is_already_seeded(
    adapter: "MempillAdapter",
    agent_id: str,
    claims: "list[tuple] | None" = None,
) -> bool:
    """Return True if the store already contains seed data for *agent_id*.

    Uses the sentinel claim — the FIRST (subject, predicate) pair in *claims*
    (defaulting to the legacy full list's first entry, alice-chen/employer,
    when claims is None) — as a proxy for seed presence. If it exists with a
    non-NoBelief status the store was previously seeded. This check is
    critical for file-backed (persistent) engines where re-running the
    showcase must not duplicate seed writes and produce Contested beliefs.

    Deriving the sentinel from *claims* (rather than a single hardcoded
    alice-chen/employer constant) makes this idempotency check correct for
    domain-scoped seed lists too (e.g. ORG_REGISTRY_SEED_CLAIMS, whose store
    never contains an alice-chen claim) — TASK-31 dual-agent router.
    """
    if claims:
        sentinel_subject, sentinel_predicate = claims[0][0], claims[0][1]
    else:
        sentinel_subject, sentinel_predicate = _SEED_SENTINEL_SUBJECT, _SEED_SENTINEL_PREDICATE

    try:
        belief = adapter.recall(agent_id, sentinel_subject, sentinel_predicate)
        return belief.status not in ("NoBelief", None)
    except Exception as exc:
        log.debug("is_already_seeded: recall check failed (%s) — assuming not seeded", exc)
        return False


def load_seed_claims(
    adapter: "MempillAdapter",
    agent_id: str = AGENT_ID,
    claims: "list[tuple] | None" = None,
) -> list[str]:
    """Write Day-0 seed claims into *adapter* for *agent_id*.

    Idempotent: if the store already contains seed data for *agent_id*
    (detected via is_already_seeded()), this function returns an empty list
    without writing anything.  This prevents duplicate Contested beliefs on
    file-backed engines across restarts.

    Args:
        adapter:  MempillAdapter to write into.
        agent_id: Agent identity to seed under.
        claims:   Optional list of (subject, predicate, value, valid_from,
                  valid_until, provenance_type) tuples. Defaults to None, which
                  uses the full legacy 7-fact list (_SEED_CLAIMS) — backward
                  compatible for existing single-agent callers. Pass
                  PEOPLE_OPS_SEED_CLAIMS or ORG_REGISTRY_SEED_CLAIMS for the
                  TASK-31 dual-agent router's domain-scoped seeding.

    Returns a list of claim_ref UUIDs (one per ingested claim), or an empty
    list when the seed was skipped because data was already present.
    """
    if claims is None:
        claims = _SEED_CLAIMS

    if is_already_seeded(adapter, agent_id, claims=claims):
        log.info(
            "load_seed_claims: store already seeded for agent_id=%r — skipping",
            agent_id,
        )
        return []

    from mempill import ProvenanceLabel

    refs: list[str] = []
    for subject, predicate, value, valid_from, valid_until, prov_type in claims:
        if prov_type == "ExternalFirstHand":
            prov = ProvenanceLabel.external_first_hand()
        else:
            prov = ProvenanceLabel.external_user_asserted()

        claim = ClaimInput(
            subject=subject,
            predicate=predicate,
            value=value,
            valid_from=valid_from,
            valid_until=valid_until,
            confidence=1.0,
            provenance=prov,
            cardinality="Functional",
            criticality="Medium",
        )
        receipt = adapter.write_claim(agent_id, claim)
        refs.append(receipt.claim_ref)

    log.info("load_seed_claims: wrote %d seed claims for agent_id=%r", len(refs), agent_id)
    return refs
