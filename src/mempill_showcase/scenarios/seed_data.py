"""
mempill_showcase.scenarios.seed_data — Day-0 seed claims for agent_id "jordan-park-001".

Loads the 7 initial facts from SCENARIO.md into a MemoryStore adapter via write_claim().
Designed to be idempotent: calling twice produces the same final state
(mempill engine handles duplicate detection via succession).

Seed claims (from SCENARIO.md § System Starting State):
  alice-chen  / employer           = Acme Corp / VP Engineering  valid_from=2023-06-01  UserAsserted
  alice-chen  / city               = Austin TX                    valid_from=2023-06-01  UserAsserted
  alice-chen  / dietary_restriction= vegetarian                   valid_from=2024-01-01  UserAsserted
  bob-liu     / employer           = Meridian Ventures / Partner  valid_from=2022-09-01  UserAsserted
  bob-liu     / travel_preference  = window seat, no checked bags valid_from=2023-03-01  UserAsserted
  acme-corp   / ceo                = Diane Foster                 valid_from=2021-04-01  ExternalFirstHand
  jordan-park / preferred_hotel    = Marriott Bonvoy Gold         valid_from=2024-06-01  UserAsserted
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mempill_showcase.adapters.memory.mempill_adapter import MempillAdapter

from mempill_showcase.core.domain.models import ClaimInput

AGENT_ID = "jordan-park-001"

# ── Seed definitions ──────────────────────────────────────────────────────────
# Each tuple: (subject, predicate, value, valid_from, provenance_type)
# provenance_type: "UserAsserted" | "ExternalFirstHand"

_SEED_CLAIMS = [
    # (subject, predicate, value, valid_from, valid_until, provenance_type)
    # alice-chen/city is bounded at valid_until=2025-02 (Austin is the Day-0 belief;
    # valid_until is set to the known move date so the successor NYC claim at 2025-02
    # does not overlap → CommittedCheap succession without oracle conflict).
    ("alice-chen",  "employer",            "Acme Corp / VP Engineering",   "2023-06-01",  None,       "UserAsserted"),
    ("alice-chen",  "city",                "Austin TX",                    "2023-06-01",  "2025-02",  "UserAsserted"),
    ("alice-chen",  "dietary_restriction", "vegetarian",                   "2024-01-01",  None,       "UserAsserted"),
    ("bob-liu",     "employer",            "Meridian Ventures / Partner",  "2022-09-01",  None,       "UserAsserted"),
    ("bob-liu",     "travel_preference",   "window seat, no checked bags", "2023-03-01",  None,       "UserAsserted"),
    ("acme-corp",   "ceo",                 "Diane Foster",                 "2021-04-01",  None,       "ExternalFirstHand"),
    ("jordan-park", "preferred_hotel",     "Marriott Bonvoy Gold",         "2024-06-01",  None,       "UserAsserted"),
]


def load_seed_claims(adapter: "MempillAdapter", agent_id: str = AGENT_ID) -> list[str]:
    """Write Day-0 seed claims into *adapter* for *agent_id*.

    Returns a list of claim_ref UUIDs (one per ingested claim).
    ProvenanceLabel factory is resolved here (inside adapters boundary is fine —
    but to keep seed_data importable without mempill, we import lazily).
    """
    from mempill import ProvenanceLabel

    refs: list[str] = []
    for subject, predicate, value, valid_from, valid_until, prov_type in _SEED_CLAIMS:
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

    return refs
