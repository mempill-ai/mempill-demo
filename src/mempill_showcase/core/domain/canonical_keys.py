"""
mempill_showcase.core.domain.canonical_keys — deterministic entity/predicate lookup.

Plain dicts — no LLM, no fuzzy matching, no external calls.
The LLM never invents keys; it always passes through resolve_entity/resolve_predicate
before any mempill write. Adding a new contact or predicate = adding a row here.

Scenario entities (jordan-park-001):
  alice-chen, bob-liu, acme-corp, jordan-park
"""
from __future__ import annotations

from typing import Optional


# ── Entity aliases → canonical key ───────────────────────────────────────────
#
# Keys are lowercase-hyphenated identifiers.
# Values are all the strings that should map to that canonical key.
# Matching is case-insensitive (see resolve_entity).

_ENTITY_ALIASES: dict[str, str] = {
    # Alice Chen
    "alice chen":           "alice-chen",
    "alice":                "alice-chen",
    "alice-chen":           "alice-chen",
    "alice_chen":           "alice-chen",
    # Bob Liu
    "bob liu":              "bob-liu",
    "bob":                  "bob-liu",
    "bob-liu":              "bob-liu",
    "bob_liu":              "bob-liu",
    # Acme Corp
    "acme corp":            "acme-corp",
    "acme":                 "acme-corp",
    "acme-corp":            "acme-corp",
    "acme_corp":            "acme-corp",
    "acme corporation":     "acme-corp",
    # Jordan Park
    "jordan park":          "jordan-park",
    "jordan":               "jordan-park",
    "jordan-park":          "jordan-park",
    "jordan_park":          "jordan-park",
}


# ── Predicate aliases → canonical predicate ───────────────────────────────────
#
# Predicate keys are lowercase-underscore identifiers.

_PREDICATE_ALIASES: dict[str, str] = {
    # city / location
    "city":                 "city",
    "location":             "city",
    "office city":          "city",
    "office location":      "city",
    # employer
    "employer":             "employer",
    "company":              "employer",
    "works at":             "employer",
    "works for":            "employer",
    "organization":         "employer",
    # dietary
    "dietary_restriction":  "dietary_restriction",
    "dietary restriction":  "dietary_restriction",
    "diet":                 "dietary_restriction",
    "food preference":      "dietary_restriction",
    # travel
    "travel_preference":    "travel_preference",
    "travel preference":    "travel_preference",
    "flight preference":    "travel_preference",
    # hotel
    "preferred_hotel":      "preferred_hotel",
    "preferred hotel":      "preferred_hotel",
    "hotel preference":     "preferred_hotel",
    # CEO / executive
    "ceo":                  "ceo",
    "chief executive":      "ceo",
    "chief executive officer": "ceo",
    # CTO
    "cto":                  "cto",
    "chief technology officer": "cto",
    # role / title / position — person's job/title is stored under "employer"
    # (seed + all writes use "employer" as "Company / Job Title").
    # These aliases map to "employer" so "what role does Alice hold?" resolves
    # the same belief as "who is Alice's employer?".
    "title":                "employer",
    "role":                 "employer",
    "job title":            "employer",
    "position":             "employer",
    "job":                  "employer",
}


def resolve_entity(name: str) -> Optional[str]:
    """Return the canonical entity key for *name*, or None if unknown.

    Matching is case-insensitive and strips leading/trailing whitespace.
    """
    return _ENTITY_ALIASES.get(name.strip().lower())


def resolve_predicate(name: str) -> Optional[str]:
    """Return the canonical predicate key for *name*, or None if unknown.

    Matching is case-insensitive.
    """
    return _PREDICATE_ALIASES.get(name.strip().lower())


def all_canonical_entities() -> frozenset[str]:
    """Return the set of all known canonical entity keys."""
    return frozenset(_ENTITY_ALIASES.values())


def all_canonical_predicates() -> frozenset[str]:
    """Return the set of all known canonical predicate keys."""
    return frozenset(_PREDICATE_ALIASES.values())
