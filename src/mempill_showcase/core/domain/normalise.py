"""
mempill_showcase.core.domain.normalise — shared key normalisation helper.

All memory tools that accept a subject or predicate MUST pass the value through
``normalise_key`` before calling the adapter.  This guarantees that a write of
subject "Alice Chen" and a subsequent recall_at("Alice Chen", ...) resolve to
the same internal key "alice-chen".

Soft normalisation rules (applied in order):
  1. strip()        — remove leading/trailing whitespace
  2. lower()        — fold to lowercase
  3. collapse any sequence of whitespace to a single "-"  — spaces→hyphens
"""
from __future__ import annotations

import re


def normalise_key(s: str) -> str:
    """Strip, lowercase, and replace runs of whitespace with a single hyphen.

    Examples::

        normalise_key("Alice Chen")     # "alice-chen"
        normalise_key(" Employer ")     # "employer"
        normalise_key("favorite color") # "favorite-color"
        normalise_key("already-done")   # "already-done"
    """
    return re.sub(r"\s+", "-", s.strip().lower())
