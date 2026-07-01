"""
mempill_showcase.core.domain.dates — granularity-aware valid-time comparison.

Valid-time display strings carry one of three granularities:
  "YYYY"          → year-level  (e.g. "2023")
  "YYYY-MM"       → month-level (e.g. "2023-06")
  "YYYY-MM-DD"    → day-level   (e.g. "2023-06-01")

Problem: lexicographic string comparison is WRONG across granularities.
  "2023-06-01" > "2023-06"  lexicographically, but they denote the same
  period-start (June 2023).  Treating them as "after" would cause the
  succession guard to close the open-ended incumbent even though it started
  at the same time — turning a genuine same-period contest into a false
  succession close.

Comparison rule (period-start alignment):
  Each display string is mapped to a (year, month, day) triple where missing
  components are filled with the MINIMUM value for that component (month=1,
  day=1).  This aligns every display on its period start instant so that:

    "2023-06"    → (2023, 6, 1)   ← same as "June 2023" start
    "2023-06-01" → (2023, 6, 1)   ← identical → EQUAL (same period)
    "2024-01"    → (2024, 1, 1)   ← later than June 2023 → AFTER

  This is the "coarser common granularity / period-start instant" approach:
  we do NOT try to detect half-open intervals at the finer end; we simply
  ask "do these two displays denote the same or a later period start?".

  Semantic consequence for the succession guard:
    - EQUAL period-starts → same-period contest (return False from is_later)
    - STRICTLY later period-start → succession candidate (return True from is_later)

Public API:
  parse_display_to_tuple(display: str) -> tuple[int, int, int]
    Map a YYYY / YYYY-MM / YYYY-MM-DD string to (year, month, day).
    Raises ValueError for unrecognised formats.

  is_later(candidate: str, incumbent: str) -> bool
    True iff candidate's period-start is STRICTLY after incumbent's.
    False for equal period-starts (same-period → contest, not succession).
"""
from __future__ import annotations

import re

# Regexes for the three supported ISO display granularities
_YEAR_ONLY = re.compile(r"^(\d{4})$")
_YEAR_MONTH = re.compile(r"^(\d{4})-(\d{2})$")
_YEAR_MONTH_DAY = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")


def parse_display_to_tuple(display: str) -> tuple[int, int, int]:
    """Return a (year, month, day) triple aligned to the period-start instant.

    Missing components are filled with minimum values (month=1, day=1).

    Examples::

        parse_display_to_tuple("2023")       # (2023, 1, 1)
        parse_display_to_tuple("2023-06")    # (2023, 6, 1)
        parse_display_to_tuple("2023-06-01") # (2023, 6, 1)
        parse_display_to_tuple("2024-01")    # (2024, 1, 1)

    Raises:
        ValueError: if the string does not match any supported granularity.
    """
    m = _YEAR_MONTH_DAY.match(display)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))

    m = _YEAR_MONTH.match(display)
    if m:
        return int(m.group(1)), int(m.group(2)), 1

    m = _YEAR_ONLY.match(display)
    if m:
        return int(m.group(1)), 1, 1

    raise ValueError(
        f"Cannot parse valid-time display string: {display!r}. "
        "Expected YYYY, YYYY-MM, or YYYY-MM-DD."
    )


def is_later(candidate: str, incumbent: str) -> bool:
    """Return True iff *candidate*'s period-start is STRICTLY after *incumbent*'s.

    Both arguments are valid-time display strings (YYYY / YYYY-MM / YYYY-MM-DD).
    Equal period-starts (including granularity-mismatch same-starts like
    "2023-06" vs "2023-06-01") return False — they denote the same period and
    must be treated as a same-period contest, not a succession.

    Args:
        candidate: the new claim's valid_from display string.
        incumbent: the open-ended incumbent's vt_start_display string.

    Returns:
        True  — candidate starts strictly after incumbent → succession, close the incumbent.
        False — same or earlier start → same-period contest, do NOT close incumbent.

    Raises:
        ValueError: if either string cannot be parsed.
    """
    cand_tuple = parse_display_to_tuple(candidate)
    inc_tuple = parse_display_to_tuple(incumbent)
    return cand_tuple > inc_tuple
