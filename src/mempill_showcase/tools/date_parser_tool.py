"""
mempill_showcase.tools.date_parser_tool — DateParserTool

Deterministic natural-date → (ISO valid_from string, granularity) using stdlib only.
NO LLM, NO dateutil (not guaranteed in the venv). Feeds valid_from to MempillRememberTool.

Supported input patterns:
  - "March 2020"           → "2020-03"     (granularity: month)
  - "March 15, 2020"       → "2020-03-15"  (granularity: day)
  - "2020-03"              → "2020-03"     (granularity: month)
  - "2020-03-15"           → "2020-03-15"  (granularity: day)
  - "2020"                 → "2020"        (granularity: year)
  - "since June 15, 2026"  → "2026-06-15"  (granularity: day; strips "since"/"from"/"as of")
  - "since June 2026"      → "2026-06"     (granularity: month)
  - "last month"           → error (relative dates not supported)

Granularity is inferred from the parsed date:
  - Day-level date    → "day"
  - Month+year only   → "month"
  - Year only         → "year"

Returns a JSON dict with:
  - iso_date: the ISO date string (YYYY / YYYY-MM / YYYY-MM-DD)
  - granularity: "year" | "month" | "day"
  - original: the original input string
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Any, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

# Prefixes to strip before parsing ("since June 2020" → "June 2020")
_STRIP_PREFIXES = re.compile(
    r"^(?:since|from|as\s+of|starting|starting\s+from|beginning)\s+",
    re.IGNORECASE,
)

# ISO-only patterns — fast path, preserve original precision
_ISO_YEAR_ONLY = re.compile(r"^(\d{4})$")
_ISO_YEAR_MONTH = re.compile(r"^(\d{4})-(\d{2})$")
_ISO_FULL_DATE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")

# Month name → month number
_MONTH_NAMES: dict[str, int] = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

# Natural-language: "Month Year" or "Month Day, Year" or "Month Day Year"
_MONTH_YEAR = re.compile(
    r"^([A-Za-z]+)\s+(\d{4})$",
)
_MONTH_DAY_YEAR = re.compile(
    r"^([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})$",
)


def _parse_date(raw: str) -> tuple[str, str]:
    """Return (iso_date, granularity) for a date string.

    Raises ValueError if the date cannot be parsed or is relative.
    """
    text = raw.strip()

    # Strip sentence-level prefixes ("since", "from", "as of", ...)
    text = _STRIP_PREFIXES.sub("", text).strip()

    # Fast path: ISO patterns (preserve caller-supplied precision)
    if _ISO_YEAR_ONLY.match(text):
        return text, "year"
    if _ISO_YEAR_MONTH.match(text):
        return text, "month"
    if _ISO_FULL_DATE.match(text):
        return text, "day"

    # Natural-language: "Month Day, Year" (e.g. "June 15, 2026" → "2026-06-15")
    m = _MONTH_DAY_YEAR.match(text)
    if m:
        month_str, day_str, year_str = m.group(1), m.group(2), m.group(3)
        month_num = _MONTH_NAMES.get(month_str.lower())
        if month_num is None:
            raise ValueError(f"Unknown month name: {month_str!r} in {raw!r}")
        return f"{int(year_str):04d}-{month_num:02d}-{int(day_str):02d}", "day"

    # Natural-language: "Month Year" (e.g. "March 2020" → "2020-03")
    m = _MONTH_YEAR.match(text)
    if m:
        month_str, year_str = m.group(1), m.group(2)
        month_num = _MONTH_NAMES.get(month_str.lower())
        if month_num is None:
            raise ValueError(f"Unknown month name: {month_str!r} in {raw!r}")
        return f"{int(year_str):04d}-{month_num:02d}", "month"

    # Exhausted all patterns
    raise ValueError(
        f"Cannot parse date: {raw!r}. "
        "Supported formats: YYYY, YYYY-MM, YYYY-MM-DD, 'Month YYYY', 'Month DD YYYY', "
        "or any of these with a 'since'/'from'/'as of' prefix. "
        "Relative dates (last month, yesterday) are NOT supported."
    )


class DateParserInput(BaseModel):
    date_string: str = Field(
        description=(
            "Natural-language or ISO date string to parse. "
            "Examples: 'March 2020', 'since June 15, 2026', '2023-06', '2020'. "
            "Relative dates like 'last month' are NOT supported."
        )
    )


class DateParserTool(BaseTool):
    """Deterministic natural-date → ISO string parser. No LLM, no external dependencies.

    Converts natural-language or partial ISO date strings into a normalised
    ISO date string (YYYY / YYYY-MM / YYYY-MM-DD) plus a granularity label.
    Used to produce the valid_from value for MempillRememberTool.

    Preserves original precision:
      "March 2020"      → iso_date="2020-03"    granularity="month"
      "March 15, 2020"  → iso_date="2020-03-15" granularity="day"
      "2020"            → iso_date="2020"        granularity="year"
      "since June 2026" → iso_date="2026-06"     granularity="month"

    Relative dates ("last month", "yesterday") and complex expressions are not supported;
    the tool returns an error key in the JSON response (no exception raised to the caller).
    """

    name: str = "date_parser"
    description: str = (
        "Parse a natural-language or ISO date string into a normalised ISO date "
        "and granularity label. Returns JSON with iso_date, granularity, and original. "
        "Use the iso_date as the valid_from argument to mempill_remember. "
        "Does NOT accept relative dates (last month, yesterday, etc.)."
    )
    args_schema: Type[BaseModel] = DateParserInput

    def _run(self, date_string: str, **kwargs: Any) -> str:
        log.debug("DateParserTool: parsing %r", date_string)
        try:
            iso_date, granularity = _parse_date(date_string)
        except ValueError as exc:
            return json.dumps({"error": str(exc), "original": date_string})

        result = {
            "iso_date": iso_date,
            "granularity": granularity,
            "original": date_string,
        }
        log.debug("DateParserTool result: %s", result)
        return json.dumps(result)

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("DateParserTool does not support async")
