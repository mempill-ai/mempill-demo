"""
mempill_demo.adapters.inference_llm — LLM-backed command extraction.

Guard: requires ANTHROPIC_API_KEY in environment.
Model: claude-haiku-4-5-20251001

This module is NEVER imported by selftest or the deterministic path.
It is only loaded when --llm is passed and the key guard has already
verified ANTHROPIC_API_KEY is present.
"""
from __future__ import annotations

import json
import os
from typing import Any

try:
    import anthropic as _anthropic
except ImportError:  # pragma: no cover
    _anthropic = None  # type: ignore[assignment]

from mempill_demo.domain.models import CommandKind, ParsedCommand


_MODEL = os.environ.get("MEMPILL_MODEL", "claude-haiku-4-5-20251001")

_EXTRACTION_SYSTEM = """
You are a command extraction assistant for a mempill memory agent.
Given a natural-language turn from the user, return ONLY a valid JSON object with this schema:

{
  "claims": [
    {
      "subject": "<string>",
      "predicate": "<string>",
      "value": "<string>",
      "provenance": "UserAsserted" | "ModelDerived",
      "valid_from": "<ISO date or null>",
      "valid_until": "<ISO date or null>",
      "value_confidence": <0.0-1.0>,
      "cardinality": "Functional" | "SetValued" | "Unknown"
    }
  ],
  "is_query": <true|false>,
  "query_subject": "<string or null>",
  "query_predicate": "<string or null>"
}

Rules:
- If the turn is a statement/assertion → populate "claims"; set "is_query": false.
- If the turn is a question/recall → set "is_query": true; populate query_subject/query_predicate; claims=[].
- Provenance: user-stated facts → "UserAsserted"; anything you inferred → "ModelDerived".
- Do NOT invent subjects/predicates/values that are not present in the user's message.
- Return ONLY the JSON object. No prose, no markdown fences.
""".strip()


def guard() -> None:
    """Raise SystemExit(1) with a clear message if ANTHROPIC_API_KEY is absent."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: --llm requires ANTHROPIC_API_KEY to be set in the environment.")
        raise SystemExit(1)


class LLMParser:
    """Extract structured commands from natural language using Claude."""

    def __init__(self) -> None:
        guard()
        if _anthropic is None:
            print("ERROR: anthropic package not installed. Run: uv pip install anthropic")
            raise SystemExit(1)
        self._client = _anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def parse(self, raw: str) -> ParsedCommand:
        try:
            extraction = self._extract(raw)
        except Exception as exc:
            return ParsedCommand(
                kind=CommandKind.UNKNOWN,
                raw=raw,
                error=f"LLM extraction failed: {exc}. Treating as unknown.",
            )

        if not self._validate(extraction):
            return ParsedCommand(
                kind=CommandKind.UNKNOWN,
                raw=raw,
                error="LLM returned invalid JSON schema. Treating as unknown.",
            )

        if extraction.get("is_query"):
            return ParsedCommand(
                kind=CommandKind.RECALL,
                subject=extraction.get("query_subject"),
                predicate=extraction.get("query_predicate"),
                raw=raw,
            )

        claims = extraction.get("claims", [])
        if not claims:
            return ParsedCommand(
                kind=CommandKind.UNKNOWN,
                raw=raw,
                error="No claims or query detected.",
            )

        claim = claims[0]
        prov_str = claim.get("provenance", "UserAsserted")
        conf = float(claim.get("value_confidence", 0.9))

        return ParsedCommand(
            kind=CommandKind.INGEST,
            subject=claim.get("subject"),
            predicate=claim.get("predicate"),
            value=str(claim.get("value", "")),
            since=claim.get("valid_from"),
            until=claim.get("valid_until"),
            conf=conf,
            raw=raw,
            extra={
                "provenance_str": prov_str,
                "cardinality": claim.get("cardinality", "Functional"),
                "all_claims": claims,
            },
        )

    def _extract(self, text: str) -> dict[str, Any]:
        response = self._client.messages.create(
            model=_MODEL,
            max_tokens=512,
            system=_EXTRACTION_SYSTEM,
            messages=[{"role": "user", "content": text}],
        )
        content = response.content[0].text.strip()
        return json.loads(content)

    @staticmethod
    def _validate(data: Any) -> bool:
        if not isinstance(data, dict):
            return False
        required = {"claims", "is_query", "query_subject", "query_predicate"}
        return required.issubset(data.keys())
