"""
mempill_showcase.tests.test_hitl_verdict_normalization — unit tests for
_normalize_verdict() in hitl_node.py.

All tests are purely deterministic (no LLM, no mempill, no DB).
Covers:
  Case 1: Exact keyword Affirm → "Affirm"
  Case 2: Pasted challenger value → "Affirm"
  Case 3: Exact keyword Deny / pasted incumbent value → "Deny"
  Case 4: Abstain and synonyms → "Abstain"
  Case 5: Garbage string → "_invalid_"
"""
from __future__ import annotations

import pytest

from mempill_showcase.frameworks.langgraph.hitl_node import _normalize_verdict

CHALLENGER = "Acme Corp / CTO"
INCUMBENT = "Acme Corp / VP Engineering"


class TestNormalizeVerdictExactKeywords:
    """Case 1: exact keyword (case-insensitive, stripped)."""

    def test_affirm_exact(self):
        assert _normalize_verdict("Affirm", challenger_value=CHALLENGER, incumbent_value=INCUMBENT) == "Affirm"

    def test_affirm_lowercase(self):
        assert _normalize_verdict("affirm", challenger_value=CHALLENGER, incumbent_value=INCUMBENT) == "Affirm"

    def test_affirm_uppercase(self):
        assert _normalize_verdict("AFFIRM", challenger_value=CHALLENGER, incumbent_value=INCUMBENT) == "Affirm"

    def test_affirm_with_whitespace(self):
        assert _normalize_verdict("  Affirm  ", challenger_value=CHALLENGER, incumbent_value=INCUMBENT) == "Affirm"

    def test_deny_exact(self):
        assert _normalize_verdict("Deny", challenger_value=CHALLENGER, incumbent_value=INCUMBENT) == "Deny"

    def test_deny_lowercase(self):
        assert _normalize_verdict("deny", challenger_value=CHALLENGER, incumbent_value=INCUMBENT) == "Deny"

    def test_abstain_exact(self):
        assert _normalize_verdict("Abstain", challenger_value=CHALLENGER, incumbent_value=INCUMBENT) == "Abstain"

    def test_abstain_lowercase(self):
        assert _normalize_verdict("abstain", challenger_value=CHALLENGER, incumbent_value=INCUMBENT) == "Abstain"


class TestNormalizeVerdictSynonyms:
    """Case 1+2: synonym mapping."""

    @pytest.mark.parametrize("syn", ["yes", "accept", "challenger", "approve", "confirm", "correct"])
    def test_affirm_synonyms(self, syn: str):
        assert _normalize_verdict(syn, challenger_value=CHALLENGER, incumbent_value=INCUMBENT) == "Affirm", (
            f"Synonym {syn!r} should map to Affirm"
        )

    @pytest.mark.parametrize("syn", ["no", "reject", "incumbent", "decline", "wrong", "incorrect"])
    def test_deny_synonyms(self, syn: str):
        assert _normalize_verdict(syn, challenger_value=CHALLENGER, incumbent_value=INCUMBENT) == "Deny", (
            f"Synonym {syn!r} should map to Deny"
        )

    @pytest.mark.parametrize("syn", ["defer", "skip", "later", "unsure", "unknown", "pass"])
    def test_abstain_synonyms(self, syn: str):
        assert _normalize_verdict(syn, challenger_value=CHALLENGER, incumbent_value=INCUMBENT) == "Abstain", (
            f"Synonym {syn!r} should map to Abstain"
        )


class TestNormalizeVerdictPastedCandidateValue:
    """Case 2: pasting the candidate VALUE maps to Affirm/Deny.

    This is exactly the user scenario: pasting 'Acme Corp / CTO'
    (the challenger value) should resolve to Affirm.
    """

    def test_pasted_challenger_value_maps_to_affirm(self):
        """The exact user-reported bug: pasting the challenger → Affirm."""
        result = _normalize_verdict(
            "Acme Corp / CTO",
            challenger_value=CHALLENGER,
            incumbent_value=INCUMBENT,
        )
        assert result == "Affirm", (
            f"Pasting challenger value 'Acme Corp / CTO' should map to Affirm, got {result!r}"
        )

    def test_pasted_challenger_value_case_insensitive(self):
        result = _normalize_verdict(
            "acme corp / cto",
            challenger_value=CHALLENGER,
            incumbent_value=INCUMBENT,
        )
        assert result == "Affirm"

    def test_pasted_challenger_value_extra_whitespace(self):
        result = _normalize_verdict(
            "  Acme Corp  /  CTO  ",
            challenger_value=CHALLENGER,
            incumbent_value=INCUMBENT,
        )
        assert result == "Affirm"

    def test_pasted_incumbent_value_maps_to_deny(self):
        result = _normalize_verdict(
            "Acme Corp / VP Engineering",
            challenger_value=CHALLENGER,
            incumbent_value=INCUMBENT,
        )
        assert result == "Deny", (
            f"Pasting incumbent value should map to Deny, got {result!r}"
        )

    def test_pasted_incumbent_value_case_insensitive(self):
        result = _normalize_verdict(
            "acme corp / vp engineering",
            challenger_value=CHALLENGER,
            incumbent_value=INCUMBENT,
        )
        assert result == "Deny"


class TestNormalizeVerdictAbstain:
    """Case 4: Abstain keeps pending_contested (tested here via normalization only)."""

    def test_abstain_keyword(self):
        assert _normalize_verdict("Abstain", challenger_value=CHALLENGER, incumbent_value=INCUMBENT) == "Abstain"

    def test_abstain_defer_synonym(self):
        assert _normalize_verdict("defer", challenger_value=CHALLENGER, incumbent_value=INCUMBENT) == "Abstain"

    def test_abstain_skip_synonym(self):
        assert _normalize_verdict("skip", challenger_value=CHALLENGER, incumbent_value=INCUMBENT) == "Abstain"


class TestNormalizeVerdictInvalid:
    """Case 5: garbage string → '_invalid_' sentinel (never falsely resolve)."""

    def test_garbage_string(self):
        result = _normalize_verdict("xyz", challenger_value=CHALLENGER, incumbent_value=INCUMBENT)
        assert result == "_invalid_", (
            f"Garbage verdict 'xyz' must return '_invalid_', got {result!r}"
        )

    def test_empty_string(self):
        result = _normalize_verdict("", challenger_value=CHALLENGER, incumbent_value=INCUMBENT)
        assert result == "_invalid_"

    def test_random_sentence(self):
        result = _normalize_verdict(
            "I'm not sure what to do here",
            challenger_value=CHALLENGER,
            incumbent_value=INCUMBENT,
        )
        assert result == "_invalid_"

    def test_partial_keyword_not_matched(self):
        """'affirming' is NOT a keyword — should not match."""
        result = _normalize_verdict("affirming", challenger_value=CHALLENGER, incumbent_value=INCUMBENT)
        assert result == "_invalid_"

    def test_partial_value_not_matched(self):
        """Partial candidate value must NOT match — only exact (normalized) match."""
        result = _normalize_verdict("CTO", challenger_value=CHALLENGER, incumbent_value=INCUMBENT)
        assert result == "_invalid_", (
            f"Partial candidate value 'CTO' must not match — got {result!r}"
        )

    def test_none_values_garbage_still_invalid(self):
        """When no candidate values are available, garbage is still invalid."""
        result = _normalize_verdict("xyz", challenger_value=None, incumbent_value=None)
        assert result == "_invalid_"
