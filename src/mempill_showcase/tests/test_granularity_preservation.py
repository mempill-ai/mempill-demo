"""
mempill_showcase.tests.test_granularity_preservation — remember_fact preserves the
caller-stated date granularity end-to-end (no silent day-precision upgrade).

Regression for: `_SYSTEM_PROMPT` rule 4 in frameworks/langgraph/graph.py used to
instruct the agent to copy the incumbent's EXACT stored valid_from onto a
correction write, silently upgrading a user-stated month/year into a fabricated
day-precision date. The fix: the write path (write_claim → engine.ingest_claim)
already infers and stores granularity from whatever raw date string is supplied;
this test asserts that inference is correct for year / month / day inputs, using
the stored `valid_time.start_granularity` column (via the raw engine response)
as the definitive source — NOT the rendered display string.
"""
from __future__ import annotations

import pytest

from mempill import ProvenanceLabel
from mempill_showcase.adapters.memory.mempill_adapter import MempillAdapter
from mempill_showcase.config.di import build_mempill_adapter
from mempill_showcase.core.domain.models import ClaimInput
from mempill_showcase.scenarios.seed_data import AGENT_ID


@pytest.fixture()
def mempill_adapter() -> MempillAdapter:
    """Fresh in-memory mempill adapter per test (no seed data needed)."""
    return build_mempill_adapter(in_memory=True)


def _stored_start_granularity(adapter: MempillAdapter, agent_id: str, subject: str, predicate: str) -> str:
    """Read back the DEFINITIVE stored granularity column via the raw engine response.

    BeliefView only exposes the rendered display string (vt_start_display); the
    stored `valid_time.start_granularity` column on the raw engine response is
    the ground truth this test asserts against.
    """
    raw = adapter._engine.query_memory({
        "agent_id": agent_id,
        "subject": subject,
        "predicate": predicate,
    })
    primary = raw["belief"]["primary"]
    return primary["valid_time"]["start_granularity"]


class TestGranularityPreservedOnWrite:
    """remember_fact / write_claim preserves the exact granularity of valid_from."""

    def test_year_granularity_preserved(self, mempill_adapter: MempillAdapter) -> None:
        """valid_from='2025' → stored start_granularity == 'year'."""
        claim = ClaimInput(
            subject="test-subject-year",
            predicate="employer",
            value="Acme Corp / CTO",
            valid_from="2025",
            confidence=1.0,
            provenance=ProvenanceLabel.external_user_asserted(),
            cardinality="Functional",
            criticality="Medium",
        )
        receipt = mempill_adapter.write_claim(AGENT_ID, claim)
        assert receipt.disposition == "CommittedCheap"

        gran = _stored_start_granularity(mempill_adapter, AGENT_ID, "test-subject-year", "employer")
        assert gran == "year", f"Expected stored start_granularity='year', got {gran!r}"

        belief = mempill_adapter.recall(AGENT_ID, "test-subject-year", "employer")
        assert belief.vt_start_display == "2025", (
            f"Expected honest display '2025' (year), got {belief.vt_start_display!r}"
        )

    def test_month_granularity_preserved(self, mempill_adapter: MempillAdapter) -> None:
        """valid_from='2025-03' → stored start_granularity == 'month'."""
        claim = ClaimInput(
            subject="test-subject-month",
            predicate="employer",
            value="Acme Corp / CTO",
            valid_from="2025-03",
            confidence=1.0,
            provenance=ProvenanceLabel.external_user_asserted(),
            cardinality="Functional",
            criticality="Medium",
        )
        receipt = mempill_adapter.write_claim(AGENT_ID, claim)
        assert receipt.disposition == "CommittedCheap"

        gran = _stored_start_granularity(mempill_adapter, AGENT_ID, "test-subject-month", "employer")
        assert gran == "month", f"Expected stored start_granularity='month', got {gran!r}"

        belief = mempill_adapter.recall(AGENT_ID, "test-subject-month", "employer")
        assert belief.vt_start_display == "2025-03", (
            f"Expected honest display '2025-03' (month), got {belief.vt_start_display!r}"
        )

    def test_day_granularity_preserved(self, mempill_adapter: MempillAdapter) -> None:
        """valid_from='2025-03-15' → stored start_granularity == 'day'."""
        claim = ClaimInput(
            subject="test-subject-day",
            predicate="employer",
            value="Acme Corp / CTO",
            valid_from="2025-03-15",
            confidence=1.0,
            provenance=ProvenanceLabel.external_user_asserted(),
            cardinality="Functional",
            criticality="Medium",
        )
        receipt = mempill_adapter.write_claim(AGENT_ID, claim)
        assert receipt.disposition == "CommittedCheap"

        gran = _stored_start_granularity(mempill_adapter, AGENT_ID, "test-subject-day", "employer")
        assert gran == "day", f"Expected stored start_granularity='day', got {gran!r}"

        belief = mempill_adapter.recall(AGENT_ID, "test-subject-day", "employer")
        assert belief.vt_start_display == "2025-03-15", (
            f"Expected honest display '2025-03-15' (day), got {belief.vt_start_display!r}"
        )

    def test_month_write_does_not_upgrade_to_day(self, mempill_adapter: MempillAdapter) -> None:
        """Explicit regression for the bug: a month-granularity write must NEVER be
        stored/rendered as if it were day-granularity (e.g. '2023-06' must not
        become '2023-06-01' in the stored granularity column)."""
        claim = ClaimInput(
            subject="test-subject-no-upgrade",
            predicate="employer",
            value="Acme Corp / CTO",
            valid_from="2023-06",
            confidence=1.0,
            provenance=ProvenanceLabel.external_user_asserted(),
            cardinality="Functional",
            criticality="Medium",
        )
        mempill_adapter.write_claim(AGENT_ID, claim)

        gran = _stored_start_granularity(mempill_adapter, AGENT_ID, "test-subject-no-upgrade", "employer")
        assert gran != "day", (
            f"Bug regression: month-granularity input '2023-06' must not be stored as "
            f"'day' granularity, got {gran!r}"
        )
        assert gran == "month"

        belief = mempill_adapter.recall(AGENT_ID, "test-subject-no-upgrade", "employer")
        assert belief.vt_start_display == "2023-06"
        assert belief.vt_start_display != "2023-06-01"
