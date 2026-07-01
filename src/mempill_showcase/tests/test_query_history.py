"""
mempill_showcase.tests.test_query_history — QueryHistoryTool + adapter.query_history
deterministic tests, plus the ResolveAdjudicationTool stale-handle "already_resolved"
guard.

No API key, no LLM, no external services. Uses real in-memory oracle-backed
mempill engine.

Covers:
  H1. adapter.query_history returns the correct chronologically-folded,
      TRUNCATED, non-overlapping timeline for a scripted Diane -> Joan -> John
      CEO succession — the Superseded entries' valid_until reflects the
      TRUNCATION point (where the next Affirmed claim starts), not the
      original stated end date.
  H2. QueryHistoryTool._run returns the same entries in JSON form.
  H3. QueryHistoryTool._arun parity with _run.
  H4. ResolveAdjudicationTool: resubmitting an already-resolved (stale/orphaned)
      handle_id returns status="already_resolved" (informational), not a raw
      NotFoundError propagating to the caller.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from mempill import ProvenanceLabel
from mempill_showcase.adapters.memory.mempill_adapter import MempillAdapter
from mempill_showcase.config.di import build_mempill_adapter
from mempill_showcase.core.domain.models import ClaimInput
from mempill_showcase.tools.query_history_tool import QueryHistoryTool
from mempill_showcase.tools.resolve_adjudication_tool import ResolveAdjudicationTool

AGENT_ID = "jordan-park-001"


@pytest.fixture()
def oracle_adapter() -> MempillAdapter:
    """Fresh oracle-backed in-memory adapter (open_oracle_in_memory) per test."""
    return build_mempill_adapter(in_memory=True, oracle_backed=True)


def _write_ceo(
    adapter: MempillAdapter,
    value: str,
    valid_from: str,
    valid_until: str | None = None,
) -> str:
    """Write acme-corp/ceo=value; returns the write disposition string."""
    receipt = adapter.write_claim(AGENT_ID, ClaimInput(
        subject="acme-corp",
        predicate="ceo",
        value=value,
        valid_from=valid_from,
        valid_until=valid_until,
        confidence=1.0,
        provenance=ProvenanceLabel.external_user_asserted(),
        cardinality="Functional",
        criticality="Medium",
    ))
    return receipt.disposition


def _affirm_all_pending(adapter: MempillAdapter) -> None:
    """Affirm every currently-pending adjudication (challenger wins each time)."""
    for entry in adapter.list_pending_adjudications(AGENT_ID):
        adapter.submit_adjudication(AGENT_ID, entry["handle_id"], "Affirm")


def _seed_diane_joan_john(adapter: MempillAdapter) -> None:
    """Scripted succession: Diane (open) -> Joan (contest->Affirm) ->
    John (contest->Affirm, overlaps Joan's stated window).

    Expected canonical fold after both Affirms:
      Diane  2021-04 -> 2024-09  Superseded  (truncated from open-ended)
      Joan   2024-09 -> 2025-01  Superseded  (truncated from her OWN stated
                                              end 2025-11, because John was
                                              later Affirmed over the tail)
      John   2025-01 -> None     Current
    """
    assert _write_ceo(adapter, "Diane", "2021-04") == "CommittedCheap"
    assert _write_ceo(adapter, "Joan", "2024-09", "2025-11") == "QueuedForAdjudication"
    _affirm_all_pending(adapter)  # Joan wins over Diane
    assert _write_ceo(adapter, "John", "2025-01", "2026-12") == "QueuedForAdjudication"
    _affirm_all_pending(adapter)  # John wins over Joan


# ── H1: adapter.query_history — correct truncated, non-overlapping fold ─────

class TestAdapterQueryHistory:
    def test_scripted_succession_truncates_correctly(self, oracle_adapter: MempillAdapter) -> None:
        _seed_diane_joan_john(oracle_adapter)

        entries = oracle_adapter.query_history(AGENT_ID, "acme-corp", "ceo")
        assert len(entries) == 3, f"expected 3 folded entries, got {entries}"

        diane, joan, john = entries
        assert diane["value"] == "Diane"
        assert diane["status"] == "Superseded"
        assert diane["valid_until"] is not None, "Diane's open end must be TRUNCATED, not left open"
        assert diane["valid_until"].startswith("2024-09"), (
            f"Diane's valid_until must be truncated to Joan's start (2024-09), got {diane['valid_until']!r}"
        )

        assert joan["value"] == "Joan"
        assert joan["status"] == "Superseded"
        assert joan["valid_until"] is not None
        assert joan["valid_until"].startswith("2025-01"), (
            "Joan's valid_until must be TRUNCATED to John's start (2025-01) — NOT her "
            f"originally-stated end (2025-11). Got {joan['valid_until']!r}"
        )
        assert not joan["valid_until"].startswith("2025-11"), (
            "Joan's ORIGINALLY-STATED end (2025-11) must NOT leak through as valid_until "
            "— the fold must reflect the truncation caused by John's later Affirm."
        )

        assert john["value"] == "John"
        assert john["status"] == "Current"
        assert john["valid_until"] is None, "John (current CEO) must be open-ended"

    def test_ordering_is_oldest_to_newest(self, oracle_adapter: MempillAdapter) -> None:
        _seed_diane_joan_john(oracle_adapter)
        entries = oracle_adapter.query_history(AGENT_ID, "acme-corp", "ceo")
        values = [e["value"] for e in entries]
        assert values == ["Diane", "Joan", "John"], f"expected oldest->newest order, got {values}"

    def test_empty_history_returns_empty_list(self, oracle_adapter: MempillAdapter) -> None:
        entries = oracle_adapter.query_history(AGENT_ID, "nobody", "role")
        assert entries == []


# ── H2/H3: QueryHistoryTool ───────────────────────────────────────────────────

class TestQueryHistoryTool:
    def test_run_returns_correct_truncated_chronology(self, oracle_adapter: MempillAdapter) -> None:
        _seed_diane_joan_john(oracle_adapter)
        tool = QueryHistoryTool(adapter=oracle_adapter)

        raw = tool.invoke({"agent_id": AGENT_ID, "subject": "acme-corp", "predicate": "ceo"})
        result = json.loads(raw)

        assert result["subject"] == "acme-corp"
        assert result["predicate"] == "ceo"
        entries = result["entries"]
        assert [e["value"] for e in entries] == ["Diane", "Joan", "John"]
        assert entries[0]["status"] == "Superseded"
        assert entries[0]["valid_until"].startswith("2024-09")
        assert entries[1]["status"] == "Superseded"
        assert entries[1]["valid_until"].startswith("2025-01")
        assert entries[2]["status"] == "Current"
        assert entries[2]["valid_until"] is None

    def test_subject_predicate_are_normalised(self, oracle_adapter: MempillAdapter) -> None:
        _seed_diane_joan_john(oracle_adapter)
        tool = QueryHistoryTool(adapter=oracle_adapter)
        raw = tool.invoke({"agent_id": AGENT_ID, "subject": "Acme Corp", "predicate": "CEO"})
        result = json.loads(raw)
        assert result["subject"] == "acme-corp"
        assert result["predicate"] == "ceo"
        assert len(result["entries"]) == 3

    def test_arun_matches_run(self, oracle_adapter: MempillAdapter) -> None:
        _seed_diane_joan_john(oracle_adapter)
        tool = QueryHistoryTool(adapter=oracle_adapter)

        sync_result = json.loads(tool.invoke({
            "agent_id": AGENT_ID, "subject": "acme-corp", "predicate": "ceo",
        }))
        async_result = json.loads(asyncio.run(tool.ainvoke({
            "agent_id": AGENT_ID, "subject": "acme-corp", "predicate": "ceo",
        })))
        assert sync_result == async_result


# ── H4: ResolveAdjudicationTool — stale handle is informational, not an error ─

class TestResolveAdjudicationStaleHandleGuard:
    """Simulates a handle that goes stale/orphaned BETWEEN the tool's own
    list_pending_adjudications lookup and its submit_adjudication call — e.g.
    another concurrent resolution path (hitl_node) resolves the same
    subject-line first. The tool must surface this as INFORMATIONAL
    status='already_resolved', never a raw NotFoundError."""

    def test_orphaned_handle_between_lookup_and_submit_is_informational(
        self, oracle_adapter: MempillAdapter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert _write_ceo(oracle_adapter, "Diane", "2021-04") == "CommittedCheap"
        assert _write_ceo(oracle_adapter, "John", "2023-01") == "QueuedForAdjudication"
        assert _write_ceo(oracle_adapter, "Carrie", "2024-01") == "QueuedForAdjudication"

        pending = oracle_adapter.list_pending_adjudications(AGENT_ID)
        john_handle = next(e["handle_id"] for e in pending if e["challenger_value"] == "John")

        resolve_tool = ResolveAdjudicationTool(adapter=oracle_adapter)

        # Simulate a race: by the time submit_adjudication is actually called,
        # John's handle has already been orphaned (denied) by another path
        # (e.g. hitl_node's queue-collapse resolving Carrie first).
        real_submit = oracle_adapter.submit_adjudication

        def _racy_submit(agent_id, handle_id, verdict):
            if handle_id == john_handle and not getattr(_racy_submit, "_orphaned", False):
                _racy_submit._orphaned = True
                real_submit(agent_id, handle_id, "Deny")  # orphan it out-of-band
            return real_submit(agent_id, handle_id, verdict)

        monkeypatch.setattr(oracle_adapter, "submit_adjudication", _racy_submit)

        result = json.loads(resolve_tool.invoke({
            "agent_id": AGENT_ID, "handle_id": john_handle, "verdict": "Affirm",
        }))
        assert result["status"] == "already_resolved", (
            f"expected informational already_resolved for orphaned handle, got {result}"
        )
        assert "message" in result
        assert "current_value" in result

    def test_arun_matches_run_for_already_resolved(
        self, oracle_adapter: MempillAdapter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert _write_ceo(oracle_adapter, "Diane", "2021-04") == "CommittedCheap"
        assert _write_ceo(oracle_adapter, "John", "2023-01") == "QueuedForAdjudication"
        assert _write_ceo(oracle_adapter, "Carrie", "2024-01") == "QueuedForAdjudication"

        pending = oracle_adapter.list_pending_adjudications(AGENT_ID)
        john_handle = next(e["handle_id"] for e in pending if e["challenger_value"] == "John")

        resolve_tool = ResolveAdjudicationTool(adapter=oracle_adapter)
        real_submit = oracle_adapter.submit_adjudication

        def _racy_submit(agent_id, handle_id, verdict):
            if handle_id == john_handle and not getattr(_racy_submit, "_orphaned", False):
                _racy_submit._orphaned = True
                real_submit(agent_id, handle_id, "Deny")
            return real_submit(agent_id, handle_id, verdict)

        monkeypatch.setattr(oracle_adapter, "submit_adjudication", _racy_submit)

        result = json.loads(asyncio.run(resolve_tool.ainvoke({
            "agent_id": AGENT_ID, "handle_id": john_handle, "verdict": "Affirm",
        })))
        assert result["status"] == "already_resolved"

    def test_genuinely_unknown_handle_still_returns_not_found(
        self, oracle_adapter: MempillAdapter
    ) -> None:
        """A handle_id that was never queued (or already gone from the list
        entirely) still returns the pre-existing status='not_found' path —
        this test guards against regressing that behaviour."""
        resolve_tool = ResolveAdjudicationTool(adapter=oracle_adapter)
        result = json.loads(resolve_tool.invoke({
            "agent_id": AGENT_ID,
            "handle_id": "00000000-0000-0000-0000-000000000000",
            "verdict": "Affirm",
        }))
        assert result["status"] == "not_found"
