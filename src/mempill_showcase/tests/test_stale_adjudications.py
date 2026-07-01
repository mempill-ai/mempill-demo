"""
mempill_showcase.tests.test_stale_adjudications — stale pending adjudication
surfacing + resolution + queue-collapse tests.

Proves (deterministic, no LLM):
  1. ListPendingAdjudicationsTool returns MULTIPLE stale handles when several
     contested writes stack on the same subject/predicate.
  2. ResolveAdjudicationTool clears a SPECIFIC stale handle by handle_id (its
     status transitions out of the pending queue).
  3. Queue-collapse: seed acme-corp/ceo=Diane, write John (contested, pending),
     write Carrie (contested, pending) → 2+ pending rows. Resolving the
     subject-line with winner=Carrie via ResolveAdjudicationTool collapses the
     WHOLE fan to exactly ONE live belief (Resolved, not Contested) with ZERO
     pending rows remaining for acme-corp/ceo.
  4. Both new tools implement _run AND _arun (async parity for LangGraph Studio).
  5. hitl_node._resolve_via_oracle_or_reconcile applies the SAME queue-collapse
     policy directly (the live-interrupt path), producing the same convergence.

Background: tasks/20-multiagent-showcase/RESEARCH_STALE_ADJUDICATIONS.md.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from mempill import ProvenanceLabel
from mempill_showcase.adapters.memory.mempill_adapter import MempillAdapter
from mempill_showcase.config.di import build_mempill_adapter
from mempill_showcase.core.domain.models import ClaimInput
from mempill_showcase.frameworks.langgraph.hitl_node import _resolve_via_oracle_or_reconcile
from mempill_showcase.tools.list_pending_adjudications_tool import ListPendingAdjudicationsTool
from mempill_showcase.tools.resolve_adjudication_tool import ResolveAdjudicationTool

AGENT_ID = "jordan-park-001"


@pytest.fixture()
def oracle_adapter() -> MempillAdapter:
    """Fresh oracle-backed in-memory adapter (open_oracle_in_memory) per test."""
    return build_mempill_adapter(in_memory=True, oracle_backed=True)


def _write_ceo(adapter: MempillAdapter, value: str, valid_from: str | None = None) -> str:
    """Write acme-corp/ceo=value; returns the write disposition string."""
    receipt = adapter.write_claim(AGENT_ID, ClaimInput(
        subject="acme-corp",
        predicate="ceo",
        value=value,
        valid_from=valid_from,
        confidence=0.9,
        provenance=ProvenanceLabel.external_user_asserted(),
        cardinality="Functional",
    ))
    return receipt.disposition


# ── ListPendingAdjudicationsTool ─────────────────────────────────────────────

class TestListPendingAdjudicationsTool:
    def test_returns_multiple_stale_handles(self, oracle_adapter: MempillAdapter) -> None:
        """Seeding a subject-line then stacking two contested writes surfaces BOTH
        pending rows — including the stale second one no interrupt ever saw."""
        assert _write_ceo(oracle_adapter, "Diane", "2020-01") == "CommittedCheap"
        assert _write_ceo(oracle_adapter, "John", "2023-01") == "QueuedForAdjudication"
        assert _write_ceo(oracle_adapter, "Carrie", "2024-01") == "QueuedForAdjudication"

        tool = ListPendingAdjudicationsTool(adapter=oracle_adapter)
        result = json.loads(tool.invoke({"agent_id": AGENT_ID}))

        assert result["pending_count"] >= 2, f"expected >=2 pending, got {result}"
        challengers = {e["challenger_value"] for e in result["pending"]}
        assert "John" in challengers
        assert "Carrie" in challengers
        for entry in result["pending"]:
            assert entry["handle_id"]
            assert entry["subject"] == "acme-corp"
            assert entry["predicate"] == "ceo"
            assert entry["status"] == "pending"

    def test_empty_queue_returns_empty_list(self, oracle_adapter: MempillAdapter) -> None:
        tool = ListPendingAdjudicationsTool(adapter=oracle_adapter)
        result = json.loads(tool.invoke({"agent_id": AGENT_ID}))
        assert result["pending_count"] == 0
        assert result["pending"] == []

    def test_arun_matches_run(self, oracle_adapter: MempillAdapter) -> None:
        _write_ceo(oracle_adapter, "Diane", "2020-01")
        _write_ceo(oracle_adapter, "John", "2023-01")
        tool = ListPendingAdjudicationsTool(adapter=oracle_adapter)
        sync_result = json.loads(tool.invoke({"agent_id": AGENT_ID}))
        async_result = json.loads(asyncio.run(tool.ainvoke({"agent_id": AGENT_ID})))
        assert sync_result["pending_count"] == async_result["pending_count"]


# ── ResolveAdjudicationTool ───────────────────────────────────────────────────

class TestResolveAdjudicationTool:
    def test_resolves_specific_stale_handle_by_id(self, oracle_adapter: MempillAdapter) -> None:
        """Deny a SPECIFIC stale handle by id — it clears from the pending queue."""
        _write_ceo(oracle_adapter, "Diane", "2020-01")
        _write_ceo(oracle_adapter, "John", "2023-01")

        list_tool = ListPendingAdjudicationsTool(adapter=oracle_adapter)
        pending = json.loads(list_tool.invoke({"agent_id": AGENT_ID}))["pending"]
        assert len(pending) == 1
        handle_id = pending[0]["handle_id"]

        resolve_tool = ResolveAdjudicationTool(adapter=oracle_adapter)
        result = json.loads(resolve_tool.invoke({
            "agent_id": AGENT_ID,
            "handle_id": handle_id,
            "verdict": "Deny",
        }))
        assert result["status"] == "resolved"
        assert result["verdict"] == "Deny"
        assert result["winning_value"] == "Diane"

        pending_after = json.loads(list_tool.invoke({"agent_id": AGENT_ID}))["pending"]
        assert pending_after == [], "resolved handle must clear from the pending queue"

    def test_unknown_handle_id_returns_not_found(self, oracle_adapter: MempillAdapter) -> None:
        tool = ResolveAdjudicationTool(adapter=oracle_adapter)
        result = json.loads(tool.invoke({
            "agent_id": AGENT_ID,
            "handle_id": "00000000-0000-0000-0000-000000000000",
            "verdict": "Affirm",
        }))
        assert result["status"] == "not_found"

    def test_forgiving_verdict_synonym(self, oracle_adapter: MempillAdapter) -> None:
        """'yes' synonym maps to Affirm via the shared _normalize_verdict."""
        _write_ceo(oracle_adapter, "Diane", "2020-01")
        _write_ceo(oracle_adapter, "John", "2023-01")
        list_tool = ListPendingAdjudicationsTool(adapter=oracle_adapter)
        handle_id = json.loads(list_tool.invoke({"agent_id": AGENT_ID}))["pending"][0]["handle_id"]

        resolve_tool = ResolveAdjudicationTool(adapter=oracle_adapter)
        result = json.loads(resolve_tool.invoke({
            "agent_id": AGENT_ID,
            "handle_id": handle_id,
            "verdict": "yes",
        }))
        assert result["verdict"] == "Affirm"
        assert result["winning_value"] == "John"

    def test_pasted_candidate_value_verdict(self, oracle_adapter: MempillAdapter) -> None:
        """Pasting the exact challenger value resolves as Affirm."""
        _write_ceo(oracle_adapter, "Diane", "2020-01")
        _write_ceo(oracle_adapter, "John", "2023-01")
        list_tool = ListPendingAdjudicationsTool(adapter=oracle_adapter)
        handle_id = json.loads(list_tool.invoke({"agent_id": AGENT_ID}))["pending"][0]["handle_id"]

        resolve_tool = ResolveAdjudicationTool(adapter=oracle_adapter)
        result = json.loads(resolve_tool.invoke({
            "agent_id": AGENT_ID,
            "handle_id": handle_id,
            "verdict": "John",
        }))
        assert result["verdict"] == "Affirm"
        assert result["winning_value"] == "John"

    def test_arun_matches_run(self, oracle_adapter: MempillAdapter) -> None:
        _write_ceo(oracle_adapter, "Diane", "2020-01")
        _write_ceo(oracle_adapter, "John", "2023-01")
        list_tool = ListPendingAdjudicationsTool(adapter=oracle_adapter)
        handle_id = json.loads(list_tool.invoke({"agent_id": AGENT_ID}))["pending"][0]["handle_id"]

        resolve_tool = ResolveAdjudicationTool(adapter=oracle_adapter)
        result = json.loads(asyncio.run(resolve_tool.ainvoke({
            "agent_id": AGENT_ID,
            "handle_id": handle_id,
            "verdict": "Affirm",
        })))
        assert result["status"] == "resolved"


# ── Queue-collapse convergence ────────────────────────────────────────────────

class TestQueueCollapse:
    """The exact user-reported scenario: multiple stacked CEO conflicts must
    collapse to a SINGLE live belief with zero stale pending rows."""

    def test_resolve_adjudication_collapses_whole_subject_line(
        self, oracle_adapter: MempillAdapter
    ) -> None:
        # Seed acme-corp/ceo=Diane, then two competing successors both contest.
        assert _write_ceo(oracle_adapter, "Diane", "2020-01") == "CommittedCheap"
        assert _write_ceo(oracle_adapter, "John", "2023-01") == "QueuedForAdjudication"
        assert _write_ceo(oracle_adapter, "Carrie", "2024-01") == "QueuedForAdjudication"

        list_tool = ListPendingAdjudicationsTool(adapter=oracle_adapter)
        before = json.loads(list_tool.invoke({"agent_id": AGENT_ID}))
        assert before["pending_count"] == 2, f"expected 2 pending before resolution: {before}"

        carrie_handle = next(
            e["handle_id"] for e in before["pending"] if e["challenger_value"] == "Carrie"
        )

        resolve_tool = ResolveAdjudicationTool(adapter=oracle_adapter)
        outcome = json.loads(resolve_tool.invoke({
            "agent_id": AGENT_ID,
            "handle_id": carrie_handle,
            "verdict": "Affirm",  # Carrie wins
        }))
        assert outcome["status"] == "resolved"
        assert outcome["winning_value"] == "Carrie"
        assert outcome["swept_count"] == 1, "the John row must be swept (Denied)"

        after = json.loads(list_tool.invoke({"agent_id": AGENT_ID}))
        assert after["pending_count"] == 0, (
            f"queue-collapse must leave ZERO pending rows for acme-corp/ceo: {after}"
        )

        belief = oracle_adapter.recall(AGENT_ID, "acme-corp", "ceo")
        assert belief.value == "Carrie"
        assert belief.status == "Resolved", (
            f"expected a single converged Resolved belief, got status={belief.status!r}"
        )
        assert not belief.is_contested()
        assert belief.alternatives == [] or all(
            a.value == "Carrie" for a in belief.alternatives
        ), f"no lingering alternative live winners expected: {belief.alternatives}"

    def test_hitl_node_helper_applies_same_collapse_policy(
        self, oracle_adapter: MempillAdapter
    ) -> None:
        """_resolve_via_oracle_or_reconcile (the live-interrupt path) uses the
        identical queue-collapse policy — same convergence guarantee."""
        _write_ceo(oracle_adapter, "Diane", "2020-01")
        _write_ceo(oracle_adapter, "John", "2023-01")
        _write_ceo(oracle_adapter, "Carrie", "2024-01")

        winning_value, disposition = _resolve_via_oracle_or_reconcile(
            oracle_adapter, AGENT_ID, "acme-corp", "ceo", "Affirm", claim_refs=[],
        )
        assert winning_value in ("John", "Carrie")  # first matching entry decides
        assert disposition == "CommittedCheap"

        pending_after = oracle_adapter.list_pending_adjudications(AGENT_ID)
        assert pending_after == [], "hitl_node collapse must leave zero pending rows"

        belief = oracle_adapter.recall(AGENT_ID, "acme-corp", "ceo")
        assert not belief.is_contested()
        assert belief.status == "Resolved"
