"""
mempill_showcase.tests.test_w7_oracle — W7 end-to-end oracle tests.

Proves the REAL mempill HITL oracle path end-to-end without any LLM:

  1. Oracle engine construction: build_mempill_adapter(oracle_backed=True) creates
     an open_oracle_in_memory engine; adapter gains list_pending_adjudications /
     submit_adjudication methods.

  2. Non-oracle engine: build_mempill_adapter(oracle_backed=False) raises
     AttributeError on oracle methods (clean boundary).

  3. Affirm path: ingest genuine conflict → list_pending → interrupt → resume Affirm →
     submit_adjudication → recall returns Resolved (challenger wins).

  4. Deny path: same flow but resume Deny → recall returns Resolved (incumbent wins).

  5. Graph HITL: crew_a write creates QueuedForAdjudication → graph pauses at
     hitl_node → Command(resume='Affirm') → hitl_node calls submit_adjudication →
     post-resume recall is Resolved (challenger) via recall_tool.

  6. CLI oracle: CliOracle.request_adjudication returns a UUID; display/prompt methods
     are callable without errors.

All tests are CI-safe (no API key, no LLM, no LangSmith required).
"""
from __future__ import annotations

import json
import uuid

import pytest

from mempill import ProvenanceLabel
from mempill_showcase.adapters.memory.mempill_adapter import MempillAdapter
from mempill_showcase.config.di import build_mempill_adapter, build_tools
from mempill_showcase.core.domain.models import ClaimInput
from mempill_showcase.frameworks.langgraph.graph import build_graph
from mempill_showcase.scenarios.seed_data import AGENT_ID
from langgraph.types import Command


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def oracle_adapter() -> MempillAdapter:
    """Oracle-backed in-memory adapter (open_oracle_in_memory)."""
    return build_mempill_adapter(in_memory=True, oracle_backed=True)


@pytest.fixture()
def plain_adapter() -> MempillAdapter:
    """Non-oracle in-memory adapter (open_in_memory)."""
    return build_mempill_adapter(in_memory=True, oracle_backed=False)


def _seed_vp(adapter: MempillAdapter) -> None:
    """Seed alice-chen/employer = VP Engineering (open-ended, 2023-06)."""
    adapter.write_claim(AGENT_ID, ClaimInput(
        subject="alice-chen",
        predicate="employer",
        value="Acme Corp / VP Engineering",
        valid_from="2023-06",
        confidence=1.0,
        provenance=ProvenanceLabel.external_user_asserted(),
        cardinality="Functional",
    ))


def _write_cto_conflict(adapter: MempillAdapter) -> str:
    """Write CTO (same valid_from → genuine overlap → QueuedForAdjudication).

    Returns the write disposition string.
    """
    receipt = adapter.write_claim(AGENT_ID, ClaimInput(
        subject="alice-chen",
        predicate="employer",
        value="Acme Corp / CTO",
        valid_from="2023-06",
        confidence=0.9,
        provenance=ProvenanceLabel.external_first_hand(),
        cardinality="Functional",
    ))
    return receipt.disposition


def _cfg(thread_id: str | None = None) -> dict:
    return {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}


# ── TestOracleEngineConstruction ──────────────────────────────────────────────

class TestOracleEngineConstruction:
    """build_mempill_adapter oracle_backed flag wires the correct engine."""

    def test_oracle_adapter_has_list_pending(self, oracle_adapter: MempillAdapter) -> None:
        """oracle_backed=True adapter has list_pending_adjudications method."""
        assert hasattr(oracle_adapter, "list_pending_adjudications"), (
            "Oracle-backed adapter must expose list_pending_adjudications"
        )

    def test_oracle_adapter_has_submit_adjudication(self, oracle_adapter: MempillAdapter) -> None:
        """oracle_backed=True adapter has submit_adjudication method."""
        assert hasattr(oracle_adapter, "submit_adjudication"), (
            "Oracle-backed adapter must expose submit_adjudication"
        )

    def test_oracle_adapter_list_pending_empty_on_fresh(self, oracle_adapter: MempillAdapter) -> None:
        """list_pending_adjudications returns empty list on a fresh engine."""
        pending = oracle_adapter.list_pending_adjudications(AGENT_ID)
        assert isinstance(pending, list)
        assert len(pending) == 0

    def test_plain_adapter_list_pending_raises(self, plain_adapter: MempillAdapter) -> None:
        """oracle_backed=False adapter raises AttributeError on list_pending_adjudications."""
        with pytest.raises(AttributeError, match="oracle"):
            plain_adapter.list_pending_adjudications(AGENT_ID)

    def test_plain_adapter_submit_raises(self, plain_adapter: MempillAdapter) -> None:
        """oracle_backed=False adapter raises AttributeError on submit_adjudication."""
        with pytest.raises(AttributeError, match="oracle"):
            plain_adapter.submit_adjudication(AGENT_ID, "fake-handle", "Affirm")

    def test_oracle_engine_has_submit_adjudication(self, oracle_adapter: MempillAdapter) -> None:
        """The underlying oracle engine exposes submit_adjudication directly."""
        assert hasattr(oracle_adapter._engine, "submit_adjudication"), (
            "PyOracleEngine must have submit_adjudication"
        )


# ── TestOracleAffirmPath ──────────────────────────────────────────────────────

class TestOracleAffirmPath:
    """Ingest conflict → queue → submit Affirm → recall Resolved (challenger wins)."""

    def test_conflict_creates_queued_entry(self, oracle_adapter: MempillAdapter) -> None:
        """A genuine temporal overlap creates a QueuedForAdjudication entry in the queue."""
        _seed_vp(oracle_adapter)
        disposition = _write_cto_conflict(oracle_adapter)
        assert disposition == "QueuedForAdjudication", (
            f"Oracle engine must return QueuedForAdjudication for a genuine conflict, "
            f"got {disposition!r}"
        )

    def test_list_pending_returns_handle(self, oracle_adapter: MempillAdapter) -> None:
        """After a conflict, list_pending_adjudications returns a handle_id."""
        _seed_vp(oracle_adapter)
        _write_cto_conflict(oracle_adapter)
        pending = oracle_adapter.list_pending_adjudications(AGENT_ID)
        assert len(pending) >= 1, "At least one pending adjudication must exist"
        entry = pending[0]
        assert entry.get("handle_id"), "Pending entry must have a handle_id"
        assert entry.get("subject") == "alice-chen"
        assert entry.get("predicate") == "employer"

    def test_affirm_resolves_challenger_wins(self, oracle_adapter: MempillAdapter) -> None:
        """submit_adjudication(Affirm) → recall returns Resolved (CTO, the challenger)."""
        _seed_vp(oracle_adapter)
        _write_cto_conflict(oracle_adapter)

        pending = oracle_adapter.list_pending_adjudications(AGENT_ID)
        handle_id = pending[0]["handle_id"]

        result = oracle_adapter.submit_adjudication(AGENT_ID, handle_id, "Affirm")
        assert result.get("disposition") in ("CommittedCheap", "Resolved"), (
            f"Affirm should commit the challenger, got {result.get('disposition')!r}"
        )

        belief = oracle_adapter.recall(AGENT_ID, "alice-chen", "employer")
        assert belief.status == "Resolved", (
            f"After Affirm, belief must be Resolved, got {belief.status!r}"
        )
        assert belief.value == "Acme Corp / CTO", (
            f"Challenger (CTO) should win after Affirm, got {belief.value!r}"
        )

    def test_affirm_clears_pending_queue(self, oracle_adapter: MempillAdapter) -> None:
        """After Affirm, the oracle queue is empty for this agent/subject/predicate."""
        _seed_vp(oracle_adapter)
        _write_cto_conflict(oracle_adapter)

        pending = oracle_adapter.list_pending_adjudications(AGENT_ID)
        handle_id = pending[0]["handle_id"]
        oracle_adapter.submit_adjudication(AGENT_ID, handle_id, "Affirm")

        pending_after = oracle_adapter.list_pending_adjudications(AGENT_ID)
        assert len(pending_after) == 0, (
            f"Oracle queue must be empty after Affirm, got {len(pending_after)} remaining"
        )


# ── TestOracleDenyPath ────────────────────────────────────────────────────────

class TestOracleDenyPath:
    """Ingest conflict → queue → submit Deny → recall Resolved (incumbent wins)."""

    def test_deny_resolves_incumbent_wins(self, oracle_adapter: MempillAdapter) -> None:
        """submit_adjudication(Deny) → recall returns Resolved (VP Engineering, the incumbent)."""
        _seed_vp(oracle_adapter)
        _write_cto_conflict(oracle_adapter)

        pending = oracle_adapter.list_pending_adjudications(AGENT_ID)
        handle_id = pending[0]["handle_id"]

        result = oracle_adapter.submit_adjudication(AGENT_ID, handle_id, "Deny")
        assert result.get("disposition") in ("Superseded", "Rejected"), (
            f"Deny should supersede/reject the challenger, got {result.get('disposition')!r}"
        )

        belief = oracle_adapter.recall(AGENT_ID, "alice-chen", "employer")
        assert belief.status == "Resolved", (
            f"After Deny, belief must be Resolved, got {belief.status!r}"
        )
        assert belief.value == "Acme Corp / VP Engineering", (
            f"Incumbent (VP Engineering) should win after Deny, got {belief.value!r}"
        )

    def test_deny_clears_pending_queue(self, oracle_adapter: MempillAdapter) -> None:
        """After Deny, the oracle queue is empty."""
        _seed_vp(oracle_adapter)
        _write_cto_conflict(oracle_adapter)

        pending = oracle_adapter.list_pending_adjudications(AGENT_ID)
        oracle_adapter.submit_adjudication(AGENT_ID, pending[0]["handle_id"], "Deny")

        pending_after = oracle_adapter.list_pending_adjudications(AGENT_ID)
        assert len(pending_after) == 0


# ── TestGraphHITLRealOracle ───────────────────────────────────────────────────

@pytest.mark.live
class TestGraphHITLRealOracle:
    """End-to-end graph HITL via real oracle: LLM writes CTO → interrupt → resume → Resolved.

    Wave B NOTE: The ReAct graph no longer has a deterministic supervisor_node that
    routes to hitl_node after a contested write.  In the new design the HITL interrupt
    is fired from inside the request_adjudication tool, which the LLM must choose to
    call after observing is_contested=True from remember_fact.  These tests therefore
    require a live LLM (ANTHROPIC_API_KEY) to drive the ReAct loop.

    All tests in this class are marked @pytest.mark.live (skipped with -m "not live").

    The oracle mechanics (queue, Affirm, Deny) are proven without a graph in
    TestOracleAffirmPath / TestOracleDenyPath above.
    """

    @pytest.fixture()
    def graph_fixture(self, oracle_adapter: MempillAdapter):
        """Oracle-backed adapter + graph."""
        tools = build_tools(oracle_adapter)
        app = build_graph(adapter=oracle_adapter, tools=tools)
        return oracle_adapter, tools, app

    def test_crew_a_conflict_pauses_at_hitl(self, graph_fixture) -> None:
        """LLM write of CTO (same valid_from as VP Eng) → graph pauses at request_adjudication."""
        from langchain_core.messages import HumanMessage
        adapter, _, app = graph_fixture
        _seed_vp(adapter)

        cfg = _cfg()
        result = app.invoke(
            {"messages": [HumanMessage(content="Alice was promoted to CTO at Acme since 2023-06")]},
            cfg,
        )
        assert "__interrupt__" in result, (
            "Graph must pause at request_adjudication interrupt after contested write"
        )
        interrupts = result["__interrupt__"]
        assert len(interrupts) >= 1
        payload = interrupts[0].value
        assert "subject" in payload or "question" in payload or "incumbent" in payload, (
            f"Interrupt payload must describe the conflict, got {payload!r}"
        )

    def test_affirm_via_command_resolves_belief(self, graph_fixture) -> None:
        """Command(resume='Affirm') → request_adjudication resolves → CTO wins."""
        from langchain_core.messages import HumanMessage
        adapter, _, app = graph_fixture
        _seed_vp(adapter)

        cfg = _cfg()
        result1 = app.invoke(
            {"messages": [HumanMessage(content="Alice was promoted to CTO at Acme since 2023-06")]},
            cfg,
        )
        assert "__interrupt__" in result1, "Graph must pause before resume test"

        # Resume with Affirm — the tool resolves and the agent continues to END
        result2 = app.invoke(Command(resume="Affirm"), cfg)

        # Post-resume recall via adapter must be Resolved (challenger = CTO wins)
        belief = adapter.recall(AGENT_ID, "alice-chen", "employer")
        assert belief.status == "Resolved", (
            f"Belief must be Resolved after oracle Affirm, got {belief.status!r}"
        )
        assert belief.value == "Acme Corp / CTO", (
            f"CTO (challenger) must win after Affirm, got {belief.value!r}"
        )

    def test_deny_via_command_keeps_incumbent(self, graph_fixture) -> None:
        """Command(resume='Deny') → request_adjudication resolves → VP Engineering wins."""
        from langchain_core.messages import HumanMessage
        adapter, _, app = graph_fixture
        _seed_vp(adapter)

        cfg = _cfg()
        result1 = app.invoke(
            {"messages": [HumanMessage(content="Alice was promoted to CTO at Acme since 2023-06")]},
            cfg,
        )
        assert "__interrupt__" in result1, "Graph must pause before resume test"

        # Resume with Deny — incumbent survives
        app.invoke(Command(resume="Deny"), cfg)

        # VP Engineering (incumbent) must survive
        belief = adapter.recall(AGENT_ID, "alice-chen", "employer")
        assert belief.status == "Resolved", (
            f"Belief must be Resolved after oracle Deny, got {belief.status!r}"
        )
        assert belief.value == "Acme Corp / VP Engineering", (
            f"VP Engineering (incumbent) must win after Deny, got {belief.value!r}"
        )

    def test_oracle_queue_empty_after_graph_resolution(self, graph_fixture) -> None:
        """After graph HITL resolution, the oracle queue is empty for this agent."""
        from langchain_core.messages import HumanMessage
        adapter, _, app = graph_fixture
        _seed_vp(adapter)

        cfg = _cfg()
        app.invoke(
            {"messages": [HumanMessage(content="Alice was promoted to CTO at Acme since 2023-06")]},
            cfg,
        )
        app.invoke(Command(resume="Affirm"), cfg)

        pending = adapter.list_pending_adjudications(AGENT_ID)
        assert len(pending) == 0, (
            f"Oracle queue must be empty after graph HITL resolution, "
            f"got {len(pending)} remaining"
        )

    def test_no_pending_before_conflict_ingested(self, graph_fixture) -> None:
        """No pending adjudications on a fresh adapter (no conflicts yet)."""
        adapter, _, app = graph_fixture
        pending = adapter.list_pending_adjudications(AGENT_ID)
        assert len(pending) == 0, "Fresh adapter must have empty oracle queue"


# ── TestCliOracle ─────────────────────────────────────────────────────────────

class TestCliOracle:
    """CliOracle: request_adjudication returns a UUID; display/prompt work without error."""

    def test_request_adjudication_returns_uuid(self) -> None:
        """CliOracle.request_adjudication returns a valid UUID string."""
        from mempill_showcase.adapters.oracle.cli_oracle import CliOracle
        oracle = CliOracle()
        handle = oracle.request_adjudication("agent-001", {"some": "request"})
        # Must be a valid UUID
        try:
            uuid.UUID(handle)
        except ValueError:
            pytest.fail(f"request_adjudication must return a UUID, got {handle!r}")

    def test_request_adjudication_unique_per_call(self) -> None:
        """Each call to request_adjudication returns a unique UUID."""
        from mempill_showcase.adapters.oracle.cli_oracle import CliOracle
        oracle = CliOracle()
        h1 = oracle.request_adjudication("agent-001", {})
        h2 = oracle.request_adjudication("agent-001", {})
        assert h1 != h2, "Each adjudication must get a unique handle_id"

    def test_display_pending_no_error_empty(self, capsys) -> None:
        """display_pending([]) prints 'No pending adjudications.' without error."""
        from mempill_showcase.adapters.oracle.cli_oracle import CliOracle
        import io
        oracle = CliOracle()
        buf = io.StringIO()
        oracle.display_pending([], file=buf)
        output = buf.getvalue()
        assert "No pending" in output

    def test_display_pending_shows_handle_and_values(self, capsys) -> None:
        """display_pending shows handle, subject/predicate, and incumbent/challenger values."""
        from mempill_showcase.adapters.oracle.cli_oracle import CliOracle
        import io
        oracle = CliOracle()
        fake_entries = [{
            "handle_id": "abc-123-def",
            "subject": "alice-chen",
            "predicate": "employer",
            "incumbent_value": "VP Engineering",
            "challenger_value": "CTO",
        }]
        buf = io.StringIO()
        oracle.display_pending(fake_entries, file=buf)
        output = buf.getvalue()
        assert "alice-chen/employer" in output
        assert "VP Engineering" in output
        assert "CTO" in output

    def test_prompt_verdict_affirm(self) -> None:
        """prompt_verdict returns 'Affirm' when operator enters 'Affirm'."""
        from mempill_showcase.adapters.oracle.cli_oracle import CliOracle
        import io
        oracle = CliOracle()
        buf = io.StringIO()
        fake_entry = {
            "handle_id": "abc-123",
            "subject": "alice",
            "predicate": "employer",
            "incumbent_value": "VP",
            "challenger_value": "CTO",
        }
        verdict = oracle.prompt_verdict(fake_entry, file=buf, input_fn=lambda: "Affirm")
        assert verdict == "Affirm"

    def test_prompt_verdict_deny(self) -> None:
        """prompt_verdict returns 'Deny' when operator enters 'deny' (case-insensitive)."""
        from mempill_showcase.adapters.oracle.cli_oracle import CliOracle
        import io
        oracle = CliOracle()
        buf = io.StringIO()
        fake_entry = {"handle_id": "x", "subject": "a", "predicate": "b",
                      "incumbent_value": "v1", "challenger_value": "v2"}
        verdict = oracle.prompt_verdict(fake_entry, file=buf, input_fn=lambda: "deny")
        assert verdict == "Deny"

    def test_prompt_verdict_retries_on_invalid(self) -> None:
        """prompt_verdict retries until a valid verdict is entered."""
        from mempill_showcase.adapters.oracle.cli_oracle import CliOracle
        import io
        import itertools
        oracle = CliOracle()
        buf = io.StringIO()
        inputs = iter(["invalid", "Yep", "Affirm"])
        fake_entry = {"handle_id": "x", "subject": "a", "predicate": "b",
                      "incumbent_value": "v1", "challenger_value": "v2"}
        verdict = oracle.prompt_verdict(fake_entry, file=buf, input_fn=lambda: next(inputs))
        assert verdict == "Affirm"

    def test_cli_oracle_usable_with_open_oracle_in_memory(self) -> None:
        """CliOracle can be passed to mempill.open_oracle_in_memory without error."""
        import mempill
        from mempill_showcase.adapters.oracle.cli_oracle import CliOracle
        oracle = CliOracle()
        engine = mempill.open_oracle_in_memory(oracle)
        assert engine is not None
        # Verify the engine is functional
        pending = engine.list_pending_adjudications(agent_id="test")
        assert isinstance(pending, list)
