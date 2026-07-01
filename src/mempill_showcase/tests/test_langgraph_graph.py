"""
mempill_showcase.tests.test_langgraph_graph — Wave B ReAct agent integration tests.

No live LLM required for the structural tests.  The HITL interrupt/resume path
is verified deterministically by calling the tool directly (bypassing the LLM),
then testing the interrupt/Command(resume=...) mechanics against the compiled graph.

Test classes:
  TestBuildGraph          — build_graph() produces a compiled app from a ShowcaseTools.
  TestBuildApp            — build_app() factory returns (app, adapter) + app is runnable.
  TestHITLInterrupt       — RequestAdjudicationTool calls interrupt(); Command(resume)
                            resumes and resolves the contested belief.
  TestStudioGraph         — studio_graph module imports + exposes a compiled graph
                            without MemorySaver; seeded adapter has Day-0 data.
"""
from __future__ import annotations

import json
import uuid

import pytest

from langgraph.types import Command

from mempill_showcase.config.di import build_mempill_adapter, build_agent_tools, build_app
from mempill_showcase.frameworks.langgraph.graph import build_graph, ShowcaseTools
from mempill_showcase.scenarios.seed_data import AGENT_ID


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def adapter():
    """Fresh oracle-backed in-memory mempill adapter per test."""
    return build_mempill_adapter(in_memory=True, oracle_backed=True)


@pytest.fixture()
def tools(adapter):
    return build_agent_tools(adapter)


@pytest.fixture()
def app(adapter, tools):
    """Compiled ReAct agent app (MemorySaver checkpointer)."""
    return build_graph(adapter=adapter, tools=tools)


def _cfg(thread_id: str | None = None) -> dict:
    return {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}


# ── TestBuildGraph ────────────────────────────────────────────────────────────

class TestBuildGraph:
    """build_graph() produces a compiled app from a ShowcaseTools."""

    def test_build_graph_returns_non_none(self, adapter, tools):
        """build_graph() returns a non-None compiled app."""
        compiled = build_graph(adapter=adapter, tools=tools)
        assert compiled is not None, "build_graph must return a compiled app"

    def test_tools_is_showcase_tools(self, tools):
        """build_agent_tools returns a ShowcaseTools NamedTuple with 10 tools."""
        assert isinstance(tools, ShowcaseTools)
        assert len(tools) == 10, f"ShowcaseTools must have 10 tools, got {len(tools)}"

    def test_all_tool_names_are_unique(self, tools):
        """All 10 tool names are distinct."""
        names = [t.name for t in tools]
        assert len(set(names)) == len(names), f"Duplicate tool names: {names}"

    def test_build_graph_with_no_checkpointer(self, adapter, tools):
        """build_graph(checkpointer=None) compiles without a MemorySaver (Studio path)."""
        compiled = build_graph(adapter=adapter, tools=tools, checkpointer=None)
        assert compiled is not None
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = getattr(compiled, "checkpointer", None)
        assert not isinstance(checkpointer, MemorySaver), (
            "Graph with checkpointer=None must NOT carry a MemorySaver"
        )


# ── TestBuildApp ──────────────────────────────────────────────────────────────

class TestBuildApp:
    """build_app() factory returns (app, adapter)."""

    def test_build_app_returns_tuple(self):
        """build_app() returns (app, adapter) where both are non-None."""
        app, adapter = build_app()
        assert app is not None, "app must not be None"
        assert adapter is not None, "adapter must not be None"

    def test_build_app_reuses_adapter(self):
        """build_app(adapter=<existing>) reuses the provided adapter."""
        existing = build_mempill_adapter(in_memory=True)
        app, returned = build_app(adapter=existing)
        assert returned is existing, "build_app must reuse the passed adapter"

    def test_build_app_has_memory_saver(self):
        """build_app() default path attaches a MemorySaver checkpointer."""
        from langgraph.checkpoint.memory import MemorySaver
        app, _ = build_app()
        checkpointer = getattr(app, "checkpointer", None)
        assert isinstance(checkpointer, MemorySaver), (
            "Default build_app() must carry a MemorySaver checkpointer"
        )


# ── TestHITLInterrupt ─────────────────────────────────────────────────────────

class TestHITLInterrupt:
    """RequestAdjudicationTool triggers interrupt(); Command(resume) resolves it.

    We test the interrupt/resume mechanics by invoking the tool DIRECTLY inside
    a compiled graph (bypassing the LLM router) via a pre-seeded conflict + a
    minimal state machine test.

    AC-2 assertions:
      (a) Calling request_adjudication_tool.invoke() raises a LangGraph interrupt.
      (b) The interrupt payload describes the conflict.
      (c) _normalize_verdict correctly maps 'Affirm' → 'Affirm'.
      (d) After oracle Affirm on a seeded conflict, post-resolution recall returns CTO.
    """

    def _seed_vp(self, adapter):
        """Seed alice-chen/employer = VP Engineering (2023-06)."""
        from mempill_showcase.core.domain.models import ClaimInput
        import mempill

        adapter.write_claim(AGENT_ID, ClaimInput(
            subject="alice-chen",
            predicate="employer",
            value="Acme Corp / VP Engineering",
            valid_from="2023-06",
            confidence=1.0,
            provenance=mempill.ProvenanceLabel.external_user_asserted(),
        ))

    def test_normalize_verdict_affirm(self):
        """_normalize_verdict maps 'affirm' (case-insensitive) → 'Affirm'."""
        from mempill_showcase.frameworks.langgraph.hitl_node import _normalize_verdict

        assert _normalize_verdict("Affirm", challenger_value="CTO", incumbent_value="VP") == "Affirm"
        assert _normalize_verdict("affirm", challenger_value="CTO", incumbent_value="VP") == "Affirm"
        assert _normalize_verdict("yes", challenger_value="CTO", incumbent_value="VP") == "Affirm"
        assert _normalize_verdict("CTO", challenger_value="CTO", incumbent_value="VP") == "Affirm"

    def test_normalize_verdict_deny(self):
        """_normalize_verdict maps 'deny' → 'Deny'."""
        from mempill_showcase.frameworks.langgraph.hitl_node import _normalize_verdict

        assert _normalize_verdict("Deny", challenger_value="CTO", incumbent_value="VP") == "Deny"
        assert _normalize_verdict("no", challenger_value="CTO", incumbent_value="VP") == "Deny"
        assert _normalize_verdict("VP", challenger_value="CTO", incumbent_value="VP") == "Deny"

    def test_normalize_verdict_abstain(self):
        """_normalize_verdict maps 'abstain' → 'Abstain'."""
        from mempill_showcase.frameworks.langgraph.hitl_node import _normalize_verdict

        assert _normalize_verdict("Abstain", challenger_value=None, incumbent_value=None) == "Abstain"
        assert _normalize_verdict("defer", challenger_value=None, incumbent_value=None) == "Abstain"

    def test_normalize_verdict_invalid(self):
        """_normalize_verdict returns '_invalid_' for unrecognised input."""
        from mempill_showcase.frameworks.langgraph.hitl_node import _normalize_verdict

        result = _normalize_verdict("not_a_verdict_xyz", challenger_value=None, incumbent_value=None)
        assert result == "_invalid_"

    def test_request_adjudication_tool_raises_interrupt(self, adapter, tools):
        """Calling request_adjudication_tool.invoke() inside a graph pauses at interrupt.

        We build a minimal graph with the request_adjudication tool as a tool call
        and verify the graph yields __interrupt__ on the first invoke.
        """
        from mempill_showcase.core.domain.models import ClaimInput
        import mempill

        # Seed a conflict so the oracle queue has a pending entry
        adapter.write_claim(AGENT_ID, ClaimInput(
            subject="alice-chen",
            predicate="employer",
            value="Acme Corp / VP Engineering",
            valid_from="2023-06",
            confidence=1.0,
            provenance=mempill.ProvenanceLabel.external_user_asserted(),
        ))
        adapter.write_claim(AGENT_ID, ClaimInput(
            subject="alice-chen",
            predicate="employer",
            value="Acme Corp / CTO",
            valid_from="2023-06",
            confidence=1.0,
            provenance=mempill.ProvenanceLabel.external_user_asserted(),
        ))

        # Build a tiny graph that calls request_adjudication deterministically
        from langgraph.graph import StateGraph, END
        from langgraph.checkpoint.memory import MemorySaver
        from mempill_showcase.frameworks.langgraph.state import ExecAssistantState

        adj_tool = tools.request_adjudication_tool

        def _call_adjudication(state):
            result = adj_tool.invoke({
                "agent_id": AGENT_ID,
                "subject": "alice-chen",
                "predicate": "employer",
                "reason": "Conflict between VP Engineering and CTO",
                "incumbent_value": "Acme Corp / VP Engineering",
                "challenger_value": "Acme Corp / CTO",
                "claim_refs": [],
            })
            return {"output_text": result}

        g = StateGraph(ExecAssistantState)
        g.add_node("adjudicate", _call_adjudication)
        g.set_entry_point("adjudicate")
        g.add_edge("adjudicate", END)

        compiled = g.compile(checkpointer=MemorySaver())
        thread_id = str(uuid.uuid4())
        cfg = {"configurable": {"thread_id": thread_id}}

        result = compiled.invoke({"user_input": "test", "agent_id": AGENT_ID}, cfg)

        assert "__interrupt__" in result, (
            f"Graph must pause at interrupt(), got keys: {list(result.keys())}"
        )
        interrupts = result["__interrupt__"]
        assert len(interrupts) >= 1
        payload = interrupts[0].value
        assert "question" in payload or "subject" in payload, (
            f"Interrupt payload must describe the conflict, got: {payload}"
        )

    def test_hitl_affirm_resolves_belief(self, adapter, tools):
        """After oracle Affirm on a seeded same-period conflict, recall returns CTO."""
        from mempill_showcase.core.domain.models import ClaimInput
        import mempill

        # Seed conflict
        adapter.write_claim(AGENT_ID, ClaimInput(
            subject="alice-chen",
            predicate="employer",
            value="Acme Corp / VP Engineering",
            valid_from="2023-06",
            confidence=1.0,
            provenance=mempill.ProvenanceLabel.external_user_asserted(),
        ))
        adapter.write_claim(AGENT_ID, ClaimInput(
            subject="alice-chen",
            predicate="employer",
            value="Acme Corp / CTO",
            valid_from="2023-06",
            confidence=1.0,
            provenance=mempill.ProvenanceLabel.external_user_asserted(),
        ))

        # Submit Affirm via oracle
        pending = adapter.list_pending_adjudications(AGENT_ID)
        assert pending, "Oracle queue must have pending entry"
        result = adapter.submit_adjudication(AGENT_ID, pending[0]["handle_id"], "Affirm")
        assert result.get("disposition") == "CommittedCheap"

        # Post-resolution recall must return CTO
        belief = adapter.recall(AGENT_ID, "alice-chen", "employer")
        assert belief.status == "Resolved", f"After Affirm, recall must be Resolved, got {belief.status!r}"
        assert "CTO" in (belief.value or ""), f"After Affirm, CTO must win, got {belief.value!r}"

    def test_hitl_interrupt_and_resume_via_graph(self, adapter, tools):
        """Full interrupt → Command(resume='Affirm') → resolved cycle via compiled graph."""
        from mempill_showcase.core.domain.models import ClaimInput
        import mempill
        from langgraph.graph import StateGraph, END
        from langgraph.checkpoint.memory import MemorySaver
        from mempill_showcase.frameworks.langgraph.state import ExecAssistantState

        # Seed conflict
        adapter.write_claim(AGENT_ID, ClaimInput(
            subject="alice-chen",
            predicate="employer",
            value="Acme Corp / VP Engineering",
            valid_from="2023-06",
            confidence=1.0,
            provenance=mempill.ProvenanceLabel.external_user_asserted(),
        ))
        adapter.write_claim(AGENT_ID, ClaimInput(
            subject="alice-chen",
            predicate="employer",
            value="Acme Corp / CTO",
            valid_from="2023-06",
            confidence=1.0,
            provenance=mempill.ProvenanceLabel.external_user_asserted(),
        ))

        adj_tool = tools.request_adjudication_tool

        def _call_adjudication(state):
            raw = adj_tool.invoke({
                "agent_id": AGENT_ID,
                "subject": "alice-chen",
                "predicate": "employer",
                "reason": "VP vs CTO conflict",
                "incumbent_value": "Acme Corp / VP Engineering",
                "challenger_value": "Acme Corp / CTO",
                "claim_refs": [],
            })
            parsed = json.loads(raw)
            return {
                "output_text": raw,
                "hitl_verdict": parsed.get("verdict"),
                "hitl_resolved_belief": json.dumps({"value": parsed.get("winning_value")}),
            }

        g = StateGraph(ExecAssistantState)
        g.add_node("adjudicate", _call_adjudication)
        g.set_entry_point("adjudicate")
        g.add_edge("adjudicate", END)

        compiled = g.compile(checkpointer=MemorySaver())
        thread_id = str(uuid.uuid4())
        cfg = {"configurable": {"thread_id": thread_id}}

        # First invoke → interrupt
        result1 = compiled.invoke({"user_input": "test", "agent_id": AGENT_ID}, cfg)
        assert "__interrupt__" in result1, "Graph must pause at interrupt"

        # Resume with Affirm
        result2 = compiled.invoke(Command(resume="Affirm"), cfg)

        # Graph completed; check verdict
        assert result2.get("hitl_verdict") == "Affirm", (
            f"hitl_verdict should be 'Affirm', got {result2.get('hitl_verdict')!r}"
        )
        hrb = result2.get("hitl_resolved_belief")
        assert hrb is not None, "hitl_resolved_belief must be set after Affirm"
        rb = json.loads(hrb)
        assert "CTO" in (rb.get("value") or ""), (
            f"Resolved belief must contain CTO, got {rb.get('value')!r}"
        )


# ── TestStudioGraph ───────────────────────────────────────────────────────────

class TestStudioGraph:
    """Guard tests: studio_graph module exposes a compiled graph without MemorySaver."""

    def test_studio_graph_imports(self):
        """studio_graph.py is importable and exposes a module-level `graph`."""
        from mempill_showcase.frameworks.langgraph.studio_graph import graph
        assert graph is not None, "studio_graph.graph must be non-None"

    def test_studio_graph_has_no_memory_saver(self):
        """studio_graph.graph has no MemorySaver (Studio injects its own checkpointer)."""
        from langgraph.checkpoint.memory import MemorySaver
        from mempill_showcase.frameworks.langgraph.studio_graph import graph
        checkpointer = getattr(graph, "checkpointer", None)
        assert not isinstance(checkpointer, MemorySaver), (
            "studio_graph.graph must NOT have a MemorySaver checkpointer. "
            f"Got checkpointer={type(checkpointer).__name__!r}"
        )

    def test_studio_adapter_seeded_with_day0_data(self):
        """Importing studio_graph seeds the adapter: alice-chen/city belief exists."""
        from mempill_showcase.frameworks.langgraph.studio_graph import studio_adapter
        from mempill_showcase.scenarios.seed_data import AGENT_ID

        belief = studio_adapter.recall(AGENT_ID, "alice-chen", "city")
        assert belief is not None
        assert belief.status != "NoBelief", (
            f"Day-0 seed city should be present (not NoBelief), got {belief.status!r}"
        )

    def test_studio_adapter_seeded_dietary(self):
        """Seeded adapter has alice-chen/dietary_restriction=vegetarian."""
        from mempill_showcase.frameworks.langgraph.studio_graph import studio_adapter
        from mempill_showcase.scenarios.seed_data import AGENT_ID

        belief = studio_adapter.recall(AGENT_ID, "alice-chen", "dietary_restriction")
        assert belief is not None
        assert belief.value == "vegetarian", (
            f"Expected 'vegetarian', got {belief.value!r}"
        )
