"""
mempill_showcase.tests.test_router_graph_unit — TASK-31 T31-1 deterministic
router unit tests (NO API key required).

These tests exercise the router's OWN code — RouteDecision parsing, the
default-route fallback, and agent_id injection into state — by monkeypatching
the structured-output classifier call at the boundary (`_build_classifier`).
This is testing the router node's control flow, not introducing a mock
framework for LLM behavior; live classification accuracy tests belong to
Wave 3 (test_router_graph.py, `live` marker).

No resume-guard is tested here (SPIKE_RESULTS.md confirmed it is unnecessary
and intentionally omitted from router_graph.py).
"""
from __future__ import annotations

import pytest

from mempill_showcase.frameworks.langgraph.agents_config import (
    ORG_REGISTRY_SPEC,
    PEOPLE_OPS_SPEC,
)
from mempill_showcase.frameworks.langgraph.router_graph import (
    RouteDecision,
    _select_route,
    build_router_graph,
    make_route_query_node,
)
from mempill_showcase.frameworks.langgraph.router_state import RouterState


class _FakeHumanMessage:
    """Minimal stand-in for a langchain HumanMessage (has .content)."""
    def __init__(self, content: str):
        self.content = content


class _FakeStructuredClassifier:
    """Stand-in for `ChatAnthropic(...).with_structured_output(RouteDecision)`.

    invoke() returns a pre-configured RouteDecision, or raises a pre-configured
    exception, regardless of the prompt passed in.
    """
    def __init__(self, decision: RouteDecision | None = None, raises: Exception | None = None):
        self._decision = decision
        self._raises = raises
        self.last_prompt = None

    def invoke(self, prompt):
        self.last_prompt = prompt
        if self._raises is not None:
            raise self._raises
        return self._decision


class TestRouteDecisionParsing:
    """RouteDecision Pydantic model: valid construction + literal enum enforcement."""

    def test_valid_people_ops_decision(self):
        d = RouteDecision(agent="people_ops", rationale="mentions Alice")
        assert d.agent == "people_ops"
        assert d.rationale == "mentions Alice"

    def test_valid_org_registry_decision(self):
        d = RouteDecision(agent="org_registry", rationale="mentions Acme's CEO")
        assert d.agent == "org_registry"

    def test_invalid_agent_value_rejected(self):
        with pytest.raises(Exception):
            RouteDecision(agent="not_a_real_route", rationale="bad")


class TestRouteQueryNodeDefaultRoute:
    """route_query defaults to people_ops on classifier failure (ambiguity policy)."""

    def test_classifier_exception_defaults_to_people_ops(self, monkeypatch):
        # Monkeypatch _build_classifier so make_route_query_node uses our fake.
        import mempill_showcase.frameworks.langgraph.router_graph as rg_mod

        fake = _FakeStructuredClassifier(raises=RuntimeError("no API key"))
        monkeypatch.setattr(rg_mod, "_build_classifier", lambda model_name=None: fake)

        route_query = rg_mod.make_route_query_node()
        state: RouterState = {"messages": [_FakeHumanMessage("something ambiguous")]}
        result = route_query(state)

        assert result["route"] == "people_ops"
        assert result["agent_id"] == PEOPLE_OPS_SPEC.agent_id
        assert "classifier error" in result["route_rationale"]

    def test_classifier_success_people_ops_injects_correct_agent_id(self, monkeypatch):
        import mempill_showcase.frameworks.langgraph.router_graph as rg_mod

        decision = RouteDecision(agent="people_ops", rationale="mentions Bob")
        fake = _FakeStructuredClassifier(decision=decision)
        monkeypatch.setattr(rg_mod, "_build_classifier", lambda model_name=None: fake)

        route_query = rg_mod.make_route_query_node()
        state: RouterState = {"messages": [_FakeHumanMessage("What is Bob's travel preference?")]}
        result = route_query(state)

        assert result["route"] == "people_ops"
        assert result["agent_id"] == "people-ops-001"
        assert result["route_rationale"] == "mentions Bob"

    def test_classifier_success_org_registry_injects_correct_agent_id(self, monkeypatch):
        import mempill_showcase.frameworks.langgraph.router_graph as rg_mod

        decision = RouteDecision(agent="org_registry", rationale="mentions Acme's CEO")
        fake = _FakeStructuredClassifier(decision=decision)
        monkeypatch.setattr(rg_mod, "_build_classifier", lambda model_name=None: fake)

        route_query = rg_mod.make_route_query_node()
        state: RouterState = {"messages": [_FakeHumanMessage("Who is Acme's CEO?")]}
        result = route_query(state)

        assert result["route"] == "org_registry"
        assert result["agent_id"] == ORG_REGISTRY_SPEC.agent_id
        assert result["agent_id"] == "org-registry-001"


class TestAgentIdInjection:
    """agent_id is injected into state BEFORE dispatch (state-injection, C2)."""

    def test_agent_id_matches_route_for_both_routes(self, monkeypatch):
        import mempill_showcase.frameworks.langgraph.router_graph as rg_mod

        for route, expected_agent_id in [
            ("people_ops", "people-ops-001"),
            ("org_registry", "org-registry-001"),
        ]:
            decision = RouteDecision(agent=route, rationale="test")
            fake = _FakeStructuredClassifier(decision=decision)
            monkeypatch.setattr(rg_mod, "_build_classifier", lambda model_name=None, _fake=fake: _fake)

            route_query = rg_mod.make_route_query_node()
            state: RouterState = {"messages": [_FakeHumanMessage("irrelevant text")]}
            result = route_query(state)

            assert result["agent_id"] == expected_agent_id, (
                f"route={route!r} must inject agent_id={expected_agent_id!r}, "
                f"got {result['agent_id']!r}"
            )


class TestSelectRouteConditionalEdge:
    """_select_route reads state['route'] and defaults to people_ops if unset."""

    def test_select_route_people_ops(self):
        assert _select_route({"route": "people_ops"}) == "people_ops"

    def test_select_route_org_registry(self):
        assert _select_route({"route": "org_registry"}) == "org_registry"

    def test_select_route_missing_defaults_to_people_ops(self):
        assert _select_route({}) == "people_ops"


class TestAgentSpecConstants:
    """agents_config.py AgentSpec literals match ARCHITECTURE.md §1 exactly."""

    def test_people_ops_agent_id(self):
        assert PEOPLE_OPS_SPEC.agent_id == "people-ops-001"

    def test_org_registry_agent_id(self):
        assert ORG_REGISTRY_SPEC.agent_id == "org-registry-001"

    def test_people_ops_known_entities(self):
        for name in ("alice-chen", "bob-liu", "jordan-park"):
            assert name in PEOPLE_OPS_SPEC.known_entities

    def test_org_registry_known_entities(self):
        assert "acme-corp" in ORG_REGISTRY_SPEC.known_entities


class TestBuildRouterGraph:
    """build_router_graph(sub1, sub2) compiles a CompiledGraph (dummy subgraphs,
    no API key required — the classifier model client is only constructed, never
    invoked, during graph compilation).
    """

    @staticmethod
    def _dummy_subgraph():
        from langgraph.graph import StateGraph, END

        def _echo(state):
            return {"messages": state.get("messages", [])}

        g = StateGraph(RouterState)
        g.add_node("n", _echo)
        g.set_entry_point("n")
        g.add_edge("n", END)
        return g.compile()

    def test_build_router_graph_returns_compiled_graph(self):
        sub1 = self._dummy_subgraph()
        sub2 = self._dummy_subgraph()
        compiled = build_router_graph(sub1, sub2)
        assert compiled is not None

    def test_build_router_graph_has_no_memory_saver_by_default(self):
        """checkpointer defaults to None (Studio/router path)."""
        from langgraph.checkpoint.memory import MemorySaver

        sub1 = self._dummy_subgraph()
        sub2 = self._dummy_subgraph()
        compiled = build_router_graph(sub1, sub2)
        checkpointer = getattr(compiled, "checkpointer", None)
        assert not isinstance(checkpointer, MemorySaver)


class TestBuildAgentInstance:
    """di.build_agent_instance() composes adapter + tools + prompt + graph."""

    def test_people_ops_instance_isolated_db(self):
        from mempill_showcase.config.di import build_agent_instance

        adapter, subgraph = build_agent_instance(
            agent_id=PEOPLE_OPS_SPEC.agent_id,
            responsibility=PEOPLE_OPS_SPEC.responsibility,
            known_entities=PEOPLE_OPS_SPEC.known_entities,
            db_dir=None,
        )
        assert adapter is not None
        assert subgraph is not None

    def test_org_registry_instance_isolated_db(self):
        from mempill_showcase.config.di import build_agent_instance

        adapter, subgraph = build_agent_instance(
            agent_id=ORG_REGISTRY_SPEC.agent_id,
            responsibility=ORG_REGISTRY_SPEC.responsibility,
            known_entities=ORG_REGISTRY_SPEC.known_entities,
            db_dir=None,
        )
        assert adapter is not None
        assert subgraph is not None

    def test_instance_subgraph_has_no_memory_saver(self):
        """build_agent_instance compiles with checkpointer=None (router/Studio path)."""
        from langgraph.checkpoint.memory import MemorySaver
        from mempill_showcase.config.di import build_agent_instance

        _, subgraph = build_agent_instance(
            agent_id=PEOPLE_OPS_SPEC.agent_id,
            responsibility=PEOPLE_OPS_SPEC.responsibility,
            known_entities=PEOPLE_OPS_SPEC.known_entities,
            db_dir=None,
        )
        checkpointer = getattr(subgraph, "checkpointer", None)
        assert not isinstance(checkpointer, MemorySaver)
