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
    _normalize_messages,
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


class TestNormalizeMessages:
    """_normalize_messages: router input hardening (TASK-33).

    Regression: LangGraph Studio's raw-array input form for {messages}-only
    schemas can coerce a scalar into a primitive (e.g. int) list entry. A
    real Studio session crashed with NotImplementedError inside LangChain
    message coercion, deep inside a subgraph, after route_query had already
    committed to a route — normalizing at route_query's entry point means
    subgraphs never see junk.
    """

    def test_int_entry_dropped_not_raised(self):
        normalized, dropped = _normalize_messages([2025])
        assert normalized == []
        assert dropped == [2025]

    def test_empty_list_normalizes_to_empty(self):
        normalized, dropped = _normalize_messages([])
        assert normalized == []
        assert dropped == []

    def test_empty_string_entry_dropped(self):
        normalized, dropped = _normalize_messages([""])
        assert normalized == []
        assert dropped == [""]

    def test_contentless_dict_dropped(self):
        normalized, dropped = _normalize_messages([{"no": "content"}])
        assert normalized == []
        assert dropped == [{"no": "content"}]

    def test_mixed_valid_and_junk_survives_valid_parts(self):
        valid_msg = _FakeHumanMessage("hello")
        normalized, dropped = _normalize_messages(
            ["a real question", 2025, None, True, {"no": "content"}, valid_msg]
        )
        assert len(normalized) == 2
        assert normalized[0].content == "a real question"
        assert normalized[1] is valid_msg
        assert dropped == [2025, None, True, {"no": "content"}]

    def test_valid_dict_message_kept_as_is(self):
        d = {"role": "user", "content": "hi there"}
        normalized, dropped = _normalize_messages([d])
        assert normalized == [d]
        assert dropped == []

    def test_non_string_primitives_all_dropped(self):
        normalized, dropped = _normalize_messages([2025, 1.5, None, False, True])
        assert normalized == []
        assert dropped == [2025, 1.5, None, False, True]

    def test_happy_path_string_coerced_to_human_message(self):
        from langchain_core.messages import HumanMessage

        normalized, dropped = _normalize_messages(["What is Alice's role?"])
        assert dropped == []
        assert len(normalized) == 1
        assert isinstance(normalized[0], HumanMessage)
        assert normalized[0].content == "What is Alice's role?"


class TestRouteQueryNodeMessageHardening:
    """route_query's entry-point normalization: the THE regression + friendly
    ambiguous-default fallback when nothing usable survives."""

    def test_int_message_does_not_raise_and_does_not_reach_subgraph(self, monkeypatch):
        """{"messages": [2025]} — must not raise, and the returned state's
        `messages` must NOT contain the raw int (so a downstream subgraph never
        sees it)."""
        import mempill_showcase.frameworks.langgraph.router_graph as rg_mod

        fake = _FakeStructuredClassifier(raises=AssertionError("classifier should not be reached"))
        monkeypatch.setattr(rg_mod, "_build_classifier", lambda model_name=None: fake)

        route_query = rg_mod.make_route_query_node()
        state: RouterState = {"messages": [2025]}
        result = route_query(state)  # must not raise

        assert result["route"] == "people_ops"
        assert result["agent_id"] == PEOPLE_OPS_SPEC.agent_id
        assert 2025 not in result["messages"]
        assert all(not isinstance(m, int) for m in result["messages"])

    def test_empty_messages_list_defaults_ambiguous_with_friendly_message(self, monkeypatch):
        import mempill_showcase.frameworks.langgraph.router_graph as rg_mod

        fake = _FakeStructuredClassifier(raises=AssertionError("classifier should not be reached"))
        monkeypatch.setattr(rg_mod, "_build_classifier", lambda model_name=None: fake)

        route_query = rg_mod.make_route_query_node()
        result = route_query({"messages": []})

        assert result["route"] == "people_ops"
        assert result["agent_id"] == PEOPLE_OPS_SPEC.agent_id
        assert len(result["messages"]) == 1
        assert result["messages"][0].content  # friendly, non-empty text
        assert fake.last_prompt is None  # classifier never invoked

    def test_blank_string_message_defaults_ambiguous(self, monkeypatch):
        import mempill_showcase.frameworks.langgraph.router_graph as rg_mod

        fake = _FakeStructuredClassifier(raises=AssertionError("classifier should not be reached"))
        monkeypatch.setattr(rg_mod, "_build_classifier", lambda model_name=None: fake)

        route_query = rg_mod.make_route_query_node()
        result = route_query({"messages": [""]})

        assert result["route"] == "people_ops"
        assert fake.last_prompt is None

    def test_contentless_dict_message_defaults_ambiguous(self, monkeypatch):
        import mempill_showcase.frameworks.langgraph.router_graph as rg_mod

        fake = _FakeStructuredClassifier(raises=AssertionError("classifier should not be reached"))
        monkeypatch.setattr(rg_mod, "_build_classifier", lambda model_name=None: fake)

        route_query = rg_mod.make_route_query_node()
        result = route_query({"messages": [{"no": "content"}]})

        assert result["route"] == "people_ops"
        assert fake.last_prompt is None

    def test_mixed_valid_and_junk_routes_using_only_valid_parts(self, monkeypatch):
        """Junk entries are dropped; the valid entry still drives routing."""
        import mempill_showcase.frameworks.langgraph.router_graph as rg_mod

        decision = RouteDecision(agent="org_registry", rationale="mentions Acme's CEO")
        fake = _FakeStructuredClassifier(decision=decision)
        monkeypatch.setattr(rg_mod, "_build_classifier", lambda model_name=None: fake)

        route_query = rg_mod.make_route_query_node()
        result = route_query({"messages": [2025, None, "Who is Acme's CEO?"]})

        assert result["route"] == "org_registry"
        assert result["agent_id"] == ORG_REGISTRY_SPEC.agent_id
        assert "Who is Acme's CEO?" in fake.last_prompt
        assert len(result["messages"]) == 1
        assert result["messages"][0].content == "Who is Acme's CEO?"

    def test_happy_path_single_valid_message_unchanged_behavior(self, monkeypatch):
        """Existing happy-path behavior (single valid HumanMessage) is unaffected."""
        import mempill_showcase.frameworks.langgraph.router_graph as rg_mod

        decision = RouteDecision(agent="people_ops", rationale="mentions Bob")
        fake = _FakeStructuredClassifier(decision=decision)
        monkeypatch.setattr(rg_mod, "_build_classifier", lambda model_name=None: fake)

        route_query = rg_mod.make_route_query_node()
        result = route_query({"messages": [_FakeHumanMessage("What is Bob's travel preference?")]})

        assert result["route"] == "people_ops"
        assert result["agent_id"] == "people-ops-001"
        assert result["route_rationale"] == "mentions Bob"
        assert len(result["messages"]) == 1
        assert result["messages"][0].content == "What is Bob's travel preference?"

    def test_full_router_graph_survives_int_input_end_to_end(self, monkeypatch):
        """Full build_router_graph().invoke({"messages": [2025]}) — the literal
        Studio regression scenario — must not raise, using a dummy subgraph."""
        import mempill_showcase.frameworks.langgraph.router_graph as rg_mod

        fake = _FakeStructuredClassifier(raises=AssertionError("classifier should not be reached"))
        monkeypatch.setattr(rg_mod, "_build_classifier", lambda model_name=None: fake)

        sub1 = TestBuildRouterGraph._dummy_subgraph()
        sub2 = TestBuildRouterGraph._dummy_subgraph()
        compiled = build_router_graph(sub1, sub2)

        result = compiled.invoke({"messages": [2025]})  # must not raise

        assert result["route"] == "people_ops"
        assert all(not isinstance(m, int) for m in result["messages"])


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

    def test_compiled_graph_input_schema_exposes_only_messages(self):
        """TASK-31-W5: Studio's input form must show only Messages — not the
        full internal RouterState (agent_id/route/route_rationale). A prior
        Studio input error occurred because the raw-array input form coerced
        a comma-separated value into an int, crashing message coercion."""
        sub1 = self._dummy_subgraph()
        sub2 = self._dummy_subgraph()
        compiled = build_router_graph(sub1, sub2)

        schema = compiled.get_input_jsonschema()
        assert list(schema.get("properties", {}).keys()) == ["messages"], (
            f"router graph input schema must expose ONLY 'messages', got {schema}"
        )


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
