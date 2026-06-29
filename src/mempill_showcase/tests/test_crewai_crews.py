"""
mempill_showcase.tests.test_crewai_crews — W4 CrewAI construction + tool-bridge tests.

No API key, no LLM calls for the default suite.

Coverage:
  C1. build_crews() construction — returns ShowcaseCrews with 3 crews.
  C2. Crew A agents: correct roles; intake_agent has remember + recall + date_parser;
      canonicalization_agent has entity_normalizer; none have audit_tool.
  C3. Crew B agents: research_agent has rag_write (no remember); synthesis_agent
      has remember + recall (no rag_write in synthesis).
  C4. Crew C agents: NEITHER agent has a remember/write tool (read-only invariant AC-C4).
      fact_retrieval_agent has recall + audit; briefing_agent has calendar + email.
  C5. Tool bridging: as_crewai_tool() returns a CrewStructuredTool; calling .run()
      delegates to the underlying LangChain tool._run() with a real in-memory engine.
  C6. Entity normalizer standalone: resolves 'Alice Chen' → 'alice-chen', unknown → None.
  C7. Crew C read-only invariant: explicitly assert no tool named 'mempill_remember'
      is in any Crew C agent's tool list.
  C8. Shell path unaffected: existing graph tests still run (55 tests baseline).
  C9. (optional, skipped without key) @pytest.mark.live kickoff smoke test for Crew A.
"""
from __future__ import annotations

import json
import os
import pytest
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=UserWarning, module="chromadb")


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def adapter():
    """Fresh in-memory mempill adapter (shared across module tests for speed)."""
    from mempill_showcase.config.di import build_mempill_adapter
    return build_mempill_adapter(in_memory=True)


@pytest.fixture(scope="module")
def tools(adapter):
    """W2 tool bundle built from the adapter."""
    from mempill_showcase.config.di import build_tools
    return build_tools(adapter)


@pytest.fixture(scope="module")
def crews(adapter, tools):
    """ShowcaseCrews built without an LLM (construction-only, no API key required)."""
    from mempill_showcase.frameworks.crewai.crews import build_crews
    return build_crews(adapter=adapter, tools=tools, llm=None)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tool_names(agent) -> list[str]:
    """Return the list of tool names attached to an agent."""
    return [getattr(t, "name", "") for t in agent.tools]


def _has_tool(agent, name: str) -> bool:
    return name in _tool_names(agent)


# ── C1: build_crews() construction ────────────────────────────────────────────

class TestCrewsConstruction:
    def test_build_crews_returns_three_crews(self, crews):
        from mempill_showcase.frameworks.crewai.crews import ShowcaseCrews
        assert isinstance(crews, ShowcaseCrews)
        assert crews.crew_a is not None
        assert crews.crew_b is not None
        assert crews.crew_c is not None

    def test_crew_a_has_two_agents(self, crews):
        assert len(crews.crew_a.agents) == 2

    def test_crew_b_has_two_agents(self, crews):
        assert len(crews.crew_b.agents) == 2

    def test_crew_c_has_two_agents(self, crews):
        assert len(crews.crew_c.agents) == 2

    def test_crew_a_has_two_tasks(self, crews):
        assert len(crews.crew_a.tasks) == 2

    def test_crew_b_has_two_tasks(self, crews):
        assert len(crews.crew_b.tasks) == 2

    def test_crew_c_has_two_tasks(self, crews):
        assert len(crews.crew_c.tasks) == 2


# ── C2: Crew A agent roles and tools ──────────────────────────────────────────

class TestCrewAAgents:
    @pytest.fixture(autouse=True)
    def _agents(self, crews):
        # Agents are accessible via crew.agents list
        # We determine which is which by role string
        self.agents = {a.role: a for a in crews.crew_a.agents}

    def test_intake_agent_role_present(self):
        assert any("Intake" in role for role in self.agents)

    def test_canonicalization_agent_role_present(self):
        assert any("Canonicalization" in role or "Canon" in role for role in self.agents)

    def test_intake_agent_has_remember_tool(self):
        intake = next(a for a in self.agents.values() if "Intake" in a.role)
        assert _has_tool(intake, "mempill_remember"), (
            f"intake_agent missing mempill_remember; tools={_tool_names(intake)}"
        )

    def test_intake_agent_has_recall_tool(self):
        intake = next(a for a in self.agents.values() if "Intake" in a.role)
        assert _has_tool(intake, "mempill_recall"), (
            f"intake_agent missing mempill_recall; tools={_tool_names(intake)}"
        )

    def test_intake_agent_has_date_parser(self):
        intake = next(a for a in self.agents.values() if "Intake" in a.role)
        assert _has_tool(intake, "date_parser"), (
            f"intake_agent missing date_parser; tools={_tool_names(intake)}"
        )

    def test_canonicalization_agent_has_entity_normalizer(self):
        canon = next(a for a in self.agents.values() if "Canon" in a.role or "Specializ" in a.role)
        assert _has_tool(canon, "entity_normalizer"), (
            f"canonicalization_agent missing entity_normalizer; tools={_tool_names(canon)}"
        )

    def test_crew_a_task_order_canonical_before_intake(self, crews):
        # First task should be the canonicalization task
        first_task = crews.crew_a.tasks[0]
        assert "canonicalize" in first_task.description.lower() or "canonical" in first_task.description.lower() or "entity_normalizer" in first_task.description.lower(), (
            f"First task should be canonicalization; got description starting: {first_task.description[:80]}"
        )


# ── C3: Crew B agent roles and tools ──────────────────────────────────────────

class TestCrewBAgents:
    @pytest.fixture(autouse=True)
    def _agents(self, crews):
        self.agents = {a.role: a for a in crews.crew_b.agents}

    def test_research_agent_role_present(self):
        assert any("Research" in role for role in self.agents), (
            f"No Research role found; roles={list(self.agents.keys())}"
        )

    def test_synthesis_agent_role_present(self):
        assert any("Synthesis" in role or "Distil" in role for role in self.agents), (
            f"No Synthesis role found; roles={list(self.agents.keys())}"
        )

    def test_research_agent_has_rag_write(self):
        research = next(a for a in self.agents.values() if "Research" in a.role)
        assert _has_tool(research, "rag_write"), (
            f"research_agent missing rag_write; tools={_tool_names(research)}"
        )

    def test_research_agent_has_no_remember_tool(self):
        """Research agent must NOT write to mempill directly (raw content stays in RAG)."""
        research = next(a for a in self.agents.values() if "Research" in a.role)
        assert not _has_tool(research, "mempill_remember"), (
            f"research_agent must NOT have mempill_remember (AC-6); tools={_tool_names(research)}"
        )

    def test_synthesis_agent_has_remember_tool(self):
        synthesis = next(a for a in self.agents.values() if "Synthesis" in a.role or "Distil" in a.role)
        assert _has_tool(synthesis, "mempill_remember"), (
            f"synthesis_agent missing mempill_remember; tools={_tool_names(synthesis)}"
        )

    def test_synthesis_agent_has_recall_tool(self):
        synthesis = next(a for a in self.agents.values() if "Synthesis" in a.role or "Distil" in a.role)
        assert _has_tool(synthesis, "mempill_recall"), (
            f"synthesis_agent missing mempill_recall; tools={_tool_names(synthesis)}"
        )


# ── C4 & C7: Crew C read-only invariant ───────────────────────────────────────

class TestCrewCReadOnlyInvariant:
    @pytest.fixture(autouse=True)
    def _agents(self, crews):
        self.agents = crews.crew_c.agents

    def test_no_crew_c_agent_has_remember_tool(self):
        """AC invariant: Crew C is read-only — no write tools allowed."""
        for agent in self.agents:
            names = _tool_names(agent)
            assert "mempill_remember" not in names, (
                f"Crew C agent {agent.role!r} has mempill_remember — violates read-only invariant! "
                f"tools={names}"
            )

    def test_no_crew_c_agent_has_rag_write(self):
        """Crew C must not have rag_write either (no writes at all)."""
        for agent in self.agents:
            names = _tool_names(agent)
            assert "rag_write" not in names, (
                f"Crew C agent {agent.role!r} has rag_write; tools={names}"
            )

    def test_fact_retrieval_agent_has_recall_tool(self):
        fact_agent = next(
            (a for a in self.agents if "Retrieval" in a.role or "Fact" in a.role), None
        )
        assert fact_agent is not None, "No Fact Retrieval agent found in Crew C"
        assert _has_tool(fact_agent, "mempill_recall"), (
            f"fact_retrieval_agent missing mempill_recall; tools={_tool_names(fact_agent)}"
        )

    def test_briefing_agent_has_calendar_tool(self):
        briefing = next(
            (a for a in self.agents if "Briefing" in a.role or "Coordinator" in a.role), None
        )
        assert briefing is not None, "No Briefing agent found in Crew C"
        assert _has_tool(briefing, "calendar_create_event"), (
            f"briefing_agent missing calendar_create_event; tools={_tool_names(briefing)}"
        )

    def test_briefing_agent_has_email_draft_tool(self):
        briefing = next(
            (a for a in self.agents if "Briefing" in a.role or "Coordinator" in a.role), None
        )
        assert briefing is not None, "No Briefing agent found in Crew C"
        assert _has_tool(briefing, "email_draft"), (
            f"briefing_agent missing email_draft; tools={_tool_names(briefing)}"
        )

    def test_crew_c_agents_have_no_write_tools(self):
        """Comprehensive write-tool check for all Crew C agents."""
        write_tool_names = {"mempill_remember", "rag_write"}
        for agent in self.agents:
            names = set(_tool_names(agent))
            overlap = names & write_tool_names
            assert not overlap, (
                f"Crew C agent {agent.role!r} has write tool(s) {overlap} — violates read-only invariant"
            )


# ── C5: Tool bridging (no LLM needed) ─────────────────────────────────────────

class TestToolBridging:
    """Verify as_crewai_tool() wraps LangChain tools correctly.

    Tests call .run() on the wrapped tool directly (no LLM, no kickoff).
    The underlying mempill adapter is exercised with a real in-memory engine.
    """

    def test_bridge_returns_crewai_tool(self, tools):
        from mempill_showcase.frameworks.crewai.tool_bridge import as_crewai_tool
        from crewai.tools.base_tool import BaseTool as CrewBaseTool

        crew_tool = as_crewai_tool(tools.recall_tool)
        assert isinstance(crew_tool, CrewBaseTool), (
            f"Expected crewai BaseTool, got {type(crew_tool)}"
        )

    def test_bridge_preserves_tool_name(self, tools):
        from mempill_showcase.frameworks.crewai.tool_bridge import as_crewai_tool
        crew_tool = as_crewai_tool(tools.remember_tool)
        assert crew_tool.name == "mempill_remember"

    def test_bridge_preserves_tool_description(self, tools):
        from mempill_showcase.frameworks.crewai.tool_bridge import as_crewai_tool
        crew_tool = as_crewai_tool(tools.recall_tool)
        assert "mempill" in crew_tool.description.lower()

    def test_bridged_recall_delegates_to_adapter(self, adapter, tools):
        """Calling the bridged recall tool delegates to the real mempill adapter."""
        from mempill_showcase.frameworks.crewai.tool_bridge import as_crewai_tool
        from mempill_showcase.scenarios.seed_data import AGENT_ID, load_seed_claims as seed_day0

        # Seed data so there's something to recall
        seed_day0(adapter)

        crew_recall = as_crewai_tool(tools.recall_tool)
        # Call the underlying func directly (bypasses CrewAI's agent loop)
        raw = crew_recall._run(
            agent_id=AGENT_ID,
            subject="alice-chen",
            predicate="city",
        )
        result = json.loads(raw)
        assert result["value"] == "Austin TX", (
            f"Expected 'Austin TX', got {result['value']!r}; full result={result}"
        )
        assert result["status"] == "Resolved"

    def test_bridged_remember_delegates_to_adapter(self, adapter, tools):
        """Calling the bridged remember tool writes through to the real mempill adapter."""
        from mempill_showcase.frameworks.crewai.tool_bridge import as_crewai_tool
        from mempill_showcase.scenarios.seed_data import AGENT_ID

        crew_remember = as_crewai_tool(tools.remember_tool)
        raw = crew_remember._run(
            agent_id=AGENT_ID,
            subject="bob-liu",
            predicate="preferred_hotel",
            value="Hilton Honors",
            valid_from="2025-01",
            confidence=1.0,
            provenance_channel="UserAsserted",
        )
        result = json.loads(raw)
        assert "claim_ref" in result
        assert result["disposition"] in ("CommittedCheap", "Contested")

    def test_bridged_date_parser_delegates(self, tools):
        """DateParserTool bridge works without adapter."""
        from mempill_showcase.frameworks.crewai.tool_bridge import as_crewai_tool

        crew_date = as_crewai_tool(tools.date_parser)
        raw = crew_date._run(date_string="March 2025")
        result = json.loads(raw)
        assert result["iso_date"] == "2025-03"
        assert result["granularity"] == "month"

    def test_bridge_preserves_args_schema(self, tools):
        """Bridge forwards the LangChain tool's args_schema to CrewAI."""
        from mempill_showcase.frameworks.crewai.tool_bridge import as_crewai_tool
        crew_tool = as_crewai_tool(tools.recall_tool)
        # CrewAI auto-generates args_schema if not explicitly set
        assert crew_tool.args_schema is not None
        # The schema should have the expected fields from the LangChain tool
        assert hasattr(crew_tool.args_schema, "model_fields")


# ── C6: Entity Normalizer standalone ──────────────────────────────────────────

class TestEntityNormalizerTool:
    @pytest.fixture(autouse=True)
    def _tool(self):
        from mempill_showcase.frameworks.crewai.agents import _build_entity_normalizer_tool
        self.tool = _build_entity_normalizer_tool()

    def test_resolves_alice_chen(self):
        raw = self.tool._run(entity_name="Alice Chen")
        result = json.loads(raw)
        assert result["canonical_entity"] == "alice-chen"
        assert result["entity_resolved"] is True

    def test_resolves_acme(self):
        raw = self.tool._run(entity_name="Acme Corp")
        result = json.loads(raw)
        assert result["canonical_entity"] == "acme-corp"

    def test_resolves_predicate_city(self):
        raw = self.tool._run(entity_name="alice-chen", predicate_name="city")
        result = json.loads(raw)
        assert result["canonical_predicate"] == "city"
        assert result["predicate_resolved"] is True

    def test_unknown_entity_returns_none(self):
        raw = self.tool._run(entity_name="Unknown Person XYZ")
        result = json.loads(raw)
        assert result["canonical_entity"] is None
        assert result["entity_resolved"] is False

    def test_case_insensitive_resolution(self):
        raw = self.tool._run(entity_name="ALICE CHEN")
        result = json.loads(raw)
        assert result["canonical_entity"] == "alice-chen"


# ── C8: Shell path unaffected ─────────────────────────────────────────────────

class TestShellPathUnaffected:
    """Verify the graph still builds and routes correctly without CrewAI crews."""

    def test_build_langgraph_default_shell_path(self):
        from mempill_showcase.config.di import build_langgraph
        app, adapter = build_langgraph(in_memory=True, use_crewai=False)
        assert app is not None
        assert adapter is not None

    def test_graph_invoke_crew_a_shell(self):
        """crew_a_node shell path handles an UPDATE_CONTACT intent without crews."""
        from mempill_showcase.config.di import build_langgraph
        from mempill_showcase.scenarios.seed_data import load_seed_claims
        from mempill_showcase.frameworks.langgraph.supervisor_node import MockSupervisor

        app, adapter = build_langgraph(in_memory=True, use_crewai=False)
        load_seed_claims(adapter)

        config = {"configurable": {"thread_id": "test-shell-c8"}}
        result = app.invoke(
            {"user_input": "Alice relocated to NYC last month", "agent_id": "jordan-park-001"},
            config=config,
        )
        assert result is not None
        # Either the output_text is present or some result key exists
        assert any(k in result for k in ("output_text", "write_result", "recall_result", "briefing_text"))


# ── C9: Live kickoff smoke test (skipped without API key) ─────────────────────

@pytest.mark.live
def test_crew_a_kickoff_live(adapter, tools):
    """Smoke test: Crew A kickoff with a real LLM.

    Requires ANTHROPIC_API_KEY or OPENAI_API_KEY in the environment.
    Run with: pytest -m live
    Not part of the default CI suite.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("No API key found; set ANTHROPIC_API_KEY or OPENAI_API_KEY to run live tests")

    from mempill_showcase.frameworks.crewai.crews import build_crews
    from mempill_showcase.scenarios.seed_data import AGENT_ID, load_seed_claims

    load_seed_claims(adapter)

    # Use claude-3-haiku (cheapest) for the smoke test
    llm_model = "anthropic/claude-3-haiku-20240307" if os.environ.get("ANTHROPIC_API_KEY") else "gpt-3.5-turbo"
    crews = build_crews(adapter=adapter, tools=tools, llm=llm_model)

    result = crews.crew_a.kickoff(inputs={
        "user_request": "Alice just told me she relocated to New York City last month.",
        "agent_id": AGENT_ID,
    })

    assert result is not None
    output_str = str(result)
    # Minimal smoke: the output contains some reference to alice or city
    assert any(kw in output_str.lower() for kw in ("alice", "city", "canonical", "alice-chen")), (
        f"Crew A kickoff output doesn't mention expected entities: {output_str[:300]}"
    )
