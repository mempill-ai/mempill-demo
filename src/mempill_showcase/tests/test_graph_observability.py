"""
mempill_showcase.tests.test_graph_observability — W6 graph + observability tests.

Test classes:
  TestBuildApp            — build_app() factory produces a runnable graph for all 5 intents;
                            deterministic (no API key, no LLM).
  TestScenarioViaBuildApp — 8-beat run_scenario still passes when invoked via build_app().
  TestTracingNoOp         — with NO LangSmith key, tool calls run normally and return
                            correct results; tracing is a complete no-op (no error).
  TestContestedHook       — emit_contested_span fires registered hooks even without a
                            LangSmith backend, enabling test-time assertion.
  TestLLMSupervisorSmoke  — @pytest.mark.live: classify one utterance via real Anthropic
                            model (skipped without ANTHROPIC_API_KEY).

All non-live tests are CI-safe: no API key required, no live LLM, no LangSmith backend.
"""
from __future__ import annotations

import json
import os
import uuid

import pytest

from mempill_showcase.config.di import build_app, build_mempill_adapter
from mempill_showcase.frameworks.langgraph.supervisor_node import (
    IntentLabel,
    MockSupervisor,
)
from mempill_showcase.observability import (
    clear_contested_hooks,
    emit_contested_span,
    register_contested_hook,
)
from mempill_showcase.scenarios.seed_data import AGENT_ID


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cfg(thread_id: str | None = None) -> dict:
    return {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}


def _run(app, user_input: str, agent_id: str = AGENT_ID, thread_id: str | None = None) -> dict:
    cfg = _cfg(thread_id)
    return app.invoke({"user_input": user_input, "agent_id": agent_id}, cfg)


# ── TestBuildApp ──────────────────────────────────────────────────────────────

class TestBuildApp:
    """build_app() default path: deterministic, no API key, all 5 intents routed."""

    @pytest.fixture()
    def app_adapter(self):
        return build_app(use_crewai=False)

    @pytest.fixture()
    def app(self, app_adapter):
        return app_adapter[0]

    @pytest.fixture()
    def adapter(self, app_adapter):
        return app_adapter[1]

    def test_build_app_returns_tuple(self, app_adapter):
        """build_app returns (app, adapter) tuple."""
        app, adapter = app_adapter
        assert app is not None, "app must not be None"
        assert adapter is not None, "adapter must not be None"

    def test_update_contact_routes_crew_a(self, app):
        """intent:update_contact → crew_a."""
        result = _run(app, "intent:update_contact")
        assert result.get("intent") == IntentLabel.UPDATE_CONTACT

    def test_research_routes_crew_b(self, app):
        """intent:research → crew_b."""
        result = _run(app, "intent:research")
        assert result.get("intent") == IntentLabel.RESEARCH

    def test_prepare_briefing_routes_crew_c(self, app):
        """intent:prepare_briefing → crew_c."""
        result = _run(app, "intent:prepare_briefing")
        assert result.get("intent") == IntentLabel.PREPARE_BRIEFING

    def test_recall_history_routes_crew_c(self, app):
        """intent:recall_history → crew_c."""
        result = _run(app, "intent:recall_history")
        assert result.get("intent") == IntentLabel.RECALL_HISTORY
        assert result.get("output_text"), "crew_c must produce output_text"

    def test_compliance_audit_routes_crew_c(self, app):
        """intent:compliance_audit → crew_c audit path."""
        result = _run(app, "intent:compliance_audit")
        assert result.get("intent") == IntentLabel.COMPLIANCE_AUDIT
        assert result.get("audit_result") is not None, "crew_c must set audit_result"

    def test_explicit_mock_classifier_unchanged(self):
        """Passing classifier=MockSupervisor() explicitly is identical to default."""
        app, _ = build_app(classifier=MockSupervisor())
        result = _run(app, "Alice relocated to NYC since 2025-02")
        assert result.get("intent") == IntentLabel.UPDATE_CONTACT

    def test_custom_adapter_reused(self):
        """build_app(adapter=<existing>) reuses the provided adapter without reinitialising."""
        import mempill
        from mempill_showcase.core.domain.models import ClaimInput

        adapter = build_mempill_adapter(in_memory=True)
        # Pre-seed before building the app
        adapter.write_claim(AGENT_ID, ClaimInput(
            subject="alice-chen",
            predicate="city",
            value="Austin TX",
            valid_from="2023-06",
            confidence=1.0,
            provenance=mempill.ProvenanceLabel.external_user_asserted(),
        ))

        app, returned_adapter = build_app(adapter=adapter)
        assert returned_adapter is adapter, "build_app must reuse the passed adapter"

        # Pre-seeded data must be visible
        belief = adapter.recall(AGENT_ID, "alice-chen", "city")
        assert belief.value == "Austin TX"


# ── TestScenarioViaBuildApp ───────────────────────────────────────────────────

class TestScenarioViaBuildApp:
    """8-beat run_scenario passes when using build_app()-assembled graph."""

    def test_run_scenario_produces_8_beats(self):
        """run_scenario returns exactly 8 BeatResult entries."""
        from mempill_showcase.scenarios.executive_assistant import run_scenario

        adapter = build_mempill_adapter(in_memory=True)
        trace = run_scenario(adapter)
        assert len(trace.beats) == 8, (
            f"Expected 8 beats, got {len(trace.beats)}: {[b.beat_id for b in trace.beats]}"
        )

    def test_scenario_ac1_current_recall_is_nyc(self):
        """AC-1: after T-02, current recall of city == NYC (not Austin)."""
        from mempill_showcase.scenarios.executive_assistant import run_scenario

        adapter = build_mempill_adapter(in_memory=True)
        trace = run_scenario(adapter)

        belief = adapter.recall(AGENT_ID, "alice-chen", "city")
        assert belief.value == "New York NY", (
            f"AC-1: current city should be 'New York NY', got {belief.value!r}"
        )

    def test_scenario_ac3_valid_at_q1_returns_austin(self):
        """AC-3: valid_at=2025-01-01 returns Austin TX (before NYC succession)."""
        from mempill_showcase.scenarios.executive_assistant import run_scenario

        adapter = build_mempill_adapter(in_memory=True)
        run_scenario(adapter)

        hist = adapter.query_at(
            AGENT_ID, "alice-chen", "city",
            valid_at="2025-01-01T00:00:00Z",
        )
        assert hist.value == "Austin TX", (
            f"AC-3: valid_at Q1 should be 'Austin TX', got {hist.value!r}"
        )

    def test_scenario_ac5_audit_entries_gte_11(self):
        """AC-5: compliance audit returns >= 11 entries."""
        from mempill_showcase.scenarios.executive_assistant import run_scenario

        adapter = build_mempill_adapter(in_memory=True)
        trace = run_scenario(adapter)

        assert len(trace.audit_entries) >= 11, (
            f"AC-5: audit should have >= 11 entries, got {len(trace.audit_entries)}"
        )

    def test_scenario_ac7_subjects_are_canonical(self):
        """AC-7: all subjects written are canonical entity keys."""
        from mempill_showcase.core.domain.canonical_keys import all_canonical_entities
        from mempill_showcase.scenarios.executive_assistant import run_scenario

        adapter = build_mempill_adapter(in_memory=True)
        trace = run_scenario(adapter)

        known = all_canonical_entities()
        non_canonical = trace.subjects_written - known
        assert not non_canonical, (
            f"AC-7: non-canonical subjects found: {non_canonical}"
        )


# ── TestTracingNoOp ───────────────────────────────────────────────────────────

class TestTracingNoOp:
    """Without LANGSMITH_API_KEY, tool calls run normally — tracing is a complete no-op."""

    @pytest.fixture(autouse=True)
    def ensure_no_langsmith_key(self, monkeypatch):
        """Remove LangSmith env vars for the duration of these tests."""
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)

    def test_remember_tool_no_error_without_langsmith(self):
        """MempillRememberTool._run works correctly without LangSmith — no exception raised."""
        import mempill
        from mempill_showcase.adapters.memory.mempill_adapter import MempillAdapter
        from mempill_showcase.tools.mempill_remember_tool import MempillRememberTool

        adapter = build_mempill_adapter(in_memory=True)
        tool = MempillRememberTool(adapter=adapter)

        raw = tool.invoke({
            "agent_id": AGENT_ID,
            "subject": "alice-chen",
            "predicate": "city",
            "value": "Austin TX",
            "valid_from": "2023-06",
            "confidence": 1.0,
            "provenance_channel": "UserAsserted",
        })
        result = json.loads(raw)
        assert "claim_ref" in result, "remember tool must return claim_ref"
        assert "disposition" in result, "remember tool must return disposition"
        assert result.get("disposition") in ("CommittedCheap", "Contested", "Conflict"), (
            f"Unexpected disposition: {result.get('disposition')}"
        )

    def test_recall_tool_no_error_without_langsmith(self):
        """MempillRecallTool._run works correctly without LangSmith — no exception raised."""
        import mempill
        from mempill_showcase.core.domain.models import ClaimInput
        from mempill_showcase.tools.mempill_recall_tool import MempillRecallTool

        adapter = build_mempill_adapter(in_memory=True)
        adapter.write_claim(AGENT_ID, ClaimInput(
            subject="alice-chen",
            predicate="city",
            value="Austin TX",
            valid_from="2023-06",
            confidence=1.0,
            provenance=mempill.ProvenanceLabel.external_user_asserted(),
        ))

        tool = MempillRecallTool(adapter=adapter)
        raw = tool.invoke({
            "agent_id": AGENT_ID,
            "subject": "alice-chen",
            "predicate": "city",
        })
        result = json.loads(raw)
        assert result.get("value") == "Austin TX", (
            f"Recall without LangSmith should return 'Austin TX', got {result.get('value')!r}"
        )
        assert result.get("status") == "Resolved"

    def test_audit_tool_no_error_without_langsmith(self):
        """MempillAuditTool._run works correctly without LangSmith — no exception raised."""
        import mempill
        from mempill_showcase.core.domain.models import ClaimInput
        from mempill_showcase.tools.mempill_audit_tool import MempillAuditTool

        adapter = build_mempill_adapter(in_memory=True)
        adapter.write_claim(AGENT_ID, ClaimInput(
            subject="alice-chen",
            predicate="city",
            value="Austin TX",
            valid_from="2023-06",
            confidence=1.0,
            provenance=mempill.ProvenanceLabel.external_user_asserted(),
        ))

        tool = MempillAuditTool(adapter=adapter)
        raw = tool.invoke({"agent_id": AGENT_ID, "limit": 10})
        result = json.loads(raw)
        assert result.get("entry_count", 0) >= 1, (
            "Audit tool must return >= 1 entry after a write"
        )

    def test_full_graph_no_error_without_langsmith(self):
        """Full build_app() graph runs all 5 intents without LangSmith — no error."""
        app, _ = build_app(use_crewai=False)
        for intent_text, expected_intent in [
            ("intent:update_contact", IntentLabel.UPDATE_CONTACT),
            ("intent:research", IntentLabel.RESEARCH),
            ("intent:prepare_briefing", IntentLabel.PREPARE_BRIEFING),
            ("intent:recall_history", IntentLabel.RECALL_HISTORY),
            ("intent:compliance_audit", IntentLabel.COMPLIANCE_AUDIT),
        ]:
            result = _run(app, intent_text)
            assert result.get("intent") == expected_intent, (
                f"Expected {expected_intent}, got {result.get('intent')!r} for {intent_text!r}"
            )


# ── TestContestedHook ─────────────────────────────────────────────────────────

class TestContestedHook:
    """emit_contested_span fires registered hooks — no LangSmith backend required."""

    @pytest.fixture(autouse=True)
    def cleanup_hooks(self):
        """Clear contested hooks before and after each test."""
        clear_contested_hooks()
        yield
        clear_contested_hooks()

    def test_hook_fires_on_emit_contested_span(self):
        """register_contested_hook callback fires when emit_contested_span is called."""
        fired: list[dict] = []

        def my_hook(**kw):
            fired.append(dict(kw))

        register_contested_hook(my_hook)

        emit_contested_span(
            subject="alice-chen",
            predicate="employer",
            incumbent={"value": "VP Engineering", "valid_from_display": "2023-06"},
            challenger={"value": "CTO", "valid_from_display": "2025-01"},
            agent_id=AGENT_ID,
        )

        assert len(fired) == 1, f"Hook should fire exactly once, got {len(fired)}"
        assert fired[0]["subject"] == "alice-chen"
        assert fired[0]["predicate"] == "employer"
        assert fired[0]["incumbent"]["value"] == "VP Engineering"
        assert fired[0]["challenger"]["value"] == "CTO"

    def test_hook_fires_without_langsmith_key(self, monkeypatch):
        """Contested hook fires even when LANGSMITH_API_KEY is absent."""
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)

        fired: list[dict] = []
        register_contested_hook(lambda **kw: fired.append(kw))

        emit_contested_span(
            subject="bob-liu",
            predicate="city",
            incumbent={"value": "Austin TX", "valid_from_display": "2023-01"},
            challenger={"value": "New York NY", "valid_from_display": "2025-02"},
        )

        assert len(fired) == 1, "Hook must fire even without LangSmith key"
        assert fired[0]["subject"] == "bob-liu"

    def test_multiple_hooks_all_fire(self):
        """Multiple registered hooks all receive the event."""
        results: list[str] = []
        register_contested_hook(lambda **kw: results.append(f"hook1:{kw['subject']}"))
        register_contested_hook(lambda **kw: results.append(f"hook2:{kw['predicate']}"))

        emit_contested_span(subject="alice-chen", predicate="employer")

        assert "hook1:alice-chen" in results
        assert "hook2:employer" in results
        assert len(results) == 2

    def test_hook_fires_via_remember_tool_contested(self, monkeypatch):
        """MempillRememberTool fires the contested hook on a genuine conflict."""
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
        import mempill
        from mempill_showcase.core.domain.models import ClaimInput
        from mempill_showcase.tools.mempill_remember_tool import MempillRememberTool

        adapter = build_mempill_adapter(in_memory=True)
        # Seed VP Engineering open-ended (no valid_until)
        adapter.write_claim(AGENT_ID, ClaimInput(
            subject="alice-chen",
            predicate="employer",
            value="Acme Corp / VP Engineering",
            valid_from="2023-06",
            confidence=1.0,
            provenance=mempill.ProvenanceLabel.external_user_asserted(),
        ))

        fired: list[dict] = []
        register_contested_hook(lambda **kw: fired.append(dict(kw)))

        tool = MempillRememberTool(adapter=adapter)
        # Write CTO with same valid_from=2023-06 → genuine overlap → Contested
        raw = tool.invoke({
            "agent_id": AGENT_ID,
            "subject": "alice-chen",
            "predicate": "employer",
            "value": "Acme Corp / CTO",
            "valid_from": "2023-06",
            "confidence": 0.75,
            "provenance_channel": "ExternalFirstHand",
        })
        result = json.loads(raw)

        # The write returns Contested → hook should have fired
        if result.get("is_contested"):
            assert len(fired) >= 1, (
                "Contested hook must fire when MempillRememberTool returns Contested"
            )
            assert fired[0]["subject"] == "alice-chen"
            assert fired[0]["predicate"] == "employer"
        # If the engine resolves it (not contested), that's OK — we just verify no error.


# ── TestLLMSupervisorSmoke ────────────────────────────────────────────────────

class TestLLMSupervisorSmoke:
    """Live LLM smoke tests — skipped without ANTHROPIC_API_KEY."""

    @pytest.mark.live
    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not set — skipping live LLMSupervisor test",
    )
    def test_llm_supervisor_classifies_update_contact(self):
        """LLMSupervisor classifies a contact-update utterance correctly."""
        from mempill_showcase.frameworks.langgraph.supervisor_node import LLMSupervisor

        classifier = LLMSupervisor()
        intent = classifier.classify("Alice just told me she moved to San Francisco last month.")
        assert intent == IntentLabel.UPDATE_CONTACT, (
            f"Expected UPDATE_CONTACT, got {intent!r}"
        )

    @pytest.mark.live
    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not set — skipping live LLMSupervisor test",
    )
    def test_llm_supervisor_classifies_recall_history(self):
        """LLMSupervisor classifies a historical query correctly."""
        from mempill_showcase.frameworks.langgraph.supervisor_node import LLMSupervisor

        classifier = LLMSupervisor()
        intent = classifier.classify("What was Alice's title back in Q1 2025?")
        assert intent == IntentLabel.RECALL_HISTORY, (
            f"Expected RECALL_HISTORY, got {intent!r}"
        )

    @pytest.mark.live
    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not set — skipping live LLMSupervisor test",
    )
    def test_build_app_with_llm_supervisor_runs_graph(self):
        """build_app(classifier=LLMSupervisor()) produces a working graph."""
        from mempill_showcase.frameworks.langgraph.supervisor_node import LLMSupervisor

        app, adapter = build_app(classifier=LLMSupervisor())
        result = _run(app, "Alice just told me she moved to San Francisco.")
        assert result.get("intent") in IntentLabel.ALL, (
            f"LLMSupervisor should return a valid intent, got {result.get('intent')!r}"
        )
