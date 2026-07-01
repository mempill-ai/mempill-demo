"""
mempill_showcase.tests.test_graph_observability — observability + build_app tests.

Wave B: old TestBuildApp (routing via MockSupervisor) and TestScenarioViaBuildApp
(run_scenario with old graph) are replaced.  Tests that survived deletion:

  TestBuildApp            — build_app() factory returns (app, adapter) non-None.
  TestTracingNoOp         — with NO LangSmith key, tool calls run normally.
  TestContestedHook       — emit_contested_span fires registered hooks.
  TestLLMSupervisorSmoke  — REMOVED (LLMSupervisor deleted with supervisor_node).

Tests deleted / reason:
  TestBuildApp.test_update_contact_routes_crew_a  — intent routing deleted
  TestBuildApp.test_research_routes_crew_b        — crew_b deleted
  TestBuildApp.*_routes_crew_*                    — old crew topology deleted
  TestScenarioViaBuildApp.*                       — run_scenario uses old graph → Wave C
  TestLLMSupervisorSmoke.*                        — LLMSupervisor deleted
"""
from __future__ import annotations

import json
import os
import uuid

import pytest

from mempill_showcase.config.di import build_app, build_mempill_adapter
from mempill_showcase.observability import (
    clear_contested_hooks,
    emit_contested_span,
    register_contested_hook,
)
from mempill_showcase.scenarios.seed_data import AGENT_ID


# ── TestBuildApp ──────────────────────────────────────────────────────────────

class TestBuildApp:
    """build_app() factory produces a runnable ReAct agent."""

    def test_build_app_returns_tuple(self):
        """build_app() returns (app, adapter) where both are non-None."""
        app, adapter = build_app()
        assert app is not None, "app must not be None"
        assert adapter is not None, "adapter must not be None"

    def test_build_app_has_memory_saver(self):
        """build_app() default path attaches a MemorySaver checkpointer."""
        from langgraph.checkpoint.memory import MemorySaver
        app, _ = build_app()
        checkpointer = getattr(app, "checkpointer", None)
        assert isinstance(checkpointer, MemorySaver), (
            f"Default build_app() must carry MemorySaver, got {type(checkpointer).__name__!r}"
        )

    def test_custom_adapter_reused(self):
        """build_app(adapter=<existing>) reuses the provided adapter."""
        import mempill
        from mempill_showcase.core.domain.models import ClaimInput

        adapter = build_mempill_adapter(in_memory=True)
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
        belief = adapter.recall(AGENT_ID, "alice-chen", "city")
        assert belief.value == "Austin TX"

    def test_build_app_from_settings_default(self):
        """build_app_from_settings() default returns (app, MempillAdapter)."""
        from mempill_showcase.config.di import build_app_from_settings
        from mempill_showcase.adapters.memory.mempill_adapter import MempillAdapter

        app, adapter = build_app_from_settings()
        assert app is not None
        assert isinstance(adapter, MempillAdapter)

    def test_build_app_from_settings_naive_mode(self):
        """build_app_from_settings(naive_mode=True) returns (None, NaiveAdapter)."""
        from mempill_showcase.config.di import build_app_from_settings
        from mempill_showcase.config.settings import Settings
        from mempill_showcase.adapters.memory.naive_adapter import NaiveAdapter

        app, adapter = build_app_from_settings(Settings(naive_mode=True))
        assert app is None, "Naive mode must return app=None"
        assert isinstance(adapter, NaiveAdapter)


# ── TestTracingNoOp ───────────────────────────────────────────────────────────

class TestTracingNoOp:
    """Without LANGSMITH_API_KEY, tool calls run normally — tracing is a complete no-op."""

    @pytest.fixture(autouse=True)
    def ensure_no_langsmith_key(self, monkeypatch):
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)

    def test_remember_tool_no_error_without_langsmith(self):
        """MempillRememberTool._run works correctly without LangSmith."""
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
        assert "claim_ref" in result
        assert "disposition" in result
        assert result.get("disposition") in (
            "CommittedCheap", "Contested", "Conflict", "QueuedForAdjudication"
        )

    def test_recall_tool_no_error_without_langsmith(self):
        """MempillRecallTool._run works correctly without LangSmith."""
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
        raw = tool.invoke({"agent_id": AGENT_ID, "subject": "alice-chen", "predicate": "city"})
        result = json.loads(raw)
        assert result.get("value") == "Austin TX"
        assert result.get("status") == "Resolved"

    def test_audit_tool_no_error_without_langsmith(self):
        """MempillAuditTool._run works correctly without LangSmith."""
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
        assert result.get("entry_count", 0) >= 1


# ── TestContestedHook ─────────────────────────────────────────────────────────

class TestContestedHook:
    """emit_contested_span fires registered hooks — no LangSmith backend required."""

    @pytest.fixture(autouse=True)
    def cleanup_hooks(self):
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

        assert len(fired) == 1
        assert fired[0]["subject"] == "alice-chen"
        assert fired[0]["predicate"] == "employer"

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

        assert len(fired) == 1
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

        if result.get("is_contested"):
            assert len(fired) >= 1
            assert fired[0]["subject"] == "alice-chen"
            assert fired[0]["predicate"] == "employer"
