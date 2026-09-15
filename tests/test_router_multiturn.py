"""
tests.test_router_multiturn — Multi-turn router `messages` semantics
(TASK-33 item 3, test-first).

FINDING (empirically verified, see class docstrings below): plain
`list[Any]` router state, with NO reducer, is last-write-wins — every
`app.invoke(..., config={"thread_id": X})` call on an EXISTING thread fully
OVERWRITES the checkpointed `messages` channel with just that call's input,
discarding all prior turns before any node runs. A live end-to-end
reproduction (people_ops "What is Alice Chen's dietary restriction?" then,
same thread, "What about her city?") showed the SECOND turn losing all
context: the agent asked the user to clarify who "her" was instead of
answering "Austin TX" from turn 1.

VERDICT: genuinely broken (history lost causing wrong behavior — not a
crash, but an incorrect/degraded end-to-end result). Minimal fix applied:
`RouterState.messages` (and, for consistency, `ExecAssistantState.messages`)
now carry `Annotated[list, _safe_add_messages]` — see
frameworks/langgraph/router_state.py's module docstring for the full
rationale, in particular why the built-in `langgraph.graph.message.
add_messages` reducer could NOT be used as-is: it raises on the exact
primitive-junk items (int/float/bool/None, malformed dicts) that TASK-33
item 1's `_normalize_messages` router-input-hardening fix defends against,
and that coercion happens at the CHANNEL-MERGE step — BEFORE route_query's
own normalization ever runs. `_safe_add_messages` gets real cross-turn
accumulation WITHOUT reintroducing that crash.

This file:
  TestSafeAddMessagesReducer      — unit tests of the reducer function itself.
  TestRouterGraphAccumulation     — deterministic, dummy-subgraph, full-graph
                                     proof that a router-level MemorySaver +
                                     the new reducer accumulates (not wipes)
                                     `messages` across two invokes on one
                                     thread_id.
  TestMultiTurnFollowupLive       — live end-to-end reproduction of the
                                     original bug + proof it is now fixed
                                     (marked @pytest.mark.live; requires a
                                     real classifier + real ReAct subgraphs).
"""
from __future__ import annotations

import os
import uuid

import pytest
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

from mempill_showcase.config.di import build_agent_instance  # noqa: E402
from mempill_showcase.frameworks.langgraph.agents_config import (  # noqa: E402
    ORG_REGISTRY_SPEC,
    PEOPLE_OPS_SPEC,
)
from mempill_showcase.frameworks.langgraph.router_graph import build_router_graph  # noqa: E402
from mempill_showcase.frameworks.langgraph.router_state import (  # noqa: E402
    RouterState,
    _safe_add_messages,
)
from mempill_showcase.scenarios.seed_data import (  # noqa: E402
    ORG_REGISTRY_SEED_CLAIMS,
    PEOPLE_OPS_SEED_CLAIMS,
    load_seed_claims,
)


def _cfg(thread_id: str | None = None) -> dict:
    return {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}


class TestSafeAddMessagesReducer:
    """_safe_add_messages: real accumulation for valid items, graceful
    (warning-logged) drop instead of crash for non-coercible junk."""

    def test_accumulates_valid_messages_across_calls(self):
        left = _safe_add_messages([], [HumanMessage(content="turn1")])
        merged = _safe_add_messages(left, [AIMessage(content="turn1 answer")])
        merged = _safe_add_messages(merged, [HumanMessage(content="turn2")])
        assert [m.content for m in merged] == ["turn1", "turn1 answer", "turn2"]

    def test_same_id_message_updates_in_place_not_duplicated(self):
        msg = HumanMessage(content="hello", id="fixed-id-1")
        left = _safe_add_messages([], [msg])
        # Re-merging the SAME object/id must not duplicate it.
        merged = _safe_add_messages(left, [msg])
        assert len(merged) == 1

    def test_raw_int_dropped_not_raised(self):
        merged = _safe_add_messages([], [2025])
        assert merged == []

    def test_mixed_valid_and_junk_keeps_only_valid(self):
        merged = _safe_add_messages(
            [], [HumanMessage(content="ok"), 2025, None, True, {"bad": "dict"}]
        )
        assert len(merged) == 1
        assert merged[0].content == "ok"

    def test_junk_in_second_call_does_not_wipe_first_call_history(self):
        left = _safe_add_messages([], [HumanMessage(content="turn1")])
        merged = _safe_add_messages(left, [2025])  # junk-only second update
        assert len(merged) == 1
        assert merged[0].content == "turn1"

    def test_junk_in_existing_left_history_does_not_block_resume(self):
        """Copilot review (PR #57): a pre-existing checkpoint (e.g. written
        before this reducer existed, or corrupted by any other path) could
        already carry non-coercible junk in `left`. The reducer must filter
        BOTH sides, not just `right` — otherwise add_messages(left, ...)
        raises on the same junk forever, permanently blocking that thread."""
        junky_left = [HumanMessage(content="old turn"), 2025, None]
        merged = _safe_add_messages(junky_left, [HumanMessage(content="new turn")])
        contents = [m.content for m in merged]
        assert contents == ["old turn", "new turn"], (
            f"Expected junk in existing history to be filtered and resume to "
            f"succeed, got {contents!r}"
        )

    def test_dropped_item_warning_never_logs_raw_content(self, caplog):
        """Copilot review (PR #57): dropped items may carry user-provided
        content (e.g. a malformed dict) and must never be repr()'d into logs —
        only counts/types."""
        import logging

        sensitive_dict = {"secret": "super-sensitive-pii-value-12345"}
        with caplog.at_level(logging.WARNING):
            _safe_add_messages([], [2025, sensitive_dict])

        log_text = "\n".join(r.getMessage() for r in caplog.records)
        assert "super-sensitive-pii-value-12345" not in log_text
        assert "secret" not in log_text


class TestRouterGraphAccumulation:
    """Deterministic, full-graph (dummy subgraph, monkeypatched classifier)
    proof: two invokes on ONE thread_id with a MemorySaver checkpointer
    accumulate `messages` instead of the pre-fix last-write-wins wipe."""

    @staticmethod
    def _dummy_subgraph():
        def _echo(state):
            return {"messages": [AIMessage(content=f"ack (saw {len(state.get('messages', []))} msgs)")]}

        g = StateGraph(RouterState)
        g.add_node("n", _echo)
        g.set_entry_point("n")
        g.add_edge("n", END)
        return g.compile()

    def test_two_invokes_same_thread_accumulate_messages(self, monkeypatch):
        import mempill_showcase.frameworks.langgraph.router_graph as rg_mod

        class _FakeClassifier:
            def invoke(self, prompt):
                from mempill_showcase.frameworks.langgraph.router_graph import RouteDecision
                return RouteDecision(agent="people_ops", rationale="test")

        monkeypatch.setattr(rg_mod, "_build_classifier", lambda model_name=None: _FakeClassifier())

        sub1 = self._dummy_subgraph()
        sub2 = self._dummy_subgraph()
        app = build_router_graph(sub1, sub2, checkpointer=MemorySaver())

        cfg = _cfg("accum-thread")
        r1 = app.invoke({"messages": [HumanMessage(content="turn1")]}, cfg)
        assert [m.content for m in r1["messages"]] == ["turn1", "ack (saw 1 msgs)"]

        r2 = app.invoke({"messages": [HumanMessage(content="turn2")]}, cfg)
        # Pre-fix (no reducer) this would be exactly ["turn2", "ack (saw 1 msgs)"]
        # (history from turn 1 wiped). Fixed behavior: turn 1's history survives.
        contents = [m.content for m in r2["messages"]]
        assert contents == ["turn1", "ack (saw 1 msgs)", "turn2", "ack (saw 3 msgs)"], (
            f"Expected full cross-turn accumulation, got {contents!r}"
        )

    def test_junk_second_turn_does_not_wipe_first_turn_or_crash(self, monkeypatch):
        """A junk-only second invoke (the router-hardening regression, item 1)
        combined with an existing thread history (item 3) must neither crash
        nor wipe turn 1 — the two fixes must compose safely."""
        import mempill_showcase.frameworks.langgraph.router_graph as rg_mod

        class _FakeClassifier:
            def invoke(self, prompt):
                from mempill_showcase.frameworks.langgraph.router_graph import RouteDecision
                return RouteDecision(agent="people_ops", rationale="test")

        monkeypatch.setattr(rg_mod, "_build_classifier", lambda model_name=None: _FakeClassifier())

        sub1 = self._dummy_subgraph()
        sub2 = self._dummy_subgraph()
        app = build_router_graph(sub1, sub2, checkpointer=MemorySaver())

        cfg = _cfg("accum-junk-thread")
        r1 = app.invoke({"messages": [HumanMessage(content="turn1")]}, cfg)
        assert len(r1["messages"]) == 2

        r2 = app.invoke({"messages": [2025]}, cfg)  # must not raise
        assert all(not isinstance(m, int) for m in r2["messages"])
        # turn 1's history must still be present.
        assert r2["messages"][0].content == "turn1"


class TestEmptyMessagesTerminalRoute:
    """Copilot review (PR #58 comment 1): empty/junk-only input should
    short-circuit to END (terminal route "_no_route") instead of dispatching
    to a subgraph. This avoids unnecessary downstream work and correctly
    reflects the intent: a friendly message, not a subgraph response."""

    @staticmethod
    def _dummy_subgraph():
        def _echo(state):
            return {"messages": [AIMessage(content=f"ack (saw {len(state.get('messages', []))} msgs)")]}

        g = StateGraph(RouterState)
        g.add_node("n", _echo)
        g.set_entry_point("n")
        g.add_edge("n", END)
        return g.compile()

    def test_empty_messages_terminal_route_no_subgraph_call(self, monkeypatch):
        """Junk-only input (e.g. [2025]) should set route='_no_route' and
        terminate with a friendly message, not dispatch to a subgraph."""
        import mempill_showcase.frameworks.langgraph.router_graph as rg_mod

        class _FakeClassifier:
            def invoke(self, prompt):
                from mempill_showcase.frameworks.langgraph.router_graph import RouteDecision
                return RouteDecision(agent="people_ops", rationale="test")

        monkeypatch.setattr(rg_mod, "_build_classifier", lambda model_name=None: _FakeClassifier())

        sub1 = self._dummy_subgraph()
        sub2 = self._dummy_subgraph()
        app = build_router_graph(sub1, sub2, checkpointer=MemorySaver())

        cfg = _cfg("empty-messages-thread")
        result = app.invoke({"messages": [2025]}, cfg)  # junk-only input

        # Route must be set to terminal marker, not a real route.
        assert result.get("route") == "_no_route", (
            f"Expected route='_no_route' for junk-only input, got {result.get('route')!r}"
        )

        # The friendly message must be present (the terminal message,
        # not an agent's subgraph response).
        msgs = result.get("messages", [])
        assert len(msgs) == 1
        assert "didn't receive a usable message" in msgs[0].content.lower()


@pytest.mark.live
class TestMultiTurnFollowupLive:
    """Live end-to-end reproduction: a pronoun follow-up on the SAME thread_id
    now resolves correctly using turn 1's context (was broken pre-fix — the
    agent asked the user to clarify who "her" was instead of answering)."""

    @pytest.fixture()
    def seeded_checkpointed_router(self, tmp_path):
        db_dir = str(tmp_path / ".mempill")
        people_ops_adapter, people_ops_subgraph = build_agent_instance(
            agent_id=PEOPLE_OPS_SPEC.agent_id,
            responsibility=PEOPLE_OPS_SPEC.responsibility,
            known_entities=PEOPLE_OPS_SPEC.known_entities,
            db_dir=db_dir,
        )
        org_registry_adapter, org_registry_subgraph = build_agent_instance(
            agent_id=ORG_REGISTRY_SPEC.agent_id,
            responsibility=ORG_REGISTRY_SPEC.responsibility,
            known_entities=ORG_REGISTRY_SPEC.known_entities,
            db_dir=db_dir,
        )
        load_seed_claims(
            people_ops_adapter, agent_id=PEOPLE_OPS_SPEC.agent_id, claims=PEOPLE_OPS_SEED_CLAIMS,
        )
        load_seed_claims(
            org_registry_adapter, agent_id=ORG_REGISTRY_SPEC.agent_id, claims=ORG_REGISTRY_SEED_CLAIMS,
        )
        app = build_router_graph(people_ops_subgraph, org_registry_subgraph, checkpointer=MemorySaver())
        return app

    def test_pronoun_followup_resolves_from_turn1_context(self, seeded_checkpointed_router):
        app = seeded_checkpointed_router
        cfg = _cfg("multiturn-followup")

        r1 = app.invoke(
            {"messages": [HumanMessage(content="What is Alice Chen's dietary restriction?")]},
            cfg,
        )
        assert r1.get("route") == "people_ops"
        turn1_len = len(r1["messages"])
        assert turn1_len >= 2  # at least the human turn + a final AI answer

        r2 = app.invoke({"messages": [HumanMessage(content="What about her city?")]}, cfg)
        assert r2.get("route") == "people_ops"
        # Cross-turn accumulation: turn 2's message list must be STRICTLY
        # longer than turn 1's, not reset to just the new turn.
        assert len(r2["messages"]) > turn1_len, (
            f"Expected accumulated history (> {turn1_len} messages), "
            f"got {len(r2['messages'])}"
        )

        final_answer = r2["messages"][-1].content
        assert "austin" in final_answer.lower(), (
            f"Expected the follow-up to resolve 'her' to Alice Chen and answer "
            f"'Austin' from turn 1's context, got: {final_answer!r}"
        )
