"""
tests.test_router_graph — Live E2E routing tests for the dual-agent router
(TASK-31 T31-4, Wave 3).

Marked with @pytest.mark.live — excluded from the default test run:
  pytest -m "not live"          # fast CI suite (skips this file)
  pytest -m live                # run only live tests (requires ANTHROPIC_API_KEY)

Exercises the REAL structured-output classifier (route_query) through the
top-level router graph built by build_router_graph(), dispatching to two
build_agent_instance() subgraphs (people_ops / org_registry). Uses a
pytest tmp_path db_dir so these tests never touch the repo's own .mempill/
directory.

Coverage:
  test_people_ops_query_routes_and_answers_from_people_db —
    "What is Alice Chen's dietary restriction?" routes to people_ops and the
    answer draws from the seeded people DB (vegetarian).
  test_org_registry_query_routes_and_answers_from_org_db —
    "Who is Acme's CEO?" routes to org_registry and the answer draws from the
    seeded org DB (Diane Foster).
  test_ambiguous_query_routes_to_people_ops_default —
    An ambiguous message routes to the people_ops default (ARCHITECTURE.md §2
    ambiguity policy), confirmed via state['route']/state['agent_id'].
"""
from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# Load .env from the project root so ANTHROPIC_API_KEY is available when
# pytest is invoked from outside the project directory.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

from mempill_showcase.config.di import build_agent_instance  # noqa: E402
from mempill_showcase.frameworks.langgraph.agents_config import (  # noqa: E402
    ORG_REGISTRY_SPEC,
    PEOPLE_OPS_SPEC,
)
from mempill_showcase.frameworks.langgraph.router_graph import build_router_graph  # noqa: E402
from mempill_showcase.scenarios.seed_data import (  # noqa: E402
    ORG_REGISTRY_SEED_CLAIMS,
    PEOPLE_OPS_SEED_CLAIMS,
    load_seed_claims,
)


def _last_ai_text(result: dict) -> str:
    """Extract the last non-empty AI message text from a graph result."""
    msgs = result.get("messages", [])
    for msg in reversed(msgs):
        cls = msg.__class__.__name__
        if cls not in ("AIMessage", "AIMessageChunk"):
            continue
        content = msg.content
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            texts = [
                c.get("text", "") if isinstance(c, dict) else str(c)
                for c in content
            ]
            joined = " ".join(t for t in texts if t.strip())
            if joined.strip():
                return joined.strip()
    return ""


def _build_seeded_router(tmp_path):
    """Build a router graph with both agent instances, seeded, file-backed under tmp_path.

    Returns (router_app, people_ops_adapter, org_registry_adapter).
    """
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

    router_app = build_router_graph(people_ops_subgraph, org_registry_subgraph)
    return router_app, people_ops_adapter, org_registry_adapter


@pytest.fixture()
def seeded_router(tmp_path):
    """Fresh file-backed (tmp_path) dual-agent router, both DBs seeded."""
    return _build_seeded_router(tmp_path)


@pytest.mark.live
def test_people_ops_query_routes_and_answers_from_people_db(seeded_router):
    """Real classifier routes a person-named question to people_ops."""
    app, _, _ = seeded_router
    config = {"configurable": {"thread_id": "router-people-ops"}}

    result = app.invoke(
        {"messages": [HumanMessage(content="What is Alice Chen's dietary restriction?")]},
        config=config,
    )

    assert result.get("route") == "people_ops", (
        f"Expected route='people_ops', got {result.get('route')!r}"
    )
    assert result.get("agent_id") == PEOPLE_OPS_SPEC.agent_id, (
        f"Expected agent_id={PEOPLE_OPS_SPEC.agent_id!r}, got {result.get('agent_id')!r}"
    )

    answer = _last_ai_text(result)
    assert "vegetarian" in answer.lower(), (
        f"Expected 'vegetarian' in agent answer (from people DB), got: {answer!r}"
    )


@pytest.mark.live
def test_org_registry_query_routes_and_answers_from_org_db(seeded_router):
    """Real classifier routes an org/role-named question to org_registry."""
    app, _, _ = seeded_router
    config = {"configurable": {"thread_id": "router-org-registry"}}

    result = app.invoke(
        {"messages": [HumanMessage(content="Who is Acme's CEO?")]},
        config=config,
    )

    assert result.get("route") == "org_registry", (
        f"Expected route='org_registry', got {result.get('route')!r}"
    )
    assert result.get("agent_id") == ORG_REGISTRY_SPEC.agent_id, (
        f"Expected agent_id={ORG_REGISTRY_SPEC.agent_id!r}, got {result.get('agent_id')!r}"
    )

    answer = _last_ai_text(result)
    assert "diane" in answer.lower() or "foster" in answer.lower(), (
        f"Expected Diane Foster in agent answer (from org DB), got: {answer!r}"
    )


@pytest.mark.live
def test_ambiguous_query_routes_to_people_ops_default(seeded_router):
    """An ambiguous message (no person/org named) routes to the people_ops default."""
    app, _, _ = seeded_router
    config = {"configurable": {"thread_id": "router-ambiguous"}}

    result = app.invoke(
        {"messages": [HumanMessage(content="tell me about the situation")]},
        config=config,
    )

    assert result.get("route") == "people_ops", (
        f"Expected ambiguous input to default-route to 'people_ops', got "
        f"{result.get('route')!r}"
    )
    assert result.get("agent_id") == PEOPLE_OPS_SPEC.agent_id, (
        f"Expected agent_id={PEOPLE_OPS_SPEC.agent_id!r}, got {result.get('agent_id')!r}"
    )
