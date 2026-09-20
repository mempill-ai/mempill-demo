"""
tests.test_router_hitl_live — HITL interrupt/resume driven through the TOP-LEVEL
router graph (TASK-33 item 2), not the bare single-agent subgraph as in
src/mempill_showcase/tests/test_w7_oracle.py::TestGraphHITLRealOracle.

Proves the router's OWN checkpointer (build_router_graph(..., checkpointer=...))
correctly persists/resumes state INSIDE the dispatched subgraph, and that a
resolved verdict lands in the CORRECT agent's independent oracle queue only —
the other agent's oracle queue (people-ops vs org-registry) is untouched.

Why @pytest.mark.live: reaching the interrupt requires (1) the real classifier
to route the message, and (2) the real ReAct agent LLM to decide to call
remember_fact -> observe is_contested=True -> call request_adjudication. Per
SPIKE_RESULTS.md and test_w7_oracle.py::TestGraphHITLRealOracle's precedent,
this tool-invocation decision cannot be reached deterministically (no LLM) —
the oracle MECHANICS (queue/Affirm/Deny) are already proven without a graph in
test_w7_oracle.py::TestOracleAffirmPath / TestOracleDenyPath.
"""
from __future__ import annotations

import os
import uuid

import pytest
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

# Load .env from the project root so ANTHROPIC_API_KEY is available when
# pytest is invoked from outside the project directory.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

from mempill import ProvenanceLabel  # noqa: E402
from mempill_showcase.config.di import build_agent_instance  # noqa: E402
from mempill_showcase.core.domain.models import ClaimInput  # noqa: E402
from mempill_showcase.frameworks.langgraph.agents_config import (  # noqa: E402
    ORG_REGISTRY_SPEC,
    PEOPLE_OPS_SPEC,
)
from mempill_showcase.frameworks.langgraph.router_graph import build_router_graph  # noqa: E402


def _cfg(thread_id: str | None = None) -> dict:
    return {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}


def _seed_vp(adapter) -> None:
    """Seed alice-chen/employer = VP Engineering (open-ended, 2023-06) on the
    people-ops adapter — the incumbent the LLM's CTO write will contest."""
    adapter.write_claim(PEOPLE_OPS_SPEC.agent_id, ClaimInput(
        subject="alice-chen",
        predicate="employer",
        value="Acme Corp / VP Engineering",
        valid_from="2023-06",
        confidence=1.0,
        provenance=ProvenanceLabel.external_user_asserted(),
        cardinality="Functional",
    ))


def _seed_org_decoy_conflict(adapter) -> None:
    """Seed an INDEPENDENT contested conflict on the org-registry adapter —
    a decoy pending adjudication that must remain untouched by the people-ops
    resolution, proving the two agents' oracle queues do not interfere."""
    adapter.write_claim(ORG_REGISTRY_SPEC.agent_id, ClaimInput(
        subject="acme-corp",
        predicate="ceo",
        value="Diane Foster",
        valid_from="2021-04",
        confidence=1.0,
        provenance=ProvenanceLabel.external_first_hand(),
        cardinality="Functional",
    ))
    adapter.write_claim(ORG_REGISTRY_SPEC.agent_id, ClaimInput(
        subject="acme-corp",
        predicate="ceo",
        value="Marcus Reyes",
        valid_from="2021-04",
        confidence=0.9,
        provenance=ProvenanceLabel.external_first_hand(),
        cardinality="Functional",
    ))


@pytest.fixture()
def checkpointed_router():
    """Router graph with a MemorySaver checkpointer + REAL build_agent_instance
    subgraphs (in-memory engines) for both people-ops and org-registry.

    Each subgraph is still compiled with checkpointer=None internally (per
    build_agent_instance's contract) — the ROUTER's MemorySaver governs
    persistence for the whole tree, including interrupt/resume inside a
    dispatched subgraph (SPIKE_RESULTS.md Smoke 1).

    Returns (router_app, people_ops_adapter, org_registry_adapter).
    """
    people_ops_adapter, people_ops_subgraph = build_agent_instance(
        agent_id=PEOPLE_OPS_SPEC.agent_id,
        responsibility=PEOPLE_OPS_SPEC.responsibility,
        known_entities=PEOPLE_OPS_SPEC.known_entities,
        db_dir=None,
    )
    org_registry_adapter, org_registry_subgraph = build_agent_instance(
        agent_id=ORG_REGISTRY_SPEC.agent_id,
        responsibility=ORG_REGISTRY_SPEC.responsibility,
        known_entities=ORG_REGISTRY_SPEC.known_entities,
        db_dir=None,
    )
    router_app = build_router_graph(
        people_ops_subgraph, org_registry_subgraph, checkpointer=MemorySaver()
    )
    return router_app, people_ops_adapter, org_registry_adapter


@pytest.mark.live
class TestRouterLevelHITL:
    """Router-level (not bare-subgraph) interrupt/resume + cross-agent isolation."""

    def test_interrupt_surfaces_at_router_level(self, checkpointed_router):
        """(a) The subgraph's interrupt propagates up through the router's
        top-level invoke() result as `__interrupt__`."""
        app, people_ops_adapter, org_registry_adapter = checkpointed_router
        _seed_vp(people_ops_adapter)
        _seed_org_decoy_conflict(org_registry_adapter)  # non-interference setup

        cfg = _cfg()
        result = app.invoke(
            {"messages": [HumanMessage(content="Alice was promoted to CTO at Acme since 2023-06")]},
            cfg,
        )

        assert result.get("route") == "people_ops", (
            f"Expected route='people_ops' (message names Alice), got {result.get('route')!r}"
        )
        assert "__interrupt__" in result, (
            "Router-level invoke() result must surface the dispatched subgraph's "
            f"interrupt; got keys={list(result.keys())}"
        )
        interrupts = result["__interrupt__"]
        assert len(interrupts) >= 1
        payload = interrupts[0].value
        assert "subject" in payload or "question" in payload or "incumbent" in payload, (
            f"Interrupt payload must describe the conflict, got {payload!r}"
        )

    def test_resume_completes_inside_subgraph_and_lands_in_correct_queue(
        self, checkpointed_router
    ):
        """(b) Command(resume='Affirm') on the SAME thread_id resumes INSIDE the
        interrupted people_ops subgraph and completes (no dangling interrupt).
        (c) The verdict lands in the CORRECT agent's oracle queue: people-ops's
        pending queue is empty afterward, AND org-registry's independently
        seeded decoy pending item is untouched — proving no cross-agent
        interference despite sharing one router-level checkpointer."""
        app, people_ops_adapter, org_registry_adapter = checkpointed_router
        _seed_vp(people_ops_adapter)
        _seed_org_decoy_conflict(org_registry_adapter)

        decoy_before = org_registry_adapter.list_pending_adjudications(ORG_REGISTRY_SPEC.agent_id)
        assert len(decoy_before) == 1, "Decoy conflict must be queued on org-registry before test"

        cfg = _cfg()
        result1 = app.invoke(
            {"messages": [HumanMessage(content="Alice was promoted to CTO at Acme since 2023-06")]},
            cfg,
        )
        assert "__interrupt__" in result1, "Graph must pause before resume test"

        result2 = app.invoke(Command(resume="Affirm"), cfg)
        assert "__interrupt__" not in result2, (
            "Graph must fully complete (resume inside the subgraph) after "
            f"Command(resume=...); got __interrupt__={result2.get('__interrupt__')!r}"
        )

        # (c) Correct-agent verdict landing: people-ops resolved + empty queue...
        people_ops_pending = people_ops_adapter.list_pending_adjudications(PEOPLE_OPS_SPEC.agent_id)
        assert people_ops_pending == [], (
            f"people-ops oracle queue must be empty after resolution, got {people_ops_pending!r}"
        )
        belief = people_ops_adapter.recall(PEOPLE_OPS_SPEC.agent_id, "alice-chen", "employer")
        assert belief.status == "Resolved" and belief.value == "Acme Corp / CTO", (
            f"CTO (challenger) must win after Affirm, got status={belief.status!r} value={belief.value!r}"
        )

        # ...and org-registry's independent decoy queue is UNTOUCHED.
        org_pending_after = org_registry_adapter.list_pending_adjudications(ORG_REGISTRY_SPEC.agent_id)
        assert len(org_pending_after) == 1, (
            "org-registry's independently seeded decoy pending item must be "
            f"untouched by the people-ops resolution, got {len(org_pending_after)} remaining"
        )
