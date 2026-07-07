"""
tests.test_dual_agent_isolation — dual-agent router DB isolation tests
(TASK-31 T31-4, Wave 3).

Writes one NEW people fact and one NEW org fact through the ROUTER graph
(real classifier + real writes — requires ANTHROPIC_API_KEY, @pytest.mark.live),
then asserts DETERMINISTICALLY (no LLM involved) that each agent's SQLite
file contains ONLY its own domain's rows: zero cross-contamination.

The write step is live (@pytest.mark.live); the row-set assertions that
follow are pure sqlite3 file inspection and would fail loudly on any
cross-domain leakage regardless of LLM behavior.
"""
from __future__ import annotations

import os
import sqlite3

import pytest
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

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


def _distinct_agent_ids(db_path) -> set[str]:
    with sqlite3.connect(str(db_path)) as conn:
        rows = conn.execute("select distinct agent_id from claims").fetchall()
    return {row[0] for row in rows}


def _all_subjects(db_path) -> set[str]:
    with sqlite3.connect(str(db_path)) as conn:
        rows = conn.execute("select distinct subject from claims").fetchall()
    return {row[0] for row in rows}


def _rows(db_path) -> list[tuple]:
    with sqlite3.connect(str(db_path)) as conn:
        return conn.execute(
            "select subject, predicate, value, agent_id from claims order by subject, predicate"
        ).fetchall()


@pytest.mark.live
def test_dual_agent_write_isolation_no_cross_contamination(tmp_path):
    """Write a people fact and an org fact via the router; assert exact DB separation."""
    db_dir = tmp_path / ".mempill"

    people_ops_adapter, people_ops_subgraph = build_agent_instance(
        agent_id=PEOPLE_OPS_SPEC.agent_id,
        responsibility=PEOPLE_OPS_SPEC.responsibility,
        known_entities=PEOPLE_OPS_SPEC.known_entities,
        db_dir=str(db_dir),
    )
    org_registry_adapter, org_registry_subgraph = build_agent_instance(
        agent_id=ORG_REGISTRY_SPEC.agent_id,
        responsibility=ORG_REGISTRY_SPEC.responsibility,
        known_entities=ORG_REGISTRY_SPEC.known_entities,
        db_dir=str(db_dir),
    )

    load_seed_claims(
        people_ops_adapter, agent_id=PEOPLE_OPS_SPEC.agent_id, claims=PEOPLE_OPS_SEED_CLAIMS,
    )
    load_seed_claims(
        org_registry_adapter, agent_id=ORG_REGISTRY_SPEC.agent_id, claims=ORG_REGISTRY_SEED_CLAIMS,
    )

    router_app = build_router_graph(people_ops_subgraph, org_registry_subgraph)

    people_db = db_dir / f"agent_{PEOPLE_OPS_SPEC.agent_id}.db"
    org_db = db_dir / f"agent_{ORG_REGISTRY_SPEC.agent_id}.db"
    assert people_db.exists(), "people-ops DB file was not created after seeding"
    assert org_db.exists(), "org-registry DB file was not created after seeding"

    # ── Live write step: one new people fact, one new org fact, via the router ──
    router_app.invoke(
        {"messages": [HumanMessage(
            content="Remember that Bob Liu is vegetarian, effective 2025."
        )]},
        config={"configurable": {"thread_id": "isolation-write-people"}},
    )
    router_app.invoke(
        {"messages": [HumanMessage(
            content="Remember that Acme's fiscal year ends in January, effective 2025."
        )]},
        config={"configurable": {"thread_id": "isolation-write-org"}},
    )

    # ── Deterministic assertions: raw sqlite3 file inspection, no LLM involved ──

    people_agent_ids = _distinct_agent_ids(people_db)
    org_agent_ids = _distinct_agent_ids(org_db)

    assert people_agent_ids == {PEOPLE_OPS_SPEC.agent_id}, (
        f"people DB must contain ONLY agent_id={PEOPLE_OPS_SPEC.agent_id!r}, "
        f"found {people_agent_ids!r}"
    )
    assert org_agent_ids == {ORG_REGISTRY_SPEC.agent_id}, (
        f"org DB must contain ONLY agent_id={ORG_REGISTRY_SPEC.agent_id!r}, "
        f"found {org_agent_ids!r}"
    )

    people_subjects = _all_subjects(people_db)
    org_subjects = _all_subjects(org_db)

    # people DB: seeded subjects + the new bob-liu row; ZERO acme-corp rows.
    assert "acme-corp" not in people_subjects, (
        f"CROSS-CONTAMINATION: people DB contains an acme-corp row. "
        f"Subjects found: {people_subjects!r}"
    )
    assert "bob-liu" in people_subjects, (
        f"Expected the new bob-liu vegetarian fact in the people DB. "
        f"Subjects found: {people_subjects!r}"
    )
    assert people_subjects == {"alice-chen", "bob-liu", "jordan-park"}, (
        f"people DB must contain exactly the 3 known people-ops subjects, "
        f"got {people_subjects!r}"
    )

    # org DB: seeded acme-corp facts + the new fiscal_year fact; zero person-subject rows.
    assert org_subjects == {"acme-corp"}, (
        f"org DB must contain ONLY acme-corp rows (zero person-subject rows), "
        f"got {org_subjects!r}"
    )
    person_subjects_in_org_db = org_subjects & {"alice-chen", "bob-liu", "jordan-park"}
    assert not person_subjects_in_org_db, (
        f"CROSS-CONTAMINATION: org DB contains person-subject rows: "
        f"{person_subjects_in_org_db!r}"
    )

    # Predicate spelling is model-chosen free text (not a fixed enum) — accept
    # either underscore or hyphen form (e.g. "dietary_restriction" vs
    # "dietary-restriction"); the assertion is on domain scoping, not spelling.
    people_predicates = {row[1] for row in _rows(people_db) if row[0] == "bob-liu"}
    assert any("dietary" in p for p in people_predicates), (
        f"Expected a dietary-related predicate to be written to the people DB, "
        f"predicates found for bob-liu: {people_predicates!r}"
    )

    org_predicates = {row[1] for row in _rows(org_db) if row[0] == "acme-corp"}
    assert any("fiscal" in p for p in org_predicates), (
        f"Expected a fiscal-year-related predicate written to the org DB, "
        f"predicates found for acme-corp: {org_predicates!r}"
    )

    # Row-set separation is exact: no row in either file references the other's agent_id.
    assert people_agent_ids.isdisjoint(org_agent_ids), (
        "people DB and org DB must never share an agent_id value"
    )
