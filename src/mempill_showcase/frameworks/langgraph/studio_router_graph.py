"""
mempill_showcase.frameworks.langgraph.studio_router_graph — dual-agent router
Studio entry point (TASK-31, ARCHITECTURE.md §4).

Exposes three module-level compiled graphs for `langgraph.json`/`langgraph dev`:
  people_ops_graph   — the people-ops-001 ReAct subgraph, addressable directly.
  org_registry_graph — the org-registry-001 ReAct subgraph, addressable directly.
  graph              — the top-level dual-agent router (route_query -> dispatch).

Design (mirrors studio_graph.py's pattern, extended for two agents):
  1. bootstrap() to load .env.
  2. Resolve db_dir = settings.mempill_db_dir or "./.mempill" — the approved
     Studio default (ARCHITECTURE.md §4): file-backed (NOT in-memory) so the
     per-agent SQLite files are visible/inspectable on disk.
  3. Build both agent instances via di.build_agent_instance(...) — each opens
     its own MempillAdapter (via mempill.open_oracle_for_agent) and compiles
     its own ReAct subgraph with checkpointer=None.
  4. Seed each adapter with ITS OWN domain claims (PEOPLE_OPS_SEED_CLAIMS /
     ORG_REGISTRY_SEED_CLAIMS) via load_seed_claims() — idempotent, so
     re-running `langgraph dev` / re-importing this module does not duplicate
     seed rows (guarded by is_already_seeded()'s claims-derived sentinel).
  5. Build the router via build_router_graph(people_ops_subgraph,
     org_registry_subgraph, checkpointer=None) — Studio injects its own
     checkpointer; do NOT pass an explicit one (SPIKE_RESULTS.md).

Usage (LangGraph Studio):
  1. Set ANTHROPIC_API_KEY in .env. Optionally set MEMPILL_DB_DIR to override
     the "./.mempill" default.
  2. .venv/bin/langgraph dev
  3. Select "dual_agent_router" to exercise the classifier + dispatch, or
     "people_ops_agent" / "org_registry_agent" to talk to one subgraph
     directly (its agent_id defaults per ARCHITECTURE.md §2's per-instance
     system prompt).
  4. Inspect persisted state on disk:
       ls -la .mempill/
       sqlite3 .mempill/agent_people-ops-001.db "select subject, predicate, value from claims;"
       sqlite3 .mempill/agent_org-registry-001.db "select subject, predicate, value from claims;"
"""
from __future__ import annotations

import logging

from mempill_showcase.config.bootstrap import bootstrap
from mempill_showcase.config.di import build_agent_instance
from mempill_showcase.frameworks.langgraph.agents_config import (
    ORG_REGISTRY_SPEC,
    PEOPLE_OPS_SPEC,
)
from mempill_showcase.frameworks.langgraph.router_graph import build_router_graph
from mempill_showcase.scenarios.seed_data import (
    ORG_REGISTRY_SEED_CLAIMS,
    PEOPLE_OPS_SEED_CLAIMS,
    load_seed_claims,
)

log = logging.getLogger(__name__)


def _resolve_db_dir() -> str:
    """Resolve the Studio db_dir: settings.mempill_db_dir or "./.mempill" default.

    Per ARCHITECTURE.md §4: Studio must use a file-backed engine (not in-memory)
    so the per-agent DB files are visible/inspectable on disk.
    """
    try:
        from mempill_showcase.config.settings import get_settings
        configured = get_settings().mempill_db_dir
        return configured or "./.mempill"
    except Exception:
        return "./.mempill"


def _build_studio_router():
    """Build both agent instances + the router graph for LangGraph Studio.

    Returns (router_graph, people_ops_graph, org_registry_graph,
    people_ops_adapter, org_registry_adapter).
    """
    bootstrap()

    from mempill_showcase.config.settings import get_settings
    _settings = get_settings()

    db_dir = _resolve_db_dir()

    people_ops_adapter, people_ops_subgraph = build_agent_instance(
        agent_id=PEOPLE_OPS_SPEC.agent_id,
        responsibility=PEOPLE_OPS_SPEC.responsibility,
        known_entities=PEOPLE_OPS_SPEC.known_entities,
        db_dir=db_dir,
        model_name=_settings.anthropic_model,
        oracle_backed=True,
    )
    org_registry_adapter, org_registry_subgraph = build_agent_instance(
        agent_id=ORG_REGISTRY_SPEC.agent_id,
        responsibility=ORG_REGISTRY_SPEC.responsibility,
        known_entities=ORG_REGISTRY_SPEC.known_entities,
        db_dir=db_dir,
        model_name=_settings.anthropic_model,
        oracle_backed=True,
    )

    for label, adapter, agent_id, claims in (
        ("people_ops", people_ops_adapter, PEOPLE_OPS_SPEC.agent_id, PEOPLE_OPS_SEED_CLAIMS),
        ("org_registry", org_registry_adapter, ORG_REGISTRY_SPEC.agent_id, ORG_REGISTRY_SEED_CLAIMS),
    ):
        try:
            refs = load_seed_claims(adapter, agent_id=agent_id, claims=claims)
            if refs:
                log.info(
                    "studio_router_graph: seeded %d Day-0 claims for agent_id=%r (%s)",
                    len(refs), agent_id, label,
                )
            else:
                log.info(
                    "studio_router_graph: store already seeded for agent_id=%r (%s) — seed skipped",
                    agent_id, label,
                )
        except Exception as exc:
            log.warning(
                "studio_router_graph: seed failed for agent_id=%r (%s): %s — "
                "starting with existing store state",
                agent_id, label, exc,
            )

    # No checkpointer — Studio injects its own persistence layer.
    router = build_router_graph(
        people_ops_subgraph,
        org_registry_subgraph,
        model_name=_settings.anthropic_model,
        checkpointer=None,
    )

    log.info(
        "studio_router_graph: compiled dual-agent router (db_dir=%r, model=%s)",
        db_dir, _settings.anthropic_model,
    )
    return router, people_ops_subgraph, org_registry_subgraph, people_ops_adapter, org_registry_adapter


# ── Module-level compiled graphs ──────────────────────────────────────────────

# Built once at import time. `langgraph dev` imports this module once and
# serves all Studio turns from the same process — mirrors studio_graph.py.
(
    graph,
    people_ops_graph,
    org_registry_graph,
    people_ops_adapter,
    org_registry_adapter,
) = _build_studio_router()
