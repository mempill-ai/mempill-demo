"""
mempill_showcase.frameworks.langgraph.agents_config — dual-agent AgentSpec literals.

Per ARCHITECTURE.md §1: splits the existing jordan-park executive-assistant
scenario along its two natural fact domains — PEOPLE facts (alice-chen,
bob-liu, jordan-park) vs. ORG facts (acme-corp) — instead of inventing a new
scenario. jordan-park himself becomes the implicit "principal" whose two
assistants split by domain.

Both agent_ids match `[A-Za-z0-9_-]`. These are the only two agent instances
this module defines; `exec_assistant` (the legacy single-agent graph, agent_id
"jordan-park-001") is untouched and lives on in graph.py/studio_graph.py per
ARCHITECTURE.md §7.
"""
from __future__ import annotations

from typing import NamedTuple


class AgentSpec(NamedTuple):
    """Per-instance configuration for one dual-agent-router subgraph agent."""
    agent_id: str
    responsibility: str
    known_entities: str


PEOPLE_OPS_SPEC = AgentSpec(
    agent_id="people-ops-001",
    responsibility=(
        "You manage personnel, colleague, and travel/logistics facts for "
        "people the principal (Jordan Park) works with."
    ),
    known_entities="alice-chen, bob-liu, jordan-park",
)

ORG_REGISTRY_SPEC = AgentSpec(
    agent_id="org-registry-001",
    responsibility=(
        "You manage organisational facts — corporate roles, leadership "
        "seats, and company-level records — for organisations the principal "
        "deals with."
    ),
    known_entities="acme-corp",
)
