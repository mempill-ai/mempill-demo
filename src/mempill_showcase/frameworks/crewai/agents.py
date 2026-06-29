"""
mempill_showcase.frameworks.crewai.agents — CrewAI Agent definitions (W4).

Three crews, six agents total (two per crew):

Crew A — Intake & Memory
  intake_agent         — parse user utterance → extract atomic claims → write to mempill
  canonicalization_agent — normalise entity names to canonical keys before any write

Crew B — Research & Intelligence
  research_agent       — search external sources for contact/company facts
  synthesis_agent      — distil raw research into ≤3 atomic claims for mempill

Crew C — Scheduling & Briefing (READ-ONLY)
  fact_retrieval_agent — recall all relevant facts from mempill; flag Contested
  briefing_agent       — compose human-readable output using only non-Contested facts

Tool assignment per SCENARIO.md:
  Crew A: remember + recall + date_parser + entity_normalizer (canonical lookup)
  Crew B: rag_write + remember (distilled only) + recall (cross-check)
  Crew C: recall + (optional) audit; calendar + email stubs — NO remember/write

Safety rule (AC-7): Canonicalization Agent always runs first within Crew A.
Crew C invariant:    NO remember-capable tool is passed to any Crew C agent.

The `llm` param is injectable so tests can pass a fake/configured model or
leave it as None to use environment config (OPENAI_API_KEY / ANTHROPIC_API_KEY).

CrewAI uses LiteLLM under the hood; if no LLM is configured and no API key is
present, agents will raise at kickoff time (not at construction time) — which
means CONSTRUCTION tests can run without any API key.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional

log = logging.getLogger(__name__)


# ── Crew A agents ─────────────────────────────────────────────────────────────

def build_crew_a_agents(
    remember_tool: Any,
    recall_tool: Any,
    date_parser_tool: Any,
    llm: Optional[Any] = None,
) -> tuple[Any, Any]:
    """Build (intake_agent, canonicalization_agent) for Crew A.

    Tools received are already CrewAI-bridged (as_crewai_tool wrappers or
    native CrewAI tools).

    Returns:
        (intake_agent, canonicalization_agent)
    """
    from crewai import Agent

    # Entity normalizer is a lightweight inline function tool wrapping canonical_keys
    entity_normalizer = _build_entity_normalizer_tool()

    canonicalization_agent = Agent(
        role="Canonicalization Specialist",
        goal=(
            "Resolve any entity name in the user request to its deterministic canonical key "
            "(e.g. 'Alice Chen' → 'alice-chen', 'Acme' → 'acme-corp') using the project-maintained "
            "lookup table. Never invent a key. Return the canonical subject and predicate so the "
            "Intake Agent can write with correct keys."
        ),
        backstory=(
            "You are a data steward who ensures entity identifiers are always stable and consistent. "
            "You have memorised the canonical key table (alice-chen, bob-liu, acme-corp, jordan-park). "
            "You never guess or hallucinate a key: if an entity is unknown, you report it as unknown. "
            "Correct canonical keys are the foundation of the entire memory system."
        ),
        tools=[entity_normalizer],
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    intake_agent = Agent(
        role="Intake & Memory Specialist",
        goal=(
            "Extract atomic fact claims from the user's utterance. For each claim determine "
            "the canonical entity key (from the Canonicalization Agent), the canonical predicate, "
            "the value, the valid_from date, and the provenance channel (UserAsserted). "
            "Write each claim to mempill via mempill_remember. "
            "If the write returns is_contested=true, surface it immediately — do NOT resolve it."
        ),
        backstory=(
            "You are a meticulous data entry specialist for an AI executive assistant. "
            "You extract only confirmed atomic facts from natural language ('Alice relocated to NYC') "
            "and record them with precise temporal metadata. "
            "You never write prose or summaries to mempill — only subject/predicate/value/valid_from. "
            "You always check existing beliefs before writing (recall) to avoid redundant writes."
        ),
        tools=[remember_tool, recall_tool, date_parser_tool],
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    log.debug("build_crew_a_agents: intake + canonicalization agents built")
    return intake_agent, canonicalization_agent


# ── Crew B agents ─────────────────────────────────────────────────────────────

def build_crew_b_agents(
    rag_write_tool: Any,
    remember_tool: Any,
    recall_tool: Any,
    llm: Optional[Any] = None,
) -> tuple[Any, Any]:
    """Build (research_agent, synthesis_agent) for Crew B.

    Crew B distillation rule (AC-6):
      - research_agent writes ONLY bulk context to RAG (never to mempill)
      - synthesis_agent writes ≤3 atomic claims to mempill per run
      - No prose, summaries, or article text ever reaches mempill

    Returns:
        (research_agent, synthesis_agent)
    """
    from crewai import Agent

    research_agent = Agent(
        role="Research & Intelligence Analyst",
        goal=(
            "Find external information about contacts and companies from available sources. "
            "Store ALL raw research content (full articles, quotes, context) in the RAG store "
            "using rag_write. Never write raw content directly to mempill. "
            "Produce a structured research report listing discovered facts with their sources and dates."
        ),
        backstory=(
            "You are an intelligence analyst who gathers information about business contacts and companies. "
            "You save all raw content to the RAG working context for later synthesis. "
            "You are disciplined: you never short-cut to mempill — all raw findings go to RAG first. "
            "Your output is a structured list of discovered facts, each with a source reference."
        ),
        tools=[rag_write_tool],
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    synthesis_agent = Agent(
        role="Research Synthesis Specialist",
        goal=(
            "Distil research findings into atomic mempill claims. "
            "Write at most 3 claims per research run (AC-6). "
            "Each claim must be: subject (canonical key), predicate (canonical key), value, "
            "valid_from date, confidence, and provenance=ExternalFirstHand. "
            "Cross-check existing mempill beliefs using mempill_recall before writing. "
            "If a write returns is_contested=true, surface it immediately for HITL resolution."
        ),
        backstory=(
            "You are a specialist in converting raw intelligence into precise, machine-readable memory entries. "
            "You follow the distillation rule strictly: one claim per fact, no prose, no summaries. "
            "You are conservative: when uncertain about a date, you use lower confidence and coarser granularity. "
            "You always use canonical entity and predicate keys — never free-form strings."
        ),
        tools=[remember_tool, recall_tool],
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    log.debug("build_crew_b_agents: research + synthesis agents built")
    return research_agent, synthesis_agent


# ── Crew C agents ─────────────────────────────────────────────────────────────

def build_crew_c_agents(
    recall_tool: Any,
    calendar_tool: Any,
    email_draft_tool: Any,
    audit_tool: Optional[Any] = None,
    llm: Optional[Any] = None,
) -> tuple[Any, Any]:
    """Build (fact_retrieval_agent, briefing_agent) for Crew C.

    SAFETY INVARIANT: NO remember/write-capable tool is ever passed to Crew C agents.
    Crew C is strictly read-only — it recalls facts and produces action output.

    The audit_tool is optional; pass it when COMPLIANCE_AUDIT intent is active.

    Returns:
        (fact_retrieval_agent, briefing_agent)
    """
    from crewai import Agent

    # Build tool list for fact retrieval agent — recall always present, audit optional
    fact_retrieval_tools: List[Any] = [recall_tool]
    if audit_tool is not None:
        fact_retrieval_tools.append(audit_tool)

    fact_retrieval_agent = Agent(
        role="Fact Retrieval Specialist",
        goal=(
            "Recall all relevant facts about named contacts and companies from mempill. "
            "Use mempill_recall with valid_at for historical queries and as_of_tx_time for "
            "transaction-time replays. Flag any fact that returns is_contested=true — "
            "the Briefing Agent MUST NOT use Contested facts in any output. "
            "For COMPLIANCE_AUDIT intent: use mempill_audit to retrieve the full audit ledger."
        ),
        backstory=(
            "You are a precise data retrieval specialist who knows how to query bi-temporal memory systems. "
            "You understand that 'current' and 'as of a past date' are different queries. "
            "You always report the full status (Resolved / Contested / NoBelief) of every recalled fact. "
            "You never modify or write data — you only read."
        ),
        tools=fact_retrieval_tools,
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    briefing_agent = Agent(
        role="Briefing & Scheduling Coordinator",
        goal=(
            "Compose human-readable briefings, calendar events, and email drafts using only "
            "confirmed, non-Contested facts provided by the Fact Retrieval Agent. "
            "If any required fact is Contested or missing, explicitly flag it in the output "
            "('Alice\\'s [attribute] is unresolved — omitted from briefing'). "
            "Never use a Contested fact in any action output."
        ),
        backstory=(
            "You are Jordan Park's executive assistant coordinator. "
            "You produce polished briefings and scheduling outputs for high-stakes meetings. "
            "You are disciplined about data quality: if a fact is unresolved, you flag it rather "
            "than guess. You use CalendarTool for event creation and EmailDraftTool for communications."
        ),
        tools=[calendar_tool, email_draft_tool],
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    log.debug("build_crew_c_agents: fact_retrieval + briefing agents built")
    return fact_retrieval_agent, briefing_agent


# ── Entity Normalizer Tool ────────────────────────────────────────────────────

def _build_entity_normalizer_tool() -> Any:
    """Build a CrewAI tool that resolves entity/predicate names to canonical keys.

    This is a pure Python function tool — no LLM, no external call.
    Wraps canonical_keys.resolve_entity + resolve_predicate.
    Returns a crewai.tools.base_tool.BaseTool subclass instance.
    """
    from crewai.tools.base_tool import BaseTool as CrewBaseTool
    from pydantic import BaseModel, Field

    class EntityNormalizerInput(BaseModel):
        entity_name: str = Field(
            description=(
                "The entity name to normalise (e.g. 'Alice Chen', 'alice', 'Acme Corp'). "
                "Returns the canonical key (e.g. 'alice-chen') or None if unknown."
            )
        )
        predicate_name: str = Field(
            default="",
            description=(
                "Optional predicate name to normalise (e.g. 'city', 'employer', 'role'). "
                "Returns the canonical predicate key or None if unknown."
            ),
        )

    _INPUT_SCHEMA = EntityNormalizerInput

    class EntityNormalizerTool(CrewBaseTool):
        name: str = "entity_normalizer"
        description: str = (
            "Resolve an entity name or predicate name to its deterministic canonical key. "
            "Supply entity_name (e.g. 'Alice Chen') and optionally predicate_name (e.g. 'city'). "
            "Returns JSON with canonical_entity and canonical_predicate. "
            "Always call this before any mempill write to ensure canonical key compliance (AC-7)."
        )
        args_schema: type = _INPUT_SCHEMA

        def _run(self, entity_name: str, predicate_name: str = "") -> str:
            import json
            from mempill_showcase.core.domain.canonical_keys import (
                resolve_entity,
                resolve_predicate,
            )
            canonical_entity = resolve_entity(entity_name)
            canonical_predicate = resolve_predicate(predicate_name) if predicate_name else None
            result = {
                "entity_name": entity_name,
                "canonical_entity": canonical_entity,
                "predicate_name": predicate_name or None,
                "canonical_predicate": canonical_predicate,
                "entity_resolved": canonical_entity is not None,
                "predicate_resolved": canonical_predicate is not None if predicate_name else None,
            }
            return json.dumps(result)

    return EntityNormalizerTool()
