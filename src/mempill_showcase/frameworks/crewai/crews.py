"""
mempill_showcase.frameworks.crewai.crews — Crew definitions + factory (W4).

Builds three CrewAI Crew objects corresponding to the three LangGraph crew nodes:

  Crew A — Intake & Memory
    Tasks: canonicalize_task (runs first) → intake_task
    Called when supervisor intent = UPDATE_CONTACT / CORRECT_FACT

  Crew B — Research & Intelligence
    Tasks: research_task → synthesis_task
    Called when supervisor intent = RESEARCH_CONTACT / RESEARCH_COMPANY

  Crew C — Scheduling & Briefing  (READ-ONLY)
    Tasks: fact_retrieval_task → briefing_task
    Called when supervisor intent = PREPARE_BRIEFING / RECALL_HISTORY / COMPLIANCE_AUDIT

Factory:
  build_crews(adapter, tools, llm=None) -> ShowcaseCrews
    - adapter: MempillAdapter (used to construct W2 tools if not pre-built)
    - tools:   ShowcaseTools NamedTuple (pre-built W2 LangChain tool instances)
    - llm:     optional LiteLLM-compatible model string or crewai LLM instance.
               When None, CrewAI uses env-configured OPENAI_API_KEY / ANTHROPIC_API_KEY.
               Pass a fake/configured model in tests to control the LLM seam.

ShowcaseCrews is a NamedTuple with .crew_a, .crew_b, .crew_c so callers can
inject the appropriate crew into make_crew_X_node(crew=...).

Tool bridging:
  W2 tools (LangChain BaseTool) → as_crewai_tool() → CrewStructuredTool
  The bridge is transparent: CrewAI calls the same underlying _run() method.
"""
from __future__ import annotations

import logging
from typing import Any, NamedTuple, Optional

log = logging.getLogger(__name__)


class ShowcaseCrews(NamedTuple):
    """Bundle of the three CrewAI Crew objects for injection into LangGraph nodes."""
    crew_a: Any  # Crew — Intake & Memory
    crew_b: Any  # Crew — Research & Intelligence
    crew_c: Any  # Crew — Scheduling & Briefing (read-only)


def build_crews(
    adapter: Any,
    tools: Any,
    llm: Optional[Any] = None,
) -> ShowcaseCrews:
    """Build all three CrewAI crews wired to W2 tools.

    Args:
        adapter: MempillAdapter (the mempill boundary; used in tool construction).
        tools:   ShowcaseTools NamedTuple from config/di.py (all W2 tool instances).
        llm:     Optional LiteLLM model string (e.g. "anthropic/claude-3-5-sonnet-20241022")
                 or a crewai.LLM instance. When None, CrewAI uses env API keys.

    Returns:
        ShowcaseCrews(crew_a, crew_b, crew_c)

    Note: Construction does NOT call any LLM. kickoff() is the first LLM call.
    """
    from mempill_showcase.frameworks.crewai.tool_bridge import as_crewai_tool
    from mempill_showcase.frameworks.crewai.agents import (
        build_crew_a_agents,
        build_crew_b_agents,
        build_crew_c_agents,
    )

    # Bridge W2 LangChain tools → CrewAI tool wrappers
    remember_crew = as_crewai_tool(tools.remember_tool)
    recall_crew = as_crewai_tool(tools.recall_tool)
    audit_crew = as_crewai_tool(tools.audit_tool)
    date_parser_crew = as_crewai_tool(tools.date_parser)
    rag_write_crew = as_crewai_tool(tools.rag_write_tool)
    # Calendar and email stubs — used by Crew C briefing agent
    from mempill_showcase.tools.stub_tools import CalendarTool, EmailDraftTool
    calendar_crew = as_crewai_tool(CalendarTool())
    email_crew = as_crewai_tool(EmailDraftTool())

    log.debug("build_crews: W2 tools bridged to CrewAI")

    # Build agents
    intake_agent, canon_agent = build_crew_a_agents(
        remember_tool=remember_crew,
        recall_tool=recall_crew,
        date_parser_tool=date_parser_crew,
        llm=llm,
    )
    research_agent, synthesis_agent = build_crew_b_agents(
        rag_write_tool=rag_write_crew,
        remember_tool=remember_crew,
        recall_tool=recall_crew,
        llm=llm,
    )
    fact_retrieval_agent, briefing_agent = build_crew_c_agents(
        recall_tool=recall_crew,
        calendar_tool=calendar_crew,
        email_draft_tool=email_crew,
        audit_tool=audit_crew,
        llm=llm,
    )

    # Build Crews with Tasks
    crew_a = _build_crew_a(intake_agent, canon_agent, llm=llm)
    crew_b = _build_crew_b(research_agent, synthesis_agent, llm=llm)
    crew_c = _build_crew_c(fact_retrieval_agent, briefing_agent, llm=llm)

    log.info("build_crews: Crew A/B/C constructed (no LLM call yet)")
    return ShowcaseCrews(crew_a=crew_a, crew_b=crew_b, crew_c=crew_c)


# ── Crew A builder ────────────────────────────────────────────────────────────

def _build_crew_a(intake_agent: Any, canon_agent: Any, llm: Optional[Any] = None) -> Any:
    """Assemble Crew A: Intake & Memory.

    Process order (sequential):
      1. canonicalize_task  — resolve entity/predicate to canonical keys
      2. intake_task        — extract claims, determine valid_from, write to mempill

    This order enforces AC-7: canonical key resolution always precedes any mempill write.
    """
    from crewai import Crew, Task

    canonicalize_task = Task(
        description=(
            "Given the user request: '{user_request}'\n"
            "Use the entity_normalizer tool to resolve all named entities to canonical keys.\n"
            "Return: the canonical subject key, the canonical predicate key, "
            "and a confidence that both are valid."
        ),
        expected_output=(
            "A JSON object with: canonical_entity (string), canonical_predicate (string), "
            "entity_resolved (bool), predicate_resolved (bool)."
        ),
        agent=canon_agent,
    )

    intake_task = Task(
        description=(
            "Given the user request: '{user_request}'\n"
            "Using the canonical keys from the Canonicalization Agent:\n"
            "1. Parse the valid_from date using date_parser.\n"
            "2. Recall any existing belief via mempill_recall to check for succession.\n"
            "3. Write the claim via mempill_remember with provenance_channel=UserAsserted.\n"
            "4. Report the claim_ref, disposition, and whether the result is_contested."
        ),
        expected_output=(
            "A JSON object with: claim_ref, disposition (CommittedCheap or Contested), "
            "is_contested (bool), subject, predicate, value, valid_from."
        ),
        agent=intake_agent,
        context=[canonicalize_task],
    )

    crew = Crew(
        agents=[canon_agent, intake_agent],
        tasks=[canonicalize_task, intake_task],
        verbose=False,
    )

    log.debug("_build_crew_a: Crew A assembled with 2 agents, 2 tasks")
    return crew


# ── Crew B builder ────────────────────────────────────────────────────────────

def _build_crew_b(research_agent: Any, synthesis_agent: Any, llm: Optional[Any] = None) -> Any:
    """Assemble Crew B: Research & Intelligence.

    Process order (sequential):
      1. research_task    — gather external facts and write bulk content to RAG
      2. synthesis_task   — distil ≤3 atomic claims to mempill (ExternalFirstHand)

    AC-6 invariant: synthesis_agent is explicitly instructed to write ≤3 claims.
    """
    from crewai import Crew, Task

    research_task = Task(
        description=(
            "Research request: '{user_request}'\n"
            "Search for relevant information about the named contacts and companies.\n"
            "Write all raw findings (full article text, quotes, source URLs) to the RAG store "
            "using rag_write. Do NOT write anything to mempill at this stage.\n"
            "Return a structured list of discovered facts with source references."
        ),
        expected_output=(
            "A structured list of discovered facts. Each entry has: "
            "entity, attribute, value, estimated_date, source, confidence (0-1). "
            "All raw content must be stored in RAG. No mempill writes in this task."
        ),
        agent=research_agent,
    )

    synthesis_task = Task(
        description=(
            "Distil the research findings into atomic mempill claims.\n"
            "Rules:\n"
            "  - Write at most 3 claims to mempill per run (AC-6 constraint).\n"
            "  - Each claim: subject (canonical key), predicate (canonical key), value, "
            "    valid_from, confidence, provenance_channel=ExternalFirstHand.\n"
            "  - Cross-check existing beliefs via mempill_recall before writing.\n"
            "  - If is_contested=true: report it immediately — do NOT resolve automatically.\n"
            "  - No prose, no summaries, no article text in mempill claims.\n"
            "Research findings to distil: {user_request}"
        ),
        expected_output=(
            "A JSON list of written claims (max 3). Each entry has: "
            "claim_ref, subject, predicate, value, valid_from, disposition, is_contested. "
            "If any claim is Contested, include contested_with references."
        ),
        agent=synthesis_agent,
        context=[research_task],
    )

    crew = Crew(
        agents=[research_agent, synthesis_agent],
        tasks=[research_task, synthesis_task],
        verbose=False,
    )

    log.debug("_build_crew_b: Crew B assembled with 2 agents, 2 tasks")
    return crew


# ── Crew C builder ────────────────────────────────────────────────────────────

def _build_crew_c(
    fact_retrieval_agent: Any, briefing_agent: Any, llm: Optional[Any] = None
) -> Any:
    """Assemble Crew C: Scheduling & Briefing (READ-ONLY).

    Process order (sequential):
      1. fact_retrieval_task — recall all relevant facts; flag Contested
      2. briefing_task       — compose output using only non-Contested facts

    SAFETY INVARIANT: Neither agent has a remember/write tool. This is enforced
    at build time by build_crew_c_agents which receives no remember_tool.
    """
    from crewai import Crew, Task

    fact_retrieval_task = Task(
        description=(
            "Action request: '{user_request}'\n"
            "Recall all relevant facts about the contacts and entities mentioned.\n"
            "For each fact:\n"
            "  - Use mempill_recall for current facts (valid_at=now).\n"
            "  - Use mempill_recall with valid_at for point-in-time queries.\n"
            "  - Use mempill_recall with as_of_tx_time for transaction-time replay.\n"
            "  - For COMPLIANCE_AUDIT: use mempill_audit.\n"
            "Flag any fact where is_contested=true — the Briefing Agent must NOT use it."
        ),
        expected_output=(
            "A structured fact report. For each recalled fact: "
            "subject, predicate, value, status (Resolved/Contested/NoBelief), "
            "is_contested (bool), valid_from_display, valid_until_display. "
            "Explicitly list any Contested facts that must be omitted from output."
        ),
        agent=fact_retrieval_agent,
    )

    briefing_task = Task(
        description=(
            "Compose the action output for: '{user_request}'\n"
            "Using ONLY the non-Contested facts from the Fact Retrieval Agent:\n"
            "  - For PREPARE_BRIEFING: write a concise briefing paragraph.\n"
            "  - For SCHEDULE_MEETING: create a calendar event via calendar_create_event.\n"
            "  - For DRAFT_COMMUNICATION: draft an email via email_draft.\n"
            "  - For any CONTESTED fact needed in the output: write "
            "    '[attribute] is currently unresolved — omitted until Jordan confirms.'\n"
            "Never fabricate facts not retrieved from mempill."
        ),
        expected_output=(
            "The final action output (briefing text, calendar event JSON, or email draft JSON). "
            "If any required fact was Contested, include an explicit omission note."
        ),
        agent=briefing_agent,
        context=[fact_retrieval_task],
    )

    crew = Crew(
        agents=[fact_retrieval_agent, briefing_agent],
        tasks=[fact_retrieval_task, briefing_task],
        verbose=False,
    )

    log.debug("_build_crew_c: Crew C assembled with 2 agents, 2 tasks (read-only)")
    return crew
