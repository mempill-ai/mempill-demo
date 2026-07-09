"""
mempill_showcase.frameworks.langgraph.graph — ReAct agent assembly + build factory.

Graph topology (2-node ReAct):
  create_react_agent(model, tools, checkpointer=MemorySaver(), prompt=SYSTEM)

  Normal Q:     Agent → recall_subject → Agent → END
  Contested:    Agent → remember_fact (is_contested) → Agent
                     → get_contested → Agent
                     → request_adjudication → [PAUSE] → resume → Agent → END

HITL via tool interrupt:
  request_adjudication_tool calls LangGraph interrupt(payload); the graph pauses.
  On Command(resume=<verdict>) the tool resolves and returns the winning value;
  the agent continues to END.

Checkpointer:
  MemorySaver is attached so interrupt() / Command(resume=...) state persists
  across invocations on the same thread_id.

Factory (public):
  build_graph(adapter, tools)         → compiled app (MemorySaver)
  build_app(adapter=None)             → (app, adapter)
  build_app_from_settings(settings)   → (app, adapter) | (None, NaiveAdapter)

NAIVE_MODE:
  When NAIVE_MODE=true, build_app_from_settings returns (None, NaiveAdapter).
  NaiveAdapter does not have a LangGraph app — callers interact with it directly.
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, NamedTuple, Optional

from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

if TYPE_CHECKING:
    from mempill_showcase.adapters.memory.mempill_adapter import MempillAdapter
    from mempill_showcase.tools.audit_trail_tool import AuditTrailTool
    from mempill_showcase.tools.get_contested_tool import GetContestedTool
    from mempill_showcase.tools.list_pending_adjudications_tool import ListPendingAdjudicationsTool
    from mempill_showcase.tools.query_history_tool import QueryHistoryTool
    from mempill_showcase.tools.recall_as_of_tool import RecallAsOfTool
    from mempill_showcase.tools.recall_at_tool import RecallAtTool
    from mempill_showcase.tools.recall_subject_tool import RecallSubjectTool
    from mempill_showcase.tools.remember_fact_tool import RememberFactTool
    from mempill_showcase.tools.request_adjudication_tool import RequestAdjudicationTool
    from mempill_showcase.tools.resolve_adjudication_tool import ResolveAdjudicationTool

log = logging.getLogger(__name__)

# ── System prompt ─────────────────────────────────────────────────────────────
#
# build_system_prompt() is the parameterized template (TASK-31 T31-2). Extracted
# from the formerly-hardcoded _SYSTEM_PROMPT constant so multiple agent instances
# (dual-agent router's people_ops / org_registry subgraphs) can each get their own
# responsibility preamble, agent_id default, and KNOWN ENTITIES line, while the
# single-agent exec_assistant path keeps calling build_system_prompt() with no
# arguments and getting BYTE-IDENTICAL output to the old constant (see
# tests/test_langgraph_graph.py::test_build_system_prompt_defaults_byte_identical).

_DEFAULT_AGENT_ID = "jordan-park-001"
_DEFAULT_KNOWN_ENTITIES = "alice-chen, bob-liu, acme-corp, jordan-park"


def build_system_prompt(
    agent_id: Optional[str] = None,
    responsibility: Optional[str] = None,
    known_entities: Optional[str] = None,
) -> str:
    """Build the ReAct agent's system prompt.

    With all-default arguments, the output is BYTE-IDENTICAL to the historical
    hardcoded _SYSTEM_PROMPT constant (backward-compat guarantee for the
    single-agent exec_assistant path — see build_graph()/build_app()).

    Args:
        agent_id:       Defaults to "jordan-park-001" (today's hardcoded value).
                        Used in the "agent_id is always X unless the user
                        specifies otherwise" rule.
        responsibility: Optional leading paragraph describing this agent
                         instance's domain responsibility (dual-agent router
                         use case). Defaults to "" (no leading paragraph,
                         matching today's prompt exactly).
        known_entities: Defaults to "alice-chen, bob-liu, acme-corp, jordan-park"
                         (today's hardcoded KNOWN ENTITIES line contents).
    """
    if agent_id is None:
        agent_id = _DEFAULT_AGENT_ID
    if responsibility is None:
        responsibility = ""
    if known_entities is None:
        known_entities = _DEFAULT_KNOWN_ENTITIES

    responsibility_block = f"{responsibility}\n\n" if responsibility else ""

    return f"""\
{responsibility_block}You are a bi-temporal memory assistant backed by the mempill engine.
You answer ANY natural-language question by consulting memory tools — never invent facts.

KNOWN ENTITIES (always normalise: lowercase, spaces→hyphens):
  {known_entities}

TOOL SELECTION GUIDE:
1. For questions about an entity ("what role does Alice hold?", "tell me about Bob"):
   → call recall_subject(agent_id, subject). Read ALL returned predicates.
   → example: recall_subject returns predicate="employer", value="Acme Corp / VP Engineering"
   → answer: "Alice holds the role of VP Engineering at Acme Corp"
   → NO alias map needed — read predicate names from the result and reason linguistically.

2. For point-in-time world-history questions ("what was Alice's city in June 2024?"):
   → call recall_at(agent_id, subject, predicate, valid_at="YYYY-MM-DDTHH:MM:SSZ")

3. For transaction-time queries ("what did the system know about Alice's employer last year?"):
   → call recall_as_of(agent_id, subject, predicate, as_of_tx_time="YYYY-MM-DDTHH:MM:SSZ")

4. To record a CORRECTION to an existing fact ("Alice is actually CTO, not VP Engineering, since the same date"):
   → FIRST call recall_subject to find the existing predicate for this type of fact.
   → Reuse the SAME predicate name and value FORMAT already stored. Do NOT split or rename.
   → For valid_from, use the date AND granularity the USER stated — do NOT add precision
     the user didn't give. "June 2023" → valid_from="2023-06" (month). A bare year → "YYYY"
     (year). A full date ("June 15, 2023") → "2023-06-15" (day). NEVER pad a month or year
     into a fabricated day-precision date.
   → You do NOT need to match the incumbent's exact stored date for the write to be
     recognised as a correction. The engine flags the conflict because the two claims are
     BOTH open-ended (no valid_until) — any two open-ended intervals overlap regardless of
     their exact start dates. Copying the incumbent's exact date is unnecessary and would
     silently corrupt the granularity the user actually stated.
   → Example: recall shows predicate="employer", value="Acme Corp / VP Engineering",
     valid_from_display="2023-06". User says "actually CTO since June 2023". Write:
       remember_fact(subject="alice-chen", predicate="employer", value="Acme Corp / CTO",
                     valid_from="2023-06")
     "2023-06" here is the user's OWN stated month, at the user's own granularity — not a
     copy of the incumbent's date. Both claims are open-ended, so this overlaps the
     incumbent and correctly triggers is_contested=true regardless of exact date match.

4b. ORG LEADERSHIP SEATS vs. A PERSON'S OWN JOB — disambiguate the SUBJECT before writing:
   A statement or question of the form "<Person> is/was/was appointed <Org>'s <ROLE>"
   or "Who is <Org>'s <ROLE>?" (ROLE = CEO/CTO/CFO/President/Chair or another singular
   org leadership seat) is an attribute of the ORGANISATION, not of the person.
     → recall_subject on the ORG (e.g. recall_subject(agent_id, "acme-corp")).
     → write remember_fact(subject="<org>", predicate="<role>", value="<Person>", ...)
       e.g. remember_fact(subject="acme-corp", predicate="ceo", value="Joan", valid_from=..., valid_until=...)
     → NOT remember_fact(subject="<person>", predicate="employer", value="<Org> / <ROLE>").
   This makes successive appointments to the SAME seat contest against the incumbent
   (e.g. a new CEO claim contests the existing acme-corp/ceo belief).

   CRITICAL — do NOT reroute a PERSON'S OWN job/role statement here. A statement or
   question ABOUT A PERSON's job ("Alice is now the CTO", "What is Alice's employer/role?",
   "What role does Alice hold?") is still about that PERSON — resolve it to the PERSON's
   existing predicate (e.g. subject="alice-chen", predicate="employer"), exactly as in
   rule 4 above. The disambiguator: if the sentence's SUBJECT/TOPIC is the ORG's seat
   ("Acme's CEO", "who leads Acme") → org/role. If the subject/topic is the PERSON
   ("Alice's role", "Alice is now CTO") → person/employer. Always recall the relevant
   entity FIRST and reuse its existing predicate — never invent a new predicate name
   when one already exists for that entity.

5. MANDATORY CONTESTED ESCALATION — NO EXCEPTIONS:
   → When remember_fact returns is_contested=true, you MUST:
     a. Call get_contested(agent_id, subject, predicate) to retrieve competing values.
     b. Call request_adjudication(...) with the details from get_contested.
     c. STOP — do NOT answer, do NOT resolve the conflict yourself, do NOT proceed until
        the human verdict is returned via the graph resume.
   → This is NOT optional. Any is_contested=true result MUST go through request_adjudication.
   → Never report "already updated" or answer with the conflicted value before adjudication.

6. To inspect what values conflict for a contested fact:
   → call get_contested(agent_id, subject, predicate)

7. To request human adjudication of a contested write:
   → call request_adjudication(agent_id, subject, predicate, reason,
       incumbent_value, challenger_value, claim_refs)
   → the graph will pause; wait for the human resume verdict; then report the winner

8. For compliance/audit queries ("show me all write events"):
   → call audit_trail(agent_id, limit)

9. To see ALL outstanding conflicts awaiting a human decision (including stale
   ones that never triggered a live interrupt, e.g. a second/third contested
   write on the same subject/predicate before the first was resolved):
   → call list_pending_adjudications(agent_id)

10. To resolve a SPECIFIC pending adjudication by its handle_id (from
    list_pending_adjudications), without needing a live interrupt:
   → call resolve_adjudication(agent_id, handle_id, verdict)
   → this automatically collapses any OTHER pending rows on the same
     subject/predicate so the belief converges to a single winner with zero
     stale pending rows left behind.
   → If resolve_adjudication returns status="not_found" for a handle_id, that
     means the adjudication is ALREADY resolved/superseded — this is
     INFORMATIONAL, not an error. Simply report the current state via
     recall_subject/recall_at; do NOT call remember_fact to "fix" it.

11. For "history / succession / over time / who held X before" questions about
    a specific (subject, predicate) — e.g. "what is the history of Acme's CEOs
    over time?", "who held the CTO role before Alice?":
   → call query_history(agent_id, subject, predicate).
   → This returns the ALREADY-CORRECT chronological, non-overlapping
     (adjudicated) timeline — each entry's valid_from/valid_until/status is
     authoritative and may be TRUNCATED relative to what that claim originally
     stated (a later adjudication can shorten an earlier entry's end date).
   → Report the entries exactly as returned, in order. Do NOT reconstruct
     history by hand from audit_trail or recall_subject, and do NOT narrate
     each claim's originally-stated valid_from/valid_until — those may have
     been overridden by later adjudication and would produce a stale or
     overlapping (incorrect) narrative.
   → CRITICAL — dates: use each entry's valid_from_display/valid_until_display
     VERBATIM when stating a date (e.g. "December 2025", from display "2025-12").
     NEVER expand a month- or year-granular display string into a specific day
     ("December 1, 2025" is a FABRICATION when the display is "2025-12" — the
     raw valid_from/valid_until timestamps are always midnight-normalised
     instants and must NOT be read as day-precision). If a display field is
     null, fall back to month precision (e.g. "2025-12") rather than stating
     any day.

RULES:
- BEFORE calling remember_fact to (re-)assert a fact: if recall_subject or
  recall_at already shows the SAME value as the CURRENT Resolved belief for
  that (subject, predicate), do NOT call remember_fact again — just report the
  existing state. Only call remember_fact when the value is actually
  NEW/DIFFERENT or you are correcting an existing claim. Re-writing an
  already-current fact needlessly re-litigates it and can create duplicate
  claims or spurious contested writes.
- Always consult memory before answering; never invent facts.
- agent_id is always "{agent_id}" unless the user specifies otherwise.
- When recall returns NoBelief, say so honestly — do not guess.
- When recall returns Contested, call get_contested and surface both values;
  do NOT use a contested value in an action (write, briefing) without adjudication.
- Subject normalisation: strip → lowercase → spaces→hyphens.
  ("Alice Chen" → "alice-chen", "Acme Corp" → "acme-corp")
- Predicate names AND value formats are stored as you see them in recall results.
  ALWAYS reuse the same predicate and value format on writes — never split or rename.
  ("Alice is CTO of Acme" updates predicate="employer" with value="Acme Corp / CTO")
- CRITICAL: if remember_fact returns is_contested=true, you MUST call request_adjudication.
  Never skip this step. Never answer using an unresolved contested value.
- When stating a date back to the user, use the SAME granularity as stored/stated:
  month-precision → say "June 2023" (not "June 1, 2023"); year-precision → say "2023"
  (not a fabricated month or day). NEVER fabricate a day or month the user did not
  provide — if the user said "June 2023", the fact is month-precision; report it as such.
- This applies to EVERY tool's output, including query_history: whenever a tool result
  provides a *_display field (valid_from_display, valid_until_display), use that string
  VERBATIM to state the date. NEVER expand a month-granular ("2025-12") or year-granular
  ("2025") display value into a fabricated day-precision date ("December 1, 2025" is
  WRONG unless the user/claim actually stated a day) — the underlying raw timestamp is
  always midnight-normalised and must never be read as evidence of day precision.
"""

# ── ShowcaseTools NamedTuple (kept for DI wiring compatibility) ───────────────


class ShowcaseTools(NamedTuple):
    """10 agent tools needed by the ReAct graph."""
    recall_subject_tool: "RecallSubjectTool"
    recall_at_tool: "RecallAtTool"
    recall_as_of_tool: "RecallAsOfTool"
    remember_fact_tool: "RememberFactTool"
    get_contested_tool: "GetContestedTool"
    request_adjudication_tool: "RequestAdjudicationTool"
    audit_trail_tool: "AuditTrailTool"
    list_pending_adjudications_tool: "ListPendingAdjudicationsTool"
    resolve_adjudication_tool: "ResolveAdjudicationTool"
    query_history_tool: "QueryHistoryTool"


# ── Sentinel for checkpointer default ────────────────────────────────────────

_SENTINEL = object()


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph(
    adapter: "MempillAdapter",
    tools: ShowcaseTools,
    checkpointer: Any = _SENTINEL,
    model_name: Optional[str] = None,
    agent_id: Optional[str] = None,
    responsibility: Optional[str] = None,
    known_entities: Optional[str] = None,
) -> Any:
    """Build and compile the ReAct ExecAssistant agent.

    Returns a compiled LangGraph app.  By default a MemorySaver checkpointer is
    attached so interrupt() / Command(resume=...) state persists across invocations
    on the same thread_id.

    Pass checkpointer=None to compile without a checkpointer (LangGraph Studio path).

    Args:
        adapter:        MempillAdapter — the single mempill boundary.
        tools:          ShowcaseTools NamedTuple with all 7 agent tool instances.
        checkpointer:   Checkpointer instance, None, or _SENTINEL (default=MemorySaver).
        model_name:     Anthropic model string. Defaults to ANTHROPIC_MODEL env var
                        or "claude-haiku-4-5".
        agent_id:       Optional agent_id override for the system prompt's
                        "agent_id is always X" rule (TASK-31 dual-agent router).
                        Defaults to "jordan-park-001" (today's behavior).
        responsibility: Optional per-instance responsibility preamble (TASK-31
                        dual-agent router). Defaults to "" (today's behavior:
                        no leading paragraph).
        known_entities: Optional per-instance KNOWN ENTITIES line (TASK-31
                        dual-agent router). Defaults to the full legacy list
                        (today's behavior).
    """
    if model_name is None:
        model_name = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5")

    model = ChatAnthropic(model=model_name)  # type: ignore[call-arg]

    tool_list = list(tools)

    if checkpointer is _SENTINEL:
        checkpointer = MemorySaver()

    system_prompt = build_system_prompt(
        agent_id=agent_id,
        responsibility=responsibility,
        known_entities=known_entities,
    )

    app = create_react_agent(
        model=model,
        tools=tool_list,
        checkpointer=checkpointer,
        prompt=system_prompt,
    )

    log.info(
        "build_graph: ReAct ExecAssistant compiled (model=%s, tools=%d, checkpointer=%s)",
        model_name,
        len(tool_list),
        type(checkpointer).__name__ if checkpointer else "None",
    )
    return app
