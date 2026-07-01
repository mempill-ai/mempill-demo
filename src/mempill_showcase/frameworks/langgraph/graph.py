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
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

if TYPE_CHECKING:
    from mempill_showcase.adapters.memory.mempill_adapter import MempillAdapter
    from mempill_showcase.tools.audit_trail_tool import AuditTrailTool
    from mempill_showcase.tools.get_contested_tool import GetContestedTool
    from mempill_showcase.tools.recall_as_of_tool import RecallAsOfTool
    from mempill_showcase.tools.recall_at_tool import RecallAtTool
    from mempill_showcase.tools.recall_subject_tool import RecallSubjectTool
    from mempill_showcase.tools.remember_fact_tool import RememberFactTool
    from mempill_showcase.tools.request_adjudication_tool import RequestAdjudicationTool

log = logging.getLogger(__name__)

# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a bi-temporal memory assistant backed by the mempill engine.
You answer ANY natural-language question by consulting memory tools — never invent facts.

KNOWN ENTITIES (always normalise: lowercase, spaces→hyphens):
  alice-chen, bob-liu, acme-corp, jordan-park

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

4. To record a new fact ("Alice is now CTO of Acme since June 2023"):
   → FIRST call recall_subject to find the existing predicate for this type of fact.
   → Reuse the SAME predicate name and value FORMAT already stored. Do NOT split or rename.
   → Example: if recall_subject shows predicate="employer", value="Acme Corp / VP Engineering"
     and the user says "Alice is now CTO", write:
       remember_fact(subject="alice-chen", predicate="employer", value="Acme Corp / CTO", ...)
     NOT a new predicate "role" — keep it in the existing "employer" predicate, same format.
   → if the result shows is_contested=true: call get_contested, then call request_adjudication
   → NEVER use a contested fact without resolving it first

5. To inspect what values conflict for a contested fact:
   → call get_contested(agent_id, subject, predicate)

6. To request human adjudication of a contested write:
   → call request_adjudication(agent_id, subject, predicate, reason,
       incumbent_value, challenger_value, claim_refs)
   → the graph will pause; wait for the human resume verdict; then report the winner

7. For compliance/audit queries ("show me all write events"):
   → call audit_trail(agent_id, limit)

RULES:
- Always consult memory before answering; never invent facts.
- agent_id is always "jordan-park-001" unless the user specifies otherwise.
- When recall returns NoBelief, say so honestly — do not guess.
- When recall returns Contested, call get_contested and surface both values;
  do NOT use a contested value in an action (write, briefing) without adjudication.
- Subject normalisation: strip → lowercase → spaces→hyphens.
  ("Alice Chen" → "alice-chen", "Acme Corp" → "acme-corp")
- Predicate names AND value formats are stored as you see them in recall results.
  ALWAYS reuse the same predicate and value format on writes — never split or rename.
  ("Alice is CTO of Acme" updates predicate="employer" with value="Acme Corp / CTO")
"""

# ── ShowcaseTools NamedTuple (kept for DI wiring compatibility) ───────────────


class ShowcaseTools(NamedTuple):
    """7 agent tools needed by the ReAct graph."""
    recall_subject_tool: "RecallSubjectTool"
    recall_at_tool: "RecallAtTool"
    recall_as_of_tool: "RecallAsOfTool"
    remember_fact_tool: "RememberFactTool"
    get_contested_tool: "GetContestedTool"
    request_adjudication_tool: "RequestAdjudicationTool"
    audit_trail_tool: "AuditTrailTool"


# ── Sentinel for checkpointer default ────────────────────────────────────────

_SENTINEL = object()


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph(
    adapter: "MempillAdapter",
    tools: ShowcaseTools,
    checkpointer: Any = _SENTINEL,
    model_name: Optional[str] = None,
) -> Any:
    """Build and compile the ReAct ExecAssistant agent.

    Returns a compiled LangGraph app.  By default a MemorySaver checkpointer is
    attached so interrupt() / Command(resume=...) state persists across invocations
    on the same thread_id.

    Pass checkpointer=None to compile without a checkpointer (LangGraph Studio path).

    Args:
        adapter:      MempillAdapter — the single mempill boundary.
        tools:        ShowcaseTools NamedTuple with all 7 agent tool instances.
        checkpointer: Checkpointer instance, None, or _SENTINEL (default=MemorySaver).
        model_name:   Anthropic model string. Defaults to ANTHROPIC_MODEL env var
                      or "claude-haiku-4-5".
    """
    if model_name is None:
        model_name = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5")

    model = ChatAnthropic(model=model_name)  # type: ignore[call-arg]

    tool_list = list(tools)

    if checkpointer is _SENTINEL:
        checkpointer = MemorySaver()

    app = create_react_agent(
        model=model,
        tools=tool_list,
        checkpointer=checkpointer,
        prompt=_SYSTEM_PROMPT,
    )

    log.info(
        "build_graph: ReAct ExecAssistant compiled (model=%s, tools=%d, checkpointer=%s)",
        model_name,
        len(tool_list),
        type(checkpointer).__name__ if checkpointer else "None",
    )
    return app
