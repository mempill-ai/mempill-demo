"""
mempill_showcase.tests.test_langgraph_graph — Wave B ReAct agent integration tests.

No live LLM required for the structural tests.  The HITL interrupt/resume path
is verified deterministically by calling the tool directly (bypassing the LLM),
then testing the interrupt/Command(resume=...) mechanics against the compiled graph.

Test classes:
  TestBuildGraph          — build_graph() produces a compiled app from a ShowcaseTools.
  TestBuildApp            — build_app() factory returns (app, adapter) + app is runnable.
  TestHITLInterrupt       — RequestAdjudicationTool calls interrupt(); Command(resume)
                            resumes and resolves the contested belief.
  TestStudioGraph         — studio_graph module imports + exposes a compiled graph
                            without MemorySaver; seeded adapter has Day-0 data.
"""
from __future__ import annotations

import json
import uuid

import pytest

from langgraph.types import Command

from mempill_showcase.config.di import build_mempill_adapter, build_agent_tools, build_app
from mempill_showcase.frameworks.langgraph.graph import build_graph, build_system_prompt, ShowcaseTools
from mempill_showcase.scenarios.seed_data import AGENT_ID


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def adapter():
    """Fresh oracle-backed in-memory mempill adapter per test."""
    return build_mempill_adapter(in_memory=True, oracle_backed=True)


@pytest.fixture()
def tools(adapter):
    return build_agent_tools(adapter)


@pytest.fixture()
def app(adapter, tools):
    """Compiled ReAct agent app (MemorySaver checkpointer)."""
    return build_graph(adapter=adapter, tools=tools)


def _cfg(thread_id: str | None = None) -> dict:
    return {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}


# ── TestBuildGraph ────────────────────────────────────────────────────────────

class TestBuildGraph:
    """build_graph() produces a compiled app from a ShowcaseTools."""

    def test_build_graph_returns_non_none(self, adapter, tools):
        """build_graph() returns a non-None compiled app."""
        compiled = build_graph(adapter=adapter, tools=tools)
        assert compiled is not None, "build_graph must return a compiled app"

    def test_tools_is_showcase_tools(self, tools):
        """build_agent_tools returns a ShowcaseTools NamedTuple with 10 tools."""
        assert isinstance(tools, ShowcaseTools)
        assert len(tools) == 10, f"ShowcaseTools must have 10 tools, got {len(tools)}"

    def test_all_tool_names_are_unique(self, tools):
        """All 10 tool names are distinct."""
        names = [t.name for t in tools]
        assert len(set(names)) == len(names), f"Duplicate tool names: {names}"

    def test_build_graph_with_no_checkpointer(self, adapter, tools):
        """build_graph(checkpointer=None) compiles without a MemorySaver (Studio path)."""
        compiled = build_graph(adapter=adapter, tools=tools, checkpointer=None)
        assert compiled is not None
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = getattr(compiled, "checkpointer", None)
        assert not isinstance(checkpointer, MemorySaver), (
            "Graph with checkpointer=None must NOT carry a MemorySaver"
        )


# ── TestBuildApp ──────────────────────────────────────────────────────────────

class TestBuildApp:
    """build_app() factory returns (app, adapter)."""

    def test_build_app_returns_tuple(self):
        """build_app() returns (app, adapter) where both are non-None."""
        app, adapter = build_app()
        assert app is not None, "app must not be None"
        assert adapter is not None, "adapter must not be None"

    def test_build_app_reuses_adapter(self):
        """build_app(adapter=<existing>) reuses the provided adapter."""
        existing = build_mempill_adapter(in_memory=True)
        app, returned = build_app(adapter=existing)
        assert returned is existing, "build_app must reuse the passed adapter"

    def test_build_app_has_memory_saver(self):
        """build_app() default path attaches a MemorySaver checkpointer."""
        from langgraph.checkpoint.memory import MemorySaver
        app, _ = build_app()
        checkpointer = getattr(app, "checkpointer", None)
        assert isinstance(checkpointer, MemorySaver), (
            "Default build_app() must carry a MemorySaver checkpointer"
        )


# ── TestHITLInterrupt ─────────────────────────────────────────────────────────

class TestHITLInterrupt:
    """RequestAdjudicationTool triggers interrupt(); Command(resume) resolves it.

    We test the interrupt/resume mechanics by invoking the tool DIRECTLY inside
    a compiled graph (bypassing the LLM router) via a pre-seeded conflict + a
    minimal state machine test.

    AC-2 assertions:
      (a) Calling request_adjudication_tool.invoke() raises a LangGraph interrupt.
      (b) The interrupt payload describes the conflict.
      (c) _normalize_verdict correctly maps 'Affirm' → 'Affirm'.
      (d) After oracle Affirm on a seeded conflict, post-resolution recall returns CTO.
    """

    def _seed_vp(self, adapter):
        """Seed alice-chen/employer = VP Engineering (2023-06)."""
        from mempill_showcase.core.domain.models import ClaimInput
        import mempill

        adapter.write_claim(AGENT_ID, ClaimInput(
            subject="alice-chen",
            predicate="employer",
            value="Acme Corp / VP Engineering",
            valid_from="2023-06",
            confidence=1.0,
            provenance=mempill.ProvenanceLabel.external_user_asserted(),
        ))

    def test_normalize_verdict_affirm(self):
        """_normalize_verdict maps 'affirm' (case-insensitive) → 'Affirm'."""
        from mempill_showcase.frameworks.langgraph.hitl_node import _normalize_verdict

        assert _normalize_verdict("Affirm", challenger_value="CTO", incumbent_value="VP") == "Affirm"
        assert _normalize_verdict("affirm", challenger_value="CTO", incumbent_value="VP") == "Affirm"
        assert _normalize_verdict("yes", challenger_value="CTO", incumbent_value="VP") == "Affirm"
        assert _normalize_verdict("CTO", challenger_value="CTO", incumbent_value="VP") == "Affirm"

    def test_normalize_verdict_deny(self):
        """_normalize_verdict maps 'deny' → 'Deny'."""
        from mempill_showcase.frameworks.langgraph.hitl_node import _normalize_verdict

        assert _normalize_verdict("Deny", challenger_value="CTO", incumbent_value="VP") == "Deny"
        assert _normalize_verdict("no", challenger_value="CTO", incumbent_value="VP") == "Deny"
        assert _normalize_verdict("VP", challenger_value="CTO", incumbent_value="VP") == "Deny"

    def test_normalize_verdict_abstain(self):
        """_normalize_verdict maps 'abstain' → 'Abstain'."""
        from mempill_showcase.frameworks.langgraph.hitl_node import _normalize_verdict

        assert _normalize_verdict("Abstain", challenger_value=None, incumbent_value=None) == "Abstain"
        assert _normalize_verdict("defer", challenger_value=None, incumbent_value=None) == "Abstain"

    def test_normalize_verdict_invalid(self):
        """_normalize_verdict returns '_invalid_' for unrecognised input."""
        from mempill_showcase.frameworks.langgraph.hitl_node import _normalize_verdict

        result = _normalize_verdict("not_a_verdict_xyz", challenger_value=None, incumbent_value=None)
        assert result == "_invalid_"

    def test_request_adjudication_tool_raises_interrupt(self, adapter, tools):
        """Calling request_adjudication_tool.invoke() inside a graph pauses at interrupt.

        We build a minimal graph with the request_adjudication tool as a tool call
        and verify the graph yields __interrupt__ on the first invoke.
        """
        from mempill_showcase.core.domain.models import ClaimInput
        import mempill

        # Seed a conflict so the oracle queue has a pending entry
        adapter.write_claim(AGENT_ID, ClaimInput(
            subject="alice-chen",
            predicate="employer",
            value="Acme Corp / VP Engineering",
            valid_from="2023-06",
            confidence=1.0,
            provenance=mempill.ProvenanceLabel.external_user_asserted(),
        ))
        adapter.write_claim(AGENT_ID, ClaimInput(
            subject="alice-chen",
            predicate="employer",
            value="Acme Corp / CTO",
            valid_from="2023-06",
            confidence=1.0,
            provenance=mempill.ProvenanceLabel.external_user_asserted(),
        ))

        # Build a tiny graph that calls request_adjudication deterministically
        from langgraph.graph import StateGraph, END
        from langgraph.checkpoint.memory import MemorySaver
        from mempill_showcase.frameworks.langgraph.state import ExecAssistantState

        adj_tool = tools.request_adjudication_tool

        def _call_adjudication(state):
            result = adj_tool.invoke({
                "agent_id": AGENT_ID,
                "subject": "alice-chen",
                "predicate": "employer",
                "reason": "Conflict between VP Engineering and CTO",
                "incumbent_value": "Acme Corp / VP Engineering",
                "challenger_value": "Acme Corp / CTO",
                "claim_refs": [],
            })
            return {"output_text": result}

        g = StateGraph(ExecAssistantState)
        g.add_node("adjudicate", _call_adjudication)
        g.set_entry_point("adjudicate")
        g.add_edge("adjudicate", END)

        compiled = g.compile(checkpointer=MemorySaver())
        thread_id = str(uuid.uuid4())
        cfg = {"configurable": {"thread_id": thread_id}}

        result = compiled.invoke({"user_input": "test", "agent_id": AGENT_ID}, cfg)

        assert "__interrupt__" in result, (
            f"Graph must pause at interrupt(), got keys: {list(result.keys())}"
        )
        interrupts = result["__interrupt__"]
        assert len(interrupts) >= 1
        payload = interrupts[0].value
        assert "question" in payload or "subject" in payload, (
            f"Interrupt payload must describe the conflict, got: {payload}"
        )

    def test_hitl_affirm_resolves_belief(self, adapter, tools):
        """After oracle Affirm on a seeded same-period conflict, recall returns CTO."""
        from mempill_showcase.core.domain.models import ClaimInput
        import mempill

        # Seed conflict
        adapter.write_claim(AGENT_ID, ClaimInput(
            subject="alice-chen",
            predicate="employer",
            value="Acme Corp / VP Engineering",
            valid_from="2023-06",
            confidence=1.0,
            provenance=mempill.ProvenanceLabel.external_user_asserted(),
        ))
        adapter.write_claim(AGENT_ID, ClaimInput(
            subject="alice-chen",
            predicate="employer",
            value="Acme Corp / CTO",
            valid_from="2023-06",
            confidence=1.0,
            provenance=mempill.ProvenanceLabel.external_user_asserted(),
        ))

        # Submit Affirm via oracle
        pending = adapter.list_pending_adjudications(AGENT_ID)
        assert pending, "Oracle queue must have pending entry"
        result = adapter.submit_adjudication(AGENT_ID, pending[0]["handle_id"], "Affirm")
        assert result.get("disposition") == "CommittedCheap"

        # Post-resolution recall must return CTO
        belief = adapter.recall(AGENT_ID, "alice-chen", "employer")
        assert belief.status == "Resolved", f"After Affirm, recall must be Resolved, got {belief.status!r}"
        assert "CTO" in (belief.value or ""), f"After Affirm, CTO must win, got {belief.value!r}"

    def test_hitl_interrupt_and_resume_via_graph(self, adapter, tools):
        """Full interrupt → Command(resume='Affirm') → resolved cycle via compiled graph."""
        from mempill_showcase.core.domain.models import ClaimInput
        import mempill
        from langgraph.graph import StateGraph, END
        from langgraph.checkpoint.memory import MemorySaver
        from mempill_showcase.frameworks.langgraph.state import ExecAssistantState

        # Seed conflict
        adapter.write_claim(AGENT_ID, ClaimInput(
            subject="alice-chen",
            predicate="employer",
            value="Acme Corp / VP Engineering",
            valid_from="2023-06",
            confidence=1.0,
            provenance=mempill.ProvenanceLabel.external_user_asserted(),
        ))
        adapter.write_claim(AGENT_ID, ClaimInput(
            subject="alice-chen",
            predicate="employer",
            value="Acme Corp / CTO",
            valid_from="2023-06",
            confidence=1.0,
            provenance=mempill.ProvenanceLabel.external_user_asserted(),
        ))

        adj_tool = tools.request_adjudication_tool

        def _call_adjudication(state):
            raw = adj_tool.invoke({
                "agent_id": AGENT_ID,
                "subject": "alice-chen",
                "predicate": "employer",
                "reason": "VP vs CTO conflict",
                "incumbent_value": "Acme Corp / VP Engineering",
                "challenger_value": "Acme Corp / CTO",
                "claim_refs": [],
            })
            parsed = json.loads(raw)
            return {
                "output_text": raw,
                "hitl_verdict": parsed.get("verdict"),
                "hitl_resolved_belief": json.dumps({"value": parsed.get("winning_value")}),
            }

        g = StateGraph(ExecAssistantState)
        g.add_node("adjudicate", _call_adjudication)
        g.set_entry_point("adjudicate")
        g.add_edge("adjudicate", END)

        compiled = g.compile(checkpointer=MemorySaver())
        thread_id = str(uuid.uuid4())
        cfg = {"configurable": {"thread_id": thread_id}}

        # First invoke → interrupt
        result1 = compiled.invoke({"user_input": "test", "agent_id": AGENT_ID}, cfg)
        assert "__interrupt__" in result1, "Graph must pause at interrupt"

        # Resume with Affirm
        result2 = compiled.invoke(Command(resume="Affirm"), cfg)

        # Graph completed; check verdict
        assert result2.get("hitl_verdict") == "Affirm", (
            f"hitl_verdict should be 'Affirm', got {result2.get('hitl_verdict')!r}"
        )
        hrb = result2.get("hitl_resolved_belief")
        assert hrb is not None, "hitl_resolved_belief must be set after Affirm"
        rb = json.loads(hrb)
        assert "CTO" in (rb.get("value") or ""), (
            f"Resolved belief must contain CTO, got {rb.get('value')!r}"
        )


# ── TestBuildSystemPrompt ─────────────────────────────────────────────────────

class TestBuildSystemPrompt:
    """build_system_prompt() (TASK-31 T31-2): defaults must be byte-identical
    to the historical hardcoded _SYSTEM_PROMPT constant.
    """

    # Captured verbatim from the pre-TASK-31 _SYSTEM_PROMPT constant (git blame:
    # graph.py before the build_system_prompt() extraction). Any future edit to
    # the DEFAULT prompt text must update this frozen copy deliberately.
    _LEGACY_SYSTEM_PROMPT = """\
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

4. To record a CORRECTION to an existing fact ("Alice is actually CTO, not VP Engineering, since the same date"):
   → FIRST call recall_subject to find the existing predicate for this type of fact.
   → Reuse the SAME predicate name and value FORMAT already stored. Do NOT split or rename.
   → For valid_from, use the date AND granularity the USER stated — do NOT add precision
     the user didn't give. "June 2023" → valid_from="2023-06" (month). A bare year → "YYYY"
     (year). A full date ("June 15, 2023") → "2023-06-15" (day). NEVER pad a month or year
     into a fabricated day-precision date.
   → For valid_until (bounded/fixed-term facts): the SAME fidelity rule applies — use
     the EXACT end date/month/year the user stated, at their stated precision. NEVER
     compute, infer, or extrapolate an end date (e.g. never derive it by adding an
     assumed term length to valid_from, and never copy a year from an unrelated fact
     elsewhere in context/memory). "John is CEO from January 2025 until December 2025"
     → valid_from="2025-01", valid_until="2025-12" — NOT a different year. If no end
     was explicitly stated, pass valid_until=None (open-ended) rather than guessing one.
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
   → This returns the engine's own chronological fold — each entry's
     valid_until reflects the engine's EFFECTIVE window for that claim (its
     own stated end, or an adjacent entry's start where that comes first).
     This is NOT evidence that "a later adjudication shortened" the claim,
     and overlapping windows on the same line are a genuine, possibly
     unresolved conflict rather than an already-settled succession — consult
     recall_subject/recall_at for the current Contested status before
     asserting which claim "won".
   → Report the entries exactly as returned, in order. Do NOT reconstruct
     history by hand from audit_trail or recall_subject, and do NOT narrate
     each claim's originally-stated valid_from/valid_until as if it were the
     current effective window — this tool's fold is the source of truth for
     the chronology.
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
- agent_id is always "jordan-park-001" unless the user specifies otherwise.
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

    # NOTE (intended, reviewed change — TASK-31-W5): the two blocks above (rule 11's
    # "CRITICAL — dates" bullet and the closing RULES' "*_display" bullet) were added
    # to fix a fabricated-day-precision bug (month-granular facts like "December 2025"
    # were verbalised as "December 1, 2025"). This snapshot was updated deliberately as
    # part of that fix, not organic drift.
    #
    # NOTE (intended, reviewed change — TASK-33-W3-DEMO): rule 11's main paragraph was
    # rewritten to drop the false "a later adjudication can shorten an earlier entry's
    # end date" narrative — query_history's valid_until reflects the engine's effective
    # window, not proof of an adjudication outcome, and overlapping entries may be an
    # UNRESOLVED conflict. See mempill-demo/src/mempill_showcase/tools/query_history_tool.py
    # docstring for the matching fix.
    #
    # NOTE (intended, reviewed change — TASK-33-W3-DEMO): rule 4 gained a "For
    # valid_until (bounded/fixed-term facts)" bullet mirroring the pre-existing
    # valid_from fidelity bullet — added while investigating a live anomaly where a
    # CEO's fixed-term end date was stored a year later than stated (2026-12 instead
    # of 2025-12). Root cause could not be conclusively proven (no raw user-turn text
    # is persisted anywhere queryable); this bullet + the matching RememberFactTool
    # schema/description hardening are the harm-reduction fix for the LLM-tool-arg
    # hallucination hypothesis.

    def test_defaults_byte_identical_to_legacy_constant(self):
        """build_system_prompt() with all-default args == the historical constant."""
        result = build_system_prompt()
        assert result == self._LEGACY_SYSTEM_PROMPT, (
            "build_system_prompt() defaults must be BYTE-IDENTICAL to the "
            "pre-TASK-31 hardcoded _SYSTEM_PROMPT constant"
        )

    def test_custom_agent_id_overrides_default(self):
        """Passing agent_id changes the 'agent_id is always X' rule line."""
        result = build_system_prompt(agent_id="people-ops-001")
        assert 'agent_id is always "people-ops-001"' in result
        assert 'agent_id is always "jordan-park-001"' not in result

    def test_custom_responsibility_adds_leading_paragraph(self):
        """Passing responsibility prepends a leading paragraph; default has none."""
        default_result = build_system_prompt()
        assert not default_result.startswith("You manage")

        custom_result = build_system_prompt(responsibility="You manage widgets.")
        assert custom_result.startswith("You manage widgets.\n\n")

    def test_custom_known_entities_overrides_default(self):
        """Passing known_entities changes the KNOWN ENTITIES line."""
        result = build_system_prompt(known_entities="acme-corp")
        assert "acme-corp" in result
        assert "alice-chen, bob-liu, acme-corp, jordan-park" not in result


# ── TestStudioGraph ───────────────────────────────────────────────────────────

class TestStudioGraph:
    """Guard tests: studio_graph module exposes a compiled graph without MemorySaver."""

    def test_studio_graph_imports(self):
        """studio_graph.py is importable and exposes a module-level `graph`."""
        from mempill_showcase.frameworks.langgraph.studio_graph import graph
        assert graph is not None, "studio_graph.graph must be non-None"

    def test_studio_graph_has_no_memory_saver(self):
        """studio_graph.graph has no MemorySaver (Studio injects its own checkpointer)."""
        from langgraph.checkpoint.memory import MemorySaver
        from mempill_showcase.frameworks.langgraph.studio_graph import graph
        checkpointer = getattr(graph, "checkpointer", None)
        assert not isinstance(checkpointer, MemorySaver), (
            "studio_graph.graph must NOT have a MemorySaver checkpointer. "
            f"Got checkpointer={type(checkpointer).__name__!r}"
        )

    def test_studio_adapter_seeded_with_day0_data(self):
        """Importing studio_graph seeds the adapter: alice-chen/city belief exists."""
        from mempill_showcase.frameworks.langgraph.studio_graph import studio_adapter
        from mempill_showcase.scenarios.seed_data import AGENT_ID

        belief = studio_adapter.recall(AGENT_ID, "alice-chen", "city")
        assert belief is not None
        assert belief.status != "NoBelief", (
            f"Day-0 seed city should be present (not NoBelief), got {belief.status!r}"
        )

    def test_studio_adapter_seeded_dietary(self):
        """Seeded adapter has alice-chen/dietary_restriction=vegetarian."""
        from mempill_showcase.frameworks.langgraph.studio_graph import studio_adapter
        from mempill_showcase.scenarios.seed_data import AGENT_ID

        belief = studio_adapter.recall(AGENT_ID, "alice-chen", "dietary_restriction")
        assert belief is not None
        assert belief.value == "vegetarian", (
            f"Expected 'vegetarian', got {belief.value!r}"
        )
