"""mempill_langgraph.prompts — All prompt templates and memory-block formatters."""
from __future__ import annotations

# ── Deterministic memory-block templates ─────────────────────────────────────

CONTESTED_BLOCK = """\
[MEMORY STATUS: CONTESTED]
Subject: {subject} | Predicate: {predicate}
  Claim A: "{value_a}"  conf={conf_a}  valid: {start_a} -> {end_a}
  Claim B: "{value_b}"  conf={conf_b}  valid: {start_b} -> {end_b}
INSTRUCTION: Surface BOTH claims. Do NOT pick one. Ask the user to /reconcile."""

RESOLVED_BLOCK = """\
[MEMORY STATUS: {status}]
Subject: {subject} | Predicate: {predicate}
  Value: "{value}"  conf={conf}  valid: {vt_start} -> {vt_end}  source={provenance}  corroborations={corroboration}"""

NO_BELIEF_BLOCK = """\
[MEMORY STATUS: NO_BELIEF]
Subject: {subject} | Predicate: {predicate}
No memory stored. Do not invent a value."""

# Empty block — no subject detected (greeting, small-talk)
EMPTY_BLOCK = ""

# ── System prompt for the respond node ──────────────────────────────────────

MEMORY_SYSTEM_PREFIX = """\
You are a helpful conversational assistant backed by mempill, a temporally-correct memory system.

Memory rules (non-negotiable):
- If memory status is CONTESTED: you MUST surface both claims to the user verbatim. Never choose between them. Suggest running /reconcile.
- If memory status is SUPERSEDED: report the current value; note prior version exists.
- If memory status is NO_BELIEF: say you have no memory of this. Never invent a value.
- If memory status is Committed/CommittedCheap: answer with confidence; cite the valid timeframe.
- If no memory block is present: respond naturally from conversation context only.
- Greetings, small-talk, and chitchat: respond naturally. There is nothing wrong with "Hi!"

{memory_context}"""

# ── Extraction prompt for write_memory node ──────────────────────────────────

EXTRACTION_PROMPT = """\
Extract any new factual claims from the following assistant message.
A factual claim is a (subject, predicate, value) triple asserting something about the world.
Return an empty claims list for: greetings, questions, expressions of uncertainty, procedural responses, or contested reports.
Do NOT extract claims already framed as contested or uncertain.

Message:
{assistant_message}"""
