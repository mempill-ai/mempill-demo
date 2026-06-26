"""mempill_langgraph.prompts — All prompt templates and memory-block formatters."""
from __future__ import annotations

# ── Canonical key convention (shared by BOTH write and read paths) ────────────
#
# Rules:
#   role/position facts  → subject="<org>:<role>" (lowercase), predicate="held_by",  value="<person>"
#   location facts       → subject="<entity>" (lowercase),     predicate="lives_in",  value="<place>"
#   generic facts        → subject="<entity>" (lowercase),     predicate="<snake_case>", value="<asserted value>"
#
# Always lowercase subject and predicate.  The purpose is that the SAME real-world
# fact always maps to the SAME (subject, predicate) key regardless of whether it
# is being written or read.

CANONICAL_KEY_CONVENTION = """\
CANONICAL KEY CONVENTION — apply to every (subject, predicate) pair you emit:
- Role/position facts : subject = "<org>:<role>"  (lowercase, e.g. "acme:ceo"), predicate = "held_by",  value = the person
- Location facts      : subject = "<entity>"       (lowercase),                  predicate = "lives_in", value = the place
- Generic facts       : subject = "<entity>"       (lowercase),                  predicate = "<snake_case_property>"
- ALWAYS lowercase subject and predicate.
- Use the canonical form consistently so write and read keys always match.

Few-shot examples:
  Statement "Alice is the CEO of Acme."          → subject="acme:ceo"  predicate="held_by"  value="Alice"
  Statement "Acme's CEO is Alice."               → subject="acme:ceo"  predicate="held_by"  value="Alice"
  Question  "Who is Acme CEO?"                   → subject="acme:ceo"  predicate="held_by"
  Question  "Who is the CEO of Acme?"            → subject="acme:ceo"  predicate="held_by"
  Statement "Bob lives in Paris."                → subject="bob"        predicate="lives_in" value="Paris"
  Question  "Where does Bob live?"               → subject="bob"        predicate="lives_in"
  Statement "Eve is the CTO of Globex."          → subject="globex:cto" predicate="held_by"  value="Eve"
  Question  "Who is Globex's CTO?"               → subject="globex:cto" predicate="held_by"\
"""

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
Extract any new factual claims from the following user message.
A factual claim is a (subject, predicate, value) triple asserting something about the world that the USER is stating as fact.
Return an empty claims list for: greetings, questions, expressions of uncertainty, commands, or procedural requests.
Do NOT extract claims framed as contested, uncertain, or that are merely questions about the world.
All claims extracted from a user message are user-asserted facts — set is_user_asserted=True for every claim.

{canonical_convention}

Message:
{{user_message}}"""

# Render EXTRACTION_PROMPT with the canonical convention embedded
EXTRACTION_PROMPT = EXTRACTION_PROMPT.format(canonical_convention=CANONICAL_KEY_CONVENTION)

# ── Key-extraction prompt for retrieve_memory node ───────────────────────────

KEY_EXTRACTION_PROMPT = """\
Given the user's question below, identify the canonical (subject, predicate) key
that should be queried in memory to answer it.

Return subject and predicate using the SAME canonical convention as the write path.
If the question is a greeting, small-talk, or has no identifiable subject, return
subject="" and predicate="" (empty strings).

{canonical_convention}

Question:
{{user_question}}"""

KEY_EXTRACTION_PROMPT = KEY_EXTRACTION_PROMPT.format(canonical_convention=CANONICAL_KEY_CONVENTION)
