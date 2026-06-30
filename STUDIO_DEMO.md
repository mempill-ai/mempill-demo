# mempill Studio Demo Script

Exhaustive demo covering every route in the ExecAssistant graph.
Verified end-to-end with real LLM (`claude-haiku-4-5-20251001`) on 2026-06-30.

---

## Setup

**1. Prerequisites**
```bash
# Ensure ANTHROPIC_API_KEY is in .env (already configured)
# Do NOT run uv sync or reinstall mempill — use .venv/bin/python

# Optional: reset the persistent DB to start fresh
rm -f .mempill/showcase_1.db
```

**2. Start LangGraph Studio**
```bash
.venv/bin/langgraph dev
# Studio opens at http://127.0.0.1:2024
```

**3. Open Studio**
- Navigate to `http://127.0.0.1:2024`
- Select the `exec_assistant` graph from the graph picker
- The graph has 5 nodes visible: `supervisor`, `crew_a`, `crew_b`, `crew_c`, `hitl_node`

**4. Configuration**
- In the **Input** panel, only fill the `user_input` field
- Leave `agent_id` blank — it defaults to `jordan-park-001`
- Day-0 facts are seeded automatically on first run:
  - alice-chen / employer = Acme Corp / VP Engineering (since 2023-06)
  - alice-chen / city = Austin TX (since 2023-06, bounded 2025-02)
  - alice-chen / dietary_restriction = vegetarian (since 2024-01)
  - bob-liu / employer = Meridian Ventures / Partner
  - bob-liu / travel_preference = window seat, no checked bags
  - acme-corp / ceo = Diane Foster
  - jordan-park / preferred_hotel = Marriott Bonvoy Gold

---

## Turn-by-Turn Demo Script

### Turn 1 — RECALL current city (R1)
**Node path:** supervisor → crew_c → END

**User Input:**
```
What is Alice Chen's current city?
```

**What lights up:** `supervisor` (RECALL_HISTORY) → `crew_c`

**Expected output:**
```
crew_c [recall_history]: alice-chen/city='Austin TX' status=Resolved
```
> **Verified:** Austin TX from Day-0 seed.

---

### Turn 2 — UPDATE succession (R2)
**Node path:** supervisor → crew_a → END (CommittedCheap, no HITL)

**User Input:**
```
Alice moved to New York in February 2025
```

**What lights up:** `supervisor` (UPDATE_CONTACT) → `crew_a`

**Expected output:**
```
crew_a [llm]: wrote alice-chen/city='New York' (disposition=CommittedCheap valid_from=2025-02)
```
> **Verified:** LLMExtractor extracts `alice-chen/city=New York valid_from=2025-02`. Succession folded — Austin bounded at 2025-02, New York committed. No HITL (different valid_from = no overlap).

---

### Turn 3 — RECALL after update (R3)
**Node path:** supervisor → crew_c → END

**User Input:**
```
What is Alice's city now?
```

**What lights up:** `supervisor` (RECALL_HISTORY) → `crew_c`

**Expected output:**
```
crew_c [recall_history]: alice-chen/city='New York' status=Resolved
```
> **Verified:** Current belief is New York (succession committed in Turn 2).

---

### Turn 4 — bi-temporal past query (R4)
**Node path:** supervisor → crew_c → END

**User Input:**
```
What was Alice's city in 2024?
```

**What lights up:** `supervisor` (RECALL_HISTORY) → `crew_c`

**Expected output:**
```
crew_c [recall_history]: alice-chen/city='Austin TX' status=Resolved (valid_at=2025-01-01T00:00:00Z ...)
```
> **Verified:** Bi-temporal query at 2024 (before the 2025-02 succession) correctly returns Austin TX.
> This is the core mempill value proposition: past beliefs are retrievable without overwriting.

---

### Turn 5 — PREPARE_BRIEFING (R5)
**Node path:** supervisor → crew_c → END

**User Input:**
```
Brief me on Alice
```

**What lights up:** `supervisor` (PREPARE_BRIEFING) → `crew_c`

**Expected output:**
```
Briefing for alice-chen: employer='Acme Corp / VP Engineering', city='New York', dietary_restriction='vegetarian'
```
> **Verified:** Briefing pulls current facts for employer, city, dietary_restriction from mempill.
> City reflects the post-succession New York value.

---

### Turn 6 — RESEARCH + distil to mempill (R6)
**Node path:** supervisor → crew_b → END

**User Input:**
```
Research Acme Corp for me
```

**What lights up:** `supervisor` (RESEARCH) → `crew_b`

**Expected output:**
```
crew_b [llm]: research complete — wrote 411 chars to RAG, distilled 2 claim(s) to mempill.
Summary: Acme Corp is a mid-market technology company founded in 2015 that produces enterprise software...
```
> **Verified (after fix):** `LLMResearcher` (single structured call) synthesises a factual summary
> written to the RAG store, AND distils ≤3 atomic claims (e.g. `acme-corp/ceo=Diane Foster`) to mempill
> via `remember_tool`. Audit entry count increases by 2. RAG store contains "Acme" text.
>
> **crew_b before fix:** CrewAI kickoff hallucinated "Final Answer" JSON without calling tools —
> 0 claims written, RAG written but mempill empty. `LLMResearcher` fixes this with a single
> bounded structured call.

---

### Turn 7 — UPDATE → Contested → HITL interrupt (R7)
**Node path:** supervisor → crew_a → hitl_node ← **PAUSED**

**User Input:**
```
Alice is now the CTO of Acme Corp
```

**What lights up:** `supervisor` (UPDATE_CONTACT) → `crew_a` → `hitl_node` (red/orange — INTERRUPTED)

**What happens:**
- `LLMExtractor` extracts: `alice-chen/employer=Acme Corp / CTO`
- Current belief: `Acme Corp / VP Engineering` (same subject/predicate, no clear succession)
- No `valid_from` → overlapping temporal claim → **Contested** → QueuedForAdjudication
- Graph **pauses** at `hitl_node`

**Interrupt payload shown in Studio (Interrupts panel):**
```
Conflict on alice-chen/employer:
  Incumbent:  'Acme Corp / VP Engineering' (valid from 2023-06, source: UserAsserted)
  Challenger: 'Acme Corp / CTO' (valid from unknown, source: UserAsserted)
Which is correct? Reply: 'Affirm' (challenger wins), 'Deny' (incumbent wins), or 'Abstain' (defer).
```

---

### Turn 8 — HITL Affirm → CTO wins (R8)
**How to resume in Studio:**
In the **Interrupts** panel that appears when the graph pauses, enter the verdict in the resume field:

**Resume payload:**
```
Affirm
```

**What lights up:** `hitl_node` resumes → END

**Expected output:**
```
HITL resolved alice-chen/employer: verdict=Affirm
```
> **Verified:** `hitl_node` calls `adapter.list_pending_adjudications()` → finds the queued handle
> → calls `adapter.submit_adjudication(handle_id, "Affirm")` → oracle engine resolves the conflict.
> The challenger (CTO) is committed; the incumbent (VP Engineering) is superseded.
> Post-resolution `recall_tool` confirms the belief is resolved.
>
> Note: if `valid_from` was not provided in Turn 7, mempill may report `TimingUncertain`
> (the oracle accepted the Affirm but the temporal window is ambiguous). This is expected
> behavior — the fact was resolved, time anchor is unknown.

---

### Turn 9 — HITL Deny (separate fresh run) (R9)
To demonstrate the Deny path, start a **new thread** in Studio (click "New Thread"):

**User Input (fresh thread):**
```
Alice is now the CTO of Acme Corp
```
→ Graph pauses at `hitl_node` again (same Contested scenario).

**Resume payload:**
```
Deny
```

**Expected output:**
```
HITL resolved alice-chen/employer: verdict=Deny → belief='Acme Corp / VP Engineering' status=Resolved
```
> **Verified:** `submit_adjudication(handle_id, "Deny")` → challenger rejected, incumbent survives.
> `adapter.recall(...)` returns `Acme Corp / VP Engineering` with `status=Resolved`.

---

### Turn 10 — COMPLIANCE_AUDIT (R10)
**Node path:** supervisor → crew_c → END (audit path)

**User Input:**
```
Show me the full compliance audit ledger for Alice
```

**What lights up:** `supervisor` (COMPLIANCE_AUDIT) → `crew_c`

**Expected output:**
```
crew_c [compliance_audit]: 12-14 audit entries retrieved for agent_id=jordan-park-001.
```
> **Verified:** `MempillAuditTool` retrieves the full bi-temporal ledger — every claim written,
> every succession fold, every Contested write and resolution — as an ordered audit trail.
> `entry_count > 0` confirms the ledger is populated.

---

## Reset Note

To run the demo again from scratch:
```bash
# Stop langgraph dev (Ctrl+C)
rm -f .mempill/showcase_1.db
.venv/bin/langgraph dev
# Day-0 facts re-seeded on next Studio invocation
```

---

## Route Summary Table (Verified)

| Route | Intent          | Node Path                      | Result                                    | Status |
|-------|-----------------|--------------------------------|-------------------------------------------|--------|
| R1    | RECALL_HISTORY  | supervisor → crew_c            | alice-chen/city = Austin TX               | PASS   |
| R2    | UPDATE_CONTACT  | supervisor → crew_a → END      | city = New York CommittedCheap (2025-02)  | PASS   |
| R3    | RECALL_HISTORY  | supervisor → crew_c            | alice-chen/city = New York (current)      | PASS   |
| R4    | RECALL_HISTORY  | supervisor → crew_c            | bi-temporal: city@2024 = Austin TX        | PASS   |
| R5    | PREPARE_BRIEFING| supervisor → crew_c            | Briefing: employer + city + dietary       | PASS   |
| R6    | RESEARCH        | supervisor → crew_b            | RAG+411 chars; 2 claims distilled         | PASS   |
| R7    | UPDATE_CONTACT  | supervisor → crew_a → hitl_node| Contested → INTERRUPTED                   | PASS   |
| R8    | HITL Affirm     | hitl_node resume               | CTO accepted; oracle Affirm submitted     | PASS   |
| R9    | HITL Deny       | hitl_node resume (fresh thread)| VP Engineering survives; oracle Deny      | PASS   |
| R10   | COMPLIANCE_AUDIT| supervisor → crew_c            | 12+ audit entries returned               | PASS   |

---

## Key Architecture Notes

**LLMSupervisor:** Single Anthropic call classifies intent into one of 5 labels.
Falls back to `RECALL_HISTORY` on any error (safe read-only default).

**LLMExtractor (crew_a):** Single structured Anthropic call extracts
`{entity, predicate, value, valid_from}` from free-form text. Python
`remember_tool` writes to mempill (no tool-loop, no hallucination risk).

**LLMResearcher (crew_b, NEW):** Single structured Anthropic call synthesises
a factual summary (→ RAG store) AND ≤3 distilled atomic claims (→ mempill via
`remember_tool`). Replaces the unreliable CrewAI kickoff which hallucinated
"Final Answer" JSON without calling tools. Before fix: 0 claims written.
After fix: 2 claims reliably written per research turn.

**Oracle HITL:** Contested writes queue in `list_pending_adjudications`.
`hitl_node` submits the human verdict via `submit_adjudication`. No LLM involved
in resolution — the human is the authority.

**Bi-temporal engine:** mempill stores valid-time AND transaction-time for every
claim. R4 proves this: asking about 2024 after a 2025-02 succession correctly
returns the Austin TX value from before the move.
