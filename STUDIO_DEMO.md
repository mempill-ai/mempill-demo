# mempill Studio Demo Script

Interactive demo for the free-form ReAct agent in LangGraph Studio.
Verified end-to-end with real LLM (`claude-haiku-4-5-20251001`).

---

## Setup

**1. Prerequisites**
```bash
# Ensure ANTHROPIC_API_KEY is in .env (already configured)
# Do NOT run uv sync or reinstall mempill — use .venv/bin/python

# Optional: reset the persistent DB to start fresh
rm -f .mempill/showcase.db
```

**2. Start LangGraph Studio**
```bash
.venv/bin/langgraph dev
# Studio opens at http://127.0.0.1:2024
```

**3. Open Studio**
- Navigate to `http://127.0.0.1:2024`
- Select the `exec_assistant` graph from the graph picker
- The graph is a single free-form ReAct agent (one node visible)

**4. Configuration**
- In the **Input** panel, fill the `messages` field with a natural-language question
- Day-0 facts are seeded automatically at graph module-import time (when `langgraph dev` starts):
  - alice-chen / employer = Acme Corp / VP Engineering (since 2023-06)
  - alice-chen / city = Austin TX (since 2023-06, bounded 2025-02)
  - alice-chen / dietary_restriction = vegetarian (since 2024-01)
  - bob-liu / employer = Meridian Ventures / Partner
  - bob-liu / travel_preference = window seat, no checked bags
  - acme-corp / ceo = Diane Foster
  - jordan-park / preferred_hotel = Marriott Bonvoy Gold

---

## Turn-by-Turn Demo Script

Run these turns IN ORDER in a single Studio thread to cover the full scenario.

---

### Turn 1 — Ask-anything (dietary restriction)

**User Input:**
```
What is Alice Chen's dietary restriction?
```

**Expected behaviour:**
- Agent calls `recall_subject(agent_id="jordan-park-001", subject="alice-chen")`
- Reads all predicates returned (employer, city, dietary_restriction)
- Answers: "Alice Chen's dietary restriction is **vegetarian**."
- No interrupt — simple recall query.

**What to verify:** answer contains "vegetarian"; no Interrupts panel appears.

---

### Turn 2 — Point-in-time query

**User Input:**
```
What city was Alice living in during early 2024, say around March 2024?
```

**Expected behaviour:**
- Agent calls `recall_at(agent_id="jordan-park-001", subject="alice-chen", predicate="city", valid_at="2024-03-01T00:00:00Z")`
- Returns: **Austin TX** (Alice was in Austin from 2023-06 until 2025-02; March 2024 is within that window)
- Answer: "In early 2024, Alice Chen was living in **Austin, TX**."

**What to verify:** answer contains "Austin"; no interrupt.

---

### Turn 3 — New fact write (CommittedCheap)

**User Input:**
```
Alice Chen prefers business class travel as of 2025. Please store this in memory.
```

**Expected behaviour:**
- Agent calls `recall_subject(alice-chen)` first to check existing predicates
- Calls `remember_fact(subject="alice-chen", predicate="travel-preference", value="business class", valid_from="2025", agent_id="jordan-park-001")`
- alice-chen has no prior `travel-preference` claim → **CommittedCheap** (no conflict)
- Answer: "I've stored Alice Chen's travel preference: business class, effective 2025."
- No interrupt.

**What to verify:** no Interrupts panel; answer confirms the write.

---

### Turn 4 — Contested update → HITL interrupt

**User Input (use this exact phrasing):**
```
Alice has actually been CTO of Acme since June 2023, not VP Engineering
```

> **Why this exact phrasing?** The agent extracts `valid_from=2023-06` from "since June 2023".
> The seeded VP Engineering claim also has `valid_from=2023-06-01`. A same-period overlap
> on a Functional predicate is a genuine contradiction → **Contested** → QueuedForAdjudication
> → the agent calls `request_adjudication` → graph **PAUSES**.

**Expected behaviour:**
1. Agent calls `recall_subject(alice-chen)` → sees employer = "Acme Corp / VP Engineering"
2. Agent calls `remember_fact(subject="alice-chen", predicate="employer", value="Acme Corp / CTO", valid_from="2023-06", ...)` → `is_contested=true`
3. Agent calls `get_contested(alice-chen, employer)` → sees incumbent vs challenger
4. Agent calls `request_adjudication(...)` → **graph PAUSES here**
5. Studio shows the **Interrupts** panel on the right

**Interrupt payload shown in Studio:**
```
Conflict on alice-chen/employer:
  Incumbent:  'Acme Corp / VP Engineering'
  Challenger: 'Acme Corp / CTO'
Reason: [agent's explanation of the conflict]
Which is correct? Reply: 'Affirm' (challenger wins), 'Deny' (incumbent wins), or 'Abstain' (defer).
```

---

### Turn 5 — HITL resume (Affirm — CTO wins)

In the **Interrupts** panel that appears when the graph pauses, enter the verdict:

**Resume payload** (any of these are accepted):
```
Affirm
```
Or paste the candidate value directly (maps to Affirm because it matches the challenger):
```
Acme Corp / CTO
```

**Accepted verdict formats:**
- Exact keywords (case-insensitive): `Affirm`, `Deny`, `Abstain`
- Synonyms: `yes` / `accept` / `challenger` → Affirm; `no` / `reject` / `incumbent` → Deny; `defer` / `skip` → Abstain
- Pasted candidate value verbatim: `Acme Corp / CTO` → Affirm; `Acme Corp / VP Engineering` → Deny

**What happens after Affirm:**
- `request_adjudication_tool` receives the verdict
- Calls `adapter.list_pending_adjudications()` → finds the queued handle for alice-chen/employer
- Calls `adapter.submit_adjudication(handle_id, "Affirm")` → oracle resolves the conflict
- CTO claim committed (CommittedCheap); VP Engineering superseded
- Post-resolution recall confirms `status=Resolved, value="Acme Corp / CTO"`
- Agent resumes and answers: "Conflict resolved: Alice Chen is now the CTO at Acme Corp."

**What to verify:** Interrupts panel clears; answer confirms CTO won.

---

### Turn 6 — Confirm CTO resolution (recall after HITL)

**User Input:**
```
What role does Alice currently hold at Acme?
```

**Expected behaviour:**
- Agent calls `recall_subject(alice-chen)` → employer = "Acme Corp / CTO" (status=Resolved)
- Answer: "Alice Chen is the **CTO** (Chief Technology Officer) at Acme Corp."

**What to verify:** answer contains "CTO"; no interrupt.

---

### Turn 7 — Compliance audit

**User Input:**
```
Show me the full audit trail for agent jordan-park-001
```

**Expected behaviour:**
- Agent calls `audit_trail(agent_id="jordan-park-001", limit=50)`
- Returns descriptions of: 7 seed writes + travel_preference write + CTO contested write + oracle resolution
- Answer describes the events (write events, succession events, adjudication events)

**What to verify:** answer mentions events (write, commit, resolve, or similar); no interrupt.

---

## Deny and Abstain paths (separate fresh threads)

### Deny path

Start a **new thread** in Studio (click "New Thread"), then repeat Turn 4:
```
Alice has actually been CTO of Acme since June 2023, not VP Engineering
```
→ Graph pauses at HITL interrupt.

In the **Interrupts** panel, enter:
```
Deny
```

**Expected:** VP Engineering survives; `submit_adjudication(handle_id, "Deny")` commits the incumbent.
Post-resolution recall returns `Resolved` with `value="Acme Corp / VP Engineering"`.

### Abstain path

Same fresh thread as above, but enter:
```
Abstain
```

**Expected:** Both claims remain Contested; `pending_contested` is preserved.
The agent reports that the decision is deferred.

---

## Reset

To run the demo again from scratch:
```bash
# Stop langgraph dev (Ctrl+C)
rm -f .mempill/showcase.db
.venv/bin/langgraph dev
# Day-0 facts re-seeded at graph module-import time on next start
```

---

## Architecture Notes

**ReAct topology:** A single `create_react_agent(model, tools, checkpointer=MemorySaver())`.
The agent selects tools freely based on the question. No routing rules, no crew nodes.

**7 memory tools:**
- `recall_subject` — retrieve all predicates for an entity (ask-anything)
- `recall_at` — bi-temporal point-in-time query (world-history axis)
- `recall_as_of` — bi-temporal transaction-time query (what-did-we-know axis)
- `remember_fact` — write a new fact (triggers Contested if predicate is Functional and overlaps)
- `get_contested` — inspect conflicting values for a Contested predicate
- `request_adjudication` — HITL interrupt gate (pauses graph for human verdict)
- `audit_trail` — full compliance audit log

**HITL via tool interrupt:** `request_adjudication` calls `LangGraph interrupt(payload)` inside
`_run()`. The graph pauses mid-tool-call. `Command(resume=verdict)` resumes the tool at the
`interrupt()` return. The tool submits the verdict to the mempill oracle queue and recalls
the resolved belief before returning.

**Verdict normalization (#38 forgiving-verdict):** Accepts Affirm/Deny/Abstain (any case),
synonyms (yes/no/defer), and pasted candidate values. Unrecognized strings return an error
message and keep the claim Contested — never falsely report "resolved".

**Bi-temporal engine:** mempill stores valid-time AND transaction-time for every claim.
Turn 2 proves this: asking about March 2024 after a 2025-02 succession correctly returns
the Austin TX value from before the move — without overwriting history.
