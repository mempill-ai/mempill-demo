# mempill Console Agent — Command Grammar

All commands are keyword-insensitive (INGEST, ingest, Ingest all work).
Slash commands must start with `/`.

---

## CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--db-dir DIR` | `.mempill` | Base directory for per-agent SQLite files (actual file: `<dir>/agent_<agent_id>.db`) |
| `--agent AGENT_ID` | `console-user` | Agent identifier (determines which DB file is used) |
| `--llm` | off | Enable LLM-backed natural language parsing (requires `ANTHROPIC_API_KEY`) |
| `--scenario` | off | Auto-play the 3-act demonstration scenario then hand off to REPL |
| `--selftest` | off | Run deterministic assertion suite and exit (no API key required) |
| `--reset` | off | Delete the persistent per-agent DB file and exit |
| `--verbose` | off | Enable verbose logging of engine calls (use `-v` for INFO, `-vv` for DEBUG) |

**Example:**
```bash
.venv/bin/python -m mempill_demo --db-dir /tmp/mp-verify --agent alice-01
.venv/bin/python -m mempill_demo --scenario                 # run demo, then REPL
.venv/bin/python -m mempill_demo --selftest                # run tests, exit
.venv/bin/python -m mempill_demo --reset                   # delete DB, exit
```

---

## Write Commands

### INGEST
```
INGEST <subject> <predicate> "<value>" [SINCE <ISO>] [UNTIL <ISO>] [CONF <0-1>]
```
Ingests a claim into the persistent memory engine.

| Field      | Required | Description |
|------------|----------|-------------|
| subject    | yes      | Entity identifier (e.g. `acme:ceo`) |
| predicate  | yes      | Relationship (e.g. `held_by`) |
| "value"    | yes      | Quoted string value |
| SINCE      | no       | Valid-time start (ISO date or datetime, e.g. `2020-01-01`) |
| UNTIL      | no       | Valid-time end |
| CONF       | no       | Confidence 0–1 (default 0.9) |

**SDK mapping:**
```python
engine.ingest_claim({
    "agent_id": agent_id,
    "subject": subject,
    "predicate": predicate,
    "value": value,
    "provenance": ProvenanceLabel.external_user_asserted(),
    "cardinality": "Functional",
    "valid_time": {"start": SINCE, "end": UNTIL, "valid_time_confidence": CONF},
    "confidence": {"value_confidence": CONF, "valid_time_confidence": CONF},
    "criticality": "Medium",
    "derived_from": [],
})
```

**Dispositions:** `CommittedCheap` (fast path), `Contested` (conflict), `Superseded`, etc.

**Example:**
```
INGEST acme:ceo held_by "Alice" SINCE 2020-01-01 CONF 0.95
INGEST acme:ceo held_by "Bob" SINCE 2023-03-15
```

---

### RECALL_REENTRY
```
RECALL_REENTRY <subject> <predicate> "<value>" <source_claim_ref>
```
Re-ingests a value the agent previously recalled from memory, using `RecallReEntry` provenance.
Tests/demonstrates the amplification firewall (C6 guard).

**SDK mapping:**
```python
engine.ingest_claim({
    "provenance": ProvenanceLabel.recall_re_entry(),
    "cardinality": "Functional",
    "valid_time": {"valid_time_confidence": 0.7},
    "confidence": {"value_confidence": 0.7, "valid_time_confidence": 0.7},
    "criticality": "Low",
    "derived_from": [source_claim_ref],
})
```
Then auto-queries to show `Belief unchanged (firewall held)`.

---

## Read Commands

### RECALL
```
RECALL <subject> <predicate> [valid=<ISO>] [tx=<ISO>]
```
Queries the current canonical belief (or historical belief at specified times). Applies agent rules R1-R5:
- R1: Committed → show value + valid_time + conf
- R2: Contested → surface BOTH values + suggest `/reconcile` (never guesses)
- R3: Superseded → show current value + suggest `/history`
- R5: No belief → "No memory … Use INGEST"

| Modifier | Effect |
|----------|--------|
| `valid=<ISO>` | Point-in-time query (world-history axis); omit to query now |
| `tx=<ISO>` | Transaction-time query (what-did-we-know axis); omit to query as-of-now |

**SDK mapping:**
```python
engine.query_memory({"agent_id": agent_id, "subject": subject, "predicate": predicate,
                     "point_in_time": valid_at, "as_of_tx_time": tx_time})
```

**Examples:**
```
RECALL acme:ceo held_by
RECALL acme:ceo held_by valid=2024-06
RECALL acme:ceo held_by valid=2024-06 tx=2025-02-01
```

---

## Inspector Commands

### /memory
Lists all claims ingested in the current session with their current belief status.

### /history \<subject\> \<predicate\>
Shows the full audit event timeline for all claims in the session matching the given subject/predicate pair.

**SDK mapping:** `engine.query_audit({..., "limit": 500})` filtered by claim_refs from session registry.

**Example:**
```
/history acme:ceo held_by
```

### /why \<subject\> \<predicate\>
Shows the disposition transition timeline for a subject/predicate pair (why does it have its current state).

### /contested
Lists all (subject, predicate) pairs whose current belief is `Contested` or `PendingConflict`.

### /audit [N]
Shows the last N raw audit ledger entries (default 10).
Entries contain: `entry_id`, `claim_ref`, `event_kind`, `disposition`, `rationale`, `recorded_at`.

### /reconcile \<subject\> \<predicate\>
Runs conflict adjudication for the given subject/predicate pair.
The engine promotes the highest-confidence non-contested claim and marks others as Superseded.

**SDK mapping:**
```python
engine.reconcile({"agent_id": agent_id, "subject_lines": [(subject, predicate)]})
```

### /review
Enters the human-in-the-loop adjudication UI for all `Queued`/`PendingReview` Contested claims.
Prompts for each pending pair: enter `c` (challenger wins), `i` (incumbent wins), or `s`/empty (defer).

**Example:**
```
/review
[/review] 2 pending adjudication(s):

  Incumbent: Alice | Challenger: Bob | Queued at 2025-02-15T10:30Z
  → [c]hallenger, [i]ncumbent, or [s]kip? c
  ✓ Affirmed challenger.

  Incumbent: Acme HQ/Austin | Challenger: Acme HQ/Boston | Queued at 2025-02-16T14:22Z
  → [c]hallenger, [i]ncumbent, or [s]kip? i
  ✓ Affirmed incumbent.
```

### /sweep
Expires all adjudications that have been queued longer than the oracle timeout (default 24h).
Reverts queued claims back to `Contested` (no decision made).

### /reset (CLI flag)
Delete the persistent per-agent database and exit. Use `--reset` as a command-line flag:
```bash
.venv/bin/python -m mempill_demo --reset
```

### /help
Prints the command grammar summary.

### /quit (also /q, /exit)
Exits the REPL.

---

## Audit Entry Shape

```python
{
    "entry_id":    "<uuid>",
    "agent_id":    "<string>",
    "claim_ref":   "<uuid>",          # links to the ingested claim
    "event_kind":  "ClaimCommitted" | "ValidityAsserted" | "AdjudicationResolved" | ...,
    "disposition": "CommittedCheap" | "Contested" | "Superseded" | ...,
    "rationale":   {...},             # engine-internal reason dict
    "recorded_at": "<ISO datetime>",
}
```

Note: audit entries do NOT carry `subject`, `predicate`, or `value`.
These are tracked via the session-local claim registry keyed by `claim_ref`.

---

## Provenance Labels

| Abbreviation | Wire shape | When used |
|---|---|---|
| USER | `{"type":"External","kind":"UserAsserted"}` | INGEST (default) |
| EXT  | `{"type":"External","kind":"ExternalFirstHand"}` | Scenario (first-hand) |
| RECALL | `{"type":"RecallReEntry"}` | RECALL_REENTRY command |
| LLM  | `{"type":"ModelDerived"}` | --llm mode inferred claims |

---

## Disposition Badge Map (presenter_rich.py)

| Disposition(s) | Badge | Color |
|---|---|---|
| Committed, CommittedCheap, CommittedInferred, Reinstated, Resolved | COMMITTED | green |
| Contested, PendingConflict | CONTESTED | yellow |
| Superseded, Invalidated | SUPERSEDED | grey/dim |
| Queued, QueuedForAdjudication, PendingReview | PENDING | cyan |
| PendingLowConfidence | LOW-CONF | yellow |
| Quarantined, Rejected | REJECTED | red |
