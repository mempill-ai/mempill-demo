# mempill Console Agent — Command Grammar

All commands are keyword-insensitive (INGEST, ingest, Ingest all work).
Slash commands must start with `/`.

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
RECALL <subject> <predicate>
```
Queries the current canonical belief. Applies agent rules R1-R5:
- R1: Committed → show value + valid_time + conf
- R2: Contested → surface BOTH values + suggest `/reconcile` (never guesses)
- R3: Superseded → show current value + suggest `/history`
- R5: No belief → "No memory … Use INGEST"

**SDK mapping:**
```python
engine.query_memory({"agent_id": agent_id, "subject": subject, "predicate": predicate})
```

**Example:**
```
RECALL acme:ceo held_by
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

## Disposition Badge Map (panel.py)

| Disposition(s) | Badge | Color |
|---|---|---|
| Committed, CommittedCheap, CommittedInferred, Reinstated, Resolved | COMMITTED | green |
| Contested, PendingConflict | CONTESTED | yellow |
| Superseded, Invalidated | SUPERSEDED | grey/dim |
| Queued, QueuedForAdjudication, PendingReview | PENDING | cyan |
| PendingLowConfidence | LOW-CONF | yellow |
| Quarantined, Rejected | REJECTED | red |
