# mempill Multi-Agent Showcase — Run Guide

## Requirements

- **Python 3.12** (required — mempill ships an abi3 wheel targeting 3.12+)
- The `.venv` virtualenv at the repo root (set up by `uv` from `pyproject.toml`)
- **ANTHROPIC_API_KEY** required for the ReAct agent (scenario + Studio) — place in `.env`

### mempill dependency

mempill 0.4.0 (per-agent file storage: `open_for_agent` / `open_oracle_for_agent`) is
**not yet published on PyPI** — the `mempill>=0.4.0,<0.5` pin in `pyproject.toml` will fail to
resolve via the default `scripts/setup.sh` until it is published.

Until then, run `scripts/setup.sh --local-engine` — it builds and installs mempill from the
sibling `../mempill` repo instead of PyPI.

If you ever need to rebuild the venv from scratch, follow the project `README.md`
setup instructions.

---

## Architecture Overview

### Primary: Single free-form ReAct agent

The exec_assistant (legacy) uses a single `create_react_agent` (LangGraph) with 10 memory tools
and a mempill bi-temporal memory backend. There is no multi-agent topology —
one ReAct agent handles all question types directly via tool selection.

```
User question (free-form natural language)
       │
       ▼
 ┌─────────────────────────────────────────────────┐
 │  ReAct Agent  (claude-haiku-4-5-20251001)        │
 │                                                 │
 │  Tools:                                         │
 │    recall_subject   — ask about an entity       │
 │    recall_at        — point-in-time query       │
 │    recall_as_of     — tx-time query             │
 │    remember_fact    — write a new fact          │
 │    get_contested    — inspect conflicting facts │
 │    request_adjudication — HITL interrupt gate   │
 │    audit_trail      — compliance audit log      │
 │    list_pending_adjudications — view conflicts  │
 │    resolve_adjudication — human verdict gate    │
 │    query_history    — search fact history       │
 │                                                 │
 │  Capabilities:                                  │
 │    • Bounded valid-time intervals (valid_from  │
 │      + valid_until) on remember_fact           │
 │    • Role-holder modeling: org leadership seats │
 │      are org attributes (acme-corp/ceo), so    │
 │      re-appointments conflict with incumbents  │
 │    • Audit capped at 200 entries with cursor   │
 │      pagination (from_tx_time)                 │
 └─────────────┬───────────────────────────────────┘
               │  tool calls
               ▼
 ┌─────────────────────────────────────────────────┐
 │  mempill bi-temporal memory engine              │
 │                                                 │
 │  Stores every fact with:                        │
 │    valid_time  — when it was true in the world  │
 │    tx_time     — when it was recorded           │
 │    provenance  — who asserted it                │
 │    disposition — CommittedCheap / Contested /   │
 │                  QueuedForAdjudication          │
 └─────────────────────────────────────────────────┘
```

### HITL (Human-In-The-Loop)

When the agent writes a fact that conflicts with an existing belief on the same
valid-time window (`remember_fact` → `Contested` / `QueuedForAdjudication`), it
calls `request_adjudication`. That tool calls `LangGraph interrupt(payload)` —
the graph **pauses**. The human sends `Command(resume=verdict)` with one of:

- `Affirm` — challenger is correct; incumbent is superseded.
- `Deny` — incumbent survives; challenger is rejected.
- `Abstain` — defer; both remain Contested.
- Pasted candidate value (e.g. `Acme Corp / CTO`) → maps to Affirm or Deny automatically.

The oracle queue in mempill receives the verdict via `submit_adjudication()` and
resolves the conflict. Post-resolution recall returns `Resolved` status.

### Dual-agent router (mempill 0.4.0 showcase)

The dual-agent router (`people_ops_agent` and `org_registry_agent` in LangGraph Studio)
is an alternative topology that splits the executive-assistant workload along domain lines.
A lightweight structured-output classifier routes each natural-language query to the appropriate
agent: **people_ops_agent** handles facts about people (Alice Chen, Bob Liu, Jordan Park) and
their roles/attributes; **org_registry_agent** handles organizational facts (Acme Corp's CEO
seat and headquarters). Each agent maintains its own SQLite database (`agent_<id>.db` under
`.mempill/`), demonstrating mempill 0.4.0's per-agent storage feature. Ambiguous queries
default to people_ops_agent.

### Naive-vs-mempill contrast

The `NaiveAdapter` demonstrates what a simpler non-temporal store cannot do:
- No `query_at` (no bi-temporal) — asking about past state is impossible.
- No Contested detection — conflicting facts silently overwrite each other.
- No provenance — who said it and when is not recorded.
- No audit trail — compliance replay is structurally impossible.

`build_app_from_settings(Settings(naive_mode=True))` returns `(None, NaiveAdapter)` —
the LangGraph toolchain literally cannot be built without the bi-temporal API.

### mempill value demonstration

| Question | Without mempill | With mempill |
|---|---|---|
| "What was Alice's city in March 2024?" | Overwrites — no history | `recall_at(valid_at=2024-03)` → Austin TX |
| "Did we believe VP Eng before the CTO update?" | No tx-time axis | `recall_as_of(as_of_tx_time=<before>)` → VP Engineering |
| "Two sources conflict on Alice's title" | Silent overwrite | `Contested` → HITL → human resolves |
| "Show the provenance chain for compliance" | Impossible | `audit_trail` → full ledger |

---

## Using the venv

Either activate the venv:

```bash
source .venv/bin/activate
python -m mempill_showcase.scenarios.executive_assistant
```

Or call `.venv/bin/python` / the installed console scripts directly (no activation needed):

```bash
.venv/bin/python -m mempill_showcase.scenarios.executive_assistant
.venv/bin/mempill-showcase
```

All commands below assume the venv is active OR you prefix with `.venv/bin/`.

---

## Commands

### Run the 6-beat ReAct agent scenario

```bash
# As a module:
.venv/bin/python -m mempill_showcase.scenarios.executive_assistant

# As a console script (after editable install):
.venv/bin/mempill-showcase
```

Runs all 6 scenario beats (B-01..B-06) via the free-form ReAct agent:

| Beat | Description |
|---|---|
| B-01 | Ask-anything: Alice's dietary restriction (recall_subject) |
| B-02 | Point-in-time: Alice's city in March 2024 (recall_at → Austin TX) |
| B-03 | New fact write: alice-chen/travel_preference (remember_fact, CommittedCheap) |
| B-04 | Contested: "Alice is CTO since June 2023" → HITL interrupt → Affirm → CTO wins |
| B-05 | Confirm resolution: recall Alice's employer → CTO (Resolved) |
| B-06 | Compliance audit: audit_trail → full write event history |

**Requires `ANTHROPIC_API_KEY` in `.env` at the repo root.**

---

### Run the naive-vs-mempill comparison (4 contrasts)

```bash
.venv/bin/python -m mempill_showcase.scenarios.compare

# or:
.venv/bin/mempill-showcase-compare
```

Side-by-side comparison of NaiveAdapter vs MempillAdapter across the 4
"money-shot" contrast moments from the scenario.

---

### Run the compliance replay (T-08 audit report)

```bash
.venv/bin/python -m mempill_showcase.scenarios.compliance_replay

# or:
.venv/bin/mempill-showcase-audit
```

Runs the full scenario, captures real engine-stamped tx timestamps, then
answers: "What did the assistant believe about Alice Chen at the compliance
moment, and with what provenance?" Outputs:

- Belief state AS OF the captured tx time (before NYC write + CTO resolution)
- Current belief state (for contrast)
- Full audit ledger (query_audit output)
- AC-4 / AC-5 confirmation summary

The tx timestamp is **engine-stamped** (not injected). The narrative "decision
on date X" maps to a real captured timestamp — proving the bi-temporal axis
without faking past dates.

---

### Try it in Studio — interactive ReAct agent demo

LangGraph Studio lets you visualize and interactively run the showcase agents with a UI.

**Default model:** `claude-haiku-4-5-20251001` (short alias: `claude-haiku-4-5`; overridable via `ANTHROPIC_MODEL` in `.env`).

**Setup:**

1. Add your Anthropic API key to `.env` at the repo root (required for the ReAct LLM and router classifier):

   ```dotenv
   ANTHROPIC_API_KEY=sk-ant-...
   # Optional — override the default model:
   # ANTHROPIC_MODEL=claude-haiku-4-5-20251001
   ```

2. Start Studio (requires `langgraph-cli[inmem]` — an opt-in extra):

   ```bash
   pip install 'langgraph-cli[inmem]'
   .venv/bin/langgraph dev
   ```

   Studio opens at `http://127.0.0.1:2024` and prints a link to the Studio UI at
   `https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`.

3. Select one of four graphs from the Studio sidebar:

   | Graph | Purpose |
   |---|---|
   | **exec_assistant** | Legacy single-agent entry point (jordan-park-001 agent, all facts in one DB) |
   | **people_ops_agent** | Domain-scoped ReAct agent managing personnel facts (alice-chen, bob-liu, jordan-park) |
   | **org_registry_agent** | Domain-scoped ReAct agent managing organizational facts (acme-corp) |
   | **dual_agent_router** | Natural-language router that classifies queries and dispatches to people_ops or org_registry |

**What happens at startup:**

Day-0 facts are seeded automatically at graph module-import time (when `langgraph dev` starts):
- **exec_assistant:** 7 facts for agent `jordan-park-001` (legacy single agent with all domains)
- **people_ops_agent:** 6 facts for agent `people-ops-001` (person subjects only)
- **org_registry_agent:** 2 facts for agent `org-registry-001` (org subjects only: ceo + new hq_location)

For the dual-agent graphs, the in-memory store persists across turns within a single `langgraph dev` session.
To inspect per-agent databases on disk (persistent storage):

```bash
# After running dual_agent_router a few times, inspect the per-agent SQLite files:
sqlite3 .mempill/agent_people-ops-001.db "select subject, predicate, value from claims;" 
# Expected seed (6 rows):
#   alice-chen|employer|Acme Corp / VP Engineering
#   alice-chen|city|Austin TX
#   alice-chen|dietary_restriction|vegetarian
#   bob-liu|employer|Meridian Ventures / Partner
#   bob-liu|travel_preference|window seat, no checked bags
#   jordan-park|preferred_hotel|Marriott Bonvoy Gold

sqlite3 .mempill/agent_org-registry-001.db "select subject, predicate, value from claims;"
# Expected seed (2 rows):
#   acme-corp|ceo|Diane Foster
#   acme-corp|hq_location|Austin, TX

# Inspect which agent wrote each claim:
sqlite3 .mempill/agent_people-ops-001.db "select distinct agent_id from claims;"
# Output: people-ops-001
```

This per-agent storage is mempill 0.4.0's headline: one SQLite file per agent, each isolated and auditable.

**How to use the Input form:**

Fill in only the **messages** field as a natural-language question. The agent
normalises entity names (e.g. "Alice Chen" → "alice-chen") and selects tools
automatically.

See `STUDIO_DEMO.md` for the full turn-by-turn script with exact inputs.

---

### Run the deterministic test suite

```bash
.venv/bin/python -m pytest -m "not live" -q
```

Runs all non-live tests (no API key required). Expected result: **407 passed** (includes console + showcase + router tests).

```bash
.venv/bin/python -m pytest -m live -q
```

Runs live semantic E2E tests (requires `ANTHROPIC_API_KEY`). Expected: **26 passed**.

Live tests covered:

| Test | What it asserts |
|---|---|
| LIVE-A | Dietary query → "vegetarian", no spurious interrupt |
| LIVE-B | Role query (no alias map) → employer fact answered linguistically |
| LIVE-C | Novel attribute write + read-back (bob-liu/preferred_airline) |
| LIVE-D | Contested → HITL interrupt → Affirm → recall returns CTO |
| LIVE-E | Step-aside (Acme CEO query) → Diane Foster |
| LIVE-F | Point-in-time valid_at (city early 2024 → Austin TX) |
| LIVE-G | Audit trail returns event descriptions |

---

### LangSmith tracing and Anthropic API key (`.env` auto-load)

The CLI entry points (`mempill-showcase`, `mempill-showcase-compare`,
`mempill-showcase-audit`) automatically load a `.env` file from the current
working directory at startup.

**Supported `.env` keys:**

```dotenv
# Anthropic (required for the ReAct agent LLM)
ANTHROPIC_API_KEY=sk-ant-...

# Optional: override the default model
ANTHROPIC_MODEL=claude-haiku-4-5-20251001

# LangSmith observability (all optional — tracing is a no-op without a key)
LANGSMITH_API_KEY=ls__...        # your LangSmith API key
LANGSMITH_TRACING=true           # set to true to enable trace export
LANGSMITH_PROJECT=mempill-demo   # project name in LangSmith UI (default: mempill-showcase)
```

---

### NAIVE_MODE toggle

Flip the whole showcase to the NaiveAdapter to watch it misbehave:

```bash
NAIVE_MODE=true .venv/bin/python -m mempill_showcase.scenarios.compare
```

With `NAIVE_MODE=true`:
- `build_app_from_settings()` returns `(None, NaiveAdapter)` instead of `(app, MempillAdapter)`
- The NaiveAdapter has no `query_at` (no bi-temporal), no Contested detection,
  no provenance — all 4 contrasts demonstrate failure.

```python
from mempill_showcase.config.di import build_app_from_settings
from mempill_showcase.config.settings import Settings

# mempill (default):
app, adapter = build_app_from_settings(Settings(naive_mode=False))

# naive — adapter is NaiveAdapter; app is None (bi-temporal tools can't be built):
app, adapter = build_app_from_settings(Settings(naive_mode=True))
assert app is None
assert not hasattr(adapter, "query_at")
```

---

## Console scripts reference

| Script | Module | Purpose |
|---|---|---|
| `mempill-showcase` | `mempill_showcase.scenarios.executive_assistant:main` | 6-beat ReAct scenario |
| `mempill-showcase-compare` | `mempill_showcase.scenarios.compare:main` | 4 naive-vs-mempill contrasts |
| `mempill-showcase-audit` | `mempill_showcase.scenarios.compliance_replay:main` | Compliance replay |
| `mempill-console` | `mempill_demo.__main__:main` | Legacy console demo |

Scripts are registered in `pyproject.toml [project.scripts]` and installed by:

```bash
.venv/bin/uv pip install -e . --no-deps
```

`--no-deps` is required to avoid pulling `mempill` from PyPI.
