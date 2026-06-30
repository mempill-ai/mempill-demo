# mempill Multi-Agent Showcase — Run Guide

## Requirements

- **Python 3.12** (required — the prerelease mempill wheel is an abi3 build targeting 3.12+)
- The `.venv` virtualenv at the repo root (set up by `uv` from `pyproject.toml`)
- **No API key needed** for all scenario and test commands below

### Prerelease wheel note

`mempill 0.3.0` is not yet on PyPI. The version in `.venv` is a locally built
wheel installed from the sibling `mempill/` repo. Do NOT run `uv sync` or
`pip install mempill` — that would pull an older `0.2.x` release from PyPI
which lacks `query_at` / `as_of_tx_time` / `valid_from_display` support.

If you ever need to rebuild the venv from scratch, follow the project `README.md`
setup instructions (which path-install the prerelease wheel first).

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

### Run the 8-beat scenario

```bash
# As a module (always works):
.venv/bin/python -m mempill_showcase.scenarios.executive_assistant

# As a console script (after editable install):
.venv/bin/mempill-showcase
```

Runs all 8 scenario beats (T-01..T-08) deterministically. No LLM calls,
no API key. Prints a Rich beat summary table.

---

### Run the naive-vs-mempill comparison (4 contrasts)

```bash
.venv/bin/python -m mempill_showcase.scenarios.compare

# or:
.venv/bin/mempill-showcase-compare
```

Side-by-side comparison of NaiveAdapter vs MempillAdapter across the 4
"money-shot" contrast moments from SCENARIO.md §4.

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

### Try it in Studio — interactive demo recipe

LangGraph Studio lets you visualize and interactively run the supervisor→crews→hitl
graph with a UI.

**Default model:** `claude-haiku-4-5` (overridable via `ANTHROPIC_MODEL` in `.env`).

**Setup:**

1. Add your Anthropic API key to `.env` at the repo root to enable the full LLM
   routing + extraction path (required for free-form input to work correctly):

   ```dotenv
   ANTHROPIC_API_KEY=sk-ant-...
   # Optional — override the default model:
   # ANTHROPIC_MODEL=claude-haiku-4-5
   ```

   With `ANTHROPIC_API_KEY` set, the graph uses:
   - `LLMSupervisor` (claude-haiku-4-5) for intent classification, and
   - `LLMExtractor` (single structured call) to parse entity/predicate/value/valid_from
     from free-form sentences before writing to mempill.

   Without the key, `MockSupervisor` (deterministic keyword routing) + shell heuristics
   are used — the demo still works for standard inputs but may misroute unusual phrasing.

2. Start Studio:

   ```bash
   .venv/bin/langgraph dev
   ```

   Studio opens at `http://127.0.0.1:2024` and prints a link to the Studio UI at
   `https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`.

3. Select the **exec_assistant** graph in the Studio sidebar.

**What happens at startup:**

The graph module seeds 7 Day-0 facts for agent `jordan-park-001` automatically
(Alice Chen: Austin TX, Acme Corp / VP Engineering, vegetarian; Bob Liu: Meridian
Ventures / Partner, travel prefs; Acme Corp CEO Diane Foster; Jordan Park hotel).
The in-memory store persists across turns within a single `langgraph dev` session.

**How to use the Input form:**

Fill in **only the `User Input` field** — leave all other fields blank.
`agent_id` defaults to `jordan-park-001` automatically.

**Verified demo sequence (with ANTHROPIC_API_KEY):**

| Turn | User Input | Expected behaviour |
|------|-----------|-------------------|
| 1 | `What's Alice Chen's current city?` | crew_c RECALL_HISTORY → `Austin TX` (seeded Day-0) |
| 2 | `Alice moved to New York in February 2025` | LLMSupervisor → UPDATE_CONTACT → crew_a: LLMExtractor extracts `alice-chen/city=New York valid_from=2025-02` → mempill writes CommittedCheap succession; current belief becomes `New York` |
| 3 | `What's Alice Chen's city now?` | crew_c → `New York` (succession committed in turn 2) |
| 4 | `What was Alice's city in Q1 2024?` | crew_c RECALL_HISTORY → bi-temporal query at 2024-06-01 → `Austin TX` (before the move) |
| 5 | `Alice is now CTO of Acme` | LLMExtractor maps to `alice-chen/employer=Acme Corp / CTO` (no date → overlaps open-ended VP Engineering belief) → **Contested** → graph routes to `hitl_node` |

**LLM extraction details:**

`LLMExtractor` makes one structured Anthropic API call per UPDATE_CONTACT turn to
extract `{entity, predicate, value, valid_from}` from free-form text. The Python
`mempill_remember` tool then writes to mempill (reliable — no agent tool-loops).

Examples verified with `claude-haiku-4-5`:
- `"Alice moved to New York in February 2025"` → `alice-chen / city = "New York" @ 2025-02` → CommittedCheap (clean succession over Austin TX 2023-06..2025-02)
- `"Alice is now CTO of Acme"` → `alice-chen / employer = "Acme Corp / CTO" @ null` → Contested vs VP Engineering (no date → overlapping) → HITL
- Bi-temporal: `query_at(alice-chen, city, valid_at=2024-06-01)` → `"Austin TX"` ✓

**Handling Contested writes (HITL interrupt):**

When turn 5 triggers a conflict, Studio shows an **Interrupts** panel on the right.
The interrupt payload describes the incumbent belief vs. the challenger claim.
To resume:

1. Open the **Interrupts** panel.
2. Enter a verdict in the resume field: `Affirm` (accept the new claim), `Deny`
   (keep the incumbent), or `Abstain` (leave it Contested).
3. Click **Submit**. The graph resumes at `hitl_node`, resolves the belief, and
   sets `hitl_verdict` in the final state.

**Notes:**

- Writes within a session accumulate — turn 2's NYC write is visible in turn 3.
- The store resets when you restart `langgraph dev` (in-memory only).
- Stop Studio with `Ctrl-C`.

The `langgraph.json` manifest at the repo root points Studio to:
```
src/mempill_showcase/frameworks/langgraph/studio_graph.py:graph
```

---

### Run the deterministic test suite

```bash
.venv/bin/python -m pytest src/mempill_showcase/tests/ -v -m "not live"
```

Runs all non-live tests (no API key required). Expected result: **255 passed**.

Tests are located in `src/mempill_showcase/tests/`. The `-m "not live"` flag
excludes tests that require `ANTHROPIC_API_KEY`.

---

### LangSmith tracing and Anthropic API key (`.env` auto-load)

The CLI entry points (`mempill-showcase`, `mempill-showcase-compare`,
`mempill-showcase-audit`) automatically load a `.env` file from the current
working directory at startup. This means you can place your LangSmith and
Anthropic credentials in a `.env` file at the repo root and they will be
picked up without any shell export.

**Supported `.env` keys:**

```dotenv
# LangSmith observability (all optional — tracing is a no-op without a key)
LANGSMITH_API_KEY=ls__...        # your LangSmith API key
LANGSMITH_TRACING=true           # set to true to enable trace export
LANGSMITH_PROJECT=mempill-demo   # project name in LangSmith UI (default: mempill-showcase)

# Anthropic (required only for the live LLM supervisor — MockSupervisor is used otherwise)
ANTHROPIC_API_KEY=sk-ant-...
```

When `LANGSMITH_API_KEY` and `LANGSMITH_TRACING=true` are present, running any
CLI scenario will produce LangSmith traces in the named project. Each scenario
run generates:

- A top-level LangGraph run with all graph nodes as child spans.
- `mempill.remember`, `mempill.recall`, `mempill.audit` tool spans (tagged
  with `run_type="tool"`).
- `mempill.contested` spans whenever a Functional write triggers a conflict
  (shows the incumbent vs. challenger values in the LangSmith UI).

The `.env` load is handled by `mempill_showcase.config.bootstrap.bootstrap()`,
which is called only inside CLI `main()` functions — never at module import
time. Test code that imports scenario or tool modules will never accidentally
pick up a developer's local `.env`.

**Shell env vars take precedence over `.env`:** if `LANGSMITH_API_KEY` is
already set in your shell, the `.env` value is ignored (dotenv `override=False`).

---

### NAIVE_MODE toggle

Flip the whole showcase to the NaiveAdapter to watch it misbehave:

```bash
NAIVE_MODE=true .venv/bin/python -m mempill_showcase.scenarios.compare
```

Or set it in a `.env` file at the repo root:

```
NAIVE_MODE=true
```

With `NAIVE_MODE=true`:
- `build_app_from_settings()` returns `(None, NaiveAdapter)` instead of
  `(app, MempillAdapter)`
- The NaiveAdapter has no `query_at` (no bi-temporal), no Contested detection,
  no provenance — all 4 contrasts demonstrate failure

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
| `mempill-showcase` | `mempill_showcase.scenarios.executive_assistant:main` | 8-beat scenario |
| `mempill-showcase-compare` | `mempill_showcase.scenarios.compare:main` | 4 naive-vs-mempill contrasts |
| `mempill-showcase-audit` | `mempill_showcase.scenarios.compliance_replay:main` | T-08 compliance replay |
| `mempill-console` | `mempill_demo.__main__:main` | Legacy console demo |

Scripts are registered in `pyproject.toml [project.scripts]` and installed by:

```bash
.venv/bin/uv pip install -e . --no-deps
```

`--no-deps` is required to avoid pulling `mempill` from PyPI.
