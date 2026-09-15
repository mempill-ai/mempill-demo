# mempill-demo

A runnable demonstration of mempill's bi-temporal memory engine inside a multi-agent system.
The primary deliverable is `mempill_showcase` — a reference app showing
a single free-form ReAct agent (`create_react_agent`) with 10 memory tools + a thin LangGraph
shell for durable HITL `interrupt()`, backed by mempill bi-temporal memory, with a naive
last-write-wins adapter (`NaiveAdapter`) for contrast and a full compliance audit replay.

---

## Quickstart — mempill_showcase

**Python 3.12 required.** mempill 0.4.0 is not yet published on PyPI, so setup requires
`--local-engine` (builds mempill from a sibling `../mempill` checkout) — see
[mempill dependency](#mempill-dependency) below for details.

```bash
git clone <this-repo> mempill-demo
cd mempill-demo
bash scripts/setup.sh --local-engine   # creates .venv, builds + installs mempill from ../mempill + LangGraph/pytest
```

Run the 6-beat executive-assistant scenario (B-01..B-06) via the free-form ReAct agent:

```bash
.venv/bin/mempill-showcase       # full scenario: ingestion, conflict, HITL, bi-temporal recall
.venv/bin/mempill-showcase-compare  # mempill vs naive side-by-side (4 money-shot contrasts)
.venv/bin/mempill-showcase-audit    # compliance audit: tx-time replay + full provenance ledger
```

Run the test suite (407 deterministic + 26 live tests):

```bash
.venv/bin/python -m pytest -m "not live" -q
# Expected: 407 passed
```

See [SHOWCASE.md](SHOWCASE.md) for the full architecture, 6-beat scenario walkthrough,
design decisions, and bi-temporal query examples.

---

## Architecture

**Primary (mempill_showcase):** A single free-form ReAct agent (`create_react_agent`) with 10 memory tools (7 core + 3 adjudication/history tools) + a thin LangGraph
shell for durable HITL `interrupt()`, backed by mempill bi-temporal memory;
naive-vs-mempill adapter toggle via `NAIVE_MODE=true`.

**Dual-agent router (mempill_showcase Studio graphs):** A natural-language router entry point (`dual_agent_router`) that classifies incoming questions
and dispatches to two isolated ReAct agents — `people_ops_agent` (handles person facts: Alice, Bob, Jordan's colleagues and roles)
and `org_registry_agent` (handles organization facts: Acme Corp's leadership and structure). Each agent owns its own per-agent
SQLite database file under `.mempill/`, making mempill 0.4.0's per-agent storage visible and auditable. Ambiguous queries
default-route to `people_ops_agent`. Router input hardening (PR #57) coerces junk message entries before routing, multi-turn
state accumulates across turns safely, and invalid agent IDs exit with a clean error.

---

## Console agent (mempill_demo)

A simpler REPL demonstrating mempill's three core acts (temporal validity, contested conflict,
amplification firewall) without any multi-agent framework:

```bash
.venv/bin/python -m mempill_demo --scenario    # auto-play 3-act story then REPL
.venv/bin/python -m mempill_demo --selftest    # deterministic assertion suite (13 checks)
.venv/bin/mempill-console                      # interactive REPL
.venv/bin/python -m mempill_demo --db-dir .mempill --agent my-agent  # custom per-agent DB dir
```

> **Migration note (mempill 0.4.0):** the console agent's `--db` flag (a single file path) was
> replaced by `--db-dir` (a base directory). The actual database file is now derived
> automatically as `<db-dir>/agent_<agent_id>.db`, one file per agent. Pre-0.4.0 database files
> are **not** auto-migrated — to keep existing data, move/rename the old shared-file database to
> `<db-dir>/agent_<agent_id>.db` before first use with 0.4.0.

### Date Granularity + Honest Display (0.4.0 read-path)

The `query_history` tool surfaces facts with their original date granularity. A fact ingested as "September 2024" is never fabricated as "September 1, 2024". See [STUDIO_DEMO.md](STUDIO_DEMO.md#date-granularity--honest-display-on-history-04-headline-read-path-feature) for a walkthrough (fresh thread, 3-step CEO succession scenario).

---

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `NAIVE_MODE` | `false` | `true` → use NaiveAdapter (last-write-wins, no bi-temporal) |
| `MEMPILL_AGENT_ID` | `jordan-park-001` | Agent/session owner ID for the single-agent exec_assistant. Ignored by dual-agent router graphs (which use per-graph agent IDs: `people-ops-001`, `org-registry-001`). |
| `MEMPILL_DB_DIR` | — (in-memory) | Base directory for persistent SQLite engines; each agent's file is derived as `MEMPILL_DB_DIR/agent_{AGENT_ID}.db`. For dual-agent router, defaults to `./.mempill/` (not in-memory) to persist separate per-agent databases. |
| `ANTHROPIC_API_KEY` | — | Required for the free-form ReAct agent LLM (tool-calling) |
| `ANTHROPIC_MODEL` | `claude-haiku-4-5-20251001` | Anthropic model for the ReAct agent and router classifier |
| `LANGSMITH_API_KEY` | — | Enables LangSmith tracing; absent → no-op |
| `LANGSMITH_TRACING` | — | `true` to force-enable tracing |
| `LANGSMITH_PROJECT` | `mempill-showcase` | LangSmith project name |

Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY` for live runs.
All settings are also exposed via `mempill_showcase.config.settings.Settings`
(pydantic-settings, env-file aware) and wired into the observability layer.

---

## Optional: MCP integration

The MCP demo requires the sibling repo `../mempill/mempill-mcp/` (pure Python, not on PyPI).
`scripts/setup.sh` path-installs it when the sibling is present.

```bash
.venv/bin/python mcp/verify_stdio.py   # MCP stdio client verification
```

See the MCP section in this file (below) for Claude Desktop / Claude Code setup.

---

## MCP integration (detail)

### Verify the MCP server

```bash
.venv/bin/python mcp/verify_stdio.py
```

Launches `mempill-mcp` via stdio, handshakes, lists 4 tools, calls `ingest_claim` +
`query_memory`, asserts the round-trip, prints `[VERIFIED]`. Exit code 0 = success.

### Connect to Claude Desktop

1. Copy `mcp/claude_desktop_config.json.example` to:
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`
2. Replace `/ABSOLUTE/PATH/TO/mempill-demo` with the absolute path to this repo
3. Fully quit and restart Claude Desktop

### Connect to Claude Code

1. Copy `mcp/.mcp.json.example` to `.mcp.json` at the repo root
2. Replace `/ABSOLUTE/PATH/TO/mempill-demo` with the absolute path
3. Set env vars and launch Claude Code from this directory

### Available MCP tools

| Tool | Description |
|---|---|
| `ingest_claim` | Write a belief claim to the engine |
| `query_memory` | Read the canonical belief for a (subject, predicate) pair |
| `reconcile` | Resolve conflicts for a set of subject lines |
| `audit` | Inspect the claim history ledger |

---

## Repository layout

```
mempill-demo/
  src/
    mempill_showcase/    ← PRIMARY: reference app (LangGraph ReAct agent + mempill bi-temporal memory)
    mempill_demo/        ← simpler console REPL (3-act mempill demo)
  tests/                 ← root-level tests for mempill_demo + edge cases
  scripts/setup.sh       ← creates .venv, installs prerelease mempill wheel
  examples/temporal_validity.py  ← standalone 3-act demo
  mcp/                   ← MCP stdio client + config examples
  SHOWCASE.md            ← full showcase architecture and walkthrough
  pyproject.toml
```

---

## mempill dependency

mempill 0.4.0 (per-agent file storage via `open_for_agent` / `open_oracle_for_agent`) is
**not yet published on PyPI** — it is deliberately unpublished until this migration is fully
verified. Until it is published, `scripts/setup.sh --local-engine` is **required**: it builds
and installs mempill from the sibling `../mempill` repo instead of PyPI. The default
`scripts/setup.sh` (no flag) resolves the `mempill>=0.4.0,<0.5` pin in `pyproject.toml` against
PyPI and will fail until 0.4.0 is published — do not use it yet.
