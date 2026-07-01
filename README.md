# mempill-demo

A runnable demonstration of mempill's bi-temporal memory engine inside a multi-agent system.
The primary deliverable is `mempill_showcase` — a production-style reference app showing
a single free-form ReAct agent (`create_react_agent`) with 7 memory tools + a thin LangGraph
shell for durable HITL `interrupt()`, backed by mempill bi-temporal memory, with a naive
last-write-wins adapter (`NaiveAdapter`) for contrast and a full compliance audit replay.

---

## Quickstart — mempill_showcase

**Python 3.12 required.** mempill is installed from a local prerelease wheel (0.2.1, built
from source — not on PyPI). `scripts/setup.sh` handles this.

```bash
git clone <this-repo> mempill-demo
cd mempill-demo
bash scripts/setup.sh        # creates .venv, installs prerelease mempill wheel + all extras
```

Run the 6-beat executive-assistant scenario (B-01..B-06) via the free-form ReAct agent:

```bash
.venv/bin/mempill-showcase       # full scenario: ingestion, conflict, HITL, bi-temporal recall
.venv/bin/mempill-showcase-compare  # mempill vs naive side-by-side (4 money-shot contrasts)
.venv/bin/mempill-showcase-audit    # compliance audit: tx-time replay + full provenance ledger
```

Run the test suite (304 deterministic + 12 live tests):

```bash
.venv/bin/python -m pytest -m "not live" -q
# Expected: 304 passed
```

See [SHOWCASE.md](SHOWCASE.md) for the full architecture, 6-beat scenario walkthrough,
design decisions, and bi-temporal query examples.

---

## Architecture (one line)

A single free-form ReAct agent (`create_react_agent`) with 7 memory tools + a thin LangGraph
shell for durable HITL `interrupt()`, backed by mempill bi-temporal memory;
naive-vs-mempill adapter toggle via `NAIVE_MODE=true`.

---

## Console agent (mempill_demo)

A simpler REPL demonstrating mempill's three core acts (temporal validity, contested conflict,
amplification firewall) without any multi-agent framework:

```bash
.venv/bin/python -m mempill_demo --scenario    # auto-play 3-act story then REPL
.venv/bin/python -m mempill_demo --selftest    # deterministic assertion suite (13 checks)
.venv/bin/mempill-console                      # interactive REPL
```

---

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `NAIVE_MODE` | `false` | `true` → use NaiveAdapter (last-write-wins, no bi-temporal) |
| `ANTHROPIC_API_KEY` | — | Required for the free-form ReAct agent LLM (tool-calling) |
| `ANTHROPIC_MODEL` | `claude-haiku-4-5-20251001` | Anthropic model for the ReAct agent |
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

## Prerelease wheel note

mempill is installed from the **local source-built wheel** (version 0.2.1, prerelease — not on
PyPI) — includes `valid_at` + granularity + oracle HITL features. `scripts/setup.sh` installs
it automatically from the sibling repo.

To rebuild the wheel manually:
```bash
cd ../mempill/mempill-python && ./.venv/bin/maturin build --release
```
