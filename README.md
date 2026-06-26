# mempill-demo

A runnable demonstration of mempill's temporal-validity memory engine and its MCP integration. Runs entirely offline — no LLM, no vector DB, no network. All intelligence is structural.

**Honest status:** this demo uses the local mempill wheel built from the private sibling repo (`mempill/`). It depends on an unpublished Rust/PyO3 package built via maturin. The demo will not run without the sibling repo present (see layout below).

---

## What it demonstrates

**The temporal-validity problem:** AI agents read their own earlier outputs and re-ingest them as if they were fresh evidence. Without guardrails, agent memory becomes a self-reinforcing echo chamber where the agent's confidence in stale facts grows over time with no new information.

**mempill's structural solution (3 acts):**

1. **Temporal validity** — beliefs are bounded in time (`valid_time`). The engine tracks WHEN something was true, not just WHAT was true.
2. **Contested conflict, not silent overwrite** — when two Functional claims overlap temporally (e.g., Alice then Bob as CEO), mempill flags them as Contested and requires explicit reconciliation. History is never silently overwritten.
3. **Amplification firewall** — an agent can ingest the same recall-re-entry claim 808 times; the engine does not treat repetition as corroboration. The belief after 808 re-ingests is identical to after 1.

---

## Repository layout

```
mempill-demo/             ← this repo
  pyproject.toml
  scripts/setup.sh
  examples/temporal_validity.py   # 3-act demo
  mcp/verify_stdio.py         # real MCP stdio client verification
  mcp/claude_desktop_config.json.example
  mcp/.mcp.json.example
  README.md
  .gitignore

../mempill/               ← sibling repo (required; not included here)
  mempill-python/         # Rust/PyO3 wheel source
  mempill-mcp/            # FastMCP server (pure Python)
```

The two repos must be checked out side by side under the same parent directory.

---

## Prerequisites

- **Rust toolchain** (stable): https://rustup.rs
- **Python 3.11+**
- **uv** (Python package manager): https://docs.astral.sh/uv/

Rust is required to build the mempill wheel (Rust/PyO3 via maturin). The build happens automatically during setup via PEP-517.

---

## Setup

```bash
cd mempill-demo
bash scripts/setup.sh
```

The setup script:
1. Creates a `.venv` (if absent)
2. Builds and installs the mempill wheel from `../mempill/mempill-python/` (PEP-517 / maturin — ~1-2 min on first run)
3. Installs `mempill-mcp` (editable) from `../mempill/mempill-mcp/`
4. Installs `mcp>=1.9,<2`
5. Installs `mempill-demo` (editable)
6. Installs LangGraph conversational-agent dependencies by DEFAULT

For a lean install without LangGraph:
```bash
SKIP_LANGGRAPH=true bash scripts/setup.sh
```

Verify the install:
```bash
uv run python -c "import mempill, mempill_mcp, mcp; print('imports OK')"
```

---

## Run the demo

```bash
uv run python examples/temporal_validity.py
```

The demo runs all 3 acts and prints the actual engine output with inline narration explaining the temporal-validity behavior.

---

## MCP integration

### Verify the MCP server (stdio client)

```bash
uv run python mcp/verify_stdio.py
```

This launches `mempill-mcp` as a subprocess via stdio, performs the MCP handshake, lists all 4 tools, calls `ingest_claim` + `query_memory`, asserts the round-trip, and prints `[VERIFIED]`.

Exit code 0 = success. Use `--verify` flag for CI-friendly exit code mode.

### Connect to Claude Desktop

The config uses the demo venv's Python directly (not `uv run`) because `mempill-mcp` depends on the locally-built mempill wheel which is not on PyPI.

1. Copy `mcp/claude_desktop_config.json.example` to:
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`
2. Replace `/ABSOLUTE/PATH/TO/mempill-demo` with the absolute path to this repo
3. Fully quit and restart Claude Desktop (Cmd+Q on macOS, not just close window)
4. Look for the mempill tools in the Claude Desktop tool picker

Logs: `~/Library/Logs/Claude/mcp*.log` (macOS)

### Connect to Claude Code

1. Copy `mcp/.mcp.json.example` to `.mcp.json` at the repo root
2. Replace `/ABSOLUTE/PATH/TO/mempill-demo` with the absolute path to this repo
3. Set env vars in your shell:
   ```bash
   export MEMPILL_AGENT_ID="my-agent"
   export MEMPILL_DB_PATH="/path/to/demo.db"  # optional; omit for in-memory
   ```
4. Launch Claude Code from this directory — it picks up `.mcp.json` automatically

`.mcp.json` is in `.gitignore` (personal paths stay local). The `.example` file is tracked.

### Available MCP tools

| Tool | Description |
|---|---|
| `ingest_claim` | Write a belief claim to the engine |
| `query_memory` | Read the canonical belief for a (subject, predicate) pair |
| `reconcile` | Resolve conflicts for a set of subject lines |
| `audit` | Inspect the claim history ledger |

---

## Interactive Console Agent

A mempill-aware REPL agent with a rich memory panel, persistent file-backed storage, and an optional LLM extraction layer.

### Entry point

```bash
# 3-act auto-play scenario then REPL
uv run python -m mempill_demo --scenario

# Plain REPL (deterministic grammar)
uv run python -m mempill_demo

# Assertion suite for CI (no API key required)
uv run python -m mempill_demo --selftest

# LLM-backed natural language parsing (requires ANTHROPIC_API_KEY)
# Copy .env.example to .env and set ANTHROPIC_API_KEY (or export it in your shell).
# Optionally set MEMPILL_MODEL to override the default model.
uv run python -m mempill_demo --llm

# Delete the persistent DB and exit
uv run python -m mempill_demo --reset
```

### Flags

| Flag | Description |
|------|-------------|
| `--db PATH` | File-backed DB path (default: `.mempill/console.db`) |
| `--agent AGENT_ID` | Agent ID string (default: `console-user`) |
| `--llm` | Enable LLM extraction via Claude (requires `ANTHROPIC_API_KEY`) |
| `--scenario` | Auto-play the 3-act story then hand off to REPL |
| `--selftest` | Run deterministic assertion suite; exit 0 on pass |
| `--reset` | Delete the DB file and exit |

### Quickstart

```
$ uv run python -m mempill_demo
mempill Console Agent  [deterministic]
Type /help for commands, /quit to exit.

mempill> INGEST acme:ceo held_by "Alice" SINCE 2020-01-01
Ingested: acme:ceo held_by = "Alice"
  disposition: CommittedCheap
  claim_ref:   c97b91b2...

mempill> INGEST acme:ceo held_by "Bob" SINCE 2023-03-15
Ingested: acme:ceo held_by = "Bob"
  disposition: Contested
  contested_with: ['c97b91b2...']
  [!] Contested — use /reconcile to resolve.

mempill> /reconcile acme:ceo held_by
Reconcile acme:ceo held_by:
  f27bbdbc...  → CommittedCheap

mempill> RECALL acme:ceo held_by
Memory: acme:ceo held_by = "Bob" (conf 0.90)
  status: Resolved  ref: f27bbdbc...
```

### --scenario output highlights

The auto-play scenario demonstrates:
- **CONTESTED** badge — two open-ended Functional claims overlap
- **SUPERSEDED** in audit — Alice's claim bounded by reconcile
- **FIREWALL HELD** — `RecallReEntry` × 5 does not alter the belief

### --selftest CI usage

```bash
uv run python -m mempill_demo --selftest && echo "CI: selftest passed"
```

Exit 0 = all 13 assertions passed (T1–T7). No API key required.
The selftest uses `open_in_memory()` — never touches the persistent DB.

### --llm setup

Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY` (or export it in your shell). The app auto-loads `.env` on startup — no manual `source` needed. Deterministic mode (`--selftest`, plain REPL, `--scenario`) requires no key.

Optionally set `MEMPILL_MODEL` to override the default model (e.g. `MEMPILL_MODEL=claude-haiku-4-5` for a cheaper run).

### --llm no-key guard

```bash
# No key set and no .env:
$ uv run python -m mempill_demo --llm
ERROR: --llm requires ANTHROPIC_API_KEY to be set in the environment.
# exits 1
```

### Grammar reference

See `console/GRAMMAR.md` for the full command grammar with SDK mappings.

---

## LangGraph conversational agent

A natural-language CHAT agent that uses mempill as long-term memory. The LLM replies naturally each turn while reading and writing the mempill memory store — multi-turn history, contested-belief surfacing, structured-output extraction (no `json.loads`).

### Setup (included by default)

The LangGraph dependencies are installed by default during `bash scripts/setup.sh`. If you ran a lean setup with `SKIP_LANGGRAPH=true`, install them now:

```bash
uv pip install -e ".[langgraph]"
```

Set your API key in `.env` or the environment (runtime-only; not needed during setup):

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...
# Optional — override the default model
MEMPILL_MODEL=claude-sonnet-4-6
```

**Note:** `SKIP_LANGGRAPH` and `INSTALL_LANGGRAPH` are setup-time flags only — they do not belong in `.env` and have no effect at runtime. Only `ANTHROPIC_API_KEY` is needed at runtime.

### Run the agent

```bash
uv run python -m mempill_langgraph
```

If `ANTHROPIC_API_KEY` is not set:

```
ERROR: ANTHROPIC_API_KEY not set. Add it to .env or export it.
# exits 1
```

### What it demonstrates

- **Read before speak** — `retrieve_memory` queries mempill before the LLM replies.
- **Contested surfacing** — if the queried belief is Contested, both claims are formatted into the system prompt. The LLM is instructed never to pick one and to suggest `/reconcile`.
- **Write after speak** — `write_memory` uses `llm.with_structured_output(ClaimExtractResult)` (Claude tool-use API) to extract factual claims from the AI reply and ingest them into mempill with `ModelDerived` provenance. Greetings and questions produce an empty claim list — never an error.
- **Amplification firewall** — if the AI reply merely restates a value already in memory (recall re-entry), the node detects it and skips the ingest to prevent self-amplification.
- **Multi-turn history** — LangGraph's `MemorySaver` checkpointer restores the full message history on every turn via `thread_id`.

### Offline tests (no API key)

```bash
uv run pytest tests/test_langgraph_graph.py -q
# 4 offline assertions + 1 skipped (live smoke)
```

The offline tests use `FakeMessagesListChatModel` and an injectable extractor callable to bypass tool-calling in the fake model.

### Architecture

Graph: `START → retrieve_memory → respond → write_memory → END` (unconditional edges). Nodes import the `MemoryStore` Protocol only — never `mempill` directly. Only `__main__.py` constructs `MempillMemoryStore` and imports `mempill`.

---

## Architecture notes

- The mempill engine is a structural, bi-temporal memory store — no LLM, no embeddings.
- Temporal validity is enforced at the claim level via `valid_time` (valid-time dimension) and `as_of_tx_time` (transaction-time dimension).
- `mempill-mcp` is a thin FastMCP wrapper — same engine, two interfaces (Python SDK and MCP).
- The in-memory engine (no `MEMPILL_DB_PATH`) is ephemeral — all data is lost when the process exits.
