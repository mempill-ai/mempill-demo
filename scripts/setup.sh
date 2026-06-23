#!/usr/bin/env bash
# scripts/setup.sh — idempotent setup for mempill-demo
# Builds the mempill Rust/PyO3 wheel and installs all demo dependencies.
# Prerequisites: cargo (Rust toolchain), uv

set -euo pipefail

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MEMPILL_DIR="$(cd "$DEMO_DIR/../mempill" && pwd 2>/dev/null || true)"

# ── Guard: cargo (Rust toolchain) ────────────────────────────────────────────
if ! command -v cargo &>/dev/null; then
    echo "ERROR: cargo not found. Install the Rust toolchain first: https://rustup.rs" >&2
    exit 1
fi

# ── Guard: uv ────────────────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "ERROR: uv not found. Install it first: https://docs.astral.sh/uv/" >&2
    exit 1
fi

# ── Guard: sibling mempill repo ──────────────────────────────────────────────
if [ ! -d "$MEMPILL_DIR" ]; then
    echo "ERROR: sibling mempill repo not found at $MEMPILL_DIR" >&2
    echo "Expected layout:" >&2
    echo "  .../mempill-ai/mempill/       (core repo)" >&2
    echo "  .../mempill-ai/mempill-demo/  (this repo)" >&2
    exit 1
fi

MEMPILL_PYTHON_DIR="$MEMPILL_DIR/mempill-python"
MEMPILL_MCP_DIR="$MEMPILL_DIR/mempill-mcp"

if [ ! -d "$MEMPILL_PYTHON_DIR" ]; then
    echo "ERROR: mempill-python not found at $MEMPILL_PYTHON_DIR" >&2
    exit 1
fi

if [ ! -d "$MEMPILL_MCP_DIR" ]; then
    echo "ERROR: mempill-mcp not found at $MEMPILL_MCP_DIR" >&2
    exit 1
fi

echo "=== mempill-demo setup ==="
echo "Demo dir:         $DEMO_DIR"
echo "mempill dir:      $MEMPILL_DIR"
echo ""

# ── Create venv (idempotent) ──────────────────────────────────────────────────
cd "$DEMO_DIR"
if [ ! -d ".venv" ]; then
    echo "[1/5] Creating virtual environment..."
    uv venv
else
    echo "[1/5] Virtual environment already exists, skipping."
fi

# ── Build + install the mempill Rust/PyO3 wheel (PEP-517 via maturin) ────────
echo "[2/5] Building mempill wheel (PEP-517 / maturin — may take ~1-2 min on first run)..."
uv pip install "$MEMPILL_PYTHON_DIR"
echo "      mempill wheel installed."

# ── Install mempill-mcp (pure Python, editable) ──────────────────────────────
echo "[3/5] Installing mempill-mcp (editable)..."
uv pip install -e "$MEMPILL_MCP_DIR"
echo "      mempill-mcp installed."

# ── Install MCP client library ────────────────────────────────────────────────
echo "[4/5] Installing mcp>=1.9,<2 and rich>=13..."
uv pip install "mcp>=1.9,<2" "rich>=13"
echo "      mcp + rich installed."

# ── Install this demo package ─────────────────────────────────────────────────
echo "[5/5] Installing mempill-demo (editable)..."
uv pip install -e .
echo "      mempill-demo installed."

# ── [6/6] Install LangGraph conversational-agent deps (default; opt-out via SKIP_LANGGRAPH)
if [ "${SKIP_LANGGRAPH:-false}" != "true" ]; then
    echo "[6/6] Installing LangGraph conversational-agent dependencies..."
    uv pip install -e ".[langgraph]"
    echo "      LangGraph deps installed.  (skip with SKIP_LANGGRAPH=true)"
else
    echo "[6/6] Skipping LangGraph deps (SKIP_LANGGRAPH=true)."
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Verify with:"
echo "  uv run python -c \"import mempill, mempill_mcp, mcp; print('imports OK')\""
echo ""
echo "Run the demo:"
echo "  uv run python examples/temporal_validity.py"
echo ""
echo "Run the MCP verification:"
echo "  uv run python mcp/verify_stdio.py"
echo ""
echo "Run the interactive console agent:"
echo "  uv run python -m mempill_demo --scenario    # 3-act demo then REPL"
echo "  uv run python -m mempill_demo --selftest    # CI assertion suite (no API key)"
echo "  uv run python -m mempill_demo               # plain REPL"
echo ""
echo "Run the LangGraph conversational agent (requires ANTHROPIC_API_KEY in .env):"
echo "  uv run python -m mempill_langgraph"
