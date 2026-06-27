#!/usr/bin/env bash
# scripts/setup.sh — idempotent setup for mempill-demo
# Installs all demo dependencies. mempill is pulled from PyPI (no Rust needed).
# Prerequisites: uv
# Optional: sibling repo ../mempill/mempill-mcp/ (only needed for the MCP demo)

set -euo pipefail

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MEMPILL_MCP_DIR="$(cd "$DEMO_DIR/../mempill/mempill-mcp" 2>/dev/null && pwd || echo "")"

# ── Guard: uv ────────────────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "ERROR: uv not found. Install it first: https://docs.astral.sh/uv/" >&2
    exit 1
fi

echo "=== mempill-demo setup ==="
echo "Demo dir: $DEMO_DIR"
echo ""

# ── Create venv (idempotent) ──────────────────────────────────────────────────
cd "$DEMO_DIR"
if [ ! -d ".venv" ]; then
    echo "[1/4] Creating virtual environment..."
    uv venv
else
    echo "[1/4] Virtual environment already exists, skipping."
fi

# ── Install this demo package (pulls mempill from PyPI via dependencies) ─────
echo "[2/4] Installing mempill-demo (editable) — mempill pulled from PyPI..."
uv pip install -e .
echo "      mempill-demo + mempill installed."

# ── Install LangGraph conversational-agent deps (default; opt-out via SKIP_LANGGRAPH)
if [ "${SKIP_LANGGRAPH:-false}" != "true" ]; then
    echo "[3/4] Installing LangGraph conversational-agent dependencies..."
    uv pip install -e ".[langgraph]"
    echo "      LangGraph deps installed.  (skip with SKIP_LANGGRAPH=true)"
else
    echo "[3/4] Skipping LangGraph deps (SKIP_LANGGRAPH=true)."
fi

# ── Install mempill-mcp (pure Python, editable) — OPTIONAL ──────────────────
if [ -n "$MEMPILL_MCP_DIR" ] && [ -d "$MEMPILL_MCP_DIR" ]; then
    echo "[4/4] Installing mempill-mcp (editable) from $MEMPILL_MCP_DIR..."
    uv pip install -e "$MEMPILL_MCP_DIR"
    echo "      mempill-mcp installed."
    MCP_INSTALLED=true
else
    echo "[4/4] Sibling repo ../mempill/mempill-mcp/ not found — skipping MCP demo install."
    echo "      The console + LangGraph demos work without it."
    echo "      To enable the MCP demo, check out the sibling mempill repo and re-run setup.sh."
    MCP_INSTALLED=false
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Verify with:"
if [ "$MCP_INSTALLED" = "true" ]; then
    echo "  uv run python -c \"import mempill, mempill_mcp, mcp; print('imports OK')\""
else
    echo "  uv run python -c \"import mempill, mcp; print('imports OK')\""
fi
echo ""
echo "Run the demo:"
echo "  uv run python examples/temporal_validity.py"
echo ""
if [ "$MCP_INSTALLED" = "true" ]; then
    echo "Run the MCP verification:"
    echo "  uv run python mcp/verify_stdio.py"
    echo ""
fi
echo "Run the interactive console agent:"
echo "  uv run python -m mempill_demo --scenario    # 3-act demo then REPL"
echo "  uv run python -m mempill_demo --selftest    # CI assertion suite (no API key)"
echo "  uv run python -m mempill_demo               # plain REPL"
echo ""
echo "Run the LangGraph conversational agent (requires ANTHROPIC_API_KEY in .env):"
echo "  uv run python -m mempill_langgraph"
echo ""
