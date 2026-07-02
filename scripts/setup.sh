#!/usr/bin/env bash
# scripts/setup.sh — idempotent setup for mempill-demo on Python 3.12
#
# This script creates a Python 3.12 venv and installs all showcase dependencies
# (LangGraph, langchain-core, mempill) via PyPI. mempill 0.3.0 (query_subject,
# valid_at, granularity) is published on PyPI and resolves as a normal
# dependency — no local wheel build required.
#
# Prerequisites:
#   - uv  (https://docs.astral.sh/uv/)
#
# Optional:
#   - Sibling repo ../mempill/mempill-mcp/ (only needed for the MCP demo;
#     mempill-mcp is pure Python and not published on PyPI)

set -euo pipefail

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MEMPILL_MCP_DIR="$(cd "$DEMO_DIR/../mempill/mempill-mcp" 2>/dev/null && pwd || echo "")"

# ── Guard: uv ────────────────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "ERROR: uv not found. Install it first: https://docs.astral.sh/uv/" >&2
    exit 1
fi

echo "=== mempill-demo setup (Python 3.12) ==="
echo "Demo dir: $DEMO_DIR"
echo ""

# ── Ensure Python 3.12 is available ──────────────────────────────────────────
if ! uv python list 2>&1 | grep -q "cpython-3.12"; then
    echo "[0/4] Installing Python 3.12 via uv..."
    uv python install 3.12
else
    echo "[0/4] Python 3.12 already installed."
fi

# ── Create venv on Python 3.12 (replaces any existing venv) ──────────────────
cd "$DEMO_DIR"
if [ ! -d ".venv" ]; then
    echo "[1/4] Creating Python 3.12 virtual environment..."
    uv venv --python 3.12
else
    CURRENT_PY=$(.venv/bin/python --version 2>&1 | awk '{print $2}')
    if [[ "$CURRENT_PY" != 3.12* ]]; then
        echo "[1/4] Existing venv is Python $CURRENT_PY — recreating on 3.12..."
        rm -rf .venv
        uv venv --python 3.12
    else
        echo "[1/4] Python 3.12 venv already exists, skipping."
    fi
fi

# ── Install base runtime deps (including mempill, resolved from PyPI) ────────
echo "[2/4] Installing base runtime deps (anthropic, mcp, mempill, python-dotenv, rich)..."
uv pip install \
    "anthropic>=0.111,<1" \
    "mcp>=1.9,<2" \
    "mempill>=0.3.0,<0.4" \
    "python-dotenv>=1.0" \
    "rich>=13"
echo "      Base runtime deps installed."

# ── Install showcase extras (LangGraph) ──────────────────────────────────────
echo "[3/4] Installing showcase extras (LangGraph + pytest)..."
uv pip install \
    "langgraph>=1.2.6,<2" \
    "langgraph-prebuilt>=1.1.0,<2" \
    "langchain-core>=1.4.8,<2" \
    "langchain-anthropic>=1.4.7,<2" \
    "pytest>=7"
echo "      LangGraph deps installed."

# ── Install the demo package itself (editable, resolves mempill from PyPI) ───
echo "[4/4] Installing mempill-demo (editable)..."
uv pip install -e .
echo "      mempill-demo installed."

# ── Install mempill-mcp (optional, pure Python, editable) ───────────────────
if [ -n "$MEMPILL_MCP_DIR" ] && [ -d "$MEMPILL_MCP_DIR" ]; then
    echo ""
    echo "[opt] Installing mempill-mcp (editable) from $MEMPILL_MCP_DIR..."
    uv pip install -e "$MEMPILL_MCP_DIR"
    echo "      mempill-mcp installed."
    MCP_INSTALLED=true
else
    MCP_INSTALLED=false
fi

echo ""
echo "=== Setup complete (Python 3.12) ==="
echo ""
echo "Verify with:"
echo "  .venv/bin/python -c \"import mempill, langgraph, langchain_core; print('imports OK')\""
echo ""
echo "Run the showcase test suite (no API key required):"
echo "  .venv/bin/python -m pytest src/mempill_showcase/tests/ -v -m 'not live'"
echo ""
echo "Run the demo:"
echo "  .venv/bin/python examples/temporal_validity.py"
echo ""
echo "Run the interactive console agent:"
echo "  .venv/bin/python -m mempill_demo --scenario    # 3-act demo then REPL"
echo "  .venv/bin/python -m mempill_demo --selftest    # CI assertion suite (no API key)"
echo "  .venv/bin/python -m mempill_demo               # plain REPL"
echo ""
echo "Run the LangGraph conversational agent (requires ANTHROPIC_API_KEY in .env):"
echo "  .venv/bin/python -m mempill_langgraph"
echo ""
if [ "$MCP_INSTALLED" = "true" ]; then
    echo "Run the MCP verification:"
    echo "  .venv/bin/python mcp/verify_stdio.py"
    echo ""
fi
