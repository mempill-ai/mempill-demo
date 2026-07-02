#!/usr/bin/env bash
# scripts/setup.sh — idempotent setup for mempill-demo on Python 3.12
#
# This script creates a Python 3.12 venv and installs all showcase dependencies
# (LangGraph, langchain-core, mempill) via PyPI. mempill 0.3.0 (query_subject,
# valid_at, granularity) is published on PyPI and resolves as a normal
# dependency — no local wheel build required.
#
# Usage:
#   scripts/setup.sh                # default: install mempill from PyPI (mempill>=0.3.0,<0.4)
#   scripts/setup.sh --local-engine # install mempill from a LOCAL source build in ../mempill
#                                    # instead — use this when developing/testing unreleased
#                                    # mempill engine changes before they're published to PyPI.
#
# Prerequisites:
#   - uv  (https://docs.astral.sh/uv/)
#
# Optional:
#   - Sibling repo ../mempill/mempill-mcp/ (only needed for the MCP demo;
#     mempill-mcp is pure Python and not published on PyPI)
#   - Sibling repo ../mempill/ with mempill-python's .venv (maturin installed)
#     — only needed when passing --local-engine

set -euo pipefail

LOCAL_ENGINE=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --local-engine)
            LOCAL_ENGINE=true
            shift
            ;;
        -h|--help)
            grep '^#' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            echo "       Usage: $0 [--local-engine]" >&2
            exit 1
            ;;
    esac
done

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MEMPILL_REPO="$(cd "$DEMO_DIR/../mempill" 2>/dev/null && pwd || echo "")"
MEMPILL_MCP_DIR="$(cd "$DEMO_DIR/../mempill/mempill-mcp" 2>/dev/null && pwd || echo "")"

# ── Guard: uv ────────────────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "ERROR: uv not found. Install it first: https://docs.astral.sh/uv/" >&2
    exit 1
fi

echo "=== mempill-demo setup (Python 3.12) ==="
echo "Demo dir: $DEMO_DIR"
if [ "$LOCAL_ENGINE" = "true" ]; then
    echo "Mode: --local-engine (installing mempill from local source build)"
else
    echo "Mode: default (installing mempill from PyPI)"
fi
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

# ── Install base runtime deps ────────────────────────────────────────────────
if [ "$LOCAL_ENGINE" = "true" ]; then
    echo "[2/4] Installing base runtime deps (anthropic, mcp, python-dotenv, rich)..."
    uv pip install \
        "anthropic>=0.111,<1" \
        "mcp>=1.9,<2" \
        "python-dotenv>=1.0" \
        "rich>=13"
else
    echo "[2/4] Installing base runtime deps (anthropic, mcp, mempill, python-dotenv, rich)..."
    uv pip install \
        "anthropic>=0.111,<1" \
        "mcp>=1.9,<2" \
        "mempill>=0.3.0,<0.4" \
        "python-dotenv>=1.0" \
        "rich>=13"
fi
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

# ── Install the demo package itself ──────────────────────────────────────────
if [ "$LOCAL_ENGINE" = "true" ]; then
    echo "[4/4] Installing mempill-demo (editable, local build)..."
    uv pip install --no-deps -e .
else
    echo "[4/4] Installing mempill (PyPI)..."
    uv pip install -e .
fi
echo "      mempill-demo installed."

# ── --local-engine: build and force-install the local mempill wheel ─────────
if [ "$LOCAL_ENGINE" = "true" ]; then
    echo ""
    echo "[local-engine] Building mempill from ../mempill/mempill-python via maturin..."
    if [ -z "$MEMPILL_REPO" ] || [ ! -d "$MEMPILL_REPO" ]; then
        echo "ERROR: Sibling repo ../mempill/ not found." >&2
        echo "       --local-engine requires the mempill repo checked out next to mempill-demo." >&2
        exit 1
    fi

    MATURIN_VENV="$MEMPILL_REPO/mempill-python/.venv"
    if [ ! -x "$MATURIN_VENV/bin/maturin" ]; then
        echo "ERROR: maturin not found at $MATURIN_VENV/bin/maturin" >&2
        echo "       Run: cd $MEMPILL_REPO/mempill-python && uv venv && uv pip install maturin" >&2
        exit 1
    fi

    (cd "$MEMPILL_REPO/mempill-python" && "$MATURIN_VENV/bin/maturin" build --release)

    WHEEL=$(ls -t "$MEMPILL_REPO/target/wheels/mempill-"*"-cp311-abi3-"*".whl" 2>/dev/null | head -1)
    if [ -z "$WHEEL" ]; then
        echo "ERROR: no mempill-*-cp311-abi3-*.whl found in $MEMPILL_REPO/target/wheels/ after build." >&2
        exit 1
    fi

    echo "      Installing wheel: $WHEEL"
    uv pip install --force-reinstall "$WHEEL"
    echo "      Installed mempill from local build: $(basename "$WHEEL")"
else
    echo "      Installed mempill from PyPI (0.3.0+)"
fi

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
