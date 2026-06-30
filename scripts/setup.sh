#!/usr/bin/env bash
# scripts/setup.sh — idempotent setup for mempill-demo on Python 3.12
#
# This script creates a Python 3.12 venv and installs all showcase dependencies
# (CrewAI, LangGraph, langchain-core) WITHOUT pulling mempill from PyPI.
# Instead it installs the LOCAL source-built mempill abi3 wheel from the sibling
# mempill repo.  That wheel is built from the source (valid_at + granularity
# support) and runs on Python 3.11–3.14 (abi3 tag: cp311-abi3).
#
# WHY NOT PyPI?
#   PyPI has mempill 0.2.x which lacks valid_at + granularity.
#   The showcase requires the prerelease features.
#   Once mempill 0.3.0 is published, update pyproject.toml to >=0.3.0 and
#   remove the local-wheel step below.
#
# Prerequisites:
#   - uv  (https://docs.astral.sh/uv/)
#   - Sibling repo ../mempill/  checked out at main (for the abi3 wheel)
#     The wheel is at ../mempill/target/wheels/mempill-*-cp311-abi3-*.whl
#     If no wheel exists yet, run:
#       cd ../mempill/mempill-python && ./.venv/bin/maturin build --release
#     (maturin lives in ../mempill/mempill-python/.venv)
#
# Optional:
#   - Sibling repo ../mempill/mempill-mcp/ (only needed for the MCP demo)

set -euo pipefail

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
echo ""

# ── Ensure Python 3.12 is available ──────────────────────────────────────────
if ! uv python list 2>&1 | grep -q "cpython-3.12"; then
    echo "[0/5] Installing Python 3.12 via uv..."
    uv python install 3.12
else
    echo "[0/5] Python 3.12 already installed."
fi

# ── Create venv on Python 3.12 (replaces any existing venv) ──────────────────
cd "$DEMO_DIR"
if [ ! -d ".venv" ]; then
    echo "[1/5] Creating Python 3.12 virtual environment..."
    uv venv --python 3.12
else
    CURRENT_PY=$(.venv/bin/python --version 2>&1 | awk '{print $2}')
    if [[ "$CURRENT_PY" != 3.12* ]]; then
        echo "[1/5] Existing venv is Python $CURRENT_PY — recreating on 3.12..."
        rm -rf .venv
        uv venv --python 3.12
    else
        echo "[1/5] Python 3.12 venv already exists, skipping."
    fi
fi

# ── Install base runtime deps (NOT mempill — we install the local wheel below) ─
echo "[2/5] Installing base runtime deps (anthropic, mcp, python-dotenv, rich)..."
uv pip install \
    "anthropic>=0.111,<1" \
    "mcp>=1.9,<2" \
    "python-dotenv>=1.0" \
    "rich>=13"
echo "      Base runtime deps installed."

# ── Install showcase extras (LangGraph + CrewAI) ─────────────────────────────
echo "[3/5] Installing showcase extras (LangGraph + CrewAI + pytest)..."
uv pip install \
    "langgraph>=1.2.6,<2" \
    "langgraph-prebuilt>=1.1.0,<2" \
    "langchain-core>=1.4.8,<2" \
    "langchain-anthropic>=1.4.7,<2" \
    "crewai>=1.0,<2" \
    "pytest>=7"
echo "      LangGraph + CrewAI deps installed."

# ── Install the demo package itself (no deps to avoid pulling mempill from PyPI)
echo "[4/5] Installing mempill-demo (editable, --no-deps)..."
uv pip install --no-deps -e .
echo "      mempill-demo installed."

# ── Install the LOCAL source-built mempill abi3 wheel ─────────────────────────
# This is needed until mempill 0.3.0 is published on PyPI.
# The wheel supports Python 3.11–3.14 (abi3 tag).
echo "[5/5] Installing local mempill prerelease wheel (source-built, NOT PyPI)..."
if [ -z "$MEMPILL_REPO" ] || [ ! -d "$MEMPILL_REPO" ]; then
    echo "ERROR: Sibling repo ../mempill/ not found." >&2
    echo "       Check out the mempill repo next to mempill-demo and rebuild:" >&2
    echo "         cd ../mempill/mempill-python && ./.venv/bin/maturin build --release" >&2
    exit 1
fi

WHEEL=$(ls -t "$MEMPILL_REPO/target/wheels/mempill-"*"-cp311-abi3-"*".whl" 2>/dev/null | head -1)
if [ -z "$WHEEL" ]; then
    echo "      No abi3 wheel found — building now..."
    MATURIN_VENV="$MEMPILL_REPO/mempill-python/.venv"
    if [ ! -x "$MATURIN_VENV/bin/maturin" ]; then
        echo "ERROR: maturin not found at $MATURIN_VENV/bin/maturin" >&2
        echo "       Run: cd $MEMPILL_REPO/mempill-python && uv venv && uv pip install maturin" >&2
        exit 1
    fi
    (cd "$MEMPILL_REPO/mempill-python" && "$MATURIN_VENV/bin/maturin" build --release)
    WHEEL=$(ls -t "$MEMPILL_REPO/target/wheels/mempill-"*"-cp311-abi3-"*".whl" 2>/dev/null | head -1)
fi

echo "      Installing wheel: $WHEEL"
uv pip install --force-reinstall "$WHEEL"
echo "      mempill prerelease installed from local wheel."

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
echo "  .venv/bin/python -c \"import mempill, crewai, langgraph, langchain_core; print('imports OK')\""
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
echo "NOTE: mempill is installed from the LOCAL source wheel, not PyPI."
echo "      Once mempill 0.3.0 is published, update pyproject.toml to >=0.3.0"
echo "      and remove the local-wheel step from this script."
echo ""
