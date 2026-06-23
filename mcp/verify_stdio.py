"""
mcp/verify_stdio.py — Real MCP client over stdio verifying mempill-mcp

Launches the mempill-mcp server as a subprocess (stdio transport), performs
the full MCP handshake, lists tools, calls ingest_claim + query_memory, and
asserts the round-trip is correct.

Usage:
  uv run python mcp/verify_stdio.py           # prints [VERIFIED] on success
  uv run python mcp/verify_stdio.py --verify  # exits 0 on success, 1 on failure

No MEMPILL_DB_PATH → in-memory engine (ephemeral, no files written).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# ── Paths ──────────────────────────────────────────────────────────────────────

DEMO_DIR = Path(__file__).resolve().parent.parent
MEMPILL_MCP_DIR = (DEMO_DIR.parent / "mempill" / "mempill-mcp").resolve()

EXPECTED_TOOLS = {"ingest_claim", "query_memory", "reconcile", "audit"}


# ── Verification logic ─────────────────────────────────────────────────────────

async def verify() -> None:
    """Run the full MCP verification suite."""

    if not MEMPILL_MCP_DIR.exists():
        raise RuntimeError(
            f"mempill-mcp not found at {MEMPILL_MCP_DIR}\n"
            "Expected layout: ../mempill/mempill-mcp/ (sibling of this repo)"
        )

    # Use the demo venv's Python directly so we share the already-built mempill
    # wheel. Using `uv run --project mempill-mcp` would create a NEW venv for that
    # project and fail to find the unpublished mempill wheel (not on PyPI).
    demo_python = DEMO_DIR / ".venv" / "bin" / "python"
    if not demo_python.exists():
        raise RuntimeError(
            f"Demo venv not found at {demo_python}\n"
            "Run `bash scripts/setup.sh` first."
        )

    # Build env: pass PATH + HOME so the subprocess can find system tools.
    # No MEMPILL_DB_PATH → server opens an in-memory engine (ephemeral).
    server_env = {
        "MEMPILL_AGENT_ID": "verify-agent",
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
    }

    server_params = StdioServerParameters(
        command=str(demo_python),
        args=["-m", "mempill_mcp"],
        env=server_env,
    )

    print(f"[verify_stdio] Launching mempill-mcp via demo venv Python")
    print(f"[verify_stdio] Command: {demo_python} -m mempill_mcp")
    print()

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:

            # ── Step 1: MCP handshake ─────────────────────────────────────────
            await session.initialize()
            print("[step 1] MCP initialize: OK")

            # ── Step 2: List tools — assert all 4 are present ─────────────────
            tools_result = await session.list_tools()
            tool_names = {t.name for t in tools_result.tools}
            print(f"[step 2] list_tools: {sorted(tool_names)}")

            missing = EXPECTED_TOOLS - tool_names
            if missing:
                raise AssertionError(
                    f"Missing tools from mempill-mcp: {missing}. Got: {tool_names}"
                )
            print(f"[step 2] All 4 tools present: {sorted(EXPECTED_TOOLS)}")

            # ── Step 3: ingest_claim ──────────────────────────────────────────
            # Note: valid_time must include valid_time_confidence (part of the
            # ValidTime struct, not just the top-level confidence scores).
            ingest_result = await session.call_tool(
                name="ingest_claim",
                arguments={
                    "subject": "verify:ceo",
                    "predicate": "held_by",
                    "value": "VerifyAlice",
                    "provenance": "External:UserAsserted",
                    "cardinality": "Functional",
                    "confidence_value": 0.95,
                    "confidence_valid_time": 0.9,
                    "criticality": "Low",
                    "valid_time": {
                        "start": "2020-01-01T00:00:00Z",
                        "valid_time_confidence": 0.9,
                    },
                },
            )

            # CallToolResult.content is a list of TextContent/etc.
            ingest_text = ""
            if ingest_result.content:
                first = ingest_result.content[0]
                ingest_text = getattr(first, "text", str(first))

            print(f"[step 3] ingest_claim result: {ingest_text}")

            # Parse the ingest response to get disposition
            try:
                ingest_data = json.loads(ingest_text)
                disposition = ingest_data.get("disposition", "UNKNOWN")
                claim_ref = ingest_data.get("claim_ref", "UNKNOWN")
                print(f"[step 3] disposition={disposition}, claim_ref={claim_ref[:8]}...")
            except (json.JSONDecodeError, AttributeError):
                print(f"[step 3] raw response (not JSON): {ingest_text}")
                ingest_data = {}
                disposition = "UNKNOWN"

            # ── Step 4: query_memory — assert belief reflects the ingest ──────
            query_result = await session.call_tool(
                name="query_memory",
                arguments={
                    "subject": "verify:ceo",
                    "predicate": "held_by",
                },
            )

            query_text = ""
            if query_result.content:
                first = query_result.content[0]
                query_text = getattr(first, "text", str(first))

            print(f"[step 4] query_memory result: {query_text}")

            try:
                query_data = json.loads(query_text)
                belief = query_data.get("belief", {})
                belief_status = belief.get("status")
                # BeliefProjection nests the value at belief.primary.fact.value
                primary = belief.get("primary") or {}
                fact = primary.get("fact") or {}
                belief_value = fact.get("value")
                print(f"[step 4] belief.status={belief_status!r}, belief.primary.fact.value={belief_value!r}")
            except (json.JSONDecodeError, AttributeError):
                print(f"[step 4] raw response: {query_text}")
                belief_value = None
                belief_status = None

            # Assert the round-trip: the ingested value should appear in belief.primary.fact.value
            if belief_value == "VerifyAlice":
                print("[step 4] Round-trip assertion PASSED: belief.primary.fact.value == 'VerifyAlice'")
            else:
                raise AssertionError(
                    f"Round-trip FAILED: expected belief.primary.fact.value='VerifyAlice', "
                    f"got {belief_value!r}. belief.status={belief_status!r}. Full response: {query_text}"
                )

            print()
            print("[VERIFIED]")
            print()
            print("Summary:")
            print(f"  Server:                  {demo_python} -m mempill_mcp (in-memory engine)")
            print(f"  MCP initialize:          OK")
            print(f"  Tools listed:            {sorted(tool_names)}")
            print(f"  ingest_claim:            disposition={disposition}")
            print(f"  query_memory round-trip: belief.primary.fact.value='VerifyAlice' (PASS)")


def main() -> int:
    """Entry point. Returns exit code 0 on success, 1 on failure."""
    verify_mode = "--verify" in sys.argv

    try:
        asyncio.run(verify())
        return 0
    except AssertionError as e:
        print(f"\n[FAILED] Assertion error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        if verify_mode:
            return 1
        raise


if __name__ == "__main__":
    sys.exit(main())
