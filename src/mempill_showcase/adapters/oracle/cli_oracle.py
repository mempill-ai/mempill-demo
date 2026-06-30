"""
mempill_showcase.adapters.oracle.cli_oracle — stdin-based oracle for manual/live runs.

Presents a contested adjudication request on stdout and reads a verdict from stdin.
This is the MANUAL / LIVE RUN oracle.  Deterministic tests use the automated
Command(resume=<verdict>) path through LangGraph; they do not touch this module.

Interface:
  The mempill engine duck-types the oracle: any object implementing
      request_adjudication(agent_id: str, request: dict) -> str
  qualifies.  The engine calls this when a conflicting Functional claim is ingested
  and returns the handle_id UUID string back to the engine to queue the request.

Design:
  - This class is STATELESS: it only generates a handle_id UUID on request.
  - The durable store lives inside the mempill engine.
  - Verdicts are submitted LATER via adapter.submit_adjudication(handle_id, verdict).
  - For live interactive runs, the operator uses the provided helper display() method
    or pipes stdin from a script.

Pluggable surface:
  A future Slack oracle, web-hook oracle, or test oracle implements the same
  request_adjudication() method and is passed to open_oracle_in_memory() unchanged.

Usage (live run):
    from mempill_showcase.adapters.oracle.cli_oracle import CliOracle
    import mempill

    oracle = CliOracle()
    engine = mempill.open_oracle_in_memory(oracle)
    # ... ingest conflicting claims ...
    pending = engine.list_pending_adjudications(agent_id="my-agent")
    oracle.display_pending(pending)
    verdict = oracle.prompt_verdict(pending[0])
    engine.submit_adjudication({
        "handle_id": pending[0]["handle_id"],
        "verdict": verdict,
        "evidence_provenance": mempill.ProvenanceLabel.external_first_hand(),
    })
"""
from __future__ import annotations

import sys
import uuid

VALID_VERDICTS = frozenset({"Affirm", "Deny", "Unknown"})


class CliOracle:
    """Stdin oracle for manual/live HITL adjudication sessions.

    Implements the mempill oracle duck-type:
        request_adjudication(agent_id, request) -> handle_id

    For INTERACTIVE USE ONLY.  Tests and automated flows use Command(resume=...).
    """

    def request_adjudication(self, agent_id: str, request: dict) -> str:
        """Return a fresh UUID handle_id; the engine records the adjudication request."""
        return str(uuid.uuid4())

    # ── Display helpers ───────────────────────────────────────────────────────

    def display_pending(self, pending: list[dict], file=None) -> None:
        """Print all pending adjudication entries in a human-readable format.

        Args:
            pending: list returned by engine.list_pending_adjudications(agent_id=...).
            file:    output stream (default: sys.stdout).
        """
        out = file or sys.stdout
        if not pending:
            print("No pending adjudications.", file=out)
            return

        print(f"\n{'='*60}", file=out)
        print(f"  {len(pending)} PENDING ADJUDICATION(S)", file=out)
        print(f"{'='*60}", file=out)
        for i, entry in enumerate(pending, 1):
            handle = entry.get("handle_id", "?")
            subj = entry.get("subject", "?")
            pred = entry.get("predicate", "?")
            inc_val = entry.get("incumbent_value", "?")
            chal_val = entry.get("challenger_value", "?")
            print(f"\n  [{i}] handle={handle}", file=out)
            print(f"      {subj}/{pred}", file=out)
            print(f"      Incumbent:  {inc_val!r}", file=out)
            print(f"      Challenger: {chal_val!r}", file=out)
        print(f"\n{'='*60}\n", file=out)

    def prompt_verdict(self, entry: dict, file=None, input_fn=None) -> str:
        """Prompt the operator for a verdict for a single pending entry.

        Args:
            entry:    A single pending adjudication dict.
            file:     Output stream for the prompt (default: sys.stdout).
            input_fn: Callable() -> str for reading input (default: built-in input()).

        Returns:
            "Affirm" | "Deny" | "Unknown"
        """
        out = file or sys.stdout
        read = input_fn or input

        handle = entry.get("handle_id", "?")
        subj = entry.get("subject", "?")
        pred = entry.get("predicate", "?")
        inc_val = entry.get("incumbent_value", "?")
        chal_val = entry.get("challenger_value", "?")

        print(f"\nConflict on {subj}/{pred}:", file=out)
        print(f"  Incumbent:  {inc_val!r}", file=out)
        print(f"  Challenger: {chal_val!r}", file=out)
        print(f"  Handle: {handle}", file=out)
        print("Verdict? [Affirm / Deny / Unknown]: ", end="", file=out)
        out.flush()

        while True:
            raw = read().strip()
            # Normalize: accept case-insensitive variants
            normalized = raw.capitalize()
            if normalized in VALID_VERDICTS:
                return normalized
            print(
                f"  Invalid verdict {raw!r}. Enter 'Affirm', 'Deny', or 'Unknown': ",
                end="",
                file=out,
            )
            out.flush()
