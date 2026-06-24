"""
mempill_demo.app.review — /review command: human-in-the-loop adjudication UI.

4-choice mapping (ARCHITECTURE §C.2 updated):
  c / challenger → submit verdict "Affirm"  (challenger wins)
  i / incumbent  → submit verdict "Deny"    (incumbent stands)
  s / skip / empty / EOF → DEFER: leave pending (reappears next /review)
  a / abstain    → submit verdict "Unknown" (stays Contested; removed from queue)

No mempill imports here. All engine calls go through the OracleMemoryStore port.
"""
from __future__ import annotations

from typing import Callable, Optional


def run_review(
    store: object,
    presenter: object,
    *,
    _input_fn: Optional[Callable[[], str]] = None,
) -> None:
    """
    Run the interactive /review loop over pending adjudications.

    Args:
        store:      An OracleMemoryStore (has list_pending() + submit()).
        presenter:  Presenter (has render() for messages).
        _input_fn:  Injectable input function for non-interactive testing.
                    Defaults to built-in input().
    """
    input_fn = _input_fn if _input_fn is not None else input

    # Retrieve pending adjudications
    list_pending = getattr(store, "list_pending", None)
    submit = getattr(store, "submit", None)

    if list_pending is None or submit is None:
        print("[/review] This store does not support oracle adjudication.")
        return

    pending = list_pending()

    if not pending:
        print("[/review] No pending adjudications — queue is empty.")
        return

    print(f"\n[/review] {len(pending)} pending adjudication(s):\n")

    for item in pending:
        handle_id = item.get("handle_id", "")
        subject = item.get("subject", "?")
        predicate = item.get("predicate", "?")
        incumbent_value = item.get("incumbent_value", "?")
        challenger_value = item.get("challenger_value", "?")
        queued_at = str(item.get("queued_at", ""))[:19]

        print(f"  Subject:    {subject}")
        print(f"  Predicate:  {predicate}")
        print(f"  Incumbent:  \"{incumbent_value}\"")
        print(f"  Challenger: \"{challenger_value}\"")
        print(f"  Queued at:  {queued_at}")
        print()

        prompt = "  [c]hallenger / [i]ncumbent / [s]kip-ask-later / [a]bstain? "
        try:
            raw = input_fn().strip().lower() if _input_fn else input(prompt).strip().lower()
        except EOFError:
            raw = "s"

        if raw in ("c", "challenger"):
            # Affirm: challenger wins
            result = submit(handle_id, "Affirm")
            disp = result.get("disposition", "?")
            resolved_ref = result.get("claim_ref", "")
            print(f"  Resolved: challenger (\"{challenger_value}\") wins — disposition={disp}  ref={resolved_ref[:8]}...")
            print()

        elif raw in ("i", "incumbent"):
            # Deny: incumbent stands
            result = submit(handle_id, "Deny")
            disp = result.get("disposition", "?")
            resolved_ref = result.get("claim_ref", "")
            print(f"  Resolved: incumbent (\"{incumbent_value}\") stands — disposition={disp}  ref={resolved_ref[:8]}...")
            print()

        elif raw in ("a", "abstain"):
            # Unknown: stays Contested, removed from queue
            result = submit(handle_id, "Unknown")
            disp = result.get("disposition", "?")
            print(f"  Left Contested (abstained) — disposition={disp}")
            print()

        else:
            # s / skip / empty / any other input → defer
            print("  Deferred — will ask again next /review.")
            print()
