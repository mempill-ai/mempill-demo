"""
mempill_demo.app.scenario — auto-play the 3-act story then hand off to the REPL.

Act 1: Ingest Alice as CEO (2020-)                → CommittedCheap
Act 2: Ingest Bob as CEO (2023-)                  → Contested (or actual disposition)
       /reconcile                                 → Superseded + Resolved
Act 3: RECALL_REENTRY ×5 (amplification firewall) → belief unchanged

No imports of mempill/rich/anthropic in this module. All mempill calls are
delegated to the MemoryStore port and the adapter's run_scenario helper.
"""
from __future__ import annotations

from mempill_demo.domain.models import (
    CommandKind,
    ParsedCommand,
)
from mempill_demo.ports.memory import MemoryStore
from mempill_demo.ports.presenter import Presenter


def run_scenario(store: MemoryStore, presenter: Presenter) -> None:
    """
    Run the 3-act auto-play scenario.

    Delegates all mempill engine calls to the adapter's internal run_scenario
    helper so this module stays clean of mempill imports.
    """
    # The adapter exposes a run_scenario helper that holds the engine calls.
    # If the store is not a MempillMemoryStore, we report and bail.
    _run = getattr(store, "_run_scenario", None)
    if _run is None:
        print("[scenario] Cannot run: store does not support _run_scenario (not MempillMemoryStore).")
        return
    _run()
