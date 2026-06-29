"""
mempill_showcase.config.di — minimal dependency injection factory.

Builds the engine + chosen adapter (mempill or naive) based on settings.
All mempill imports are deferred to this module and the mempill_adapter.
"""
from __future__ import annotations

from enum import Enum
from typing import Union

from mempill_showcase.adapters.memory.naive_adapter import NaiveAdapter


class AdapterMode(str, Enum):
    MEMPILL = "mempill"
    NAIVE = "naive"


def build_mempill_adapter(in_memory: bool = True):
    """Build a MempillAdapter wrapping a real mempill engine.

    in_memory=True  → ephemeral in-memory engine (for tests and demos)
    in_memory=False → raises NotImplementedError (file-backed not wired in W1)
    """
    from mempill import open_in_memory
    from mempill_showcase.adapters.memory.mempill_adapter import MempillAdapter

    if not in_memory:
        raise NotImplementedError("File-backed mempill engine not wired in W1; use in_memory=True")

    engine = open_in_memory()
    return MempillAdapter(engine)


def build_naive_adapter() -> NaiveAdapter:
    """Build a NaiveAdapter (no mempill, no valid-time, no provenance)."""
    return NaiveAdapter()


def build_adapter(
    mode: AdapterMode = AdapterMode.MEMPILL,
    in_memory: bool = True,
) -> Union["MempillAdapter", NaiveAdapter]:  # type: ignore[name-defined]
    """Factory: return the adapter for the requested mode.

    mode=MEMPILL → MempillAdapter (BiTemporalMemoryStore)
    mode=NAIVE   → NaiveAdapter (MemoryStore only)
    """
    if mode == AdapterMode.MEMPILL:
        return build_mempill_adapter(in_memory=in_memory)
    elif mode == AdapterMode.NAIVE:
        return build_naive_adapter()
    else:
        raise ValueError(f"Unknown AdapterMode: {mode!r}")
