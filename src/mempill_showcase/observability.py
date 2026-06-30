"""
mempill_showcase.observability — Optional LangSmith tracing helpers.

All tracing is a NO-OP when LangSmith is not configured. The module never
raises an ImportError or a RuntimeError when keys are absent.

Environment variables (all optional):
  LANGSMITH_API_KEY   — enables LangSmith trace export when set to a valid key.
  LANGSMITH_TRACING   — set to "true" to force-enable tracing (requires API key).
  LANGSMITH_PROJECT   — LangSmith project name (default: "mempill-showcase").

Usage:
  from mempill_showcase.observability import traceable_mempill, emit_contested_span

  # Wrap a tool _run method:
  @traceable_mempill(name="mempill.remember")
  def _run(self, ...): ...

  # Emit a dedicated contested event:
  emit_contested_span(subject="alice-chen", predicate="employer",
                      incumbent={"value": "VP Engineering", ...},
                      challenger={"value": "CTO", ...})

Design decisions:
  - Uses langsmith.traceable() when available; gracefully falls back to a
    no-op passthrough wrapper when the library is absent or key is missing.
  - The `mempill.contested` span is emitted as a child run within the current
    trace context via langsmith.get_current_run_tree() (no-op if no active run).
  - Tracing does NOT modify return values — callers see identical results.
  - Thread-safe: uses langsmith's own context vars; no shared mutable state here.
"""
from __future__ import annotations

import functools
import logging
import os
from typing import Any, Callable, Optional, TypeVar

log = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# ── LangSmith availability check ──────────────────────────────────────────────

def _langsmith_enabled() -> bool:
    """Return True only when langsmith is installed AND tracing is configured.

    Checks LANGSMITH_API_KEY and LANGSMITH_TRACING env vars.
    Import is attempted lazily — never raises if langsmith is absent.
    """
    api_key = os.environ.get("LANGSMITH_API_KEY", "")
    tracing = os.environ.get("LANGSMITH_TRACING", "").lower()
    if not api_key and tracing != "true":
        return False
    try:
        import langsmith  # noqa: F401
        return True
    except ImportError:
        return False


# ── @traceable_mempill decorator ──────────────────────────────────────────────

def traceable_mempill(name: str, **extra_metadata: Any) -> Callable[[F], F]:
    """Return a decorator that wraps a function with a LangSmith span named *name*.

    When LangSmith is not configured, the decorator is a transparent passthrough.
    The span is tagged with run_type="tool" and any extra_metadata provided.

    Example:
        @traceable_mempill(name="mempill.remember", tool="MempillRememberTool")
        def _run(self, agent_id, subject, predicate, value, ...):
            ...
    """
    def decorator(fn: F) -> F:
        if not _langsmith_enabled():
            # No-op passthrough — zero overhead, no import required.
            return fn

        try:
            from langsmith import traceable

            metadata = {"mempill_tool": name, **extra_metadata}

            wrapped = traceable(
                name=name,
                run_type="tool",
                metadata=metadata,
            )(fn)
            log.debug("traceable_mempill: wrapped %s with LangSmith span", name)
            return wrapped  # type: ignore[return-value]
        except Exception as exc:
            log.warning(
                "traceable_mempill: failed to apply @traceable to %s (%s); falling back to no-op",
                name, exc,
            )
            return fn

    return decorator


# ── mempill.contested dedicated span ─────────────────────────────────────────

def emit_contested_span(
    subject: str,
    predicate: str,
    incumbent: Optional[dict] = None,
    challenger: Optional[dict] = None,
    agent_id: Optional[str] = None,
    extra: Optional[dict] = None,
) -> None:
    """Emit a dedicated 'mempill.contested' span/event into the active LangSmith trace.

    This is the key observability moment: when a write or recall returns Contested,
    this function emits a child span with both candidate values attached as metadata.
    In the LangSmith UI this appears as a discrete event within the tool run tree,
    making the conflict visible to operators and debuggers.

    Args:
        subject:    Canonical entity key (e.g. "alice-chen").
        predicate:  Canonical predicate key (e.g. "employer").
        incumbent:  Dict with at least {"value": ..., "valid_from_display": ...}.
        challenger: Dict with at least {"value": ..., "valid_from_display": ...}.
        agent_id:   Optional session owner for trace metadata.
        extra:      Any additional metadata to attach to the span.

    When LangSmith is not active, this function is a no-op (returns immediately).
    """
    if not _langsmith_enabled():
        return

    try:
        from langsmith import traceable

        contested_metadata: dict[str, Any] = {
            "event": "mempill.contested",
            "subject": subject,
            "predicate": predicate,
            "incumbent_value": (incumbent or {}).get("value"),
            "incumbent_valid_from": (incumbent or {}).get("valid_from_display"),
            "challenger_value": (challenger or {}).get("value"),
            "challenger_valid_from": (challenger or {}).get("valid_from_display"),
        }
        if agent_id:
            contested_metadata["agent_id"] = agent_id
        if extra:
            contested_metadata.update(extra)

        # Emit as a traceable sub-call so it shows up as a child span in LangSmith.
        # We use a small inline function to satisfy the traceable protocol.
        @traceable(name="mempill.contested", run_type="tool", metadata=contested_metadata)
        def _contested_event() -> dict:
            return contested_metadata

        _contested_event()

        log.debug(
            "emit_contested_span: emitted mempill.contested for %s/%s "
            "(incumbent=%r vs challenger=%r)",
            subject, predicate,
            (incumbent or {}).get("value"),
            (challenger or {}).get("value"),
        )

    except Exception as exc:
        # Never let tracing break tool execution.
        log.debug("emit_contested_span: suppressed error: %s", exc)


# ── Contested hook registry (for test assertions without a real backend) ──────
#
# Tests can register a callback to verify that emit_contested_span was called.
# This avoids requiring a live LangSmith backend in CI.
#
# Usage in tests:
#   from mempill_showcase.observability import register_contested_hook, clear_contested_hooks
#
#   fired = []
#   register_contested_hook(lambda **kw: fired.append(kw))
#   # ... run tool ...
#   assert len(fired) == 1
#   assert fired[0]["subject"] == "alice-chen"
#   clear_contested_hooks()

_contested_hooks: list[Callable[..., None]] = []


def register_contested_hook(fn: Callable[..., None]) -> None:
    """Register a callback to be called whenever emit_contested_span fires.

    The callback receives the same keyword arguments as emit_contested_span.
    Multiple hooks may be registered; they are called in registration order.
    """
    _contested_hooks.append(fn)


def clear_contested_hooks() -> None:
    """Remove all registered contested hooks (call in test teardown)."""
    _contested_hooks.clear()


def _fire_contested_hooks(**kwargs: Any) -> None:
    """Internal: fire all registered hooks (always, even when tracing is off)."""
    for hook in _contested_hooks:
        try:
            hook(**kwargs)
        except Exception as exc:
            log.debug("_fire_contested_hooks: hook error suppressed: %s", exc)


# Monkey-patch emit_contested_span to also fire hooks.
# We keep the original implementation and wrap it.
_orig_emit_contested_span = emit_contested_span


def emit_contested_span(  # type: ignore[no-redef]
    subject: str,
    predicate: str,
    incumbent: Optional[dict] = None,
    challenger: Optional[dict] = None,
    agent_id: Optional[str] = None,
    extra: Optional[dict] = None,
) -> None:
    """Emit a 'mempill.contested' LangSmith span and fire any registered test hooks."""
    # Always fire hooks first (no-op in production if no hooks registered)
    _fire_contested_hooks(
        subject=subject,
        predicate=predicate,
        incumbent=incumbent,
        challenger=challenger,
        agent_id=agent_id,
        extra=extra,
    )
    # Then do the real LangSmith span (no-op without a key)
    _orig_emit_contested_span(
        subject=subject,
        predicate=predicate,
        incumbent=incumbent,
        challenger=challenger,
        agent_id=agent_id,
        extra=extra,
    )
