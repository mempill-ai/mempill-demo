"""
mempill_showcase.config.bootstrap — CLI startup bootstrap.

Loads the repo `.env` file into ``os.environ`` so that LangSmith and Anthropic
config set there takes effect for LangGraph, the ``langsmith`` library, and any
other library that reads ``os.environ`` directly (not via pydantic-settings).

Design decisions:
  - ``bootstrap()`` is the single public entry point.
  - It is safe to call multiple times (idempotent via a module-level sentinel).
  - It MUST be called only from CLI ``main()`` functions, never at import time,
    so that test code that imports scenario/tool modules does not accidentally
    pick up a developer's local ``.env`` file.
  - ``dotenv.load_dotenv()`` with ``override=False`` (the default) means env vars
    already in ``os.environ`` (e.g. set in the shell) are never clobbered.
  - After loading ``.env``, ``configure_tracing_from_settings()`` is called to
    sync the LangSmith vars into ``os.environ`` from pydantic-settings (which may
    pick up additional validation / defaults).

Usage in CLI entry points only:

    def main() -> None:
        from mempill_showcase.config.bootstrap import bootstrap
        bootstrap()
        ...
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

_bootstrapped: bool = False


def bootstrap() -> None:
    """Load ``.env`` from the current working directory and configure LangSmith.

    Steps:
      1. ``dotenv.load_dotenv()`` — populates ``os.environ`` from ``.env``
         (safe no-op if the file is absent; never overrides already-set vars).
      2. ``configure_tracing_from_settings()`` — syncs LANGSMITH_* vars from
         pydantic-settings into ``os.environ`` so the ``langsmith`` library and
         LangGraph pick them up (no-op if no API key is present).

    Idempotent: the second and subsequent calls are no-ops.  Thread-safe for
    typical CLI use (called once before any threads are spawned).
    """
    global _bootstrapped
    if _bootstrapped:
        return

    # Step 1: load .env into os.environ
    # Use an explicit dotenv_path (cwd/.env) to avoid dotenv's default upward-search
    # behaviour, which would find a .env in a parent directory.  Specifying the path
    # explicitly means "load THIS file if it exists, otherwise silently skip".
    try:
        from pathlib import Path
        from dotenv import load_dotenv
        dotenv_path = Path.cwd() / ".env"
        loaded = load_dotenv(dotenv_path=dotenv_path)  # override=False — never clobbers shell env
        if loaded:
            log.debug("bootstrap: loaded .env from %s", dotenv_path)
        else:
            log.debug("bootstrap: .env not found at %s — continuing without it", dotenv_path)
    except ImportError:
        log.debug("bootstrap: python-dotenv not installed — skipping .env load")

    # Step 2: sync LangSmith / Anthropic vars from pydantic-settings
    try:
        from mempill_showcase.observability import configure_tracing_from_settings
        configure_tracing_from_settings()

        # Log whether tracing is active (without printing any secret values)
        langsmith_key_present = bool(os.environ.get("LANGSMITH_API_KEY", ""))
        tracing_flag = os.environ.get("LANGSMITH_TRACING", "").lower()
        if langsmith_key_present or tracing_flag == "true":
            project = os.environ.get("LANGSMITH_PROJECT", "mempill-showcase")
            log.info("bootstrap: LangSmith tracing ENABLED (project=%s)", project)
            print(f"[mempill] LangSmith tracing enabled (project={project})")
        else:
            log.debug("bootstrap: LangSmith tracing is off (no LANGSMITH_API_KEY)")
    except Exception as exc:
        log.debug("bootstrap: configure_tracing_from_settings failed (%s); skipping", exc)

    _bootstrapped = True
