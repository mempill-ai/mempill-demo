"""Shared pytest configuration for the showcase test suite."""
from __future__ import annotations

import os


def pytest_configure(config) -> None:
    # Set before any test module imports langsmith: langsmith caches env reads
    # (lru_cache on get_env_var), so a later session fixture would be too late.
    # Tests must never export traces; developer test runs exhausted the LangSmith quota.
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
