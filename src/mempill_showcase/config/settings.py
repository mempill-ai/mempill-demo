"""
mempill_showcase.config.settings — Application settings via pydantic-settings.

Fields are loaded from environment variables (case-insensitive) or .env files.
All fields have safe defaults so the demo runs without any configuration.

Key env vars:
  NAIVE_MODE=true          → use the NaiveAdapter instead of mempill (flip to watch it misbehave)
  MEMPILL_DB_PATH          → optional path for a file-backed engine (not wired; reserved)
  ANTHROPIC_API_KEY        → required for LLMSupervisor; absent → MockSupervisor (CI-safe)
  ANTHROPIC_MODEL          → model string for LLMSupervisor (default: claude-3-5-haiku-20241022)
  LANGSMITH_API_KEY        → enables LangSmith tracing (optional; absent → no-op)
  LANGSMITH_TRACING        → "true" to force-enable tracing
  LANGSMITH_PROJECT        → LangSmith project name (default: "mempill-showcase")
"""
from __future__ import annotations

from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the mempill showcase.

    Populated from environment variables (case-insensitive). All fields
    have safe defaults so the demo runs without any configuration.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Adapter toggle ────────────────────────────────────────────────────────
    naive_mode: bool = False
    """When True, use NaiveAdapter (last-write-wins, no bi-temporal).
    Flip to True via NAIVE_MODE=true to watch the showcase misbehave.
    Default: False → use MempillAdapter (bi-temporal, provenance, audit).
    """

    mempill_db_path: Optional[str] = None
    """Optional file path for a persistent mempill engine.
    Reserved for future use; in-memory engine is used when None.
    """

    # ── LLM / API keys ────────────────────────────────────────────────────────
    anthropic_api_key: Optional[str] = None
    """Anthropic API key. Required for LLMSupervisor. Absent → MockSupervisor."""

    anthropic_model: str = "claude-3-5-haiku-20241022"
    """Anthropic model for LLMSupervisor."""

    langsmith_api_key: Optional[str] = None
    """LangSmith API key. Optional; absent → tracing is a no-op."""

    langsmith_tracing: Optional[str] = None
    """Set to 'true' to force-enable LangSmith tracing."""

    langsmith_project: str = "mempill-showcase"
    """LangSmith project name."""

    @property
    def has_anthropic_key(self) -> bool:
        """True if an Anthropic API key is configured."""
        return bool(self.anthropic_api_key)

    @property
    def has_langsmith_key(self) -> bool:
        """True if LangSmith tracing can be enabled."""
        return bool(self.langsmith_api_key)


def get_settings() -> Settings:
    """Return a Settings instance populated from the environment.

    Cached per-process (Settings is stateless once constructed). Safe to call
    multiple times; pydantic-settings reads env vars once at construction time.
    """
    return Settings()
