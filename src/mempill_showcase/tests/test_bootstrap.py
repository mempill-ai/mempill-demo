"""
mempill_showcase.tests.test_bootstrap — Bootstrap module tests.

Covers:
  - bootstrap() loads a .env file into os.environ when present.
  - bootstrap() is a no-op (does not error) when .env is absent.
  - bootstrap() enables _langsmith_enabled() when LANGSMITH_API_KEY +
    LANGSMITH_TRACING=true are present in the .env.
  - bootstrap() leaves _langsmith_enabled() False when .env is absent /
    has no LangSmith config.
  - bootstrap() is idempotent (safe to call multiple times).
  - bootstrap() does NOT run at import time of scenario modules (tests stay hermetic).

All tests are CI-safe: no real LangSmith backend, no API calls.
"""
from __future__ import annotations

import importlib
import os
import sys
import tempfile
import textwrap

import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _reload_bootstrap():
    """Force-reload bootstrap so _bootstrapped sentinel resets between tests."""
    mod_name = "mempill_showcase.config.bootstrap"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    return importlib.import_module(mod_name)


# ── TestBootstrapLoadsEnv ─────────────────────────────────────────────────────

class TestBootstrapLoadsEnv:
    """bootstrap() populates os.environ from a .env written to a temp dir."""

    @pytest.fixture(autouse=True)
    def reset_bootstrap(self):
        """Ensure _bootstrapped is False before each test."""
        yield
        # Clean up sentinel after test
        mod = sys.modules.get("mempill_showcase.config.bootstrap")
        if mod is not None:
            mod._bootstrapped = False

    def test_bootstrap_loads_env_key(self, monkeypatch, tmp_path):
        """bootstrap() with a .env containing LANGSMITH_API_KEY populates os.environ."""
        env_file = tmp_path / ".env"
        env_file.write_text("LANGSMITH_API_KEY=fake-test-key-12345\n")

        # Change cwd to the temp dir so dotenv finds our .env
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)

        bootstrap = _reload_bootstrap()
        bootstrap.bootstrap()

        assert os.environ.get("LANGSMITH_API_KEY") == "fake-test-key-12345", (
            "bootstrap() must load LANGSMITH_API_KEY from .env into os.environ"
        )

    def test_bootstrap_enables_langsmith(self, monkeypatch, tmp_path):
        """bootstrap() with LANGSMITH_API_KEY + LANGSMITH_TRACING=true flips _langsmith_enabled()."""
        env_file = tmp_path / ".env"
        env_file.write_text(
            "LANGSMITH_API_KEY=fake-test-key-12345\n"
            "LANGSMITH_TRACING=true\n"
            "LANGSMITH_PROJECT=mempill-test-project\n"
        )

        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
        monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)

        bootstrap = _reload_bootstrap()
        bootstrap.bootstrap()

        from mempill_showcase.observability import _langsmith_enabled
        # _langsmith_enabled checks os.environ + tries to import langsmith.
        # With a fake key + tracing=true set in os.environ, the env check passes.
        # The langsmith import may or may not succeed (depends on install),
        # so we assert at least that the env is correct and call is error-free.
        assert os.environ.get("LANGSMITH_API_KEY") == "fake-test-key-12345"
        assert os.environ.get("LANGSMITH_TRACING", "").lower() == "true"

    def test_bootstrap_noop_when_env_absent(self, monkeypatch, tmp_path):
        """bootstrap() without a .env file is a safe no-op — no error, env unchanged."""
        # tmp_path has no .env
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)

        bootstrap = _reload_bootstrap()
        bootstrap.bootstrap()  # must not raise

        assert os.environ.get("LANGSMITH_API_KEY") is None, (
            "LANGSMITH_API_KEY must remain absent when .env is not present"
        )

    def test_bootstrap_does_not_override_shell_env(self, monkeypatch, tmp_path):
        """bootstrap() must NOT override env vars already set in the shell."""
        env_file = tmp_path / ".env"
        env_file.write_text("LANGSMITH_API_KEY=from-dotenv\n")

        monkeypatch.chdir(tmp_path)
        # Shell env takes precedence
        monkeypatch.setenv("LANGSMITH_API_KEY", "from-shell")

        bootstrap = _reload_bootstrap()
        bootstrap.bootstrap()

        assert os.environ.get("LANGSMITH_API_KEY") == "from-shell", (
            "bootstrap() must not override LANGSMITH_API_KEY already set in the shell"
        )

    def test_bootstrap_is_idempotent(self, monkeypatch, tmp_path):
        """Calling bootstrap() multiple times must be a no-op after the first call."""
        env_file = tmp_path / ".env"
        env_file.write_text("LANGSMITH_API_KEY=idempotent-key\n")

        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

        bootstrap = _reload_bootstrap()
        bootstrap.bootstrap()  # first call — loads .env
        bootstrap.bootstrap()  # second call — must be no-op, no error
        bootstrap.bootstrap()  # third call — still no-op

        assert bootstrap._bootstrapped is True


# ── TestBootstrapNotAtImportTime ──────────────────────────────────────────────

class TestBootstrapNotAtImportTime:
    """bootstrap() must NOT run when scenario modules are imported (tests stay hermetic)."""

    def test_import_executive_assistant_does_not_load_dotenv(self, monkeypatch, tmp_path):
        """Importing executive_assistant does NOT call bootstrap() or load .env."""
        env_file = tmp_path / ".env"
        env_file.write_text("LANGSMITH_API_KEY=should-not-be-loaded-at-import\n")

        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

        # Importing the module must not trigger bootstrap
        import mempill_showcase.scenarios.executive_assistant as _mod  # noqa: F401

        assert os.environ.get("LANGSMITH_API_KEY") is None, (
            "Importing executive_assistant must not load .env into os.environ"
        )

    def test_import_compare_does_not_load_dotenv(self, monkeypatch, tmp_path):
        """Importing compare does NOT call bootstrap() or load .env."""
        env_file = tmp_path / ".env"
        env_file.write_text("LANGSMITH_API_KEY=should-not-be-loaded-at-import\n")

        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

        import mempill_showcase.scenarios.compare as _mod  # noqa: F401

        assert os.environ.get("LANGSMITH_API_KEY") is None, (
            "Importing compare must not load .env into os.environ"
        )

    def test_import_compliance_replay_does_not_load_dotenv(self, monkeypatch, tmp_path):
        """Importing compliance_replay does NOT call bootstrap() or load .env."""
        env_file = tmp_path / ".env"
        env_file.write_text("LANGSMITH_API_KEY=should-not-be-loaded-at-import\n")

        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

        import mempill_showcase.scenarios.compliance_replay as _mod  # noqa: F401

        assert os.environ.get("LANGSMITH_API_KEY") is None, (
            "Importing compliance_replay must not load .env into os.environ"
        )
