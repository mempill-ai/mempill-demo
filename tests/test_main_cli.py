"""
tests.test_main_cli — deterministic tests for mempill_demo.__main__'s
arg-parsing + engine-construction path (TASK-33 item 4).

Runs `python -m mempill_demo` as a real subprocess (no --llm, no API key
needed) with `/quit\\n` piped to stdin so the interactive REPL exits
immediately after startup — this exercises the FULL composition-root path
(arg parsing -> DB-dir/agent_id resolution -> mempill.open_oracle_for_agent
-> adapter/tools construction -> run_repl's startup banner) without needing
to factor the REPL out of __main__.py.

Coverage:
  - --db-dir + --agent creates the expected per-agent DB file
    (<dir>/agent_<agent_id>.db).
  - Default agent_id ("console-user") is used when --agent is omitted.
  - An invalid --agent id (characters outside [A-Za-z0-9_-], which
    mempill.open_oracle_for_agent rejects with mempill.StorageError) now
    prints a clean single-line error and exits 1 — NOT a raw Python
    traceback (the __main__.py fix this test file drives: item 4).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _run_cli(args: list[str], stdin_text: str = "/quit\n", timeout: float = 60.0):
    """Run `python -m mempill_demo <args>` as a subprocess with piped stdin.

    Returns the completed CompletedProcess (stdout/stderr captured as text).
    """
    return subprocess.run(
        [sys.executable, "-m", "mempill_demo", *args],
        input=stdin_text,
        cwd=str(_PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


class TestDbDirAndAgentFlag:
    """--db-dir <dir> --agent <agent_id> creates <dir>/agent_<agent_id>.db."""

    def test_custom_db_dir_and_agent_creates_expected_db_file(self, tmp_path):
        db_dir = tmp_path / "tmp"
        result = _run_cli(["--db-dir", str(db_dir), "--agent", "alice-01"])

        assert result.returncode == 0, (
            f"CLI must exit 0, got {result.returncode}. stderr={result.stderr!r}"
        )
        expected_db = db_dir / "agent_alice-01.db"
        assert expected_db.exists(), (
            f"Expected {expected_db} to be created, dir contents: "
            f"{list(db_dir.iterdir()) if db_dir.exists() else '<missing>'}"
        )
        assert "Goodbye." in result.stdout

    def test_different_agent_ids_create_isolated_db_files(self, tmp_path):
        db_dir = tmp_path / "tmp"
        _run_cli(["--db-dir", str(db_dir), "--agent", "bob-02"])
        _run_cli(["--db-dir", str(db_dir), "--agent", "carol-03"])

        assert (db_dir / "agent_bob-02.db").exists()
        assert (db_dir / "agent_carol-03.db").exists()


class TestDefaultAgentId:
    """--agent omitted -> default agent_id "console-user" is used."""

    def test_default_agent_id_used_when_flag_absent(self, tmp_path):
        db_dir = tmp_path / "tmp"
        result = _run_cli(["--db-dir", str(db_dir)])

        assert result.returncode == 0, f"stderr={result.stderr!r}"
        expected_db = db_dir / "agent_console-user.db"
        assert expected_db.exists(), (
            f"Expected default-agent DB {expected_db} to be created, "
            f"dir contents: {list(db_dir.iterdir()) if db_dir.exists() else '<missing>'}"
        )


class TestInvalidAgentIdCleanError:
    """An invalid --agent id must produce a clean error message, not a
    raw Python traceback (item 4 fix in mempill_demo/__main__.py)."""

    def test_invalid_agent_id_exits_nonzero_with_clean_message(self, tmp_path):
        db_dir = tmp_path / "tmp"
        result = _run_cli(["--db-dir", str(db_dir), "--agent", "bad agent id!"])

        assert result.returncode == 1, (
            f"Expected clean exit code 1, got {result.returncode}. "
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )
        assert "Traceback" not in result.stderr, (
            f"Invalid agent id must not raise a raw traceback, got stderr={result.stderr!r}"
        )
        assert "Traceback" not in result.stdout
        assert "ERROR" in result.stderr
        assert "bad agent id!" in result.stderr

    def test_invalid_agent_id_does_not_create_a_db_file(self, tmp_path):
        db_dir = tmp_path / "tmp"
        _run_cli(["--db-dir", str(db_dir), "--agent", "bad/agent"])
        # No agent_<id>.db should ever be created for a rejected agent_id.
        assert not any(db_dir.glob("agent_*.db")) if db_dir.exists() else True
