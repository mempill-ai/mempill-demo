"""
mempill_showcase.tests.test_studio_router_graph_smoke — TASK-31 T31-3
deterministic import-smoke test for studio_router_graph.py (NO API key
required, NO LLM invocation).

Scope: confirms the module imports cleanly, exposes the three expected
compiled-graph attributes, and that a fresh file-backed db_dir is seeded with
EXACTLY each agent's own domain claims (no cross-contamination) — mirroring
the manual verification performed during T31-3 implementation. Full router
behavior/classification tests are Wave 3's scope (test_router_graph.py).

Isolation: runs the import in a subprocess with MEMPILL_DB_DIR pointed at a
pytest tmp_path, so this test never touches the repo's own `.mempill/` dir
and never depends on network/API keys (module import constructs ChatAnthropic
clients but never calls .invoke()).
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

_IMPORT_SNIPPET = textwrap.dedent(
    """
    import json
    from mempill_showcase.frameworks.langgraph import studio_router_graph as m

    result = {
        "graph": type(m.graph).__name__,
        "people_ops_graph": type(m.people_ops_graph).__name__,
        "org_registry_graph": type(m.org_registry_graph).__name__,
    }
    print(json.dumps(result))
    """
)


def _run_import(db_dir: Path, env_extra: dict | None = None) -> dict:
    import os

    env = os.environ.copy()
    env["MEMPILL_DB_DIR"] = str(db_dir)
    env.pop("ANTHROPIC_API_KEY", None)
    if env_extra:
        env.update(env_extra)

    proc = subprocess.run(
        [sys.executable, "-c", _IMPORT_SNIPPET],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, (
        f"studio_router_graph import failed (rc={proc.returncode}):\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    # Last non-empty stdout line is the JSON result.
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    return json.loads(lines[-1])


def test_studio_router_graph_imports_and_exposes_three_graphs(tmp_path):
    result = _run_import(tmp_path / ".mempill")

    assert result["graph"] == "CompiledStateGraph"
    assert result["people_ops_graph"] == "CompiledStateGraph"
    assert result["org_registry_graph"] == "CompiledStateGraph"


def test_studio_router_graph_seeds_domain_scoped_dbs_no_cross_contamination(tmp_path):
    db_dir = tmp_path / ".mempill"
    _run_import(db_dir)

    import sqlite3

    people_db = db_dir / "agent_people-ops-001.db"
    org_db = db_dir / "agent_org-registry-001.db"
    assert people_db.exists(), "people-ops DB file was not created"
    assert org_db.exists(), "org-registry DB file was not created"

    with sqlite3.connect(str(people_db)) as conn:
        people_rows = conn.execute(
            "select subject, predicate from claims order by subject, predicate"
        ).fetchall()
    with sqlite3.connect(str(org_db)) as conn:
        org_rows = conn.execute(
            "select subject, predicate from claims order by subject, predicate"
        ).fetchall()

    people_subjects = {row[0] for row in people_rows}
    org_subjects = {row[0] for row in org_rows}

    assert people_subjects == {"alice-chen", "bob-liu", "jordan-park"}
    assert org_subjects == {"acme-corp"}
    assert len(people_rows) == 6
    assert len(org_rows) == 2
    # No cross-domain leakage.
    assert people_subjects.isdisjoint(org_subjects)


def test_studio_router_graph_reimport_is_idempotent(tmp_path):
    db_dir = tmp_path / ".mempill"

    _run_import(db_dir)

    import sqlite3

    def _count(db_path: Path) -> int:
        with sqlite3.connect(str(db_path)) as conn:
            return conn.execute("select count(*) from claims").fetchone()[0]

    people_db = db_dir / "agent_people-ops-001.db"
    org_db = db_dir / "agent_org-registry-001.db"

    first_people_count = _count(people_db)
    first_org_count = _count(org_db)

    # Re-run the import in a fresh subprocess against the SAME db_dir.
    _run_import(db_dir)

    second_people_count = _count(people_db)
    second_org_count = _count(org_db)

    assert first_people_count == second_people_count == 6
    assert first_org_count == second_org_count == 2
