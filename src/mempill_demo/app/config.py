"""mempill_demo.app.config — AppConfig dataclass from argparse."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AppConfig:
    db_path: str = ".mempill/console.db"
    agent_id: str = "console-user"
    llm_mode: bool = False
    scenario: bool = False
    selftest: bool = False
    reset: bool = False
