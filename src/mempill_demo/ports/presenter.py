"""mempill_demo.ports.presenter — Presenter Protocol."""
from __future__ import annotations

from typing import Protocol

from mempill_demo.domain.models import (
    AgentResponse,
    AuditEntry,
    BeliefView,
    ClaimMeta,
    SessionStats,
)


class Presenter(Protocol):
    def render(self, response: AgentResponse) -> None: ...
    def render_panel(
        self,
        audit_entries: list[AuditEntry],
        beliefs: list[BeliefView],
        registry: dict[str, ClaimMeta],
        stats: SessionStats,
    ) -> None: ...
    def render_startup_audit(self, entries: list[AuditEntry]) -> None: ...
