"""
mempill_showcase.core.ports.oracle — HumanOracle interface for HITL adjudication.

Lifted from mempill_demo.adapters.human_oracle and extracted as a port.
The engine duck-types this: any object with request_adjudication() qualifies.

Used in W3+ (LangGraph HITL wave). Kept minimal in W1 — just the interface.
"""
from __future__ import annotations

import uuid


class HumanOracle:
    """Stateless oracle that queues every conflict for human review.

    The mempill engine calls request_adjudication() when a write returns Contested
    and the engine was opened with open_oracle / open_oracle_in_memory. This method
    returns a handle_id UUID; the engine stores the full adjudication request in its
    durable queue. The human resolves it later via submit_adjudication().

    No mempill imports here — the engine accepts any duck-typed oracle object.
    """

    def request_adjudication(self, agent_id: str, request: dict) -> str:
        """Return a fresh UUID handle_id; the engine records the adjudication request."""
        return str(uuid.uuid4())
