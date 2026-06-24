"""
mempill_demo.adapters.human_oracle — stateless HumanOracle adapter.

Duck-typed oracle for mempill.open_oracle / open_oracle_in_memory.
The engine calls request_adjudication() when a conflict needs human review;
we return a fresh handle_id UUID. The engine stores the adjudication request
in its durable queue; the human resolves it later via /review.
"""
from __future__ import annotations

import uuid


class HumanOracle:
    """
    Stateless oracle that queues every adjudication request for human review.

    The engine is the durable store — this class only generates a handle_id.
    No mempill imports allowed here (imported by adapters/memory_mempill.py only).
    """

    def request_adjudication(self, agent_id: str, request: dict) -> str:
        """Return a fresh UUID handle_id string; the engine records the rest."""
        return str(uuid.uuid4())
