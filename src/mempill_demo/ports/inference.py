"""mempill_demo.ports.inference — InferenceSource Protocol."""
from __future__ import annotations

from typing import Protocol

from mempill_demo.domain.models import ParsedCommand


class InferenceSource(Protocol):
    def parse(self, text: str) -> ParsedCommand: ...
