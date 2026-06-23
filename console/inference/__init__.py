"""console.inference — command parsing layer (deterministic + optional LLM)."""
from console.inference.base import ParsedCommand, CommandKind
from console.inference.deterministic import DeterministicParser

__all__ = ["ParsedCommand", "CommandKind", "DeterministicParser"]
