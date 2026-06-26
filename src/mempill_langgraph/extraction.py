"""mempill_langgraph.extraction — Pydantic models for structured claim extraction.

Uses llm.with_structured_output(ClaimExtractResult) — NO json.loads anywhere.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ExtractedClaim(BaseModel):
    """A single factual claim extracted from an assistant message."""
    subject: str = Field(description="Entity the claim is about, e.g. 'acme:ceo', 'alice'")
    predicate: str = Field(description="Relationship or property, e.g. 'held_by', 'lives_in'")
    value: str = Field(description="The asserted value, e.g. 'Alice', 'Paris'")
    conf: float = Field(default=0.85, ge=0.0, le=1.0, description="Confidence 0-1")
    since: Optional[str] = Field(
        default=None,
        description=(
            "Start as a full ISO-8601 date YYYY-MM-DD; default unknown month/day to 01 "
            "(e.g. 'March 2020' -> '2020-03-01', '2020' -> '2020-01-01'). null if unknown."
        ),
    )
    until: Optional[str] = Field(
        default=None,
        description="End as a full ISO-8601 date YYYY-MM-DD (same rules as 'since'); null if open-ended",
    )
    is_user_asserted: bool = Field(
        default=False,
        description="True if the user explicitly stated this fact; False if inferred by model",
    )


class ClaimExtractResult(BaseModel):
    """Result of claim extraction from one assistant message."""
    claims: list[ExtractedClaim] = Field(
        default_factory=list,
        description=(
            "List of factual claims extracted. "
            "Empty list for greetings, small-talk, questions, contested reports."
        ),
    )


class KeyExtractResult(BaseModel):
    """Result of canonical key extraction from a user question (for retrieve_memory)."""
    subject: str = Field(
        default="",
        description=(
            "Canonical subject key, e.g. 'acme:ceo' or 'bob'. "
            "Empty string when no subject is identifiable (greeting/small-talk)."
        ),
    )
    predicate: str = Field(
        default="",
        description=(
            "Canonical predicate, e.g. 'held_by' or 'lives_in'. "
            "Empty string when no predicate is identifiable."
        ),
    )


class DecisionClassifyResult(BaseModel):
    """
    Result of the conversational adjudication decision classifier.

    The classifier reads the user's reply when a pending_decision is set and
    determines whether the user picked the incumbent, the challenger, or neither.
    """
    verdict: str = Field(
        default="neither",
        description=(
            "One of: 'challenger' (new claim wins), 'incumbent' (existing belief stands), "
            "or 'neither' (user did not pick, stay pending)."
        ),
    )
