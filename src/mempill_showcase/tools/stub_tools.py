"""
mempill_showcase.tools.stub_tools — CalendarTool, EmailDraftTool

Stub tools for the Scheduling crew (Crew C). These do NOT call real APIs —
they return canned, structured results so the Briefing Agent has something
deterministic to call in W2/W3/W4 before live integrations are wired.

Both tools return structured JSON so downstream agents can consume them
programmatically (not just as human-readable text).

CalendarTool:   creates a stub calendar event (no real calendar write)
EmailDraftTool: produces a stub email draft (no real send)

Replace with real API wrappers in W9/W10 without changing the tool signatures.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, List, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


# ── CalendarTool ──────────────────────────────────────────────────────────────

class CalendarInput(BaseModel):
    title: str = Field(description="Event title / subject line")
    attendees: List[str] = Field(
        default_factory=list,
        description="List of attendee names or email addresses",
    )
    date: Optional[str] = Field(
        default=None,
        description="Event date (ISO-8601 or natural language like 'next Tuesday')",
    )
    location: Optional[str] = Field(
        default=None,
        description="Event location (city, venue, or 'virtual')",
    )
    duration_minutes: int = Field(
        default=60,
        description="Duration in minutes",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Additional notes or context for the event",
    )


class CalendarTool(BaseTool):
    """Stub calendar event creator for Crew C (Scheduling & Briefing).

    Does NOT call a real calendar API. Returns a structured stub response
    containing an event_id, confirmation message, and echoed fields.
    Replace with a real integration (Google Calendar, Outlook) in W9.
    """

    name: str = "calendar_create_event"
    description: str = (
        "Create a calendar event (stub — no real calendar write). "
        "Supply title, attendees, date, location, and duration. "
        "Returns JSON with event_id, status='stub_created', and echoed fields."
    )
    args_schema: Type[BaseModel] = CalendarInput

    def _run(
        self,
        title: str,
        attendees: Optional[List[str]] = None,
        date: Optional[str] = None,
        location: Optional[str] = None,
        duration_minutes: int = 60,
        notes: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        event_id = f"stub-event-{uuid.uuid4().hex[:8]}"
        created_at = datetime.now(timezone.utc).isoformat()

        log.debug(
            "CalendarTool (stub): title=%r attendees=%s date=%s location=%s",
            title, attendees, date, location,
        )

        result = {
            "event_id": event_id,
            "status": "stub_created",
            "title": title,
            "attendees": attendees or [],
            "date": date,
            "location": location,
            "duration_minutes": duration_minutes,
            "notes": notes,
            "created_at": created_at,
            "warning": "This is a stub — no real calendar event was created.",
        }
        return json.dumps(result)

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("CalendarTool does not support async")


# ── EmailDraftTool ────────────────────────────────────────────────────────────

class EmailDraftInput(BaseModel):
    to: List[str] = Field(description="Recipient names or email addresses")
    subject: str = Field(description="Email subject line")
    body: str = Field(description="Email body text (plain text or markdown)")
    cc: Optional[List[str]] = Field(
        default=None,
        description="Optional CC recipients",
    )
    from_name: Optional[str] = Field(
        default="Jordan Park's AI Assistant",
        description="Sender display name",
    )
    priority: str = Field(
        default="normal",
        description="Email priority: 'low' | 'normal' | 'high'",
    )


class EmailDraftTool(BaseTool):
    """Stub email draft composer for Crew C (Scheduling & Briefing).

    Does NOT send a real email. Returns a structured stub response containing
    a draft_id, the full composed email fields, and a confirmation timestamp.
    Replace with a real mail integration (SendGrid, Outlook) in W9.
    """

    name: str = "email_draft"
    description: str = (
        "Compose an email draft (stub — no real send). "
        "Supply to, subject, body, and optional cc / from_name / priority. "
        "Returns JSON with draft_id, status='stub_drafted', and the full email payload."
    )
    args_schema: Type[BaseModel] = EmailDraftInput

    def _run(
        self,
        to: List[str],
        subject: str,
        body: str,
        cc: Optional[List[str]] = None,
        from_name: Optional[str] = "Jordan Park's AI Assistant",
        priority: str = "normal",
        **kwargs: Any,
    ) -> str:
        draft_id = f"stub-draft-{uuid.uuid4().hex[:8]}"
        drafted_at = datetime.now(timezone.utc).isoformat()

        log.debug(
            "EmailDraftTool (stub): to=%s subject=%r priority=%s",
            to, subject, priority,
        )

        result = {
            "draft_id": draft_id,
            "status": "stub_drafted",
            "from": from_name,
            "to": to,
            "cc": cc or [],
            "subject": subject,
            "body": body,
            "priority": priority,
            "drafted_at": drafted_at,
            "warning": "This is a stub — no real email was sent.",
        }
        return json.dumps(result)

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("EmailDraftTool does not support async")
