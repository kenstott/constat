# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Calendar event ingestion — fetches events from Google Calendar or Microsoft Graph."""

from __future__ import annotations

import base64
import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from constat.core.config import DocumentConfig

logger = logging.getLogger(__name__)


@dataclass
class EventAttachment:
    """An attachment on a calendar event."""

    filename: str
    mime_type: str
    url: str  # download URL (Google) or empty for inline MS Graph
    size: int | None
    content_bytes: bytes | None = None  # populated for MS Graph inline attachments


@dataclass
class CalendarEvent:
    """A single calendar event (or recurring instance)."""

    event_id: str
    title: str
    start: datetime
    end: datetime
    all_day: bool
    location: str | None
    organizer: str
    attendees: list[str]
    status: str  # confirmed, tentative, cancelled
    description: str | None
    recurrence_id: str | None
    attachments: list[EventAttachment] = field(default_factory=list)
    html_link: str | None = None


class CalendarFetcher:
    """Fetches events from Google Calendar or Microsoft Graph API."""

    def __init__(self, config: "DocumentConfig", config_dir: Path | None = None):
        self._config = config
        self._config_dir = config_dir

    def fetch_events(self) -> list[CalendarEvent]:
        """Fetch events from the configured provider."""
        if self._config.provider == "google":
            return self._fetch_google()
        elif self._config.provider == "microsoft":
            return self._fetch_microsoft()
        raise ValueError(f"Unknown calendar provider: {self._config.provider}")

    def _get_time_range(self) -> tuple[datetime, datetime]:
        """Parse since/until or default to +/- 90 days."""
        now = datetime.now(timezone.utc)
        since = (
            datetime.fromisoformat(self._config.since)
            if self._config.since
            else now - timedelta(days=90)
        )
        until = (
            datetime.fromisoformat(self._config.until)
            if self._config.until
            else now + timedelta(days=90)
        )
        # Ensure timezone-aware
        if since.tzinfo is None:
            since = since.replace(tzinfo=timezone.utc)
        if until.tzinfo is None:
            until = until.replace(tzinfo=timezone.utc)
        return since, until

    def _get_access_token(self) -> str:
        """Get an OAuth2 access token via the existing IMAP OAuth2 providers."""
        from constat.discovery.doc_tools._imap import (
            AzureOAuth2Provider,
            GoogleOAuth2Provider,
            RefreshTokenOAuth2Provider,
        )

        if self._config.auth_type == "oauth2_refresh":
            return RefreshTokenOAuth2Provider(self._config).get_access_token()
        if self._config.provider == "microsoft":
            return AzureOAuth2Provider(self._config).get_access_token()
        return GoogleOAuth2Provider(self._config).get_access_token()

    # -----------------------------------------------------------------------
    # Google Calendar API v3
    # -----------------------------------------------------------------------

    def _fetch_google(self) -> list[CalendarEvent]:
        """Fetch events from Google Calendar API."""
        token = self._get_access_token()
        since, until = self._get_time_range()
        calendar_ids = self._config.calendars or [self._config.calendar_id]
        all_events: list[CalendarEvent] = []

        for cal_id in calendar_ids:
            url = f"https://www.googleapis.com/calendar/v3/calendars/{cal_id}/events"
            params: dict = {
                "timeMin": since.isoformat(),
                "timeMax": until.isoformat(),
                "singleEvents": str(self._config.expand_recurring).lower(),
                "orderBy": "startTime" if self._config.expand_recurring else "updated",
                "maxResults": min(self._config.max_events, 2500),
            }
            headers = {"Authorization": f"Bearer {token}"}

            raw_items: list[dict] = []
            while True:
                resp = httpx.get(url, params=params, headers=headers, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                raw_items.extend(data.get("items", []))
                page_token = data.get("nextPageToken")
                if not page_token or len(raw_items) >= self._config.max_events:
                    break
                params["pageToken"] = page_token

            all_events.extend(
                self._parse_google_events(raw_items[: self._config.max_events])
            )

        return all_events

    def _parse_google_events(self, items: list[dict]) -> list[CalendarEvent]:
        """Parse raw Google Calendar API items into CalendarEvent objects."""
        events: list[CalendarEvent] = []
        for item in items:
            if not self._config.include_cancelled and item.get("status") == "cancelled":
                continue
            if not self._config.include_declined and self._is_declined_google(item):
                continue

            start = item.get("start", {})
            end = item.get("end", {})
            all_day = "date" in start
            start_dt = datetime.fromisoformat(start.get("dateTime") or start.get("date"))
            end_dt = datetime.fromisoformat(end.get("dateTime") or end.get("date"))

            attachments: list[EventAttachment] = []
            for att in item.get("attachments", []):
                attachments.append(
                    EventAttachment(
                        filename=att.get("title", "attachment"),
                        mime_type=att.get("mimeType", "application/octet-stream"),
                        url=att.get("fileUrl", ""),
                        size=int(att["fileSize"]) if att.get("fileSize") else None,
                    )
                )

            event_id = self._make_event_id(item, start_dt)
            events.append(
                CalendarEvent(
                    event_id=event_id,
                    title=item.get("summary", "(No title)"),
                    start=start_dt,
                    end=end_dt,
                    all_day=all_day,
                    location=item.get("location"),
                    organizer=item.get("organizer", {}).get("email", ""),
                    attendees=[
                        a.get("email", "") for a in item.get("attendees", [])
                    ],
                    status=item.get("status", "confirmed"),
                    description=item.get("description") if self._config.extract_body else None,
                    recurrence_id=item.get("recurringEventId"),
                    attachments=attachments,
                    html_link=item.get("htmlLink"),
                )
            )
        return events

    @staticmethod
    def _is_declined_google(item: dict) -> bool:
        """Check if the authenticated user declined a Google event."""
        for attendee in item.get("attendees", []):
            if attendee.get("self") and attendee.get("responseStatus") == "declined":
                return True
        return False

    # -----------------------------------------------------------------------
    # Microsoft Graph API
    # -----------------------------------------------------------------------

    def _fetch_microsoft(self) -> list[CalendarEvent]:
        """Fetch events from Microsoft Graph calendarView endpoint."""
        token = self._get_access_token()
        since, until = self._get_time_range()
        cal_id = self._config.calendar_id

        if cal_id == "me":
            url = "https://graph.microsoft.com/v1.0/me/calendarView"
        else:
            url = f"https://graph.microsoft.com/v1.0/me/calendars/{cal_id}/calendarView"

        params: dict = {
            "startDateTime": since.isoformat(),
            "endDateTime": until.isoformat(),
            "$top": min(self._config.max_events, 1000),
            "$select": (
                "id,subject,start,end,isAllDay,location,organizer,"
                "attendees,body,webLink,hasAttachments,isCancelled,"
                "responseStatus"
            ),
        }
        headers = {
            "Authorization": f"Bearer {token}",
            "Prefer": 'outlook.timezone="UTC"',
        }

        raw_items: list[dict] = []
        while True:
            resp = httpx.get(url, params=params, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            raw_items.extend(data.get("value", []))
            next_link = data.get("@odata.nextLink")
            if not next_link or len(raw_items) >= self._config.max_events:
                break
            url = next_link
            params = {}  # next_link includes params

        return self._parse_microsoft_events(raw_items[: self._config.max_events])

    def _parse_microsoft_events(self, items: list[dict]) -> list[CalendarEvent]:
        """Parse raw Microsoft Graph items into CalendarEvent objects."""
        events: list[CalendarEvent] = []
        for item in items:
            if not self._config.include_cancelled and item.get("isCancelled", False):
                continue
            if not self._config.include_declined and self._is_declined_microsoft(item):
                continue

            start_dt = datetime.fromisoformat(item["start"]["dateTime"])
            end_dt = datetime.fromisoformat(item["end"]["dateTime"])

            attachments: list[EventAttachment] = []
            if item.get("hasAttachments") and self._config.extract_attachments:
                attachments = self._fetch_ms_attachments(item["id"])

            event_id = self._make_event_id(item, start_dt)

            body_content = None
            if self._config.extract_body:
                body = item.get("body", {})
                body_content = body.get("content")
                # Convert HTML body to markdown
                if body_content and body.get("contentType") == "html":
                    from ._file_extractors import _convert_html_to_markdown

                    body_content = _convert_html_to_markdown(body_content)

            events.append(
                CalendarEvent(
                    event_id=event_id,
                    title=item.get("subject", "(No title)"),
                    start=start_dt,
                    end=end_dt,
                    all_day=item.get("isAllDay", False),
                    location=item.get("location", {}).get("displayName"),
                    organizer=item.get("organizer", {})
                    .get("emailAddress", {})
                    .get("address", ""),
                    attendees=[
                        a["emailAddress"]["address"]
                        for a in item.get("attendees", [])
                        if "emailAddress" in a and "address" in a["emailAddress"]
                    ],
                    status=item.get("responseStatus", {}).get("response", "none"),
                    description=body_content,
                    recurrence_id=None,  # calendarView already expands
                    attachments=attachments,
                    html_link=item.get("webLink"),
                )
            )
        return events

    @staticmethod
    def _is_declined_microsoft(item: dict) -> bool:
        """Check if the user declined a Microsoft event."""
        response = item.get("responseStatus", {}).get("response", "")
        return response == "declined"

    def _fetch_ms_attachments(self, event_id: str) -> list[EventAttachment]:
        """Fetch attachments for a single Microsoft event."""
        token = self._get_access_token()
        url = f"https://graph.microsoft.com/v1.0/me/events/{event_id}/attachments"
        resp = httpx.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=30)
        resp.raise_for_status()
        attachments: list[EventAttachment] = []
        for att in resp.json().get("value", []):
            if att.get("@odata.type") == "#microsoft.graph.fileAttachment":
                content_bytes = None
                if att.get("contentBytes"):
                    content_bytes = base64.b64decode(att["contentBytes"])
                attachments.append(
                    EventAttachment(
                        filename=att.get("name", "attachment"),
                        mime_type=att.get("contentType", "application/octet-stream"),
                        url="",  # data inline for MS Graph
                        size=att.get("size"),
                        content_bytes=content_bytes,
                    )
                )
        return attachments

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _make_event_id(item: dict, start_dt: datetime) -> str:
        """Deterministic short ID from event data + date."""
        raw_id = item.get("id", "")
        short_hash = hashlib.sha256(raw_id.encode()).hexdigest()[:8]
        date_str = start_dt.strftime("%Y%m%d")
        title_slug = re.sub(
            r"[^a-z0-9]",
            "",
            (item.get("summary") or item.get("subject") or "event").lower(),
        )[:20]
        return f"evt_{date_str}_{title_slug}_{short_hash}"

    def download_attachment(self, att: EventAttachment) -> bytes:
        """Download attachment bytes.

        For MS Graph, content_bytes is already populated inline.
        For Google, downloads via the fileUrl.
        """
        if att.content_bytes is not None:
            return att.content_bytes
        if not att.url:
            raise ValueError(f"No URL or content for attachment: {att.filename}")
        token = self._get_access_token()
        resp = httpx.get(
            att.url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.content


def render_event(event: CalendarEvent) -> str:
    """Render a calendar event as markdown document text."""
    parts = [f"# {event.title}"]
    if event.all_day:
        parts.append(f"Date: {event.start.strftime('%Y-%m-%d')} (all day)")
    else:
        parts.append(f"Start: {event.start.isoformat()}")
        parts.append(f"End: {event.end.isoformat()}")
    if event.location:
        parts.append(f"Location: {event.location}")
    parts.append(f"Organizer: {event.organizer}")
    if event.attendees:
        if len(event.attendees) > 50:
            shown = event.attendees[:50]
            parts.append(
                f"Attendees: {', '.join(shown)} and {len(event.attendees) - 50} more"
            )
        else:
            parts.append(f"Attendees: {', '.join(event.attendees)}")
    parts.append(f"Status: {event.status}")
    parts.append("")
    if event.description:
        parts.append(event.description)
    if event.attachments:
        parts.append("\n## Attachments")
        for att in event.attachments:
            parts.append(f"- {att.filename} ({att.mime_type})")
    return "\n".join(parts)
