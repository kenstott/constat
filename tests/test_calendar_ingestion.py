# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Tests for calendar event ingestion (_calendar.py)."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import httpx
import pytest

from constat.discovery.doc_tools._calendar import (
    CalendarEvent,
    CalendarFetcher,
    EventAttachment,
    render_event,
)


# ---------------------------------------------------------------------------
# Fixtures: canned API responses
# ---------------------------------------------------------------------------

GOOGLE_SINGLE_EVENT = {
    "id": "google_evt_001",
    "summary": "Team Standup",
    "status": "confirmed",
    "start": {"dateTime": "2026-03-15T09:00:00-04:00"},
    "end": {"dateTime": "2026-03-15T09:30:00-04:00"},
    "location": "Room 42",
    "organizer": {"email": "alice@example.com"},
    "attendees": [
        {"email": "bob@example.com", "responseStatus": "accepted"},
        {"email": "carol@example.com", "responseStatus": "accepted"},
    ],
    "description": "Daily standup meeting",
    "htmlLink": "https://calendar.google.com/event?eid=abc",
}

GOOGLE_ALL_DAY_EVENT = {
    "id": "google_evt_002",
    "summary": "Company Holiday",
    "status": "confirmed",
    "start": {"date": "2026-04-01"},
    "end": {"date": "2026-04-02"},
    "organizer": {"email": "hr@example.com"},
}

GOOGLE_RECURRING_EVENT = {
    "id": "google_evt_003_20260316",
    "summary": "Weekly Review",
    "status": "confirmed",
    "start": {"dateTime": "2026-03-16T14:00:00Z"},
    "end": {"dateTime": "2026-03-16T15:00:00Z"},
    "organizer": {"email": "alice@example.com"},
    "recurringEventId": "google_evt_003",
}

GOOGLE_CANCELLED_EVENT = {
    "id": "google_evt_004",
    "summary": "Cancelled Meeting",
    "status": "cancelled",
    "start": {"dateTime": "2026-03-17T10:00:00Z"},
    "end": {"dateTime": "2026-03-17T11:00:00Z"},
    "organizer": {"email": "alice@example.com"},
}

GOOGLE_DECLINED_EVENT = {
    "id": "google_evt_005",
    "summary": "Optional Meeting",
    "status": "confirmed",
    "start": {"dateTime": "2026-03-18T10:00:00Z"},
    "end": {"dateTime": "2026-03-18T11:00:00Z"},
    "organizer": {"email": "alice@example.com"},
    "attendees": [
        {"email": "me@example.com", "self": True, "responseStatus": "declined"},
    ],
}

GOOGLE_EVENT_WITH_ATTACHMENT = {
    "id": "google_evt_006",
    "summary": "Review with Docs",
    "status": "confirmed",
    "start": {"dateTime": "2026-03-20T10:00:00Z"},
    "end": {"dateTime": "2026-03-20T11:00:00Z"},
    "organizer": {"email": "alice@example.com"},
    "attachments": [
        {
            "title": "agenda.pdf",
            "mimeType": "application/pdf",
            "fileUrl": "https://drive.google.com/file/abc",
            "fileSize": "12345",
        }
    ],
}


MS_SINGLE_EVENT = {
    "id": "ms_evt_001",
    "subject": "Sprint Planning",
    "start": {"dateTime": "2026-03-15T09:00:00.0000000"},
    "end": {"dateTime": "2026-03-15T10:00:00.0000000"},
    "isAllDay": False,
    "isCancelled": False,
    "location": {"displayName": "Conference Room B"},
    "organizer": {"emailAddress": {"address": "dave@example.com"}},
    "attendees": [
        {"emailAddress": {"address": "eve@example.com"}},
        {"emailAddress": {"address": "frank@example.com"}},
    ],
    "body": {"contentType": "html", "content": "<p>Sprint planning for Q2</p>"},
    "responseStatus": {"response": "accepted"},
    "webLink": "https://outlook.office.com/event/abc",
    "hasAttachments": True,
}

MS_DECLINED_EVENT = {
    "id": "ms_evt_002",
    "subject": "Optional Sync",
    "start": {"dateTime": "2026-03-16T10:00:00.0000000"},
    "end": {"dateTime": "2026-03-16T10:30:00.0000000"},
    "isAllDay": False,
    "isCancelled": False,
    "location": {"displayName": ""},
    "organizer": {"emailAddress": {"address": "dave@example.com"}},
    "attendees": [],
    "body": {"contentType": "text", "content": ""},
    "responseStatus": {"response": "declined"},
    "hasAttachments": False,
}

MS_ATTACHMENT_RESPONSE = {
    "value": [
        {
            "@odata.type": "#microsoft.graph.fileAttachment",
            "name": "deck.pptx",
            "contentType": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "size": 54321,
            "contentBytes": "c2xpZGUgY29udGVudA==",  # "slide content" base64
        }
    ]
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides):
    """Build a mock DocumentConfig for calendar tests."""
    defaults = {
        "type": "calendar",
        "provider": "google",
        "calendar_id": "primary",
        "since": "2026-01-01",
        "until": "2026-12-31",
        "max_events": 500,
        "expand_recurring": True,
        "include_declined": False,
        "include_cancelled": False,
        "extract_attachments": True,
        "extract_body": True,
        "calendars": [],
        "auth_type": "oauth2",
        "oauth2_client_id": "test-client-id",
        "oauth2_client_secret": "test-secret",
        "oauth2_tenant_id": None,
        "oauth2_scopes": [],
        "oauth2_token_cache": None,
        "password": None,
    }
    defaults.update(overrides)
    config = MagicMock()
    for k, v in defaults.items():
        setattr(config, k, v)
    return config


def _mock_response(json_data, status_code=200):
    """Create a mock httpx.Response."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    resp.content = b""
    return resp


# ---------------------------------------------------------------------------
# Tests: Google event parsing
# ---------------------------------------------------------------------------

class TestGoogleParsing:
    def test_parse_single_event(self):
        config = _make_config()
        fetcher = CalendarFetcher(config)
        events = fetcher._parse_google_events([GOOGLE_SINGLE_EVENT])
        assert len(events) == 1
        ev = events[0]
        assert ev.title == "Team Standup"
        assert ev.location == "Room 42"
        assert ev.organizer == "alice@example.com"
        assert len(ev.attendees) == 2
        assert ev.status == "confirmed"
        assert ev.description == "Daily standup meeting"
        assert not ev.all_day

    def test_parse_all_day_event(self):
        config = _make_config()
        fetcher = CalendarFetcher(config)
        events = fetcher._parse_google_events([GOOGLE_ALL_DAY_EVENT])
        assert len(events) == 1
        ev = events[0]
        assert ev.all_day is True
        assert ev.start == datetime(2026, 4, 1)

    def test_parse_recurring_event(self):
        config = _make_config()
        fetcher = CalendarFetcher(config)
        events = fetcher._parse_google_events([GOOGLE_RECURRING_EVENT])
        assert len(events) == 1
        assert events[0].recurrence_id == "google_evt_003"

    def test_filter_cancelled(self):
        config = _make_config(include_cancelled=False)
        fetcher = CalendarFetcher(config)
        events = fetcher._parse_google_events([GOOGLE_CANCELLED_EVENT])
        assert len(events) == 0

    def test_include_cancelled(self):
        config = _make_config(include_cancelled=True)
        fetcher = CalendarFetcher(config)
        events = fetcher._parse_google_events([GOOGLE_CANCELLED_EVENT])
        assert len(events) == 1

    def test_filter_declined(self):
        config = _make_config(include_declined=False)
        fetcher = CalendarFetcher(config)
        events = fetcher._parse_google_events([GOOGLE_DECLINED_EVENT])
        assert len(events) == 0

    def test_include_declined(self):
        config = _make_config(include_declined=True)
        fetcher = CalendarFetcher(config)
        events = fetcher._parse_google_events([GOOGLE_DECLINED_EVENT])
        assert len(events) == 1

    def test_parse_attachment(self):
        config = _make_config()
        fetcher = CalendarFetcher(config)
        events = fetcher._parse_google_events([GOOGLE_EVENT_WITH_ATTACHMENT])
        assert len(events) == 1
        assert len(events[0].attachments) == 1
        att = events[0].attachments[0]
        assert att.filename == "agenda.pdf"
        assert att.mime_type == "application/pdf"
        assert att.size == 12345

    @patch("constat.discovery.doc_tools._calendar.httpx.get")
    def test_pagination(self, mock_get):
        config = _make_config(max_events=100)
        fetcher = CalendarFetcher(config)
        fetcher._get_access_token = MagicMock(return_value="test-token")

        page1 = _mock_response({
            "items": [GOOGLE_SINGLE_EVENT],
            "nextPageToken": "page2token",
        })
        page2 = _mock_response({
            "items": [GOOGLE_ALL_DAY_EVENT],
        })
        mock_get.side_effect = [page1, page2]

        events = fetcher._fetch_google()
        assert len(events) == 2
        assert mock_get.call_count == 2


# ---------------------------------------------------------------------------
# Tests: Microsoft event parsing
# ---------------------------------------------------------------------------

class TestMicrosoftParsing:
    def test_parse_single_event(self):
        config = _make_config(provider="microsoft")
        fetcher = CalendarFetcher(config)
        fetcher._fetch_ms_attachments = MagicMock(return_value=[])

        events = fetcher._parse_microsoft_events([MS_SINGLE_EVENT])
        assert len(events) == 1
        ev = events[0]
        assert ev.title == "Sprint Planning"
        assert ev.location == "Conference Room B"
        assert ev.organizer == "dave@example.com"
        assert len(ev.attendees) == 2
        # HTML body should be converted
        assert ev.description is not None

    def test_filter_declined(self):
        config = _make_config(provider="microsoft", include_declined=False)
        fetcher = CalendarFetcher(config)
        events = fetcher._parse_microsoft_events([MS_DECLINED_EVENT])
        assert len(events) == 0

    def test_include_declined(self):
        config = _make_config(provider="microsoft", include_declined=True)
        fetcher = CalendarFetcher(config)
        events = fetcher._parse_microsoft_events([MS_DECLINED_EVENT])
        assert len(events) == 1

    @patch("constat.discovery.doc_tools._calendar.httpx.get")
    def test_ms_attachments_fetched(self, mock_get):
        config = _make_config(provider="microsoft")
        fetcher = CalendarFetcher(config)
        fetcher._get_access_token = MagicMock(return_value="test-token")

        mock_get.return_value = _mock_response(MS_ATTACHMENT_RESPONSE)
        attachments = fetcher._fetch_ms_attachments("ms_evt_001")
        assert len(attachments) == 1
        att = attachments[0]
        assert att.filename == "deck.pptx"
        assert att.content_bytes == b"slide content"

    @patch("constat.discovery.doc_tools._calendar.httpx.get")
    def test_ms_pagination(self, mock_get):
        config = _make_config(provider="microsoft", calendar_id="me", max_events=100)
        fetcher = CalendarFetcher(config)
        fetcher._get_access_token = MagicMock(return_value="test-token")
        fetcher._fetch_ms_attachments = MagicMock(return_value=[])

        page1 = _mock_response({
            "value": [MS_SINGLE_EVENT],
            "@odata.nextLink": "https://graph.microsoft.com/v1.0/me/calendarView?page=2",
        })
        page2 = _mock_response({
            "value": [MS_SINGLE_EVENT],
        })
        mock_get.side_effect = [page1, page2]

        events = fetcher._fetch_microsoft()
        assert len(events) == 2
        assert mock_get.call_count == 2


# ---------------------------------------------------------------------------
# Tests: render_event
# ---------------------------------------------------------------------------

class TestRenderEvent:
    def test_timed_event(self):
        ev = CalendarEvent(
            event_id="evt_20260315_standup_abc",
            title="Team Standup",
            start=datetime(2026, 3, 15, 9, 0, tzinfo=timezone.utc),
            end=datetime(2026, 3, 15, 9, 30, tzinfo=timezone.utc),
            all_day=False,
            location="Room 42",
            organizer="alice@example.com",
            attendees=["bob@example.com"],
            status="confirmed",
            description="Daily sync",
            recurrence_id=None,
        )
        text = render_event(ev)
        assert "# Team Standup" in text
        assert "Start:" in text
        assert "End:" in text
        assert "Location: Room 42" in text
        assert "Organizer: alice@example.com" in text
        assert "Attendees: bob@example.com" in text
        assert "Daily sync" in text

    def test_all_day_event(self):
        ev = CalendarEvent(
            event_id="evt_20260401_holiday_def",
            title="Company Holiday",
            start=datetime(2026, 4, 1),
            end=datetime(2026, 4, 2),
            all_day=True,
            location=None,
            organizer="hr@example.com",
            attendees=[],
            status="confirmed",
            description=None,
            recurrence_id=None,
        )
        text = render_event(ev)
        assert "Date: 2026-04-01 (all day)" in text
        assert "Start:" not in text
        assert "Location:" not in text  # None location omitted

    def test_attachments_rendered(self):
        ev = CalendarEvent(
            event_id="evt_test",
            title="Meeting",
            start=datetime(2026, 3, 20, 10, 0, tzinfo=timezone.utc),
            end=datetime(2026, 3, 20, 11, 0, tzinfo=timezone.utc),
            all_day=False,
            location=None,
            organizer="alice@example.com",
            attendees=[],
            status="confirmed",
            description=None,
            recurrence_id=None,
            attachments=[
                EventAttachment(
                    filename="agenda.pdf",
                    mime_type="application/pdf",
                    url="https://example.com/file",
                    size=1234,
                )
            ],
        )
        text = render_event(ev)
        assert "## Attachments" in text
        assert "- agenda.pdf (application/pdf)" in text

    def test_attendee_truncation(self):
        attendees = [f"user{i}@example.com" for i in range(60)]
        ev = CalendarEvent(
            event_id="evt_test",
            title="Big Meeting",
            start=datetime(2026, 3, 20, 10, 0, tzinfo=timezone.utc),
            end=datetime(2026, 3, 20, 11, 0, tzinfo=timezone.utc),
            all_day=False,
            location=None,
            organizer="alice@example.com",
            attendees=attendees,
            status="confirmed",
            description=None,
            recurrence_id=None,
        )
        text = render_event(ev)
        assert "and 10 more" in text


# ---------------------------------------------------------------------------
# Tests: _make_event_id
# ---------------------------------------------------------------------------

class TestMakeEventId:
    def test_deterministic(self):
        item = {"id": "abc123", "summary": "Standup"}
        dt = datetime(2026, 3, 15, 9, 0)
        id1 = CalendarFetcher._make_event_id(item, dt)
        id2 = CalendarFetcher._make_event_id(item, dt)
        assert id1 == id2

    def test_format(self):
        item = {"id": "xyz", "summary": "Team Standup!"}
        dt = datetime(2026, 3, 15)
        eid = CalendarFetcher._make_event_id(item, dt)
        assert eid.startswith("evt_20260315_")
        assert "teamstandup" in eid

    def test_different_dates_differ(self):
        item = {"id": "abc123", "summary": "Standup"}
        id1 = CalendarFetcher._make_event_id(item, datetime(2026, 3, 15))
        id2 = CalendarFetcher._make_event_id(item, datetime(2026, 3, 16))
        assert id1 != id2

    def test_missing_title_uses_event(self):
        item = {"id": "abc123"}
        dt = datetime(2026, 3, 15)
        eid = CalendarFetcher._make_event_id(item, dt)
        assert "event" in eid

    def test_microsoft_subject_field(self):
        item = {"id": "ms123", "subject": "Sprint Planning"}
        dt = datetime(2026, 3, 15)
        eid = CalendarFetcher._make_event_id(item, dt)
        assert "sprintplanning" in eid


# ---------------------------------------------------------------------------
# Tests: time range calculation
# ---------------------------------------------------------------------------

class TestTimeRange:
    def test_custom_range(self):
        config = _make_config(since="2026-01-01", until="2026-06-30")
        fetcher = CalendarFetcher(config)
        since, until = fetcher._get_time_range()
        assert since.year == 2026
        assert since.month == 1
        assert until.month == 6

    def test_default_range(self):
        config = _make_config(since=None, until=None)
        fetcher = CalendarFetcher(config)
        since, until = fetcher._get_time_range()
        # Should be roughly +/- 90 days from now
        assert since.tzinfo is not None
        assert until.tzinfo is not None
        assert until > since

    def test_timezone_aware(self):
        config = _make_config(since="2026-01-01T00:00:00+00:00", until="2026-06-30T00:00:00+00:00")
        fetcher = CalendarFetcher(config)
        since, until = fetcher._get_time_range()
        assert since.tzinfo is not None
        assert until.tzinfo is not None
