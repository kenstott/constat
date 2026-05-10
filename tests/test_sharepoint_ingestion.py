# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Tests for SharePoint document source (_sharepoint.py)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import httpx
import pytest

from constat.discovery.doc_tools._sharepoint import (
    SharePointClient,
    SPLibrary,
    SPList,
    SPPage,
)


# ---------------------------------------------------------------------------
# Fixtures: mock config and canned API responses
# ---------------------------------------------------------------------------


def _make_config(**overrides):
    """Build a mock DocumentConfig for SharePoint tests."""
    defaults = {
        "type": "sharepoint",
        "site_url": "https://contoso.sharepoint.com/sites/analytics",
        "discover_libraries": True,
        "discover_lists": True,
        "discover_calendars": True,
        "discover_pages": True,
        "library_names": None,
        "list_names": None,
        "calendar_names": None,
        "max_rows": 5000,
        "list_as_table": True,
        "page_types": None,
        "folder_path": None,
        "recursive": True,
        "max_files": 200,
        "include_types": None,
        "exclude_patterns": None,
        "include_trashed": False,
        "since": None,
        "oauth2_client_id": "test-client-id",
        "oauth2_client_secret": "test-secret",
        "oauth2_tenant_id": "test-tenant",
        "oauth2_scopes": ["https://graph.microsoft.com/.default"],
        "oauth2_token_cache": None,
        "auth_type": "oauth2",
    }
    defaults.update(overrides)
    config = MagicMock()
    for k, v in defaults.items():
        setattr(config, k, v)
    return config


SITE_RESPONSE = {
    "id": "contoso.sharepoint.com,abc123,def456",
    "displayName": "Analytics",
    "webUrl": "https://contoso.sharepoint.com/sites/analytics",
}

LISTS_RESPONSE = {
    "value": [
        {
            "id": "lib-001",
            "displayName": "Shared Documents",
            "list": {"template": "documentLibrary", "contentTypesEnabled": 42},
            "drive": {"id": "drive-001"},
        },
        {
            "id": "list-001",
            "displayName": "Project Tracker",
            "list": {"template": "genericList", "contentTypesEnabled": 10},
        },
        {
            "id": "cal-001",
            "displayName": "Team Calendar",
            "list": {"template": "events", "contentTypesEnabled": 5},
        },
    ],
}

COLUMNS_RESPONSE = {
    "value": [
        {"name": "Title", "text": {}},
        {"name": "Status", "text": {}},
        {"name": "Priority", "number": {}},
    ],
}

LIST_ITEMS_RESPONSE = {
    "value": [
        {"id": "item-1", "fields": {"Title": "Task A", "Status": "Done", "Priority": "1"}},
        {"id": "item-2", "fields": {"Title": "Task B", "Status": "Open", "Priority": "2"}},
    ],
}

LIST_ITEMS_PAGE2_RESPONSE = {
    "value": [
        {"id": "item-3", "fields": {"Title": "Task C", "Status": "Open", "Priority": "3"}},
    ],
}

PAGES_RESPONSE = {
    "value": [
        {
            "id": "page-001",
            "name": "Welcome.aspx",
            "title": "Welcome",
            "pageLayout": "article",
            "webUrl": "https://contoso.sharepoint.com/sites/analytics/SitePages/Welcome.aspx",
            "description": "Welcome to the analytics site",
        },
        {
            "id": "page-002",
            "name": "OldWiki.aspx",
            "title": "Old Wiki Page",
            "pageLayout": "wiki",
            "webUrl": "https://contoso.sharepoint.com/sites/analytics/SitePages/OldWiki.aspx",
            "description": "Legacy wiki content",
        },
    ],
}

CALENDAR_ITEMS_RESPONSE = {
    "value": [
        {
            "id": "evt-001",
            "fields": {
                "Title": "Team Standup",
                "EventDate": "2026-03-15T09:00:00Z",
                "EndDate": "2026-03-15T09:30:00Z",
                "fAllDayEvent": False,
                "Location": "Room 42",
                "Author": {"email": "alice@contoso.com"},
                "Description": "Daily standup meeting",
            },
        },
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_response(data: dict, status_code: int = 200) -> httpx.Response:
    """Create a mock httpx.Response."""
    return httpx.Response(
        status_code=status_code,
        json=data,
        request=httpx.Request("GET", "https://graph.microsoft.com/v1.0/test"),
    )


# ---------------------------------------------------------------------------
# Tests: API auto-detection
# ---------------------------------------------------------------------------


class TestAPIDetection:
    def test_graph_for_sharepoint_com(self):
        config = _make_config(site_url="https://contoso.sharepoint.com/sites/analytics")
        client = SharePointClient(config)
        assert client._api == "graph"

    def test_rest_for_on_premises(self):
        config = _make_config(site_url="https://sp.company.local/sites/hr")
        client = SharePointClient(config)
        assert client._api == "rest"

    def test_rest_for_custom_domain(self):
        config = _make_config(site_url="https://intranet.corp.com/sites/docs")
        client = SharePointClient(config)
        assert client._api == "rest"


# ---------------------------------------------------------------------------
# Tests: Site ID resolution
# ---------------------------------------------------------------------------


class TestSiteIDResolution:
    @patch("constat.discovery.doc_tools._sharepoint.httpx.get")
    def test_resolve_site_id(self, mock_get):
        mock_get.return_value = _mock_response(SITE_RESPONSE)
        config = _make_config()
        client = SharePointClient(config)
        site_id = client._get_site_id("test-token")

        assert site_id == "contoso.sharepoint.com,abc123,def456"
        call_args = mock_get.call_args
        assert "contoso.sharepoint.com:/sites/analytics" in call_args[0][0]


# ---------------------------------------------------------------------------
# Tests: Site discovery
# ---------------------------------------------------------------------------


class TestSiteDiscovery:
    @patch("constat.discovery.doc_tools._sharepoint.httpx.get")
    @patch.object(SharePointClient, "_get_access_token", return_value="test-token")
    def test_discover_site_filters_by_template(self, _mock_token, mock_get):
        def side_effect(url, **kwargs):
            if "/sites/" in url and url.endswith("/analytics"):
                return _mock_response(SITE_RESPONSE)
            if "/lists" in url and "/columns" in url:
                return _mock_response(COLUMNS_RESPONSE)
            if "/lists" in url and "/items" not in url:
                return _mock_response(LISTS_RESPONSE)
            if "/pages" in url:
                return _mock_response(PAGES_RESPONSE)
            # Page content fetch
            if "sitePage" in url:
                return _mock_response({"id": "page-001", "title": "Welcome"})
            return _mock_response({})

        mock_get.side_effect = side_effect
        config = _make_config()
        client = SharePointClient(config)
        result = client.discover_site()

        assert len(result["libraries"]) == 1
        assert result["libraries"][0].name == "Shared Documents"
        assert result["libraries"][0].drive_id == "drive-001"

        assert len(result["lists"]) == 1
        assert result["lists"][0].name == "Project Tracker"

        assert len(result["calendars"]) == 1
        assert result["calendars"][0].name == "Team Calendar"

        assert len(result["pages"]) == 2


# ---------------------------------------------------------------------------
# Tests: Library/list name filtering
# ---------------------------------------------------------------------------


class TestNameFiltering:
    @patch("constat.discovery.doc_tools._sharepoint.httpx.get")
    @patch.object(SharePointClient, "_get_access_token", return_value="test-token")
    def test_library_name_filter(self, _mock_token, mock_get):
        def side_effect(url, **kwargs):
            if url.endswith("/analytics"):
                return _mock_response(SITE_RESPONSE)
            if "/lists" in url and "/columns" not in url:
                return _mock_response(LISTS_RESPONSE)
            if "/pages" in url:
                return _mock_response({"value": []})
            return _mock_response(COLUMNS_RESPONSE)

        mock_get.side_effect = side_effect
        config = _make_config(library_names=["Nonexistent Library"])
        client = SharePointClient(config)
        result = client.discover_site()

        assert len(result["libraries"]) == 0

    @patch("constat.discovery.doc_tools._sharepoint.httpx.get")
    @patch.object(SharePointClient, "_get_access_token", return_value="test-token")
    def test_list_name_filter(self, _mock_token, mock_get):
        def side_effect(url, **kwargs):
            if url.endswith("/analytics"):
                return _mock_response(SITE_RESPONSE)
            if "/lists" in url and "/columns" not in url:
                return _mock_response(LISTS_RESPONSE)
            if "/pages" in url:
                return _mock_response({"value": []})
            return _mock_response(COLUMNS_RESPONSE)

        mock_get.side_effect = side_effect
        config = _make_config(list_names=["Project Tracker"])
        client = SharePointClient(config)
        result = client.discover_site()

        assert len(result["lists"]) == 1
        assert result["lists"][0].name == "Project Tracker"


# ---------------------------------------------------------------------------
# Tests: List item fetching with pagination
# ---------------------------------------------------------------------------


class TestListItemFetching:
    @patch("constat.discovery.doc_tools._sharepoint.httpx.get")
    @patch.object(SharePointClient, "_get_access_token", return_value="test-token")
    def test_fetch_list_items_paginated(self, _mock_token, mock_get):
        page1 = dict(LIST_ITEMS_RESPONSE)
        page1["@odata.nextLink"] = "https://graph.microsoft.com/v1.0/next-page"

        call_count = 0

        def side_effect(url, **kwargs):
            nonlocal call_count
            if url.endswith("/analytics"):
                return _mock_response(SITE_RESPONSE)
            if "/items" in url or "next-page" in url:
                call_count += 1
                if call_count == 1:
                    return _mock_response(page1)
                return _mock_response(LIST_ITEMS_PAGE2_RESPONSE)
            return _mock_response({})

        mock_get.side_effect = side_effect
        config = _make_config()
        client = SharePointClient(config)

        sp_list = SPList(id="list-001", name="Test List", item_count=3)
        items = client.fetch_list_items(sp_list)

        assert len(items) == 3
        assert items[0]["fields"]["Title"] == "Task A"
        assert items[2]["fields"]["Title"] == "Task C"


# ---------------------------------------------------------------------------
# Tests: render_list_as_markdown
# ---------------------------------------------------------------------------


class TestRenderListAsMarkdown:
    def test_render_non_empty_list(self):
        config = _make_config()
        client = SharePointClient(config)
        sp_list = SPList(
            id="list-001",
            name="Project Tracker",
            item_count=2,
            columns=[{"name": "Title"}, {"name": "Status"}],
        )
        items = [
            {"fields": {"Title": "Task A", "Status": "Done"}},
            {"fields": {"Title": "Task B", "Status": "Open"}},
        ]
        md = client.render_list_as_markdown(sp_list, items)

        assert "# Project Tracker" in md
        assert "| Title | Status |" in md
        assert "| --- | --- |" in md
        assert "| Task A | Done |" in md
        assert "| Task B | Open |" in md

    def test_render_empty_list(self):
        config = _make_config()
        client = SharePointClient(config)
        sp_list = SPList(
            id="list-001",
            name="Empty List",
            item_count=0,
            columns=[{"name": "Title"}],
        )
        md = client.render_list_as_markdown(sp_list, [])

        assert "# Empty List" in md
        assert "(empty list)" in md


# ---------------------------------------------------------------------------
# Tests: Canvas content extraction
# ---------------------------------------------------------------------------


class TestCanvasContentExtraction:
    def test_extract_canvas_content_json(self):
        config = _make_config()
        client = SharePointClient(config)
        canvas_json = json.dumps([
            {
                "columns": [
                    {
                        "controls": [
                            {"innerHTML": "<p>Hello <strong>world</strong></p>"},
                            {"innerHTML": "<p>Second paragraph</p>"},
                        ]
                    }
                ]
            }
        ])

        text = client._extract_canvas_content(canvas_json)
        assert "Hello world" in text
        assert "Second paragraph" in text
        # HTML tags should be stripped
        assert "<p>" not in text
        assert "<strong>" not in text

    def test_extract_canvas_content_invalid_json(self):
        config = _make_config()
        client = SharePointClient(config)
        text = client._extract_canvas_content("not valid json")
        assert text == ""

    def test_extract_canvas_content_empty(self):
        config = _make_config()
        client = SharePointClient(config)
        text = client._extract_canvas_content("[]")
        assert text == ""


# ---------------------------------------------------------------------------
# Tests: Calendar event conversion
# ---------------------------------------------------------------------------


class TestCalendarEventConversion:
    @patch("constat.discovery.doc_tools._sharepoint.httpx.get")
    @patch.object(SharePointClient, "_get_access_token", return_value="test-token")
    def test_fetch_calendar_events(self, _mock_token, mock_get):
        def side_effect(url, **kwargs):
            if url.endswith("/analytics"):
                return _mock_response(SITE_RESPONSE)
            if "/items" in url:
                return _mock_response(CALENDAR_ITEMS_RESPONSE)
            return _mock_response({})

        mock_get.side_effect = side_effect
        config = _make_config()
        client = SharePointClient(config)

        cal = SPList(id="cal-001", name="Team Calendar", item_count=1)
        events = client.fetch_calendar_events(cal)

        assert len(events) == 1
        evt = events[0]
        assert evt.title == "Team Standup"
        assert evt.location == "Room 42"
        assert evt.all_day is False
        assert evt.start == datetime(2026, 3, 15, 9, 0, tzinfo=timezone.utc)
        assert evt.end == datetime(2026, 3, 15, 9, 30, tzinfo=timezone.utc)
        assert evt.event_id.startswith("evt_sp_")

    def test_parse_sp_datetime_with_z(self):
        dt = SharePointClient._parse_sp_datetime("2026-03-15T09:00:00Z")
        assert dt == datetime(2026, 3, 15, 9, 0, tzinfo=timezone.utc)

    def test_parse_sp_datetime_empty(self):
        dt = SharePointClient._parse_sp_datetime("")
        assert dt == datetime(1970, 1, 1, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Tests: Page type filtering
# ---------------------------------------------------------------------------


class TestPageTypeFiltering:
    @patch("constat.discovery.doc_tools._sharepoint.httpx.get")
    @patch.object(SharePointClient, "_get_access_token", return_value="test-token")
    def test_filter_modern_pages_only(self, _mock_token, mock_get):
        def side_effect(url, **kwargs):
            if url.endswith("/analytics"):
                return _mock_response(SITE_RESPONSE)
            if "/pages" in url and "sitePage" not in url:
                return _mock_response(PAGES_RESPONSE)
            if "sitePage" in url:
                return _mock_response({"id": "page-001", "title": "Welcome"})
            return _mock_response({})

        mock_get.side_effect = side_effect
        config = _make_config(page_types=["modern"])
        client = SharePointClient(config)
        pages = client._list_pages("test-token", "site-id")

        # Only "article" (modern) pages should be included
        assert len(pages) == 1
        assert pages[0].name == "Welcome.aspx"
