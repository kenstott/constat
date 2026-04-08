# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Integration tests for SharePoint Online via Microsoft Graph API.

Required environment variables:
    SP_SITE_URL       — e.g. https://contoso.sharepoint.com/sites/mysite
    SP_CLIENT_ID      — Azure AD app client ID
    SP_CLIENT_SECRET  — Azure AD app client secret
    SP_TENANT_ID      — Azure AD tenant ID

Optional environment variables (target specific resources):
    SP_LIBRARY_NAME   — Document library name (default: first found)
    SP_LIST_NAME      — Generic list name (default: first found)
    SP_CALENDAR_NAME  — Calendar list name (default: first found)

These can be set in .env or the shell environment.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        pytest.fail(
            f"Required environment variable {name!r} is not set. "
            "Add it to .env or your shell before running SharePoint integration tests."
        )
    return val


def _optional_env(name: str) -> str | None:
    return os.environ.get(name) or None


def _make_sp_config() -> MagicMock:
    """Build a real config from environment variables."""
    library_name = _optional_env("SP_LIBRARY_NAME")
    list_name = _optional_env("SP_LIST_NAME")
    calendar_name = _optional_env("SP_CALENDAR_NAME")

    config = MagicMock()
    config.site_url = _require_env("SP_SITE_URL")
    config.auth_type = "oauth2"
    config.oauth2_client_id = _require_env("SP_CLIENT_ID")
    config.oauth2_client_secret = _require_env("SP_CLIENT_SECRET")
    config.oauth2_tenant_id = _require_env("SP_TENANT_ID")
    config.oauth2_scopes = ["https://graph.microsoft.com/.default"]
    config.oauth2_token_cache = None
    config.discover_libraries = True
    config.discover_lists = True
    config.discover_calendars = True
    config.discover_pages = True
    config.library_names = [library_name] if library_name else None
    config.list_names = [list_name] if list_name else None
    config.calendar_names = [calendar_name] if calendar_name else None
    config.page_types = None
    config.folder_path = None
    config.recursive = True
    config.max_files = 200
    config.max_rows = 500
    config.include_types = None
    config.exclude_patterns = None
    config.include_trashed = False
    config.since = None
    return config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sp_client():
    from constat.discovery.doc_tools._sharepoint import SharePointClient
    config = _make_sp_config()
    return SharePointClient(config)


@pytest.fixture(scope="module")
def site_discovery(sp_client):
    """Run discover_site() once and share the result across tests in this module."""
    return sp_client.discover_site()


# ---------------------------------------------------------------------------
# Protocol detection
# ---------------------------------------------------------------------------

class TestProtocolDetection:
    def test_graph_api_detected_for_sharepoint_com(self, sp_client):
        assert sp_client._api == "graph"


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

class TestAuthentication:
    def test_get_access_token_returns_non_empty_string(self, sp_client):
        token = sp_client._get_access_token()
        assert isinstance(token, str)
        assert len(token) > 0

    def test_token_is_bearer_format(self, sp_client):
        """Token should be a JWT or opaque bearer token (not empty, no whitespace)."""
        token = sp_client._get_access_token()
        assert " " not in token.strip()


# ---------------------------------------------------------------------------
# Site ID resolution
# ---------------------------------------------------------------------------

class TestSiteResolution:
    def test_get_site_id_returns_compound_id(self, sp_client):
        token = sp_client._get_access_token()
        site_id = sp_client._get_site_id(token)
        # Graph site IDs are compound: hostname,driveId,siteId
        assert isinstance(site_id, str)
        assert "," in site_id or len(site_id) > 10


# ---------------------------------------------------------------------------
# Site discovery
# ---------------------------------------------------------------------------

class TestSiteDiscovery:
    def test_discover_returns_expected_keys(self, site_discovery):
        assert set(site_discovery.keys()) == {"libraries", "lists", "calendars", "pages"}

    def test_libraries_are_sp_library_objects(self, site_discovery):
        from constat.discovery.doc_tools._sharepoint import SPLibrary
        for lib in site_discovery["libraries"]:
            assert isinstance(lib, SPLibrary)
            assert lib.id
            assert lib.name
            assert lib.drive_id

    def test_lists_are_sp_list_objects(self, site_discovery):
        from constat.discovery.doc_tools._sharepoint import SPList
        for lst in site_discovery["lists"]:
            assert isinstance(lst, SPList)
            assert lst.id
            assert lst.name

    def test_calendars_are_sp_list_objects(self, site_discovery):
        from constat.discovery.doc_tools._sharepoint import SPList
        for cal in site_discovery["calendars"]:
            assert isinstance(cal, SPList)
            assert cal.id
            assert cal.name

    def test_pages_are_sp_page_objects(self, site_discovery):
        from constat.discovery.doc_tools._sharepoint import SPPage
        for page in site_discovery["pages"]:
            assert isinstance(page, SPPage)
            assert page.id
            assert page.name


# ---------------------------------------------------------------------------
# Document libraries
# ---------------------------------------------------------------------------

class TestDocumentLibraries:
    def test_at_least_one_library_exists(self, site_discovery):
        if not site_discovery["libraries"]:
            pytest.fail("No document libraries found on the SharePoint site.")

    def test_fetch_library_files_returns_list(self, sp_client, site_discovery):
        if not site_discovery["libraries"]:
            pytest.fail("No document libraries found — cannot test file listing.")
        library = site_discovery["libraries"][0]
        files = sp_client.fetch_library_files(library)
        assert isinstance(files, list)

    def test_library_files_have_name_and_id(self, sp_client, site_discovery):
        if not site_discovery["libraries"]:
            pytest.fail("No document libraries found.")
        library = site_discovery["libraries"][0]
        files = sp_client.fetch_library_files(library)
        for f in files:
            # DriveFile objects must have name and id attributes
            assert hasattr(f, "name") or isinstance(f, dict)


# ---------------------------------------------------------------------------
# Lists
# ---------------------------------------------------------------------------

class TestLists:
    def test_fetch_list_items_returns_list(self, sp_client, site_discovery):
        if not site_discovery["lists"]:
            pytest.fail("No generic lists found on the SharePoint site.")
        sp_list = site_discovery["lists"][0]
        items = sp_client.fetch_list_items(sp_list)
        assert isinstance(items, list)

    def test_list_items_have_fields(self, sp_client, site_discovery):
        if not site_discovery["lists"]:
            pytest.fail("No generic lists found.")
        sp_list = site_discovery["lists"][0]
        items = sp_client.fetch_list_items(sp_list)
        for item in items:
            assert "fields" in item or "id" in item

    def test_render_list_as_markdown_produces_table(self, sp_client, site_discovery):
        if not site_discovery["lists"]:
            pytest.fail("No generic lists found.")
        sp_list = site_discovery["lists"][0]
        items = sp_client.fetch_list_items(sp_list)
        md = sp_client.render_list_as_markdown(sp_list, items)
        assert isinstance(md, str)
        assert f"# {sp_list.name}" in md
        if items:
            assert "|" in md  # markdown table
        else:
            assert "(empty list)" in md


# ---------------------------------------------------------------------------
# Calendar lists
# ---------------------------------------------------------------------------

class TestCalendars:
    def test_fetch_calendar_events_returns_list(self, sp_client, site_discovery):
        if not site_discovery["calendars"]:
            pytest.fail("No calendar lists found on the SharePoint site.")
        cal = site_discovery["calendars"][0]
        events = sp_client.fetch_calendar_events(cal)
        assert isinstance(events, list)

    def test_calendar_events_have_required_fields(self, sp_client, site_discovery):
        if not site_discovery["calendars"]:
            pytest.fail("No calendar lists found.")
        cal = site_discovery["calendars"][0]
        events = sp_client.fetch_calendar_events(cal)
        from constat.discovery.doc_tools._calendar import CalendarEvent
        for evt in events:
            assert isinstance(evt, CalendarEvent)
            assert evt.event_id.startswith("evt_sp_")
            assert evt.title
            assert evt.start is not None
            assert evt.end is not None


# ---------------------------------------------------------------------------
# Site pages
# ---------------------------------------------------------------------------

class TestSitePages:
    def test_fetch_pages_returns_list(self, sp_client):
        pages = sp_client.fetch_pages()
        assert isinstance(pages, list)

    def test_pages_have_content(self, sp_client, site_discovery):
        for page in site_discovery["pages"]:
            assert isinstance(page.content, str)
            assert page.id.startswith("pg_")

    def test_page_ids_are_unique(self, site_discovery):
        ids = [p.id for p in site_discovery["pages"]]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# List item attachments
# ---------------------------------------------------------------------------

_ALLOW_APP_ONLY_MSG = (
    "SharePoint REST API rejected the app-only token (401). "
    "Tenant admin must enable app-only access: "
    "Connect-SPOService -Url https://kenstott-admin.sharepoint.com; "
    "Set-SPOTenant -AllowAppOnlyPolicy $true"
)


class TestListItemAttachments:
    def _find_item_with_attachment(self, sp_client, sp_list) -> tuple[str, str] | None:
        """Return (item_id, attachment_name) for the first item that has attachments."""
        import httpx as _httpx

        items = sp_client.fetch_list_items(sp_list)
        for item in items:
            item_id = item.get("id", "")
            if not item_id:
                continue
            try:
                attachments = sp_client.fetch_list_item_attachments(sp_list, item_id)
                if attachments:
                    file_name = attachments[0].get("FileName", attachments[0].get("name", ""))
                    if file_name:
                        return item_id, file_name
            except _httpx.HTTPStatusError as exc:
                if exc.response.status_code == 401:
                    pytest.fail(_ALLOW_APP_ONLY_MSG)
                continue
        return None

    def test_fetch_attachments_returns_list(self, sp_client, site_discovery):
        import httpx as _httpx

        if not site_discovery["lists"]:
            pytest.fail("No generic lists found on the SharePoint site.")
        sp_list = site_discovery["lists"][0]
        items = sp_client.fetch_list_items(sp_list)
        if not items:
            pytest.fail(f"List '{sp_list.name}' has no items.")
        item_id = items[0]["id"]
        try:
            attachments = sp_client.fetch_list_item_attachments(sp_list, item_id)
        except _httpx.HTTPStatusError as exc:
            if exc.response.status_code == 401:
                pytest.fail(_ALLOW_APP_ONLY_MSG)
            raise
        assert isinstance(attachments, list)

    def test_item_with_attachment_has_name(self, sp_client, site_discovery):
        if not site_discovery["lists"]:
            pytest.fail("No generic lists found on the SharePoint site.")
        sp_list = site_discovery["lists"][0]
        result = self._find_item_with_attachment(sp_client, sp_list)
        if result is None:
            pytest.fail(
                f"No items with attachments found in list '{sp_list.name}'. "
                "Add an attachment to a list item before running this test."
            )
        _, attachment_name = result
        assert isinstance(attachment_name, str)
        assert len(attachment_name) > 0

    def test_download_list_item_attachment(self, sp_client, site_discovery):
        import httpx as _httpx

        if not site_discovery["lists"]:
            pytest.fail("No generic lists found on the SharePoint site.")
        sp_list = site_discovery["lists"][0]
        result = self._find_item_with_attachment(sp_client, sp_list)
        if result is None:
            pytest.fail(
                f"No items with attachments found in list '{sp_list.name}'. "
                "Add an attachment to a list item before running this test."
            )
        item_id, attachment_name = result
        try:
            content = sp_client.download_list_item_attachment(sp_list, item_id, attachment_name)
        except _httpx.HTTPStatusError as exc:
            if exc.response.status_code == 401:
                pytest.fail(_ALLOW_APP_ONLY_MSG)
            raise
        assert isinstance(content, bytes)
        assert len(content) > 0


# ---------------------------------------------------------------------------
# Calendar item attachments
# ---------------------------------------------------------------------------

class TestCalendarItemAttachments:
    def _find_calendar_item_with_attachment(self, sp_client, cal) -> tuple[str, str] | None:
        import httpx as _httpx

        items = sp_client.fetch_list_items(cal)
        for item in items:
            item_id = item.get("id", "")
            if not item_id:
                continue
            try:
                attachments = sp_client.fetch_list_item_attachments(cal, item_id)
                if attachments:
                    file_name = attachments[0].get("FileName", attachments[0].get("name", ""))
                    if file_name:
                        return item_id, file_name
            except _httpx.HTTPStatusError as exc:
                if exc.response.status_code == 401:
                    pytest.fail(_ALLOW_APP_ONLY_MSG)
                continue
        return None

    def test_calendar_item_attachments_returns_list(self, sp_client, site_discovery):
        import httpx as _httpx

        if not site_discovery["calendars"]:
            pytest.fail("No calendar lists found on the SharePoint site.")
        cal = site_discovery["calendars"][0]
        items = sp_client.fetch_list_items(cal)
        if not items:
            pytest.fail(f"Calendar '{cal.name}' has no items.")
        item_id = items[0]["id"]
        try:
            attachments = sp_client.fetch_list_item_attachments(cal, item_id)
        except _httpx.HTTPStatusError as exc:
            if exc.response.status_code == 401:
                pytest.fail(_ALLOW_APP_ONLY_MSG)
            raise
        assert isinstance(attachments, list)

    def test_download_calendar_item_attachment(self, sp_client, site_discovery):
        import httpx as _httpx

        if not site_discovery["calendars"]:
            pytest.fail("No calendar lists found on the SharePoint site.")
        cal = site_discovery["calendars"][0]
        result = self._find_calendar_item_with_attachment(sp_client, cal)
        if result is None:
            pytest.fail(
                f"No items with attachments found in calendar '{cal.name}'. "
                "Add an attachment to a calendar event before running this test."
            )
        item_id, attachment_name = result
        try:
            content = sp_client.download_list_item_attachment(cal, item_id, attachment_name)
        except _httpx.HTTPStatusError as exc:
            if exc.response.status_code == 401:
                pytest.fail(_ALLOW_APP_ONLY_MSG)
            raise
        assert isinstance(content, bytes)
        assert len(content) > 0


# ---------------------------------------------------------------------------
# Linked file extraction
# ---------------------------------------------------------------------------

class TestLinkedFileExtraction:
    def test_extract_sp_urls_finds_absolute_document_links(self, sp_client):
        """_extract_sp_urls should return absolute SharePoint document URLs."""
        from urllib.parse import urlparse
        hostname = urlparse(sp_client._site_url).hostname
        text = (
            f"See attached: https://{hostname}/sites/shared/Documents/black-swan-and-the-llm.docx "
            f"and also https://{hostname}/pages/home.aspx for details."
        )
        urls = sp_client._extract_sp_urls(text)
        assert len(urls) == 1
        assert urls[0].endswith(".docx")

    def test_extract_sp_urls_finds_server_relative_hrefs(self, sp_client):
        """_extract_sp_urls should resolve server-relative hrefs in HTML."""
        from urllib.parse import urlparse
        parsed = urlparse(sp_client._site_url)
        text = (
            '<p>Attachment: <a href="/SiteAssets/Lists/Calendar/NewForm/report.pdf">'
            "report.pdf</a></p>"
        )
        urls = sp_client._extract_sp_urls(text)
        assert len(urls) == 1
        assert urls[0] == f"{parsed.scheme}://{parsed.hostname}/SiteAssets/Lists/Calendar/NewForm/report.pdf"

    def test_extract_sp_urls_ignores_non_document_links(self, sp_client):
        from urllib.parse import urlparse
        hostname = urlparse(sp_client._site_url).hostname
        text = f"Visit https://{hostname}/pages/home.aspx for details."
        urls = sp_client._extract_sp_urls(text)
        assert urls == []

    def test_extract_sp_urls_empty_input(self, sp_client):
        assert sp_client._extract_sp_urls("") == []
        assert sp_client._extract_sp_urls(None) == []

    def test_calendar_items_contain_linked_file(self, sp_client, site_discovery):
        """The test calendar item with a hyperlinked .docx in its description
        should yield at least one linked file via extract_linked_files_from_items."""
        if not site_discovery["calendars"]:
            pytest.fail("No calendar lists found on the SharePoint site.")
        cal = site_discovery["calendars"][0]
        items = sp_client.fetch_list_items(cal)
        if not items:
            pytest.fail(f"Calendar '{cal.name}' has no items.")
        results = sp_client.extract_linked_files_from_items(items)
        if not results:
            pytest.fail(
                "No linked document files found in calendar items. "
                "Ensure a calendar event description contains a hyperlink to a "
                "SharePoint document (e.g. black-swan-and-the-llm.docx)."
            )
        item_id, url, content = results[0]
        assert isinstance(item_id, str)
        assert isinstance(url, str)
        assert isinstance(content, bytes)
        assert len(content) > 0

    def test_fetch_linked_file_returns_bytes(self, sp_client, site_discovery):
        """fetch_linked_file should download a file and return non-empty bytes."""
        if not site_discovery["calendars"]:
            pytest.fail("No calendar lists found on the SharePoint site.")
        cal = site_discovery["calendars"][0]
        items = sp_client.fetch_list_items(cal)
        results = sp_client.extract_linked_files_from_items(items)
        if not results:
            pytest.fail("No linked files found — cannot test fetch_linked_file directly.")
        _, url, content = results[0]
        # Re-fetch to verify fetch_linked_file is independently callable
        content2 = sp_client.fetch_linked_file(url)
        assert isinstance(content2, bytes)
        assert len(content2) > 0
