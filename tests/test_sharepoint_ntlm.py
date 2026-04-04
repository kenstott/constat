"""Tests for SharePoint on-premises NTLM authentication (mock-based)."""

import base64
from unittest.mock import MagicMock, patch

import pytest

from constat.discovery.doc_tools._sharepoint import SharePointClient


def _make_config(**overrides) -> MagicMock:
    """Create a mock DocumentConfig for SharePoint."""
    config = MagicMock()
    config.site_url = overrides.get("site_url", "https://intranet.company.local/sites/analytics")
    config.auth_type = overrides.get("auth_type", "ntlm")
    config.username = overrides.get("username", r"DOMAIN\user")
    config.password = overrides.get("password", "pass123")
    config.discover_libraries = overrides.get("discover_libraries", True)
    config.discover_lists = overrides.get("discover_lists", True)
    config.discover_calendars = overrides.get("discover_calendars", False)
    config.discover_pages = overrides.get("discover_pages", False)
    return config


def test_detect_api_onprem():
    """Non-sharepoint.com URLs use REST API."""
    config = _make_config(site_url="https://intranet.company.local/sites/analytics")
    client = SharePointClient(config)
    assert client._api == "rest"


def test_detect_api_online():
    """sharepoint.com URLs use Graph API."""
    config = _make_config(site_url="https://contoso.sharepoint.com/sites/analytics")
    client = SharePointClient(config)
    assert client._api == "graph"


def test_ntlm_auth_header():
    """NTLM auth uses username/password from config to build Basic header."""
    config = _make_config(auth_type="ntlm", username=r"DOMAIN\user", password="pass123")
    client = SharePointClient(config)
    headers = client._get_auth_headers()

    expected_creds = base64.b64encode(r"DOMAIN\user:pass123".encode()).decode()
    assert headers["Authorization"] == f"Basic {expected_creds}"


def test_basic_auth_header():
    """Basic auth also builds Basic Authorization header."""
    config = _make_config(auth_type="basic", username="admin", password="secret")
    client = SharePointClient(config)
    headers = client._get_auth_headers()

    expected_creds = base64.b64encode(b"admin:secret").decode()
    assert headers["Authorization"] == f"Basic {expected_creds}"


def test_rest_api_site_discovery():
    """On-prem site discovery uses /_api/web/lists endpoint."""
    config = _make_config()
    client = SharePointClient(config)

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "d": {
            "results": [
                {"Title": "Documents", "BaseTemplate": 101, "Id": "list-1"},
                {"Title": "Tasks", "BaseTemplate": 107, "Id": "list-2"},
            ]
        }
    }

    with patch("httpx.get", return_value=mock_response) as mock_get:
        headers = client._get_auth_headers()
        result = client._discover_rest(headers)

        called_url = mock_get.call_args[0][0]
        assert "/_api/web/lists" in called_url
        assert "intranet.company.local" in called_url
        assert len(result["lists"]) == 2


def test_rest_api_list_items():
    """On-prem list items use /_api/web/lists/getbytitle() endpoint."""
    config = _make_config()
    client = SharePointClient(config)

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "d": {
            "results": [
                {"Id": 1, "Title": "Item 1"},
                {"Id": 2, "Title": "Item 2"},
            ]
        }
    }

    with patch("httpx.get", return_value=mock_response) as mock_get:
        headers = client._get_auth_headers()
        items = client._list_items_rest("Tasks", headers)

        called_url = mock_get.call_args[0][0]
        assert "getbytitle('Tasks')" in called_url
        assert len(items) == 2
        assert items[0]["Title"] == "Item 1"


def test_rest_api_library_files():
    """On-prem library files use /_api/web/GetFolderByServerRelativeUrl()."""
    config = _make_config()
    client = SharePointClient(config)

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "d": {
            "results": [
                {"Name": "report.pdf", "ServerRelativeUrl": "/sites/analytics/Shared Documents/report.pdf"},
                {"Name": "data.xlsx", "ServerRelativeUrl": "/sites/analytics/Shared Documents/data.xlsx"},
            ]
        }
    }

    with patch("httpx.get", return_value=mock_response) as mock_get:
        headers = client._get_auth_headers()
        files = client._list_files_rest("/sites/analytics/Shared Documents", headers)

        called_url = mock_get.call_args[0][0]
        assert "GetFolderByServerRelativeUrl" in called_url
        assert len(files) == 2
        assert files[0]["Name"] == "report.pdf"


def test_ntlm_fallback_to_basic():
    """If auth_type is basic, use basic auth (fallback from NTLM)."""
    config = _make_config(auth_type="basic", username="admin", password="secret")
    client = SharePointClient(config)

    # Still produces a valid auth header
    headers = client._get_auth_headers()
    assert headers["Authorization"].startswith("Basic ")

    # Verify the credentials decode correctly
    encoded = headers["Authorization"].split(" ", 1)[1]
    decoded = base64.b64decode(encoded).decode()
    assert decoded == "admin:secret"
