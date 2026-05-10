# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Tests for Cloud Drive document source (Google Drive / OneDrive)."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from constat.core.source_config import DocumentConfig
from constat.discovery.doc_tools._drive import (
    DriveFetcher,
    DriveFile,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_config(**kwargs) -> DocumentConfig:
    """Build a DocumentConfig with drive defaults."""
    defaults = {
        "type": "drive",
        "provider": "google",
        "folder_id": "root_folder_123",
        "oauth2_client_id": "client-id",
        "oauth2_client_secret": "client-secret",
    }
    defaults.update(kwargs)
    return DocumentConfig(**defaults)


def _google_file_item(
    file_id: str,
    name: str,
    mime: str = "application/pdf",
    size: int = 1024,
    modified: str = "2026-03-15T10:00:00Z",
    web_link: str | None = None,
) -> dict:
    """Build a Google Drive API file item dict."""
    item: dict = {
        "id": file_id,
        "name": name,
        "mimeType": mime,
        "modifiedTime": modified,
        "parents": ["parent_1"],
    }
    if size is not None:
        item["size"] = str(size)
    if web_link:
        item["webViewLink"] = web_link
    return item


def _ms_drive_item(
    item_id: str,
    name: str,
    mime: str = "application/pdf",
    size: int = 1024,
    modified: str = "2026-03-15T10:00:00Z",
    is_folder: bool = False,
) -> dict:
    """Build a Microsoft Graph drive item dict."""
    item: dict = {
        "id": item_id,
        "name": name,
        "lastModifiedDateTime": modified,
        "size": size,
        "parentReference": {"id": "parent_1", "path": "/drive/root:/Docs"},
        "webUrl": f"https://onedrive.live.com/{item_id}",
    }
    if is_folder:
        item["folder"] = {"childCount": 2}
    else:
        item["file"] = {"mimeType": mime}
    return item


# ---------------------------------------------------------------------------
# _matches_filters
# ---------------------------------------------------------------------------

class TestMatchesFilters:
    def test_no_filters_passes_all(self):
        config = _make_config()
        fetcher = DriveFetcher(config)
        assert fetcher._matches_filters("report.pdf", "application/pdf") is True

    def test_include_types_match(self):
        config = _make_config(include_types=[".pdf", ".docx"])
        fetcher = DriveFetcher(config)
        assert fetcher._matches_filters("report.pdf", "application/pdf") is True
        assert fetcher._matches_filters("data.xlsx", "application/vnd.ms-excel") is False

    def test_include_types_google_native(self):
        config = _make_config(include_types=[".docx"])
        fetcher = DriveFetcher(config)
        assert fetcher._matches_filters(
            "My Doc", "application/vnd.google-apps.document"
        ) is True
        assert fetcher._matches_filters(
            "My Sheet", "application/vnd.google-apps.spreadsheet"
        ) is False

    def test_exclude_patterns(self):
        config = _make_config(exclude_patterns=[r"^~\$", r"^\."])
        fetcher = DriveFetcher(config)
        assert fetcher._matches_filters("~$temp.docx", "application/docx") is False
        assert fetcher._matches_filters(".hidden.pdf", "application/pdf") is False
        assert fetcher._matches_filters("report.pdf", "application/pdf") is True


# ---------------------------------------------------------------------------
# make_file_id
# ---------------------------------------------------------------------------

class TestMakeFileId:
    def test_deterministic(self):
        f = DriveFile(
            file_id="abc123",
            name="Quarterly Report.pdf",
            mime_type="application/pdf",
            size=1024,
            modified_time=datetime(2026, 3, 15, tzinfo=timezone.utc),
            path="Quarterly Report.pdf",
            parent_id=None,
            web_url=None,
        )
        id1 = DriveFetcher.make_file_id(f)
        id2 = DriveFetcher.make_file_id(f)
        assert id1 == id2
        assert id1.startswith("f_")
        assert "quarterly_report" in id1

    def test_different_files_different_ids(self):
        f1 = DriveFile(
            file_id="abc123", name="A.pdf", mime_type="application/pdf",
            size=100, modified_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
            path="A.pdf", parent_id=None, web_url=None,
        )
        f2 = DriveFile(
            file_id="def456", name="B.pdf", mime_type="application/pdf",
            size=100, modified_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
            path="B.pdf", parent_id=None, web_url=None,
        )
        assert DriveFetcher.make_file_id(f1) != DriveFetcher.make_file_id(f2)

    def test_slug_truncation(self):
        f = DriveFile(
            file_id="x",
            name="A" * 100 + ".pdf",
            mime_type="application/pdf",
            size=100,
            modified_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
            path="long.pdf",
            parent_id=None,
            web_url=None,
        )
        fid = DriveFetcher.make_file_id(f)
        # slug portion should be at most 30 chars
        slug = fid.split("_", 2)[2]
        assert len(slug) <= 30


# ---------------------------------------------------------------------------
# Google Drive listing
# ---------------------------------------------------------------------------

class TestGoogleDriveListing:
    @patch("constat.discovery.doc_tools._drive.DriveFetcher._get_access_token")
    def test_list_flat_folder(self, mock_token):
        mock_token.return_value = "test-token"
        config = _make_config(folder_id="folder_1", max_files=10)
        fetcher = DriveFetcher(config)

        response_data = {
            "files": [
                _google_file_item("f1", "report.pdf"),
                _google_file_item("f2", "data.xlsx", "application/vnd.ms-excel"),
            ],
        }

        mock_resp = MagicMock()
        mock_resp.json.return_value = response_data
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_resp) as mock_get:
            files = fetcher._list_google()

        assert len(files) == 2
        assert files[0].name == "report.pdf"
        assert files[1].name == "data.xlsx"
        assert files[0].is_google_native is False

    @patch("constat.discovery.doc_tools._drive.DriveFetcher._get_access_token")
    def test_list_with_subfolder_traversal(self, mock_token):
        mock_token.return_value = "test-token"
        config = _make_config(folder_id="root", recursive=True, max_files=10)
        fetcher = DriveFetcher(config)

        # First call: root folder with a subfolder and a file
        root_response = {
            "files": [
                _google_file_item("sub1", "Subfolder", "application/vnd.google-apps.folder"),
                _google_file_item("f1", "root_file.pdf"),
            ],
        }
        # Second call: subfolder contents
        sub_response = {
            "files": [
                _google_file_item("f2", "sub_file.docx", "application/msword"),
            ],
        }

        mock_resp_root = MagicMock()
        mock_resp_root.json.return_value = root_response
        mock_resp_root.raise_for_status = MagicMock()

        mock_resp_sub = MagicMock()
        mock_resp_sub.json.return_value = sub_response
        mock_resp_sub.raise_for_status = MagicMock()

        with patch("httpx.get", side_effect=[mock_resp_root, mock_resp_sub]):
            files = fetcher._list_google()

        assert len(files) == 2
        names = {f.name for f in files}
        assert "root_file.pdf" in names
        assert "sub_file.docx" in names

    @patch("constat.discovery.doc_tools._drive.DriveFetcher._get_access_token")
    def test_list_google_native_detection(self, mock_token):
        mock_token.return_value = "test-token"
        config = _make_config(folder_id="folder_1", max_files=10)
        fetcher = DriveFetcher(config)

        response_data = {
            "files": [
                _google_file_item(
                    "gdoc1", "My Document",
                    "application/vnd.google-apps.document",
                    size=None,
                ),
            ],
        }

        mock_resp = MagicMock()
        mock_resp.json.return_value = response_data
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_resp):
            files = fetcher._list_google()

        assert len(files) == 1
        assert files[0].is_google_native is True
        assert files[0].mime_type == "application/vnd.google-apps.document"

    @patch("constat.discovery.doc_tools._drive.DriveFetcher._get_access_token")
    def test_pagination(self, mock_token):
        mock_token.return_value = "test-token"
        config = _make_config(folder_id="folder_1", max_files=10)
        fetcher = DriveFetcher(config)

        page1 = {
            "files": [_google_file_item("f1", "page1.pdf")],
            "nextPageToken": "token_page2",
        }
        page2 = {
            "files": [_google_file_item("f2", "page2.pdf")],
        }

        mock_resp1 = MagicMock()
        mock_resp1.json.return_value = page1
        mock_resp1.raise_for_status = MagicMock()

        mock_resp2 = MagicMock()
        mock_resp2.json.return_value = page2
        mock_resp2.raise_for_status = MagicMock()

        with patch("httpx.get", side_effect=[mock_resp1, mock_resp2]):
            files = fetcher._list_google()

        assert len(files) == 2
        assert files[0].name == "page1.pdf"
        assert files[1].name == "page2.pdf"


# ---------------------------------------------------------------------------
# Google Docs export
# ---------------------------------------------------------------------------

class TestGoogleDocsExport:
    @patch("constat.discovery.doc_tools._drive.DriveFetcher._get_access_token")
    def test_download_native_exports_docx(self, mock_token):
        mock_token.return_value = "test-token"
        config = _make_config()
        fetcher = DriveFetcher(config)

        file = DriveFile(
            file_id="gdoc1",
            name="My Doc",
            mime_type="application/vnd.google-apps.document",
            size=None,
            modified_time=datetime(2026, 3, 15, tzinfo=timezone.utc),
            path="My Doc",
            parent_id="folder_1",
            web_url=None,
            is_google_native=True,
        )

        mock_resp = MagicMock()
        mock_resp.content = b"fake-docx-bytes"
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_resp) as mock_get:
            data = fetcher._download_google(file)

        assert data == b"fake-docx-bytes"
        # Should have called export endpoint
        call_args = mock_get.call_args
        assert "/export" in call_args[0][0]
        export_mime = call_args[1]["params"]["mimeType"]
        assert "wordprocessingml" in export_mime

    @patch("constat.discovery.doc_tools._drive.DriveFetcher._get_access_token")
    def test_download_regular_file(self, mock_token):
        mock_token.return_value = "test-token"
        config = _make_config()
        fetcher = DriveFetcher(config)

        file = DriveFile(
            file_id="f1",
            name="report.pdf",
            mime_type="application/pdf",
            size=5000,
            modified_time=datetime(2026, 3, 15, tzinfo=timezone.utc),
            path="report.pdf",
            parent_id="folder_1",
            web_url=None,
            is_google_native=False,
        )

        mock_resp = MagicMock()
        mock_resp.content = b"fake-pdf-bytes"
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_resp) as mock_get:
            data = fetcher._download_google(file)

        assert data == b"fake-pdf-bytes"
        call_args = mock_get.call_args
        assert "/export" not in call_args[0][0]
        assert call_args[1]["params"]["alt"] == "media"


# ---------------------------------------------------------------------------
# Microsoft OneDrive listing
# ---------------------------------------------------------------------------

class TestMicrosoftOneDriveListing:
    @patch("constat.discovery.doc_tools._drive.DriveFetcher._get_access_token")
    def test_list_onedrive_flat(self, mock_token):
        mock_token.return_value = "test-token"
        config = _make_config(provider="microsoft", folder_id="folder_ms_1",
                              oauth2_tenant_id="tenant-1")
        fetcher = DriveFetcher(config)

        response_data = {
            "value": [
                _ms_drive_item("ms_f1", "report.pdf"),
                _ms_drive_item("ms_f2", "notes.docx", "application/msword"),
            ],
        }

        mock_resp = MagicMock()
        mock_resp.json.return_value = response_data
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_resp):
            files = fetcher._list_microsoft()

        assert len(files) == 2
        assert files[0].name == "report.pdf"
        assert files[1].name == "notes.docx"

    @patch("constat.discovery.doc_tools._drive.DriveFetcher._get_access_token")
    def test_list_with_since_filter(self, mock_token):
        mock_token.return_value = "test-token"
        config = _make_config(
            provider="microsoft",
            folder_id="folder_ms_1",
            since="2026-03-10",
            oauth2_tenant_id="tenant-1",
        )
        fetcher = DriveFetcher(config)

        response_data = {
            "value": [
                _ms_drive_item("ms_f1", "new.pdf", modified="2026-03-15T10:00:00Z"),
                _ms_drive_item("ms_f2", "old.pdf", modified="2026-01-01T10:00:00Z"),
            ],
        }

        mock_resp = MagicMock()
        mock_resp.json.return_value = response_data
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_resp):
            files = fetcher._list_microsoft()

        assert len(files) == 1
        assert files[0].name == "new.pdf"

    @patch("constat.discovery.doc_tools._drive.DriveFetcher._get_access_token")
    def test_microsoft_pagination(self, mock_token):
        mock_token.return_value = "test-token"
        config = _make_config(provider="microsoft", folder_id="folder_ms_1",
                              oauth2_tenant_id="tenant-1")
        fetcher = DriveFetcher(config)

        page1 = {
            "value": [_ms_drive_item("ms_f1", "page1.pdf")],
            "@odata.nextLink": "https://graph.microsoft.com/v1.0/next-page",
        }
        page2 = {
            "value": [_ms_drive_item("ms_f2", "page2.pdf")],
        }

        mock_resp1 = MagicMock()
        mock_resp1.json.return_value = page1
        mock_resp1.raise_for_status = MagicMock()

        mock_resp2 = MagicMock()
        mock_resp2.json.return_value = page2
        mock_resp2.raise_for_status = MagicMock()

        with patch("httpx.get", side_effect=[mock_resp1, mock_resp2]):
            files = fetcher._list_microsoft()

        assert len(files) == 2

    @patch("constat.discovery.doc_tools._drive.DriveFetcher._get_access_token")
    def test_sharepoint_base_url(self, mock_token):
        mock_token.return_value = "test-token"
        config = _make_config(
            provider="microsoft",
            site_id="contoso.sharepoint.com,guid1,guid2",
            folder_id="folder_sp_1",
            oauth2_tenant_id="tenant-1",
        )
        fetcher = DriveFetcher(config)
        base = fetcher._resolve_microsoft_base_url()
        assert "sites/contoso.sharepoint.com,guid1,guid2" in base


# ---------------------------------------------------------------------------
# Transport detection
# ---------------------------------------------------------------------------

class TestTransportDetection:
    def test_drive_transport(self):
        from constat.discovery.doc_tools._transport import infer_transport
        config = _make_config()
        assert infer_transport(config) == "drive"


# ---------------------------------------------------------------------------
# Unknown provider
# ---------------------------------------------------------------------------

class TestUnknownProvider:
    def test_list_files_unknown_provider(self):
        config = _make_config(provider="dropbox")
        fetcher = DriveFetcher(config)
        with pytest.raises(ValueError, match="Unknown drive provider"):
            fetcher.list_files()

    def test_download_file_unknown_provider(self):
        config = _make_config(provider="dropbox")
        fetcher = DriveFetcher(config)
        f = DriveFile(
            file_id="x", name="x.pdf", mime_type="application/pdf",
            size=100, modified_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
            path="x.pdf", parent_id=None, web_url=None,
        )
        with pytest.raises(ValueError, match="Unknown drive provider"):
            fetcher.download_file(f)
