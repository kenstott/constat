# Copyright (c) 2025 Kenneth Stott
# Canary: 921320cd-3676-4db5-8bea-ce09fc5ec838
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tests for constat.server.source_refresher."""

import os
import tempfile
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from constat.server.source_refresher import (
    _classify_source,
    _needs_refresh,
    _refresh_calendar_source,
    _refresh_file_source,
    _refresh_http_source,
    _refresh_imap_source,
    refresh_session_sources,
)


# ---------------------------------------------------------------------------
# _needs_refresh
# ---------------------------------------------------------------------------


class TestNeedsRefresh:
    def test_no_document_config(self):
        ref = {"name": "test"}
        assert _needs_refresh(ref, 900) is False

    def test_auto_refresh_disabled(self):
        ref = {"document_config": {"auto_refresh": False}}
        assert _needs_refresh(ref, 900) is False

    def test_never_refreshed(self):
        ref = {"document_config": {"url": "http://example.com"}}
        assert _needs_refresh(ref, 900) is True

    def test_recently_refreshed(self):
        now = datetime.now(timezone.utc)
        ref = {
            "document_config": {"url": "http://example.com"},
            "last_refreshed": now.isoformat(),
        }
        assert _needs_refresh(ref, 900) is False

    def test_stale_refresh(self):
        old = datetime.now(timezone.utc) - timedelta(seconds=1000)
        ref = {
            "document_config": {"url": "http://example.com"},
            "last_refreshed": old.isoformat(),
        }
        assert _needs_refresh(ref, 900) is True

    def test_custom_interval_override(self):
        old = datetime.now(timezone.utc) - timedelta(seconds=500)
        ref = {
            "document_config": {"url": "http://example.com", "refresh_interval": 300},
            "last_refreshed": old.isoformat(),
        }
        assert _needs_refresh(ref, 900) is True

    def test_custom_interval_not_yet(self):
        recent = datetime.now(timezone.utc) - timedelta(seconds=100)
        ref = {
            "document_config": {"url": "http://example.com", "refresh_interval": 300},
            "last_refreshed": recent.isoformat(),
        }
        assert _needs_refresh(ref, 900) is False


# ---------------------------------------------------------------------------
# _classify_source
# ---------------------------------------------------------------------------


class TestClassifySource:
    def test_imap(self):
        ref = {"document_config": {"url": "imaps://imap.gmail.com"}}
        assert _classify_source(ref) == "imap"

    def test_http(self):
        ref = {"document_config": {"url": "https://example.com/doc.pdf"}}
        assert _classify_source(ref) == "http"

    def test_file(self):
        ref = {"document_config": {"path": "/tmp/doc.md"}}
        assert _classify_source(ref) == "file"

    def test_no_config(self):
        ref = {"name": "test"}
        assert _classify_source(ref) is None

    def test_no_url_or_path(self):
        ref = {"document_config": {"content": "inline text"}}
        assert _classify_source(ref) is None

    def test_imap_plain(self):
        ref = {"document_config": {"url": "imap://mail.example.com"}}
        assert _classify_source(ref) == "imap"

    def test_calendar(self):
        ref = {"document_config": {"type": "calendar", "auth_type": "oauth2_refresh"}}
        assert _classify_source(ref) == "calendar"

    def test_calendar_takes_precedence_over_url(self):
        # type=calendar should be classified as calendar even if a URL is present
        ref = {"document_config": {"type": "calendar", "url": "https://www.googleapis.com/calendar/v3"}}
        assert _classify_source(ref) == "calendar"


# ---------------------------------------------------------------------------
# _refresh_file_source
# ---------------------------------------------------------------------------


class TestRefreshFileSource:
    def test_file_unchanged(self):
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            f.write(b"hello")
            path = f.name

        try:
            stat = os.stat(path)
            ref = {
                "document_config": {"path": path},
                "file_stat_key": f"{stat.st_mtime}:{stat.st_size}",
            }
            managed = MagicMock()
            success, msg, count = _refresh_file_source(managed, ref, "test")
            assert success is True
            assert count == 0
            assert "unchanged" in msg.lower()
        finally:
            os.unlink(path)

    def test_file_not_found(self):
        ref = {"document_config": {"path": "/nonexistent/file.md"}}
        managed = MagicMock()
        success, msg, count = _refresh_file_source(managed, ref, "test")
        assert success is False
        assert count == 0

    def test_file_changed(self):
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            f.write(b"hello")
            path = f.name

        try:
            ref = {
                "document_config": {"path": path},
                "file_stat_key": "0:0",  # old stat won't match
            }
            managed = MagicMock()
            mock_vs = MagicMock()
            managed.session.doc_tools._vector_store = mock_vs
            managed.session.doc_tools.add_document_from_config.return_value = (True, "5 chunks indexed")
            managed.session_id = "test-session"

            success, msg, count = _refresh_file_source(managed, ref, "test")
            assert success is True
            assert count == 5
            assert "file_stat_key" in ref
            mock_vs.delete_resource_chunks.assert_called_once()
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# _refresh_calendar_source
# ---------------------------------------------------------------------------


class TestRefreshCalendarSource:
    def test_calendar_refresh_success(self):
        ref = {
            "document_config": {
                "type": "calendar",
                "auth_type": "oauth2_refresh",
                "oauth2_client_id": "client-id",
                "oauth2_client_secret": "secret",
            },
        }
        managed = MagicMock()
        mock_vs = MagicMock()
        managed.session.doc_tools._vector_store = mock_vs
        managed.session.doc_tools.add_document_from_config.return_value = (True, "7 chunks indexed")
        managed.session_id = "test-session"

        success, msg, count = _refresh_calendar_source(managed, ref, "my-calendar")

        assert success is True
        assert count == 7
        mock_vs.delete_resource_chunks.assert_called_once_with(
            "test-session", "document", "my-calendar"
        )
        managed.session.doc_tools.add_document_from_config.assert_called_once()

    def test_calendar_refresh_failure(self):
        ref = {
            "document_config": {
                "type": "calendar",
                "auth_type": "oauth2_refresh",
            },
        }
        managed = MagicMock()
        mock_vs = MagicMock()
        managed.session.doc_tools._vector_store = mock_vs
        managed.session.doc_tools.add_document_from_config.return_value = (False, "auth error")
        managed.session_id = "test-session"

        success, msg, count = _refresh_calendar_source(managed, ref, "my-calendar")

        assert success is False
        assert count == 0
        assert "auth error" in msg

    def test_calendar_refresh_no_chunks_in_msg(self):
        ref = {
            "document_config": {
                "type": "calendar",
                "auth_type": "oauth2_refresh",
            },
        }
        managed = MagicMock()
        managed.session.doc_tools._vector_store = MagicMock()
        managed.session.doc_tools.add_document_from_config.return_value = (True, "ok")
        managed.session_id = "test-session"

        success, msg, count = _refresh_calendar_source(managed, ref, "my-calendar")

        assert success is True
        assert count == 0


# ---------------------------------------------------------------------------
# _refresh_http_source
# ---------------------------------------------------------------------------


class TestRefreshHttpSource:
    @patch("constat.server.source_refresher.urllib.request.urlopen")
    def test_http_unchanged(self, mock_urlopen):
        import hashlib
        content = b"hello world"
        content_hash = hashlib.sha256(content).hexdigest()

        mock_resp = MagicMock()
        mock_resp.read.return_value = content
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        ref = {
            "document_config": {"url": "https://example.com/doc.md"},
            "content_hash": content_hash,
        }
        managed = MagicMock()

        success, msg, count = _refresh_http_source(managed, ref, "test")
        assert success is True
        assert count == 0
        assert "unchanged" in msg.lower()

    @patch("constat.server.source_refresher.urllib.request.urlopen")
    def test_http_changed(self, mock_urlopen):
        content = b"new content"

        mock_resp = MagicMock()
        mock_resp.read.return_value = content
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        ref = {
            "document_config": {"url": "https://example.com/doc.md"},
            "content_hash": "old-hash",
        }
        managed = MagicMock()
        mock_vs = MagicMock()
        managed.session.doc_tools._vector_store = mock_vs
        managed.session.doc_tools.add_document_from_config.return_value = (True, "3 chunks indexed")
        managed.session_id = "test-session"

        success, msg, count = _refresh_http_source(managed, ref, "test")
        assert success is True
        assert count == 3
        assert "content_hash" in ref
        mock_vs.delete_resource_chunks.assert_called_once()


# ---------------------------------------------------------------------------
# refresh_session_sources
# ---------------------------------------------------------------------------


class TestRefreshSessionSources:
    def test_no_doc_tools(self):
        managed = MagicMock()
        managed.session = None
        sm = MagicMock()
        assert refresh_session_sources(managed, sm, 900) == 0

    def test_skips_non_eligible(self):
        managed = MagicMock()
        managed._file_refs = [
            {"name": "test", "document_config": {"auto_refresh": False}},
        ]
        managed.session.doc_tools = MagicMock()
        sm = MagicMock()
        assert refresh_session_sources(managed, sm, 900) == 0

    def test_refreshes_eligible_file(self):
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            f.write(b"content")
            path = f.name

        try:
            managed = MagicMock()
            managed.session_id = "test-session"
            managed._file_refs = [
                {
                    "name": "doc",
                    "document_config": {"path": path},
                    # no last_refreshed -> needs refresh
                },
            ]
            mock_vs = MagicMock()
            managed.session.doc_tools._vector_store = mock_vs
            managed.session.doc_tools.add_document_from_config.return_value = (True, "2 chunks")

            sm = MagicMock()
            count = refresh_session_sources(managed, sm, 900)
            assert count == 1
            managed.save_resources.assert_called_once()
        finally:
            os.unlink(path)

    def test_publishes_ingest_start_event(self):
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            f.write(b"content")
            path = f.name

        try:
            managed = MagicMock()
            managed.session_id = "test-session"
            managed._file_refs = [
                {
                    "name": "doc",
                    "document_config": {"path": path},
                },
            ]
            mock_vs = MagicMock()
            managed.session.doc_tools._vector_store = mock_vs
            managed.session.doc_tools.add_document_from_config.return_value = (True, "2 chunks")

            sm = MagicMock()
            refresh_session_sources(managed, sm, 900)

            from constat.server.models import EventType
            push_calls = sm._push_event.call_args_list
            event_types = [c.args[1] for c in push_calls]
            assert EventType.SOURCE_INGEST_START in event_types
        finally:
            os.unlink(path)

    def test_refreshes_calendar_source(self):
        managed = MagicMock()
        managed.session_id = "test-session"
        managed._file_refs = [
            {
                "name": "my-cal",
                "document_config": {
                    "type": "calendar",
                    "auth_type": "oauth2_refresh",
                },
            },
        ]
        mock_vs = MagicMock()
        managed.session.doc_tools._vector_store = mock_vs
        managed.session.doc_tools.add_document_from_config.return_value = (True, "4 chunks indexed")

        sm = MagicMock()
        count = refresh_session_sources(managed, sm, 900)

        assert count == 1
        mock_vs.delete_resource_chunks.assert_called_once()
        managed.save_resources.assert_called_once()

        from constat.server.models import EventType
        push_calls = sm._push_event.call_args_list
        event_types = [c.args[1] for c in push_calls]
        assert EventType.SOURCE_INGEST_START in event_types
        assert EventType.SOURCE_REFRESH_COMPLETE in event_types
