# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Tests for transport abstraction."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from constat.discovery.doc_tools._transport import (
    FetchResult,
    infer_transport,
    fetch_document,
)


class _FakeConfig:
    """Minimal stand-in for DocumentConfig for testing."""

    def __init__(self, **kwargs):
        self.content = kwargs.get("content")
        self.path = kwargs.get("path")
        self.url = kwargs.get("url")
        self.headers = kwargs.get("headers", {})
        self.username = kwargs.get("username")
        self.password = kwargs.get("password")
        self.port = kwargs.get("port")
        self.key_path = kwargs.get("key_path")
        self.aws_profile = kwargs.get("aws_profile")
        self.aws_region = kwargs.get("aws_region")


class TestInferTransport:
    def test_inline(self):
        assert infer_transport(_FakeConfig(content="hello")) == "inline"

    def test_file(self):
        assert infer_transport(_FakeConfig(path="/tmp/x.md")) == "file"

    def test_http(self):
        assert infer_transport(_FakeConfig(url="https://example.com/doc.pdf")) == "http"

    def test_s3(self):
        assert infer_transport(_FakeConfig(url="s3://bucket/key.pdf")) == "s3"

    def test_ftp(self):
        assert infer_transport(_FakeConfig(url="ftp://host/file.txt")) == "ftp"

    def test_sftp(self):
        assert infer_transport(_FakeConfig(url="sftp://host/file.txt")) == "sftp"

    def test_content_takes_priority(self):
        """content set overrides path and url."""
        assert infer_transport(_FakeConfig(content="x", path="/y", url="https://z")) == "inline"

    def test_no_source_raises(self):
        with pytest.raises(ValueError, match="no content, path, or url"):
            infer_transport(_FakeConfig())


class TestFetchInline:
    def test_basic(self):
        cfg = _FakeConfig(content="hello world")
        result = fetch_document(cfg)
        assert result.data == b"hello world"
        assert result.detected_mime is None
        assert result.source_path is None

    def test_empty_content(self):
        cfg = _FakeConfig(content="")
        result = fetch_document(cfg)
        assert result.data == b""


class TestFetchFile:
    def test_absolute_path(self, tmp_path):
        f = tmp_path / "doc.md"
        f.write_text("# Hello")
        cfg = _FakeConfig(path=str(f))
        result = fetch_document(cfg)
        assert result.data == b"# Hello"
        assert result.source_path == str(f)

    def test_relative_path(self, tmp_path):
        f = tmp_path / "doc.md"
        f.write_text("content")
        cfg = _FakeConfig(path="doc.md")
        result = fetch_document(cfg, config_dir=str(tmp_path))
        assert result.data == b"content"

    def test_missing_file_raises(self, tmp_path):
        cfg = _FakeConfig(path=str(tmp_path / "nonexistent.md"))
        with pytest.raises(FileNotFoundError):
            fetch_document(cfg)


class TestFetchHttp:
    @patch("requests.get")
    def test_basic_http(self, mock_get):
        mock_response = MagicMock()
        mock_response.content = b"<html>page</html>"
        mock_response.headers = {"content-type": "text/html; charset=utf-8"}
        mock_get.return_value = mock_response

        cfg = _FakeConfig(url="https://example.com/page.html")
        result = fetch_document(cfg)

        assert result.data == b"<html>page</html>"
        assert result.detected_mime == "text/html; charset=utf-8"
        assert result.source_path == "https://example.com/page.html"
        mock_get.assert_called_once_with(
            "https://example.com/page.html", headers={}, timeout=30
        )
