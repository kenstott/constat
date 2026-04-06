# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Unit tests for validate_uri query logic."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


async def _validate(uri: str):
    """Call validate_uri directly without a running GraphQL server."""
    from constat.server.graphql.source_resolvers import Query
    q = Query()
    # info is unused by validate_uri — pass None
    return await q.validate_uri(info=None, uri=uri)  # type: ignore[arg-type]


def _mock_http_session(status_code: int = 200, raises: Exception | None = None):
    """Return a mock requests.Session where head() returns the given status."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.close = MagicMock()

    mock_session = MagicMock()
    if raises:
        mock_session.head.side_effect = raises
        mock_session.get.side_effect = raises
    else:
        mock_session.head.return_value = mock_resp
        mock_session.get.return_value = mock_resp

    return mock_session


class TestValidateUri:
    @pytest.mark.asyncio
    async def test_http_reachable(self):
        mock_session = _mock_http_session(200)
        with patch("constat.discovery.doc_tools._transport._get_http_session", return_value=mock_session):
            result = await _validate("https://example.com/page")
        assert result.reachable is True
        assert result.error is None

    @pytest.mark.asyncio
    async def test_http_unreachable(self):
        mock_session = _mock_http_session(raises=Exception("connection refused"))
        with patch("constat.discovery.doc_tools._transport._get_http_session", return_value=mock_session):
            result = await _validate("https://example.com/page")
        assert result.reachable is False
        assert "connection refused" in result.error

    @pytest.mark.asyncio
    async def test_http_4xx_treated_as_reachable(self):
        mock_session = _mock_http_session(401)
        with patch("constat.discovery.doc_tools._transport._get_http_session", return_value=mock_session):
            result = await _validate("https://example.com/protected")
        assert result.reachable is True

    @pytest.mark.asyncio
    async def test_http_5xx_not_reachable(self):
        mock_session = _mock_http_session(503)
        with patch("constat.discovery.doc_tools._transport._get_http_session", return_value=mock_session):
            result = await _validate("https://example.com/broken")
        assert result.reachable is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_file_scheme_exists(self):
        with patch("os.path.exists", return_value=True):
            result = await _validate("file:///etc/passwd")
        assert result.reachable is True
        assert result.error is None

    @pytest.mark.asyncio
    async def test_file_scheme_missing(self):
        with patch("os.path.exists", return_value=False):
            result = await _validate("file:///etc/nonexistent")
        assert result.reachable is False
        assert "/etc/nonexistent" in result.error

    @pytest.mark.asyncio
    async def test_s3_valid_format(self):
        result = await _validate("s3://my-bucket/data/file.parquet")
        assert result.reachable is True

    @pytest.mark.asyncio
    async def test_s3_missing_bucket(self):
        result = await _validate("s3:///key-without-bucket")
        assert result.reachable is False

    @pytest.mark.asyncio
    async def test_s3a_valid_format(self):
        result = await _validate("s3a://my-bucket/key")
        assert result.reachable is True

    @pytest.mark.asyncio
    async def test_ftp_valid_format(self):
        result = await _validate("ftp://ftp.example.com/file.txt")
        assert result.reachable is True

    @pytest.mark.asyncio
    async def test_ftp_missing_host(self):
        result = await _validate("ftp:///file.txt")
        assert result.reachable is False

    @pytest.mark.asyncio
    async def test_bare_path_exists(self):
        with patch("os.path.exists", return_value=True):
            result = await _validate("/data/rules.md")
        assert result.reachable is True

    @pytest.mark.asyncio
    async def test_bare_path_missing(self):
        with patch("os.path.exists", return_value=False):
            result = await _validate("/data/missing.md")
        assert result.reachable is False
