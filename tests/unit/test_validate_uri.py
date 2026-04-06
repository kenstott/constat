# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Unit tests for validate_uri mutation logic."""

from __future__ import annotations

import urllib.error
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


async def _validate(uri: str):
    """Call validate_uri directly without a running GraphQL server."""
    from constat.server.graphql.source_resolvers import Mutation
    m = Mutation()
    # info is unused by validate_uri — pass None
    return await m.validate_uri(info=None, uri=uri)  # type: ignore[arg-type]


class TestValidateUri:
    @pytest.mark.asyncio
    async def test_http_reachable(self):
        with patch("urllib.request.urlopen") as mock_open:
            mock_open.return_value.__enter__ = lambda s: s
            mock_open.return_value.__exit__ = MagicMock(return_value=False)
            result = await _validate("https://example.com/page")
        assert result.reachable is True
        assert result.error is None

    @pytest.mark.asyncio
    async def test_http_unreachable(self):
        with patch("urllib.request.urlopen",
                   side_effect=urllib.error.URLError("connection refused")):
            result = await _validate("https://example.com/page")
        assert result.reachable is False
        assert "connection refused" in result.error

    @pytest.mark.asyncio
    async def test_http_4xx_treated_as_reachable(self):
        err = urllib.error.HTTPError(
            url="https://example.com", code=401, msg="Unauthorized",
            hdrs=None, fp=None,
        )
        with patch("urllib.request.urlopen", side_effect=err):
            result = await _validate("https://example.com/protected")
        assert result.reachable is True

    @pytest.mark.asyncio
    async def test_http_5xx_not_reachable(self):
        err = urllib.error.HTTPError(
            url="https://example.com", code=503, msg="Service Unavailable",
            hdrs=None, fp=None,
        )
        with patch("urllib.request.urlopen", side_effect=err):
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
