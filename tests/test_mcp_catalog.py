# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tests for MCP catalog: seed loading, remote fetch, cache TTL, filters."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from constat.mcp.catalog import McpCatalog


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_cache(tmp_path: Path) -> Path:
    return tmp_path / "cache"


@pytest.fixture
def catalog(tmp_cache: Path) -> McpCatalog:
    return McpCatalog(cache_dir=tmp_cache)


REMOTE_SERVERS = [
    {"name": "Remote A", "slug": "remote-a", "description": "Remote server A", "category": "development", "capabilities": ["tools"]},
    {"name": "Remote B", "slug": "remote-b", "description": "Remote server B", "category": "communication", "capabilities": ["resources"]},
]


# ---------------------------------------------------------------------------
# Seed loading
# ---------------------------------------------------------------------------

class TestSeedLoading:
    def test_load_seed_returns_servers(self, catalog: McpCatalog) -> None:
        """Seed file should load and return a non-empty list."""
        servers = catalog._load_seed()
        slugs = [s["slug"] for s in servers]
        assert "github" in slugs, f"Expected 'github' in seed slugs, got: {slugs}"
        assert servers[0]["slug"] == "github"

    def test_load_seed_has_expected_fields(self, catalog: McpCatalog) -> None:
        servers = catalog._load_seed()
        for s in servers:
            assert "name" in s
            assert "slug" in s
            assert "description" in s
            assert "category" in s
            assert "capabilities" in s


# ---------------------------------------------------------------------------
# Remote fetch with mocked httpx
# ---------------------------------------------------------------------------

class TestRemoteFetch:
    def test_fetch_remote_success(self, catalog: McpCatalog) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"servers": REMOTE_SERVERS}
        mock_resp.raise_for_status = MagicMock()

        with patch("constat.mcp.catalog.httpx.get", return_value=mock_resp):
            result = catalog._fetch_remote()

        assert result is not None
        assert len(result) == 2
        assert result[0]["slug"] == "remote-a"

    def test_fetch_remote_list_format(self, catalog: McpCatalog) -> None:
        """API may return a bare list instead of {servers: [...]}."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = REMOTE_SERVERS
        mock_resp.raise_for_status = MagicMock()

        with patch("constat.mcp.catalog.httpx.get", return_value=mock_resp):
            result = catalog._fetch_remote()

        assert result == REMOTE_SERVERS

    def test_fetch_remote_failure_returns_none(self, catalog: McpCatalog) -> None:
        with patch("constat.mcp.catalog.httpx.get", side_effect=Exception("network")):
            result = catalog._fetch_remote()
        assert result is None


# ---------------------------------------------------------------------------
# Cache TTL
# ---------------------------------------------------------------------------

class TestCacheTTL:
    def test_fresh_cache_is_used(self, catalog: McpCatalog, tmp_cache: Path) -> None:
        tmp_cache.mkdir(parents=True, exist_ok=True)
        cache_data = {"cached_at": time.time(), "servers": REMOTE_SERVERS}
        (tmp_cache / "mcp_catalog.json").write_text(json.dumps(cache_data))

        # Should use cache, not fetch remote
        with patch("constat.mcp.catalog.httpx.get") as mock_get:
            servers = catalog.list_servers()
            mock_get.assert_not_called()

        assert len(servers) == 2

    def test_stale_cache_triggers_remote(self, catalog: McpCatalog, tmp_cache: Path) -> None:
        tmp_cache.mkdir(parents=True, exist_ok=True)
        stale_time = time.time() - McpCatalog.CACHE_TTL - 100
        cache_data = {"cached_at": stale_time, "servers": REMOTE_SERVERS}
        (tmp_cache / "mcp_catalog.json").write_text(json.dumps(cache_data))

        with patch("constat.mcp.catalog.httpx.get", side_effect=Exception("down")):
            servers = catalog.list_servers()

        # Falls back to stale cache
        assert len(servers) == 2

    def test_stale_cache_replaced_by_remote(self, catalog: McpCatalog, tmp_cache: Path) -> None:
        tmp_cache.mkdir(parents=True, exist_ok=True)
        stale_time = time.time() - McpCatalog.CACHE_TTL - 100
        cache_data = {"cached_at": stale_time, "servers": [{"slug": "old"}]}
        (tmp_cache / "mcp_catalog.json").write_text(json.dumps(cache_data))

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"servers": REMOTE_SERVERS}
        mock_resp.raise_for_status = MagicMock()

        with patch("constat.mcp.catalog.httpx.get", return_value=mock_resp):
            servers = catalog.list_servers()

        assert len(servers) == 2
        assert servers[0]["slug"] == "remote-a"


# ---------------------------------------------------------------------------
# Fallback chain: remote down -> cache -> seed
# ---------------------------------------------------------------------------

class TestFallbackChain:
    def test_remote_down_no_cache_falls_to_seed(self, catalog: McpCatalog) -> None:
        with patch("constat.mcp.catalog.httpx.get", side_effect=Exception("down")):
            servers = catalog.list_servers()

        slugs = [s["slug"] for s in servers]
        assert "github" in slugs, f"Expected 'github' in fallback seed slugs, got: {slugs}"
        assert servers[0]["slug"] == "github"  # from seed

    def test_remote_down_stale_cache_used(self, catalog: McpCatalog, tmp_cache: Path) -> None:
        tmp_cache.mkdir(parents=True, exist_ok=True)
        stale_time = time.time() - McpCatalog.CACHE_TTL - 100
        cache_data = {"cached_at": stale_time, "servers": REMOTE_SERVERS}
        (tmp_cache / "mcp_catalog.json").write_text(json.dumps(cache_data))

        with patch("constat.mcp.catalog.httpx.get", side_effect=Exception("down")):
            servers = catalog.list_servers()

        assert servers == REMOTE_SERVERS


# ---------------------------------------------------------------------------
# Category filter
# ---------------------------------------------------------------------------

class TestCategoryFilter:
    def test_filter_by_category(self, catalog: McpCatalog) -> None:
        with patch("constat.mcp.catalog.httpx.get", side_effect=Exception("down")):
            servers = catalog.list_servers(category="development")

        slugs = [s["slug"] for s in servers]
        assert "github" in slugs
        assert "slack" not in slugs

    def test_filter_by_nonexistent_category(self, catalog: McpCatalog) -> None:
        with patch("constat.mcp.catalog.httpx.get", side_effect=Exception("down")):
            servers = catalog.list_servers(category="nonexistent")
        assert servers == []


# ---------------------------------------------------------------------------
# Search query
# ---------------------------------------------------------------------------

class TestSearchQuery:
    def test_search_by_name(self, catalog: McpCatalog) -> None:
        with patch("constat.mcp.catalog.httpx.get", side_effect=Exception("down")):
            servers = catalog.list_servers(query="github")
        assert len(servers) == 1
        assert servers[0]["slug"] == "github"

    def test_search_by_description(self, catalog: McpCatalog) -> None:
        with patch("constat.mcp.catalog.httpx.get", side_effect=Exception("down")):
            servers = catalog.list_servers(query="channels")
        assert len(servers) == 1
        assert servers[0]["slug"] == "slack"

    def test_search_case_insensitive(self, catalog: McpCatalog) -> None:
        with patch("constat.mcp.catalog.httpx.get", side_effect=Exception("down")):
            servers = catalog.list_servers(query="GITHUB")
        assert len(servers) == 1


# ---------------------------------------------------------------------------
# Slug lookup
# ---------------------------------------------------------------------------

class TestSlugLookup:
    def test_get_server_found(self, catalog: McpCatalog) -> None:
        with patch("constat.mcp.catalog.httpx.get", side_effect=Exception("down")):
            server = catalog.get_server("jira")
        assert server is not None
        assert server["name"] == "Jira"

    def test_get_server_not_found(self, catalog: McpCatalog) -> None:
        with patch("constat.mcp.catalog.httpx.get", side_effect=Exception("down")):
            server = catalog.get_server("nonexistent")
        assert server is None
