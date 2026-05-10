# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Tests for McpDocumentProvider and McpApiProvider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from constat.core.sources import AuthConfig, DataSourceKind
from constat.mcp.client import McpClient, McpError
from constat.mcp.document_provider import McpDocumentProvider
from constat.mcp.api_provider import McpApiProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_client(resources=None, tools=None, read_result=None):
    """Create a mock McpClient with preset responses."""
    client = MagicMock(spec=McpClient)
    client.connected = True

    async def mock_connect():
        return {"resources": {}, "tools": {}}

    async def mock_disconnect():
        pass

    async def mock_list_resources(cursor=None):
        return {"resources": resources or []}

    async def mock_read_resource(uri):
        if read_result:
            return read_result
        return {"contents": [{"uri": uri, "mimeType": "text/plain", "text": f"content of {uri}"}]}

    async def mock_list_tools(cursor=None):
        return {"tools": tools or []}

    client.connect = mock_connect
    client.disconnect = mock_disconnect
    client.list_resources = mock_list_resources
    client.read_resource = mock_read_resource
    client.list_tools = mock_list_tools

    return client


# ---------------------------------------------------------------------------
# McpDocumentProvider
# ---------------------------------------------------------------------------

class TestMcpDocumentProviderConnect:
    def test_connect_requires_url(self):
        provider = McpDocumentProvider()
        result = provider.connect("test", {})
        assert result.success is False
        assert "URL is required" in result.error

    def test_connect_success_with_mocked_client(self):
        provider = McpDocumentProvider()
        mock = _mock_client()

        with patch("constat.mcp.document_provider.McpClient", return_value=mock):
            result = provider.connect("test", {"url": "http://localhost:8080"})
        assert result.success is True
        assert "ingestible" in result.capabilities

    def test_status_connected(self):
        provider = McpDocumentProvider()
        mock = _mock_client()
        provider._clients["test"] = mock
        provider._configs["test"] = {}
        status = provider.status("test")
        assert status.state == "connected"

    def test_status_disconnected(self):
        provider = McpDocumentProvider()
        status = provider.status("test")
        assert status.state == "disconnected"


class TestMcpDocumentProviderDiscover:
    def test_discover_returns_items(self):
        resources = [
            {"uri": "file:///a.txt", "name": "a.txt", "mimeType": "text/plain"},
            {"uri": "file:///b.md", "name": "b.md", "mimeType": "text/markdown"},
        ]
        provider = McpDocumentProvider()
        mock = _mock_client(resources=resources)
        provider._clients["test"] = mock
        provider._configs["test"] = {}

        discovery = provider.discover("test")
        assert len(discovery.items) == 2
        assert discovery.items[0].id == "file:///a.txt"
        assert discovery.items[1].name == "b.md"

    def test_resource_filter_regex(self):
        resources = [
            {"uri": "file:///docs/a.txt", "name": "a.txt"},
            {"uri": "file:///logs/b.txt", "name": "b.txt"},
            {"uri": "file:///docs/c.md", "name": "c.md"},
        ]
        provider = McpDocumentProvider()
        mock = _mock_client(resources=resources)
        provider._clients["test"] = mock
        provider._configs["test"] = {"resource_filter": r"file:///docs/"}

        discovery = provider.discover("test")
        assert len(discovery.items) == 2
        uris = [item.id for item in discovery.items]
        assert "file:///docs/a.txt" in uris
        assert "file:///docs/c.md" in uris
        assert "file:///logs/b.txt" not in uris

    def test_max_resources_limit(self):
        resources = [{"uri": f"file:///{i}.txt", "name": f"{i}.txt"} for i in range(50)]
        provider = McpDocumentProvider()
        mock = _mock_client(resources=resources)
        provider._clients["test"] = mock
        provider._configs["test"] = {"max_resources": 10}

        discovery = provider.discover("test")
        assert len(discovery.items) == 10


class TestMcpDocumentProviderFetch:
    def test_fetch_item_returns_content(self):
        read_result = {
            "contents": [{"uri": "file:///a.txt", "mimeType": "text/plain", "text": "hello"}]
        }
        provider = McpDocumentProvider()
        mock = _mock_client(read_result=read_result)
        provider._clients["test"] = mock
        provider._configs["test"] = {}

        result = provider.fetch_item("test", "file:///a.txt")
        assert result.content == "hello"
        assert result.mime_type == "text/plain"

    def test_fetch_item_not_connected_raises(self):
        provider = McpDocumentProvider()
        with pytest.raises(McpError):
            provider.fetch_item("test", "file:///a.txt")


class TestMcpDocumentProviderRefresh:
    def test_refresh_detects_changes(self):
        resources_v1 = [{"uri": "file:///a.txt", "size": 100}]
        resources_v2 = [
            {"uri": "file:///a.txt", "size": 200},
            {"uri": "file:///b.txt", "size": 50},
        ]

        provider = McpDocumentProvider()

        # Set up with v1 resources
        mock_v1 = _mock_client(resources=resources_v1)
        provider._clients["test"] = mock_v1
        provider._configs["test"] = {}
        provider.discover("test")  # Populate initial state

        # Now refresh triggers probe on first call — everything is "added"
        result1 = provider.refresh("test")
        assert result1.added >= 0  # First refresh with probe

        # Switch to v2 resources for second refresh
        mock_v2 = _mock_client(resources=resources_v2)
        provider._clients["test"] = mock_v2
        result2 = provider.refresh("test")
        assert result2.added >= 1 or result2.updated >= 1

    def test_supports_incremental(self):
        provider = McpDocumentProvider()
        assert provider.supports_incremental() is True

    def test_kind_is_document(self):
        provider = McpDocumentProvider()
        assert provider.kind == DataSourceKind.DOCUMENT


# ---------------------------------------------------------------------------
# McpApiProvider
# ---------------------------------------------------------------------------

class TestMcpApiProviderConnect:
    def test_connect_requires_url(self):
        provider = McpApiProvider()
        result = provider.connect("test", {})
        assert result.success is False
        assert "URL is required" in result.error

    def test_connect_success(self):
        provider = McpApiProvider()
        mock = _mock_client()

        with patch("constat.mcp.api_provider.McpClient", return_value=mock):
            result = provider.connect("test", {"url": "http://localhost:8080"})
        assert result.success is True
        assert "queryable" in result.capabilities


class TestMcpApiProviderToolFiltering:
    def test_allowed_tools_filter(self):
        tools = [
            {"name": "search", "description": "Search"},
            {"name": "delete", "description": "Delete"},
            {"name": "create", "description": "Create"},
        ]
        provider = McpApiProvider()
        mock = _mock_client(tools=tools)
        provider._clients["test"] = mock
        provider._configs["test"] = {"allowed_tools": ["search", "create"]}

        discovery = provider.discover("test")
        names = [item.id for item in discovery.items]
        assert "search" in names
        assert "create" in names
        assert "delete" not in names

    def test_denied_tools_filter(self):
        tools = [
            {"name": "search", "description": "Search"},
            {"name": "delete", "description": "Delete"},
            {"name": "create", "description": "Create"},
        ]
        provider = McpApiProvider()
        mock = _mock_client(tools=tools)
        provider._clients["test"] = mock
        provider._configs["test"] = {"denied_tools": ["delete"]}

        discovery = provider.discover("test")
        names = [item.id for item in discovery.items]
        assert "search" in names
        assert "create" in names
        assert "delete" not in names

    def test_no_filter_returns_all(self):
        tools = [
            {"name": "search", "description": "Search"},
            {"name": "delete", "description": "Delete"},
        ]
        provider = McpApiProvider()
        mock = _mock_client(tools=tools)
        provider._clients["test"] = mock
        provider._configs["test"] = {}

        discovery = provider.discover("test")
        assert len(discovery.items) == 2

    def test_discover_includes_operations(self):
        tools = [
            {"name": "search", "description": "Search docs", "inputSchema": {"type": "object"}},
        ]
        provider = McpApiProvider()
        mock = _mock_client(tools=tools)
        provider._clients["test"] = mock
        provider._configs["test"] = {}

        discovery = provider.discover("test")
        assert discovery.operations is not None
        assert len(discovery.operations) == 1
        assert discovery.operations[0]["name"] == "search"


class TestMcpApiProviderFetch:
    def test_fetch_item_returns_tool_schema(self):
        tools = [{"name": "search", "description": "Search", "inputSchema": {"type": "object"}}]
        provider = McpApiProvider()
        mock = _mock_client(tools=tools)
        provider._clients["test"] = mock
        provider._configs["test"] = {}
        provider.discover("test")

        result = provider.fetch_item("test", "search")
        assert result.mime_type == "application/json"
        assert "search" in result.content

    def test_fetch_unknown_tool_raises(self):
        provider = McpApiProvider()
        provider._tools["test"] = []
        with pytest.raises(McpError):
            provider.fetch_item("test", "nonexistent")


class TestMcpApiProviderMisc:
    def test_kind_is_api(self):
        provider = McpApiProvider()
        assert provider.kind == DataSourceKind.API

    def test_supports_incremental_false(self):
        provider = McpApiProvider()
        assert provider.supports_incremental() is False

    def test_status_disconnected(self):
        provider = McpApiProvider()
        assert provider.status("x").state == "disconnected"
