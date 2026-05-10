# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Tests for constat.mcp.client — McpClient and McpError."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from constat.core.sources import AuthConfig
from constat.mcp.client import McpClient, McpError, JSONRPC_VERSION


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_REQUEST = httpx.Request("POST", "http://localhost:8080/")


def _jsonrpc_response(result: dict, request_id: int = 1, session_id: str = "sess-1") -> httpx.Response:
    """Build a mock httpx.Response with a JSON-RPC result."""
    body = {"jsonrpc": JSONRPC_VERSION, "id": request_id, "result": result}
    return httpx.Response(
        status_code=200,
        json=body,
        headers={"mcp-session-id": session_id, "content-type": "application/json"},
        request=_FAKE_REQUEST,
    )


def _jsonrpc_error(code: int, message: str, request_id: int = 1) -> httpx.Response:
    """Build a mock httpx.Response with a JSON-RPC error."""
    body = {"jsonrpc": JSONRPC_VERSION, "id": request_id, "error": {"code": code, "message": message}}
    return httpx.Response(
        status_code=200,
        json=body,
        headers={"content-type": "application/json"},
        request=_FAKE_REQUEST,
    )


def _notification_response() -> httpx.Response:
    """Response for notifications (no body needed, just 200)."""
    return httpx.Response(status_code=200, content=b"", headers={}, request=_FAKE_REQUEST)


# ---------------------------------------------------------------------------
# McpError
# ---------------------------------------------------------------------------

class TestMcpError:
    def test_attributes(self):
        err = McpError(-32600, "Invalid Request")
        assert err.code == -32600
        assert err.message == "Invalid Request"
        assert "-32600" in str(err)
        assert "Invalid Request" in str(err)


# ---------------------------------------------------------------------------
# McpClient.connect
# ---------------------------------------------------------------------------

class TestConnect:
    @pytest.mark.asyncio
    async def test_connect_returns_capabilities(self):
        capabilities = {"resources": {"listChanged": True}, "tools": {}}
        init_result = {
            "protocolVersion": "2025-03-26",
            "capabilities": capabilities,
            "serverInfo": {"name": "test-server", "version": "1.0"},
        }

        call_count = 0

        async def mock_post(path, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _jsonrpc_response(init_result, request_id=1)
            # notification response
            return _notification_response()

        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.post = mock_post

        with patch("constat.mcp.client.httpx.AsyncClient", return_value=mock_http):
            client = McpClient("http://localhost:8080")
            result = await client.connect()
        assert result == capabilities
        assert client.connected is True

    @pytest.mark.asyncio
    async def test_connect_stores_session_id(self):
        init_result = {"protocolVersion": "2025-03-26", "capabilities": {}, "serverInfo": {"name": "s"}}

        call_count = 0

        async def mock_post(path, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _jsonrpc_response(init_result, session_id="my-session-42")
            return _notification_response()

        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.post = mock_post

        with patch("constat.mcp.client.httpx.AsyncClient", return_value=mock_http):
            client = McpClient("http://localhost:8080")
            await client.connect()
        assert client._session_id == "my-session-42"


# ---------------------------------------------------------------------------
# McpClient.list_resources / read_resource
# ---------------------------------------------------------------------------

class TestResources:
    @pytest.mark.asyncio
    async def test_list_resources(self):
        resources = [
            {"uri": "file:///a.txt", "name": "a.txt", "mimeType": "text/plain"},
            {"uri": "file:///b.md", "name": "b.md", "mimeType": "text/markdown"},
        ]

        client = McpClient("http://localhost:8080")
        client._http = AsyncMock(spec=httpx.AsyncClient)
        client._session_id = "s1"
        client._http.post = AsyncMock(return_value=_jsonrpc_response({"resources": resources}))

        result = await client.list_resources()
        assert len(result["resources"]) == 2
        assert result["resources"][0]["uri"] == "file:///a.txt"

    @pytest.mark.asyncio
    async def test_list_resources_with_cursor(self):
        client = McpClient("http://localhost:8080")
        client._http = AsyncMock(spec=httpx.AsyncClient)
        client._session_id = "s1"
        client._http.post = AsyncMock(return_value=_jsonrpc_response({"resources": [], "nextCursor": "page2"}))

        result = await client.list_resources(cursor="page1")
        # Verify cursor was passed in params
        call_args = client._http.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert payload["params"]["cursor"] == "page1"

    @pytest.mark.asyncio
    async def test_read_resource(self):
        contents = [{"uri": "file:///a.txt", "mimeType": "text/plain", "text": "hello world"}]

        client = McpClient("http://localhost:8080")
        client._http = AsyncMock(spec=httpx.AsyncClient)
        client._session_id = "s1"
        client._http.post = AsyncMock(return_value=_jsonrpc_response({"contents": contents}))

        result = await client.read_resource("file:///a.txt")
        assert result["contents"][0]["text"] == "hello world"


# ---------------------------------------------------------------------------
# McpClient.list_tools / call_tool
# ---------------------------------------------------------------------------

class TestTools:
    @pytest.mark.asyncio
    async def test_list_tools(self):
        tools = [
            {"name": "search", "description": "Search docs", "inputSchema": {"type": "object"}},
        ]

        client = McpClient("http://localhost:8080")
        client._http = AsyncMock(spec=httpx.AsyncClient)
        client._session_id = "s1"
        client._http.post = AsyncMock(return_value=_jsonrpc_response({"tools": tools}))

        result = await client.list_tools()
        assert result["tools"][0]["name"] == "search"

    @pytest.mark.asyncio
    async def test_call_tool(self):
        tool_result = {"content": [{"type": "text", "text": "result data"}]}

        client = McpClient("http://localhost:8080")
        client._http = AsyncMock(spec=httpx.AsyncClient)
        client._session_id = "s1"
        client._http.post = AsyncMock(return_value=_jsonrpc_response(tool_result))

        result = await client.call_tool("search", {"query": "test"})
        assert result["content"][0]["text"] == "result data"

        call_args = client._http.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert payload["params"]["name"] == "search"
        assert payload["params"]["arguments"] == {"query": "test"}


# ---------------------------------------------------------------------------
# JSON-RPC error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_jsonrpc_error_raises_mcp_error(self):
        client = McpClient("http://localhost:8080")
        client._http = AsyncMock(spec=httpx.AsyncClient)
        client._session_id = "s1"
        client._http.post = AsyncMock(return_value=_jsonrpc_error(-32601, "Method not found"))

        with pytest.raises(McpError) as exc_info:
            await client.list_resources()
        assert exc_info.value.code == -32601
        assert "Method not found" in exc_info.value.message

    @pytest.mark.asyncio
    async def test_not_connected_raises(self):
        client = McpClient("http://localhost:8080")
        with pytest.raises(McpError) as exc_info:
            await client.list_resources()
        assert "Not connected" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Auth header construction
# ---------------------------------------------------------------------------

class TestAuth:
    def test_bearer_auth_header(self):
        auth = AuthConfig(method="bearer", token="my-token-123")
        client = McpClient("http://localhost:8080", auth=auth)
        headers = client._build_headers()
        assert headers["Authorization"] == "Bearer my-token-123"

    def test_api_key_auth_header(self):
        auth = AuthConfig(method="api_key", api_key="key-456", api_key_header="X-Custom-Key")
        client = McpClient("http://localhost:8080", auth=auth)
        headers = client._build_headers()
        assert headers["X-Custom-Key"] == "key-456"

    def test_basic_auth_header(self):
        import base64
        auth = AuthConfig(method="basic", username="user", password="pass")
        client = McpClient("http://localhost:8080", auth=auth)
        headers = client._build_headers()
        expected = base64.b64encode(b"user:pass").decode()
        assert headers["Authorization"] == f"Basic {expected}"

    def test_no_auth_empty_headers(self):
        client = McpClient("http://localhost:8080")
        headers = client._build_headers()
        assert headers == {}


# ---------------------------------------------------------------------------
# Disconnect
# ---------------------------------------------------------------------------

class TestDisconnect:
    @pytest.mark.asyncio
    async def test_disconnect_clears_state(self):
        client = McpClient("http://localhost:8080")
        client._http = AsyncMock(spec=httpx.AsyncClient)
        client._session_id = "s1"
        client._capabilities = {"resources": {}}

        await client.disconnect()
        assert client._http is None
        assert client._session_id is None
        assert client._capabilities == {}
        assert client.connected is False
