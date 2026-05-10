# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tests for MCP stdio transport: URL detection, subprocess spawn, JSON-RPC, disconnect."""

from __future__ import annotations

import json
from io import BytesIO
from unittest.mock import patch, MagicMock

import pytest

from constat.mcp.client import McpClient, McpError


# ---------------------------------------------------------------------------
# Stdio URL detection
# ---------------------------------------------------------------------------

class TestStdioDetection:
    def test_stdio_url_detected(self) -> None:
        client = McpClient("stdio://python -m my_server")
        assert client._is_stdio is True

    def test_http_url_not_stdio(self) -> None:
        client = McpClient("http://localhost:8080")
        assert client._is_stdio is False

    def test_https_url_not_stdio(self) -> None:
        client = McpClient("https://mcp.example.com")
        assert client._is_stdio is False


# ---------------------------------------------------------------------------
# Subprocess spawn
# ---------------------------------------------------------------------------

class TestSubprocessSpawn:
    @pytest.mark.asyncio
    async def test_connect_spawns_process(self) -> None:
        mock_proc = MagicMock()
        mock_proc.stdin = BytesIO()
        mock_proc.poll.return_value = None

        # Build a response for initialize
        init_response = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"capabilities": {"resources": {}, "tools": {}}},
        }).encode() + b"\n"
        mock_proc.stdout = BytesIO(init_response)

        with patch("constat.mcp.client.subprocess.Popen", return_value=mock_proc) as mock_popen:
            client = McpClient("stdio://python -m my_server --port 0")
            result = await client.connect()

        mock_popen.assert_called_once_with(
            ["python", "-m", "my_server", "--port", "0"],
            stdin=-1,  # subprocess.PIPE
            stdout=-1,
            stderr=-1,
        )
        assert "resources" in result
        assert client.connected is True


# ---------------------------------------------------------------------------
# JSON-RPC over stdin/stdout
# ---------------------------------------------------------------------------

class TestStdioJsonRpc:
    @pytest.mark.asyncio
    async def test_rpc_sends_and_receives(self) -> None:
        client = McpClient("stdio://echo")
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None

        stdin_buf = BytesIO()
        mock_proc.stdin = stdin_buf
        mock_proc.stdin.write = stdin_buf.write
        mock_proc.stdin.flush = MagicMock()

        # Queue two responses: initialize + list_resources
        init_resp = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}}) + "\n"
        list_resp = json.dumps({"jsonrpc": "2.0", "id": 2, "result": {"resources": []}}) + "\n"
        mock_proc.stdout = BytesIO((init_resp + list_resp).encode())

        with patch("constat.mcp.client.subprocess.Popen", return_value=mock_proc):
            await client.connect()
            result = await client._rpc("resources/list", {})

        assert result == {"resources": []}

    @pytest.mark.asyncio
    async def test_rpc_error_raises(self) -> None:
        client = McpClient("stdio://echo")
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None

        mock_proc.stdin = BytesIO()
        mock_proc.stdin.flush = MagicMock()

        init_resp = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}}) + "\n"
        err_resp = json.dumps({"jsonrpc": "2.0", "id": 2, "error": {"code": -32601, "message": "Method not found"}}) + "\n"
        mock_proc.stdout = BytesIO((init_resp + err_resp).encode())

        with patch("constat.mcp.client.subprocess.Popen", return_value=mock_proc):
            await client.connect()
            with pytest.raises(McpError, match="Method not found"):
                await client._rpc("unknown/method", {})


# ---------------------------------------------------------------------------
# Disconnect kills process
# ---------------------------------------------------------------------------

class TestStdioDisconnect:
    @pytest.mark.asyncio
    async def test_disconnect_terminates_process(self) -> None:
        client = McpClient("stdio://echo")
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.stdin = BytesIO()
        mock_proc.stdin.flush = MagicMock()

        init_resp = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}}) + "\n"
        mock_proc.stdout = BytesIO(init_resp.encode())

        with patch("constat.mcp.client.subprocess.Popen", return_value=mock_proc):
            await client.connect()

        await client.disconnect()

        mock_proc.terminate.assert_called_once()
        mock_proc.wait.assert_called_once_with(timeout=5)
        assert client._process is None
