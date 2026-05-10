# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""MCP client — JSON-RPC 2.0 over Streamable HTTP and stdio transports."""

from __future__ import annotations

import itertools
import json
import logging
import subprocess
from typing import Any, Optional

import httpx

from constat.core.sources import AuthConfig

logger = logging.getLogger(__name__)

# Protocol constants
JSONRPC_VERSION = "2.0"
MCP_PROTOCOL_VERSION = "2025-03-26"
CLIENT_NAME = "constat"
CLIENT_VERSION = "1.0.0"


class McpError(Exception):
    """MCP protocol error with JSON-RPC error code."""

    def __init__(self, code: int, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"MCP error {code}: {message}")


class McpClient:
    """JSON-RPC 2.0 client for MCP servers over Streamable HTTP transport.

    Provides typed methods for MCP protocol operations: resource listing/reading,
    tool listing/calling, and lifecycle management (connect/disconnect).
    """

    def __init__(
        self,
        base_url: str,
        auth: Optional[AuthConfig] = None,
        timeout: int = 30,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._auth = auth
        self._timeout = timeout
        self._id_counter = itertools.count(1)
        self._session_id: Optional[str] = None
        self._capabilities: dict[str, Any] = {}
        self._http: Optional[httpx.AsyncClient] = None
        self._process: Optional[subprocess.Popen] = None

    @property
    def capabilities(self) -> dict[str, Any]:
        """Server capabilities returned during initialization."""
        return self._capabilities

    @property
    def _is_stdio(self) -> bool:
        """True when the transport is stdio (subprocess)."""
        return self._base_url.startswith("stdio://")

    @property
    def connected(self) -> bool:
        if self._is_stdio:
            return self._process is not None and self._process.poll() is None
        return self._http is not None and self._session_id is not None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> dict:
        """Send initialize request and return server capabilities.

        Performs the MCP initialization handshake:
        1. Send ``initialize`` with client info
        2. Store session ID from response header
        3. Send ``notifications/initialized`` notification
        4. Return server capabilities dict
        """
        if self._is_stdio:
            cmd = self._base_url[len("stdio://"):]
            self._process = subprocess.Popen(
                cmd.split(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        else:
            self._http = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
                headers=self._build_headers(),
            )

        result = await self._rpc("initialize", {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": CLIENT_NAME, "version": CLIENT_VERSION},
        })

        self._capabilities = result.get("capabilities", {})

        # Send initialized notification (no id = notification)
        await self._notify("notifications/initialized")

        logger.info("MCP connected to %s (capabilities: %s)", self._base_url, list(self._capabilities))
        return self._capabilities

    async def disconnect(self) -> None:
        """Clean shutdown — close the HTTP client or terminate subprocess."""
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
        if self._http is not None:
            await self._http.aclose()
            self._http = None
        self._session_id = None
        self._capabilities = {}
        logger.info("MCP disconnected from %s", self._base_url)

    # ------------------------------------------------------------------
    # Resources
    # ------------------------------------------------------------------

    async def list_resources(self, cursor: Optional[str] = None) -> dict:
        """List available resources.

        Returns:
            Dict with ``resources`` list and optional ``nextCursor``.
        """
        params: dict[str, Any] = {}
        if cursor is not None:
            params["cursor"] = cursor
        return await self._rpc("resources/list", params)

    async def read_resource(self, uri: str) -> dict:
        """Read a single resource by URI.

        Returns:
            Dict with ``contents`` list, each entry having ``uri``,
            ``mimeType``, and ``text`` or ``blob``.
        """
        return await self._rpc("resources/read", {"uri": uri})

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    async def list_tools(self, cursor: Optional[str] = None) -> dict:
        """List available tools.

        Returns:
            Dict with ``tools`` list and optional ``nextCursor``.
        """
        params: dict[str, Any] = {}
        if cursor is not None:
            params["cursor"] = cursor
        return await self._rpc("tools/list", params)

    async def call_tool(self, name: str, arguments: dict) -> dict:
        """Call a tool by name with the given arguments.

        Returns:
            Dict with ``content`` list, optional ``isError`` flag,
            and optional ``structuredContent``.
        """
        return await self._rpc("tools/call", {"name": name, "arguments": arguments})

    # ------------------------------------------------------------------
    # Internal transport
    # ------------------------------------------------------------------

    async def _rpc(self, method: str, params: Optional[dict] = None) -> dict:
        """Send a JSON-RPC 2.0 request and return the result.

        Raises:
            McpError: If the server returns a JSON-RPC error response.
            httpx.HTTPStatusError: On non-2xx HTTP status.
        """
        if self._is_stdio:
            return self._rpc_stdio(method, params)

        if self._http is None:
            raise McpError(-32000, "Not connected — call connect() first")

        request_id = next(self._id_counter)
        payload: dict[str, Any] = {
            "jsonrpc": JSONRPC_VERSION,
            "id": request_id,
            "method": method,
        }
        if params is not None:
            payload["params"] = params

        headers: dict[str, str] = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._session_id is not None:
            headers["Mcp-Session-Id"] = self._session_id

        response = await self._http.post("/", json=payload, headers=headers)
        response.raise_for_status()

        # Capture session ID from response header
        if "mcp-session-id" in response.headers:
            self._session_id = response.headers["mcp-session-id"]

        body = response.json()

        if "error" in body:
            err = body["error"]
            raise McpError(err["code"], err["message"])

        return body.get("result", {})

    async def _notify(self, method: str, params: Optional[dict] = None) -> None:
        """Send a JSON-RPC 2.0 notification (no id, no response expected)."""
        if self._is_stdio:
            payload: dict[str, Any] = {"jsonrpc": JSONRPC_VERSION, "method": method}
            if params is not None:
                payload["params"] = params
            self._send_stdio(payload)
            return

        if self._http is None:
            raise McpError(-32000, "Not connected — call connect() first")

        payload: dict[str, Any] = {
            "jsonrpc": JSONRPC_VERSION,
            "method": method,
        }
        if params is not None:
            payload["params"] = params

        headers: dict[str, str] = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._session_id is not None:
            headers["Mcp-Session-Id"] = self._session_id

        response = await self._http.post("/", json=payload, headers=headers)
        response.raise_for_status()

    # ------------------------------------------------------------------
    # Stdio transport helpers
    # ------------------------------------------------------------------

    def _send_stdio(self, payload: dict[str, Any]) -> None:
        """Write a JSON-RPC message to the subprocess stdin."""
        if self._process is None or self._process.stdin is None:
            raise McpError(-32000, "Stdio process not running")
        line = json.dumps(payload) + "\n"
        self._process.stdin.write(line.encode())
        self._process.stdin.flush()

    def _recv_stdio(self) -> dict[str, Any]:
        """Read a JSON-RPC response line from the subprocess stdout."""
        if self._process is None or self._process.stdout is None:
            raise McpError(-32000, "Stdio process not running")
        line = self._process.stdout.readline()
        if not line:
            raise McpError(-32000, "Stdio process closed stdout")
        return json.loads(line)

    def _rpc_stdio(self, method: str, params: Optional[dict] = None) -> dict:
        """Send a JSON-RPC request over stdio and return the result."""
        request_id = next(self._id_counter)
        payload: dict[str, Any] = {
            "jsonrpc": JSONRPC_VERSION,
            "id": request_id,
            "method": method,
        }
        if params is not None:
            payload["params"] = params

        self._send_stdio(payload)
        body = self._recv_stdio()

        if "error" in body:
            err = body["error"]
            raise McpError(err["code"], err["message"])

        return body.get("result", {})

    def _build_headers(self) -> dict[str, str]:
        """Build default headers including auth if configured."""
        headers: dict[str, str] = {}
        if self._auth is None:
            return headers

        if self._auth.method == "bearer" and self._auth.token:
            headers["Authorization"] = f"Bearer {self._auth.token}"
        elif self._auth.method == "api_key" and self._auth.api_key:
            headers[self._auth.api_key_header] = self._auth.api_key
        elif self._auth.method == "basic" and self._auth.username and self._auth.password:
            import base64
            credentials = base64.b64encode(
                f"{self._auth.username}:{self._auth.password}".encode()
            ).decode()
            headers["Authorization"] = f"Basic {credentials}"

        return headers
