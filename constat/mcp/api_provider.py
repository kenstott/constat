# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""MCP API provider — exposes MCP tools as callable API operations."""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from constat.core.sources import (
    AuthConfig,
    ConnectionResult,
    DataSourceKind,
    DiscoveryResult,
    FetchResult,
    RefreshResult,
    SourceItem,
    SourceStatus,
)
from constat.mcp.client import McpClient, McpError

logger = logging.getLogger(__name__)


class McpApiProvider:
    """DataSourceProvider that maps MCP server tools to API operations.

    Config keys:
        ``url`` — MCP server base URL (required).
        ``allowed_tools`` — explicit allowlist of tool names (optional).
        ``denied_tools`` — denylist of tool names (optional).

    If both ``allowed_tools`` and ``denied_tools`` are set, a tool must be
    in the allowlist AND not in the denylist.
    """

    kind = DataSourceKind.API

    def __init__(self) -> None:
        self._clients: dict[str, McpClient] = {}
        self._configs: dict[str, dict] = {}
        self._tools: dict[str, list[dict]] = {}

    def connect(self, name: str, config: dict, auth: Optional[AuthConfig] = None) -> ConnectionResult:
        url = config.get("url", "")
        if not url:
            return ConnectionResult(success=False, error="MCP server URL is required")

        client = McpClient(url, auth=auth, timeout=config.get("timeout", 30))

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    capabilities = pool.submit(asyncio.run, client.connect()).result()
            else:
                capabilities = loop.run_until_complete(client.connect())
        except Exception as exc:
            return ConnectionResult(success=False, error=str(exc))

        self._clients[name] = client
        self._configs[name] = config
        return ConnectionResult(success=True, capabilities={"queryable"})

    def disconnect(self, name: str) -> None:
        client = self._clients.pop(name, None)
        if client is not None:
            try:
                asyncio.get_event_loop().run_until_complete(client.disconnect())
            except RuntimeError:
                asyncio.run(client.disconnect())
        self._configs.pop(name, None)
        self._tools.pop(name, None)

    def status(self, name: str) -> SourceStatus:
        client = self._clients.get(name)
        if client is None:
            return SourceStatus(state="disconnected")
        if not client.connected:
            return SourceStatus(state="error", error="Client not connected")
        item_count = len(self._tools.get(name, []))
        return SourceStatus(state="connected", item_count=item_count)

    def discover(self, name: str) -> DiscoveryResult:
        tools = self._fetch_tools(name)
        self._tools[name] = tools
        items = [self._tool_to_item(t) for t in tools]
        operations = [
            {
                "name": t["name"],
                "description": t.get("description", ""),
                "inputSchema": t.get("inputSchema", {}),
            }
            for t in tools
        ]
        return DiscoveryResult(items=items, operations=operations)

    def list_items(self, name: str) -> list[SourceItem]:
        tools = self._tools.get(name)
        if tools is None:
            tools = self._fetch_tools(name)
            self._tools[name] = tools
        return [self._tool_to_item(t) for t in tools]

    def fetch_item(self, name: str, item_id: str) -> FetchResult:
        """Fetch tool metadata (schema). Use call_tool for execution."""
        tools = self._tools.get(name, [])
        for t in tools:
            if t["name"] == item_id:
                import json
                content = json.dumps(t, indent=2)
                return FetchResult(content=content, mime_type="application/json", metadata={"tool": item_id})
        raise McpError(-32000, f"Tool '{item_id}' not found")

    def refresh(self, name: str) -> RefreshResult:
        old_names = {t["name"] for t in self._tools.get(name, [])}
        tools = self._fetch_tools(name)
        new_names = {t["name"] for t in tools}
        self._tools[name] = tools

        added = len(new_names - old_names)
        removed = len(old_names - new_names)
        return RefreshResult(added=added, removed=removed)

    def supports_incremental(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_tools(self, name: str) -> list[dict]:
        """Fetch and filter tools from the MCP server."""
        client = self._clients.get(name)
        if client is None:
            raise McpError(-32000, f"Source '{name}' not connected")

        config = self._configs.get(name, {})
        allowed = set(config.get("allowed_tools") or [])
        denied = set(config.get("denied_tools") or [])

        all_tools: list[dict] = []
        cursor: Optional[str] = None

        while True:
            result = self._run_async(client.list_tools(cursor=cursor))
            for t in result.get("tools", []):
                tool_name = t.get("name", "")
                if allowed and tool_name not in allowed:
                    continue
                if denied and tool_name in denied:
                    continue
                all_tools.append(t)

            cursor = result.get("nextCursor")
            if cursor is None:
                break

        return all_tools

    @staticmethod
    def _tool_to_item(tool: dict) -> SourceItem:
        """Convert an MCP tool dict to a SourceItem."""
        return SourceItem(
            id=tool.get("name", ""),
            name=tool.get("name", ""),
            description=tool.get("description", ""),
            mime_type="application/json",
        )

    @staticmethod
    def _run_async(coro: object) -> dict:
        """Run an async coroutine from sync context."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    return pool.submit(asyncio.run, coro).result()
            return loop.run_until_complete(coro)
        except RuntimeError:
            return asyncio.run(coro)
