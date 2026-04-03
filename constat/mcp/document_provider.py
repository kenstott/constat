# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""MCP document provider — exposes MCP resources as ingestible documents."""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
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
from constat.mcp.change_probe import ChangeProbe
from constat.mcp.client import McpClient, McpError

logger = logging.getLogger(__name__)


class McpDocumentProvider:
    """DataSourceProvider that maps MCP server resources to documents.

    Config keys:
        ``url`` — MCP server base URL (required).
        ``resource_filter`` — regex to filter resource URIs (optional).
        ``max_resources`` — cap on resources to list (default 100).
    """

    kind = DataSourceKind.DOCUMENT

    def __init__(self) -> None:
        self._clients: dict[str, McpClient] = {}
        self._configs: dict[str, dict] = {}
        self._probes: dict[str, ChangeProbe] = {}
        self._resources: dict[str, list[dict]] = {}

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
        self._probes[name] = ChangeProbe()

        caps = {"ingestible", "refreshable"}
        return ConnectionResult(success=True, capabilities=caps)

    def disconnect(self, name: str) -> None:
        client = self._clients.pop(name, None)
        if client is not None:
            try:
                asyncio.get_event_loop().run_until_complete(client.disconnect())
            except RuntimeError:
                asyncio.run(client.disconnect())
        self._configs.pop(name, None)
        self._probes.pop(name, None)
        self._resources.pop(name, None)

    def status(self, name: str) -> SourceStatus:
        client = self._clients.get(name)
        if client is None:
            return SourceStatus(state="disconnected")
        if not client.connected:
            return SourceStatus(state="error", error="Client not connected")
        item_count = len(self._resources.get(name, []))
        return SourceStatus(state="connected", item_count=item_count)

    def discover(self, name: str) -> DiscoveryResult:
        resources = self._fetch_resources(name)
        self._resources[name] = resources
        items = [self._resource_to_item(r) for r in resources]
        return DiscoveryResult(items=items)

    def list_items(self, name: str) -> list[SourceItem]:
        resources = self._resources.get(name)
        if resources is None:
            resources = self._fetch_resources(name)
            self._resources[name] = resources
        return [self._resource_to_item(r) for r in resources]

    def fetch_item(self, name: str, item_id: str) -> FetchResult:
        client = self._clients.get(name)
        if client is None:
            raise McpError(-32000, f"Source '{name}' not connected")

        result = self._run_async(client.read_resource(item_id))
        contents = result.get("contents", [])
        if not contents:
            raise McpError(-32000, f"No content returned for {item_id}")

        entry = contents[0]
        mime_type = entry.get("mimeType", "text/plain")
        text = entry.get("text")
        blob = entry.get("blob")
        content = text if text is not None else blob

        if content is None:
            raise McpError(-32000, f"Resource {item_id} has no text or blob content")

        return FetchResult(content=content, mime_type=mime_type, metadata={"uri": item_id})

    def refresh(self, name: str) -> RefreshResult:
        resources = self._fetch_resources(name)
        probe = self._probes.get(name)
        if probe is None:
            probe = ChangeProbe()
            self._probes[name] = probe

        result = probe.probe(resources)
        self._resources[name] = resources
        return RefreshResult(
            added=len(result.added),
            updated=len(result.changed),
            removed=len(result.removed),
        )

    def supports_incremental(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_resources(self, name: str) -> list[dict]:
        """Fetch and filter resources from the MCP server."""
        client = self._clients.get(name)
        if client is None:
            raise McpError(-32000, f"Source '{name}' not connected")

        config = self._configs.get(name, {})
        max_resources = config.get("max_resources", 100)
        resource_filter = config.get("resource_filter")
        pattern = re.compile(resource_filter) if resource_filter else None

        all_resources: list[dict] = []
        cursor: Optional[str] = None

        while len(all_resources) < max_resources:
            result = self._run_async(client.list_resources(cursor=cursor))
            resources = result.get("resources", [])

            for r in resources:
                if pattern and not pattern.search(r.get("uri", "")):
                    continue
                all_resources.append(r)
                if len(all_resources) >= max_resources:
                    break

            cursor = result.get("nextCursor")
            if cursor is None:
                break

        return all_resources

    @staticmethod
    def _resource_to_item(resource: dict) -> SourceItem:
        """Convert an MCP resource dict to a SourceItem."""
        uri = resource.get("uri", "")
        return SourceItem(
            id=uri,
            name=resource.get("name", uri),
            description=resource.get("description", ""),
            mime_type=resource.get("mimeType"),
            size=resource.get("size"),
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
