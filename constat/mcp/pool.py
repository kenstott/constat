# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Connection pool for MCP clients — same URL shares one McpClient instance."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from constat.core.sources import AuthConfig
from constat.mcp.client import McpClient

logger = logging.getLogger(__name__)


@dataclass
class _PoolEntry:
    client: McpClient
    refcount: int = 1


class McpClientPool:
    """Refcounted connection pool for MCP clients.

    When multiple data sources point at the same MCP server URL, they share
    a single ``McpClient`` instance. The client is released (disconnected)
    only when the last reference is released.
    """

    def __init__(self) -> None:
        self._entries: dict[str, _PoolEntry] = {}

    def acquire(self, url: str, auth: Optional[AuthConfig] = None) -> McpClient:
        """Get or create a client for *url*.

        If a client already exists for this URL, its reference count is
        incremented and the existing instance is returned.
        """
        key = url.rstrip("/")
        if key in self._entries:
            self._entries[key].refcount += 1
            logger.debug("Pool reuse %s (refcount=%d)", key, self._entries[key].refcount)
            return self._entries[key].client

        client = McpClient(url, auth=auth)
        self._entries[key] = _PoolEntry(client=client)
        logger.debug("Pool new %s", key)
        return client

    def release(self, url: str) -> None:
        """Decrement reference count for *url*.

        When the count reaches zero the entry is removed. The caller is
        responsible for calling ``client.disconnect()`` before releasing
        the last reference.
        """
        key = url.rstrip("/")
        entry = self._entries.get(key)
        if entry is None:
            return

        entry.refcount -= 1
        logger.debug("Pool release %s (refcount=%d)", key, entry.refcount)
        if entry.refcount <= 0:
            del self._entries[key]

    @property
    def size(self) -> int:
        """Number of distinct URLs in the pool."""
        return len(self._entries)
