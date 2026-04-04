# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""MCP server catalog — three-tier loading: remote, cache, bundled seed."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_SEED_PATH = Path(__file__).parent.parent / "data" / "mcp_catalog_seed.json"


class McpCatalog:
    """Three-tier catalog: remote -> cache -> bundled seed."""

    REMOTE_URL = "https://api.pulsemcp.com/v0/servers"
    CACHE_TTL = 86400  # 24 hours

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._cache_dir = cache_dir or Path(".constat")
        self._cache_file = self._cache_dir / "mcp_catalog.json"

    def list_servers(
        self,
        category: str | None = None,
        query: str | None = None,
    ) -> list[dict[str, Any]]:
        """List servers with optional category/search filter."""
        servers = self._load()
        if category:
            servers = [s for s in servers if s.get("category") == category]
        if query:
            q = query.lower()
            servers = [
                s
                for s in servers
                if q in s.get("name", "").lower()
                or q in s.get("description", "").lower()
            ]
        return servers

    def get_server(self, slug: str) -> dict[str, Any] | None:
        """Get a specific server by slug."""
        for s in self._load():
            if s.get("slug") == slug:
                return s
        return None

    # ------------------------------------------------------------------
    # Internal loading
    # ------------------------------------------------------------------

    def _load(self) -> list[dict[str, Any]]:
        """Load from cache (if fresh) -> remote (with cache write) -> seed."""
        # 1. Check cache freshness
        cached = self._load_cache()
        if cached is not None:
            return cached

        # 2. Try remote fetch
        remote = self._fetch_remote()
        if remote is not None:
            self._write_cache(remote)
            return remote

        # 3. Try stale cache before seed
        stale = self._load_cache(ignore_ttl=True)
        if stale is not None:
            return stale

        # 4. Fall back to bundled seed
        return self._load_seed()

    def _load_cache(self, ignore_ttl: bool = False) -> list[dict[str, Any]] | None:
        """Load from cache file. Returns None if missing or stale."""
        if not self._cache_file.exists():
            return None
        try:
            data = json.loads(self._cache_file.read_text())
            if not ignore_ttl:
                cached_at = data.get("cached_at", 0)
                if time.time() - cached_at > self.CACHE_TTL:
                    return None
            return data.get("servers", [])
        except (json.JSONDecodeError, KeyError, OSError):
            return None

    def _write_cache(self, servers: list[dict[str, Any]]) -> None:
        """Write servers to cache file."""
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            payload = {"cached_at": time.time(), "servers": servers}
            self._cache_file.write_text(json.dumps(payload))
        except OSError:
            logger.warning("Failed to write MCP catalog cache")

    def _fetch_remote(self) -> list[dict[str, Any]] | None:
        """Fetch from PulseMCP API. Returns None on failure."""
        try:
            resp = httpx.get(self.REMOTE_URL, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            # PulseMCP returns {"servers": [...]} or a list directly
            if isinstance(data, list):
                return data
            return data.get("servers", [])
        except Exception:
            logger.warning("Failed to fetch remote MCP catalog")
            return None

    def _load_seed(self) -> list[dict[str, Any]]:
        """Load bundled seed catalog."""
        try:
            data = json.loads(_SEED_PATH.read_text())
            return data.get("servers", [])
        except (json.JSONDecodeError, OSError):
            logger.error("Failed to load MCP seed catalog")
            return []
