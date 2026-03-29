# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Document source providers — wraps existing doc_tools for the DataSourceProvider contract."""

from __future__ import annotations

import logging
from pathlib import Path
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

logger = logging.getLogger(__name__)


class FileDocumentProvider:
    """DataSourceProvider for local file documents (single files, globs, directories)."""

    kind = DataSourceKind.DOCUMENT

    def __init__(self) -> None:
        self._sources: dict[str, dict] = {}

    def connect(self, name: str, config: dict, auth: Optional[AuthConfig] = None) -> ConnectionResult:
        path = config.get("path", "")
        if path and not Path(path).exists() and "*" not in path:
            return ConnectionResult(success=False, error=f"Path not found: {path}")
        self._sources[name] = config
        return ConnectionResult(success=True, capabilities={"ingestible"})

    def disconnect(self, name: str) -> None:
        self._sources.pop(name, None)

    def status(self, name: str) -> SourceStatus:
        if name in self._sources:
            return SourceStatus(state="connected")
        return SourceStatus(state="disconnected")

    def discover(self, name: str) -> DiscoveryResult:
        config = self._sources.get(name, {})
        path = config.get("path", "")
        items = []
        if path and "*" in path:
            import glob
            for p in glob.glob(path):
                items.append(SourceItem(id=p, name=Path(p).name, viewable=True))
        elif path:
            items.append(SourceItem(id=path, name=Path(path).name, viewable=True))
        return DiscoveryResult(items=items)

    def list_items(self, name: str) -> list[SourceItem]:
        return self.discover(name).items

    def fetch_item(self, name: str, item_id: str) -> FetchResult:
        p = Path(item_id)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {item_id}")
        return FetchResult(content=p.read_text(), mime_type="text/plain")

    def refresh(self, name: str) -> RefreshResult:
        return RefreshResult()

    def supports_incremental(self) -> bool:
        return False


class HttpDocumentProvider:
    """DataSourceProvider for HTTP/HTTPS document sources."""

    kind = DataSourceKind.DOCUMENT

    def __init__(self) -> None:
        self._sources: dict[str, dict] = {}

    def connect(self, name: str, config: dict, auth: Optional[AuthConfig] = None) -> ConnectionResult:
        url = config.get("url", "")
        if not url:
            return ConnectionResult(success=False, error="URL is required")
        self._sources[name] = config
        return ConnectionResult(success=True, capabilities={"ingestible", "refreshable"})

    def disconnect(self, name: str) -> None:
        self._sources.pop(name, None)

    def status(self, name: str) -> SourceStatus:
        if name in self._sources:
            return SourceStatus(state="connected")
        return SourceStatus(state="disconnected")

    def discover(self, name: str) -> DiscoveryResult:
        config = self._sources.get(name, {})
        items = [SourceItem(id=config.get("url", ""), name=name, viewable=True)]
        return DiscoveryResult(items=items)

    def list_items(self, name: str) -> list[SourceItem]:
        return self.discover(name).items

    def fetch_item(self, name: str, item_id: str) -> FetchResult:
        raise NotImplementedError("HTTP fetching delegated to existing HTTPFetcher")

    def refresh(self, name: str) -> RefreshResult:
        return RefreshResult()

    def supports_incremental(self) -> bool:
        return False


class ImapDocumentProvider:
    """DataSourceProvider for IMAP email sources."""

    kind = DataSourceKind.DOCUMENT

    def __init__(self) -> None:
        self._sources: dict[str, dict] = {}

    def connect(self, name: str, config: dict, auth: Optional[AuthConfig] = None) -> ConnectionResult:
        url = config.get("url", "")
        if not url:
            return ConnectionResult(success=False, error="IMAP URL is required")
        self._sources[name] = config
        return ConnectionResult(success=True, capabilities={"ingestible", "refreshable"})

    def disconnect(self, name: str) -> None:
        self._sources.pop(name, None)

    def status(self, name: str) -> SourceStatus:
        if name in self._sources:
            return SourceStatus(state="connected")
        return SourceStatus(state="disconnected")

    def discover(self, name: str) -> DiscoveryResult:
        return DiscoveryResult(items=[SourceItem(id=name, name=name, viewable=False)])

    def list_items(self, name: str) -> list[SourceItem]:
        return []

    def fetch_item(self, name: str, item_id: str) -> FetchResult:
        raise NotImplementedError("IMAP fetching delegated to existing IMAP fetcher")

    def refresh(self, name: str) -> RefreshResult:
        return RefreshResult()

    def supports_incremental(self) -> bool:
        return True
