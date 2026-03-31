# Copyright (c) 2025 Kenneth Stott
# Canary: 2de679a3-2442-4cc2-86c8-6253c009de93
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""API source providers — wraps existing API integrations for the DataSourceProvider contract."""

from __future__ import annotations

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

logger = logging.getLogger(__name__)


class GraphQLApiProvider:
    """DataSourceProvider for GraphQL API sources."""

    kind = DataSourceKind.API

    def __init__(self) -> None:
        self._sources: dict[str, dict] = {}

    def connect(self, name: str, config: dict, auth: Optional[AuthConfig] = None) -> ConnectionResult:
        url = config.get("url", "")
        if not url:
            return ConnectionResult(success=False, error="GraphQL URL is required")
        self._sources[name] = config
        return ConnectionResult(success=True, capabilities={"queryable"})

    def disconnect(self, name: str) -> None:
        self._sources.pop(name, None)

    def status(self, name: str) -> SourceStatus:
        if name in self._sources:
            return SourceStatus(state="connected")
        return SourceStatus(state="disconnected")

    def discover(self, name: str) -> DiscoveryResult:
        return DiscoveryResult(items=[], operations=[])

    def list_items(self, name: str) -> list[SourceItem]:
        return []

    def fetch_item(self, name: str, item_id: str) -> FetchResult:
        raise NotImplementedError("GraphQL operations executed via query engine")

    def refresh(self, name: str) -> RefreshResult:
        return RefreshResult()

    def supports_incremental(self) -> bool:
        return False


class OpenApiProvider:
    """DataSourceProvider for OpenAPI/REST API sources."""

    kind = DataSourceKind.API

    def __init__(self) -> None:
        self._sources: dict[str, dict] = {}

    def connect(self, name: str, config: dict, auth: Optional[AuthConfig] = None) -> ConnectionResult:
        base_url = config.get("base_url", "") or config.get("url", "")
        if not base_url:
            return ConnectionResult(success=False, error="API base_url is required")
        self._sources[name] = config
        return ConnectionResult(success=True, capabilities={"queryable"})

    def disconnect(self, name: str) -> None:
        self._sources.pop(name, None)

    def status(self, name: str) -> SourceStatus:
        if name in self._sources:
            return SourceStatus(state="connected")
        return SourceStatus(state="disconnected")

    def discover(self, name: str) -> DiscoveryResult:
        return DiscoveryResult(items=[], operations=[])

    def list_items(self, name: str) -> list[SourceItem]:
        return []

    def fetch_item(self, name: str, item_id: str) -> FetchResult:
        raise NotImplementedError("REST operations executed via query engine")

    def refresh(self, name: str) -> RefreshResult:
        return RefreshResult()

    def supports_incremental(self) -> bool:
        return False
