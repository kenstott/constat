# Copyright (c) 2025 Kenneth Stott
# Canary: a1b2e8b2-1483-4438-92c8-50ef44de2be7
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Unified data source contract — types, registry, and provider protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional, Protocol, runtime_checkable

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DataSourceKind(str, Enum):
    DATABASE = "database"
    DOCUMENT = "document"
    API = "api"


class ConfigSource(str, Enum):
    SYSTEM = "system"
    SYSTEM_DOMAIN = "system_domain"
    USER = "user"
    USER_DOMAIN = "user_domain"
    SESSION = "session"


# ---------------------------------------------------------------------------
# Supporting dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ConnectionResult:
    success: bool
    error: Optional[str] = None
    capabilities: set[str] = field(default_factory=set)


@dataclass
class SourceStatus:
    state: Literal["connected", "disconnected", "error", "stale"]
    error: Optional[str] = None
    item_count: Optional[int] = None
    last_refreshed: Optional[datetime] = None


@dataclass
class SourceItem:
    id: str
    name: str
    description: str = ""
    mime_type: Optional[str] = None
    size: Optional[int] = None
    last_modified: Optional[datetime] = None
    viewable: bool = True
    children: Optional[list[str]] = None


@dataclass
class DiscoveryResult:
    items: list[SourceItem]
    schema: Optional[dict] = None
    operations: Optional[list[dict]] = None


@dataclass
class FetchResult:
    content: str | bytes
    mime_type: str = "text/plain"
    metadata: dict = field(default_factory=dict)


@dataclass
class RefreshResult:
    added: int = 0
    updated: int = 0
    removed: int = 0
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Auth config
# ---------------------------------------------------------------------------

class AuthConfig(BaseModel):
    method: Literal["none", "basic", "bearer", "api_key", "oauth2", "ntlm"] = "none"

    # basic / ntlm
    username: Optional[str] = None
    password: Optional[str] = None
    domain: Optional[str] = None

    # bearer
    token: Optional[str] = None

    # api_key
    api_key: Optional[str] = None
    api_key_header: str = "X-API-Key"

    # oauth2
    oauth2_provider: Optional[str] = None
    oauth2_scopes: list[str] = []
    oauth2_tenant_id: Optional[str] = None
    token_ref: Optional[str] = None


# ---------------------------------------------------------------------------
# Unified response model
# ---------------------------------------------------------------------------

class DataSourceInfo(BaseModel):
    name: str
    kind: DataSourceKind
    type: str
    description: Optional[str] = None
    state: str = "connected"
    error: Optional[str] = None

    # Capability flags
    queryable: bool = False
    ingestible: bool = False
    refreshable: bool = False
    viewable: bool = False

    # Counts
    item_count: Optional[int] = None
    indexed_count: Optional[int] = None

    # Provenance
    source: str = "config"
    tier: Optional[str] = None
    is_dynamic: bool = False
    scope: Optional[str] = None

    # Type-specific metadata (flat)
    uri: Optional[str] = None
    base_url: Optional[str] = None
    dialect: Optional[str] = None
    path: Optional[str] = None


# ---------------------------------------------------------------------------
# Provider protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class DataSourceProvider(Protocol):
    """Standard contract for all data source types."""

    kind: DataSourceKind

    def connect(self, name: str, config: dict, auth: Optional[AuthConfig] = None) -> ConnectionResult: ...

    def disconnect(self, name: str) -> None: ...

    def status(self, name: str) -> SourceStatus: ...

    def discover(self, name: str) -> DiscoveryResult: ...

    def list_items(self, name: str) -> list[SourceItem]: ...

    def fetch_item(self, name: str, item_id: str) -> FetchResult: ...

    def refresh(self, name: str) -> RefreshResult: ...

    def supports_incremental(self) -> bool: ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class DataSourceRegistry:
    """Central dispatcher — maps (kind, type) to provider implementations."""

    def __init__(self) -> None:
        self._providers: dict[tuple[DataSourceKind, str], DataSourceProvider] = {}
        self._sources: dict[str, tuple[DataSourceKind, str]] = {}

    def register(self, kind: DataSourceKind, source_type: str, provider: DataSourceProvider) -> None:
        self._providers[(kind, source_type)] = provider

    def get_provider(self, kind: DataSourceKind, source_type: str) -> DataSourceProvider:
        key = (kind, source_type)
        if key not in self._providers:
            raise KeyError(f"No provider for {kind.value}:{source_type}")
        return self._providers[key]

    def has_provider(self, kind: DataSourceKind, source_type: str) -> bool:
        return (kind, source_type) in self._providers

    def add_source(
        self, name: str, kind: DataSourceKind, source_type: str,
        config: dict, auth: Optional[AuthConfig] = None,
    ) -> ConnectionResult:
        provider = self.get_provider(kind, source_type)
        result = provider.connect(name, config, auth)
        if result.success:
            self._sources[name] = (kind, source_type)
        return result

    def remove_source(self, name: str, kind: DataSourceKind, source_type: str) -> None:
        provider = self.get_provider(kind, source_type)
        provider.disconnect(name)
        self._sources.pop(name, None)

    def list_all(self) -> list[DataSourceInfo]:
        results: list[DataSourceInfo] = []
        for name, (kind, source_type) in self._sources.items():
            provider = self._providers.get((kind, source_type))
            if not provider:
                continue
            st = provider.status(name)
            results.append(DataSourceInfo(
                name=name,
                kind=kind,
                type=source_type,
                state=st.state,
                error=st.error,
                item_count=st.item_count,
            ))
        return results
