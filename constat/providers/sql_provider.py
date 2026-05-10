# Copyright (c) 2025 Kenneth Stott
# Canary: aea03763-1d90-4b64-972f-94c8785b1d7d
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""SQL database provider — wraps SchemaManager for the DataSourceProvider contract."""

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


class SqlDatabaseProvider:
    """DataSourceProvider for SQL databases (SQLite, PostgreSQL, MySQL, DuckDB)."""

    kind = DataSourceKind.DATABASE

    def __init__(self) -> None:
        self._schema_managers: dict[str, object] = {}

    def connect(self, name: str, config: dict, auth: Optional[AuthConfig] = None) -> ConnectionResult:
        try:
            from constat.catalog.schema_manager import SchemaManager
            from constat.core.config import Config, DatabaseConfig

            db_config = DatabaseConfig(
                type=config.get("type", "sql"),
                uri=config.get("uri", ""),
                description=config.get("description", ""),
            )
            # SchemaManager needs a full Config — store reference for later
            self._schema_managers[name] = {"config": db_config}
            return ConnectionResult(
                success=True,
                capabilities={"queryable", "schema_introspection"},
            )
        except Exception as e:
            return ConnectionResult(success=False, error=str(e))

    def disconnect(self, name: str) -> None:
        self._schema_managers.pop(name, None)

    def status(self, name: str) -> SourceStatus:
        if name in self._schema_managers:
            return SourceStatus(state="connected")
        return SourceStatus(state="disconnected")

    def discover(self, name: str) -> DiscoveryResult:
        return DiscoveryResult(items=[])

    def list_items(self, name: str) -> list[SourceItem]:
        return []

    def fetch_item(self, name: str, item_id: str) -> FetchResult:
        raise NotImplementedError("SQL databases are queried via DuckDB, not fetched")

    def refresh(self, name: str) -> RefreshResult:
        return RefreshResult()

    def supports_incremental(self) -> bool:
        return False
