# Copyright (c) 2025 Kenneth Stott
# Canary: 3db06a77-93f9-497f-85c6-98c496c4fe9b
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Tests for constat.core.sources — registry, types, auth."""

import pytest

from constat.core.sources import (
    AuthConfig,
    ConfigSource,
    ConnectionResult,
    DataSourceInfo,
    DataSourceKind,
    DataSourceProvider,
    DataSourceRegistry,
    DiscoveryResult,
    FetchResult,
    RefreshResult,
    SourceItem,
    SourceStatus,
)


class StubProvider:
    """Minimal provider for testing the registry."""

    kind = DataSourceKind.DATABASE

    def __init__(self, connect_success: bool = True):
        self._connect_success = connect_success
        self._connected: dict[str, dict] = {}

    def connect(self, name, config, auth=None):
        if self._connect_success:
            self._connected[name] = config
            return ConnectionResult(success=True, capabilities={"query"})
        return ConnectionResult(success=False, error="connection refused")

    def disconnect(self, name):
        self._connected.pop(name, None)

    def status(self, name):
        if name in self._connected:
            return SourceStatus(state="connected", item_count=0)
        return SourceStatus(state="disconnected")

    def discover(self, name):
        return DiscoveryResult(items=[])

    def list_items(self, name):
        return []

    def fetch_item(self, name, item_id):
        return FetchResult(content="", mime_type="text/plain")

    def refresh(self, name):
        return RefreshResult()

    def supports_incremental(self):
        return False


class TestDataSourceRegistry:

    def test_register_and_get(self):
        reg = DataSourceRegistry()
        provider = StubProvider()
        reg.register(DataSourceKind.DATABASE, "sql", provider)
        assert reg.get_provider(DataSourceKind.DATABASE, "sql") is provider

    def test_missing_provider(self):
        reg = DataSourceRegistry()
        with pytest.raises(KeyError, match="No provider for database:nosql"):
            reg.get_provider(DataSourceKind.DATABASE, "nosql")

    def test_has_provider(self):
        reg = DataSourceRegistry()
        assert not reg.has_provider(DataSourceKind.DATABASE, "sql")
        reg.register(DataSourceKind.DATABASE, "sql", StubProvider())
        assert reg.has_provider(DataSourceKind.DATABASE, "sql")

    def test_add_source_success(self):
        reg = DataSourceRegistry()
        reg.register(DataSourceKind.DATABASE, "sql", StubProvider())
        result = reg.add_source("mydb", DataSourceKind.DATABASE, "sql", {"uri": "sqlite://"})
        assert result.success
        assert "query" in result.capabilities

    def test_add_source_failure(self):
        reg = DataSourceRegistry()
        reg.register(DataSourceKind.DATABASE, "sql", StubProvider(connect_success=False))
        result = reg.add_source("mydb", DataSourceKind.DATABASE, "sql", {})
        assert not result.success
        assert result.error == "connection refused"

    def test_remove_source(self):
        reg = DataSourceRegistry()
        provider = StubProvider()
        reg.register(DataSourceKind.DATABASE, "sql", provider)
        reg.add_source("mydb", DataSourceKind.DATABASE, "sql", {"uri": "x"})
        assert "mydb" in provider._connected
        reg.remove_source("mydb", DataSourceKind.DATABASE, "sql")
        assert "mydb" not in provider._connected

    def test_list_all(self):
        reg = DataSourceRegistry()
        reg.register(DataSourceKind.DATABASE, "sql", StubProvider())
        reg.add_source("db1", DataSourceKind.DATABASE, "sql", {})
        reg.add_source("db2", DataSourceKind.DATABASE, "sql", {})
        items = reg.list_all()
        assert len(items) == 2
        names = {i.name for i in items}
        assert names == {"db1", "db2"}


class TestDataSourceInfo:

    def test_creation_all_fields(self):
        info = DataSourceInfo(
            name="sales",
            kind=DataSourceKind.DATABASE,
            type="sql",
            description="Sales DB",
            state="connected",
            queryable=True,
            ingestible=False,
            refreshable=False,
            viewable=True,
            item_count=42,
            indexed_count=10,
            source="config",
            tier="system",
            is_dynamic=False,
            scope="user",
            uri="postgresql://host/db",
            dialect="postgresql",
        )
        assert info.name == "sales"
        assert info.kind == DataSourceKind.DATABASE
        assert info.queryable is True
        assert info.viewable is True
        assert info.scope == "user"
        assert info.uri == "postgresql://host/db"

    def test_defaults(self):
        info = DataSourceInfo(name="x", kind=DataSourceKind.DOCUMENT, type="file")
        assert info.state == "connected"
        assert info.queryable is False
        assert info.viewable is False
        assert info.scope is None


class TestAuthConfig:

    def test_none_auth(self):
        auth = AuthConfig(method="none")
        assert auth.method == "none"

    def test_basic_auth(self):
        auth = AuthConfig(method="basic", username="user", password="pass")
        assert auth.username == "user"
        assert auth.password == "pass"

    def test_bearer_auth(self):
        auth = AuthConfig(method="bearer", token="tok123")
        assert auth.token == "tok123"

    def test_api_key_auth(self):
        auth = AuthConfig(method="api_key", api_key="key", api_key_header="Authorization")
        assert auth.api_key == "key"
        assert auth.api_key_header == "Authorization"

    def test_oauth2_auth(self):
        auth = AuthConfig(
            method="oauth2",
            oauth2_provider="google",
            oauth2_scopes=["mail.read"],
            token_ref="my-gmail",
        )
        assert auth.oauth2_provider == "google"
        assert auth.token_ref == "my-gmail"

    def test_ntlm_auth(self):
        auth = AuthConfig(method="ntlm", username="u", password="p", domain="CORP")
        assert auth.domain == "CORP"


class TestConfigSource:

    def test_all_tiers(self):
        expected = {"system", "system_domain", "user", "user_domain", "session"}
        actual = {cs.value for cs in ConfigSource}
        assert actual == expected
