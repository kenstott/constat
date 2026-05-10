# Copyright (c) 2025 Kenneth Stott
# Canary: 2b281a19-36a6-48c7-b937-5e3b50ddb012
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Tests for provider wrappers (sql, document, api)."""

from __future__ import annotations
import tempfile
from pathlib import Path

import pytest

from constat.core.sources import DataSourceKind
from constat.providers.sql_provider import SqlDatabaseProvider
from constat.providers.document_providers import (
    FileDocumentProvider,
    HttpDocumentProvider,
    ImapDocumentProvider,
)
from constat.providers.api_providers import (
    GraphQLApiProvider,
    OpenApiProvider,
)


class TestSqlDatabaseProvider:

    def test_connect_success(self):
        p = SqlDatabaseProvider()
        result = p.connect("mydb", {"type": "sql", "uri": "sqlite://"})
        assert result.success
        assert "queryable" in result.capabilities

    def test_disconnect(self):
        p = SqlDatabaseProvider()
        p.connect("mydb", {"uri": "sqlite://"})
        assert p.status("mydb").state == "connected"
        p.disconnect("mydb")
        assert p.status("mydb").state == "disconnected"

    def test_kind(self):
        assert SqlDatabaseProvider.kind == DataSourceKind.DATABASE


class TestFileDocumentProvider:

    def test_connect_with_existing_path(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        p = FileDocumentProvider()
        result = p.connect("doc", {"path": str(f)})
        assert result.success
        assert "ingestible" in result.capabilities

    def test_connect_missing_path(self, tmp_path):
        p = FileDocumentProvider()
        result = p.connect("doc", {"path": str(tmp_path / "nonexistent.txt")})
        assert not result.success
        assert "not found" in result.error.lower()

    def test_connect_glob_path(self):
        p = FileDocumentProvider()
        result = p.connect("docs", {"path": "/tmp/*.txt"})
        assert result.success

    def test_discover_single_file(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("a,b")
        p = FileDocumentProvider()
        p.connect("doc", {"path": str(f)})
        items = p.discover("doc").items
        assert len(items) == 1
        assert items[0].name == "data.csv"

    def test_fetch_item(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("content here")
        p = FileDocumentProvider()
        p.connect("doc", {"path": str(f)})
        result = p.fetch_item("doc", str(f))
        assert result.content == "content here"

    def test_fetch_item_not_found(self):
        p = FileDocumentProvider()
        with pytest.raises(FileNotFoundError):
            p.fetch_item("doc", "/nonexistent/file.txt")

    def test_kind(self):
        assert FileDocumentProvider.kind == DataSourceKind.DOCUMENT


class TestHttpDocumentProvider:

    def test_connect_success(self):
        p = HttpDocumentProvider()
        result = p.connect("web", {"url": "https://example.com/doc.pdf"})
        assert result.success
        assert "ingestible" in result.capabilities
        assert "refreshable" in result.capabilities

    def test_connect_no_url(self):
        p = HttpDocumentProvider()
        result = p.connect("web", {})
        assert not result.success

    def test_discover(self):
        p = HttpDocumentProvider()
        p.connect("web", {"url": "https://example.com"})
        items = p.discover("web").items
        assert len(items) == 1
        assert items[0].id == "https://example.com"

    def test_kind(self):
        assert HttpDocumentProvider.kind == DataSourceKind.DOCUMENT


class TestImapDocumentProvider:

    def test_connect_success(self):
        p = ImapDocumentProvider()
        result = p.connect("mail", {"url": "imaps://imap.gmail.com:993"})
        assert result.success
        assert "refreshable" in result.capabilities

    def test_connect_no_url(self):
        p = ImapDocumentProvider()
        result = p.connect("mail", {})
        assert not result.success

    def test_supports_incremental(self):
        assert ImapDocumentProvider().supports_incremental() is True

    def test_kind(self):
        assert ImapDocumentProvider.kind == DataSourceKind.DOCUMENT


class TestGraphQLApiProvider:

    def test_connect_success(self):
        p = GraphQLApiProvider()
        result = p.connect("gql", {"url": "https://api.example.com/graphql"})
        assert result.success
        assert "queryable" in result.capabilities

    def test_connect_no_url(self):
        p = GraphQLApiProvider()
        result = p.connect("gql", {})
        assert not result.success

    def test_kind(self):
        assert GraphQLApiProvider.kind == DataSourceKind.API


class TestOpenApiProvider:

    def test_connect_with_base_url(self):
        p = OpenApiProvider()
        result = p.connect("rest", {"base_url": "https://api.example.com/v1"})
        assert result.success

    def test_connect_with_url(self):
        p = OpenApiProvider()
        result = p.connect("rest", {"url": "https://api.example.com/v1"})
        assert result.success

    def test_connect_no_url(self):
        p = OpenApiProvider()
        result = p.connect("rest", {})
        assert not result.success

    def test_status_lifecycle(self):
        p = OpenApiProvider()
        assert p.status("rest").state == "disconnected"
        p.connect("rest", {"base_url": "https://api.example.com"})
        assert p.status("rest").state == "connected"
        p.disconnect("rest")
        assert p.status("rest").state == "disconnected"

    def test_kind(self):
        assert OpenApiProvider.kind == DataSourceKind.API
