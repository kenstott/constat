# Copyright (c) 2025 Kenneth Stott
# Canary: 620e1215-1bc1-418f-849e-76ff728fad2c
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Unit tests for GraphQL source query resolvers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# user_sources module has a broken import (user_vault_dir), fix it for testing
import constat.server.config as _sc
if not hasattr(_sc, "user_vault_dir"):
    from constat.core.paths import user_vault_dir as _uvd
    _sc.user_vault_dir = _uvd
import constat.server.routes.user_sources  # noqa: F401


def _make_context(user_id="test-user"):
    from constat.server.graphql.session_context import GraphQLContext

    mock_sm = MagicMock()
    mock_server_config = MagicMock()
    mock_server_config.auth_disabled = True
    mock_server_config.base_url = "http://localhost:3000"
    mock_server_config.data_dir = Path("/tmp/test-graphql-sources")

    ctx = GraphQLContext(
        session_manager=mock_sm,
        server_config=mock_server_config,
        user_id=user_id,
    )
    mock_request = MagicMock()
    ctx.request = mock_request
    return ctx


def _make_managed(session_id="s1", user_id="test-user"):
    from constat.server.models import SessionStatus

    managed = MagicMock()
    managed.session_id = session_id
    managed.user_id = user_id
    managed.status = SessionStatus("idle")
    managed.created_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
    managed.last_activity = datetime(2025, 1, 1, tzinfo=timezone.utc)
    managed.active_domains = []
    managed.session.session_id = "hist-s1"
    managed.session.config = MagicMock()
    managed.session.config.databases = {}
    managed.session.config.apis = {}
    managed.session.config.documents = {}
    managed.session.config.load_domain.return_value = None
    managed.session.config.data_dir = None
    managed.session.schema_manager = None
    managed.session.doc_tools = None
    managed.session.api_schema_manager = None
    managed.session.datastore = MagicMock()
    managed._dynamic_dbs = []
    managed._dynamic_apis = []
    managed._file_refs = []
    managed.resolved_config = None
    return managed


# ============================================================================
# Query resolver tests
# ============================================================================


class TestUploadedFilesQuery:
    @pytest.mark.asyncio
    async def test_uploaded_files_empty(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        with patch("constat.server.routes.files._get_uploaded_files_for_session", return_value=[]):
            result = await schema.execute(
                '{ files(sessionId: "s1") { files { id filename } } }',
                context_value=ctx,
            )
        assert result.errors is None
        assert result.data["files"]["files"] == []

    @pytest.mark.asyncio
    async def test_uploaded_files_with_data(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        files = [{
            "id": "f_abc123",
            "filename": "data.csv",
            "file_uri": "file:///tmp/data.csv",
            "size_bytes": 1024,
            "content_type": "text/csv",
            "uploaded_at": "2025-01-01T00:00:00+00:00",
        }]

        with patch("constat.server.routes.files._get_uploaded_files_for_session", return_value=files):
            result = await schema.execute(
                '{ files(sessionId: "s1") { files { id filename fileUri sizeBytes contentType } } }',
                context_value=ctx,
            )
        assert result.errors is None
        data = result.data["files"]["files"]
        assert len(data) == 1
        assert data[0]["id"] == "f_abc123"
        assert data[0]["filename"] == "data.csv"
        assert data[0]["sizeBytes"] == 1024


class TestFileRefsQuery:
    @pytest.mark.asyncio
    async def test_file_refs_empty(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            '{ fileRefs(sessionId: "s1") { fileRefs { name uri } } }',
            context_value=ctx,
        )
        assert result.errors is None
        assert result.data["fileRefs"]["fileRefs"] == []

    @pytest.mark.asyncio
    async def test_file_refs_with_data(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed._file_refs = [{
            "name": "manual",
            "uri": "https://example.com/doc.pdf",
            "has_auth": False,
            "description": "User manual",
            "added_at": "2025-01-01T00:00:00+00:00",
        }]
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            '{ fileRefs(sessionId: "s1") { fileRefs { name uri hasAuth description } } }',
            context_value=ctx,
        )
        assert result.errors is None
        refs = result.data["fileRefs"]["fileRefs"]
        assert len(refs) == 1
        assert refs[0]["name"] == "manual"
        assert refs[0]["hasAuth"] is False


class TestDatabasesQuery:
    @pytest.mark.asyncio
    async def test_databases_empty(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            '{ databases(sessionId: "s1") { databases { name type connected } } }',
            context_value=ctx,
        )
        assert result.errors is None
        assert result.data["databases"]["databases"] == []

    @pytest.mark.asyncio
    async def test_databases_config(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        db_config = MagicMock()
        db_config.type = "sqlite"
        db_config.description = "Test DB"
        managed.session.config.databases = {"testdb": db_config}
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            '{ databases(sessionId: "s1") { databases { name type connected isDynamic source } } }',
            context_value=ctx,
        )
        assert result.errors is None
        dbs = result.data["databases"]["databases"]
        assert len(dbs) == 1
        assert dbs[0]["name"] == "testdb"
        assert dbs[0]["type"] == "sqlite"
        assert dbs[0]["isDynamic"] is False
        assert dbs[0]["source"] == "config"

    @pytest.mark.asyncio
    async def test_databases_dynamic(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed._dynamic_dbs = [{
            "name": "upload_csv",
            "type": "csv",
            "dialect": "duckdb",
            "description": "Uploaded CSV",
            "connected": True,
            "table_count": 1,
            "added_at": "2025-01-01T00:00:00+00:00",
            "is_dynamic": True,
            "file_id": "f_123",
        }]
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            '{ databases(sessionId: "s1") { databases { name type connected isDynamic fileId } } }',
            context_value=ctx,
        )
        assert result.errors is None
        dbs = result.data["databases"]["databases"]
        assert len(dbs) == 1
        assert dbs[0]["name"] == "upload_csv"
        assert dbs[0]["isDynamic"] is True
        assert dbs[0]["fileId"] == "f_123"

    @pytest.mark.asyncio
    async def test_databases_dedup(self):
        """Dynamic db with same name as config db should not appear."""
        from constat.server.graphql import schema

        managed = _make_managed()
        db_config = MagicMock()
        db_config.type = "sqlite"
        db_config.description = "Config DB"
        managed.session.config.databases = {"mydb": db_config}
        managed._dynamic_dbs = [{
            "name": "mydb",
            "type": "csv",
            "dialect": "duckdb",
            "description": "Duplicate",
            "connected": True,
            "table_count": 1,
            "added_at": "2025-01-01T00:00:00+00:00",
            "is_dynamic": True,
        }]
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            '{ databases(sessionId: "s1") { databases { name isDynamic } } }',
            context_value=ctx,
        )
        assert result.errors is None
        dbs = result.data["databases"]["databases"]
        assert len(dbs) == 1
        assert dbs[0]["isDynamic"] is False


class TestDataSourcesQuery:
    @pytest.mark.asyncio
    async def test_data_sources_combined(self):
        from constat.server.graphql import schema

        managed = _make_managed()

        # Config db
        db_config = MagicMock()
        db_config.type = "sqlite"
        db_config.description = "DB"
        managed.session.config.databases = {"db1": db_config}

        # Config API
        api_config = MagicMock()
        api_config.type = "rest"
        api_config.description = "API"
        api_config.url = "https://api.example.com"
        managed.session.config.apis = {"api1": api_config}

        # Config document
        doc_config = MagicMock()
        doc_config.type = "pdf"
        doc_config.description = "Doc"
        doc_config.path = "/docs/manual.pdf"
        managed.session.config.documents = {"doc1": doc_config}

        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            '{ dataSources(sessionId: "s1") { databases { name } apis { name } documents { name } } }',
            context_value=ctx,
        )
        assert result.errors is None
        data = result.data["dataSources"]
        assert len(data["databases"]) == 1
        assert len(data["apis"]) == 1
        assert len(data["documents"]) == 1

    @pytest.mark.asyncio
    async def test_session_not_found(self):
        from constat.server.graphql import schema

        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = None

        result = await schema.execute(
            '{ dataSources(sessionId: "nope") { databases { name } } }',
            context_value=ctx,
        )
        assert result.errors is not None


class TestDocumentQuery:
    @pytest.mark.asyncio
    async def test_document_success(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed.session.doc_tools = MagicMock()
        managed.session.doc_tools.get_document.return_value = {
            "content": "Hello world",
            "metadata": {"pages": 3},
        }
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            '{ document(sessionId: "s1", name: "manual") { name content metadata } }',
            context_value=ctx,
        )
        assert result.errors is None
        data = result.data["document"]
        assert data["name"] == "manual"
        assert data["content"] == "Hello world"

    @pytest.mark.asyncio
    async def test_document_not_found(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed.session.doc_tools = MagicMock()
        managed.session.doc_tools.get_document.return_value = {"error": "not found"}
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            '{ document(sessionId: "s1", name: "nope") { name } }',
            context_value=ctx,
        )
        assert result.errors is not None

    @pytest.mark.asyncio
    async def test_document_no_doc_tools(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed.session.doc_tools = None
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            '{ document(sessionId: "s1", name: "nope") { name } }',
            context_value=ctx,
        )
        assert result.errors is not None


class TestUserSourcesQuery:
    @pytest.mark.asyncio
    async def test_user_sources(self):
        from constat.server.graphql import schema

        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = _make_managed()

        user_config = {
            "databases": {"mydb": {"source": "user", "type": "sqlite"}},
            "documents": {},
            "apis": {"myapi": {"source": "user", "type": "rest"}},
        }

        with patch("constat.server.routes.user_sources._load_user_config", return_value=user_config):
            result = await schema.execute(
                '{ userSources { databases documents apis } }',
                context_value=ctx,
            )
        assert result.errors is None
        data = result.data["userSources"]
        assert "mydb" in str(data["databases"])
        assert "myapi" in str(data["apis"])


class TestSecurityHardeningQueries:
    """Security hardening tests for query resolvers."""

    @pytest.mark.asyncio
    async def test_database_table_preview_rejects_without_schema_manager(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed.session.schema_manager = None
        managed.has_database = MagicMock(return_value=True)
        managed.get_database_connection = MagicMock(return_value=MagicMock())
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            """query {
                databaseTablePreview(sessionId: "s1", dbName: "mydb", tableName: "users") {
                    database
                }
            }""",
            context_value=ctx,
        )
        assert result.errors is not None
        assert "Schema manager not available" in str(result.errors[0])

    @pytest.mark.asyncio
    async def test_database_table_preview_rejects_unknown_table(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        mock_sm = MagicMock()
        mock_sm.metadata_cache = {"mydb.known_table": MagicMock()}
        managed.session.schema_manager = mock_sm
        managed.has_database = MagicMock(return_value=True)
        managed.get_database_connection = MagicMock(return_value=MagicMock())
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            """query {
                databaseTablePreview(sessionId: "s1", dbName: "mydb", tableName: "'; DROP TABLE users; --") {
                    database
                }
            }""",
            context_value=ctx,
        )
        assert result.errors is not None
        assert "Table not found" in str(result.errors[0])
