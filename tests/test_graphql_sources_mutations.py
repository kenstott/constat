# Copyright (c) 2025 Kenneth Stott
# Canary: 620e1215-1bc1-418f-849e-76ff728fad2c
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Unit tests for GraphQL source mutation resolvers."""

from __future__ import annotations

import base64
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
# Mutation resolver tests
# ============================================================================


class TestDatabaseMutations:
    @pytest.mark.asyncio
    async def test_add_database(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed.session.add_database = MagicMock()
        managed.session.session_databases = {}
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            """mutation {
                addDatabase(sessionId: "s1", input: {name: "mydb", uri: "sqlite:///test.db", type: "sqlalchemy"}) {
                    name type connected isDynamic
                }
            }""",
            context_value=ctx,
        )
        assert result.errors is None
        data = result.data["addDatabase"]
        assert data["name"] == "mydb"
        assert data["connected"] is True
        assert data["isDynamic"] is True
        managed.session.add_database.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_database_no_uri(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            """mutation {
                addDatabase(sessionId: "s1", input: {name: "mydb"}) {
                    name
                }
            }""",
            context_value=ctx,
        )
        assert result.errors is not None

    @pytest.mark.asyncio
    async def test_add_database_xlsx_blocked(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            """mutation {
                addDatabase(sessionId: "s1", input: {name: "mydb", uri: "file:///data.xlsx"}) {
                    name
                }
            }""",
            context_value=ctx,
        )
        assert result.errors is not None

    @pytest.mark.asyncio
    async def test_remove_database(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed.session.session_databases = {}
        managed._dynamic_dbs = [{
            "name": "mydb",
            "type": "csv",
            "uri": "sqlite:///test.db",
            "connected": True,
            "table_count": 1,
            "added_at": "2025-01-01T00:00:00+00:00",
        }]
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        with patch("constat.server.session_manager.ManagedSession") as mock_ms:
            mock_ms._remove_db_from_user_config = MagicMock()
            result = await schema.execute(
                'mutation { removeDatabase(sessionId: "s1", name: "mydb") { status name } }',
                context_value=ctx,
            )
        assert result.errors is None
        assert result.data["removeDatabase"]["status"] == "deleted"
        assert result.data["removeDatabase"]["name"] == "mydb"

    @pytest.mark.asyncio
    async def test_remove_database_config_protected(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        db_config = MagicMock()
        managed.session.config.databases = {"protected": db_config}
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            'mutation { removeDatabase(sessionId: "s1", name: "protected") { status } }',
            context_value=ctx,
        )
        assert result.errors is not None

    @pytest.mark.asyncio
    async def test_remove_database_not_found(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            'mutation { removeDatabase(sessionId: "s1", name: "nope") { status } }',
            context_value=ctx,
        )
        assert result.errors is not None

    @pytest.mark.asyncio
    async def test_test_database(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed.has_database.return_value = True
        mock_sm = MagicMock()
        mock_sm.get_tables_for_db.return_value = ["t1", "t2"]
        managed.session.schema_manager = mock_sm
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            'mutation { testDatabase(sessionId: "s1", name: "mydb") { name connected tableCount error } }',
            context_value=ctx,
        )
        assert result.errors is None
        data = result.data["testDatabase"]
        assert data["name"] == "mydb"
        assert data["connected"] is True
        assert data["tableCount"] == 2
        assert data["error"] is None

    @pytest.mark.asyncio
    async def test_test_database_not_found(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed.has_database.return_value = False
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            'mutation { testDatabase(sessionId: "s1", name: "nope") { name } }',
            context_value=ctx,
        )
        assert result.errors is not None


class TestApiMutations:
    @pytest.mark.asyncio
    async def test_add_api(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed.session.resources = MagicMock()
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            """mutation {
                addApi(sessionId: "s1", input: {name: "myapi", type: "rest", baseUrl: "https://api.example.com"}) {
                    name type connected isDynamic source
                }
            }""",
            context_value=ctx,
        )
        assert result.errors is None
        data = result.data["addApi"]
        assert data["name"] == "myapi"
        assert data["connected"] is True
        assert data["isDynamic"] is True
        assert data["source"] == "session"

    @pytest.mark.asyncio
    async def test_add_api_duplicate(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed._dynamic_apis = [{"name": "myapi"}]
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            """mutation {
                addApi(sessionId: "s1", input: {name: "myapi", baseUrl: "https://api.example.com"}) {
                    name
                }
            }""",
            context_value=ctx,
        )
        assert result.errors is not None

    @pytest.mark.asyncio
    async def test_remove_api(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed._dynamic_apis = [{"name": "myapi", "type": "rest"}]
        managed.session.resources = MagicMock()
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            'mutation { removeApi(sessionId: "s1", name: "myapi") { status name } }',
            context_value=ctx,
        )
        assert result.errors is None
        assert result.data["removeApi"]["status"] == "deleted"
        assert result.data["removeApi"]["name"] == "myapi"

    @pytest.mark.asyncio
    async def test_remove_api_config_protected(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        api_config = MagicMock()
        managed.session.config.apis = {"protected": api_config}
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            'mutation { removeApi(sessionId: "s1", name: "protected") { status } }',
            context_value=ctx,
        )
        assert result.errors is not None

    @pytest.mark.asyncio
    async def test_remove_api_not_found(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            'mutation { removeApi(sessionId: "s1", name: "nope") { status } }',
            context_value=ctx,
        )
        assert result.errors is not None


class TestFileRefMutations:
    @pytest.mark.asyncio
    async def test_add_file_ref(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed.session.add_file = MagicMock()
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            """mutation {
                addFileRef(sessionId: "s1", input: {name: "manual", uri: "https://example.com/doc.pdf"}) {
                    name uri hasAuth description
                }
            }""",
            context_value=ctx,
        )
        assert result.errors is None
        data = result.data["addFileRef"]
        assert data["name"] == "manual"
        assert data["uri"] == "https://example.com/doc.pdf"
        assert data["hasAuth"] is False

    @pytest.mark.asyncio
    async def test_add_file_ref_with_auth(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed.session.add_file = MagicMock()
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            """mutation {
                addFileRef(sessionId: "s1", input: {name: "private", uri: "https://example.com/secret.pdf", auth: "Bearer tok"}) {
                    name hasAuth
                }
            }""",
            context_value=ctx,
        )
        assert result.errors is None
        assert result.data["addFileRef"]["hasAuth"] is True

    @pytest.mark.asyncio
    async def test_delete_file_ref(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed._file_refs = [{"name": "manual", "uri": "https://example.com", "has_auth": False, "added_at": "2025-01-01T00:00:00+00:00"}]
        managed.session.doc_tools = None
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            'mutation { deleteFileRef(sessionId: "s1", name: "manual") { status name } }',
            context_value=ctx,
        )
        assert result.errors is None
        assert result.data["deleteFileRef"]["status"] == "deleted"
        assert result.data["deleteFileRef"]["name"] == "manual"

    @pytest.mark.asyncio
    async def test_delete_file_ref_not_found(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            'mutation { deleteFileRef(sessionId: "s1", name: "nope") { status } }',
            context_value=ctx,
        )
        assert result.errors is not None


class TestDeleteFileMutation:
    @pytest.mark.asyncio
    async def test_delete_file(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        file_info = {"id": "f_abc", "filename": "data.csv", "file_path": "/tmp/does_not_exist.csv", "file_uri": "file:///tmp/does_not_exist.csv", "size_bytes": 100, "content_type": "text/csv", "uploaded_at": "2025-01-01T00:00:00+00:00"}

        with patch("constat.server.routes.files._get_uploaded_files_for_session", return_value=[file_info]), \
             patch("constat.server.routes.files._save_uploaded_files_for_session"):
            result = await schema.execute(
                'mutation { deleteFile(sessionId: "s1", fileId: "f_abc") { status name } }',
                context_value=ctx,
            )
        assert result.errors is None
        assert result.data["deleteFile"]["status"] == "deleted"
        assert result.data["deleteFile"]["name"] == "f_abc"

    @pytest.mark.asyncio
    async def test_delete_file_not_found(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        with patch("constat.server.routes.files._get_uploaded_files_for_session", return_value=[]), \
             patch("constat.server.routes.files._save_uploaded_files_for_session"):
            result = await schema.execute(
                'mutation { deleteFile(sessionId: "s1", fileId: "nope") { status } }',
                context_value=ctx,
            )
        assert result.errors is not None


class TestUploadFileDataUri:
    @pytest.mark.asyncio
    async def test_upload_file_data_uri(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        csv_content = b"a,b\n1,2\n"
        encoded = base64.b64encode(csv_content).decode()
        data_uri = f"data:text/csv;base64,{encoded}"

        with patch("constat.server.routes.files._get_upload_dir_for_session") as mock_dir, \
             patch("constat.server.routes.files._get_uploaded_files_for_session", return_value=[]), \
             patch("constat.server.routes.files._save_uploaded_files_for_session"), \
             patch("builtins.open", MagicMock()):
            mock_dir.return_value = Path("/tmp/test-uploads")
            result = await schema.execute(
                'mutation { uploadFileDataUri(sessionId: "s1", filename: "test.csv", dataUri: "' + data_uri + '") { id filename contentType } }',
                context_value=ctx,
            )
        assert result.errors is None
        assert result.data["uploadFileDataUri"]["filename"] == "test.csv"
        assert result.data["uploadFileDataUri"]["contentType"] == "text/csv"

    @pytest.mark.asyncio
    async def test_upload_file_data_uri_raw_base64(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        content = b"hello world"
        encoded = base64.b64encode(content).decode()

        with patch("constat.server.routes.files._get_upload_dir_for_session") as mock_dir, \
             patch("constat.server.routes.files._get_uploaded_files_for_session", return_value=[]), \
             patch("constat.server.routes.files._save_uploaded_files_for_session"), \
             patch("builtins.open", MagicMock()):
            mock_dir.return_value = Path("/tmp/test-uploads")
            result = await schema.execute(
                'mutation { uploadFileDataUri(sessionId: "s1", filename: "data.bin", dataUri: "' + encoded + '") { id filename contentType } }',
                context_value=ctx,
            )
        assert result.errors is None
        assert result.data["uploadFileDataUri"]["contentType"] == "application/octet-stream"

    @pytest.mark.asyncio
    async def test_upload_file_data_uri_invalid_base64(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            'mutation { uploadFileDataUri(sessionId: "s1", filename: "bad.txt", dataUri: "not-valid-base64!!!") { id } }',
            context_value=ctx,
        )
        assert result.errors is not None


class TestDocumentMutations:
    @pytest.mark.asyncio
    async def test_add_document_uri(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed.session.doc_tools = MagicMock()
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        with patch("constat.server.routes.files._ingest_source_async"):
            result = await schema.execute(
                """mutation {
                    addDocumentUri(sessionId: "s1", input: {name: "wiki", url: "https://example.com/docs"}) {
                        status name
                    }
                }""",
                context_value=ctx,
            )
        assert result.errors is None
        data = result.data["addDocumentUri"]
        assert data["status"] == "accepted"
        assert data["name"] == "wiki"

    @pytest.mark.asyncio
    async def test_add_document_uri_no_doc_tools(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed.session.doc_tools = None
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            """mutation {
                addDocumentUri(sessionId: "s1", input: {name: "wiki", url: "https://example.com/docs"}) {
                    status
                }
            }""",
            context_value=ctx,
        )
        assert result.errors is not None

    @pytest.mark.asyncio
    async def test_add_email_source(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed.session.doc_tools = MagicMock()
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        with patch("constat.server.routes.files._ingest_source_async"):
            result = await schema.execute(
                """mutation {
                    addEmailSource(sessionId: "s1", input: {name: "inbox", url: "imap://mail.example.com", username: "user@example.com"}) {
                        status name
                    }
                }""",
                context_value=ctx,
            )
        assert result.errors is None
        data = result.data["addEmailSource"]
        assert data["status"] == "accepted"
        assert data["name"] == "inbox"

    @pytest.mark.asyncio
    async def test_refresh_documents(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed._file_refs = [{"name": "doc1", "uri": "https://example.com", "has_auth": False, "added_at": "2025-01-01T00:00:00+00:00"}]
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        with patch("constat.server.source_refresher.refresh_session_sources"):
            result = await schema.execute(
                'mutation { refreshDocuments(sessionId: "s1") { status name } }',
                context_value=ctx,
            )
        assert result.errors is None
        assert result.data["refreshDocuments"]["status"] == "started"

    @pytest.mark.asyncio
    async def test_refresh_documents_no_sources(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            'mutation { refreshDocuments(sessionId: "s1") { status name } }',
            context_value=ctx,
        )
        assert result.errors is None
        assert result.data["refreshDocuments"]["status"] == "skipped"


class TestUserSourceMutations:
    @pytest.mark.asyncio
    async def test_remove_user_source(self):
        from constat.server.graphql import schema

        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = _make_managed()

        user_config = {"databases": {"mydb": {"source": "user", "type": "sqlite"}}}

        with patch("constat.server.routes.user_sources._load_user_config", return_value=user_config), \
             patch("constat.server.routes.user_sources._save_user_config") as mock_save:
            result = await schema.execute(
                'mutation { removeUserSource(sourceType: "databases", sourceName: "mydb") { status name sourceType } }',
                context_value=ctx,
            )
        assert result.errors is None
        data = result.data["removeUserSource"]
        assert data["status"] == "removed"
        assert data["name"] == "mydb"
        assert data["sourceType"] == "databases"
        mock_save.assert_called_once()

    @pytest.mark.asyncio
    async def test_remove_user_source_not_found(self):
        from constat.server.graphql import schema

        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = _make_managed()

        with patch("constat.server.routes.user_sources._load_user_config", return_value={"databases": {}}):
            result = await schema.execute(
                'mutation { removeUserSource(sourceType: "databases", sourceName: "nope") { status } }',
                context_value=ctx,
            )
        assert result.errors is not None

    @pytest.mark.asyncio
    async def test_remove_user_source_invalid_type(self):
        from constat.server.graphql import schema

        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = _make_managed()

        result = await schema.execute(
            'mutation { removeUserSource(sourceType: "invalid", sourceName: "x") { status } }',
            context_value=ctx,
        )
        assert result.errors is not None

    @pytest.mark.asyncio
    async def test_move_source(self):
        from constat.server.graphql import schema

        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = _make_managed()

        with patch("constat.server.routes.user_sources._load_user_config", return_value={"databases": {}}), \
             patch("constat.server.routes.user_sources._save_user_config") as mock_save:
            result = await schema.execute(
                """mutation {
                    moveSource(sourceType: "databases", sourceName: "mydb", fromDomain: "session", toDomain: "user") {
                        status name sourceType
                    }
                }""",
                context_value=ctx,
            )
        assert result.errors is None
        data = result.data["moveSource"]
        assert data["status"] == "moved"
        assert data["name"] == "mydb"
        assert data["sourceType"] == "databases"
        mock_save.assert_called_once()

    @pytest.mark.asyncio
    async def test_move_source_invalid_type(self):
        from constat.server.graphql import schema

        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = _make_managed()

        result = await schema.execute(
            """mutation {
                moveSource(sourceType: "invalid", sourceName: "x", fromDomain: "a", toDomain: "b") { status }
            }""",
            context_value=ctx,
        )
        assert result.errors is not None


class TestSecurityHardeningMutations:
    """Security hardening tests for mutation resolvers."""

    @pytest.mark.asyncio
    async def test_remove_database_propagates_unlink_error(self, tmp_path):
        from constat.server.graphql import schema

        # Create a real file, then make unlink fail
        test_file = tmp_path / "test.csv"
        test_file.write_text("a,b\n1,2\n")

        managed = _make_managed()
        managed.session.session_databases = {}
        managed._dynamic_dbs = [{
            "name": "mydb",
            "type": "csv",
            "uri": str(test_file),
            "connected": True,
            "table_count": 1,
            "added_at": "2025-01-01T00:00:00+00:00",
            "file_id": "f_abc123",
        }]
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        with patch("constat.server.session_manager.ManagedSession") as mock_ms, \
             patch("pathlib.Path.unlink", side_effect=OSError("Permission denied")):
            mock_ms._remove_db_from_user_config = MagicMock()
            result = await schema.execute(
                'mutation { removeDatabase(sessionId: "s1", name: "mydb") { status name } }',
                context_value=ctx,
            )
        assert result.errors is not None
        assert "Failed to delete file" in str(result.errors[0])


class TestDatabaseExtraConfig:
    @pytest.mark.asyncio
    async def test_add_database_with_extra_config(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed.session.add_database = MagicMock()
        managed.session.session_databases = {}
        mock_sm = MagicMock()
        mock_sm.metadata_cache = {}
        managed.session.schema_manager = mock_sm
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        with patch("constat.core.config.DatabaseConfig") as mock_db_config:
            mock_db_config.return_value = MagicMock()
            result = await schema.execute(
                """mutation AddDB($input: DatabaseAddInput!) {
                    addDatabase(sessionId: "s1", input: $input) { name connected }
                }""",
                variable_values={"input": {
                    "name": "dynamo",
                    "uri": "dynamodb://us-east-1",
                    "type": "dynamodb",
                    "extraConfig": {"region": "us-east-1", "awsAccessKeyId": "AKIA123"},
                }},
                context_value=ctx,
            )
        assert result.errors is None
        assert result.data["addDatabase"]["connected"] is True
        call_kwargs = mock_db_config.call_args[1]
        assert call_kwargs.get("region") == "us-east-1"
        assert call_kwargs.get("awsAccessKeyId") == "AKIA123"

    @pytest.mark.asyncio
    async def test_add_database_extra_config_none(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed.session.add_database = MagicMock()
        managed.session.session_databases = {}
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            """mutation AddDB($input: DatabaseAddInput!) {
                addDatabase(sessionId: "s1", input: $input) { name connected }
            }""",
            variable_values={"input": {
                "name": "mydb",
                "uri": "sqlite:///test.db",
                "type": "sqlalchemy",
            }},
            context_value=ctx,
        )
        assert result.errors is None
        assert result.data["addDatabase"]["name"] == "mydb"

    @pytest.mark.asyncio
    async def test_add_database_extra_config_merged_into_db_config(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed.session.add_database = MagicMock()
        managed.session.session_databases = {}
        mock_sm = MagicMock()
        mock_sm.metadata_cache = {}
        managed.session.schema_manager = mock_sm
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        with patch("constat.core.config.DatabaseConfig") as mock_db_config:
            mock_db_config.return_value = MagicMock()
            result = await schema.execute(
                """mutation AddDB($input: DatabaseAddInput!) {
                    addDatabase(sessionId: "s1", input: $input) { name connected }
                }""",
                variable_values={"input": {
                    "name": "elastic",
                    "uri": "http://localhost:9200",
                    "type": "elasticsearch",
                    "extraConfig": {"apiKey": "mykey"},
                }},
                context_value=ctx,
            )
        assert result.errors is None
        call_kwargs = mock_db_config.call_args[1]
        assert call_kwargs.get("apiKey") == "mykey"


class TestApiAuthMutations:
    @pytest.mark.asyncio
    async def test_add_api_bearer_auth(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed.session.resources = MagicMock()
        managed.session.api_schema_manager = MagicMock()
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            """mutation {
                addApi(sessionId: "s1", input: {
                    name: "testapi",
                    baseUrl: "https://api.example.com",
                    type: "rest",
                    authType: "bearer",
                    authToken: "mytoken"
                }) { name connected }
            }""",
            context_value=ctx,
        )
        assert result.errors is None
        assert result.data["addApi"]["connected"] is True
        call_args = managed.session.api_schema_manager.add_api_dynamic.call_args
        api_config = call_args[0][1]
        assert api_config.headers == {"Authorization": "Bearer mytoken"}

    @pytest.mark.asyncio
    async def test_add_api_basic_auth(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed.session.resources = MagicMock()
        managed.session.api_schema_manager = MagicMock()
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            """mutation {
                addApi(sessionId: "s1", input: {
                    name: "basicapi",
                    baseUrl: "https://api.example.com",
                    type: "rest",
                    authType: "basic",
                    authUsername: "user",
                    authPassword: "pass"
                }) { name connected }
            }""",
            context_value=ctx,
        )
        assert result.errors is None
        call_args = managed.session.api_schema_manager.add_api_dynamic.call_args
        api_config = call_args[0][1]
        expected = base64.b64encode(b"user:pass").decode()
        assert api_config.headers == {"Authorization": f"Basic {expected}"}

    @pytest.mark.asyncio
    async def test_add_api_api_key_auth(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed.session.resources = MagicMock()
        managed.session.api_schema_manager = MagicMock()
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            """mutation {
                addApi(sessionId: "s1", input: {
                    name: "apikeyapi",
                    baseUrl: "https://api.example.com",
                    type: "rest",
                    authType: "api_key",
                    authHeader: "X-API-Key",
                    authToken: "secret"
                }) { name connected }
            }""",
            context_value=ctx,
        )
        assert result.errors is None
        call_args = managed.session.api_schema_manager.add_api_dynamic.call_args
        api_config = call_args[0][1]
        assert api_config.headers == {"X-API-Key": "secret"}

    @pytest.mark.asyncio
    async def test_add_api_no_auth(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed.session.resources = MagicMock()
        managed.session.api_schema_manager = MagicMock()
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            """mutation {
                addApi(sessionId: "s1", input: {
                    name: "noauthapi",
                    baseUrl: "https://api.example.com",
                    type: "rest"
                }) { name connected }
            }""",
            context_value=ctx,
        )
        assert result.errors is None
        call_args = managed.session.api_schema_manager.add_api_dynamic.call_args
        api_config = call_args[0][1]
        assert not api_config.headers

    def test_add_api_auth_fields_in_sdl(self):
        from constat.server.graphql import schema
        sdl = schema.as_str()
        assert "authToken" in sdl
        assert "authUsername" in sdl
        assert "authPassword" in sdl
        assert "authClientId" in sdl
        assert "authClientSecret" in sdl
        assert "authTokenUrl" in sdl
