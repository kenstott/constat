# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Unit tests for GraphQL update_database, update_api, update_document mutations."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit

# Fix broken import in user_sources module
import constat.server.config as _sc
if not hasattr(_sc, "user_vault_dir"):
    from constat.core.paths import user_vault_dir as _uvd
    _sc.user_vault_dir = _uvd
import constat.server.routes.user_sources  # noqa: F401


def _make_context(user_id: str = "test-user"):
    from constat.server.graphql.session_context import GraphQLContext

    mock_sm = MagicMock()
    mock_server_config = MagicMock()
    mock_server_config.auth_disabled = True
    mock_server_config.base_url = "http://localhost:3000"
    mock_server_config.data_dir = Path("/tmp/test-graphql-source-updates")

    ctx = GraphQLContext(
        session_manager=mock_sm,
        server_config=mock_server_config,
        user_id=user_id,
    )
    ctx.request = MagicMock()
    return ctx


def _make_managed(session_id: str = "s1", user_id: str = "test-user"):
    from constat.server.models import SessionStatus

    managed = MagicMock()
    managed.session_id = session_id
    managed.user_id = user_id
    managed.status = SessionStatus("idle")
    managed.created_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
    managed.last_activity = datetime(2025, 1, 1, tzinfo=timezone.utc)
    managed.active_domains = []
    managed.session.session_id = f"hist-{session_id}"
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


def _dynamic_db(name: str = "mydb", **kwargs) -> dict:
    return {
        "name": name,
        "type": "sqlalchemy",
        "dialect": "sqlite",
        "description": "original description",
        "uri": "sqlite:///original.db",
        "connected": True,
        "table_count": 1,
        "added_at": "2025-01-01T00:00:00+00:00",
        "is_dynamic": True,
        "file_id": None,
        **kwargs,
    }


def _dynamic_api(name: str = "myapi", **kwargs) -> dict:
    return {
        "name": name,
        "type": "rest",
        "base_url": "https://api.example.com",
        "description": "original api description",
        "auth_type": None,
        "auth_header": None,
        "auth_token": None,
        "connected": True,
        "added_at": "2025-01-01T00:00:00+00:00",
        "is_dynamic": True,
        **kwargs,
    }


def _dynamic_doc(name: str = "mydoc", **kwargs) -> dict:
    return {
        "name": name,
        "uri": "file:///original.pdf",
        "has_auth": False,
        "description": "original doc description",
        "added_at": "2025-01-01T00:00:00+00:00",
        **kwargs,
    }


# ============================================================================
# update_database tests
# ============================================================================


class TestUpdateDatabase:
    @pytest.mark.asyncio
    async def test_update_database_name(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed._dynamic_dbs = [_dynamic_db("mydb")]
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            """mutation {
                updateDatabase(sessionId: "s1", input: {name: "mydb", newName: "renamed_db"}) {
                    name isDynamic
                }
            }""",
            context_value=ctx,
        )
        assert result.errors is None
        data = result.data["updateDatabase"]
        assert data["name"] == "renamed_db"
        # Original name must no longer be in the dynamic DB list
        names = [d["name"] for d in managed._dynamic_dbs]
        assert "mydb" not in names
        assert "renamed_db" in names

    @pytest.mark.asyncio
    async def test_update_database_description(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed._dynamic_dbs = [_dynamic_db("mydb")]
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            """mutation {
                updateDatabase(sessionId: "s1", input: {name: "mydb", description: "new description"}) {
                    name description
                }
            }""",
            context_value=ctx,
        )
        assert result.errors is None
        data = result.data["updateDatabase"]
        assert data["description"] == "new description"
        assert managed._dynamic_dbs[0]["description"] == "new description"

    @pytest.mark.asyncio
    async def test_update_database_uri(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed._dynamic_dbs = [_dynamic_db("mydb")]
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            """mutation {
                updateDatabase(sessionId: "s1", input: {name: "mydb", uri: "sqlite:///new.db"}) {
                    name
                }
            }""",
            context_value=ctx,
        )
        assert result.errors is None
        assert managed._dynamic_dbs[0]["uri"] == "sqlite:///new.db"

    @pytest.mark.asyncio
    async def test_update_database_rejects_config_defined(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed.session.config.databases = {"config_db": MagicMock()}
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            """mutation {
                updateDatabase(sessionId: "s1", input: {name: "config_db", description: "hack"}) {
                    name
                }
            }""",
            context_value=ctx,
        )
        assert result.errors is not None

    @pytest.mark.asyncio
    async def test_update_database_name_not_found(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed._dynamic_dbs = []
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            """mutation {
                updateDatabase(sessionId: "s1", input: {name: "ghost", description: "irrelevant"}) {
                    name
                }
            }""",
            context_value=ctx,
        )
        assert result.errors is not None


# ============================================================================
# update_api tests
# ============================================================================


class TestUpdateApi:
    @pytest.mark.asyncio
    async def test_update_api_base_url(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed._dynamic_apis = [_dynamic_api("myapi")]
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            """mutation {
                updateApi(sessionId: "s1", input: {name: "myapi", baseUrl: "https://new.example.com"}) {
                    name baseUrl
                }
            }""",
            context_value=ctx,
        )
        assert result.errors is None
        data = result.data["updateApi"]
        assert data["baseUrl"] == "https://new.example.com"
        assert managed._dynamic_apis[0]["base_url"] == "https://new.example.com"

    @pytest.mark.asyncio
    async def test_update_api_auth_credentials(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed._dynamic_apis = [_dynamic_api("myapi")]
        # Provide a real api_schema_manager mock so the bearer branch executes
        mock_api_sm = MagicMock()
        managed.session.api_schema_manager = mock_api_sm

        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            """mutation {
                updateApi(sessionId: "s1", input: {
                    name: "myapi",
                    authType: "bearer",
                    authToken: "secret-token"
                }) {
                    name
                }
            }""",
            context_value=ctx,
        )
        assert result.errors is None
        assert managed._dynamic_apis[0]["auth_type"] == "bearer"
        # Confirm the api_schema_manager was called with a config that has the bearer header
        mock_api_sm.add_api_dynamic.assert_called_once()
        _name, api_config = mock_api_sm.add_api_dynamic.call_args[0]
        assert api_config.headers.get("Authorization") == "Bearer secret-token"

    @pytest.mark.asyncio
    async def test_update_api_rejects_config_defined(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed.session.config.apis = {"config_api": MagicMock()}
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            """mutation {
                updateApi(sessionId: "s1", input: {name: "config_api", baseUrl: "https://evil.com"}) {
                    name
                }
            }""",
            context_value=ctx,
        )
        assert result.errors is not None


# ============================================================================
# update_document tests
# ============================================================================


class TestUpdateDocument:
    @pytest.mark.asyncio
    async def test_update_document_description(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed._file_refs = [_dynamic_doc("mydoc")]
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            """mutation {
                updateDocument(sessionId: "s1", input: {name: "mydoc", description: "updated desc"}) {
                    name description
                }
            }""",
            context_value=ctx,
        )
        assert result.errors is None
        data = result.data["updateDocument"]
        assert data["description"] == "updated desc"
        assert managed._file_refs[0]["description"] == "updated desc"

    @pytest.mark.asyncio
    async def test_update_document_uri(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed._file_refs = [_dynamic_doc("mydoc")]
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            """mutation {
                updateDocument(sessionId: "s1", input: {name: "mydoc", uri: "file:///new.pdf"}) {
                    name path
                }
            }""",
            context_value=ctx,
        )
        assert result.errors is None
        data = result.data["updateDocument"]
        assert data["path"] == "file:///new.pdf"
        assert managed._file_refs[0]["uri"] == "file:///new.pdf"

    @pytest.mark.asyncio
    async def test_update_document_http_uri_routes_to_url(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed._file_refs = [_dynamic_doc("mydoc", document_config={})]
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            """mutation {
                updateDocument(sessionId: "s1", input: {name: "mydoc", uri: "https://example.com/new"}) {
                    name path
                }
            }""",
            context_value=ctx,
        )
        assert result.errors is None
        ref = managed._file_refs[0]
        assert ref["document_config"]["url"] == "https://example.com/new"
        assert "path" not in ref["document_config"]

    @pytest.mark.asyncio
    async def test_update_document_s3_uri_routes_to_url(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed._file_refs = [_dynamic_doc("mydoc", document_config={})]
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            """mutation {
                updateDocument(sessionId: "s1", input: {name: "mydoc", uri: "s3://bucket/key.pdf"}) {
                    name path
                }
            }""",
            context_value=ctx,
        )
        assert result.errors is None
        ref = managed._file_refs[0]
        assert ref["document_config"]["url"] == "s3://bucket/key.pdf"
        assert "path" not in ref["document_config"]

    @pytest.mark.asyncio
    async def test_update_document_bare_path_routes_to_path(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed._file_refs = [_dynamic_doc("mydoc", document_config={})]
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            """mutation {
                updateDocument(sessionId: "s1", input: {name: "mydoc", uri: "/data/rules.md"}) {
                    name path
                }
            }""",
            context_value=ctx,
        )
        assert result.errors is None
        ref = managed._file_refs[0]
        assert ref["document_config"]["path"] == "/data/rules.md"
        assert "url" not in ref["document_config"]

    @pytest.mark.asyncio
    async def test_update_document_clears_old_url_when_switching_to_path(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed._file_refs = [
            _dynamic_doc("mydoc", document_config={"url": "https://old.com"})
        ]
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            """mutation {
                updateDocument(sessionId: "s1", input: {name: "mydoc", uri: "/new/file.md"}) {
                    name path
                }
            }""",
            context_value=ctx,
        )
        assert result.errors is None
        ref = managed._file_refs[0]
        assert "url" not in ref["document_config"]
        assert ref["document_config"]["path"] == "/new/file.md"

    @pytest.mark.asyncio
    async def test_update_document_rejects_config_defined(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed.session.config.documents = {"config_doc": MagicMock()}
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            """mutation {
                updateDocument(sessionId: "s1", input: {name: "config_doc", description: "hack"}) {
                    name
                }
            }""",
            context_value=ctx,
        )
        assert result.errors is not None
