# Copyright (c) 2025 Kenneth Stott
# Canary: ee9ce44b-7b77-42f6-a690-edab84212dcf
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Unit tests for public (unauthenticated) GraphQL resolvers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_context(user_id=None):
    from constat.server.graphql.session_context import GraphQLContext

    mock_sm = MagicMock()
    mock_server_config = MagicMock()
    mock_server_config.auth_disabled = True
    mock_server_config.base_url = "http://localhost:3000"
    mock_server_config.data_dir = Path("/tmp/test-graphql-public")

    ctx = GraphQLContext(
        session_manager=mock_sm,
        server_config=mock_server_config,
        user_id=user_id,
    )
    mock_request = MagicMock()
    ctx.request = mock_request
    return ctx


def _make_public_managed(session_id="pub-1", user_id="test-user", status="idle"):
    from constat.server.models import SessionStatus

    managed = MagicMock()
    managed.session_id = session_id
    managed.user_id = user_id
    managed.status = SessionStatus(status)
    managed.session.datastore = MagicMock()
    managed.session.datastore.is_public.return_value = True
    managed.session.datastore.get_session_meta.return_value = "Test summary"
    return managed


def _make_private_managed(session_id="priv-1"):
    managed = MagicMock()
    managed.session_id = session_id
    managed.user_id = "test-user"
    managed.session.datastore = MagicMock()
    managed.session.datastore.is_public.return_value = False
    return managed


class TestPublicSchemaStitching:
    """Verify all 7 public queries appear in the SDL."""

    def _get_sdl(self):
        from constat.server.graphql import schema
        return schema.as_str()

    def test_public_session_query(self):
        assert "publicSession(" in self._get_sdl()

    def test_public_messages_query(self):
        assert "publicMessages(" in self._get_sdl()

    def test_public_artifacts_query(self):
        assert "publicArtifacts(" in self._get_sdl()

    def test_public_artifact_query(self):
        assert "publicArtifact(" in self._get_sdl()

    def test_public_tables_query(self):
        assert "publicTables(" in self._get_sdl()

    def test_public_table_data_query(self):
        assert "publicTableData(" in self._get_sdl()

    def test_public_proof_facts_query(self):
        assert "publicProofFacts(" in self._get_sdl()


class TestPublicSessionQuery:
    @pytest.mark.asyncio
    async def test_returns_summary_for_public_session(self):
        from constat.server.graphql.public_resolvers import Query

        ctx = _make_context()
        managed = _make_public_managed()
        ctx.session_manager.get_session_or_none.return_value = managed

        info = MagicMock()
        info.context = ctx

        q = Query()
        result = await q.public_session(info, "pub-1")
        assert result.session_id == "pub-1"
        assert result.summary == "Test summary"
        assert result.status == "idle"

    @pytest.mark.asyncio
    async def test_raises_for_nonexistent_session(self):
        from constat.server.graphql.public_resolvers import Query

        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = None

        info = MagicMock()
        info.context = ctx

        q = Query()
        with pytest.raises(ValueError, match="Not found"):
            await q.public_session(info, "nonexistent")

    @pytest.mark.asyncio
    async def test_raises_for_private_session(self):
        from constat.server.graphql.public_resolvers import Query

        ctx = _make_context()
        managed = _make_private_managed()
        ctx.session_manager.get_session_or_none.return_value = managed

        info = MagicMock()
        info.context = ctx

        q = Query()
        with pytest.raises(ValueError, match="Not found"):
            await q.public_session(info, "priv-1")


class TestPublicMessagesQuery:
    @pytest.mark.asyncio
    async def test_returns_messages(self):
        from constat.server.graphql.public_resolvers import Query

        ctx = _make_context()
        managed = _make_public_managed()
        ctx.session_manager.get_session_or_none.return_value = managed

        info = MagicMock()
        info.context = ctx

        raw_messages = [
            {"id": "m1", "type": "user", "content": "hello", "timestamp": "2025-01-01T00:00:00Z"},
            {"id": "m2", "type": "bot", "content": "hi", "timestamp": "2025-01-01T00:00:01Z"},
        ]

        with patch("constat.storage.history.SessionHistory") as MockHistory:
            MockHistory.return_value.load_messages_by_server_id.return_value = raw_messages
            q = Query()
            result = await q.public_messages(info, "pub-1")

        assert len(result.messages) == 2
        assert result.messages[0].id == "m1"
        assert result.messages[1].content == "hi"


class TestPublicArtifactsQuery:
    @pytest.mark.asyncio
    async def test_returns_artifacts(self):
        from constat.server.graphql.public_resolvers import Query

        ctx = _make_context()
        managed = _make_public_managed()
        ctx.session_manager.get_session_or_none.return_value = managed

        managed.session.datastore.list_artifacts.return_value = [
            {"id": 1, "name": "chart1", "type": "plotly", "step_number": 1},
        ]
        mock_artifact = MagicMock()
        mock_artifact.metadata = {"key": "value"}
        managed.session.datastore.get_artifact_by_id.return_value = mock_artifact

        info = MagicMock()
        info.context = ctx

        q = Query()
        result = await q.public_artifacts(info, "pub-1")
        assert len(result) == 1
        assert result[0].name == "chart1"
        assert result[0].is_starred is False

    @pytest.mark.asyncio
    async def test_raises_for_private(self):
        from constat.server.graphql.public_resolvers import Query

        ctx = _make_context()
        managed = _make_private_managed()
        ctx.session_manager.get_session_or_none.return_value = managed

        info = MagicMock()
        info.context = ctx

        q = Query()
        with pytest.raises(ValueError, match="Not found"):
            await q.public_artifacts(info, "priv-1")


class TestPublicArtifactQuery:
    @pytest.mark.asyncio
    async def test_returns_artifact_content(self):
        from constat.server.graphql.public_resolvers import Query

        ctx = _make_context()
        managed = _make_public_managed()
        ctx.session_manager.get_session_or_none.return_value = managed

        mock_artifact = MagicMock()
        mock_artifact.id = 1
        mock_artifact.name = "chart1"
        mock_artifact.artifact_type.value = "plotly"
        mock_artifact.content = '{"data": []}'
        mock_artifact.mime_type = "application/json"
        mock_artifact.is_binary = False
        managed.session.datastore.get_artifact_by_id.return_value = mock_artifact

        info = MagicMock()
        info.context = ctx

        q = Query()
        result = await q.public_artifact(info, "pub-1", 1)
        assert result.id == 1
        assert result.content == '{"data": []}'

    @pytest.mark.asyncio
    async def test_raises_for_missing_artifact(self):
        from constat.server.graphql.public_resolvers import Query

        ctx = _make_context()
        managed = _make_public_managed()
        ctx.session_manager.get_session_or_none.return_value = managed
        managed.session.datastore.get_artifact_by_id.return_value = None

        info = MagicMock()
        info.context = ctx

        q = Query()
        with pytest.raises(ValueError, match="Not found"):
            await q.public_artifact(info, "pub-1", 999)


class TestPublicTablesQuery:
    @pytest.mark.asyncio
    async def test_returns_tables_excluding_internal(self):
        from constat.server.graphql.public_resolvers import Query

        ctx = _make_context()
        managed = _make_public_managed()
        ctx.session_manager.get_session_or_none.return_value = managed

        managed.session.datastore.list_tables.return_value = [
            {"name": "sales", "row_count": 100, "step_number": 1},
            {"name": "_metadata", "row_count": 5, "step_number": 0},
        ]
        managed.session.datastore.get_table_schema.return_value = [
            {"name": "id"}, {"name": "amount"},
        ]

        info = MagicMock()
        info.context = ctx

        q = Query()
        result = await q.public_tables(info, "pub-1")
        assert len(result) == 1
        assert result[0].name == "sales"
        assert result[0].columns == ["id", "amount"]


class TestPublicTableDataQuery:
    @pytest.mark.asyncio
    async def test_returns_paginated_data(self):
        import pandas as pd
        from constat.server.graphql.public_resolvers import Query

        ctx = _make_context()
        managed = _make_public_managed()
        ctx.session_manager.get_session_or_none.return_value = managed

        df = pd.DataFrame({"id": range(10), "value": range(10, 20)})
        managed.session.datastore.load_dataframe.return_value = df

        info = MagicMock()
        info.context = ctx

        with patch("constat.server.routes.data._sanitize_df_for_json",
                    side_effect=lambda d: d.to_dict(orient="records")):
            q = Query()
            result = await q.public_table_data(info, "pub-1", "sales", page=1, page_size=5)

        assert result.name == "sales"
        assert result.total_rows == 10
        assert result.has_more is True
        assert result.page == 1
        assert result.page_size == 5

    @pytest.mark.asyncio
    async def test_raises_for_missing_table(self):
        from constat.server.graphql.public_resolvers import Query

        ctx = _make_context()
        managed = _make_public_managed()
        ctx.session_manager.get_session_or_none.return_value = managed
        managed.session.datastore.load_dataframe.return_value = None

        info = MagicMock()
        info.context = ctx

        q = Query()
        with pytest.raises(ValueError, match="Not found"):
            await q.public_table_data(info, "pub-1", "nonexistent")


class TestPublicProofFactsQuery:
    @pytest.mark.asyncio
    async def test_returns_proof_facts(self):
        from constat.server.graphql.public_resolvers import Query

        ctx = _make_context()
        managed = _make_public_managed()
        ctx.session_manager.get_session_or_none.return_value = managed

        raw_facts = [
            {"id": "f1", "name": "Revenue", "status": "resolved", "value": 1000},
        ]

        info = MagicMock()
        info.context = ctx

        with patch("constat.storage.history.SessionHistory") as MockHistory:
            MockHistory.return_value.load_proof_facts_by_server_id.return_value = (raw_facts, "Summary text")
            q = Query()
            result = await q.public_proof_facts(info, "pub-1")

        assert len(result.facts) == 1
        assert result.facts[0].name == "Revenue"
        assert result.summary == "Summary text"
