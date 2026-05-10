# Copyright (c) 2025 Kenneth Stott
# Canary: 265bf437-5bd6-4979-ac48-19e2cef9d5fe
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Unit tests for GraphQL session resolvers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_context(user_id="test-user"):
    from constat.server.graphql.session_context import GraphQLContext

    mock_sm = MagicMock()
    mock_server_config = MagicMock()
    mock_server_config.auth_disabled = True
    mock_server_config.base_url = "http://localhost:3000"
    mock_server_config.data_dir = Path("/tmp/test-graphql-sessions")

    ctx = GraphQLContext(
        session_manager=mock_sm,
        server_config=mock_server_config,
        user_id=user_id,
    )
    mock_request = MagicMock()
    ctx.request = mock_request
    return ctx


def _make_managed(session_id="s1", user_id="test-user", status="idle"):
    from constat.server.models import SessionStatus

    managed = MagicMock()
    managed.session_id = session_id
    managed.user_id = user_id
    managed.status = SessionStatus(status)
    managed.created_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
    managed.last_activity = datetime(2025, 1, 1, tzinfo=timezone.utc)
    managed.current_query = None
    managed.active_domains = ["sales-analytics.yaml"]
    managed.session.datastore = MagicMock()
    managed.session.datastore.list_tables.return_value = ["t1", "t2"]
    managed.session.datastore.list_artifacts.return_value = ["a1"]
    managed.session.datastore.get_session_meta.return_value = "Test summary"
    managed.session.datastore.get_shared_users.return_value = []
    managed.session.datastore.is_public.return_value = False
    managed.session.history = None
    return managed


class TestSchemaStitching:
    def test_schema_has_sessions_query(self):
        from constat.server.graphql import schema

        sdl = schema.as_str()
        assert "sessions:" in sdl or "sessions(" in sdl

    def test_schema_has_session_query(self):
        from constat.server.graphql import schema

        sdl = schema.as_str()
        assert "session(" in sdl

    def test_schema_has_session_shares_query(self):
        from constat.server.graphql import schema

        sdl = schema.as_str()
        assert "sessionShares(" in sdl

    def test_schema_has_active_domains_query(self):
        from constat.server.graphql import schema

        sdl = schema.as_str()
        assert "activeDomains(" in sdl

    def test_schema_has_create_session_mutation(self):
        from constat.server.graphql import schema

        sdl = schema.as_str()
        assert "createSession(" in sdl

    def test_schema_has_delete_session_mutation(self):
        from constat.server.graphql import schema

        sdl = schema.as_str()
        assert "deleteSession(" in sdl

    def test_schema_has_toggle_public_sharing_mutation(self):
        from constat.server.graphql import schema

        sdl = schema.as_str()
        assert "togglePublicSharing(" in sdl

    def test_schema_has_share_session_mutation(self):
        from constat.server.graphql import schema

        sdl = schema.as_str()
        assert "shareSession(" in sdl

    def test_schema_has_remove_share_mutation(self):
        from constat.server.graphql import schema

        sdl = schema.as_str()
        assert "removeShare(" in sdl

    def test_schema_has_reset_context_mutation(self):
        from constat.server.graphql import schema

        sdl = schema.as_str()
        assert "resetContext(" in sdl

    def test_schema_has_set_active_domains_mutation(self):
        from constat.server.graphql import schema

        sdl = schema.as_str()
        assert "setActiveDomains(" in sdl

    def test_schema_has_session_type(self):
        from constat.server.graphql import schema

        sdl = schema.as_str()
        assert "type SessionType" in sdl
        assert "sessionId:" in sdl
        assert "userId:" in sdl
        assert "tablesCount:" in sdl
        assert "artifactsCount:" in sdl
        assert "isPublic:" in sdl

    def test_schema_has_session_status_enum(self):
        from constat.server.graphql import schema

        sdl = schema.as_str()
        assert "enum SessionStatusEnum" in sdl
        assert "IDLE" in sdl
        assert "PLANNING" in sdl
        assert "EXECUTING" in sdl


class TestSessionResolvers:
    @pytest.mark.asyncio
    async def test_sessions_list(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        ctx = _make_context()
        ctx.session_manager.list_sessions.side_effect = [
            [managed],  # first call: user's sessions
            [managed],  # second call: all sessions
        ]

        with patch("constat.storage.history.SessionHistory") as mock_hist:
            mock_hist.return_value.list_sessions.return_value = []
            result = await schema.execute(
                "{ sessions { sessions { sessionId userId status summary tablesCount artifactsCount } total } }",
                context_value=ctx,
            )

        assert result.errors is None
        data = result.data["sessions"]
        assert data["total"] == 1
        assert data["sessions"][0]["sessionId"] == "s1"
        assert data["sessions"][0]["userId"] == "test-user"
        assert data["sessions"][0]["status"] == "IDLE"
        assert data["sessions"][0]["summary"] == "Test summary"
        assert data["sessions"][0]["tablesCount"] == 2
        assert data["sessions"][0]["artifactsCount"] == 1

    @pytest.mark.asyncio
    async def test_session_by_id(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            '{ session(sessionId: "s1") { sessionId status activeDomains } }',
            context_value=ctx,
        )
        assert result.errors is None
        data = result.data["session"]
        assert data["sessionId"] == "s1"
        assert data["activeDomains"] == ["sales-analytics.yaml"]

    @pytest.mark.asyncio
    async def test_session_not_found_returns_null(self):
        from constat.server.graphql import schema

        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = None

        result = await schema.execute(
            '{ session(sessionId: "no-such") { sessionId } }',
            context_value=ctx,
        )
        assert result.errors is None
        assert result.data["session"] is None

    @pytest.mark.asyncio
    async def test_session_shares_query(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed.session.datastore.get_shared_users.return_value = ["bob@example.com"]
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            '{ sessionShares(sessionId: "s1") }',
            context_value=ctx,
        )
        assert result.errors is None
        assert result.data["sessionShares"] == ["bob@example.com"]

    @pytest.mark.asyncio
    async def test_session_shares_forbidden(self):
        from constat.server.graphql import schema

        managed = _make_managed(user_id="other-user")
        ctx = _make_context(user_id="test-user")
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            '{ sessionShares(sessionId: "s1") }',
            context_value=ctx,
        )
        assert result.errors is not None

    @pytest.mark.asyncio
    async def test_active_domains_query(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed.active_domains = ["sales-analytics.yaml", "hr-reporting.yaml"]
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            '{ activeDomains(sessionId: "s1") }',
            context_value=ctx,
        )
        assert result.errors is None
        assert result.data["activeDomains"] == ["sales-analytics.yaml", "hr-reporting.yaml"]

    @pytest.mark.asyncio
    async def test_create_session_reconnect(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        with patch("constat.server.routes.learnings._ensure_user_domain_config"):
            result = await schema.execute(
                'mutation { createSession(sessionId: "s1") { sessionId userId status } }',
                context_value=ctx,
            )

        assert result.errors is None
        data = result.data["createSession"]
        assert data["sessionId"] == "s1"
        managed.touch.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_session_new(self):
        from constat.server.graphql import schema

        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = None
        ctx.session_manager.create_session.return_value = "new-session-id"

        result = await schema.execute(
            'mutation { createSession(sessionId: "new-session-id") { sessionId status } }',
            context_value=ctx,
        )
        assert result.errors is None
        data = result.data["createSession"]
        assert data["sessionId"] == "new-session-id"
        assert data["status"] == "IDLE"

    @pytest.mark.asyncio
    async def test_delete_session(self):
        from constat.server.graphql import schema

        ctx = _make_context()
        ctx.session_manager.delete_session.return_value = True

        result = await schema.execute(
            'mutation { deleteSession(sessionId: "s1") }',
            context_value=ctx,
        )
        assert result.errors is None
        assert result.data["deleteSession"] is True

    @pytest.mark.asyncio
    async def test_delete_session_not_found(self):
        from constat.server.graphql import schema

        ctx = _make_context()
        ctx.session_manager.delete_session.return_value = False

        result = await schema.execute(
            'mutation { deleteSession(sessionId: "no-such") }',
            context_value=ctx,
        )
        assert result.errors is not None

    @pytest.mark.asyncio
    async def test_toggle_public_sharing(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            'mutation { togglePublicSharing(sessionId: "s1", public: true) { status public shareUrl } }',
            context_value=ctx,
        )
        assert result.errors is None
        data = result.data["togglePublicSharing"]
        assert data["status"] == "updated"
        assert data["public"] is True
        assert "/s/s1" in data["shareUrl"]
        managed.session.datastore.set_public.assert_called_once_with(True)

    @pytest.mark.asyncio
    async def test_toggle_public_forbidden(self):
        from constat.server.graphql import schema

        managed = _make_managed(user_id="other-user")
        ctx = _make_context(user_id="test-user")
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            'mutation { togglePublicSharing(sessionId: "s1", public: true) { status } }',
            context_value=ctx,
        )
        assert result.errors is not None

    @pytest.mark.asyncio
    async def test_share_session(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            'mutation { shareSession(sessionId: "s1", email: "Bob@Example.com") { status shareUrl } }',
            context_value=ctx,
        )
        assert result.errors is None
        data = result.data["shareSession"]
        assert data["status"] == "shared"
        assert "/s/s1" in data["shareUrl"]
        managed.session.datastore.add_shared_user.assert_called_once_with("bob@example.com")

    @pytest.mark.asyncio
    async def test_share_session_forbidden(self):
        from constat.server.graphql import schema

        managed = _make_managed(user_id="other-user")
        ctx = _make_context(user_id="test-user")
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            'mutation { shareSession(sessionId: "s1", email: "bob@example.com") { status } }',
            context_value=ctx,
        )
        assert result.errors is not None

    @pytest.mark.asyncio
    async def test_remove_share(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            'mutation { removeShare(sessionId: "s1", sharedUserId: "bob@example.com") }',
            context_value=ctx,
        )
        assert result.errors is None
        assert result.data["removeShare"] is True
        managed.session.datastore.remove_shared_user.assert_called_once_with("bob@example.com")

    @pytest.mark.asyncio
    async def test_reset_context(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        with patch("constat.storage.history.SessionHistory") as mock_hist:
            result = await schema.execute(
                'mutation { resetContext(sessionId: "s1") }',
                context_value=ctx,
            )

        assert result.errors is None
        assert result.data["resetContext"] is True
        managed.api.reset_context.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_active_domains(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed.session.config = MagicMock()
        managed.session.config.load_domain.return_value = MagicMock()
        managed.session.doc_tools = None
        managed.session.clear_domain_apis = MagicMock()
        managed._domain_databases = set()
        managed.active_domains = []
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed
        ctx.session_manager.resolve_config.return_value = None

        with patch(
            "constat.server.routes.sessions._load_domains_into_session",
            return_value=(["sales-analytics.yaml"], []),
        ) as mock_load:
            with patch("constat.server.user_preferences.set_selected_domains"):
                result = await schema.execute(
                    'mutation { setActiveDomains(sessionId: "s1", domains: ["sales-analytics.yaml"]) }',
                    context_value=ctx,
                )

        assert result.errors is None
        mock_load.assert_called_once()
