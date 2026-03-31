# Copyright (c) 2025 Kenneth Stott
# Canary: cf645fe7-e64c-4903-9893-cb3ebfb24ce5
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Unit tests for GraphQL infrastructure — schema, context, pub/sub."""

import asyncio
from unittest.mock import MagicMock

import pytest


class TestGraphQLSchema:
    def test_schema_has_query_mutation_subscription(self):
        from constat.server.graphql import schema

        sdl = schema.as_str()
        assert "type Query" in sdl
        assert "type Mutation" in sdl
        assert "type Subscription" in sdl

    def test_schema_has_glossary_query(self):
        from constat.server.graphql import schema

        sdl = schema.as_str()
        assert "glossary(" in sdl
        assert "sessionId:" in sdl

    def test_schema_has_glossary_changed_subscription(self):
        from constat.server.graphql import schema

        sdl = schema.as_str()
        assert "glossaryChanged(" in sdl

    def test_schema_stitching_merges_types(self):
        """Verify merge_types produced combined root types."""
        from constat.server.graphql import Query, Mutation, Subscription

        assert hasattr(Query, "__strawberry_definition__")
        assert hasattr(Mutation, "__strawberry_definition__")
        assert hasattr(Subscription, "__strawberry_definition__")


class TestTypedContext:
    @pytest.mark.asyncio
    async def test_context_returns_graphql_context_dataclass(self):
        from constat.server.graphql import get_context
        from constat.server.graphql.session_context import GraphQLContext

        mock_request = MagicMock()
        mock_request.app.state.session_manager = MagicMock()
        mock_request.app.state.server_config = MagicMock()
        mock_request.app.state.server_config.auth_disabled = True
        mock_request.headers = {}

        ctx = await get_context(request=mock_request)
        assert isinstance(ctx, GraphQLContext)

    @pytest.mark.asyncio
    async def test_context_session_manager(self):
        from constat.server.graphql import get_context

        mock_request = MagicMock()
        mock_request.app.state.session_manager = MagicMock()
        mock_request.app.state.server_config = MagicMock()
        mock_request.app.state.server_config.auth_disabled = True
        mock_request.headers = {}

        ctx = await get_context(request=mock_request)
        assert ctx.session_manager is mock_request.app.state.session_manager

    @pytest.mark.asyncio
    async def test_context_server_config(self):
        from constat.server.graphql import get_context

        mock_request = MagicMock()
        mock_request.app.state.session_manager = MagicMock()
        mock_request.app.state.server_config = MagicMock()
        mock_request.app.state.server_config.auth_disabled = True
        mock_request.headers = {}

        ctx = await get_context(request=mock_request)
        assert ctx.server_config is mock_request.app.state.server_config

    @pytest.mark.asyncio
    async def test_context_user_id_default_when_auth_disabled(self):
        from constat.server.graphql import get_context

        mock_request = MagicMock()
        mock_request.app.state.session_manager = MagicMock()
        mock_request.app.state.server_config = MagicMock()
        mock_request.app.state.server_config.auth_disabled = True
        mock_request.headers = {}

        ctx = await get_context(request=mock_request)
        assert ctx.user_id == "default"

    @pytest.mark.asyncio
    async def test_context_user_id_admin_token(self):
        from constat.server.graphql import get_context

        mock_request = MagicMock()
        mock_request.app.state.session_manager = MagicMock()
        mock_request.app.state.server_config = MagicMock()
        mock_request.app.state.server_config.auth_disabled = False
        mock_request.app.state.server_config.admin_token = "secret-admin"
        mock_request.headers = {"authorization": "Bearer secret-admin"}

        ctx = await get_context(request=mock_request)
        assert ctx.user_id == "admin"

    @pytest.mark.asyncio
    async def test_context_ws_fallback(self):
        from constat.server.graphql import get_context

        mock_ws = MagicMock()
        mock_ws.app.state.session_manager = MagicMock()
        mock_ws.app.state.server_config = MagicMock()
        mock_ws.app.state.server_config.auth_disabled = True
        mock_ws.headers = {}

        ctx = await get_context(ws=mock_ws)
        assert ctx.session_manager is mock_ws.app.state.session_manager
        assert ctx.user_id == "default"


class TestAuthenticateToken:
    def test_auth_disabled_returns_default(self):
        from constat.server.auth import authenticate_token

        server_config = MagicMock()
        server_config.auth_disabled = True
        assert authenticate_token(None, server_config) == "default"

    def test_admin_token(self):
        from constat.server.auth import authenticate_token

        server_config = MagicMock()
        server_config.auth_disabled = False
        server_config.admin_token = "my-admin-token"
        assert authenticate_token("my-admin-token", server_config) == "admin"

    def test_missing_token_raises_401(self):
        from constat.server.auth import authenticate_token
        from fastapi import HTTPException

        server_config = MagicMock()
        server_config.auth_disabled = False
        with pytest.raises(HTTPException) as exc_info:
            authenticate_token(None, server_config)
        assert exc_info.value.status_code == 401


class TestGlossaryPubSub:
    def _make_sm(self):
        from constat.server.session_manager import SessionManager

        config = MagicMock()
        config.data_dir = "/tmp/test"
        server_config = MagicMock()
        server_config.tiered_config = None
        return SessionManager(config, server_config)

    def test_subscribe_publish(self):
        sm = self._make_sm()
        queue = sm.subscribe_glossary("s1")
        event = {"action": "CREATED", "term": "test"}
        sm.publish_glossary_change("s1", event)
        assert queue.get_nowait() == event

    def test_unsubscribe(self):
        sm = self._make_sm()
        queue = sm.subscribe_glossary("s1")
        sm.unsubscribe_glossary("s1", queue)
        sm.publish_glossary_change("s1", {"action": "CREATED"})
        assert queue.empty()

    def test_publish_to_multiple_subscribers(self):
        sm = self._make_sm()
        q1 = sm.subscribe_glossary("s1")
        q2 = sm.subscribe_glossary("s1")
        q3 = sm.subscribe_glossary("s1")
        event = {"action": "UPDATED"}
        sm.publish_glossary_change("s1", event)
        assert q1.get_nowait() == event
        assert q2.get_nowait() == event
        assert q3.get_nowait() == event

    def test_publish_to_wrong_session(self):
        sm = self._make_sm()
        queue = sm.subscribe_glossary("s1")
        sm.publish_glossary_change("s2", {"action": "CREATED"})
        assert queue.empty()

    def test_unsubscribe_nonexistent(self):
        sm = self._make_sm()
        queue = asyncio.Queue()
        # Should not raise
        sm.unsubscribe_glossary("s1", queue)
