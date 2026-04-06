# Copyright (c) 2025 Kenneth Stott
# Canary: 2d610714-b7dd-4c67-87a1-956753b068bb
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Unit tests for GraphQL auth, config, and permissions resolvers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestSchemaStitching:
    def test_schema_has_login_mutation(self):
        from constat.server.graphql import schema

        sdl = schema.as_str()
        assert "login(" in sdl

    def test_schema_has_config_query(self):
        from constat.server.graphql import schema

        sdl = schema.as_str()
        assert "config:" in sdl or "config(" in sdl

    def test_schema_has_my_permissions_query(self):
        from constat.server.graphql import schema

        sdl = schema.as_str()
        assert "myPermissions" in sdl

    def test_schema_has_passkey_mutations(self):
        from constat.server.graphql import schema

        sdl = schema.as_str()
        assert "passkeyRegisterBegin(" in sdl
        assert "passkeyRegisterComplete(" in sdl
        assert "passkeyAuthBegin(" in sdl
        assert "passkeyAuthComplete(" in sdl

    def test_schema_has_logout_mutation(self):
        from constat.server.graphql import schema

        sdl = schema.as_str()
        assert "logout" in sdl


def _make_context(
    auth_disabled=True,
    local_users=None,
    config=None,
    personas_config=None,
    user_id="test-user",
):
    from constat.server.graphql.session_context import GraphQLContext

    mock_sm = MagicMock()
    mock_server_config = MagicMock()
    mock_server_config.auth_disabled = auth_disabled
    mock_server_config.local_users = local_users or {}
    mock_server_config.firebase_api_key = None
    mock_server_config.data_dir = Path("/tmp/test-graphql-auth")

    ctx = GraphQLContext(
        session_manager=mock_sm,
        server_config=mock_server_config,
        user_id=user_id,
        config=config,
    )
    # Simulate Strawberry setting request on BaseContext
    mock_request = MagicMock()
    mock_request.app.state.personas_config = personas_config
    ctx.request = mock_request
    return ctx


class TestAuthResolvers:
    @pytest.mark.asyncio
    async def test_login_local_valid(self):
        from constat.server.graphql import schema
        from constat.server.local_auth import hash_password

        pw_hash = hash_password("secret123")
        from constat.server.config import LocalUser

        local_users = {
            "alice": LocalUser(password_hash=pw_hash, email="alice@example.com")
        }
        ctx = _make_context(local_users=local_users)

        result = await schema.execute(
            'mutation { login(email: "alice", password: "secret123") { token userId email } }',
            context_value=ctx,
        )
        assert result.errors is None
        data = result.data["login"]
        assert data["userId"] == "alice"
        assert data["email"] == "alice@example.com"
        assert len(data["token"]) >= 20, f"Token too short to be valid: {data['token']!r}"

    @pytest.mark.asyncio
    async def test_login_local_invalid_password(self):
        from constat.server.graphql import schema
        from constat.server.local_auth import hash_password

        pw_hash = hash_password("secret123")
        from constat.server.config import LocalUser

        local_users = {
            "alice": LocalUser(password_hash=pw_hash, email="alice@example.com")
        }
        ctx = _make_context(local_users=local_users)

        result = await schema.execute(
            'mutation { login(email: "alice", password: "wrong") { token userId email } }',
            context_value=ctx,
        )
        assert result.errors is not None

    @pytest.mark.asyncio
    async def test_login_no_user_no_firebase(self):
        from constat.server.graphql import schema

        ctx = _make_context(local_users={})

        result = await schema.execute(
            'mutation { login(email: "nobody", password: "pw") { token userId email } }',
            context_value=ctx,
        )
        assert result.errors is not None

    @pytest.mark.asyncio
    async def test_logout_returns_true(self):
        from constat.server.graphql import schema

        ctx = _make_context()

        result = await schema.execute("mutation { logout }", context_value=ctx)
        assert result.errors is None
        assert result.data["logout"] is True

    @pytest.mark.asyncio
    async def test_config_returns_expected_shape(self):
        from constat.server.graphql import schema

        mock_config = MagicMock()
        mock_config.databases.keys.return_value = ["sales"]
        mock_config.apis.keys.return_value = ["rest-api"]
        mock_config.documents.keys.return_value = ["doc1"]
        mock_config.llm.provider = "anthropic"
        mock_config.llm.model = "claude-3"
        mock_config.execution.timeout_seconds = 30

        mock_routing = MagicMock()
        mock_routing.routes = {}
        mock_config.llm.get_task_routing.return_value = mock_routing

        ctx = _make_context(config=mock_config)

        result = await schema.execute(
            "{ config { databases apis documents llmProvider llmModel executionTimeout taskRouting } }",
            context_value=ctx,
        )
        assert result.errors is None
        data = result.data["config"]
        assert data["databases"] == ["sales"]
        assert data["apis"] == ["rest-api"]
        assert data["documents"] == ["doc1"]
        assert data["llmProvider"] == "anthropic"
        assert data["llmModel"] == "claude-3"
        assert data["executionTimeout"] == 30
        assert data["taskRouting"] == {}

    @pytest.mark.asyncio
    async def test_my_permissions_returns_perms(self):
        from constat.server.graphql import schema
        from constat.server.persona_config import PersonaDefinition, PersonasConfig

        personas_config = PersonasConfig(
            personas={
                "domain_user": PersonaDefinition(
                    visibility={"glossary": True},
                    writes={"glossary": False},
                    feedback={"flag_answers": True},
                )
            }
        )

        mock_server_config = MagicMock()
        mock_server_config.auth_disabled = True
        mock_server_config.permissions.get_user_permissions.return_value = MagicMock(
            persona="domain_user",
            domains=["sales"],
            databases=["db1"],
            documents=[],
            apis=[],
            skills=[],
            agents=[],
            rules=[],
            facts=[],
        )

        ctx = _make_context(personas_config=personas_config, user_id="u1")
        ctx.server_config = mock_server_config

        result = await schema.execute(
            "{ myPermissions { userId admin persona domains databases visibility writes feedback } }",
            context_value=ctx,
        )
        assert result.errors is None
        data = result.data["myPermissions"]
        assert data["userId"] == "u1"
        assert data["persona"] == "domain_user"
        assert data["domains"] == ["sales"]
        assert data["visibility"] == {"glossary": True}
        assert data["writes"] == {"glossary": False}
        assert data["feedback"] == {"flag_answers": True}

    @pytest.mark.asyncio
    async def test_my_permissions_no_personas_config(self):
        from constat.server.graphql import schema

        mock_server_config = MagicMock()
        mock_server_config.auth_disabled = True
        mock_server_config.permissions.get_user_permissions.return_value = MagicMock(
            persona="viewer",
            domains=[],
            databases=[],
            documents=[],
            apis=[],
            skills=[],
            agents=[],
            rules=[],
            facts=[],
        )

        ctx = _make_context(personas_config=None, user_id="u2")
        ctx.server_config = mock_server_config

        result = await schema.execute(
            "{ myPermissions { userId persona visibility writes feedback } }",
            context_value=ctx,
        )
        assert result.errors is None
        data = result.data["myPermissions"]
        assert data["visibility"] == {}
        assert data["writes"] == {}
        assert data["feedback"] == {}


class TestPasskeyResolvers:
    @pytest.mark.asyncio
    async def test_register_begin(self):
        from constat.server.graphql import schema

        ctx = _make_context()

        with patch(
            "constat.server.routes.passkey._load_credentials", return_value=[]
        ):
            result = await schema.execute(
                'mutation { passkeyRegisterBegin(userId: "test-user") { optionsJson } }',
                context_value=ctx,
            )
        assert result.errors is None
        data = result.data["passkeyRegisterBegin"]
        assert "optionsJson" in data
        opts = data["optionsJson"]
        assert "challenge" in opts

    @pytest.mark.asyncio
    async def test_register_complete_no_challenge(self):
        from constat.server.graphql import schema

        ctx = _make_context()

        result = await schema.execute(
            'mutation { passkeyRegisterComplete(userId: "no-one", credential: "{}") }',
            context_value=ctx,
        )
        assert result.errors is not None

    @pytest.mark.asyncio
    async def test_auth_begin_no_credentials(self):
        from constat.server.graphql import schema

        ctx = _make_context()

        with patch(
            "constat.server.routes.passkey._load_credentials", return_value=[]
        ):
            result = await schema.execute(
                'mutation { passkeyAuthBegin(userId: "test-user") { optionsJson } }',
                context_value=ctx,
            )
        assert result.errors is not None

    @pytest.mark.asyncio
    async def test_auth_complete_returns_vault_unlocked(self):
        """passkey_auth_complete returns vault_unlocked in AuthPayload."""
        from constat.server.graphql import schema

        ctx = _make_context()
        # vault_unlocked is in the schema and defaults to None/false
        result = await schema.execute(
            'mutation { passkeyAuthComplete(userId: "test-user", credential: "{}") { token userId email vaultUnlocked } }',
            context_value=ctx,
        )
        # This will error (no pending challenge), but validates the field exists in schema
        assert result.errors is not None

        # Verify schema has vaultUnlocked field on AuthPayload
        sdl = schema.as_str()
        assert "vaultUnlocked" in sdl

    @pytest.mark.asyncio
    async def test_auth_complete_no_challenge(self):
        from constat.server.graphql import schema

        ctx = _make_context()

        result = await schema.execute(
            'mutation { passkeyAuthComplete(userId: "test-user", credential: "{}") { token userId email } }',
            context_value=ctx,
        )
        assert result.errors is not None
