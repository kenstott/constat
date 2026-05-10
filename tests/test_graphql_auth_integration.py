# Copyright (c) 2025 Kenneth Stott
# Canary: 102d6ab5-1e77-47ab-9cdf-33b5ed0a63ae
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Integration tests for GraphQL auth resolvers via TestClient.

Uses the shared ``graphql_client`` fixture from ``conftest.py`` plus
custom apps with auth enabled for token-based tests.
"""

from __future__ import annotations
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from constat.server.graphql import graphql_router


def _make_auth_app(
    local_users=None,
    firebase_api_key=None,
    config=None,
    personas_config=None,
):
    """Create a FastAPI app with auth enabled and optional local users."""
    from constat.server.config import ServerConfig

    app = FastAPI()
    app.state.session_manager = MagicMock()
    server_config = MagicMock(spec=ServerConfig)
    server_config.auth_disabled = False
    server_config.admin_token = "test-admin-token"
    server_config.local_users = local_users or {}
    server_config.firebase_api_key = firebase_api_key
    server_config.firebase_project_id = None
    server_config.data_dir = Path("/tmp/test-graphql-auth-int")
    app.state.server_config = server_config
    app.state.config = config
    app.state.personas_config = personas_config
    app.include_router(graphql_router, prefix="/api/graphql")
    return app


class TestAuthIntegration:
    def test_login_mutation_returns_token(self):
        from constat.server.config import LocalUser
        from constat.server.local_auth import hash_password

        pw_hash = hash_password("secret123")
        local_users = {"alice": LocalUser(password_hash=pw_hash, email="alice@example.com")}
        app = _make_auth_app(local_users=local_users)

        client = TestClient(app)
        resp = client.post(
            "/api/graphql",
            json={
                "query": 'mutation { login(email: "alice", password: "secret123") { token userId email } }',
            },
            headers={"Authorization": "Bearer test-admin-token"},
        )
        assert resp.status_code == 200
        data = resp.json()["data"]["login"]
        assert data["userId"] == "alice"
        assert data["email"] == "alice@example.com"
        assert len(data["token"]) >= 20, f"Token too short to be valid: {data['token']!r}"

    def test_login_then_query_with_token(self):
        """Login via mutation, then use returned token for subsequent query."""
        from constat.server.config import LocalUser
        from constat.server.local_auth import hash_password

        pw_hash = hash_password("pass456")
        local_users = {"bob": LocalUser(password_hash=pw_hash, email="bob@example.com")}

        mock_config = MagicMock()
        mock_config.databases.keys.return_value = ["testdb"]
        mock_config.apis.keys.return_value = []
        mock_config.documents.keys.return_value = []
        mock_config.llm.provider = "anthropic"
        mock_config.llm.model = "claude-3"
        mock_config.execution.timeout_seconds = 30
        mock_routing = MagicMock()
        mock_routing.routes = {}
        mock_config.llm.get_task_routing.return_value = mock_routing

        app = _make_auth_app(local_users=local_users, config=mock_config)

        client = TestClient(app)

        # Step 1: login (use admin token to authenticate the request itself)
        login_resp = client.post(
            "/api/graphql",
            json={
                "query": 'mutation { login(email: "bob", password: "pass456") { token userId } }',
            },
            headers={"Authorization": "Bearer test-admin-token"},
        )
        token = login_resp.json()["data"]["login"]["token"]

        # Step 2: use the returned local token for a config query
        config_resp = client.post(
            "/api/graphql",
            json={"query": "{ config { databases llmProvider } }"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert config_resp.status_code == 200
        config_data = config_resp.json()["data"]["config"]
        assert config_data["databases"] == ["testdb"]
        assert config_data["llmProvider"] == "anthropic"

    def test_unauthenticated_permissions_rejected(self):
        """myPermissions query without auth token returns GraphQL error."""
        app = _make_auth_app()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/api/graphql",
            json={"query": "{ myPermissions { userId persona } }"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["data"] is None or body["data"].get("myPermissions") is None
        assert any("Authentication required" in e["message"] for e in body["errors"])

    def test_config_query_returns_data(self, graphql_client):
        """Config query returns expected fields (auth disabled)."""
        mock_config = MagicMock()
        mock_config.databases.keys.return_value = ["db1", "db2"]
        mock_config.apis.keys.return_value = ["api1"]
        mock_config.documents.keys.return_value = ["doc1"]
        mock_config.llm.provider = "openai"
        mock_config.llm.model = "gpt-4"
        mock_config.execution.timeout_seconds = 60
        mock_routing = MagicMock()
        mock_routing.routes = {}
        mock_config.llm.get_task_routing.return_value = mock_routing

        graphql_client.app.state.config = mock_config

        resp = graphql_client.post(
            "/api/graphql",
            json={"query": "{ config { databases apis documents llmProvider llmModel executionTimeout } }"},
        )
        assert resp.status_code == 200
        data = resp.json()["data"]["config"]
        assert data["databases"] == ["db1", "db2"]
        assert data["apis"] == ["api1"]
        assert data["documents"] == ["doc1"]
        assert data["llmProvider"] == "openai"
        assert data["llmModel"] == "gpt-4"
        assert data["executionTimeout"] == 60

    def test_config_query_no_sensitive_data(self, graphql_client):
        """Config query does not expose API keys or secrets."""
        from constat.server.graphql import schema

        sdl = schema.as_str()
        # ServerConfigType should not have fields for secrets
        assert "apiKey" not in sdl.split("type ServerConfigType")[1].split("}")[0] if "type ServerConfigType" in sdl else True
        assert "firebaseApiKey" not in sdl
        assert "adminToken" not in sdl


class TestPasskeyIntegration:
    def test_register_begin_via_graphql(self, graphql_client):
        """Passkey register begin returns JSON options via GraphQL."""
        from unittest.mock import patch

        with patch("constat.server.routes.passkey._load_credentials", return_value=[]):
            resp = graphql_client.post(
                "/api/graphql",
                json={
                    "query": 'mutation { passkeyRegisterBegin(userId: "test-user") { optionsJson } }',
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("errors") is None
        opts = data["data"]["passkeyRegisterBegin"]["optionsJson"]
        assert "challenge" in opts
        assert "rp" in opts
