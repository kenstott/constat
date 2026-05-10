# Copyright (c) 2025 Kenneth Stott
# Canary: 52f18995-b1b1-47a8-8846-b74330e689a7
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Integration tests for GraphQL endpoint via TestClient.

Uses the shared ``graphql_client`` fixture from ``conftest.py``.
"""

from __future__ import annotations
import pytest
from unittest.mock import MagicMock

from fastapi import FastAPI
from starlette.testclient import TestClient

from constat.server.graphql import graphql_router


class TestGraphQLEndpoint:
    def test_introspection(self, graphql_client):
        resp = graphql_client.post("/api/graphql", json={"query": "{ __typename }"})
        assert resp.status_code == 200
        assert resp.json()["data"]["__typename"] == "Query"

    def test_glossary_query_requires_valid_session(self, graphql_client):
        graphql_client.app.state.session_manager.get_session_or_none.return_value = None
        resp = graphql_client.post(
            "/api/graphql",
            json={"query": '{ glossary(sessionId: "invalid") { terms { name } } }'},
        )
        assert resp.status_code == 200
        assert resp.json().get("errors")

    def test_schema_introspection_types(self, graphql_client):
        query = "{ __schema { types { name } } }"
        resp = graphql_client.post("/api/graphql", json={"query": query})
        assert resp.status_code == 200
        type_names = [t["name"] for t in resp.json()["data"]["__schema"]["types"]]
        assert "Query" in type_names
        assert "Mutation" in type_names
        assert "Subscription" in type_names


class TestGraphQLAuth:
    def test_auth_disabled_accepts_no_header(self, graphql_client):
        """When auth_disabled=True, requests without auth header succeed."""
        resp = graphql_client.post("/api/graphql", json={"query": "{ __typename }"})
        assert resp.status_code == 200
        assert resp.json()["data"]["__typename"] == "Query"

    def test_auth_enabled_allows_introspection_without_header(self):
        """When auth_disabled=False, introspection succeeds without auth.

        Individual resolvers enforce auth — the context layer allows
        unauthenticated access (user_id=None) so login mutation works.
        """
        app = FastAPI()
        app.state.session_manager = MagicMock()
        server_config = MagicMock()
        server_config.auth_disabled = False
        server_config.admin_token = None
        app.state.server_config = server_config
        app.state.config = None
        app.include_router(graphql_router, prefix="/api/graphql")

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/api/graphql", json={"query": "{ __typename }"})
        assert resp.status_code == 200
        assert resp.json()["data"]["__typename"] == "Query"

    def test_auth_enabled_accepts_admin_token(self):
        """When auth_disabled=False, admin token in Bearer header succeeds."""
        app = FastAPI()
        app.state.session_manager = MagicMock()
        server_config = MagicMock()
        server_config.auth_disabled = False
        server_config.admin_token = "test-admin-token"
        app.state.server_config = server_config
        app.include_router(graphql_router, prefix="/api/graphql")

        client = TestClient(app)
        resp = client.post(
            "/api/graphql",
            json={"query": "{ __typename }"},
            headers={"Authorization": "Bearer test-admin-token"},
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["__typename"] == "Query"


class TestGraphQLWebSocket:
    def test_ws_subscription_handshake(self, graphql_client):
        """Verify graphql-ws protocol handshake (connection_init -> connection_ack)."""
        with graphql_client.websocket_connect(
            "/api/graphql", subprotocols=["graphql-transport-ws"]
        ) as ws:
            ws.send_json({"type": "connection_init"})
            msg = ws.receive_json()
            assert msg["type"] == "connection_ack"
