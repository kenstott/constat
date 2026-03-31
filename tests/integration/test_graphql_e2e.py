# Copyright (c) 2025 Kenneth Stott
# Canary: d1ab6fdb-2535-49c6-9703-224958986ed4
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""E2E GraphQL tests against a real Constat server.

These tests use the ``server_url`` fixture from conftest.py which starts
a full server process with ``auth_disabled=true`` and waits for warmup.
"""

import json

import pytest
import requests
import websocket as ws_client  # websocket-client library


class TestGraphQLIntrospection:
    def test_graphql_introspection_via_server(self, server_url):
        """POST /api/graphql with a simple introspection query."""
        resp = requests.post(
            f"{server_url}/api/graphql",
            json={"query": "{ __typename }"},
            timeout=10,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["data"]["__typename"] == "Query"

    def test_schema_types_via_server(self, server_url):
        """Full schema introspection returns Query, Mutation, Subscription."""
        resp = requests.post(
            f"{server_url}/api/graphql",
            json={"query": "{ __schema { types { name } } }"},
            timeout=10,
        )
        assert resp.status_code == 200
        type_names = [t["name"] for t in resp.json()["data"]["__schema"]["types"]]
        assert "Query" in type_names
        assert "Mutation" in type_names
        assert "Subscription" in type_names


class TestGraphQLSubscriptionTransport:
    def test_graphql_ws_handshake(self, server_url):
        """Connect via WebSocket with graphql-transport-ws sub-protocol.

        Sends ``connection_init`` and expects ``connection_ack`` per the
        graphql-ws protocol spec.
        """
        ws_url = server_url.replace("http://", "ws://") + "/api/graphql"
        conn = ws_client.create_connection(
            ws_url,
            subprotocols=["graphql-transport-ws"],
            timeout=10,
        )
        try:
            # Send connection_init
            conn.send(json.dumps({"type": "connection_init"}))
            # Receive connection_ack
            raw = conn.recv()
            msg = json.loads(raw)
            assert msg["type"] == "connection_ack"
        finally:
            conn.close()
