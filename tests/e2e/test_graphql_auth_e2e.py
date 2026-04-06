# Copyright (c) 2025 Kenneth Stott
# Canary: 43adb28e-92f5-433d-919f-3721fd06127e
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""E2E GraphQL auth tests against a real Constat server.

Uses the ``server_url`` fixture from conftest.py which starts
a full server process with ``auth_disabled=true`` and waits for warmup.
"""

from __future__ import annotations
import pytest
import requests

pytestmark = pytest.mark.e2e


class TestAuthE2E:
    def test_config_via_graphql(self, server_url):
        """Config query returns valid data from running server."""
        resp = requests.post(
            f"{server_url}/api/graphql",
            json={"query": "{ config { databases apis documents llmProvider llmModel executionTimeout taskRouting } }"},
            timeout=10,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("errors") is None
        config = data["data"]["config"]
        assert isinstance(config["databases"], list)
        assert isinstance(config["apis"], list)
        assert isinstance(config["documents"], list)
        assert isinstance(config["llmProvider"], str)
        assert isinstance(config["llmModel"], str)
        assert isinstance(config["executionTimeout"], int)

    def test_permissions_via_graphql(self, server_url):
        """myPermissions query returns default user perms (auth disabled)."""
        resp = requests.post(
            f"{server_url}/api/graphql",
            json={"query": "{ myPermissions { userId admin persona domains } }"},
            timeout=10,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("errors") is None
        perms = data["data"]["myPermissions"]
        assert perms["userId"] == "default"
        assert isinstance(perms["admin"], bool)
        assert isinstance(perms["persona"], str)
        assert isinstance(perms["domains"], list)

    def test_logout_via_graphql(self, server_url):
        """Logout mutation returns true."""
        resp = requests.post(
            f"{server_url}/api/graphql",
            json={"query": "mutation { logout }"},
            timeout=10,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("errors") is None
        assert data["data"]["logout"] is True
