# Copyright (c) 2025 Kenneth Stott
# Canary: 59cef366-eed0-45b0-be78-0c32b8fb5bd0
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Playwright integration tests: browser connection and session lifecycle.

Validates that the frontend can connect to the backend, create a session,
and establish a WebSocket connection for real-time communication.
"""

import json
import uuid

import pytest
import requests


pytestmark = pytest.mark.integration


def _create_session(base_url: str, user_id: str = "default") -> str:
    """Create a session and return the session_id."""
    session_id = str(uuid.uuid4())
    body = {"session_id": session_id}
    if user_id != "default":
        body["user_id"] = user_id
    resp = requests.post(f"{base_url}/api/sessions", json=body)
    assert resp.status_code == 200, f"Create session failed ({resp.status_code}): {resp.text}"
    return resp.json()["session_id"]


class TestServerConnection:
    """Verify the server responds to API requests."""

    def test_server_responds(self, server_url):
        """Server returns a valid HTTP response."""
        resp = requests.get(f"{server_url}/api/sessions")
        # auth_disabled=true → should not be 401
        assert resp.status_code == 200

    def test_permissions_endpoint(self, server_url):
        """Permissions endpoint returns valid JSON."""
        resp = requests.get(f"{server_url}/api/users/me/permissions")
        assert resp.status_code == 200
        data = resp.json()
        assert "persona" in data

    def test_create_session(self, server_url):
        """Can create a new session via POST."""
        session_id = _create_session(server_url)
        assert session_id


class TestBrowserConnection:
    """Verify Playwright can load the UI and interact with the server."""

    def test_api_reachable_from_browser(self, page, server_url):
        """Browser can reach the API and get a JSON response."""
        resp = page.request.get(f"{server_url}/api/users/me/permissions")
        assert resp.ok
        body = resp.json()
        assert "persona" in body

    def test_session_creation_from_browser(self, page, server_url):
        """Browser can create a session via the API."""
        session_id = str(uuid.uuid4())
        resp = page.request.post(
            f"{server_url}/api/sessions",
            data=json.dumps({"session_id": session_id}),
            headers={"Content-Type": "application/json"},
        )
        assert resp.ok
        body = resp.json()
        assert "session_id" in body

    def test_websocket_connection(self, page, server_url):
        """Browser can open a WebSocket to the session endpoint."""
        session_id = _create_session(server_url)

        ws_url = server_url.replace("http://", "ws://")
        ws_endpoint = f"{ws_url}/api/sessions/{session_id}/ws"

        # Use page.evaluate to test WebSocket from browser context
        result = page.evaluate(
            """async (wsUrl) => {
                return new Promise((resolve, reject) => {
                    const ws = new WebSocket(wsUrl);
                    const timeout = setTimeout(() => {
                        ws.close();
                        reject(new Error('WebSocket connection timeout'));
                    }, 10000);

                    ws.onopen = () => {
                        clearTimeout(timeout);
                        ws.close();
                        resolve({ connected: true });
                    };
                    ws.onerror = (e) => {
                        clearTimeout(timeout);
                        resolve({ connected: false, error: 'WebSocket error' });
                    };
                    ws.onclose = (e) => {
                        clearTimeout(timeout);
                        if (e.code !== 1000 && e.code !== 1005) {
                            resolve({ connected: false, error: `Closed: ${e.code} ${e.reason}` });
                        }
                    };
                });
            }""",
            ws_endpoint,
        )
        assert result["connected"], f"WebSocket failed: {result.get('error')}"


class TestSplitModeSession:
    """Verify session creation works with split vector store (non-default user).

    When user_id != 'default' and .constat/vectors.duckdb exists, the server
    creates a per-user vault with split mode: user DB + ATTACHed system DB.
    This is the path real authenticated users take.
    """

    def test_create_session_non_default_user(self, server_url, integration_data_dir):
        """Session creation succeeds for a non-default user (split mode)."""
        session_id = str(uuid.uuid4())
        resp = requests.post(
            f"{server_url}/api/sessions",
            json={"session_id": session_id, "user_id": "integration_test_user"},
        )
        if resp.status_code != 200:
            # Read server log for actual traceback
            server_log = integration_data_dir / "server.log"
            import time; time.sleep(1)  # Give server time to write
            log_tail = ""
            if server_log.exists():
                lines = server_log.read_text(errors="replace").splitlines()
                log_tail = "\n".join(lines[-50:])
            pytest.fail(f"Create session failed ({resp.status_code}).\nServer log:\n{log_tail}")
        assert resp.json()["session_id"]

    def test_websocket_non_default_user(self, page, server_url):
        """WebSocket connects for a non-default user (split mode)."""
        session_id = _create_session(server_url, user_id="integration_test_user_ws")

        ws_url = server_url.replace("http://", "ws://")
        ws_endpoint = f"{ws_url}/api/sessions/{session_id}/ws"

        result = page.evaluate(
            """async (wsUrl) => {
                return new Promise((resolve, reject) => {
                    const ws = new WebSocket(wsUrl);
                    const timeout = setTimeout(() => {
                        ws.close();
                        reject(new Error('WebSocket connection timeout'));
                    }, 10000);

                    ws.onopen = () => {
                        clearTimeout(timeout);
                        ws.close();
                        resolve({ connected: true });
                    };
                    ws.onerror = (e) => {
                        clearTimeout(timeout);
                        resolve({ connected: false, error: 'WebSocket error' });
                    };
                    ws.onclose = (e) => {
                        clearTimeout(timeout);
                        if (e.code !== 1000 && e.code !== 1005) {
                            resolve({ connected: false, error: `Closed: ${e.code} ${e.reason}` });
                        }
                    };
                });
            }""",
            ws_endpoint,
        )
        assert result["connected"], f"WebSocket failed: {result.get('error')}"
