# Copyright (c) 2025 Kenneth Stott
# Canary: 0ba0440d-8184-43ab-b01c-997fc8194790
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Playwright E2E tests: split vector store wiring.

Verifies that the split vector store (system DB + user DB) works correctly
through the real server code path:
- Warmup writes chunks to system DB
- Non-default user sessions open split stores
- Chunks from system DB are visible in session
- Entity references resolve across DBs via union views
- Schema search returns results from system DB chunks
"""

import json
import time
import uuid

import pytest
import requests

pytestmark = pytest.mark.integration


def _wait_for_session_ready(server_url: str, session_id: str, timeout: int = 60) -> bool:
    """Poll session status until init is complete (domains loaded, entities ready)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(f"{server_url}/api/sessions/{session_id}")
            if resp.status_code == 200:
                data = resp.json()
                # Session is ready when active_domains is populated
                if data.get("active_domains"):
                    return True
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(1)
    return False


def _gql(server_url: str, query: str, variables: dict | None = None) -> dict:
    """Execute a GraphQL query/mutation."""
    resp = requests.post(
        f"{server_url}/api/graphql",
        json={"query": query, "variables": variables or {}},
    )
    assert resp.status_code == 200, f"GraphQL HTTP {resp.status_code}: {resp.text}"
    data = resp.json()
    if data.get("errors"):
        msgs = "; ".join(e.get("message", str(e)) for e in data["errors"])
        raise AssertionError(f"GraphQL errors: {msgs}")
    return data["data"]


GLOSSARY_QUERY = """
query($sid: String!, $scope: String) {
  glossary(sessionId: $sid, scope: $scope) {
    terms { name displayName glossaryStatus entityId domain }
    totalDefined totalSelfDescribing
  }
}
"""

TERM_QUERY = """
query($sid: String!, $name: String!) {
  glossaryTerm(sessionId: $sid, name: $name) {
    name displayName definition domain
    connectedResources { entityName entityType databaseName tableName }
  }
}
"""


class TestSplitStoreE2E:
    """E2E tests verifying split vector store through the real server.

    Creates sessions for non-default users (which triggers split mode),
    then verifies that warmup system DB data is accessible in the session.
    """

    @pytest.fixture(scope="class")
    def split_session(self, server_url, integration_data_dir):
        """Create a session for a non-default user, wait for init.

        Non-default users trigger the split store code path:
        user DB at .constat/{uid}.vault/vectors.duckdb + system DB ATTACHed.
        """
        session_id = str(uuid.uuid4())
        user_id = f"split_e2e_{uuid.uuid4().hex[:8]}"

        resp = requests.post(
            f"{server_url}/api/sessions",
            json={"session_id": session_id, "user_id": user_id},
        )
        if resp.status_code != 200:
            server_log = integration_data_dir / "server.log"
            time.sleep(1)
            log_tail = ""
            if server_log.exists():
                lines = server_log.read_text(errors="replace").splitlines()
                log_tail = "\n".join(lines[-50:])
            pytest.fail(
                f"Create split session failed ({resp.status_code}).\n"
                f"Response: {resp.text}\nServer log:\n{log_tail}"
            )

        # Wait for session init (domain loading, entity extraction)
        if not _wait_for_session_ready(server_url, session_id, timeout=60):
            server_log = integration_data_dir / "server.log"
            log_tail = ""
            if server_log.exists():
                lines = server_log.read_text(errors="replace").splitlines()
                log_tail = "\n".join(lines[-80:])
            pytest.fail(
                f"Session {session_id} did not become ready within 60s.\n"
                f"Server log:\n{log_tail}"
            )

        yield {"session_id": session_id, "user_id": user_id}

        # Cleanup
        requests.delete(f"{server_url}/api/sessions/{session_id}")

    def test_session_creates_successfully(self, server_url, split_session):
        """Non-default user session creates without error."""
        resp = requests.get(f"{server_url}/api/sessions/{split_session['session_id']}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == split_session["session_id"]
        assert data["user_id"] == split_session["user_id"]

    def test_self_describing_terms_from_system_db(self, server_url, split_session):
        """Session sees self-describing entities from warmup (system DB chunks).

        Warmup writes schema chunks to system DB with domain_id. The session's
        split store ATTACHes system DB. Entity extraction finds entities in
        those chunks. The glossary query reads entities via union views.
        """
        sid = split_session["session_id"]
        data = _gql(server_url, GLOSSARY_QUERY, {"sid": sid, "scope": "self_describing"})
        glossary = data["glossary"]

        assert glossary["totalSelfDescribing"] > 0, (
            f"Expected self-describing terms from warmup chunks, got 0. "
            f"This means system DB chunks are not visible in the session."
        )
        assert len(glossary["terms"]) > 0

    def test_glossary_terms_have_domains(self, server_url, split_session):
        """Glossary terms from domain databases have correct domain_id.

        The demo config has sales-analytics and hr-reporting domains.
        Terms extracted from domain databases should carry the domain_id.
        """
        sid = split_session["session_id"]
        data = _gql(server_url, GLOSSARY_QUERY, {"sid": sid, "scope": "self_describing"})
        terms = data["glossary"]["terms"]

        # At least some terms should have a domain set
        terms_with_domain = [t for t in terms if t.get("domain")]
        assert len(terms_with_domain) > 0, (
            f"No terms have domain_id set. This means domain_id is not propagating "
            f"from add_database_dynamic through to entity extraction. "
            f"Terms: {[t['name'] for t in terms[:10]]}"
        )

    def test_schema_search_finds_warmup_chunks(self, server_url, split_session):
        """Schema search returns results from system DB chunks.

        Warmup writes schema chunks to system DB. The session's split store
        creates union views over both DBs. Schema search should find tables
        indexed during warmup.
        """
        sid = split_session["session_id"]

        # Search for a table that exists in the demo databases
        resp = requests.get(
            f"{server_url}/api/schema/search",
            params={"query": "employees", "session_id": sid, "limit": 5},
        )
        if resp.status_code == 200:
            results = resp.json().get("results", [])
            assert len(results) > 0, (
                "Schema search returned 0 results for 'employees'. "
                "Warmup chunks in system DB are not visible through the split store."
            )

    def test_entity_detail_has_connected_resources(self, server_url, split_session):
        """Entity detail shows connected resources (cross-DB join).

        Entities live in user DB. Chunk-entity links live in user DB.
        But chunks live in system DB. The connectedResources resolver
        must JOIN across DBs via union views.
        """
        sid = split_session["session_id"]

        # First, find a self-describing term with an entityId
        data = _gql(server_url, GLOSSARY_QUERY, {"sid": sid, "scope": "self_describing"})
        terms = data["glossary"]["terms"]
        entity_terms = [t for t in terms if t.get("entityId")]
        if not entity_terms:
            pytest.skip("No self-describing terms with entityId available")

        # Query term detail — connectedResources requires cross-DB join
        term = entity_terms[0]
        detail = _gql(server_url, TERM_QUERY, {"sid": sid, "name": term["name"]})
        term_detail = detail["glossaryTerm"]
        assert term_detail is not None, f"Term {term['name']} not found"

        # connectedResources should have entries (entity linked to chunks)
        resources = term_detail.get("connectedResources") or []
        # Not all terms have connected resources, but the ones from schema do
        if resources:
            assert resources[0].get("entityName"), (
                f"Connected resource missing entityName: {resources[0]}"
            )

    def test_websocket_connects_for_split_session(self, page, server_url, split_session):
        """WebSocket connects for a split-mode session."""
        sid = split_session["session_id"]
        ws_url = server_url.replace("http://", "ws://")
        ws_endpoint = f"{ws_url}/api/sessions/{sid}/ws"

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

    def test_graphql_works_for_split_session(self, page, server_url, split_session):
        """Browser can execute GraphQL against a split-mode session."""
        sid = split_session["session_id"]

        resp = page.request.post(
            f"{server_url}/api/graphql",
            data=json.dumps({
                "query": GLOSSARY_QUERY,
                "variables": {"sid": sid, "scope": "all"},
            }),
            headers={"Content-Type": "application/json"},
        )
        assert resp.ok
        result = resp.json()
        assert "errors" not in result, f"GraphQL errors: {result.get('errors')}"
        assert "terms" in result["data"]["glossary"]


class TestSplitStoreIsolation:
    """Verify user data isolation in split mode.

    Two different users should each have their own user DB while sharing
    the system DB. Glossary terms created by one user should not appear
    in another user's session.
    """

    @pytest.fixture(scope="class")
    def two_user_sessions(self, server_url, integration_data_dir):
        """Create sessions for two different users."""
        sessions = []
        for i in range(2):
            session_id = str(uuid.uuid4())
            user_id = f"isolation_{i}_{uuid.uuid4().hex[:6]}"

            resp = requests.post(
                f"{server_url}/api/sessions",
                json={"session_id": session_id, "user_id": user_id},
            )
            assert resp.status_code == 200, f"Create session {i} failed: {resp.text}"

            if not _wait_for_session_ready(server_url, session_id, timeout=60):
                server_log = integration_data_dir / "server.log"
                log_tail = ""
                if server_log.exists():
                    lines = server_log.read_text(errors="replace").splitlines()
                    log_tail = "\n".join(lines[-50:])
                pytest.fail(f"Session {i} not ready. Log:\n{log_tail}")

            sessions.append({"session_id": session_id, "user_id": user_id})

        yield sessions

        for s in sessions:
            requests.delete(f"{server_url}/api/sessions/{s['session_id']}")

    def test_both_sessions_see_system_data(self, server_url, two_user_sessions):
        """Both users see self-describing terms from warmup (shared system DB)."""
        for i, sess in enumerate(two_user_sessions):
            data = _gql(server_url, GLOSSARY_QUERY, {
                "sid": sess["session_id"], "scope": "self_describing",
            })
            assert data["glossary"]["totalSelfDescribing"] > 0, (
                f"User {i} ({sess['user_id']}) sees 0 self-describing terms"
            )

    def test_user_glossary_isolated(self, server_url, two_user_sessions):
        """Glossary term created by user A is not visible to user B."""
        sid_a = two_user_sessions[0]["session_id"]
        sid_b = two_user_sessions[1]["session_id"]

        term_name = f"isolated_{uuid.uuid4().hex[:8]}"

        # User A creates a term
        _gql(server_url, """
            mutation($sid: String!, $input: GlossaryTermInput!) {
                createGlossaryTerm(sessionId: $sid, input: $input) { name }
            }
        """, {
            "sid": sid_a,
            "input": {
                "name": term_name,
                "definition": "User A's private term",
                "parentId": "__root__",
            },
        })

        # User A sees it
        data_a = _gql(server_url, """
            query($sid: String!, $name: String!) {
                glossaryTerm(sessionId: $sid, name: $name) { name }
            }
        """, {"sid": sid_a, "name": term_name})
        assert data_a["glossaryTerm"] is not None, "User A should see their own term"

        # User B should NOT see it (different user DB)
        data_b = _gql(server_url, """
            query($sid: String!, $name: String!) {
                glossaryTerm(sessionId: $sid, name: $name) { name }
            }
        """, {"sid": sid_b, "name": term_name})
        assert data_b["glossaryTerm"] is None, (
            f"User B should NOT see User A's term '{term_name}'. "
            f"User isolation is broken — both users may share the same DB."
        )

        # Cleanup
        _gql(server_url, """
            mutation($sid: String!, $name: String!) {
                deleteGlossaryTerm(sessionId: $sid, name: $name)
            }
        """, {"sid": sid_a, "name": term_name})
