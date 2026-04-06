# Copyright (c) 2025 Kenneth Stott
# Canary: 0ff1c072-93e3-4d10-b102-acbc83a083ec
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Playwright browser-based glossary tests.

These tests require a running Playwright browser context (page fixture).
HTTP-only GraphQL tests remain in tests/integration/test_glossary.py.
"""

from __future__ import annotations
import uuid

import pytest
import requests

pytestmark = pytest.mark.e2e


def _find_parent_id(server_url: str, session_id: str) -> str:
    """Find an entity name usable as parent_id via GraphQL."""
    import requests as _requests
    result_resp = _requests.post(f"{server_url}/api/graphql", json={
        "query": """
            query($sid: String!) {
                glossary(sessionId: $sid, scope: "self_describing") {
                    terms { entityId name }
                }
            }
        """,
        "variables": {"sid": session_id},
    })
    assert result_resp.status_code == 200
    result = result_resp.json()
    terms = result.get("data", {}).get("glossary", {}).get("terms", [])
    for t in terms:
        if t.get("entityId"):
            return t["entityId"]
        if t.get("name"):
            return t["name"]
    return ""


class TestGlossaryBrowser:
    """Test glossary via Playwright browser interaction."""

    def test_graphql_query_from_browser(self, page, server_url, session_id):
        """Browser can execute a GraphQL query via Playwright API."""
        import json as _json
        resp = page.request.post(
            f"{server_url}/api/graphql",
            data=_json.dumps({
                "query": """query($sid: String!) {
                    glossary(sessionId: $sid) {
                        terms { name glossaryStatus }
                        totalDefined
                        totalSelfDescribing
                    }
                }""",
                "variables": {"sid": session_id},
            }),
            headers={"Content-Type": "application/json"},
        )
        assert resp.ok
        result = resp.json()
        assert "errors" not in result, f"GraphQL errors: {result.get('errors')}"
        assert "terms" in result["data"]["glossary"]

    def test_graphql_mutation_from_browser(self, page, server_url, session_id):
        """Browser can execute a GraphQL mutation via Playwright API."""
        import json as _json
        resp = page.request.post(
            f"{server_url}/api/graphql",
            data=_json.dumps({
                "query": """mutation($sid: String!, $s: String!, $v: String!, $o: String!) {
                    createRelationship(sessionId: $sid, subject: $s, verb: $v, object: $o) {
                        id verb userEdited
                    }
                }""",
                "variables": {"sid": session_id, "s": "product", "v": "BELONGS_TO", "o": "category"},
            }),
            headers={"Content-Type": "application/json"},
        )
        assert resp.ok
        result = resp.json()
        assert "errors" not in result, f"GraphQL errors: {result.get('errors')}"
        assert result["data"]["createRelationship"]["verb"] == "BELONGS_TO"

    def test_generate_glossary_from_browser(self, page, server_url, session_id):
        """Browser can trigger glossary generation via GraphQL mutation."""
        import json as _json
        resp = page.request.post(
            f"{server_url}/api/graphql",
            data=_json.dumps({
                "query": """mutation($sid: String!) {
                    generateGlossary(sessionId: $sid) {
                        status message
                    }
                }""",
                "variables": {"sid": session_id},
            }),
            headers={"Content-Type": "application/json"},
        )
        assert resp.ok
        result = resp.json()
        assert "errors" not in result, f"GraphQL errors: {result.get('errors')}"
        assert result["data"]["generateGlossary"]["status"] == "generating"

    def test_suggest_taxonomy_from_browser(self, page, server_url, session_id):
        """Browser can request taxonomy suggestions via GraphQL mutation."""
        import json as _json
        resp = page.request.post(
            f"{server_url}/api/graphql",
            data=_json.dumps({
                "query": """mutation($sid: String!) {
                    suggestTaxonomy(sessionId: $sid) {
                        suggestions { child parent parentVerb confidence reason }
                        message
                    }
                }""",
                "variables": {"sid": session_id},
            }),
            headers={"Content-Type": "application/json"},
        )
        assert resp.ok
        result = resp.json()
        assert "errors" not in result, f"GraphQL errors: {result.get('errors')}"
        assert isinstance(result["data"]["suggestTaxonomy"]["suggestions"], list)

    def test_rename_term_from_browser(self, page, server_url, session_id):
        """Browser can rename a glossary term via GraphQL mutation."""
        import json as _json

        parent_id = _find_parent_id(server_url, session_id)
        if not parent_id:
            pytest.fail("Test data precondition not met — seed required data before running this test")

        name = f"bw_rename_{uuid.uuid4().hex[:6]}"
        new_name = f"bw_renamed_{uuid.uuid4().hex[:6]}"

        # Create term
        resp = page.request.post(
            f"{server_url}/api/graphql",
            data=_json.dumps({
                "query": """mutation($sid: String!, $input: GlossaryTermInput!) {
                    createGlossaryTerm(sessionId: $sid, input: $input) { name }
                }""",
                "variables": {"sid": session_id, "input": {"name": name, "definition": "Browser rename test", "parentId": parent_id}},
            }),
            headers={"Content-Type": "application/json"},
        )
        assert resp.ok

        # Rename
        resp = page.request.post(
            f"{server_url}/api/graphql",
            data=_json.dumps({
                "query": """mutation($sid: String!, $n: String!, $nn: String!) {
                    renameTerm(sessionId: $sid, name: $n, newName: $nn) {
                        oldName newName displayName relationshipsUpdated
                    }
                }""",
                "variables": {"sid": session_id, "n": name, "nn": new_name},
            }),
            headers={"Content-Type": "application/json"},
        )
        assert resp.ok
        result = resp.json()
        assert "errors" not in result, f"GraphQL errors: {result.get('errors')}"
        assert result["data"]["renameTerm"]["newName"].lower() == new_name.lower()

        # Cleanup
        page.request.post(
            f"{server_url}/api/graphql",
            data=_json.dumps({
                "query": "mutation($sid: String!, $n: String!) { deleteGlossaryTerm(sessionId: $sid, name: $n) }",
                "variables": {"sid": session_id, "n": new_name},
            }),
            headers={"Content-Type": "application/json"},
        )

    def test_graphql_subscription_websocket(self, page, server_url):
        """Browser can open a GraphQL subscription WebSocket and get connection_ack."""
        # Navigate to server first so WebSocket is same-origin
        page.goto(server_url)
        ws_url = server_url.replace("http://", "ws://")
        result = page.evaluate(
            """async (wsUrl) => {
                return new Promise((resolve) => {
                    const client = new WebSocket(wsUrl, 'graphql-transport-ws');
                    const timeout = setTimeout(() => {
                        client.close();
                        resolve({ connected: false, error: 'timeout' });
                    }, 10000);

                    client.onopen = () => {
                        client.send(JSON.stringify({ type: 'connection_init' }));
                    };
                    client.onmessage = (event) => {
                        const msg = JSON.parse(event.data);
                        if (msg.type === 'connection_ack') {
                            clearTimeout(timeout);
                            client.close();
                            resolve({ connected: true, protocol: 'graphql-transport-ws' });
                        }
                    };
                    client.onerror = () => {
                        clearTimeout(timeout);
                        resolve({ connected: false, error: 'ws_error' });
                    };
                });
            }""",
            f"{ws_url}/api/graphql",
        )
        assert result["connected"], f"Subscription WS failed: {result.get('error')}"

    def test_subscription_receives_generation_events(self, page, server_url, session_id):
        """Subscribe to glossaryChanged and verify generation lifecycle events arrive."""
        page.goto(server_url)
        ws_url = server_url.replace("http://", "ws://")
        gql_url = f"{ws_url}/api/graphql"
        http_url = f"{server_url}/api/graphql"

        # Phases: skip LLM-heavy work to keep the test fast; lifecycle events still fire.
        skip_phases = {
            "early_relationships": False,
            "definitions": False,
            "late_relationships": False,
            "clustering": False,
        }
        result = page.evaluate(
            """async ([wsUrl, httpUrl, sid, phases]) => {
                const events = [];
                return new Promise((resolve) => {
                    const timeout = setTimeout(() => {
                        resolve({ events, error: 'timeout' });
                    }, 60000);

                    const client = new WebSocket(wsUrl, 'graphql-transport-ws');

                    client.onopen = () => {
                        client.send(JSON.stringify({ type: 'connection_init' }));
                    };

                    client.onmessage = async (event) => {
                        const msg = JSON.parse(event.data);

                        if (msg.type === 'connection_ack') {
                            // Subscribe to glossaryChanged
                            client.send(JSON.stringify({
                                id: '1',
                                type: 'subscribe',
                                payload: {
                                    query: `subscription($sessionId: String!) {
                                        glossaryChanged(sessionId: $sessionId) {
                                            sessionId action termName
                                            stage percent termsCount durationMs error
                                        }
                                    }`,
                                    variables: { sessionId: sid },
                                },
                            }));

                            // Wait a moment for subscription to register, then trigger generation
                            await new Promise(r => setTimeout(r, 500));
                            await fetch(httpUrl, {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({
                                    query: `mutation($sid: String!, $phases: JSON) {
                                        generateGlossary(sessionId: $sid, phases: $phases) { status message }
                                    }`,
                                    variables: { sid, phases },
                                }),
                            });
                        }

                        if (msg.type === 'next' && msg.payload?.data?.glossaryChanged) {
                            const ev = msg.payload.data.glossaryChanged;
                            events.push(ev);
                            if (ev.action === 'GENERATION_COMPLETE') {
                                clearTimeout(timeout);
                                client.send(JSON.stringify({ id: '1', type: 'complete' }));
                                client.close();
                                resolve({ events, error: null });
                            }
                        }
                    };

                    client.onerror = () => {
                        clearTimeout(timeout);
                        resolve({ events, error: 'ws_error' });
                    };
                });
            }""",
            [gql_url, http_url, session_id, skip_phases],
        )

        assert result["error"] is None, f"Subscription error: {result['error']}"
        actions = [e["action"] for e in result["events"]]
        assert "GENERATION_STARTED" in actions, f"Missing GENERATION_STARTED in {actions}"
        assert "GENERATION_COMPLETE" in actions, f"Missing GENERATION_COMPLETE in {actions}"

        # GENERATION_COMPLETE should have termsCount and durationMs
        complete_event = next(e for e in result["events"] if e["action"] == "GENERATION_COMPLETE")
        assert complete_event["termsCount"] is not None  # may be 0 when phases skipped
        assert complete_event["durationMs"] is not None

    def test_subscription_receives_crud_events(self, page, server_url, session_id):
        """Subscribe to glossaryChanged and verify CREATED/DELETED events from mutations."""
        page.goto(server_url)
        ws_url = server_url.replace("http://", "ws://")
        gql_url = f"{ws_url}/api/graphql"
        http_url = f"{server_url}/api/graphql"

        result = page.evaluate(
            """async ([wsUrl, httpUrl, sid]) => {
                const events = [];
                return new Promise((resolve) => {
                    const timeout = setTimeout(() => {
                        resolve({ events, error: events.length ? null : 'timeout' });
                    }, 15000);

                    const client = new WebSocket(wsUrl, 'graphql-transport-ws');

                    client.onopen = () => {
                        client.send(JSON.stringify({ type: 'connection_init' }));
                    };

                    client.onmessage = async (event) => {
                        const msg = JSON.parse(event.data);

                        if (msg.type === 'connection_ack') {
                            client.send(JSON.stringify({
                                id: '1',
                                type: 'subscribe',
                                payload: {
                                    query: `subscription($sessionId: String!) {
                                        glossaryChanged(sessionId: $sessionId) {
                                            sessionId action termName
                                        }
                                    }`,
                                    variables: { sessionId: sid },
                                },
                            }));

                            await new Promise(r => setTimeout(r, 500));

                            // Create a term
                            await fetch(httpUrl, {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({
                                    query: `mutation($sid: String!, $input: GlossaryTermInput!) {
                                        createGlossaryTerm(sessionId: $sid, input: $input) { name }
                                    }`,
                                    variables: { sid, input: {
                                        name: 'sub_test_term',
                                        definition: 'Subscription test',
                                        parentId: '__root__',
                                    }},
                                }),
                            });

                            await new Promise(r => setTimeout(r, 500));

                            // Delete the term
                            await fetch(httpUrl, {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({
                                    query: `mutation($sid: String!, $name: String!) {
                                        deleteGlossaryTerm(sessionId: $sid, name: $name)
                                    }`,
                                    variables: { sid, name: 'sub_test_term' },
                                }),
                            });
                        }

                        if (msg.type === 'next' && msg.payload?.data?.glossaryChanged) {
                            events.push(msg.payload.data.glossaryChanged);
                            if (events.length >= 2) {
                                clearTimeout(timeout);
                                client.send(JSON.stringify({ id: '1', type: 'complete' }));
                                client.close();
                                resolve({ events, error: null });
                            }
                        }
                    };

                    client.onerror = () => {
                        clearTimeout(timeout);
                        resolve({ events, error: 'ws_error' });
                    };
                });
            }""",
            [gql_url, http_url, session_id],
        )

        assert result["error"] is None, f"Subscription error: {result['error']}"
        actions = [e["action"] for e in result["events"]]
        assert "CREATED" in actions, f"Missing CREATED in {actions}"
        assert "DELETED" in actions, f"Missing DELETED in {actions}"

        created = next(e for e in result["events"] if e["action"] == "CREATED")
        assert created["termName"] == "sub_test_term"
