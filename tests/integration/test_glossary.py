# Copyright (c) 2025 Kenneth Stott
# Canary: 0ff1c072-93e3-4d10-b102-acbc83a083ec
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Playwright integration tests: glossary REST API and GraphQL endpoint.

Tests glossary CRUD operations, GraphQL query/mutation, and browser-based
GraphQL access.
"""

import uuid

import pytest
import requests

pytestmark = pytest.mark.integration


@pytest.fixture(scope="class")
def session_id(server_url):
    """Create a single session shared across tests in a class."""
    body = {"session_id": str(uuid.uuid4())}
    resp = requests.post(f"{server_url}/api/sessions", json=body)
    assert resp.status_code == 200, f"Create session failed: {resp.text}"
    sid = resp.json()["session_id"]
    yield sid
    # Cleanup: close session
    requests.delete(f"{server_url}/api/sessions/{sid}")


def _gql(server_url: str, query: str, variables: dict | None = None) -> dict:
    """Execute a GraphQL query/mutation and return the parsed response."""
    body = {"query": query}
    if variables:
        body["variables"] = variables
    resp = requests.post(f"{server_url}/api/graphql", json=body)
    assert resp.status_code == 200
    return resp.json()


def _find_parent_id(server_url: str, session_id: str) -> str:
    """Find an entity name usable as parent_id via GraphQL."""
    result = _gql(server_url, """
        query($sid: String!) {
            glossary(sessionId: $sid, scope: "self_describing") {
                terms { entityId name }
            }
        }
    """, {"sid": session_id})
    terms = result.get("data", {}).get("glossary", {}).get("terms", [])
    for t in terms:
        if t.get("entityId"):
            return t["entityId"]
        if t.get("name"):
            return t["name"]
    return ""


# ── GraphQL tests ───────────────────────────────────────────────────────────


class TestGlossaryGraphQL:
    """Test the GraphQL glossary endpoint."""

    def test_graphql_introspection(self, server_url):
        """GraphQL endpoint responds to introspection."""
        query = {"query": "{ __schema { queryType { name } } }"}
        resp = requests.post(f"{server_url}/api/graphql", json=query)
        assert resp.status_code == 200
        data = resp.json()
        assert data["data"]["__schema"]["queryType"]["name"] == "Query"

    def test_graphql_schema_types(self, server_url):
        """GraphQL schema includes expected types."""
        query = {"query": "{ __schema { types { name } } }"}
        resp = requests.post(f"{server_url}/api/graphql", json=query)
        assert resp.status_code == 200
        type_names = {t["name"] for t in resp.json()["data"]["__schema"]["types"]}
        assert "GlossaryTermType" in type_names
        assert "EntityRelationshipType" in type_names
        assert "GlossaryListType" in type_names
        assert "GlossaryChangeEvent" in type_names
        # New mutation return types
        assert "GenerateResultType" in type_names
        assert "DraftDefinitionType" in type_names
        assert "DraftAliasesType" in type_names
        assert "DraftTagsType" in type_names
        assert "RefineResultType" in type_names
        assert "TaxonomySuggestionsType" in type_names
        assert "TaxonomySuggestionType" in type_names
        assert "RenameResultType" in type_names

    def test_graphql_mutations_exist(self, server_url):
        """GraphQL schema has expected mutations."""
        query = {"query": "{ __schema { mutationType { fields { name } } } }"}
        resp = requests.post(f"{server_url}/api/graphql", json=query)
        assert resp.status_code == 200
        mutation_names = {f["name"] for f in resp.json()["data"]["__schema"]["mutationType"]["fields"]}
        assert "createGlossaryTerm" in mutation_names
        assert "updateGlossaryTerm" in mutation_names
        assert "deleteGlossaryTerm" in mutation_names
        assert "createRelationship" in mutation_names
        assert "updateRelationship" in mutation_names
        assert "deleteRelationship" in mutation_names
        assert "approveRelationship" in mutation_names
        assert "bulkUpdateStatus" in mutation_names
        # New AI/operation mutations
        assert "generateGlossary" in mutation_names
        assert "draftDefinition" in mutation_names
        assert "draftAliases" in mutation_names
        assert "draftTags" in mutation_names
        assert "refineDefinition" in mutation_names
        assert "suggestTaxonomy" in mutation_names
        assert "renameTerm" in mutation_names

    def test_graphql_subscriptions_exist(self, server_url):
        """GraphQL schema has expected subscriptions."""
        query = {"query": "{ __schema { subscriptionType { fields { name } } } }"}
        resp = requests.post(f"{server_url}/api/graphql", json=query)
        assert resp.status_code == 200
        sub_names = {f["name"] for f in resp.json()["data"]["__schema"]["subscriptionType"]["fields"]}
        assert "glossaryChanged" in sub_names

    def test_graphql_glossary_query(self, server_url, session_id):
        """GraphQL glossary query returns results."""
        query = {
            "query": """
                query($sid: String!) {
                    glossary(sessionId: $sid) {
                        terms { name displayName glossaryStatus }
                        totalDefined
                        totalSelfDescribing
                    }
                }
            """,
            "variables": {"sid": session_id},
        }
        resp = requests.post(f"{server_url}/api/graphql", json=query)
        assert resp.status_code == 200
        data = resp.json()
        assert "errors" not in data, f"GraphQL errors: {data.get('errors')}"
        glossary = data["data"]["glossary"]
        assert isinstance(glossary["terms"], list)
        assert isinstance(glossary["totalDefined"], int)
        assert isinstance(glossary["totalSelfDescribing"], int)

    def test_graphql_create_relationship(self, server_url, session_id):
        """GraphQL createRelationship mutation works."""
        mutation = {
            "query": """
                mutation($sid: String!, $s: String!, $v: String!, $o: String!) {
                    createRelationship(sessionId: $sid, subject: $s, verb: $v, object: $o) {
                        id subject verb object confidence userEdited
                    }
                }
            """,
            "variables": {
                "sid": session_id,
                "s": "customer",
                "v": "PLACES",
                "o": "order",
            },
        }
        resp = requests.post(f"{server_url}/api/graphql", json=mutation)
        assert resp.status_code == 200
        data = resp.json()
        assert "errors" not in data, f"GraphQL errors: {data.get('errors')}"
        rel = data["data"]["createRelationship"]
        assert rel["verb"] == "PLACES"
        assert rel["userEdited"] is True
        assert rel["id"]

    def test_graphql_generate_glossary(self, server_url, session_id):
        """GraphQL generateGlossary mutation returns immediately with generating status."""
        mutation = {
            "query": """
                mutation($sid: String!, $phases: JSON) {
                    generateGlossary(sessionId: $sid, phases: $phases) {
                        status message
                    }
                }
            """,
            "variables": {"sid": session_id, "phases": {"definitions": True}},
        }
        resp = requests.post(f"{server_url}/api/graphql", json=mutation)
        assert resp.status_code == 200
        data = resp.json()
        assert "errors" not in data, f"GraphQL errors: {data.get('errors')}"
        result = data["data"]["generateGlossary"]
        assert result["status"] == "generating"
        assert result["message"]

    def test_graphql_rename_term(self, server_url, session_id):
        """GraphQL renameTerm mutation renames an abstract term."""
        parent_id = _find_parent_id(server_url, session_id)
        if not parent_id:
            pytest.skip("No entities available for parent_id")

        name = f"rename_src_{uuid.uuid4().hex[:6]}"
        new_name = f"rename_dst_{uuid.uuid4().hex[:6]}"

        # Create a term to rename
        create = {
            "query": """
                mutation($sid: String!, $input: GlossaryTermInput!) {
                    createGlossaryTerm(sessionId: $sid, input: $input) { name }
                }
            """,
            "variables": {
                "sid": session_id,
                "input": {"name": name, "definition": "To be renamed", "parentId": parent_id},
            },
        }
        resp = requests.post(f"{server_url}/api/graphql", json=create)
        assert resp.status_code == 200
        assert "errors" not in resp.json(), f"Create errors: {resp.json().get('errors')}"

        # Rename
        mutation = {
            "query": """
                mutation($sid: String!, $n: String!, $nn: String!) {
                    renameTerm(sessionId: $sid, name: $n, newName: $nn) {
                        oldName newName displayName relationshipsUpdated
                    }
                }
            """,
            "variables": {"sid": session_id, "n": name, "nn": new_name},
        }
        resp = requests.post(f"{server_url}/api/graphql", json=mutation)
        assert resp.status_code == 200
        data = resp.json()
        assert "errors" not in data, f"Rename errors: {data.get('errors')}"
        result = data["data"]["renameTerm"]
        assert result["oldName"].lower() == name.lower()
        assert result["newName"].lower() == new_name.lower()
        assert result["displayName"]
        assert isinstance(result["relationshipsUpdated"], int)

        # Cleanup
        requests.post(f"{server_url}/api/graphql", json={
            "query": "mutation($sid: String!, $n: String!) { deleteGlossaryTerm(sessionId: $sid, name: $n) }",
            "variables": {"sid": session_id, "n": new_name},
        })

    def test_graphql_suggest_taxonomy(self, server_url, session_id):
        """GraphQL suggestTaxonomy mutation returns suggestions structure."""
        mutation = {
            "query": """
                mutation($sid: String!) {
                    suggestTaxonomy(sessionId: $sid) {
                        suggestions { child parent parentVerb confidence reason }
                        message
                    }
                }
            """,
            "variables": {"sid": session_id},
        }
        resp = requests.post(f"{server_url}/api/graphql", json=mutation)
        assert resp.status_code == 200
        data = resp.json()
        assert "errors" not in data, f"GraphQL errors: {data.get('errors')}"
        result = data["data"]["suggestTaxonomy"]
        assert isinstance(result["suggestions"], list)
        # May have suggestions or a message about needing more terms
        if result["suggestions"]:
            s = result["suggestions"][0]
            assert "child" in s
            assert "parent" in s
            assert "confidence" in s

    def test_graphql_glossary_term_crud(self, server_url, session_id):
        """GraphQL create/update/delete glossary term lifecycle."""
        parent_id = _find_parent_id(server_url, session_id)
        if not parent_id:
            pytest.skip("No entities available for parent_id")

        name = f"gql_crud_{uuid.uuid4().hex[:6]}"

        # Create
        create = {
            "query": """
                mutation($sid: String!, $input: GlossaryTermInput!) {
                    createGlossaryTerm(sessionId: $sid, input: $input) {
                        name definition glossaryStatus
                    }
                }
            """,
            "variables": {
                "sid": session_id,
                "input": {"name": name, "definition": "GraphQL test", "parentId": parent_id},
            },
        }
        resp = requests.post(f"{server_url}/api/graphql", json=create)
        assert resp.status_code == 200
        data = resp.json()
        assert "errors" not in data, f"Create errors: {data.get('errors')}"
        assert data["data"]["createGlossaryTerm"]["name"] == name

        # Update
        update = {
            "query": """
                mutation($sid: String!, $n: String!, $input: GlossaryTermUpdateInput!) {
                    updateGlossaryTerm(sessionId: $sid, name: $n, input: $input) {
                        name definition
                    }
                }
            """,
            "variables": {
                "sid": session_id,
                "n": name,
                "input": {"definition": "Updated via GraphQL"},
            },
        }
        resp = requests.post(f"{server_url}/api/graphql", json=update)
        assert resp.status_code == 200
        data = resp.json()
        assert "errors" not in data, f"Update errors: {data.get('errors')}"
        assert data["data"]["updateGlossaryTerm"]["definition"] == "Updated via GraphQL"

        # Delete
        delete = {
            "query": """
                mutation($sid: String!, $n: String!) {
                    deleteGlossaryTerm(sessionId: $sid, name: $n)
                }
            """,
            "variables": {"sid": session_id, "n": name},
        }
        resp = requests.post(f"{server_url}/api/graphql", json=delete)
        assert resp.status_code == 200
        data = resp.json()
        assert "errors" not in data, f"Delete errors: {data.get('errors')}"
        assert data["data"]["deleteGlossaryTerm"] is True


# ── Browser-based tests ─────────────────────────────────────────────────────


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
            pytest.skip("No entities available for parent_id")

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

        # Use page.evaluate to:
        # 1. Open a GraphQL WS subscription for glossaryChanged
        # 2. Trigger generateGlossary mutation via HTTP
        # 3. Collect events until GENERATION_COMPLETE or timeout
        result = page.evaluate(
            """async ([wsUrl, httpUrl, sid]) => {
                const events = [];
                return new Promise((resolve) => {
                    const timeout = setTimeout(() => {
                        resolve({ events, error: 'timeout' });
                    }, 30000);

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
                                    query: `mutation($sid: String!) {
                                        generateGlossary(sessionId: $sid) { status message }
                                    }`,
                                    variables: { sid },
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
            [gql_url, http_url, session_id],
        )

        assert result["error"] is None, f"Subscription error: {result['error']}"
        actions = [e["action"] for e in result["events"]]
        assert "GENERATION_STARTED" in actions, f"Missing GENERATION_STARTED in {actions}"
        assert "GENERATION_COMPLETE" in actions, f"Missing GENERATION_COMPLETE in {actions}"

        # GENERATION_COMPLETE should have termsCount and durationMs
        complete_event = next(e for e in result["events"] if e["action"] == "GENERATION_COMPLETE")
        assert complete_event["termsCount"] is not None
        assert complete_event["durationMs"] is not None

        # If there were progress events, verify they have stage and percent
        progress_events = [e for e in result["events"] if e["action"] == "GENERATION_PROGRESS"]
        for pe in progress_events:
            assert pe["stage"] is not None
            assert pe["percent"] is not None

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
