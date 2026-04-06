# Copyright (c) 2025 Kenneth Stott
# Canary: 0ff1c072-93e3-4d10-b102-acbc83a083ec
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Playwright integration tests: glossary REST API and GraphQL endpoint.

Tests glossary CRUD operations, GraphQL query/mutation, and browser-based
GraphQL access.
"""

from __future__ import annotations
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
            pytest.fail("Test data precondition not met — seed required data before running this test")

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
            pytest.fail("Test data precondition not met — seed required data before running this test")

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


# Browser-based tests have been moved to tests/e2e/test_glossary_browser.py
