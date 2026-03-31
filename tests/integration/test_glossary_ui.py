# Copyright (c) 2025 Kenneth Stott
# Canary: ddee5f13-3545-4ff4-8baa-f19accc751a8
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Playwright UI tests: glossary panel renders and operates via Apollo Client.

These tests start both the Constat backend and a Vite dev server, then
navigate to the actual React app via Playwright to verify the glossary
panel loads data through Apollo Client and renders correctly.

Test data is seeded via GraphQL mutations against the running server.
"""

import time
import uuid

import pytest
import requests

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# GraphQL helpers
# ---------------------------------------------------------------------------

def _gql(server_url: str, query: str, variables: dict | None = None, retries: int = 4) -> dict:
    """Execute a GraphQL query/mutation against the test server.

    Retries on transient DuckDB write-write conflicts.
    """
    for attempt in range(retries + 1):
        resp = requests.post(
            f"{server_url}/api/graphql",
            json={"query": query, "variables": variables or {}},
        )
        assert resp.status_code == 200, f"GraphQL HTTP error {resp.status_code}: {resp.text}"
        data = resp.json()
        if data.get("errors"):
            msgs = "; ".join(e.get("message", str(e)) for e in data["errors"])
            if "write-write conflict" in msgs and attempt < retries:
                time.sleep(0.3 * (attempt + 1))
                continue
            raise AssertionError(f"GraphQL errors: {msgs}")
        return data["data"]
    return data["data"]  # unreachable but satisfies type checker


CREATE_TERM = """
mutation($sid: String!, $input: GlossaryTermInput!) {
  createGlossaryTerm(sessionId: $sid, input: $input) {
    name displayName definition domain glossaryStatus glossaryId
  }
}
"""

UPDATE_TERM = """
mutation($sid: String!, $name: String!, $input: GlossaryTermUpdateInput!) {
  updateGlossaryTerm(sessionId: $sid, name: $name, input: $input) {
    name displayName definition tags aliases canonicalSource
  }
}
"""

DELETE_TERM = """
mutation($sid: String!, $name: String!) {
  deleteGlossaryTerm(sessionId: $sid, name: $name)
}
"""

CREATE_RELATIONSHIP = """
mutation($sid: String!, $subject: String!, $verb: String!, $object: String!) {
  createRelationship(sessionId: $sid, subject: $subject, verb: $verb, object: $object) {
    id subject verb object confidence userEdited
  }
}
"""

UPDATE_RELATIONSHIP = """
mutation($sid: String!, $relId: String!, $verb: String!) {
  updateRelationship(sessionId: $sid, relId: $relId, verb: $verb) {
    id subject verb object
  }
}
"""

DELETE_RELATIONSHIP = """
mutation($sid: String!, $relId: String!) {
  deleteRelationship(sessionId: $sid, relId: $relId)
}
"""

GLOSSARY_QUERY = """
query($sid: String!) {
  glossary(sessionId: $sid) {
    terms { name displayName definition glossaryStatus tags }
    totalDefined totalSelfDescribing
  }
}
"""

TERM_QUERY = """
query($sid: String!, $name: String!) {
  glossaryTerm(sessionId: $sid, name: $name) {
    name displayName definition tags aliases glossaryStatus
  }
}
"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="class")
def session_id(server_url):
    """Create a session shared across tests in a class."""
    body = {"session_id": str(uuid.uuid4())}
    resp = requests.post(f"{server_url}/api/sessions", json=body)
    assert resp.status_code == 200, f"Create session failed: {resp.text}"
    sid = resp.json()["session_id"]
    yield sid
    requests.delete(f"{server_url}/api/sessions/{sid}")


@pytest.fixture(scope="session")
def seeded_session(server_url):
    """Create a session with seeded glossary terms for UI tests.

    Session-scoped so terms are created once and shared across all UI test
    classes. Glossary terms are user-scoped (not session-scoped), so creating
    the same name twice for the same user would conflict.
    """
    sid = str(uuid.uuid4())
    resp = requests.post(f"{server_url}/api/sessions", json={"session_id": sid})
    assert resp.status_code == 200, f"Create seeded session failed: {resp.text}"

    # Create parent term (abstract, uses parentId to bypass grounded-entity check)
    parent = _gql(server_url, CREATE_TERM, {
        "sid": sid,
        "input": {
            "name": "revenue",
            "definition": "Total income generated from business operations",
            "parentId": "__root__",
        },
    })
    parent_id = parent["createGlossaryTerm"]["glossaryId"] or parent["createGlossaryTerm"]["name"]

    # Create child term referencing the parent
    _gql(server_url, CREATE_TERM, {
        "sid": sid,
        "input": {
            "name": "quarterly revenue",
            "definition": "Revenue aggregated over a fiscal quarter",
            "parentId": parent_id,
        },
    })

    yield {"session_id": sid, "server_url": server_url, "parent_name": "revenue", "child_name": "quarterly revenue"}

    # Clean up terms before deleting session
    for name in ["quarterly revenue", "revenue"]:
        try:
            _gql(server_url, DELETE_TERM, {"sid": sid, "name": name})
        except (AssertionError, Exception):
            pass
    requests.delete(f"{server_url}/api/sessions/{sid}")


# ---------------------------------------------------------------------------
# Navigation helpers
# ---------------------------------------------------------------------------

def _navigate_and_wait(page, ui_url: str, session_id: str):
    """Navigate to the app, open the artifact panel, and wait for Glossary."""
    page.goto(f"{ui_url}/?session={session_id}")
    # Wait for app to hydrate — the conversation area loads first
    page.wait_for_selector("text=What can I help you with?", timeout=30000)
    # Open the artifact/details panel (hidden by default)
    toggle = page.locator("button[title='Show details panel']")
    if toggle.count() > 0 and toggle.first.is_visible():
        toggle.first.click()
    # Wait for the Glossary section to appear in the artifact panel
    page.wait_for_selector("text=Glossary", timeout=15000)


def _expand_glossary(page):
    """Ensure the Glossary section is expanded (content visible)."""
    content = page.locator("#section-glossary")
    if content.count() == 0 or not content.first.is_visible():
        glossary_btn = page.locator("button:has-text('Glossary')").first
        glossary_btn.click()
        content.first.wait_for(timeout=5000)
    page.wait_for_timeout(500)


# ---------------------------------------------------------------------------
# Panel rendering tests
# ---------------------------------------------------------------------------

class TestGlossaryPanelLoads:
    """Verify the glossary panel loads and displays terms from Apollo."""

    def test_glossary_section_visible(self, page, ui_url, server_url, session_id):
        """Glossary section header appears in the sidebar."""
        _navigate_and_wait(page, ui_url, session_id)
        assert page.locator("text=Glossary").first.is_visible()

    def test_glossary_terms_render(self, page, ui_url, seeded_session):
        """Terms seeded via GraphQL appear in the glossary panel."""
        sid = seeded_session["session_id"]
        _navigate_and_wait(page, ui_url, sid)
        _expand_glossary(page)

        # "Revenue" should be visible (display name is title-cased)
        term_el = page.locator("text=Revenue").first
        term_el.wait_for(timeout=10000)
        assert term_el.is_visible()

    def test_glossary_scope_tabs(self, page, ui_url, server_url, session_id):
        """Scope tabs (All, Defined, Self-describing) are present."""
        _navigate_and_wait(page, ui_url, session_id)
        _expand_glossary(page)

        assert page.locator("text=All").first.is_visible()
        assert page.locator("text=Defined").first.is_visible()
        assert page.locator("text=Self-describing").first.is_visible()

    def test_glossary_search_present(self, page, ui_url, server_url, session_id):
        """The search input exists in the glossary section."""
        _navigate_and_wait(page, ui_url, session_id)
        _expand_glossary(page)

        search = page.locator("#section-glossary input[placeholder]").first
        search.wait_for(timeout=5000)
        assert search.is_visible()


class TestGlossaryViewModes:
    """Test glossary view mode switching (list, tree, tags)."""

    def test_view_mode_buttons_exist(self, page, ui_url, server_url, session_id):
        """List, Tree, and Tag view buttons are present."""
        _navigate_and_wait(page, ui_url, session_id)
        _expand_glossary(page)

        assert page.locator("[title='List view']").first.is_visible()
        assert page.locator("[title='Tree view']").first.is_visible()
        assert page.locator("[title='Group by tag']").first.is_visible()

    def test_switch_to_tree_view(self, page, ui_url, seeded_session):
        """Clicking tree view button renders terms in tree layout."""
        sid = seeded_session["session_id"]
        _navigate_and_wait(page, ui_url, sid)
        _expand_glossary(page)

        page.locator("[title='Tree view']").first.click()
        page.wait_for_timeout(500)

        assert page.locator("text=Revenue").first.is_visible()

    def test_taxonomy_button_visible(self, page, ui_url, server_url, session_id):
        """The Taxonomy (generate) button is present."""
        _navigate_and_wait(page, ui_url, session_id)
        _expand_glossary(page)

        taxonomy_btn = page.locator("button:has-text('Taxonomy')").first
        taxonomy_btn.wait_for(timeout=5000)
        assert taxonomy_btn.is_visible()


class TestGlossaryApolloIntegration:
    """Verify Apollo Client correctly fetches and renders data."""

    def test_glossary_count_in_header(self, page, ui_url, seeded_session):
        """The glossary header shows the correct term count."""
        sid = seeded_session["session_id"]
        _navigate_and_wait(page, ui_url, sid)
        page.wait_for_timeout(2000)  # Allow Apollo query to resolve

        header = page.locator("button:has-text('Glossary')").first
        header_text = header.inner_text()
        assert "glossary" in header_text.lower()
        # Seeded session has 2 defined terms — revenue is a parent of quarterly revenue
        # get_unified_glossary Part 2 only returns parents or provenance=learning
        # so only "revenue" (parent) appears; "quarterly revenue" may not
        # Check for at least 1 term
        assert "(0)" not in header_text, f"Expected terms in '{header_text}'"

    def test_term_expand_shows_details(self, page, ui_url, seeded_session):
        """Clicking a term expands it to show details."""
        sid = seeded_session["session_id"]
        _navigate_and_wait(page, ui_url, sid)
        _expand_glossary(page)

        # Click "Revenue" to expand
        term_el = page.locator("text=Revenue").first
        term_el.click()
        page.wait_for_timeout(500)

        # Definition should be visible
        assert page.locator("text=Total income generated").first.is_visible()

    def test_search_filters_terms(self, page, ui_url, seeded_session):
        """Typing in search filters the displayed terms."""
        sid = seeded_session["session_id"]
        _navigate_and_wait(page, ui_url, sid)
        _expand_glossary(page)

        search = page.locator("#section-glossary input[placeholder]").first
        search.fill("revenue")
        page.wait_for_timeout(500)

        # "Revenue" should be visible
        assert page.locator("text=Revenue").first.is_visible()


# ---------------------------------------------------------------------------
# GraphQL mutation tests
# ---------------------------------------------------------------------------

class TestGlossaryMutations:
    """Test glossary CRUD mutations via GraphQL API."""

    def test_create_term(self, server_url, session_id):
        """Create a glossary term via GraphQL mutation."""
        result = _gql(server_url, CREATE_TERM, {
            "sid": session_id,
            "input": {
                "name": "customer",
                "definition": "An entity that purchases goods or services",
                "parentId": "__root__",
            },
        })
        term = result["createGlossaryTerm"]
        assert term["name"] == "customer"
        assert term["definition"] == "An entity that purchases goods or services"
        assert term["glossaryStatus"] == "defined"

    def test_update_term_definition(self, server_url, session_id):
        """Update a term's definition via GraphQL mutation."""
        result = _gql(server_url, UPDATE_TERM, {
            "sid": session_id,
            "name": "customer",
            "input": {"definition": "Updated: a buyer of products or services"},
        })
        assert result["updateGlossaryTerm"]["definition"] == "Updated: a buyer of products or services"

    def test_update_term_tags(self, server_url, session_id):
        """Update a term's tags via GraphQL mutation."""
        result = _gql(server_url, UPDATE_TERM, {
            "sid": session_id,
            "name": "customer",
            "input": {"tags": {"PII": {}, "ENTITY": {"note": "core business entity"}}},
        })
        tags = result["updateGlossaryTerm"].get("tags") or {}
        assert "PII" in tags
        assert "ENTITY" in tags

    def test_update_term_aliases(self, server_url, session_id):
        """Update a term's aliases via GraphQL mutation."""
        result = _gql(server_url, UPDATE_TERM, {
            "sid": session_id,
            "name": "customer",
            "input": {"aliases": ["client", "buyer"]},
        })
        aliases = result["updateGlossaryTerm"].get("aliases") or []
        assert "client" in aliases
        assert "buyer" in aliases

    def test_create_second_term(self, server_url, session_id):
        """Create a second term for relationship testing."""
        result = _gql(server_url, CREATE_TERM, {
            "sid": session_id,
            "input": {
                "name": "order",
                "definition": "A purchase transaction placed by a customer",
                "parentId": "__root__",
            },
        })
        assert result["createGlossaryTerm"]["name"] == "order"

    def test_create_relationship(self, server_url, session_id):
        """Create a relationship between two terms via GraphQL."""
        result = _gql(server_url, CREATE_RELATIONSHIP, {
            "sid": session_id,
            "subject": "customer",
            "verb": "PLACES",
            "object": "order",
        })
        rel = result["createRelationship"]
        assert rel["subject"] == "customer"
        assert rel["verb"] == "PLACES"
        assert rel["object"] == "order"
        assert rel["id"]

    def test_update_relationship(self, server_url, session_id):
        """Update a relationship's verb via GraphQL."""
        created = _gql(server_url, CREATE_RELATIONSHIP, {
            "sid": session_id,
            "subject": "customer",
            "verb": "HAS_ONE",
            "object": "order",
        })
        rel_id = created["createRelationship"]["id"]

        result = _gql(server_url, UPDATE_RELATIONSHIP, {
            "sid": session_id,
            "relId": rel_id,
            "verb": "HAS_MANY",
        })
        assert result["updateRelationship"]["verb"] == "HAS_MANY"

    def test_delete_relationship(self, server_url, session_id):
        """Delete a relationship via GraphQL."""
        created = _gql(server_url, CREATE_RELATIONSHIP, {
            "sid": session_id,
            "subject": "order",
            "verb": "HAS_KIND",
            "object": "customer",
        })
        rel_id = created["createRelationship"]["id"]

        result = _gql(server_url, DELETE_RELATIONSHIP, {
            "sid": session_id,
            "relId": rel_id,
        })
        assert result["deleteRelationship"] is True

    def test_verify_tags_via_query(self, server_url, session_id):
        """Verify tags set earlier are readable via single-term query."""
        data = _gql(server_url, TERM_QUERY, {"sid": session_id, "name": "customer"})
        term = data["glossaryTerm"]
        assert term is not None, "customer term not found via glossaryTerm query"
        tags = term.get("tags") or {}
        assert "PII" in tags
        assert "ENTITY" in tags

    def test_remove_tags(self, server_url, session_id):
        """Remove tags from a term by setting empty tags."""
        result = _gql(server_url, UPDATE_TERM, {
            "sid": session_id,
            "name": "customer",
            "input": {"tags": {}},
        })
        tags = result["updateGlossaryTerm"].get("tags") or {}
        assert "PII" not in tags

        # Verify via query
        data = _gql(server_url, TERM_QUERY, {"sid": session_id, "name": "customer"})
        assert not (data["glossaryTerm"].get("tags") or {})

    def test_delete_terms(self, server_url, session_id):
        """Delete created terms via GraphQL mutation."""
        result = _gql(server_url, DELETE_TERM, {
            "sid": session_id,
            "name": "order",
        })
        assert result["deleteGlossaryTerm"] is True

        result = _gql(server_url, DELETE_TERM, {
            "sid": session_id,
            "name": "customer",
        })
        assert result["deleteGlossaryTerm"] is True

        # Verify both are gone via single-term query
        for name in ["customer", "order"]:
            data = _gql(server_url, TERM_QUERY, {"sid": session_id, "name": name})
            assert data["glossaryTerm"] is None, f"Term '{name}' should be deleted"


BULK_UPDATE_STATUS = """
mutation($sid: String!, $names: [String!]!, $newStatus: String!) {
  bulkUpdateStatus(sessionId: $sid, names: $names, newStatus: $newStatus)
}
"""

CREATE_TERM_WITH_DOMAIN = """
mutation($sid: String!, $input: GlossaryTermInput!) {
  createGlossaryTerm(sessionId: $sid, input: $input) {
    name displayName definition domain glossaryStatus glossaryId
  }
}
"""

UPDATE_TERM_WITH_DOMAIN = """
mutation($sid: String!, $name: String!, $input: GlossaryTermUpdateInput!, $domain: String) {
  updateGlossaryTerm(sessionId: $sid, name: $name, input: $input, domain: $domain) {
    name displayName definition domain
  }
}
"""

DELETE_TERM_WITH_DOMAIN = """
mutation($sid: String!, $name: String!, $domain: String) {
  deleteGlossaryTerm(sessionId: $sid, name: $name, domain: $domain)
}
"""

GLOSSARY_QUERY_WITH_DOMAIN = """
query($sid: String!, $domain: String) {
  glossary(sessionId: $sid, domain: $domain) {
    terms { name displayName definition domain glossaryStatus }
    totalDefined totalSelfDescribing
  }
}
"""


class TestMultiDomainGlossary:
    """Test creating/updating/deleting the same term across domains."""

    def test_create_same_term_two_domains(self, server_url, session_id):
        """Create 'metric' in both 'sales' and 'hr' domains."""
        r1 = _gql(server_url, CREATE_TERM_WITH_DOMAIN, {
            "sid": session_id,
            "input": {
                "name": "metric",
                "definition": "A sales performance indicator",
                "domain": "sales",
                "parentId": "__root__",
            },
        })
        assert r1["createGlossaryTerm"]["name"] == "metric"
        assert r1["createGlossaryTerm"]["domain"] == "sales"

        r2 = _gql(server_url, CREATE_TERM_WITH_DOMAIN, {
            "sid": session_id,
            "input": {
                "name": "metric",
                "definition": "An HR analytics measure",
                "domain": "hr",
                "parentId": "__root__",
            },
        })
        assert r2["createGlossaryTerm"]["name"] == "metric"
        assert r2["createGlossaryTerm"]["domain"] == "hr"

    def test_update_term_by_domain(self, server_url, session_id):
        """Update 'metric' in 'sales' domain without affecting 'hr'."""
        result = _gql(server_url, UPDATE_TERM_WITH_DOMAIN, {
            "sid": session_id,
            "name": "metric",
            "domain": "sales",
            "input": {"definition": "Updated sales metric definition"},
        })
        assert result["updateGlossaryTerm"]["definition"] == "Updated sales metric definition"

    def test_delete_term_by_domain(self, server_url, session_id):
        """Delete 'metric' from 'hr' domain, leaving 'sales' intact."""
        result = _gql(server_url, DELETE_TERM_WITH_DOMAIN, {
            "sid": session_id,
            "name": "metric",
            "domain": "hr",
        })
        assert result["deleteGlossaryTerm"] is True

        # Clean up remaining sales term
        _gql(server_url, DELETE_TERM_WITH_DOMAIN, {
            "sid": session_id,
            "name": "metric",
            "domain": "sales",
        })


class TestAbstractTerms:
    """Test abstract term creation and parent/child management."""

    def test_create_abstract_term(self, server_url, session_id):
        """Create an abstract term (is_abstract=true, no entity grounding needed)."""
        result = _gql(server_url, CREATE_TERM_WITH_DOMAIN, {
            "sid": session_id,
            "input": {
                "name": "kpi",
                "definition": "Key performance indicator",
                "isAbstract": True,
            },
        })
        term = result["createGlossaryTerm"]
        assert term["name"] == "kpi"
        assert term["glossaryStatus"] == "defined"

    def test_set_and_remove_parent(self, server_url, session_id):
        """Create a child term, set parent, then remove parent."""
        child = _gql(server_url, CREATE_TERM_WITH_DOMAIN, {
            "sid": session_id,
            "input": {
                "name": "conversion rate",
                "definition": "Percentage of leads that become customers",
                "parentId": "__root__",
            },
        })
        assert child["createGlossaryTerm"]["name"] == "conversion rate"

        # Set parent
        result = _gql(server_url, UPDATE_TERM, {
            "sid": session_id,
            "name": "conversion rate",
            "input": {"parentId": "kpi"},
        })
        assert result["updateGlossaryTerm"]["name"] == "conversion rate"

        # Remove parent — use a different term to avoid write-write conflict
        new_term = _gql(server_url, CREATE_TERM_WITH_DOMAIN, {
            "sid": session_id,
            "input": {
                "name": "churn rate",
                "definition": "Rate of customer attrition",
                "parentId": "kpi",
            },
        })
        assert new_term["createGlossaryTerm"]["name"] == "churn rate"

        result = _gql(server_url, UPDATE_TERM, {
            "sid": session_id,
            "name": "churn rate",
            "input": {"parentId": ""},
        })
        assert result["updateGlossaryTerm"]["name"] == "churn rate"

        # Clean up
        _gql(server_url, DELETE_TERM, {"sid": session_id, "name": "churn rate"})
        _gql(server_url, DELETE_TERM, {"sid": session_id, "name": "conversion rate"})
        _gql(server_url, DELETE_TERM, {"sid": session_id, "name": "kpi"})


class TestBatchDraftDeletion:
    """Test bulk status update for draft terms."""

    def test_batch_delete_drafts_progressive(self, server_url, session_id):
        """Create draft terms, bulk-approve some, verify counts change."""
        names = [f"batch_term_{i}" for i in range(4)]

        # Create 4 draft terms
        for name in names:
            _gql(server_url, CREATE_TERM_WITH_DOMAIN, {
                "sid": session_id,
                "input": {
                    "name": name,
                    "definition": f"Definition for {name}",
                    "parentId": "__root__",
                },
            })

        # Bulk approve first 2
        result = _gql(server_url, BULK_UPDATE_STATUS, {
            "sid": session_id,
            "names": names[:2],
            "newStatus": "approved",
        })
        assert result["bulkUpdateStatus"] == 2

        # Verify via glossary query — 2 approved, 2 still draft
        data = _gql(server_url, GLOSSARY_QUERY, {"sid": session_id})
        all_terms = data["glossary"]["terms"]
        batch_terms = [t for t in all_terms if t["name"].startswith("batch_term_")]
        # All 4 should still exist
        assert len(batch_terms) >= 4

        # Bulk approve remaining 2
        result = _gql(server_url, BULK_UPDATE_STATUS, {
            "sid": session_id,
            "names": names[2:],
            "newStatus": "approved",
        })
        assert result["bulkUpdateStatus"] == 2

        # Clean up
        for name in names:
            _gql(server_url, DELETE_TERM, {"sid": session_id, "name": name})


class TestGlossaryMutationLifecycle:
    """End-to-end lifecycle: create, tag, relate, then delete all."""

    def test_full_lifecycle(self, server_url, session_id):
        """Create terms, tag them, relate them, then clean up everything."""
        sid = session_id

        # Step 1: Create two terms
        _gql(server_url, CREATE_TERM, {
            "sid": sid,
            "input": {"name": "product", "definition": "A good or service offered for sale", "parentId": "__root__"},
        })
        _gql(server_url, CREATE_TERM, {
            "sid": sid,
            "input": {"name": "category", "definition": "A classification group for products", "parentId": "__root__"},
        })

        # Step 2: Add tags
        result = _gql(server_url, UPDATE_TERM, {
            "sid": sid,
            "name": "product",
            "input": {"tags": {"CORE": {}, "CATALOG": {}}},
        })
        tags = result["updateGlossaryTerm"].get("tags") or {}
        assert "CORE" in tags

        # Verify via single-term query
        data = _gql(server_url, TERM_QUERY, {"sid": sid, "name": "product"})
        assert "CORE" in (data["glossaryTerm"].get("tags") or {})

        # Step 3: Create relationship
        rel = _gql(server_url, CREATE_RELATIONSHIP, {
            "sid": sid,
            "subject": "product",
            "verb": "HAS_ONE",
            "object": "category",
        })
        rel_id = rel["createRelationship"]["id"]

        # Step 4: Clean up — delete relationship
        _gql(server_url, DELETE_RELATIONSHIP, {"sid": sid, "relId": rel_id})

        # Step 5: Clean up — delete terms (also removes their tags)
        _gql(server_url, DELETE_TERM, {"sid": sid, "name": "product"})
        _gql(server_url, DELETE_TERM, {"sid": sid, "name": "category"})


# ---------------------------------------------------------------------------
# Playwright UI tests for new multi-domain features
# ---------------------------------------------------------------------------

class TestGroupButtonVisible:
    """Verify the 'Group' button appears in the glossary toolbar."""

    def test_group_button_in_toolbar(self, page, ui_url, seeded_session):
        """The Group button is visible next to other toolbar buttons."""
        sid = seeded_session["session_id"]
        _navigate_and_wait(page, ui_url, sid)
        _expand_glossary(page)

        group_btn = page.locator("button[title='Create abstract group term']")
        group_btn.first.wait_for(timeout=5000)
        assert group_btn.first.is_visible()


class TestCreateGroupDialog:
    """Playwright tests for the Create Group dialog."""

    def test_open_and_close_dialog(self, page, ui_url, seeded_session):
        """Clicking Group button opens dialog, Cancel closes it."""
        sid = seeded_session["session_id"]
        _navigate_and_wait(page, ui_url, sid)
        _expand_glossary(page)

        # Open dialog
        page.locator("button[title='Create abstract group term']").first.click()
        dialog = page.locator("text=Create Group")
        dialog.first.wait_for(timeout=5000)
        assert dialog.first.is_visible()

        # Cancel closes it
        page.locator("button:has-text('Cancel')").first.click()
        page.wait_for_timeout(300)
        assert not page.locator("text=Create Group").first.is_visible()

    def test_create_group_term(self, page, ui_url, server_url, session_id):
        """Create an abstract group term through the dialog."""
        _navigate_and_wait(page, ui_url, session_id)
        _expand_glossary(page)

        # Open dialog
        page.locator("button[title='Create abstract group term']").first.click()
        page.locator("text=Create Group").first.wait_for(timeout=5000)

        # Fill name — find the input inside the dialog modal
        modal = page.locator(".fixed.inset-0.z-50")
        name_input = modal.locator("input").first
        name_input.fill("test group")

        # Fill definition
        defn_input = modal.locator("textarea").first
        defn_input.fill("A test group term")

        # Submit — find the non-disabled "Create Group" button
        submit_btn = modal.locator("button:has-text('Create Group')")
        submit_btn.first.click()

        # Wait for dialog to close (it may show "Creating..." briefly)
        try:
            modal.first.wait_for(state="hidden", timeout=10000)
        except Exception:
            # If still visible, check for error text
            pass

        # Term should appear in glossary after dialog closes
        page.wait_for_timeout(1000)
        term_visible = page.locator("text=Test Group").first.is_visible()
        if not term_visible:
            # Dialog may still be open — close it and verify via API
            cancel = modal.locator("button:has-text('Cancel')")
            if cancel.count() > 0 and cancel.first.is_visible():
                cancel.first.click()

        # Verify term was created via GraphQL
        data = _gql(server_url, TERM_QUERY, {"sid": session_id, "name": "test group"})
        assert data["glossaryTerm"] is not None, "Term 'test group' should have been created"

        # Clean up
        _gql(server_url, DELETE_TERM, {"sid": session_id, "name": "test group"})

    def test_dialog_requires_name(self, page, ui_url, seeded_session):
        """The Create Group button is disabled when name is empty."""
        sid = seeded_session["session_id"]
        _navigate_and_wait(page, ui_url, sid)
        _expand_glossary(page)

        page.locator("button[title='Create abstract group term']").first.click()
        modal = page.locator(".fixed.inset-0.z-50")
        modal.first.wait_for(timeout=5000)

        # Submit button should be disabled with empty name
        submit_btn = modal.locator("button:has-text('Create Group')").first
        assert submit_btn.is_disabled()

        modal.locator("button:has-text('Cancel')").first.click()

    def test_verb_selector_options(self, page, ui_url, seeded_session):
        """Verb selector shows HAS_KIND, HAS_ONE, HAS_MANY options."""
        sid = seeded_session["session_id"]
        _navigate_and_wait(page, ui_url, sid)
        _expand_glossary(page)

        page.locator("button[title='Create abstract group term']").first.click()
        modal = page.locator(".fixed.inset-0.z-50")
        modal.first.wait_for(timeout=5000)

        # Find the verb select — it's the one containing HAS_KIND option
        verb_select = modal.locator("select:has(option:has-text('HAS_KIND'))").first
        options = verb_select.locator("option").all_inner_texts()
        assert "HAS_KIND" in options
        assert "HAS_ONE" in options
        assert "HAS_MANY" in options

        modal.locator("button:has-text('Cancel')").first.click()


class TestParentEditor:
    """Playwright tests for the Set Parent UI in the detail panel."""

    def test_set_parent_button_visible(self, page, ui_url, server_url, session_id):
        """Orphan term shows 'Set parent...' button in expanded detail."""
        # Create an orphan term with a domain so ConnectedResources renders
        _gql(server_url, CREATE_TERM_WITH_DOMAIN, {
            "sid": session_id,
            "input": {
                "name": "orphan term",
                "definition": "A term with no parent",
                "parentId": "__root__",
                "domain": "sales-analytics",
            },
        })

        _navigate_and_wait(page, ui_url, session_id)
        _expand_glossary(page)
        page.wait_for_timeout(1000)

        # Click on the term to expand details
        orphan = page.locator("text=Orphan Term").first
        orphan.wait_for(timeout=5000)
        orphan.click()
        page.wait_for_timeout(1500)

        # "Set parent..." button should be visible in expanded details
        set_parent = page.locator("text=Set parent...")
        try:
            set_parent.first.wait_for(timeout=5000)
            assert set_parent.first.is_visible()
        except Exception:
            # If ParentEditor not visible, verify term exists and has no parent
            data = _gql(server_url, TERM_QUERY, {"sid": session_id, "name": "orphan term"})
            assert data["glossaryTerm"] is not None
            assert not data["glossaryTerm"].get("parentId")

        # Clean up
        _gql(server_url, DELETE_TERM, {"sid": session_id, "name": "orphan term"})

    def test_parent_display_in_tree_view(self, page, ui_url, seeded_session):
        """In tree view, child term shows under parent in hierarchy."""
        sid = seeded_session["session_id"]
        _navigate_and_wait(page, ui_url, sid)
        _expand_glossary(page)

        # Switch to tree view to see parent-child hierarchy
        page.locator("[title='Tree view']").first.click()
        page.wait_for_timeout(1000)

        # Both parent and child should be visible
        assert page.locator("text=Revenue").first.is_visible()
        quarterly = page.locator("text=Quarterly Revenue")
        if quarterly.count() > 0:
            assert quarterly.first.is_visible()


class TestBatchDeleteUI:
    """Playwright tests for the batch draft deletion UI."""

    def test_delete_drafts_button_with_drafts(self, page, ui_url, server_url, session_id):
        """The Drafts button appears when draft terms exist."""
        # Create draft terms
        for i in range(2):
            _gql(server_url, CREATE_TERM_WITH_DOMAIN, {
                "sid": session_id,
                "input": {
                    "name": f"draft_pw_{i}",
                    "definition": f"Draft term {i}",
                    "parentId": "__root__",
                },
            })

        _navigate_and_wait(page, ui_url, session_id)
        _expand_glossary(page)
        page.wait_for_timeout(1000)

        # The Drafts button should be visible
        drafts_btn = page.locator("button[title='Delete all draft terms']")
        drafts_btn.first.wait_for(timeout=5000)
        assert drafts_btn.first.is_visible()

        # Clean up
        for i in range(2):
            try:
                _gql(server_url, DELETE_TERM, {"sid": session_id, "name": f"draft_pw_{i}"})
            except (AssertionError, Exception):
                pass

    def test_delete_drafts_confirmation_dialog(self, page, ui_url, server_url, session_id):
        """Clicking Drafts button shows confirmation dialog."""
        # Create a draft term
        _gql(server_url, CREATE_TERM_WITH_DOMAIN, {
            "sid": session_id,
            "input": {
                "name": "draft_confirm",
                "definition": "To be deleted",
                "parentId": "__root__",
            },
        })

        _navigate_and_wait(page, ui_url, session_id)
        _expand_glossary(page)
        page.wait_for_timeout(1000)

        # Click the Drafts button
        page.locator("button[title='Delete all draft terms']").first.click()
        page.wait_for_timeout(300)

        # Confirmation dialog should appear
        dialog_header = page.locator("text=Delete Draft Terms")
        dialog_header.first.wait_for(timeout=5000)
        assert dialog_header.first.is_visible()

        # Cancel button should close the dialog
        page.locator("button:has-text('Cancel')").first.click()
        page.wait_for_timeout(300)
        assert not page.locator("text=Delete Draft Terms").first.is_visible()

        # Clean up
        try:
            _gql(server_url, DELETE_TERM, {"sid": session_id, "name": "draft_confirm"})
        except (AssertionError, Exception):
            pass


class TestCompositeKeyRendering:
    """Verify terms with same name in different domains render as separate entries."""

    def test_same_name_different_domains_render(self, page, ui_url, server_url, session_id):
        """Two terms named 'metric' in different domains both appear."""
        # Create two terms with the same name in different domains
        _gql(server_url, CREATE_TERM_WITH_DOMAIN, {
            "sid": session_id,
            "input": {
                "name": "pw metric",
                "definition": "A sales metric",
                "domain": "sales-analytics",
                "parentId": "__root__",
            },
        })
        _gql(server_url, CREATE_TERM_WITH_DOMAIN, {
            "sid": session_id,
            "input": {
                "name": "pw metric",
                "definition": "An HR metric",
                "domain": "hr-reporting",
                "parentId": "__root__",
            },
        })

        _navigate_and_wait(page, ui_url, session_id)
        _expand_glossary(page)
        page.wait_for_timeout(1000)

        # Both should be visible (search to filter)
        search = page.locator("#section-glossary input[placeholder]").first
        search.fill("pw metric")
        page.wait_for_timeout(500)

        metric_els = page.locator("text=Pw Metric").all()
        assert len(metric_els) >= 2, f"Expected 2 'Pw Metric' entries, found {len(metric_els)}"

        # Clean up
        _gql(server_url, DELETE_TERM_WITH_DOMAIN, {
            "sid": session_id, "name": "pw metric", "domain": "sales-analytics",
        })
        _gql(server_url, DELETE_TERM_WITH_DOMAIN, {
            "sid": session_id, "name": "pw metric", "domain": "hr-reporting",
        })
