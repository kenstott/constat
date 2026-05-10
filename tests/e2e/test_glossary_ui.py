from __future__ import annotations

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

import pytest
import requests

pytestmark = pytest.mark.e2e


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


# ---------------------------------------------------------------------------
# GraphQL query/mutation strings (shared with split modules)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Navigation helpers
# ---------------------------------------------------------------------------

def _navigate_and_wait(page, ui_url: str, session_id: str):
    """Navigate to the app with the given session, open the artifact panel, and wait for Glossary section."""
    page.goto(f"{ui_url}/")
    page.evaluate(f"localStorage.setItem('constat-session-id', '{session_id}');")
    page.reload()
    page.wait_for_selector("text=What can I help you with?", timeout=30000)
    toggle_btn = page.locator("button[title='Show details panel']")
    if toggle_btn.count() > 0:
        try:
            toggle_btn.first.click(timeout=2000)
        except Exception as e:
            pytest.fail(f"UI action failed: {e}")
    page.wait_for_selector("#section-glossary", timeout=30000)


def _expand_glossary(page):
    """Wait for the Glossary section to be visible (panel is always expanded)."""
    page.locator("#section-glossary").first.wait_for(timeout=5000)
    page.wait_for_timeout(500)


# ---------------------------------------------------------------------------
# Panel rendering tests
# ---------------------------------------------------------------------------

class TestGlossaryPanelLoads:
    """Verify the glossary panel loads and displays terms from Apollo."""

    def test_glossary_section_visible(self, page, ui_url, server_url, session_id):
        """Glossary section is present in the artifact panel."""
        _navigate_and_wait(page, ui_url, session_id)
        assert page.locator("#section-glossary").first.is_visible()

    def test_glossary_terms_render(self, page, ui_url, seeded_session):
        """Terms seeded via GraphQL appear in the glossary panel."""
        sid = seeded_session["session_id"]
        _navigate_and_wait(page, ui_url, sid)
        _expand_glossary(page)

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
        assert "(0)" not in header_text, f"Expected terms in '{header_text}'"

    def test_term_expand_shows_details(self, page, ui_url, seeded_session):
        """Clicking a term expands it to show details."""
        sid = seeded_session["session_id"]
        _navigate_and_wait(page, ui_url, sid)
        _expand_glossary(page)

        term_el = page.locator("text=Revenue").first
        term_el.click()
        page.wait_for_timeout(500)

        assert page.locator("text=Total income generated").first.is_visible()

    def test_search_filters_terms(self, page, ui_url, seeded_session):
        """Typing in search filters the displayed terms."""
        sid = seeded_session["session_id"]
        _navigate_and_wait(page, ui_url, sid)
        _expand_glossary(page)

        search = page.locator("#section-glossary input[placeholder]").first
        search.fill("revenue")
        page.wait_for_timeout(500)

        assert page.locator("text=Revenue").first.is_visible()
