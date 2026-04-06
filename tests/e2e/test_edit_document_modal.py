# Copyright (c) 2025 Kenneth Stott
# Canary: c697f9c4-cf16-43ae-8b64-fd9fef2fff7f
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Playwright UI tests: EditDocumentModal validation.

Covers all field-level validation errors, the URI Test button (reachable /
unreachable), save enablement, and cancel behaviour.

Requires: running Constat backend + Vite dev server (pytest.mark.e2e).
"""

from __future__ import annotations

import uuid

import pytest
import requests

pytestmark = pytest.mark.e2e

# ---------------------------------------------------------------------------
# GraphQL helpers
# ---------------------------------------------------------------------------

_ADD_DOCUMENT_URI = """
mutation AddDocumentUri($sessionId: String!, $input: DocumentUriInput!) {
  addDocumentUri(sessionId: $sessionId, input: $input) {
    status
    name
  }
}
"""


def _gql(server_url: str, query: str, variables: dict) -> dict:
    resp = requests.post(
        f"{server_url}/api/graphql",
        json={"query": query, "variables": variables},
        timeout=30,
    )
    assert resp.status_code == 200, f"GraphQL HTTP error: {resp.text}"
    body = resp.json()
    if body.get("errors"):
        pytest.fail(f"GraphQL errors: {body['errors']}")
    return body["data"]


# ---------------------------------------------------------------------------
# Navigation helpers
# ---------------------------------------------------------------------------

def _navigate_and_expand_documents(page, ui_url: str, session_id: str) -> None:
    """Navigate to the app, inject session, pre-expand context + documents sections."""
    page.goto(f"{ui_url}/")
    page.evaluate(f"localStorage.setItem('constat-session-id', '{session_id}');")
    page.evaluate("""
        const stored = JSON.parse(localStorage.getItem('constat-ui-storage') || '{}');
        const state = stored.state || {};
        const sections = new Set(state.expandedArtifactSections || ['artifacts']);
        sections.add('context');
        sections.add('documents');
        state.expandedArtifactSections = Array.from(sections);
        stored.state = state;
        localStorage.setItem('constat-ui-storage', JSON.stringify(stored));
    """)
    page.reload()
    page.wait_for_selector("text=What can I help you with?", timeout=30000)
    toggle_btn = page.locator("button[title='Show details panel']")
    if toggle_btn.count() > 0:
        try:
            toggle_btn.first.click(timeout=2000)
        except Exception as e:
            pytest.fail(f"UI action failed: {e}")
    page.wait_for_selector("#section-documents", timeout=30000)


def _open_edit_document_modal(page, doc_name: str) -> None:
    """Hover over the document row and click the Edit document pencil button."""
    page.locator(f"text={doc_name}").first.hover()
    edit_btn = page.locator("button[title='Edit document']").first
    edit_btn.wait_for(timeout=5000)
    edit_btn.click()
    page.wait_for_selector(f"h3:has-text('Edit Document — {doc_name}')", timeout=5000)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="class")
def doc_session(server_url):
    """Create a session with a dynamic document source pre-seeded.

    The document is added via GraphQL so the Edit button is available in the
    Documents section (only session/dynamic docs expose the pencil button).
    """
    sid = str(uuid.uuid4())
    resp = requests.post(f"{server_url}/api/sessions", json={"session_id": sid})
    assert resp.status_code == 200, f"Create session failed: {resp.text}"

    doc_name = f"testdoc_{sid[:8]}"
    _gql(server_url, _ADD_DOCUMENT_URI, {
        "sessionId": sid,
        "input": {
            "name": doc_name,
            "url": "https://example.com",
            "description": "A test document",
        },
    })

    yield {"session_id": sid, "doc_name": doc_name, "server_url": server_url}

    requests.delete(f"{server_url}/api/sessions/{sid}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEditDocumentModalValidation:
    """Field-level validation in EditDocumentModal."""

    def test_blank_name_shows_error(self, page, ui_url, doc_session):
        """Clearing Name and clicking Save shows 'Name is required'."""
        _navigate_and_expand_documents(page, ui_url, doc_session["session_id"])
        _open_edit_document_modal(page, doc_session["doc_name"])

        # Clear the Name field
        name_input = page.locator("input[type='text']").first
        name_input.triple_click()
        name_input.fill("")

        # Intercept the mutation — Save must not be called
        mutation_called = {"called": False}

        def on_request(req):
            if "updateDocument" in (req.post_data or ""):
                mutation_called["called"] = True

        page.on("request", on_request)
        page.locator("button:has-text('Save')").click()

        page.wait_for_selector("text=Name is required", timeout=5000)
        assert not mutation_called["called"], "updateDocument mutation must NOT be called on validation failure"

    def test_blank_description_shows_error(self, page, ui_url, doc_session):
        """Clearing Description and clicking Save shows 'Description is required'."""
        _navigate_and_expand_documents(page, ui_url, doc_session["session_id"])
        _open_edit_document_modal(page, doc_session["doc_name"])

        desc = page.locator("textarea").first
        desc.triple_click()
        desc.fill("")

        page.locator("button:has-text('Save')").click()
        page.wait_for_selector("text=Description is required", timeout=5000)

    def test_blank_uri_shows_error(self, page, ui_url, doc_session):
        """Clearing URI and clicking Save shows 'URI or path is required'."""
        _navigate_and_expand_documents(page, ui_url, doc_session["session_id"])
        _open_edit_document_modal(page, doc_session["doc_name"])

        # URI input is the second text input (after Name)
        uri_input = page.locator("input[placeholder*='https://example.com or']")
        uri_input.triple_click()
        uri_input.fill("")

        page.locator("button:has-text('Save')").click()
        page.wait_for_selector("text=URI or path is required", timeout=5000)

    def test_invalid_url_shows_error(self, page, ui_url, doc_session):
        """Entering an invalid URL and clicking Save shows 'Enter a valid URL'."""
        _navigate_and_expand_documents(page, ui_url, doc_session["session_id"])
        _open_edit_document_modal(page, doc_session["doc_name"])

        uri_input = page.locator("input[placeholder*='https://example.com or']")
        uri_input.triple_click()
        uri_input.fill("not-a-url")

        page.locator("button:has-text('Save')").click()
        page.wait_for_selector("text=Enter a valid URL", timeout=5000)

    def test_cancel_closes_modal(self, page, ui_url, doc_session):
        """Clicking Cancel dismisses the modal."""
        _navigate_and_expand_documents(page, ui_url, doc_session["session_id"])
        _open_edit_document_modal(page, doc_session["doc_name"])

        page.locator("button:has-text('Cancel')").click()
        page.wait_for_selector(
            f"h3:has-text('Edit Document — {doc_session['doc_name']}')",
            state="detached",
            timeout=5000,
        )

    def test_uri_field_prepopulated(self, page, ui_url, doc_session):
        """URI field must be pre-populated with the document's URL when modal opens."""
        _navigate_and_expand_documents(page, ui_url, doc_session["session_id"])
        _open_edit_document_modal(page, doc_session["doc_name"])

        uri_input = page.locator("input[placeholder*='https://example.com or']")
        actual_uri = uri_input.input_value()
        assert actual_uri == "https://example.com", (
            f"URI field must be pre-populated with the document URL; got: {actual_uri!r}"
        )

    def test_valid_form_enables_save(self, page, ui_url, doc_session):
        """With all fields filled validly, Save is enabled and shows no errors."""
        _navigate_and_expand_documents(page, ui_url, doc_session["session_id"])
        _open_edit_document_modal(page, doc_session["doc_name"])

        # All fields must be pre-filled (Name, Description, URI)
        desc = page.locator("textarea").first
        assert desc.input_value(), "Description must be pre-populated when modal opens"

        uri_input = page.locator("input[placeholder*='https://example.com or']")
        assert uri_input.input_value() == "https://example.com", (
            "URI must be pre-populated with 'https://example.com' when modal opens"
        )

        save_btn = page.locator("button:has-text('Save')")
        assert save_btn.get_attribute("disabled") is None, "Save must be enabled when form is valid"
        assert page.locator("text=Name is required").count() == 0
        assert page.locator("text=Description is required").count() == 0
        assert page.locator("text=URI or path is required").count() == 0


class TestEditDocumentUriTestButton:
    """Test button reachability checks in the URI / Path field."""

    def test_test_button_unreachable_url(self, page, ui_url, doc_session):
        """Clicking Test with an unreachable URL shows the ✗ error indicator."""
        _navigate_and_expand_documents(page, ui_url, doc_session["session_id"])
        _open_edit_document_modal(page, doc_session["doc_name"])

        uri_input = page.locator("input[placeholder*='https://example.com or']")
        uri_input.triple_click()
        uri_input.fill("https://this-host-does-not-exist-xyz.invalid/doc")

        page.locator("button:has-text('Test')").click()

        # Wait for the async check to resolve; the ✗ mark and an error message must appear
        error_mark = page.locator("text=✗")
        error_mark.wait_for(timeout=30000)
        assert error_mark.is_visible(), "Error indicator ✗ must be visible after unreachable URI test"

        # An error message should also appear below the URI field
        uri_error = page.locator("p.text-red-500, p.text-red-400").first
        uri_error.wait_for(timeout=5000)
        assert uri_error.is_visible(), "Error message must appear after unreachable URI test"

    def test_test_button_reachable_url(self, page, ui_url, doc_session):
        """Clicking Test with a known-reachable URL shows the ✓ success indicator."""
        _navigate_and_expand_documents(page, ui_url, doc_session["session_id"])
        _open_edit_document_modal(page, doc_session["doc_name"])

        uri_input = page.locator("input[placeholder*='https://example.com or']")
        uri_input.triple_click()
        uri_input.fill("https://example.com")

        page.locator("button:has-text('Test')").click()

        success_mark = page.locator("text=✓")
        success_mark.wait_for(timeout=30000)
        assert success_mark.is_visible(), "Success indicator ✓ must be visible after reachable URI test"
