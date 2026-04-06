# Copyright (c) 2025 Kenneth Stott
# Canary: ddee5f13-3545-4ff4-8baa-f19accc751a8
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Playwright UI tests: Context panel (Sources, Glossary, Reasoning sub-sections).

Covers:
- Context panel toggle and sub-section visibility
- Sources: Remove Database / Remove API
- Sources: Facts CRUD (add, edit, delete)
- Reasoning: agents and skills CRUD button presence
- Session prompt edit button

Requires: running Constat backend + Vite dev server (pytest.mark.e2e).
"""

from __future__ import annotations
import os
import tempfile
import uuid

import pytest
import requests

pytestmark = pytest.mark.e2e


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _navigate_and_expand_context(page, ui_url: str, session_id: str, extra_sections=None):
    """Navigate to the app and pre-expand the context panel + given sub-sections."""
    page.goto(f"{ui_url}/")
    page.evaluate(f"localStorage.setItem('constat-session-id', '{session_id}');")
    sections = {'context', 'databases', 'apis', 'facts', 'session-prompt', 'agents', 'skills'}
    if extra_sections:
        sections.update(extra_sections)
    sections_json = str(list(sections)).replace("'", '"')
    page.evaluate(f"""
        const stored = JSON.parse(localStorage.getItem('constat-ui-storage') || '{{}}');
        const state = stored.state || {{}};
        state.expandedArtifactSections = {sections_json};
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
    page.wait_for_selector("#section-databases", timeout=30000)


def _add_database_via_ui(page, name: str, db_path: str):
    """Open Add Database modal and submit a SQLite entry."""
    page.locator("#section-databases").wait_for(timeout=10000)
    page.locator("button[title='Add database']").first.click(timeout=5000)
    page.wait_for_selector("h3:has-text('Add Database')", timeout=5000)
    page.fill("input[placeholder='Name']", name)
    page.select_option("select >> nth=0", value="sqlite")
    page.fill("input[placeholder*='.db']", db_path)
    page.locator("button:has-text('Add')").click()
    page.wait_for_selector("h3:has-text('Add Database')", state="detached", timeout=10000)
    page.wait_for_selector(f"text={name}", timeout=10000)


def _add_api_via_ui(page, name: str, url: str):
    """Open Add API modal and submit a REST API entry."""
    page.locator("#section-apis").wait_for(timeout=10000)
    page.locator("button[title='Add API']").first.click(timeout=5000)
    page.wait_for_selector("h3:has-text('Add API')", timeout=5000)
    page.fill("input[placeholder='Name']", name)
    page.fill("input[placeholder*='Base URL']", url)
    page.locator("button:has-text('Add')").click()
    page.wait_for_selector("h3:has-text('Add API')", state="detached", timeout=10000)
    page.wait_for_selector(f"text={name}", timeout=10000)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="class")
def session_id(server_url):
    body = {"session_id": str(uuid.uuid4())}
    resp = requests.post(f"{server_url}/api/sessions", json=body)
    assert resp.status_code == 200
    sid = resp.json()["session_id"]
    yield sid
    requests.delete(f"{server_url}/api/sessions/{sid}")


# ---------------------------------------------------------------------------
# Context panel: top-level structure
# ---------------------------------------------------------------------------

class TestContextPanelStructure:
    """Verify the Context panel top-level toggle and sub-section presence."""

    def test_context_toggle_button_visible(self, page, ui_url, session_id):
        _navigate_and_expand_context(page, ui_url, session_id)
        assert page.locator("button:has-text('Context')").count() > 0

    def test_context_collapses_on_click(self, page, ui_url, session_id):
        _navigate_and_expand_context(page, ui_url, session_id)
        page.locator("button:has-text('Context')").first.click()
        page.wait_for_timeout(300)
        assert not page.locator("#section-databases").is_visible()

    def test_context_re_expands(self, page, ui_url, session_id):
        _navigate_and_expand_context(page, ui_url, session_id)
        page.locator("button:has-text('Context')").first.click()
        page.wait_for_timeout(300)
        page.locator("button:has-text('Context')").first.click()
        page.wait_for_selector("#section-databases", timeout=10000)

    def test_databases_section_present(self, page, ui_url, session_id):
        _navigate_and_expand_context(page, ui_url, session_id)
        assert page.locator("#section-databases").is_visible()

    def test_apis_section_present(self, page, ui_url, session_id):
        _navigate_and_expand_context(page, ui_url, session_id)
        assert page.locator("#section-apis").is_visible()

    def test_facts_section_present(self, page, ui_url, session_id):
        _navigate_and_expand_context(page, ui_url, session_id)
        assert page.locator("#section-facts").is_visible()

    def test_glossary_section_present(self, page, ui_url, session_id):
        _navigate_and_expand_context(page, ui_url, session_id)
        assert page.locator("#section-glossary").count() > 0


# ---------------------------------------------------------------------------
# Sources: Remove Database
# ---------------------------------------------------------------------------

class TestRemoveDatabase:
    """Add then remove a database and verify it disappears from the list."""

    def test_remove_database(self, page, ui_url, server_url):
        sid = str(uuid.uuid4())
        resp = requests.post(f"{server_url}/api/sessions", json={"session_id": sid})
        assert resp.status_code == 200

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            sqlite_path = f.name

        try:
            _navigate_and_expand_context(page, ui_url, sid)
            name = f"rmdb_{sid[:8]}"
            _add_database_via_ui(page, name, sqlite_path)

            # Hover over the row to reveal the Remove button
            page.locator(f"text={name}").first.hover()
            remove_btn = page.locator("button[title='Remove database']").first
            remove_btn.wait_for(timeout=5000)
            remove_btn.click()

            # Confirm deletion dialog if present
            confirm = page.locator("button:has-text('Delete')")
            if confirm.count() > 0:
                confirm.first.click(timeout=3000)

            # Name no longer visible in the list
            page.wait_for_selector(f"text={name}", state="detached", timeout=10000)
        finally:
            if os.path.exists(sqlite_path):
                os.unlink(sqlite_path)
            requests.delete(f"{server_url}/api/sessions/{sid}")


# ---------------------------------------------------------------------------
# Sources: Remove API
# ---------------------------------------------------------------------------

class TestRemoveApi:
    """Add then remove an API and verify it disappears."""

    def test_remove_api(self, page, ui_url, server_url):
        sid = str(uuid.uuid4())
        resp = requests.post(f"{server_url}/api/sessions", json={"session_id": sid})
        assert resp.status_code == 200

        try:
            _navigate_and_expand_context(page, ui_url, sid)
            name = f"rmapi_{sid[:8]}"
            _add_api_via_ui(page, name, "https://api.example.com")

            # Hover to reveal Remove button
            page.locator(f"text={name}").first.hover()
            remove_btn = page.locator("button[title='Remove API']").first
            remove_btn.wait_for(timeout=5000)
            remove_btn.click()

            confirm = page.locator("button:has-text('Delete')")
            if confirm.count() > 0:
                confirm.first.click(timeout=3000)

            page.wait_for_selector(f"text={name}", state="detached", timeout=10000)
        finally:
            requests.delete(f"{server_url}/api/sessions/{sid}")


# ---------------------------------------------------------------------------
# Sources: Facts CRUD
# ---------------------------------------------------------------------------

class TestFactsCrud:
    """Add, edit, and delete facts via the UI."""

    def test_add_fact(self, page, ui_url, server_url):
        """Add a fact and verify it appears in the facts table."""
        sid = str(uuid.uuid4())
        resp = requests.post(f"{server_url}/api/sessions", json={"session_id": sid})
        assert resp.status_code == 200

        try:
            _navigate_and_expand_context(page, ui_url, sid)
            page.locator("#section-facts").wait_for(timeout=10000)

            page.locator("button[title='Add fact']").first.click(timeout=5000)
            page.wait_for_selector("input[placeholder='Name']", timeout=5000)

            fact_name = f"fact_{sid[:8]}"
            page.fill("input[placeholder='Name']", fact_name)
            page.fill("input[placeholder='Value']", "test_value")
            page.locator("button:has-text('Add')").last.click()

            page.wait_for_selector(f"text={fact_name}", timeout=10000)
        finally:
            requests.delete(f"{server_url}/api/sessions/{sid}")

    def test_edit_fact(self, page, ui_url, server_url):
        """Add a fact then edit its value."""
        sid = str(uuid.uuid4())
        resp = requests.post(f"{server_url}/api/sessions", json={"session_id": sid})
        assert resp.status_code == 200

        try:
            _navigate_and_expand_context(page, ui_url, sid)
            page.locator("#section-facts").wait_for(timeout=10000)

            page.locator("button[title='Add fact']").first.click(timeout=5000)
            page.wait_for_selector("input[placeholder='Name']", timeout=5000)

            fact_name = f"editfact_{sid[:8]}"
            page.fill("input[placeholder='Name']", fact_name)
            page.fill("input[placeholder='Value']", "original")
            page.locator("button:has-text('Add')").last.click()
            page.wait_for_selector(f"text={fact_name}", timeout=10000)

            # Hover over fact row to reveal Edit button
            page.locator(f"text={fact_name}").first.hover()
            edit_btn = page.locator("button[title='Edit value']").first
            edit_btn.wait_for(timeout=5000)
            edit_btn.click()

            # The inline edit input should appear
            edit_input = page.locator("input[value='original']")
            edit_input.wait_for(timeout=5000)
            edit_input.fill("updated")
            page.locator("text=Save").first.click()

            page.wait_for_selector("text=updated", timeout=10000)
        finally:
            requests.delete(f"{server_url}/api/sessions/{sid}")

    def test_delete_fact(self, page, ui_url, server_url):
        """Add a fact then forget (delete) it."""
        sid = str(uuid.uuid4())
        resp = requests.post(f"{server_url}/api/sessions", json={"session_id": sid})
        assert resp.status_code == 200

        try:
            _navigate_and_expand_context(page, ui_url, sid)
            page.locator("#section-facts").wait_for(timeout=10000)

            page.locator("button[title='Add fact']").first.click(timeout=5000)
            page.wait_for_selector("input[placeholder='Name']", timeout=5000)

            fact_name = f"delfact_{sid[:8]}"
            page.fill("input[placeholder='Name']", fact_name)
            page.fill("input[placeholder='Value']", "to_delete")
            page.locator("button:has-text('Add')").last.click()
            page.wait_for_selector(f"text={fact_name}", timeout=10000)

            # Hover to reveal Forget button
            page.locator(f"text={fact_name}").first.hover()
            forget_btn = page.locator("button[title='Forget fact']").first
            forget_btn.wait_for(timeout=5000)
            forget_btn.click()

            page.wait_for_selector(f"text={fact_name}", state="detached", timeout=10000)
        finally:
            requests.delete(f"{server_url}/api/sessions/{sid}")

    def test_add_fact_cancel(self, page, ui_url, session_id):
        """Cancelling the Add Fact modal does not add a fact."""
        _navigate_and_expand_context(page, ui_url, session_id)
        page.locator("#section-facts").wait_for(timeout=10000)

        page.locator("button[title='Add fact']").first.click(timeout=5000)
        page.wait_for_selector("input[placeholder='Name']", timeout=5000)
        page.fill("input[placeholder='Name']", "cancelled_fact")

        page.locator("button:has-text('Cancel')").last.click()
        page.wait_for_timeout(500)

        assert page.locator("text=cancelled_fact").count() == 0


# ---------------------------------------------------------------------------
# Reasoning: agents and skills button presence
# ---------------------------------------------------------------------------

class TestReasoningSection:
    """Verify the Reasoning sub-section contains CRUD controls."""

    def test_create_agent_button_present(self, page, ui_url, session_id):
        _navigate_and_expand_context(page, ui_url, session_id)
        page.locator("#section-agents").wait_for(timeout=10000)
        assert page.locator("button[title='Create agent']").count() > 0

    def test_create_skill_button_present(self, page, ui_url, session_id):
        _navigate_and_expand_context(page, ui_url, session_id)
        page.locator("#section-skills").wait_for(timeout=10000)
        assert page.locator("button[title='Create skill']").count() > 0

    def test_session_prompt_edit_button_present(self, page, ui_url, session_id):
        _navigate_and_expand_context(page, ui_url, session_id)
        page.locator("#section-session-prompt").wait_for(timeout=10000)
        assert page.locator("button[title='Edit session prompt']").count() > 0

    def test_session_prompt_edit_opens_editor(self, page, ui_url, session_id):
        _navigate_and_expand_context(page, ui_url, session_id)
        page.locator("#section-session-prompt").wait_for(timeout=10000)
        page.locator("button[title='Edit session prompt']").first.click()
        # Editor should appear (textarea or contenteditable)
        editor = page.locator("textarea").or_(page.locator("[contenteditable='true']"))
        editor.first.wait_for(timeout=5000)
        assert editor.count() > 0
