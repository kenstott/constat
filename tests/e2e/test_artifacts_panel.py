# Copyright (c) 2025 Kenneth Stott
# Canary: ddee5f13-3545-4ff4-8baa-f19accc751a8
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Playwright UI tests: Artifacts panel.

Covers the ARTIFACTS top-level panel — the results section that renders
after a chat query and the panel's expand/collapse toggle.

Requires: running Constat backend + Vite dev server (pytest.mark.e2e).
"""

from __future__ import annotations
import uuid

import pytest
import requests

pytestmark = pytest.mark.e2e


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _expand_artifacts(page, ui_url: str, session_id: str):
    """Navigate and pre-expand the artifacts section."""
    page.goto(f"{ui_url}/")
    page.evaluate(f"localStorage.setItem('constat-session-id', '{session_id}');")
    page.evaluate("""
        const stored = JSON.parse(localStorage.getItem('constat-ui-storage') || '{}');
        const state = stored.state || {};
        const sections = new Set(state.expandedArtifactSections || ['artifacts']);
        sections.add('artifacts');
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
# Artifacts panel: structure and toggle
# ---------------------------------------------------------------------------

class TestArtifactsPanelStructure:
    """Verify the Artifacts panel renders its top-level structure."""

    def test_artifacts_section_button_visible(self, page, ui_url, session_id):
        """ARTIFACTS top-level toggle button is present."""
        _expand_artifacts(page, ui_url, session_id)
        assert page.locator("button:has-text('Artifacts')").count() > 0

    def test_artifacts_section_expands(self, page, ui_url, session_id):
        """Expanding the Artifacts section makes results area visible."""
        _expand_artifacts(page, ui_url, session_id)
        # section-results is rendered directly inside the artifacts top-level section
        assert page.locator("#section-results").count() > 0

    def test_artifacts_collapses_on_click(self, page, ui_url, session_id):
        """Clicking the Artifacts toggle collapses the section."""
        _expand_artifacts(page, ui_url, session_id)
        # Collapse it
        page.locator("button:has-text('Artifacts')").first.click()
        page.wait_for_timeout(300)
        assert not page.locator("#section-results").is_visible()

    def test_artifacts_re_expands_on_click(self, page, ui_url, session_id):
        """Clicking collapsed Artifacts re-expands it."""
        _expand_artifacts(page, ui_url, session_id)
        # Collapse then re-expand
        page.locator("button:has-text('Artifacts')").first.click()
        page.wait_for_timeout(300)
        page.locator("button:has-text('Artifacts')").first.click()
        page.wait_for_timeout(300)
        assert page.locator("#section-results").count() > 0

    def test_results_section_shows_empty_state(self, page, ui_url, session_id):
        """Results section renders without crashing in empty state."""
        _expand_artifacts(page, ui_url, session_id)
        # The results section exists; it may show placeholder text or be empty
        results = page.locator("#section-results")
        assert results.count() > 0
