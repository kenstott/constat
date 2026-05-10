# Copyright (c) 2025 Kenneth Stott
# Canary: ddee5f13-3545-4ff4-8baa-f19accc751a8
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Playwright UI tests: Debug panel.

Covers the DEBUG top-level panel — expand/collapse behavior and all
sub-section accordion presence (Scratchpad, Exploratory Code,
Inference Code, Intermediate Results, Session DDL).

Requires: running Constat backend + Vite dev server (pytest.mark.e2e).
"""

from __future__ import annotations
import uuid

import pytest
import requests

pytestmark = pytest.mark.e2e

_DEBUG_SECTIONS = ["scratchpad", "code", "inference-code", "intermediate-results", "session-ddl"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _expand_debug(page, ui_url: str, session_id: str):
    """Navigate and pre-expand the debug section."""
    page.goto(f"{ui_url}/")
    page.evaluate(f"localStorage.setItem('constat-session-id', '{session_id}');")
    page.evaluate("""
        const stored = JSON.parse(localStorage.getItem('constat-ui-storage') || '{}');
        const state = stored.state || {};
        const sections = new Set(state.expandedArtifactSections || ['artifacts']);
        sections.add('debug');
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
# Debug panel: structure
# ---------------------------------------------------------------------------

class TestDebugPanelStructure:
    """Verify the Debug panel renders and contains all expected sub-sections."""

    def test_debug_toggle_button_visible(self, page, ui_url, session_id):
        """DEBUG top-level toggle button is present."""
        _expand_debug(page, ui_url, session_id)
        assert page.locator("button:has-text('Debug')").count() > 0

    def test_debug_collapses_on_click(self, page, ui_url, session_id):
        """Clicking the Debug toggle collapses the section."""
        _expand_debug(page, ui_url, session_id)
        page.locator("button:has-text('Debug')").first.click()
        page.wait_for_timeout(300)
        # Sub-sections should be hidden
        assert not page.locator("#section-scratchpad").is_visible()

    def test_debug_re_expands(self, page, ui_url, session_id):
        """Clicking collapsed Debug re-expands it."""
        _expand_debug(page, ui_url, session_id)
        page.locator("button:has-text('Debug')").first.click()
        page.wait_for_timeout(300)
        page.locator("button:has-text('Debug')").first.click()
        page.wait_for_timeout(300)
        # At least the scratchpad section toggle should reappear
        assert page.locator("button:has-text('Scratchpad')").count() > 0 or \
               page.locator("#section-scratchpad").count() > 0

    def test_scratchpad_section_present(self, page, ui_url, session_id):
        """Scratchpad accordion button is present within the Debug section."""
        _expand_debug(page, ui_url, session_id)
        assert page.locator("button:has-text('Scratchpad')").count() > 0

    def test_exploratory_code_section_present(self, page, ui_url, session_id):
        """Exploratory Code accordion button is present."""
        _expand_debug(page, ui_url, session_id)
        # The 'code' section renders as "Exploratory Code" or similar label
        assert page.locator("button:has-text('Exploratory Code')").count() > 0 or \
               page.locator("#section-code").count() > 0

    def test_inference_code_section_present(self, page, ui_url, session_id):
        """Inference Code accordion button is present."""
        _expand_debug(page, ui_url, session_id)
        assert page.locator("button:has-text('Inference Code')").count() > 0 or \
               page.locator("#section-inference-code").count() > 0

    def test_session_ddl_section_present(self, page, ui_url, session_id):
        """Session DDL accordion button is present."""
        _expand_debug(page, ui_url, session_id)
        assert page.locator("button:has-text('Session DDL')").count() > 0 or \
               page.locator("#section-session-ddl").count() > 0
