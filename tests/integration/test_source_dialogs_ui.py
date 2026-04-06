# Copyright (c) 2025 Kenneth Stott
# Canary: ddee5f13-3545-4ff4-8baa-f19accc751a8
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Playwright UI tests: Add Database and Add API dialogs.

Tests verify per-type field rendering, auth type selectors, and successful
submission through the real Apollo Client → GraphQL backend.

Requires: running Constat backend + Vite dev server (pytest.mark.integration).
"""

import uuid

import pytest
import requests

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _navigate_and_wait_sources(page, ui_url: str, session_id: str):
    """Navigate to the app, inject session, open artifact panel, wait for Sources section."""
    page.goto(f"{ui_url}/")
    page.evaluate(f"localStorage.setItem('constat-session-id', '{session_id}');")
    page.reload()
    page.wait_for_selector("text=What can I help you with?", timeout=30000)
    toggle_btn = page.locator("button[title='Show details panel']")
    if toggle_btn.count() > 0:
        try:
            toggle_btn.first.click(timeout=2000)
        except Exception:
            pass
    page.wait_for_selector("#section-databases", timeout=30000)


def _open_add_database_modal(page):
    """Click the + button next to Databases to open the Add Database modal."""
    databases_header = page.locator("#section-databases")
    databases_header.wait_for(timeout=10000)
    add_btn = page.locator("#section-databases").locator("button[title='Add database'], button[aria-label*='Add'], button:has-text('+')").first
    # Fallback: look for any + button near databases
    if not add_btn.is_visible():
        add_btn = page.locator("button[title='Add database']").first
    add_btn.click(timeout=5000)
    page.wait_for_selector("text=Add Database", timeout=5000)


def _open_add_api_modal(page):
    """Click the + button next to APIs to open the Add API modal."""
    apis_header = page.locator("#section-apis")
    apis_header.wait_for(timeout=10000)
    add_btn = page.locator("#section-apis").locator("button[title='Add API'], button[aria-label*='Add'], button:has-text('+')").first
    if not add_btn.is_visible():
        add_btn = page.locator("button[title='Add API']").first
    add_btn.click(timeout=5000)
    page.wait_for_selector("text=Add API", timeout=5000)


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


# ---------------------------------------------------------------------------
# Add Database modal: field rendering by type
# ---------------------------------------------------------------------------

class TestAddDatabaseModalFields:
    """Verify per-type field rendering in the Add Database dialog."""

    def test_modal_opens_with_type_selector(self, page, ui_url, session_id):
        _navigate_and_wait_sources(page, ui_url, session_id)
        _open_add_database_modal(page)
        assert page.locator("select").first.is_visible()
        assert page.locator("input[placeholder='Name']").is_visible()

    def test_postgresql_shows_host_port_db_credentials(self, page, ui_url, session_id):
        _navigate_and_wait_sources(page, ui_url, session_id)
        _open_add_database_modal(page)
        page.select_option("select >> nth=0", value="postgresql")
        assert page.locator("input[placeholder='localhost']").is_visible()
        assert page.locator("input[placeholder='5432']").is_visible()
        assert page.locator("input[placeholder='database']").is_visible()
        assert page.locator("input[placeholder='user']").is_visible()
        assert page.locator("input[placeholder='password']").is_visible()

    def test_mssql_shows_port_1433(self, page, ui_url, session_id):
        _navigate_and_wait_sources(page, ui_url, session_id)
        _open_add_database_modal(page)
        page.select_option("select >> nth=0", value="mssql")
        assert page.locator("input[placeholder='1433']").is_visible()

    def test_duckdb_shows_file_path_not_network(self, page, ui_url, session_id):
        _navigate_and_wait_sources(page, ui_url, session_id)
        _open_add_database_modal(page)
        page.select_option("select >> nth=0", value="duckdb")
        assert page.locator("input[placeholder*='.duckdb']").is_visible()
        assert not page.locator("input[placeholder='localhost']").is_visible()

    def test_custom_shows_connection_uri(self, page, ui_url, session_id):
        _navigate_and_wait_sources(page, ui_url, session_id)
        _open_add_database_modal(page)
        page.select_option("select >> nth=0", value="custom")
        assert page.locator("input[placeholder*='dialect']").is_visible()

    def test_elasticsearch_shows_auth_selector(self, page, ui_url, session_id):
        _navigate_and_wait_sources(page, ui_url, session_id)
        _open_add_database_modal(page)
        page.select_option("select >> nth=0", value="elasticsearch")
        assert page.locator("select >> nth=1").is_visible()
        # Switch to API key auth
        page.select_option("select >> nth=1", value="api_key")
        assert page.locator("input[placeholder*='base64']").is_visible()

    def test_elasticsearch_bearer_shows_token_field(self, page, ui_url, session_id):
        _navigate_and_wait_sources(page, ui_url, session_id)
        _open_add_database_modal(page)
        page.select_option("select >> nth=0", value="elasticsearch")
        page.select_option("select >> nth=1", value="bearer")
        assert page.locator("input[placeholder='token']").is_visible()

    def test_dynamodb_shows_env_and_credentials_options(self, page, ui_url, session_id):
        _navigate_and_wait_sources(page, ui_url, session_id)
        _open_add_database_modal(page)
        page.select_option("select >> nth=0", value="dynamodb")
        # Switch to credentials auth
        page.select_option("select >> nth=1", value="credentials")
        assert page.locator("input[placeholder*='AKIA']").is_visible()
        assert page.locator("input[placeholder='secret']").is_visible()

    def test_dynamodb_env_shows_iam_hint(self, page, ui_url, session_id):
        _navigate_and_wait_sources(page, ui_url, session_id)
        _open_add_database_modal(page)
        page.select_option("select >> nth=0", value="dynamodb")
        # Default is 'env' — should show IAM hint
        assert page.locator("text=IAM").first.is_visible()

    def test_firestore_shows_project_field_and_auth(self, page, ui_url, session_id):
        _navigate_and_wait_sources(page, ui_url, session_id)
        _open_add_database_modal(page)
        page.select_option("select >> nth=0", value="firestore")
        assert page.locator("input[placeholder='my-gcp-project']").is_visible()
        page.select_option("select >> nth=1", value="service_account")
        assert page.locator("input[placeholder*='credentials.json']").is_visible()

    def test_cancel_closes_modal(self, page, ui_url, session_id):
        _navigate_and_wait_sources(page, ui_url, session_id)
        _open_add_database_modal(page)
        page.locator("button:has-text('Cancel')").click()
        assert not page.locator("text=Add Database").is_visible()


# ---------------------------------------------------------------------------
# Add Database modal: submission
# ---------------------------------------------------------------------------

class TestAddDatabaseSubmission:
    """Verify Add Database form submits correctly for SQL types."""

    def test_add_postgresql_database(self, page, ui_url, server_url):
        """Add a PostgreSQL database entry and verify it appears in the sources panel."""
        sid = str(uuid.uuid4())
        resp = requests.post(f"{server_url}/api/sessions", json={"session_id": sid})
        assert resp.status_code == 200

        _navigate_and_wait_sources(page, ui_url, sid)
        _open_add_database_modal(page)

        unique_name = f"pgtest_{sid[:8]}"
        page.fill("input[placeholder='Name']", unique_name)
        page.select_option("select >> nth=0", value="postgresql")
        page.fill("input[placeholder='localhost']", "pg-host.example.com")
        page.fill("input[placeholder='database']", "testdb")
        page.fill("input[placeholder='user']", "pguser")
        page.fill("input[placeholder='password']", "pgpass")

        page.locator("button:has-text('Add')").click()

        # Modal closes
        page.wait_for_selector("text=Add Database", state="detached", timeout=10000)

        # DB appears in sources list
        page.wait_for_selector(f"text={unique_name}", timeout=10000)

        requests.delete(f"{server_url}/api/sessions/{sid}")


# ---------------------------------------------------------------------------
# Add API modal: field rendering by auth type
# ---------------------------------------------------------------------------

class TestAddApiModalFields:
    """Verify per-auth-type field rendering in the Add API dialog."""

    def test_modal_opens_with_name_url_type_auth(self, page, ui_url, session_id):
        _navigate_and_wait_sources(page, ui_url, session_id)
        _open_add_api_modal(page)
        assert page.locator("input[placeholder='Name']").is_visible()
        assert page.locator("input[placeholder*='Base URL']").is_visible()
        assert page.locator("text=Type").is_visible()
        assert page.locator("text=Authentication").is_visible()

    def test_default_auth_is_none_no_cred_fields(self, page, ui_url, session_id):
        _navigate_and_wait_sources(page, ui_url, session_id)
        _open_add_api_modal(page)
        assert not page.locator("input[placeholder='Bearer token']").is_visible()
        assert not page.locator("input[placeholder='user']").is_visible()

    def test_bearer_shows_token_field(self, page, ui_url, session_id):
        _navigate_and_wait_sources(page, ui_url, session_id)
        _open_add_api_modal(page)
        page.select_option("select >> nth=1", value="bearer")
        assert page.locator("input[placeholder='Bearer token']").is_visible()

    def test_basic_shows_username_password(self, page, ui_url, session_id):
        _navigate_and_wait_sources(page, ui_url, session_id)
        _open_add_api_modal(page)
        page.select_option("select >> nth=1", value="basic")
        assert page.locator("input[placeholder='user']").is_visible()
        assert page.locator("input[placeholder='password']").is_visible()

    def test_api_key_shows_header_and_value_fields(self, page, ui_url, session_id):
        _navigate_and_wait_sources(page, ui_url, session_id)
        _open_add_api_modal(page)
        page.select_option("select >> nth=1", value="api_key")
        assert page.locator("input[placeholder='X-API-Key']").is_visible()
        assert page.locator("input[placeholder='key value']").is_visible()

    def test_oauth2_shows_client_id_secret_token_url(self, page, ui_url, session_id):
        _navigate_and_wait_sources(page, ui_url, session_id)
        _open_add_api_modal(page)
        page.select_option("select >> nth=1", value="oauth2")
        assert page.locator("input[placeholder='client_id']").is_visible()
        assert page.locator("input[placeholder='client_secret']").is_visible()
        assert page.locator("input[placeholder*='auth.example.com']").is_visible()

    def test_cancel_closes_api_modal(self, page, ui_url, session_id):
        _navigate_and_wait_sources(page, ui_url, session_id)
        _open_add_api_modal(page)
        page.locator("button:has-text('Cancel')").click()
        assert not page.locator("text=Add API").is_visible()

    def test_add_button_disabled_without_url(self, page, ui_url, session_id):
        _navigate_and_wait_sources(page, ui_url, session_id)
        _open_add_api_modal(page)
        page.fill("input[placeholder='Name']", "testapi")
        add_btn = page.locator("button:has-text('Add')").last
        assert add_btn.get_attribute("disabled") is not None


# ---------------------------------------------------------------------------
# Add API modal: submission
# ---------------------------------------------------------------------------

class TestAddApiSubmission:
    """Verify Add API form submits correctly."""

    def test_add_rest_api_no_auth(self, page, ui_url, server_url):
        """Add a REST API without auth and verify it appears in sources."""
        sid = str(uuid.uuid4())
        resp = requests.post(f"{server_url}/api/sessions", json={"session_id": sid})
        assert resp.status_code == 200

        _navigate_and_wait_sources(page, ui_url, sid)
        _open_add_api_modal(page)

        unique_name = f"apitest_{sid[:8]}"
        page.fill("input[placeholder='Name']", unique_name)
        page.fill("input[placeholder*='Base URL']", "https://api.example.com")

        page.locator("button:has-text('Add')").click()

        # Modal closes
        page.wait_for_selector("text=Add API", state="detached", timeout=10000)

        # API appears in sources list
        page.wait_for_selector(f"text={unique_name}", timeout=10000)

        requests.delete(f"{server_url}/api/sessions/{sid}")

    def test_add_graphql_api_with_bearer(self, page, ui_url, server_url):
        """Add a GraphQL API with bearer token auth."""
        sid = str(uuid.uuid4())
        resp = requests.post(f"{server_url}/api/sessions", json={"session_id": sid})
        assert resp.status_code == 200

        _navigate_and_wait_sources(page, ui_url, sid)
        _open_add_api_modal(page)

        unique_name = f"gqlapitest_{sid[:8]}"
        page.fill("input[placeholder='Name']", unique_name)
        page.fill("input[placeholder*='Base URL']", "https://graphql.example.com/graphql")
        page.select_option("select >> nth=0", value="graphql")
        page.select_option("select >> nth=1", value="bearer")
        page.fill("input[placeholder='Bearer token']", "mytoken123")

        page.locator("button:has-text('Add')").click()

        page.wait_for_selector("text=Add API", state="detached", timeout=10000)
        page.wait_for_selector(f"text={unique_name}", timeout=10000)

        requests.delete(f"{server_url}/api/sessions/{sid}")
