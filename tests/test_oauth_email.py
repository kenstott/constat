"""Tests for OAuth2 email browser authentication flow."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from constat.server.config import ServerConfig


@pytest.fixture
def server_config(monkeypatch):
    monkeypatch.delenv("GOOGLE_EMAIL_CLIENT_ID", raising=False)
    monkeypatch.delenv("GOOGLE_EMAIL_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("MICROSOFT_EMAIL_CLIENT_ID", raising=False)
    monkeypatch.delenv("MICROSOFT_EMAIL_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("MICROSOFT_EMAIL_TENANT_ID", raising=False)
    return ServerConfig(
        google_email_client_id="google-client-id",
        google_email_client_secret="google-client-secret",
        microsoft_email_client_id="ms-client-id",
        microsoft_email_client_secret="ms-client-secret",
        microsoft_email_tenant_id="test-tenant",
    )


@pytest.fixture
def app(server_config):
    from fastapi import FastAPI
    from constat.server.routes.oauth_email import router

    app = FastAPI()
    app.state.server_config = server_config
    app.state.oauth_pending = {}
    app.state.oauth_completed = {}
    app.include_router(router, prefix="/api/oauth/email")
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


def test_providers_returns_configured(client):
    resp = client.get("/api/oauth/email/providers")
    assert resp.status_code == 200
    data = resp.json()
    assert data["google"] is True
    assert data["microsoft"] is True


def test_providers_none_configured(monkeypatch):
    monkeypatch.delenv("GOOGLE_EMAIL_CLIENT_ID", raising=False)
    monkeypatch.delenv("GOOGLE_EMAIL_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("MICROSOFT_EMAIL_CLIENT_ID", raising=False)
    monkeypatch.delenv("MICROSOFT_EMAIL_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("MICROSOFT_EMAIL_TENANT_ID", raising=False)
    from fastapi import FastAPI
    from fastapi.testclient import TestClient as TC
    from constat.server.routes.oauth_email import router

    app = FastAPI()
    app.state.server_config = ServerConfig()
    app.state.oauth_pending = {}
    app.state.oauth_completed = {}
    app.include_router(router, prefix="/api/oauth/email")
    c = TC(app)
    resp = c.get("/api/oauth/email/providers")
    data = resp.json()
    assert data["google"] is False
    assert data["microsoft"] is False


def test_authorize_google_redirects(client, app):
    resp = client.get(
        "/api/oauth/email/authorize?provider=google&session_id=test-session",
        follow_redirects=False,
    )
    assert resp.status_code == 307
    location = resp.headers["location"]
    assert "accounts.google.com" in location
    assert "google-client-id" in location
    assert len(app.state.oauth_pending) == 1


def test_authorize_microsoft_redirects(client, app):
    resp = client.get(
        "/api/oauth/email/authorize?provider=microsoft&session_id=test-session",
        follow_redirects=False,
    )
    assert resp.status_code == 307
    location = resp.headers["location"]
    assert "login.microsoftonline.com" in location
    assert "test-tenant" in location
    assert len(app.state.oauth_pending) == 1


def test_authorize_invalid_provider(client):
    resp = client.get(
        "/api/oauth/email/authorize?provider=yahoo&session_id=test-session",
        follow_redirects=False,
    )
    assert resp.status_code == 400


def test_callback_invalid_state(client):
    resp = client.get("/api/oauth/email/callback?code=abc&state=invalid")
    assert resp.status_code == 400


def test_callback_google_success(client, app):
    # Set up pending state
    state = "test-state-123"
    app.state.oauth_pending[state] = {
        "provider": "google",
        "session_id": "test-session",
        "created_at": time.time(),
    }

    mock_token_resp = MagicMock()
    mock_token_resp.status_code = 200
    mock_token_resp.json.return_value = {
        "access_token": "access-123",
        "refresh_token": "refresh-456",
        "id_token": "x.eyJlbWFpbCI6InVzZXJAZ21haWwuY29tIn0.z",
    }
    mock_token_resp.raise_for_status = MagicMock()

    mock_userinfo_resp = MagicMock()
    mock_userinfo_resp.status_code = 200
    mock_userinfo_resp.json.return_value = {"email": "user@gmail.com"}
    mock_userinfo_resp.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as MockClient:
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_token_resp
        mock_client.get.return_value = mock_userinfo_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_client

        resp = client.get(f"/api/oauth/email/callback?code=auth-code&state={state}")

    assert resp.status_code == 200
    assert "oauth-email-complete" in resp.text
    assert state not in app.state.oauth_pending
    assert state in app.state.oauth_completed


def test_result_found(client, app):
    state = "result-state"
    app.state.oauth_completed[state] = {
        "provider": "google",
        "email": "user@gmail.com",
        "refresh_token": "refresh-123",
        "created_at": time.time(),
    }
    resp = client.get(f"/api/oauth/email/result/{state}")
    assert resp.status_code == 200
    assert resp.json()["email"] == "user@gmail.com"


def test_result_not_found(client):
    resp = client.get("/api/oauth/email/result/nonexistent")
    assert resp.status_code == 404


def test_expired_state_cleanup(client, app):
    # Add expired entries
    old_time = time.time() - 700  # >10 min ago
    app.state.oauth_pending["expired"] = {"provider": "google", "session_id": "x", "created_at": old_time}
    app.state.oauth_completed["expired"] = {"provider": "google", "email": "x", "refresh_token": "x", "created_at": old_time}

    # Trigger cleanup via any endpoint
    client.get("/api/oauth/email/providers")

    assert "expired" not in app.state.oauth_pending
    assert "expired" not in app.state.oauth_completed
