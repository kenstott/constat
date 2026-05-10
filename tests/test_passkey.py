# Copyright (c) 2025 Kenneth Stott
# Canary: 0d68626b-f1c3-40a5-8317-b16c0e302a8e
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Tests for constat.server.routes.passkey — credential storage and endpoint logic."""

from __future__ import annotations
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from constat.server.routes.passkey import (
    _load_credentials,
    _save_credentials,
    _cred_path,
    _pending_challenges,
)


@pytest.fixture
def data_dir(tmp_path):
    return tmp_path / ".constat"


class TestCredentialStorage:
    def test_empty_when_no_file(self, data_dir):
        assert _load_credentials(data_dir, "user1") == []

    def test_save_and_load_roundtrip(self, data_dir):
        creds = [{"credential_id": "abc123", "public_key": "pk1", "sign_count": 0}]
        _save_credentials(data_dir, "user1", creds)
        loaded = _load_credentials(data_dir, "user1")
        assert loaded == creds

    def test_cred_path(self, data_dir):
        path = _cred_path(data_dir, "user1")
        assert path == data_dir / "user1.vault" / ".passkey_credentials"

    def test_save_creates_parent_dirs(self, data_dir):
        _save_credentials(data_dir, "deep/user", [{"id": "x"}])
        assert _cred_path(data_dir, "deep/user").exists()


class TestPasskeyEndpoints:
    """Test endpoint logic using FastAPI TestClient with mocked webauthn verification."""

    @pytest.fixture
    def app(self, data_dir):
        from fastapi import FastAPI
        from constat.server.routes.passkey import router

        app = FastAPI()
        app.include_router(router, prefix="/api/auth/passkey")

        # Mock app state
        mock_config = MagicMock()
        mock_config.data_dir = data_dir
        mock_config.vault_encrypt = False
        app.state.server_config = mock_config
        return app

    @pytest.fixture
    def client(self, app):
        from fastapi.testclient import TestClient
        return TestClient(app)

    def test_register_begin_returns_options(self, client):
        resp = client.post(
            "/api/auth/passkey/register/begin",
            json={"user_id": "testuser"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "challenge" in data
        assert "rp" in data
        assert "testuser" in _pending_challenges  # challenge was stored

    def test_register_complete_no_challenge(self, client):
        _pending_challenges.pop("testuser", None)
        resp = client.post(
            "/api/auth/passkey/register/complete",
            json={"user_id": "testuser", "credential": {}},
        )
        assert resp.status_code == 400

    def test_auth_begin_no_creds(self, client):
        resp = client.post(
            "/api/auth/passkey/auth/begin",
            json={"user_id": "nouser"},
        )
        assert resp.status_code == 404

    def test_auth_complete_no_challenge(self, client):
        _pending_challenges.pop("testuser", None)
        resp = client.post(
            "/api/auth/passkey/auth/complete",
            json={"user_id": "testuser", "credential": {}},
        )
        assert resp.status_code == 400

    @patch("constat.server.routes.passkey.verify_registration_response")
    def test_register_complete_stores_credential(self, mock_verify, client, data_dir):
        # Set up a pending challenge
        _pending_challenges["testuser"] = b"test-challenge"

        mock_verify.return_value = MagicMock(
            credential_id=b"\x01\x02\x03",
            credential_public_key=b"\x04\x05\x06",
            sign_count=0,
        )

        resp = client.post(
            "/api/auth/passkey/register/complete",
            json={"user_id": "testuser", "credential": {"id": "AQID", "type": "public-key"}},
        )
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

        creds = _load_credentials(data_dir, "testuser")
        assert len(creds) == 1
        assert creds[0]["sign_count"] == 0
