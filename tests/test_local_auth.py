# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tests for server-local authentication."""

import os
import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from constat.server.local_auth import (
    _local_tokens,
    create_local_token,
    hash_password,
    validate_local_token,
    verify_password,
)


# ---- Unit tests: hashing ----


def test_hash_and_verify():
    pw = "hunter2"
    hashed = hash_password(pw)
    assert hashed.startswith("scrypt:")
    assert verify_password(pw, hashed)


def test_wrong_password():
    hashed = hash_password("correct")
    assert not verify_password("wrong", hashed)


def test_hash_uniqueness():
    h1 = hash_password("same")
    h2 = hash_password("same")
    # Different salts → different hashes
    assert h1 != h2
    # Both still verify
    assert verify_password("same", h1)
    assert verify_password("same", h2)


# ---- Unit tests: tokens ----


def test_create_validate_token():
    _local_tokens.clear()
    token = create_local_token("alice", "alice@example.com")
    result = validate_local_token(token)
    assert result == ("alice", "alice@example.com")


def test_invalid_token():
    assert validate_local_token("nonexistent-token") is None


def test_expired_token():
    _local_tokens.clear()
    token = create_local_token("bob", "bob@example.com")
    # Manually expire the token
    user_id, email, _ = _local_tokens[token]
    _local_tokens[token] = (user_id, email, time.monotonic() - 1)
    assert validate_local_token(token) is None
    # Token should be cleaned up
    assert token not in _local_tokens


def test_token_eviction():
    _local_tokens.clear()
    with patch("constat.server.local_auth._TOKEN_MAX", 2):
        t1 = create_local_token("u1", "u1@x.com")
        t2 = create_local_token("u2", "u2@x.com")
        t3 = create_local_token("u3", "u3@x.com")
        # Oldest (t1) should be evicted
        assert validate_local_token(t1) is None
        assert validate_local_token(t2) is not None
        assert validate_local_token(t3) is not None
    _local_tokens.clear()


# ---- Integration tests: endpoints ----


def _make_test_app():
    """Create a minimal FastAPI app with auth routes and local users."""
    from fastapi import FastAPI

    from constat.server.config import LocalUser, ServerConfig
    from constat.server.routes.auth_routes import router as auth_router

    app = FastAPI()

    pw_hash = hash_password("testpass")
    server_config = ServerConfig(
        auth_disabled=False,
        local_users={
            "testuser": LocalUser(password_hash=pw_hash, email="test@example.com"),
        },
    )
    app.state.server_config = server_config
    app.include_router(auth_router, prefix="/api/auth")
    return app, pw_hash


def test_login_endpoint_success():
    _local_tokens.clear()
    app, _ = _make_test_app()
    client = TestClient(app)

    resp = client.post("/api/auth/login", json={"username": "testuser", "password": "testpass"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["user_id"] == "testuser"
    assert data["email"] == "test@example.com"
    assert "token" in data
    # Token should be valid
    assert validate_local_token(data["token"]) == ("testuser", "test@example.com")
    _local_tokens.clear()


def test_login_endpoint_bad_password():
    app, _ = _make_test_app()
    client = TestClient(app)

    resp = client.post("/api/auth/login", json={"username": "testuser", "password": "wrong"})
    assert resp.status_code == 401


def test_login_endpoint_unknown_user():
    app, _ = _make_test_app()
    client = TestClient(app)

    resp = client.post("/api/auth/login", json={"username": "nobody", "password": "testpass"})
    assert resp.status_code == 401


# ---- Integration test: local token accepted by auth middleware ----


def test_local_token_accepted_by_auth():
    """Token from login endpoint works on an authenticated route."""
    _local_tokens.clear()
    from fastapi import Depends, FastAPI

    from constat.server.auth import get_current_user_id
    from constat.server.config import LocalUser, ServerConfig
    from constat.server.routes.auth_routes import router as auth_router

    app = FastAPI()
    pw_hash = hash_password("secret")
    server_config = ServerConfig(
        auth_disabled=False,
        local_users={"admin": LocalUser(password_hash=pw_hash, email="admin@localhost")},
    )
    app.state.server_config = server_config
    app.include_router(auth_router, prefix="/api/auth")

    @app.get("/api/me")
    async def me(user_id: str = Depends(get_current_user_id)):
        return {"user_id": user_id}

    client = TestClient(app)

    # Login
    resp = client.post("/api/auth/login", json={"username": "admin", "password": "secret"})
    assert resp.status_code == 200
    token = resp.json()["token"]

    # Use token on authenticated endpoint
    resp = client.get("/api/me", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert resp.json()["user_id"] == "admin"
    _local_tokens.clear()


# ---- Integration test: /health auth_methods ----


@patch.dict("os.environ", {"FIREBASE_PROJECT_ID": "", "AUTH_DISABLED": ""}, clear=False)
def test_health_auth_methods_local_only():
    """Health endpoint returns auth_methods: ['local'] when local_users configured."""
    import os
    # Remove env vars that override config
    env = os.environ.copy()
    env.pop("FIREBASE_PROJECT_ID", None)
    env.pop("AUTH_DISABLED", None)
    with patch.dict("os.environ", env, clear=True):
        from constat.server.config import LocalUser, ServerConfig

        server_config = ServerConfig(
            auth_disabled=False,
            local_users={"u": LocalUser(password_hash="scrypt:aa:bb", email="u@x.com")},
        )
        methods = (
            (["local"] if server_config.local_users else [])
            + (["firebase"] if server_config.firebase_project_id else [])
        )
        assert methods == ["local"]


def test_health_auth_methods_firebase_only():
    import os
    env = os.environ.copy()
    env.pop("FIREBASE_PROJECT_ID", None)
    env.pop("AUTH_DISABLED", None)
    with patch.dict("os.environ", env, clear=True):
        from constat.server.config import ServerConfig
        server_config = ServerConfig(
            auth_disabled=False,
            firebase_project_id="my-project",
        )
        methods = (
            (["local"] if server_config.local_users else [])
            + (["firebase"] if server_config.firebase_project_id else [])
        )
        assert methods == ["firebase"]


def test_health_auth_methods_both():
    import os
    env = os.environ.copy()
    env.pop("FIREBASE_PROJECT_ID", None)
    env.pop("AUTH_DISABLED", None)
    with patch.dict("os.environ", env, clear=True):
        from constat.server.config import LocalUser, ServerConfig

        server_config = ServerConfig(
            auth_disabled=False,
            firebase_project_id="my-project",
            local_users={"u": LocalUser(password_hash="scrypt:aa:bb", email="u@x.com")},
        )
        methods = (
            (["local"] if server_config.local_users else [])
            + (["firebase"] if server_config.firebase_project_id else [])
        )
        assert methods == ["local", "firebase"]
