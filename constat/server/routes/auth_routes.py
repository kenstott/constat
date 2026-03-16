# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Authentication routes (local and Firebase)."""

import logging

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from constat.server.local_auth import create_local_token, verify_password

logger = logging.getLogger(__name__)

router = APIRouter()


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    token: str
    user_id: str
    email: str


@router.post("/login")
async def login(request: Request, body: LoginRequest) -> LoginResponse:
    """Authenticate with username/password, return opaque token."""
    server_config = request.app.state.server_config
    user = server_config.local_users.get(body.username)
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(401, "Invalid username or password")
    token = create_local_token(body.username, user.email)
    return LoginResponse(token=token, user_id=body.username, email=user.email)


class FirebaseLoginRequest(BaseModel):
    email: str
    password: str


@router.post("/firebase-login")
async def firebase_login(request: Request, body: FirebaseLoginRequest) -> LoginResponse:
    """Authenticate via Firebase REST API server-side. Returns Firebase ID token."""
    server_config = request.app.state.server_config
    api_key = server_config.firebase_api_key
    if not api_key:
        raise HTTPException(500, "Firebase API key not configured on server")

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={api_key}",
                json={"email": body.email, "password": body.password, "returnSecureToken": True},
                timeout=15,
            )
            if resp.status_code != 200:
                detail = resp.json().get("error", {}).get("message", "Authentication failed")
                raise HTTPException(401, detail)
            data = resp.json()
            return LoginResponse(
                token=data["idToken"],
                user_id=data.get("localId", ""),
                email=body.email,
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Firebase login error: {e}")
        raise HTTPException(500, "Firebase authentication failed")
