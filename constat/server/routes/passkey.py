# Copyright (c) 2025 Kenneth Stott
# Canary: 6db9cdad-bd2a-4544-a574-b8d4543ec148
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""WebAuthn passkey registration and authentication endpoints."""

import json
import logging
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from constat.core.paths import user_vault_dir

from webauthn import (
    generate_authentication_options,
    generate_registration_options,
    verify_authentication_response,
    verify_registration_response,
)
from webauthn.helpers import bytes_to_base64url, base64url_to_bytes, options_to_json
from webauthn.helpers.structs import (
    AuthenticatorSelectionCriteria,
    PublicKeyCredentialDescriptor,
    ResidentKeyRequirement,
    UserVerificationRequirement,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory challenge store (keyed by user_id, short-lived)
_pending_challenges: dict[str, bytes] = {}

RP_ID = "localhost"
RP_NAME = "Constat"
ORIGIN = "http://localhost:5173"


# ------------------------------------------------------------------
# Credential persistence (JSON file per user)
# ------------------------------------------------------------------

def _cred_path(data_dir: Path, user_id: str) -> Path:
    return user_vault_dir(data_dir, user_id) / ".passkey_credentials"


def _load_credentials(data_dir: Path, user_id: str) -> list[dict[str, Any]]:
    path = _cred_path(data_dir, user_id)
    if not path.exists():
        return []
    return json.loads(path.read_text())


def _save_credentials(data_dir: Path, user_id: str, creds: list[dict[str, Any]]) -> None:
    path = _cred_path(data_dir, user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(creds))


# ------------------------------------------------------------------
# Request / Response models
# ------------------------------------------------------------------

class RegisterCompleteRequest(BaseModel):
    user_id: str
    credential: dict[str, Any]


class AuthBeginRequest(BaseModel):
    user_id: str


class AuthCompleteRequest(BaseModel):
    user_id: str
    credential: dict[str, Any]
    prf_output: str | None = None  # base64url-encoded PRF result


# ------------------------------------------------------------------
# Registration
# ------------------------------------------------------------------

@router.post("/register/begin")
async def register_begin(request: Request, body: AuthBeginRequest) -> dict[str, Any]:
    """Generate registration options for a new passkey."""
    data_dir: Path = request.app.state.server_config.data_dir
    existing = _load_credentials(data_dir, body.user_id)
    exclude = [
        PublicKeyCredentialDescriptor(id=base64url_to_bytes(c["credential_id"]))
        for c in existing
    ]

    options = generate_registration_options(
        rp_id=RP_ID,
        rp_name=RP_NAME,
        user_id=body.user_id.encode(),
        user_name=body.user_id,
        user_display_name=body.user_id,
        authenticator_selection=AuthenticatorSelectionCriteria(
            resident_key=ResidentKeyRequirement.PREFERRED,
            user_verification=UserVerificationRequirement.PREFERRED,
        ),
        exclude_credentials=exclude,
    )

    _pending_challenges[body.user_id] = options.challenge
    return json.loads(options_to_json(options))


@router.post("/register/complete")
async def register_complete(request: Request, body: RegisterCompleteRequest) -> dict[str, str]:
    """Verify and store a new passkey credential."""
    challenge = _pending_challenges.pop(body.user_id, None)
    if challenge is None:
        raise HTTPException(400, "No pending registration challenge")

    verification = verify_registration_response(
        credential=body.credential,
        expected_challenge=challenge,
        expected_rp_id=RP_ID,
        expected_origin=ORIGIN,
    )

    data_dir: Path = request.app.state.server_config.data_dir
    creds = _load_credentials(data_dir, body.user_id)
    creds.append({
        "credential_id": bytes_to_base64url(verification.credential_id),
        "public_key": bytes_to_base64url(verification.credential_public_key),
        "sign_count": verification.sign_count,
    })
    _save_credentials(data_dir, body.user_id, creds)

    return {"status": "ok"}


# ------------------------------------------------------------------
# Authentication
# ------------------------------------------------------------------

@router.post("/auth/begin")
async def auth_begin(request: Request, body: AuthBeginRequest) -> dict[str, Any]:
    """Generate authentication options for an existing passkey."""
    data_dir: Path = request.app.state.server_config.data_dir
    creds = _load_credentials(data_dir, body.user_id)
    if not creds:
        raise HTTPException(404, "No passkey registered for this user")

    allow = [
        PublicKeyCredentialDescriptor(id=base64url_to_bytes(c["credential_id"]))
        for c in creds
    ]

    options = generate_authentication_options(
        rp_id=RP_ID,
        allow_credentials=allow,
        user_verification=UserVerificationRequirement.PREFERRED,
    )

    _pending_challenges[body.user_id] = options.challenge
    return json.loads(options_to_json(options))


@router.post("/auth/complete")
async def auth_complete(request: Request, body: AuthCompleteRequest) -> dict[str, Any]:
    """Verify passkey authentication and optionally unlock vault with PRF output."""
    challenge = _pending_challenges.pop(body.user_id, None)
    if challenge is None:
        raise HTTPException(400, "No pending authentication challenge")

    data_dir: Path = request.app.state.server_config.data_dir
    creds = _load_credentials(data_dir, body.user_id)

    # Find the matching credential
    cred_id_b64 = body.credential.get("id", "")
    matched = None
    for c in creds:
        if c["credential_id"] == cred_id_b64:
            matched = c
            break
    if matched is None:
        raise HTTPException(400, "Unknown credential")

    verification = verify_authentication_response(
        credential=body.credential,
        expected_challenge=challenge,
        expected_rp_id=RP_ID,
        expected_origin=ORIGIN,
        credential_public_key=base64url_to_bytes(matched["public_key"]),
        credential_current_sign_count=matched["sign_count"],
    )

    # Update sign count
    matched["sign_count"] = verification.new_sign_count
    _save_credentials(data_dir, body.user_id, creds)

    result: dict[str, Any] = {"status": "ok", "user_id": body.user_id}

    # If PRF output provided and vault encryption enabled, unlock vault
    server_config = request.app.state.server_config
    if body.prf_output and server_config.vault_encrypt:
        from constat.server.vault import UserVault

        prf_bytes = base64url_to_bytes(body.prf_output)
        user_dir = user_vault_dir(data_dir, body.user_id)
        vault = UserVault(user_dir, encrypt=True)
        vault_path = vault.unlock(prf_bytes)
        result["vault_unlocked"] = True
        result["vault_path"] = str(vault_path)

    return result
