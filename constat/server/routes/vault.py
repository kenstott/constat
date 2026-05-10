# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Vault status and creation endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from constat.core.paths import user_vault_dir
from constat.server.vault import UserVault

router = APIRouter(tags=["vault"])


def _get_data_dir(request: Request):
    return request.app.state.server_config.data_dir


def _get_vault_encrypt(request: Request) -> bool:
    return request.app.state.server_config.vault_encrypt


@router.get("/{user_id}/status")
async def vault_status(request: Request, user_id: str) -> dict:
    """Return whether a vault token has been established for the user."""
    if not _get_vault_encrypt(request):
        return {"has_vault": True}
    user_dir = user_vault_dir(_get_data_dir(request), user_id)
    salt_path = user_dir / UserVault.SALT_FILE
    return {"has_vault": salt_path.exists()}


class CreateVaultRequest(BaseModel):
    password: str


@router.post("/{user_id}/create")
async def create_vault(request: Request, user_id: str, body: CreateVaultRequest) -> dict:
    """Create the vault salt for a user. Errors if vault already exists."""
    if not _get_vault_encrypt(request):
        return {"status": "ok", "message": "vault encryption disabled"}
    user_dir = user_vault_dir(_get_data_dir(request), user_id)
    salt_path = user_dir / UserVault.SALT_FILE
    if salt_path.exists():
        raise HTTPException(status_code=409, detail="Vault already exists for this user")
    vault = UserVault(user_dir, encrypt=True)
    vault.create(body.password.encode())
    return {"status": "ok"}
