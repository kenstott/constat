# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Account CRUD API for personal resource management."""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from constat.server.accounts import (
    PersonalAccount,
    decrypt_token,
    encrypt_token,
    load_user_accounts,
    now_iso,
    save_user_accounts,
    validate_account,
)

logger = logging.getLogger(__name__)

router = APIRouter()


class AddAccountRequest(BaseModel):
    """Request body for adding a new personal account."""

    name: str
    type: str  # imap, drive, calendar, sharepoint
    provider: str  # google, microsoft
    display_name: str
    email: str
    refresh_token: str
    oauth2_tenant_id: str | None = None
    options: dict = {}


class UpdateAccountRequest(BaseModel):
    """Request body for updating a personal account."""

    display_name: str | None = None
    active: bool | None = None
    options: dict | None = None


class AccountSummary(BaseModel):
    """Account summary returned in API responses (no tokens)."""

    name: str
    type: str
    provider: str
    display_name: str
    email: str
    active: bool
    created_at: str


def _get_data_dir(request: Request):
    """Get data_dir from server config."""
    return request.app.state.server_config.data_dir


def _get_encryption_secret(request: Request) -> str | None:
    """Get account encryption secret from server config."""
    return request.app.state.server_config.account_encryption_secret


def _to_summary(account: PersonalAccount) -> dict:
    """Convert a PersonalAccount to an API-safe summary dict."""
    return AccountSummary(
        name=account.name,
        type=account.type,
        provider=account.provider,
        display_name=account.display_name,
        email=account.email,
        active=account.active,
        created_at=account.created_at,
    ).model_dump()


@router.get("/")
async def list_accounts(request: Request, user_id: str) -> list[dict]:
    """List user's connected accounts (no tokens in response)."""
    accounts = load_user_accounts(user_id, _get_data_dir(request))
    return [_to_summary(acct) for acct in accounts.values()]


@router.post("/")
async def add_account(request: Request, user_id: str, body: AddAccountRequest) -> dict:
    """Add a new personal account from completed OAuth flow."""
    data_dir = _get_data_dir(request)
    accounts = load_user_accounts(user_id, data_dir)

    if body.name in accounts:
        raise HTTPException(status_code=409, detail=f"Account already exists: {body.name}")

    # Encrypt refresh token if encryption secret is configured
    encryption_secret = _get_encryption_secret(request)
    encrypted_token = body.refresh_token
    if encryption_secret:
        encrypted_token = encrypt_token(body.refresh_token, encryption_secret, user_id)

    account = PersonalAccount(
        name=body.name,
        type=body.type,
        provider=body.provider,
        display_name=body.display_name,
        email=body.email,
        refresh_token=encrypted_token,
        created_at=now_iso(),
        active=True,
        oauth2_tenant_id=body.oauth2_tenant_id,
        options=body.options,
    )

    validate_account(account)

    accounts[body.name] = account
    save_user_accounts(user_id, accounts, data_dir)

    logger.info(f"Added personal account {body.name} for user {user_id}")
    return _to_summary(account)


@router.put("/{account_name}")
async def update_account(
    request: Request,
    user_id: str,
    account_name: str,
    body: UpdateAccountRequest,
) -> dict:
    """Update account options (active toggle, rename, change folder, etc.)."""
    data_dir = _get_data_dir(request)
    accounts = load_user_accounts(user_id, data_dir)

    if account_name not in accounts:
        raise HTTPException(status_code=404, detail=f"Account not found: {account_name}")

    account = accounts[account_name]

    if body.display_name is not None:
        account.display_name = body.display_name
    if body.active is not None:
        account.active = body.active
    if body.options is not None:
        account.options = body.options

    save_user_accounts(user_id, accounts, data_dir)

    logger.info(f"Updated personal account {account_name} for user {user_id}")
    return _to_summary(account)


@router.delete("/{account_name}")
async def delete_account(request: Request, user_id: str, account_name: str) -> dict:
    """Remove an account."""
    data_dir = _get_data_dir(request)
    accounts = load_user_accounts(user_id, data_dir)

    if account_name not in accounts:
        raise HTTPException(status_code=404, detail=f"Account not found: {account_name}")

    del accounts[account_name]
    save_user_accounts(user_id, accounts, data_dir)

    logger.info(f"Deleted personal account {account_name} for user {user_id}")
    return {"status": "ok", "deleted": account_name}


@router.post("/{account_name}/test")
async def test_account(request: Request, user_id: str, account_name: str) -> dict:
    """Test account connectivity (verify token can be decrypted)."""
    data_dir = _get_data_dir(request)
    accounts = load_user_accounts(user_id, data_dir)

    if account_name not in accounts:
        raise HTTPException(status_code=404, detail=f"Account not found: {account_name}")

    account = accounts[account_name]

    # Verify we can decrypt the token
    encryption_secret = _get_encryption_secret(request)
    if encryption_secret and account.refresh_token:
        try:
            decrypt_token(account.refresh_token, encryption_secret, user_id)
            return {"status": "ok", "account": account_name, "message": "Token decryption successful"}
        except Exception as e:
            return {"status": "error", "account": account_name, "message": f"Token decryption failed: {e}"}

    return {"status": "ok", "account": account_name, "message": "No encryption configured, token stored in plaintext"}
