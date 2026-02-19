# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""User management REST endpoints."""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from constat.server.auth import CurrentUserId, CurrentUserEmail
from constat.server.permissions import get_user_permissions, list_all_permissions as list_perms
from constat.server.user_preferences import load_user_preferences, save_user_preferences

logger = logging.getLogger(__name__)

router = APIRouter()


class PermissionsResponse(BaseModel):
    """User permissions response."""
    user_id: str
    email: str | None
    admin: bool
    domains: list[str]
    databases: list[str]
    documents: list[str]
    apis: list[str]


@router.get("/me/permissions", response_model=PermissionsResponse)
async def get_my_permissions(
    request: Request,
    user_id: CurrentUserId,
    email: CurrentUserEmail,
) -> PermissionsResponse:
    """Get permissions for the current authenticated user.

    Returns:
        User's permissions including admin status and accessible resources
    """
    server_config = request.app.state.server_config
    perms = get_user_permissions(server_config, email=email or "", user_id=user_id)

    return PermissionsResponse(
        user_id=perms.user_id,
        email=perms.email,
        admin=perms.admin,
        domains=perms.domains,
        databases=perms.databases,
        documents=perms.documents,
        apis=perms.apis,
    )


@router.get("/permissions", response_model=list[dict[str, Any]])
async def list_all_user_permissions(
    request: Request,
    user_id: CurrentUserId,
    email: CurrentUserEmail,
) -> list[dict[str, Any]]:
    """List all users with explicit permissions.

    Requires admin access.

    Returns:
        List of all user permissions
    """
    server_config = request.app.state.server_config

    # Check if current user is admin
    current_perms = get_user_permissions(server_config, email=email or "", user_id=user_id)
    if not current_perms.admin:
        raise HTTPException(status_code=403, detail="Admin access required")

    return list_perms(server_config)


@router.get("/me/preferences")
async def get_preferences(
    user_id: CurrentUserId,
) -> dict[str, Any]:
    """Get user preferences."""
    return load_user_preferences(user_id)


@router.patch("/me/preferences")
async def update_preferences(
    user_id: CurrentUserId,
    body: dict[str, Any],
) -> dict[str, Any]:
    """Partially update user preferences."""
    prefs = load_user_preferences(user_id)
    prefs.update(body)
    save_user_preferences(user_id, prefs)
    return prefs