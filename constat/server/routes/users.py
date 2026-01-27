# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""User management REST endpoints."""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from constat.server.auth import CurrentUserId, CurrentUserEmail
from constat.server.permissions import get_permissions_store, UserPermissions

logger = logging.getLogger(__name__)

router = APIRouter()


class PermissionsResponse(BaseModel):
    """User permissions response."""
    user_id: str
    email: str | None
    admin: bool
    projects: list[str]


class UpdatePermissionsRequest(BaseModel):
    """Request to update user permissions."""
    email: str
    admin: bool | None = None
    projects: list[str] | None = None


@router.get("/me/permissions", response_model=PermissionsResponse)
async def get_my_permissions(
    user_id: CurrentUserId,
    email: CurrentUserEmail,
) -> PermissionsResponse:
    """Get permissions for the current authenticated user.

    Returns:
        User's permissions including admin status and accessible projects
    """
    store = get_permissions_store()
    perms = store.get_user_permissions(email=email or "", user_id=user_id)

    return PermissionsResponse(
        user_id=perms.user_id,
        email=perms.email,
        admin=perms.admin,
        projects=perms.projects,
    )


@router.get("/permissions", response_model=list[dict[str, Any]])
async def list_all_permissions(
    user_id: CurrentUserId,
    email: CurrentUserEmail,
) -> list[dict[str, Any]]:
    """List all users with explicit permissions.

    Requires admin access.

    Returns:
        List of all user permissions
    """
    store = get_permissions_store()

    # Check if current user is admin
    current_perms = store.get_user_permissions(email=email or "", user_id=user_id)
    if not current_perms.admin:
        raise HTTPException(status_code=403, detail="Admin access required")

    return store.list_users()


@router.put("/permissions", response_model=PermissionsResponse)
async def update_user_permissions(
    body: UpdatePermissionsRequest,
    user_id: CurrentUserId,
    email: CurrentUserEmail,
) -> PermissionsResponse:
    """Update permissions for a user.

    Requires admin access.

    Args:
        body: Permission update request with target email and new permissions

    Returns:
        Updated user permissions
    """
    store = get_permissions_store()

    # Check if current user is admin
    current_perms = store.get_user_permissions(email=email or "", user_id=user_id)
    if not current_perms.admin:
        raise HTTPException(status_code=403, detail="Admin access required")

    # Update permissions
    updated = store.set_user_permissions(
        email=body.email,
        admin=body.admin,
        projects=body.projects,
    )

    return PermissionsResponse(
        user_id=updated.user_id,
        email=updated.email,
        admin=updated.admin,
        projects=updated.projects,
    )