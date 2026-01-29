# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Roles REST endpoints."""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from constat.server.auth import CurrentUserId
from constat.server.session_manager import SessionManager, ManagedSession
from constat.core.roles import RoleManager

logger = logging.getLogger(__name__)

router = APIRouter()


class RoleInfo(BaseModel):
    """Role information."""
    name: str
    prompt: str
    is_active: bool = False


class RolesListResponse(BaseModel):
    """Response for listing roles."""
    roles: list[RoleInfo]
    current_role: Optional[str] = None
    roles_file: str


class SetRoleRequest(BaseModel):
    """Request to set the active role."""
    role_name: Optional[str] = None  # None to clear role


class SetRoleResponse(BaseModel):
    """Response after setting role."""
    success: bool
    current_role: Optional[str] = None
    message: str


def get_session_manager(request: Request) -> SessionManager:
    """Dependency to get session manager from app state."""
    return request.app.state.session_manager


@router.get("/roles", response_model=RolesListResponse)
async def list_roles(
    session_id: str,
    user_id: str = Depends(CurrentUserId),
    session_manager: SessionManager = Depends(get_session_manager),
) -> RolesListResponse:
    """List all available roles."""
    managed = session_manager.get_session(session_id)
    if not managed or managed.user_id != user_id:
        raise HTTPException(status_code=404, detail="Session not found")

    session = managed.session
    if not hasattr(session, "role_manager"):
        raise HTTPException(status_code=500, detail="Role manager not available")

    role_manager = session.role_manager
    current_role = role_manager.active_role_name

    roles = []
    for name in role_manager.list_roles():
        role = role_manager.get_role(name)
        if role:
            roles.append(RoleInfo(
                name=name,
                prompt=role.prompt[:200] + "..." if len(role.prompt) > 200 else role.prompt,
                is_active=(name == current_role),
            ))

    return RolesListResponse(
        roles=roles,
        current_role=current_role,
        roles_file=str(role_manager.roles_file_path),
    )


@router.put("/roles/current", response_model=SetRoleResponse)
async def set_current_role(
    session_id: str,
    request: SetRoleRequest,
    user_id: str = Depends(CurrentUserId),
    session_manager: SessionManager = Depends(get_session_manager),
) -> SetRoleResponse:
    """Set the active role for the session."""
    managed = session_manager.get_session(session_id)
    if not managed or managed.user_id != user_id:
        raise HTTPException(status_code=404, detail="Session not found")

    session = managed.session
    if not hasattr(session, "role_manager"):
        raise HTTPException(status_code=500, detail="Role manager not available")

    role_manager = session.role_manager
    role_name = request.role_name

    if role_name is None or role_name.lower() == "none" or role_name == "":
        # Clear the role
        role_manager.set_active_role(None)
        return SetRoleResponse(
            success=True,
            current_role=None,
            message="Role cleared",
        )

    # Try to set the role
    if role_manager.set_active_role(role_name):
        return SetRoleResponse(
            success=True,
            current_role=role_name,
            message=f"Role set to '{role_name}'",
        )
    else:
        available = role_manager.list_roles()
        raise HTTPException(
            status_code=400,
            detail=f"Role not found: {role_name}. Available: {', '.join(available) or 'none'}",
        )
