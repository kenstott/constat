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
    description: str = ""
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


class CreateRoleRequest(BaseModel):
    """Request to create a role."""
    name: str
    prompt: str
    description: str = ""


class UpdateRoleRequest(BaseModel):
    """Request to update a role."""
    prompt: str
    description: str = ""


class RoleContentResponse(BaseModel):
    """Response with full role content."""
    name: str
    prompt: str
    description: str
    path: str


class DraftRoleRequest(BaseModel):
    """Request to draft a role using LLM."""
    name: str
    user_description: str  # Natural language description of the role


class DraftRoleResponse(BaseModel):
    """Response with LLM-drafted role content."""
    name: str
    prompt: str
    description: str


def get_session_manager(request: Request) -> SessionManager:
    """Dependency to get session manager from app state."""
    return request.app.state.session_manager


@router.get("/roles")
async def list_roles(
    request: Request,
    session_id: str,
    user_id: CurrentUserId,
    session_manager: SessionManager = Depends(get_session_manager),
):
    """List all available roles."""
    logger.info(f"[ROLES] list_roles ENTERED: session_id={session_id}, user_id={user_id}")

    managed = session_manager.get_session(session_id)
    if not managed or managed.user_id != user_id:
        logger.warning(f"[ROLES] Session not found or user mismatch: managed={managed is not None}, managed.user_id={managed.user_id if managed else 'N/A'}, user_id={user_id}")
        raise HTTPException(status_code=404, detail="Session not found")

    session = managed.session
    if not hasattr(session, "role_manager"):
        logger.error("[ROLES] Role manager not available on session")
        raise HTTPException(status_code=500, detail="Role manager not available")

    role_manager = session.role_manager
    logger.info(f"[ROLES] Role manager: roles_file={role_manager.roles_file_path}, exists={role_manager.roles_file_path.exists()}, has_roles={role_manager.has_roles}")

    current_role = role_manager.active_role_name

    roles = []
    for name in role_manager.list_roles():
        role = role_manager.get_role(name)
        if role:
            roles.append(RoleInfo(
                name=name,
                prompt=role.prompt[:200] + "..." if len(role.prompt) > 200 else role.prompt,
                description=role.description,
                is_active=(name == current_role),
            ))

    logger.info(f"[ROLES] Returning {len(roles)} roles, current_role={current_role}")
    return RolesListResponse(
        roles=roles,
        current_role=current_role,
        roles_file=str(role_manager.roles_file_path),
    )


@router.put("/roles/current", response_model=SetRoleResponse)
async def set_current_role(
    session_id: str,
    request: SetRoleRequest,
    user_id: CurrentUserId,
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


@router.get("/roles/{role_name}", response_model=RoleContentResponse)
async def get_role_content(
    session_id: str,
    role_name: str,
    user_id: CurrentUserId,
    session_manager: SessionManager = Depends(get_session_manager),
) -> RoleContentResponse:
    """Get the full content of a role."""
    managed = session_manager.get_session(session_id)
    if not managed or managed.user_id != user_id:
        raise HTTPException(status_code=404, detail="Session not found")

    session = managed.session
    if not hasattr(session, "role_manager"):
        raise HTTPException(status_code=500, detail="Role manager not available")

    role_manager = session.role_manager
    role = role_manager.get_role(role_name)
    if not role:
        raise HTTPException(status_code=404, detail=f"Role not found: {role_name}")

    return RoleContentResponse(
        name=role.name,
        prompt=role.prompt,
        description=role.description,
        path=str(role_manager.roles_file_path),
    )


@router.post("/roles", response_model=RoleInfo)
async def create_role(
    session_id: str,
    request_body: CreateRoleRequest,
    user_id: CurrentUserId,
    session_manager: SessionManager = Depends(get_session_manager),
) -> RoleInfo:
    """Create a new role."""
    managed = session_manager.get_session(session_id)
    if not managed or managed.user_id != user_id:
        raise HTTPException(status_code=404, detail="Session not found")

    session = managed.session
    if not hasattr(session, "role_manager"):
        raise HTTPException(status_code=500, detail="Role manager not available")

    role_manager = session.role_manager

    try:
        role = role_manager.create_role(
            name=request_body.name,
            prompt=request_body.prompt,
            description=request_body.description,
        )
        return RoleInfo(
            name=role.name,
            prompt=role.prompt,
            description=role.description,
            is_active=False,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/roles/{role_name}")
async def update_role(
    session_id: str,
    role_name: str,
    request_body: UpdateRoleRequest,
    user_id: CurrentUserId,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict:
    """Update a role."""
    managed = session_manager.get_session(session_id)
    if not managed or managed.user_id != user_id:
        raise HTTPException(status_code=404, detail="Session not found")

    session = managed.session
    if not hasattr(session, "role_manager"):
        raise HTTPException(status_code=500, detail="Role manager not available")

    role_manager = session.role_manager

    # Check if role exists
    if not role_manager.get_role(role_name):
        raise HTTPException(status_code=404, detail=f"Role not found: {role_name}")

    role_manager.update_role(
        name=role_name,
        prompt=request_body.prompt,
        description=request_body.description,
    )
    return {"status": "updated", "name": role_name}


@router.delete("/roles/{role_name}")
async def delete_role(
    session_id: str,
    role_name: str,
    user_id: CurrentUserId,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict:
    """Delete a role."""
    managed = session_manager.get_session(session_id)
    if not managed or managed.user_id != user_id:
        raise HTTPException(status_code=404, detail="Session not found")

    session = managed.session
    if not hasattr(session, "role_manager"):
        raise HTTPException(status_code=500, detail="Role manager not available")

    role_manager = session.role_manager

    if not role_manager.delete_role(role_name):
        raise HTTPException(status_code=404, detail=f"Role not found: {role_name}")

    return {"status": "deleted", "name": role_name}


@router.post("/roles/draft", response_model=DraftRoleResponse)
async def draft_role(
    session_id: str,
    request_body: DraftRoleRequest,
    user_id: CurrentUserId,
    session_manager: SessionManager = Depends(get_session_manager),
) -> DraftRoleResponse:
    """Use LLM to draft a role based on user description."""
    managed = session_manager.get_session(session_id)
    if not managed or managed.user_id != user_id:
        raise HTTPException(status_code=404, detail="Session not found")

    session = managed.session
    if not hasattr(session, "llm"):
        raise HTTPException(status_code=500, detail="LLM not available")
    if not hasattr(session, "role_manager"):
        raise HTTPException(status_code=500, detail="Role manager not available")

    try:
        role = session.role_manager.draft_role(
            name=request_body.name,
            user_description=request_body.user_description,
            llm=session.llm,
        )
        return DraftRoleResponse(
            name=role.name,
            prompt=role.prompt,
            description=role.description,
        )
    except ValueError as e:
        logger.error(f"Failed to draft role: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to draft role: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to draft role: {str(e)}")
