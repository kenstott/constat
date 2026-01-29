# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Skills REST endpoints (user-based, not session-based)."""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from constat.server.auth import CurrentUserId
from constat.core.skills import SkillManager, Skill

logger = logging.getLogger(__name__)

router = APIRouter()


class SkillInfo(BaseModel):
    """Skill information."""
    name: str
    prompt: str
    description: str
    filename: str
    is_active: bool = False


class SkillsListResponse(BaseModel):
    """Response for listing skills."""
    skills: list[SkillInfo]
    active_skills: list[str]
    skills_dir: str


class CreateSkillRequest(BaseModel):
    """Request to create a skill."""
    name: str
    prompt: str
    description: str = ""


class UpdateSkillRequest(BaseModel):
    """Request to update a skill."""
    content: str  # Raw YAML content


class SetActiveSkillsRequest(BaseModel):
    """Request to set active skills."""
    skill_names: list[str]


class SkillContentResponse(BaseModel):
    """Response with skill content."""
    name: str
    content: str
    path: str


# Cache skill managers per user
_skill_managers: dict[str, SkillManager] = {}


def get_skill_manager(user_id: str) -> SkillManager:
    """Get or create a skill manager for a user."""
    if user_id not in _skill_managers:
        _skill_managers[user_id] = SkillManager(user_id)
    return _skill_managers[user_id]


@router.get("/skills", response_model=SkillsListResponse)
async def list_skills(
    user_id: str = Depends(CurrentUserId),
) -> SkillsListResponse:
    """List all available skills for the user."""
    manager = get_skill_manager(user_id)

    skills = []
    for skill in manager.get_all_skills():
        skills.append(SkillInfo(
            name=skill.name,
            prompt=skill.prompt[:200] + "..." if len(skill.prompt) > 200 else skill.prompt,
            description=skill.description,
            filename=skill.filename,
            is_active=skill.name in manager.active_skills,
        ))

    return SkillsListResponse(
        skills=skills,
        active_skills=manager.active_skills,
        skills_dir=str(manager.skills_dir),
    )


@router.post("/skills", response_model=SkillInfo)
async def create_skill(
    request: CreateSkillRequest,
    user_id: str = Depends(CurrentUserId),
) -> SkillInfo:
    """Create a new skill."""
    manager = get_skill_manager(user_id)

    try:
        skill = manager.create_skill(
            name=request.name,
            prompt=request.prompt,
            description=request.description,
        )
        return SkillInfo(
            name=skill.name,
            prompt=skill.prompt,
            description=skill.description,
            filename=skill.filename,
            is_active=False,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/skills/{skill_name}", response_model=SkillContentResponse)
async def get_skill_content(
    skill_name: str,
    user_id: str = Depends(CurrentUserId),
) -> SkillContentResponse:
    """Get the raw YAML content of a skill."""
    manager = get_skill_manager(user_id)

    result = manager.get_skill_content(skill_name)
    if not result:
        raise HTTPException(status_code=404, detail=f"Skill not found: {skill_name}")

    content, path = result
    return SkillContentResponse(
        name=skill_name,
        content=content,
        path=path,
    )


@router.put("/skills/{skill_name}")
async def update_skill(
    skill_name: str,
    request: UpdateSkillRequest,
    user_id: str = Depends(CurrentUserId),
) -> dict:
    """Update a skill's YAML content."""
    manager = get_skill_manager(user_id)

    try:
        success = manager.update_skill_content(skill_name, request.content)
        if not success:
            raise HTTPException(status_code=404, detail=f"Skill not found: {skill_name}")
        return {"status": "updated", "name": skill_name}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/skills/{skill_name}")
async def delete_skill(
    skill_name: str,
    user_id: str = Depends(CurrentUserId),
) -> dict:
    """Delete a skill."""
    manager = get_skill_manager(user_id)

    if not manager.delete_skill(skill_name):
        raise HTTPException(status_code=404, detail=f"Skill not found: {skill_name}")

    return {"status": "deleted", "name": skill_name}


@router.put("/skills/active", response_model=dict)
async def set_active_skills(
    request: SetActiveSkillsRequest,
    user_id: str = Depends(CurrentUserId),
) -> dict:
    """Set the active skills for the user."""
    manager = get_skill_manager(user_id)

    activated = manager.set_active_skills(request.skill_names)
    return {
        "status": "updated",
        "active_skills": activated,
    }


@router.get("/skills/prompt", response_model=dict)
async def get_skills_prompt(
    user_id: str = Depends(CurrentUserId),
) -> dict:
    """Get the combined prompt from all active skills."""
    manager = get_skill_manager(user_id)

    return {
        "prompt": manager.get_skills_prompt(),
        "active_skills": manager.active_skills,
    }
