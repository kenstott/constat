# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Skills REST endpoints (user-based, not session-based)."""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from constat.core.skills import SkillManager
from constat.server.auth import CurrentUserId
from constat.server.config import ServerConfig
from constat.server.session_manager import SessionManager

logger = logging.getLogger(__name__)

router = APIRouter()


def get_server_config(request: Request) -> ServerConfig:
    """Get server config from app state."""
    return request.app.state.server_config


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


class DraftSkillRequest(BaseModel):
    """Request to draft a skill using LLM."""
    name: str
    user_description: str  # Natural language description of the skill


class DraftSkillResponse(BaseModel):
    """Response with LLM-drafted skill content."""
    name: str
    content: str  # Full SKILL.md content (YAML frontmatter + markdown body)
    description: str


# Cache skill managers per (user_id, base_dir)
_skill_managers: dict[tuple[str, str], SkillManager] = {}


def get_skill_manager(user_id: str, base_dir: Path) -> SkillManager:
    """Get or create a skill manager for a user.

    Args:
        user_id: User identifier
        base_dir: Base .constat directory from server config
    """
    cache_key = (user_id, str(base_dir))
    if cache_key not in _skill_managers:
        logger.info(f"Creating SkillManager for user={user_id}, base_dir={base_dir}")
        _skill_managers[cache_key] = SkillManager(user_id, base_dir)
    return _skill_managers[cache_key]


@router.get("/skills")
async def list_skills(
    request: Request,
    user_id: CurrentUserId,
):
    """List all available skills for the user."""
    logger.info(f"[SKILLS] list_skills ENTERED, user_id={user_id}")
    try:
        server_config = get_server_config(request)
        manager = get_skill_manager(user_id, server_config.data_dir)
        logger.info(f"[SKILLS] manager.skills_dir={manager.skills_dir}, active_skills={manager.active_skills}")

        skills = []
        for skill in manager.get_all_skills():
            skills.append(SkillInfo(
                name=skill.name,
                prompt=skill.prompt[:200] + "..." if len(skill.prompt) > 200 else skill.prompt,
                description=skill.description or "",
                filename=skill.filename or "",
                is_active=skill.name in manager.active_skills,
            ))

        logger.info(f"[SKILLS] Returning {len(skills)} skills")
        return SkillsListResponse(
            skills=skills,
            active_skills=list(manager.active_skills),
            skills_dir=str(manager.skills_dir),
        )
    except Exception as e:
        logger.exception(f"[SKILLS] Error in list_skills: {e}")
        raise


@router.post("/skills", response_model=SkillInfo)
async def create_skill(
    request: Request,
    skill_request: CreateSkillRequest,
    user_id: CurrentUserId,
) -> SkillInfo:
    """Create a new skill."""
    server_config = get_server_config(request)
    manager = get_skill_manager(user_id, server_config.data_dir)

    try:
        skill = manager.create_skill(
            name=skill_request.name,
            prompt=skill_request.prompt,
            description=skill_request.description,
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


# NOTE: These routes MUST come BEFORE /skills/{skill_name} to avoid wildcard matching
@router.put("/skills/active", response_model=dict)
async def set_active_skills(
    request: Request,
    active_request: SetActiveSkillsRequest,
    user_id: CurrentUserId,
) -> dict:
    """Set the active skills for the user."""
    server_config = get_server_config(request)
    manager = get_skill_manager(user_id, server_config.data_dir)

    activated = manager.set_active_skills(active_request.skill_names)
    return {
        "status": "updated",
        "active_skills": activated,
    }


@router.get("/skills/prompt", response_model=dict)
async def get_skills_prompt(
    request: Request,
    user_id: CurrentUserId,
) -> dict:
    """Get the combined prompt from all active skills."""
    server_config = get_server_config(request)
    manager = get_skill_manager(user_id, server_config.data_dir)

    return {
        "prompt": manager.get_skills_prompt(),
        "active_skills": manager.active_skills,
    }


def get_session_manager(request: Request) -> SessionManager:
    """Dependency to get session manager from app state."""
    return request.app.state.session_manager


@router.post("/skills/draft", response_model=DraftSkillResponse)
async def draft_skill(
    request: Request,
    session_id: str,
    skill_request: DraftSkillRequest,
    user_id: CurrentUserId,
) -> DraftSkillResponse:
    """Use LLM to draft a skill based on user description."""
    session_manager = get_session_manager(request)
    managed = session_manager.get_session(session_id)
    if not managed or managed.user_id != user_id:
        raise HTTPException(status_code=404, detail="Session not found")

    session = managed.session
    if not hasattr(session, "llm"):
        raise HTTPException(status_code=500, detail="LLM not available")
    if not hasattr(session, "skill_manager"):
        raise HTTPException(status_code=500, detail="Skill manager not available")

    try:
        content, description = session.skill_manager.draft_skill(
            name=skill_request.name,
            user_description=skill_request.user_description,
            llm=session.llm,
        )
        return DraftSkillResponse(
            name=skill_request.name,
            content=content,
            description=description,
        )
    except ValueError as e:
        logger.error(f"Failed to draft skill: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to draft skill: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to draft skill: {str(e)}")


class CreateSkillFromProofRequest(BaseModel):
    name: str
    description: str = ""


class CreateSkillFromProofResponse(BaseModel):
    name: str
    content: str
    description: str
    has_script: bool


@router.post("/skills/from-proof", response_model=CreateSkillFromProofResponse)
async def create_skill_from_proof(
    request: Request,
    session_id: str,
    skill_request: CreateSkillFromProofRequest,
    user_id: CurrentUserId,
) -> CreateSkillFromProofResponse:
    """Create a skill from a completed proof."""
    session_manager = get_session_manager(request)
    managed = session_manager.get_session(session_id)
    if not managed or managed.user_id != user_id:
        raise HTTPException(status_code=404, detail="Session not found")

    session = managed.session

    if not session.last_proof_result:
        raise HTTPException(status_code=404, detail="No proof result available. Run /prove first.")

    proof_result = session.last_proof_result
    proof_nodes = proof_result.get("proof_nodes", [])
    proof_summary = proof_result.get("summary")
    original_problem = proof_result.get("problem", "")

    if not hasattr(session, "skill_manager"):
        raise HTTPException(status_code=500, detail="Skill manager not available")

    # Generate SKILL.md content via LLM
    try:
        content, description = session.skill_manager.skill_from_proof(
            name=skill_request.name,
            proof_nodes=proof_nodes,
            proof_summary=proof_summary,
            original_problem=original_problem,
            llm=session.llm,
            description=skill_request.description or None,
        )
    except Exception as e:
        logger.error(f"Failed to generate skill from proof: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate skill: {str(e)}")

    # Create the skill directory and SKILL.md
    try:
        skill = session.skill_manager.create_skill(
            name=skill_request.name,
            prompt="",
            description=description,
        )
        session.skill_manager.update_skill_content(skill_request.name, content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Generate and write scripts/proof.py
    has_script = False
    if session.history and session.session_id:
        try:
            from constat.server.routes.data import generate_inference_script, _gather_source_configs

            inferences = session.history.list_inference_codes(session.session_id)
            if inferences:
                apis, databases = _gather_source_configs(managed)
                premises = session.history.list_inference_premises(session.session_id)
                script_content = generate_inference_script(
                    inferences=inferences,
                    premises=premises,
                    apis=apis,
                    databases=databases,
                    session_label=session.session_id[:8],
                )

                safe_name = "".join(
                    c if c.isalnum() or c in "-_" else "-"
                    for c in skill_request.name.lower()
                )
                scripts_dir = session.skill_manager.skills_dir / safe_name / "scripts"
                scripts_dir.mkdir(parents=True, exist_ok=True)
                (scripts_dir / "proof.py").write_text(script_content)
                has_script = True
        except Exception as e:
            logger.warning(f"Failed to write proof script: {e}")

    # Invalidate cached skill manager so list_skills sees the new skill
    server_config = get_server_config(request)
    cache_key = (user_id, str(server_config.data_dir))
    if cache_key in _skill_managers:
        _skill_managers[cache_key].reload()

    return CreateSkillFromProofResponse(
        name=skill_request.name,
        content=content,
        description=description,
        has_script=has_script,
    )


# Wildcard routes MUST come after specific path routes like /skills/draft
@router.get("/skills/{skill_name}", response_model=SkillContentResponse)
async def get_skill_content(
    request: Request,
    skill_name: str,
    user_id: CurrentUserId,
) -> SkillContentResponse:
    """Get the raw YAML content of a skill."""
    server_config = get_server_config(request)
    manager = get_skill_manager(user_id, server_config.data_dir)

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
    request: Request,
    skill_name: str,
    update_request: UpdateSkillRequest,
    user_id: CurrentUserId,
) -> dict:
    """Update a skill's YAML content."""
    server_config = get_server_config(request)
    manager = get_skill_manager(user_id, server_config.data_dir)

    try:
        success = manager.update_skill_content(skill_name, update_request.content)
        if not success:
            raise HTTPException(status_code=404, detail=f"Skill not found: {skill_name}")
        return {"status": "updated", "name": skill_name}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/skills/{skill_name}")
async def delete_skill(
    request: Request,
    skill_name: str,
    user_id: CurrentUserId,
) -> dict:
    """Delete a skill."""
    server_config = get_server_config(request)
    manager = get_skill_manager(user_id, server_config.data_dir)

    if not manager.delete_skill(skill_name):
        raise HTTPException(status_code=404, detail=f"Skill not found: {skill_name}")

    return {"status": "deleted", "name": skill_name}
