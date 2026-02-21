# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Agents REST endpoints."""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request

from constat.server.persona_config import require_write
from pydantic import BaseModel

from constat.server.auth import CurrentUserId
from constat.server.session_manager import SessionManager

logger = logging.getLogger(__name__)

router = APIRouter()


class AgentInfo(BaseModel):
    """Agent information."""
    name: str
    prompt: str
    description: str = ""
    skills: list[str] = []
    is_active: bool = False


class AgentsListResponse(BaseModel):
    """Response for listing agents."""
    agents: list[AgentInfo]
    current_agent: Optional[str] = None
    agents_file: str


class SetAgentRequest(BaseModel):
    """Request to set the active agent."""
    agent_name: Optional[str] = None  # None to clear agent


class SetAgentResponse(BaseModel):
    """Response after setting agent."""
    success: bool
    current_agent: Optional[str] = None
    message: str


class CreateAgentRequest(BaseModel):
    """Request to create an agent."""
    name: str
    prompt: str
    description: str = ""
    skills: list[str] = []


class UpdateAgentRequest(BaseModel):
    """Request to update an agent."""
    prompt: str
    description: str = ""
    skills: list[str] = []


class AgentContentResponse(BaseModel):
    """Response with full agent content."""
    name: str
    prompt: str
    description: str
    skills: list[str] = []
    path: str


class DraftAgentRequest(BaseModel):
    """Request to draft an agent using LLM."""
    name: str
    user_description: str  # Natural language description of the agent


class DraftAgentResponse(BaseModel):
    """Response with LLM-drafted agent content."""
    name: str
    prompt: str
    description: str
    skills: list[str] = []


def get_session_manager(request: Request) -> SessionManager:
    """Dependency to get session manager from app state."""
    return request.app.state.session_manager


@router.get("/agents")
async def list_agents(
    _request: Request,
    session_id: str,
    user_id: CurrentUserId,
    session_manager: SessionManager = Depends(get_session_manager),
):
    """List all available agents."""
    logger.info(f"[AGENTS] list_agents ENTERED: session_id={session_id}, user_id={user_id}")

    managed = session_manager.get_session(session_id)
    if not managed or managed.user_id != user_id:
        logger.warning(f"[AGENTS] Session not found or user mismatch: managed={managed is not None}, managed.user_id={managed.user_id if managed else 'N/A'}, user_id={user_id}")
        raise HTTPException(status_code=404, detail="Session not found")

    session = managed.session
    if not hasattr(session, "agent_manager"):
        logger.error("[AGENTS] Agent manager not available on session")
        raise HTTPException(status_code=500, detail="Agent manager not available")

    agent_manager = session.agent_manager
    logger.info(f"[AGENTS] Agent manager: agents_file={agent_manager.agents_file_path}, exists={agent_manager.agents_file_path.exists()}, has_agents={agent_manager.has_agents}")

    current_agent = agent_manager.active_agent_name

    agents = []
    for name in agent_manager.list_agents():
        agent = agent_manager.get_agent(name)
        if agent:
            agents.append(AgentInfo(
                name=name,
                prompt=agent.prompt[:200] + "..." if len(agent.prompt) > 200 else agent.prompt,
                description=agent.description,
                skills=agent.skills,
                is_active=(name == current_agent),
            ))

    logger.info(f"[AGENTS] Returning {len(agents)} agents, current_agent={current_agent}")
    return AgentsListResponse(
        agents=agents,
        current_agent=current_agent,
        agents_file=str(agent_manager.agents_file_path),
    )


@router.put("/agents/current", response_model=SetAgentResponse)
async def set_current_agent(
    session_id: str,
    request: SetAgentRequest,
    user_id: CurrentUserId,
    session_manager: SessionManager = Depends(get_session_manager),
) -> SetAgentResponse:
    """Set the active agent for the session."""
    # noinspection DuplicatedCode
    managed = session_manager.get_session(session_id)
    if not managed or managed.user_id != user_id:
        raise HTTPException(status_code=404, detail="Session not found")

    session = managed.session
    if not hasattr(session, "agent_manager"):
        raise HTTPException(status_code=500, detail="Agent manager not available")

    agent_manager = session.agent_manager
    agent_name = request.agent_name

    if agent_name is None or agent_name.lower() == "none" or agent_name == "":
        # Clear the agent
        agent_manager.set_active_agent(None)
        return SetAgentResponse(
            success=True,
            current_agent=None,
            message="Agent cleared",
        )

    # Try to set the agent
    if agent_manager.set_active_agent(agent_name):
        return SetAgentResponse(
            success=True,
            current_agent=agent_name,
            message=f"Agent set to '{agent_name}'",
        )
    else:
        available = agent_manager.list_agents()
        raise HTTPException(
            status_code=400,
            detail=f"Agent not found: {agent_name}. Available: {', '.join(available) or 'none'}",
        )


@router.get("/agents/{agent_name}", response_model=AgentContentResponse)
async def get_agent_content(
    session_id: str,
    agent_name: str,
    user_id: CurrentUserId,
    session_manager: SessionManager = Depends(get_session_manager),
) -> AgentContentResponse:
    """Get the full content of an agent."""
    # noinspection DuplicatedCode
    managed = session_manager.get_session(session_id)
    if not managed or managed.user_id != user_id:
        raise HTTPException(status_code=404, detail="Session not found")

    session = managed.session
    if not hasattr(session, "agent_manager"):
        raise HTTPException(status_code=500, detail="Agent manager not available")

    agent_manager = session.agent_manager
    agent = agent_manager.get_agent(agent_name)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_name}")

    return AgentContentResponse(
        name=agent.name,
        prompt=agent.prompt,
        description=agent.description,
        skills=agent.skills,
        path=str(agent_manager.agents_file_path),
    )


@router.post("/agents", response_model=AgentInfo, dependencies=[Depends(require_write("agents"))])
async def create_agent(
    session_id: str,
    request_body: CreateAgentRequest,
    user_id: CurrentUserId,
    session_manager: SessionManager = Depends(get_session_manager),
) -> AgentInfo:
    """Create a new agent."""
    # noinspection DuplicatedCode
    managed = session_manager.get_session(session_id)
    if not managed or managed.user_id != user_id:
        raise HTTPException(status_code=404, detail="Session not found")

    session = managed.session
    if not hasattr(session, "agent_manager"):
        raise HTTPException(status_code=500, detail="Agent manager not available")

    agent_manager = session.agent_manager

    try:
        agent = agent_manager.create_agent(
            name=request_body.name,
            prompt=request_body.prompt,
            description=request_body.description,
            skills=request_body.skills,
        )
        return AgentInfo(
            name=agent.name,
            prompt=agent.prompt,
            description=agent.description,
            skills=agent.skills,
            is_active=False,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/agents/{agent_name}", dependencies=[Depends(require_write("agents"))])
async def update_agent(
    session_id: str,
    agent_name: str,
    request_body: UpdateAgentRequest,
    user_id: CurrentUserId,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict:
    """Update an agent."""
    # noinspection DuplicatedCode
    managed = session_manager.get_session(session_id)
    if not managed or managed.user_id != user_id:
        raise HTTPException(status_code=404, detail="Session not found")

    session = managed.session
    if not hasattr(session, "agent_manager"):
        raise HTTPException(status_code=500, detail="Agent manager not available")

    agent_manager = session.agent_manager

    # Check if agent exists
    if not agent_manager.get_agent(agent_name):
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_name}")

    agent_manager.update_agent(
        name=agent_name,
        prompt=request_body.prompt,
        description=request_body.description,
        skills=request_body.skills,
    )
    return {"status": "updated", "name": agent_name}


@router.delete("/agents/{agent_name}", dependencies=[Depends(require_write("agents"))])
async def delete_agent(
    session_id: str,
    agent_name: str,
    user_id: CurrentUserId,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict:
    """Delete an agent."""
    # noinspection DuplicatedCode
    managed = session_manager.get_session(session_id)
    if not managed or managed.user_id != user_id:
        raise HTTPException(status_code=404, detail="Session not found")

    session = managed.session
    if not hasattr(session, "agent_manager"):
        raise HTTPException(status_code=500, detail="Agent manager not available")

    agent_manager = session.agent_manager

    if not agent_manager.delete_agent(agent_name):
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_name}")

    return {"status": "deleted", "name": agent_name}


@router.post("/agents/draft", response_model=DraftAgentResponse)
async def draft_agent(
    session_id: str,
    request_body: DraftAgentRequest,
    user_id: CurrentUserId,
    session_manager: SessionManager = Depends(get_session_manager),
) -> DraftAgentResponse:
    """Use LLM to draft an agent based on user description."""
    managed = session_manager.get_session(session_id)
    if not managed or managed.user_id != user_id:
        raise HTTPException(status_code=404, detail="Session not found")

    session = managed.session
    if not hasattr(session, "llm"):
        raise HTTPException(status_code=500, detail="LLM not available")
    if not hasattr(session, "agent_manager"):
        raise HTTPException(status_code=500, detail="Agent manager not available")

    # Gather available skills for the LLM to choose from
    available_skills: list[dict[str, str]] = []
    if hasattr(session, "skill_manager"):
        for skill in session.skill_manager.list_skills():
            available_skills.append({
                "name": skill.name,
                "description": skill.description or "",
            })

    try:
        agent = session.agent_manager.draft_agent(
            name=request_body.name,
            user_description=request_body.user_description,
            llm=session.llm,
            available_skills=available_skills,
        )
        return DraftAgentResponse(
            name=agent.name,
            prompt=agent.prompt,
            description=agent.description,
            skills=agent.skills,
        )
    except ValueError as e:
        logger.error(f"Failed to draft agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to draft agent: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to draft agent: {str(e)}")
