# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Session management REST endpoints."""

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request

from constat.core.api import EntityManager
from constat.server.auth import CurrentUserId, CurrentUserEmail
from constat.server.models import (
    SessionCreate,
    SessionResponse,
    SessionListResponse,
    SessionStatus,
)
from constat.server.permissions import get_user_permissions
from constat.server.session_manager import SessionManager, ManagedSession
from constat.server.user_preferences import get_selected_projects, set_selected_projects
from constat.storage.history import SessionHistory

logger = logging.getLogger(__name__)

router = APIRouter()


def get_session_manager(request: Request) -> SessionManager:
    """Dependency to get session manager from app state."""
    return request.app.state.session_manager


def _session_to_response(managed: ManagedSession) -> SessionResponse:
    """Convert ManagedSession to SessionResponse."""
    tables_count = 0
    artifacts_count = 0
    summary = None

    if managed.session.datastore:
        try:
            tables = managed.session.datastore.list_tables()
            tables_count = len(tables)
        except Exception:
            pass

        try:
            artifacts = managed.session.datastore.list_artifacts()
            artifacts_count = len(artifacts)
        except Exception:
            pass

        # Get summary from datastore meta or history
        try:
            summary = managed.session.datastore.get_session_meta("summary")
        except Exception:
            pass

    # Try to get summary from history if not in datastore
    if not summary and managed.session.history:
        try:
            # Use find_session_by_server_id to get the history session ID
            history_session_id = managed.session.history.find_session_by_server_id(managed.session_id)
            if history_session_id:
                hist = managed.session.history.get_session(history_session_id)
                if hist:
                    summary = hist.summary
        except Exception:
            pass

    return SessionResponse(
        session_id=managed.session_id,
        user_id=managed.user_id,
        status=managed.status,
        created_at=managed.created_at,
        last_activity=managed.last_activity,
        current_query=managed.current_query,
        summary=summary,
        active_projects=managed.active_projects,
        tables_count=tables_count,
        artifacts_count=artifacts_count,
    )


def _load_projects_into_session(
    managed: ManagedSession,
    project_filenames: list[str],
) -> tuple[list[str], list[str]]:
    """Load projects into a session (helper for create_session and set_active_projects).

    Args:
        managed: The managed session
        project_filenames: List of project filenames to load

    Returns:
        Tuple of (successfully_loaded, conflicts)
    """
    if not project_filenames:
        return [], []

    config = managed.session.config
    conflicts = []

    # Load all projects and check they exist
    loaded_projects = []
    logger.debug(f"[_load_projects] Loading projects: {project_filenames}")
    logger.debug(f"[_load_projects] Available projects in config: {list(config.projects.keys())}")
    for filename in project_filenames:
        project = config.load_project(filename)
        if not project:
            logger.warning(f"Project not found when loading preferences: {filename} (available: {list(config.projects.keys())})")
            continue
        loaded_projects.append((filename, project))
        logger.debug(f"[_load_projects] Loaded project: {filename} -> {project.name}")

    if not loaded_projects:
        return [], []

    # Check for conflicts: collect all names from config and projects
    all_databases = {name: "config" for name in config.databases.keys()}
    all_apis = {name: "config" for name in config.apis.keys()}
    all_documents = {name: "config" for name in config.documents.keys()}

    valid_projects = []
    for filename, project in loaded_projects:
        has_conflict = False
        # Check database conflicts
        for name in project.databases.keys():
            if name in all_databases:
                conflicts.append(f"Database '{name}' conflicts: defined in {all_databases[name]} and {filename}")
                has_conflict = True
            else:
                all_databases[name] = filename

        # Check API conflicts
        for name in project.apis.keys():
            if name in all_apis:
                conflicts.append(f"API '{name}' conflicts: defined in {all_apis[name]} and {filename}")
                has_conflict = True
            else:
                all_apis[name] = filename

        # Check document conflicts
        for name in project.documents.keys():
            if name in all_documents:
                conflicts.append(f"Document '{name}' conflicts: defined in {all_documents[name]} and {filename}")
                has_conflict = True
            else:
                all_documents[name] = filename

        if not has_conflict:
            valid_projects.append((filename, project))

    # Load valid project databases into the session
    previously_loaded = getattr(managed, "_project_databases", set())
    newly_loaded = set()

    # Phase 1: Load all databases from all projects
    for filename, project in valid_projects:
        for name, db_config in project.databases.items():
            if name not in previously_loaded:
                try:
                    if managed.session.schema_manager:
                        success = managed.session.schema_manager.add_database_dynamic(name, db_config)
                        if success:
                            newly_loaded.add(name)
                            logger.info(f"Loaded project database: {name} from {filename}")
                except Exception as e:
                    logger.exception(f"Exception loading project database {name}: {e}")
            else:
                newly_loaded.add(name)

    # Phase 2: Documents are already indexed during server warmup with project_id
    # No need to re-index here - just log what's available
    for filename, project in valid_projects:
        if project.documents:
            logger.info(f"Project {filename}: {len(project.documents)} documents available (pre-indexed during warmup)")

    # Phase 3: Update schema entities and process metadata through NER (once, with all entities)
    if managed.session.schema_manager and newly_loaded:
        schema_entities = set(managed.session.schema_manager.get_entity_names())
        logger.info(f"Updating schema entities: {len(schema_entities)} entities (newly_loaded: {newly_loaded})")
        logger.debug(f"Schema entities include: {list(schema_entities)[:20]}...")
        managed.session.doc_tools.set_schema_entities(schema_entities)

        # Process schema metadata (names + descriptions) through NER for cross-datasource linking
        schema_metadata = managed.session.schema_manager.get_description_text()
        if schema_metadata:
            managed.session.doc_tools.process_metadata_through_ner(schema_metadata, source_type="schema")

        logger.info(f"Updated doc_tools schema entities: {len(schema_entities)} entities")

    # Register project APIs
    for filename, project in valid_projects:
        for api_name, api_config in project.apis.items():
            managed.session.add_project_api(api_name, api_config)
            logger.info(f"Registered project API: {api_name} from {filename}")

    # Register resources in consolidated SessionResources
    for filename, project in valid_projects:
        managed.session.add_project_resources(
            project_filename=filename,
            databases=project.databases,
            apis=project.apis,
            documents=project.documents,
        )
        logger.info(f"Registered project resources from {filename}")

    # Sync resources to session history (session.json)
    managed.session.sync_resources_to_history()

    # Store state
    managed._project_databases = newly_loaded
    managed.active_projects = [fn for fn, _ in valid_projects]

    # Update doc_tools with active project IDs for automatic search filtering
    if managed.session.doc_tools:
        managed.session.doc_tools._active_project_ids = managed.active_projects
        logger.debug(f"Set doc_tools._active_project_ids: {managed.active_projects}")

    return managed.active_projects, conflicts


@router.post("", response_model=SessionResponse)
async def create_session(
    user_id: CurrentUserId,
    body: SessionCreate,
    session_manager: SessionManager = Depends(get_session_manager),
) -> SessionResponse:
    """Create or reconnect to a session.

    If the client-provided session_id already exists on the server, reconnects
    to that session. Otherwise, creates a new session with that ID.

    When auth is disabled, uses "default" user or the user_id from request body.
    Automatically loads projects from user preferences if creating a new session.

    Returns:
        Session details including the session ID
    """
    # Use authenticated user_id, but allow body.user_id as fallback when auth disabled
    effective_user_id = user_id if user_id != "default" else (body.user_id or "default")
    client_session_id = body.session_id
    logger.info(f"[create_session] user_id={effective_user_id}, session_id={client_session_id}")

    # Check if session already exists (reconnect case)
    logger.debug(f"[create_session] checking for existing session...")
    existing = session_manager.get_session_or_none(client_session_id)
    if existing:
        logger.info(f"Reconnecting to existing session {client_session_id}")
        existing.touch()
        return _session_to_response(existing)

    # Create new session with client-provided session_id
    logger.debug(f"[create_session] creating new session...")
    session_id = session_manager.create_session(session_id=client_session_id, user_id=effective_user_id)
    logger.debug(f"[create_session] session created, getting managed session...")
    managed = session_manager.get_session(session_id)
    logger.debug(f"[create_session] got managed session")

    # Load preferred projects from user preferences (only for new sessions)
    preferred_projects = get_selected_projects(effective_user_id)
    logger.info(f"[create_session] preferred_projects from preferences: {preferred_projects}")
    if preferred_projects:
        logger.info(f"Loading {len(preferred_projects)} preferred projects for user {effective_user_id}")
        loaded, conflicts = _load_projects_into_session(managed, preferred_projects)
        logger.info(f"[create_session] after _load_projects_into_session: loaded={loaded}, conflicts={conflicts}, active_projects={managed.active_projects}")
        if conflicts:
            logger.warning(f"Project conflicts when loading preferences: {conflicts}")
        if loaded:
            logger.info(f"Loaded preferred projects: {loaded}")
            # Run NER for newly loaded project documents
            session_manager._run_entity_extraction(session_id, managed.session)
    else:
        logger.info(f"[create_session] No preferred projects found for user {effective_user_id}")

    return _session_to_response(managed)


@router.get("", response_model=SessionListResponse)
async def list_sessions(
    current_user_id: CurrentUserId,
    user_id: Optional[str] = None,
    session_manager: SessionManager = Depends(get_session_manager),
) -> SessionListResponse:
    """List all sessions (in-memory + historical from disk).

    When authenticated, only shows sessions for the current user.
    When auth is disabled, can optionally filter by user ID.

    Args:
        user_id: Optional user ID filter (only used when auth disabled)

    Returns:
        List of sessions, newest first
    """
    # Use authenticated user_id, but allow query param when auth disabled
    effective_user_id = current_user_id if current_user_id != "default" else (user_id or "default")

    # Get in-memory sessions
    in_memory = session_manager.list_sessions(user_id=effective_user_id)
    in_memory_ids = {s.session_id for s in in_memory}
    logger.debug(f"[list_sessions] in_memory_ids: {in_memory_ids}")

    # Convert in-memory sessions to responses
    responses = [_session_to_response(s) for s in in_memory]

    # Get historical sessions from disk (not already in memory)
    try:
        history = SessionHistory(user_id=effective_user_id)
        historical = history.list_sessions(limit=50)
        logger.debug(f"[list_sessions] historical: {[(h.session_id, h.server_session_id) for h in historical]}")

        for hist in historical:
            # Skip sessions without server_session_id or already in memory
            if not hist.server_session_id or hist.server_session_id in in_memory_ids:
                continue

            try:
                created_at = datetime.fromisoformat(hist.created_at.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                created_at = datetime.now(timezone.utc)

            responses.append(SessionResponse(
                session_id=hist.server_session_id,
                user_id=hist.user_id or effective_user_id,
                status=SessionStatus.IDLE,
                created_at=created_at,
                last_activity=created_at,
                current_query=hist.summary,
                summary=hist.summary,
                tables_count=0,
                artifacts_count=0,
            ))
    except Exception as e:
        logger.warning(f"Failed to load historical sessions: {e}")

    # Sort by last_activity descending
    responses.sort(key=lambda s: s.last_activity, reverse=True)

    return SessionListResponse(
        sessions=responses,
        total=len(responses),
    )


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> SessionResponse:
    """Get session details.

    Args:
        session_id: Session ID to retrieve

    Returns:
        Session details

    Raises:
        404: Session not found
    """
    managed = session_manager.get_session(session_id)
    return _session_to_response(managed)


@router.delete("/{session_id}")
async def delete_session(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict:
    """Delete a session.

    Closes and cleans up the session, releasing any resources.

    Args:
        session_id: Session ID to delete

    Returns:
        Deletion confirmation

    Raises:
        404: Session not found
    """
    if not session_manager.delete_session(session_id):
        raise KeyError(f"Session not found: {session_id}")

    return {
        "status": "deleted",
        "session_id": session_id,
    }


@router.post("/{session_id}/projects")
async def set_active_projects(
    session_id: str,
    body: dict,
    user_id: CurrentUserId,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict:
    """Set the active projects for a session.

    Projects define collections of databases, APIs, and documents.
    Setting projects merges those sources with the session's data sources.
    Also saves the selection to user preferences for future sessions.

    Incrementally manages entities:
    - Removed projects: Clear their entities from the session
    - Added projects: Extract entities for their chunks

    Args:
        session_id: Session ID
        body: Dict with 'projects' list of filenames (or empty to clear)

    Returns:
        Updated session info with any conflicts detected

    Raises:
        404: Session or project not found
        409: Conflict detected between projects or config
    """
    managed = session_manager.get_session(session_id)
    project_filenames = body.get("projects", [])

    if not isinstance(project_filenames, list):
        project_filenames = [project_filenames] if project_filenames else []

    # Verify all projects exist before loading
    config = managed.session.config
    for filename in project_filenames:
        project = config.load_project(filename)
        if not project:
            raise HTTPException(status_code=404, detail=f"Project not found: {filename}")

    # Determine which projects are being added vs removed
    old_projects = set(managed.active_projects or [])
    new_projects = set(project_filenames)

    logger.info(f"Session {session_id}: projects changing - removed={old_projects - new_projects}, added={new_projects - old_projects}")

    # Use EntityManager for incremental entity updates
    vector_store = managed.session.doc_tools._vector_store if managed.session.doc_tools else None
    if vector_store:
        entity_manager = EntityManager(
            vector_store=vector_store,
            schema_terms_provider=lambda: managed.session.schema_manager.get_entity_names() if managed.session.schema_manager else [],
            api_terms_provider=lambda: managed.session._get_api_entity_names() if hasattr(managed.session, '_get_api_entity_names') else [],
        )
        result = entity_manager.update_projects(session_id, old_projects, new_projects)
        if result.error:
            logger.warning(f"Entity update errors: {result.error}")
        logger.info(f"Session {session_id}: entity update - added={result.entities_added}, removed={result.entities_removed}")

    # Clear existing project state before loading new projects
    managed.session.clear_project_apis()
    managed._project_databases = set()
    managed.active_projects = []

    # Load projects using helper
    loaded, conflicts = _load_projects_into_session(managed, project_filenames)

    if conflicts:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "Conflicting data source names detected",
                "conflicts": conflicts,
            }
        )

    # Save to user preferences for future sessions
    effective_user_id = user_id if user_id != "default" else managed.user_id
    set_selected_projects(effective_user_id, project_filenames)
    logger.info(f"Saved project preferences for user {effective_user_id}: {project_filenames}")

    return {
        "status": "ok",
        "session_id": session_id,
        "active_projects": managed.active_projects,
    }


@router.get("/{session_id}/messages")
async def get_messages(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict:
    """Get conversation messages for a session.

    Returns stored messages for UI restoration after refresh/reconnect.

    Args:
        session_id: Session ID

    Returns:
        Dict with messages list

    Raises:
        404: Session not found
    """
    managed = session_manager.get_session(session_id)

    # Get messages from session history using server session ID
    # (works even before first query creates a history session)
    history = SessionHistory(user_id=managed.user_id or "default")
    messages = history.load_messages_by_server_id(session_id)

    return {"messages": messages}


@router.post("/{session_id}/messages")
async def save_messages(
    session_id: str,
    body: dict,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict:
    """Save conversation messages for a session.

    Persists messages for UI restoration after refresh/reconnect.

    Args:
        session_id: Session ID
        body: Dict with messages list

    Returns:
        Status confirmation

    Raises:
        404: Session not found
    """
    managed = session_manager.get_session(session_id)
    messages = body.get("messages", [])

    # Save messages to session history using server session ID
    # (works even before first query creates a history session)
    history = SessionHistory(user_id=managed.user_id or "default")
    history.save_messages_by_server_id(session_id, messages)

    return {"status": "saved", "count": len(messages)}


@router.get("/{session_id}/proof-facts")
async def get_proof_facts(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict:
    """Get proof facts for a session.

    Returns stored proof facts for UI restoration after refresh/reconnect.

    Args:
        session_id: Session ID

    Returns:
        Dict with facts list and optional summary

    Raises:
        404: Session not found
    """
    managed = session_manager.get_session(session_id)

    # Get proof facts from session history
    history = SessionHistory(user_id=managed.user_id or "default")
    facts, summary = history.load_proof_facts_by_server_id(session_id)

    return {"facts": facts, "summary": summary}


@router.post("/{session_id}/proof-facts")
async def save_proof_facts(
    session_id: str,
    body: dict,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict:
    """Save proof facts for a session.

    Persists proof facts for UI restoration after refresh/reconnect.

    Args:
        session_id: Session ID
        body: Dict with facts list and optional summary

    Returns:
        Status confirmation

    Raises:
        404: Session not found
    """
    managed = session_manager.get_session(session_id)
    facts = body.get("facts", [])
    summary = body.get("summary")

    # Save proof facts to session history
    history = SessionHistory(user_id=managed.user_id or "default")
    history.save_proof_facts_by_server_id(session_id, facts, summary)

    return {"status": "saved", "count": len(facts)}


@router.post("/{session_id}/reset-context")
async def reset_context(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict:
    """Reset session planning context for a new query.

    Clears session-level state while preserving user-level settings:
    - Clears: session facts, conversation history, plan context
    - Keeps: data sources, projects, learnings, permanent facts

    Args:
        session_id: Session ID

    Returns:
        Status confirmation
    """
    managed = session_manager.get_session(session_id)

    # Reset the session's planning context
    if managed.api:
        managed.api.reset_context()

    # Clear persisted messages
    history = SessionHistory(user_id=managed.user_id or "default")
    history.save_messages_by_server_id(session_id, [])

    return {"status": "reset"}


@router.get("/{session_id}/prompt-context")
async def get_prompt_context(
    session_id: str,
    user_id: CurrentUserId,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict:
    """Get the prompt context for a session.

    Returns all elements that contribute to the LLM prompt:
    - System prompt (from config)
    - Active role (if any)
    - Active skills (if any)

    Args:
        session_id: Session ID

    Returns:
        Dict with system_prompt, active_role, active_skills

    Raises:
        404: Session not found
    """
    managed = session_manager.get_session(session_id)
    if not managed or managed.user_id != user_id:
        raise HTTPException(status_code=404, detail="Session not found")

    session = managed.session

    # Get the full assembled system prompt (config + role + skills)
    # This matches what's actually used during execution
    system_prompt = session._get_system_prompt() if hasattr(session, '_get_system_prompt') else (session.config.system_prompt or "")
    logger.info(f"[prompt-context] system_prompt length: {len(system_prompt)}, has _get_system_prompt: {hasattr(session, '_get_system_prompt')}")

    # Get active role
    active_role = None
    role_prompt = ""
    if hasattr(session, "role_manager"):
        role_name = session.role_manager.active_role_name
        if role_name:
            role = session.role_manager.get_role(role_name)
            active_role = {
                "name": role_name,
                "prompt": role.prompt if role else "",
            }
            role_prompt = role.prompt if role else ""

    # Get active skills
    active_skills = []
    if hasattr(session, "skill_manager"):
        for name in session.skill_manager.active_skills:
            skill = session.skill_manager.get_skill(name)
            if skill:
                active_skills.append({
                    "name": skill.name,
                    "prompt": skill.prompt,
                    "description": skill.description,
                })

    return {
        "system_prompt": system_prompt,
        "active_role": active_role,
        "active_skills": active_skills,
    }


@router.put("/{session_id}/system-prompt")
async def update_system_prompt(
    session_id: str,
    request: Request,
    body: dict,
    user_id: CurrentUserId,
    email: str = Depends(CurrentUserEmail),
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict:
    """Update the system prompt (admin only).

    Updates the in-memory config system prompt. Changes persist until server restart.

    Args:
        session_id: Session ID
        body: Dict with system_prompt field

    Returns:
        Status and updated system_prompt

    Raises:
        403: Not an admin
        404: Session not found
    """
    # Check admin permission
    server_config = request.app.state.server_config
    perms = get_user_permissions(server_config, email=email or "", user_id=user_id)
    if not perms.admin:
        raise HTTPException(status_code=403, detail="Admin access required")

    managed = session_manager.get_session(session_id)
    if not managed or managed.user_id != user_id:
        raise HTTPException(status_code=404, detail="Session not found")

    new_prompt = body.get("system_prompt", "")

    # Update the global config
    config = request.app.state.config
    config.system_prompt = new_prompt

    # Update the session's config reference
    managed.session.config.system_prompt = new_prompt

    return {
        "status": "updated",
        "system_prompt": new_prompt,
    }


@router.post("/{session_id}/match-context")
async def match_dynamic_context(
    session_id: str,
    body: dict,
    user_id: CurrentUserId,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict:
    """Preview dynamic context matching for a query.

    Returns which role and skills would be matched for the given query
    without actually running the query.

    Args:
        session_id: Session ID
        body: Dict with 'query' field

    Returns:
        Dict with:
            - role: Optional dict with name, description, similarity
            - skills: List of dicts with name, description, similarity
            - role_source: "skill" or "query" indicating how role was selected
    """
    managed = session_manager.get_session(session_id)
    if not managed or managed.user_id != user_id:
        raise HTTPException(status_code=404, detail="Session not found")

    query = body.get("query", "")
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query is required")

    session = managed.session
    context = session.get_dynamic_context(query)

    return {
        "role": context.get("role"),
        "skills": context.get("skills"),
        "role_source": context.get("role_source"),
    }
