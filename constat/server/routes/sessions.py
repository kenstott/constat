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
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request

from constat.core.api import EntityManager
from constat.server.auth import CurrentUserId
from constat.server.models import (
    SessionCreate,
    SessionResponse,
    SessionListResponse,
    SessionStatus,
    ShareSessionRequest,
    ShareSessionResponse,
    TogglePublicRequest,
    TogglePublicResponse,
)
from constat.server.persona_config import require_write
from constat.server.session_manager import SessionManager, ManagedSession
from constat.server.user_preferences import get_selected_domains, set_selected_domains
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
        except (KeyError, ValueError, OSError):
            pass

        try:
            artifacts = managed.session.datastore.list_artifacts()
            artifacts_count = len(artifacts)
        except (KeyError, ValueError, OSError):
            pass

        # Get summary from datastore meta or history
        try:
            summary = managed.session.datastore.get_session_meta("summary")
        except (KeyError, ValueError, OSError):
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
        except (KeyError, ValueError, OSError):
            pass

    # Get shared users and public flag from datastore
    shared_with: list[str] = []
    is_public = False
    if managed.session.datastore:
        try:
            shared_with = managed.session.datastore.get_shared_users()
        except (KeyError, ValueError, OSError):
            pass
        try:
            is_public = managed.session.datastore.is_public()
        except (KeyError, ValueError, OSError):
            pass

    return SessionResponse(
        session_id=managed.session_id,
        user_id=managed.user_id,
        status=managed.status,
        created_at=managed.created_at,
        last_activity=managed.last_activity,
        current_query=managed.current_query,
        summary=summary,
        active_domains=managed.active_domains,
        tables_count=tables_count,
        artifacts_count=artifacts_count,
        shared_with=shared_with,
        is_public=is_public,
    )


def _load_domains_into_session(
    managed: ManagedSession,
    domain_filenames: list[str],
) -> tuple[list[str], list[str]]:
    """Load domains into a session (helper for create_session and set_active_domains).

    Args:
        managed: The managed session
        domain_filenames: List of domain filenames to load

    Returns:
        Tuple of (successfully_loaded, conflicts)
    """
    if not domain_filenames:
        return [], []

    config = managed.session.config
    conflicts = []

    # Load all domains and check they exist
    loaded_domains = []
    logger.debug(f"[_load_domains] Loading domains: {domain_filenames}")
    logger.debug(f"[_load_domains] Available domains in config: {list(config.domains.keys())}")
    for filename in domain_filenames:
        domain = config.load_domain(filename)
        if not domain:
            logger.warning(f"Domain not found when loading preferences: {filename} (available: {list(config.domains.keys())})")
            continue
        loaded_domains.append((filename, domain))
        logger.debug(f"[_load_domains] Loaded domain: {filename} -> {domain.name}")

    if not loaded_domains:
        return [], []

    # Collect manual aliases from all loaded domains (keyed by target domain filename)
    manual_aliases: dict[str, dict[str, dict[str, str]]] = {}
    for _, domain in loaded_domains:
        for target_domain, type_aliases in domain.aliases.items():
            dest = manual_aliases.setdefault(target_domain, {})
            for resource_type, key_map in type_aliases.items():
                dest.setdefault(resource_type, {}).update(key_map)

    # Apply manual aliases to create working resource dicts per domain
    def _apply_manual(resources: dict, rename_map: dict[str, str]) -> dict:
        if not rename_map:
            return dict(resources)
        return {rename_map.get(k, k): v for k, v in resources.items()}

    aliased_domains: list[tuple[str, object, dict, dict, dict]] = []
    for filename, domain in loaded_domains:
        ma = manual_aliases.get(filename, {})
        aliased_domains.append((
            filename, domain,
            _apply_manual(domain.databases, ma.get("databases", {})),
            _apply_manual(domain.apis, ma.get("apis", {})),
            _apply_manual(domain.documents, ma.get("documents", {})),
        ))

    # Resolve conflicts: auto-alias any remaining key collisions
    all_databases = {name: "config" for name in config.databases.keys()}
    all_apis = {name: "config" for name in config.apis.keys()}
    all_documents = {name: "config" for name in config.documents.keys()}

    def _auto_alias(key: str, domain_stem: str, taken: dict[str, str]) -> str:
        candidate = f"{domain_stem}--{key}"
        if candidate not in taken:
            return candidate
        i = 2
        while f"{candidate}--{i}" in taken:
            i += 1
        return f"{candidate}--{i}"

    # Track original->final key mappings per domain (for entity resolution source remapping)
    domain_alias_map: dict[str, dict[str, dict[str, str]]] = {}

    def _resolve_resources(
        original: dict, manually_aliased: dict, taken: dict[str, str],
        stem: str, resource_type: str, filename: str, alias_map: dict,
    ) -> dict:
        """Resolve a resource dict: auto-alias any remaining conflicts, track mappings."""
        resolved = {}
        for name, cfg in manually_aliased.items():
            if name in taken:
                new_name = _auto_alias(name, stem, taken)
                logger.info(f"Auto-aliased {resource_type[:-1]} '{name}' -> '{new_name}' (conflict with {taken[name]})")
                resolved[new_name] = cfg
                taken[new_name] = filename
                # Find original key for this config value
                orig = next((k for k, v in original.items() if v is cfg), name)
                alias_map.setdefault(resource_type, {})[orig] = new_name
            else:
                resolved[name] = cfg
                taken[name] = filename
                # Track manual alias if key changed
                orig = next((k for k, v in original.items() if v is cfg), name)
                if orig != name:
                    alias_map.setdefault(resource_type, {})[orig] = name
        return resolved

    valid_domains: list[tuple[str, object, dict, dict, dict]] = []
    for filename, domain, dbs, apis, docs in aliased_domains:
        stem = Path(filename).stem
        alias_map: dict[str, dict[str, str]] = {}

        resolved_dbs = _resolve_resources(domain.databases, dbs, all_databases, stem, "databases", filename, alias_map)
        resolved_apis = _resolve_resources(domain.apis, apis, all_apis, stem, "apis", filename, alias_map)
        resolved_docs = _resolve_resources(domain.documents, docs, all_documents, stem, "documents", filename, alias_map)

        if alias_map:
            domain_alias_map[filename] = alias_map
        valid_domains.append((filename, domain, resolved_dbs, resolved_apis, resolved_docs))

    # Load domain databases into the session
    previously_loaded = getattr(managed, "_domain_databases", set())
    newly_loaded = set()

    # Phase 1: Load all databases from all domains (parallel)
    from concurrent.futures import ThreadPoolExecutor, as_completed

    to_load: list[tuple[str, str, object]] = []  # (name, filename, db_config)
    for filename, domain, dbs, apis, docs in valid_domains:
        for name, db_config in dbs.items():
            if name not in previously_loaded:
                to_load.append((name, filename, db_config))
            else:
                newly_loaded.add(name)

    if to_load and managed.session.schema_manager:
        def _load_db(item: tuple) -> tuple[str, str, bool]:
            name, filename, db_config = item
            try:
                success = managed.session.schema_manager.add_database_dynamic(name, db_config)
                return name, filename, success
            except Exception as e:
                logger.exception(f"Exception loading domain database {name}: {e}")
                return name, filename, False

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(_load_db, item) for item in to_load]
            for future in as_completed(futures):
                name, filename, success = future.result()
                if success:
                    newly_loaded.add(name)
                    logger.info(f"Loaded domain database: {name} from {filename}")

    # Phase 2: Documents are already indexed during server warmup with domain_id
    # No need to re-index here - just log what's available
    for filename, domain, dbs, apis, docs in valid_domains:
        if docs:
            logger.info(f"Domain {filename}: {len(docs)} documents available (pre-indexed during warmup)")

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

    # Register domain APIs (using aliased names)
    for filename, domain, dbs, apis, docs in valid_domains:
        for api_name, api_config in apis.items():
            managed.session.add_domain_api(api_name, api_config)
            logger.info(f"Registered domain API: {api_name} from {filename}")

    # Register resources in consolidated SessionResources (using aliased names)
    for filename, domain, dbs, apis, docs in valid_domains:
        managed.session.add_domain_resources(
            domain_filename=filename,
            databases=dbs,
            apis=apis,
            documents=docs,
        )
        logger.info(f"Registered domain resources from {filename}")

    # Register domain skills directories
    for filename, domain, dbs, apis, docs in valid_domains:
        if domain.source_path:
            domain_dir = Path(domain.source_path).parent
            domain_skills_dir = domain_dir / "skills"
            managed.session.skill_manager.add_domain_skills(domain_skills_dir, domain_filename=filename)
            # Also register domain agents
            if hasattr(managed.session, "agent_manager"):
                managed.session.agent_manager.add_domain_agents(domain_dir, domain_filename=filename)

    # Sync resources to session history (session.json)
    managed.session.sync_resources_to_history()

    # Store state
    managed._domain_databases = newly_loaded
    managed._domain_alias_map = domain_alias_map
    managed.active_domains = [fn for fn, _, _, _, _ in valid_domains]

    # Update doc_tools with active domain IDs for automatic search filtering
    if managed.session.doc_tools:
        managed.session.doc_tools._active_domain_ids = managed.active_domains
        logger.debug(f"Set doc_tools._active_domain_ids: {managed.active_domains}")

    return managed.active_domains, conflicts


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
    Automatically loads domains from user preferences if creating a new session.

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
        # ACL check: owner or shared user
        shared_with = []
        if existing.session.datastore:
            try:
                shared_with = existing.session.datastore.get_shared_users()
            except (KeyError, ValueError, OSError):
                pass
        if existing.user_id != effective_user_id and effective_user_id not in shared_with:
            raise HTTPException(status_code=403, detail="Not authorized to access this session")
        logger.info(f"Reconnecting to existing session {client_session_id}")
        existing.touch()
        return _session_to_response(existing)

    # Create new session with client-provided session_id
    logger.debug(f"[create_session] creating new session...")
    session_id = session_manager.create_session(session_id=client_session_id, user_id=effective_user_id)
    logger.debug(f"[create_session] session created, getting managed session...")
    managed = session_manager.get_session(session_id)
    logger.debug(f"[create_session] got managed session")

    # Seed per-session prompt from global config
    managed.session_prompt = managed.session.config.system_prompt

    # Load preferred domains from user preferences (only for new sessions)
    preferred_domains = get_selected_domains(effective_user_id)
    logger.info(f"[create_session] preferred_domains from preferences: {preferred_domains}")
    if preferred_domains:
        logger.info(f"Loading {len(preferred_domains)} preferred domains for user {effective_user_id}")
        loaded, conflicts = _load_domains_into_session(managed, preferred_domains)
        logger.info(f"[create_session] after _load_domains_into_session: loaded={loaded}, conflicts={conflicts}, active_domains={managed.active_domains}")
        if conflicts:
            logger.warning(f"Domain conflicts when loading preferences: {conflicts}")
        if loaded:
            logger.info(f"Loaded preferred domains: {loaded}")
            # Re-resolve config with active domains
            session_manager.resolve_config(session_id)
    else:
        logger.info(f"[create_session] No preferred domains found for user {effective_user_id}")

    # Seed 'user' domain as active by default (toggleable by user)
    if 'user' not in managed.active_domains:
        managed.active_domains.append('user')

    # Run NER after domain loading (or even with no domains) so schema
    # entities are available for pattern matching in entity extraction.
    session_manager.refresh_entities_async(session_id)

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
        current_user_id: Authenticated user ID
        user_id: Optional user ID filter (only used when auth disabled)
        session_manager: Injected session manager

    Returns:
        List of sessions, newest first
    """
    # Use authenticated user_id, but allow query param when auth disabled
    effective_user_id = current_user_id if current_user_id != "default" else (user_id or "default")

    # Get in-memory sessions (owned by user)
    in_memory = session_manager.list_sessions(user_id=effective_user_id)

    # Also include sessions shared with this user
    all_sessions = session_manager.list_sessions()
    for s in all_sessions:
        if s.user_id != effective_user_id and s.session.datastore:
            try:
                if effective_user_id in s.session.datastore.get_shared_users():
                    in_memory.append(s)
            except (KeyError, ValueError, OSError):
                pass

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
        session_manager: Injected session manager

    Returns:
        Session details

    Raises:
        404: Session not found
    """
    managed = session_manager.get_session_or_none(session_id)
    if not managed:
        raise HTTPException(status_code=404, detail="Session not found")
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
        session_manager: Injected session manager

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


@router.get("/{session_id}/routing")
async def get_session_routing(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, dict[str, list[dict]]]:
    """Get model routing layers for this session: system, user, and per-domain overrides."""
    from constat.server.models import ModelRouteInfo

    managed = session_manager.get_session_or_none(session_id)
    if not managed:
        raise HTTPException(status_code=404, detail="Session not found")

    session = managed.session
    if not hasattr(session, "router") or not session.router:
        raise HTTPException(status_code=503, detail="Router not available")

    layers = session.router.get_routing_layers(
        active_domains=managed.active_domains
    )
    default_provider = session.config.llm.provider

    result: dict[str, dict[str, list[dict]]] = {}
    for layer_name, routes in layers.items():
        result[layer_name] = {}
        for task_type, specs in routes.items():
            result[layer_name][task_type] = [
                ModelRouteInfo(
                    provider=spec.provider or default_provider,
                    model=spec.model,
                ).model_dump()
                for spec in specs
            ]
    return result


@router.post("/{session_id}/domains")
async def set_active_domains(
    session_id: str,
    body: dict,
    user_id: CurrentUserId,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict:
    """Set the active domains for a session.

    Domains define collections of databases, APIs, and documents.
    Setting domains merges those sources with the session's data sources.
    Also saves the selection to user preferences for future sessions.

    Incrementally manages entities:
    - Removed domains: Clear their entities from the session
    - Added domains: Extract entities for their chunks

    Args:
        session_id: Session ID
        body: Dict with 'domains' list of filenames (or empty to clear)
        user_id: Authenticated user ID
        session_manager: Injected session manager

    Returns:
        Updated session info with any conflicts detected

    Raises:
        404: Session or domain not found
        409: Conflict detected between domains or config
    """
    managed = session_manager.get_session_or_none(session_id)
    if not managed:
        raise HTTPException(status_code=404, detail="Session not found")
    domain_filenames = body.get("domains", body.get("projects", []))

    # noinspection PyTypeChecker
    if not isinstance(domain_filenames, list):
        domain_filenames = [domain_filenames] if domain_filenames else []

    # Verify all domains exist before loading (skip synthetic nodes)
    synthetic = {'root', 'user'}
    config = managed.session.config
    for filename in domain_filenames:
        if filename in synthetic:
            continue
        domain = config.load_domain(filename)  # type: ignore[arg-type]
        if not domain:
            raise HTTPException(status_code=404, detail=f"Domain not found: {filename}")

    # Determine which domains are being added vs removed
    old_domains = set(managed.active_domains or [])
    new_domains = set(domain_filenames)

    logger.info(f"Session {session_id}: domains changing - removed={old_domains - new_domains}, added={new_domains - old_domains}")

    # Use EntityManager for incremental entity updates
    vector_store = managed.session.doc_tools._vector_store if managed.session.doc_tools else None
    if vector_store:
        entity_manager = EntityManager(
            vector_store=vector_store,
            schema_terms_provider=lambda: managed.session.schema_manager.get_entity_names() if managed.session.schema_manager else [],
            api_terms_provider=lambda: managed.session._get_api_entity_names() if hasattr(managed.session, '_get_api_entity_names') else [],
        )
        result = entity_manager.update_domains(session_id, old_domains, new_domains)
        if result.error:
            logger.warning(f"Entity update errors: {result.error}")
        logger.info(f"Session {session_id}: entity update - added={result.entities_added}, removed={result.entities_removed}")

    # Clear existing domain state before loading new domains
    managed.session.clear_domain_apis()
    managed._domain_databases = set()
    managed.active_domains = []

    # Load real domains (skip synthetic nodes)
    real_filenames = [f for f in domain_filenames if f not in synthetic]
    synthetic_active = [f for f in domain_filenames if f in synthetic]
    loaded, conflicts = _load_domains_into_session(managed, real_filenames)

    # Re-add synthetic nodes to active_domains
    if synthetic_active:
        managed.active_domains = synthetic_active + managed.active_domains

    if conflicts:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "Conflicting data source names detected",
                "conflicts": conflicts,
            }
        )

    # Re-resolve tiered config with new domains
    session_manager.resolve_config(session_id)

    # Load single domain's system_prompt into session
    if len(real_filenames) == 1:
        single = config.load_domain(real_filenames[0])
        if single and single.system_prompt:
            managed.session.config.system_prompt = single.system_prompt
            managed.session_prompt = single.system_prompt

    # Save to user preferences for future sessions (exclude synthetic nodes)
    effective_user_id = user_id if user_id != "default" else managed.user_id
    set_selected_domains(effective_user_id, real_filenames)
    logger.info(f"Saved domain preferences for user {effective_user_id}: {domain_filenames}")

    return {
        "status": "ok",
        "session_id": session_id,
        "active_domains": managed.active_domains,
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
        session_manager: Injected session manager

    Returns:
        Dict with messages list

    Raises:
        404: Session not found
    """
    managed = session_manager.get_session_or_none(session_id)
    if not managed:
        raise HTTPException(status_code=404, detail="Session not found")

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
        session_manager: Injected session manager

    Returns:
        Status confirmation

    Raises:
        404: Session not found
    """
    managed = session_manager.get_session_or_none(session_id)
    if not managed:
        raise HTTPException(status_code=404, detail="Session not found")
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
        session_manager: Injected session manager

    Returns:
        Dict with facts list and optional summary

    Raises:
        404: Session not found
    """
    managed = session_manager.get_session_or_none(session_id)
    if not managed:
        raise HTTPException(status_code=404, detail="Session not found")

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
        session_manager: Injected session manager

    Returns:
        Status confirmation

    Raises:
        404: Session not found
    """
    managed = session_manager.get_session_or_none(session_id)
    if not managed:
        raise HTTPException(status_code=404, detail="Session not found")
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
    - Keeps: data sources, domains, learnings, permanent facts

    Args:
        session_id: Session ID
        session_manager: Injected session manager

    Returns:
        Status confirmation
    """
    managed = session_manager.get_session_or_none(session_id)
    if not managed:
        raise HTTPException(status_code=404, detail="Session not found")

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
        user_id: Authenticated user ID
        session_manager: Injected session manager

    Returns:
        Dict with system_prompt, active_agent, active_skills

    Raises:
        404: Session not found
    """
    managed = session_manager.get_session_or_none(session_id)
    if not managed or managed.user_id != user_id:
        raise HTTPException(status_code=404, detail="Session not found")

    session = managed.session

    # Get the full assembled system prompt (config + role + skills)
    # This matches what's actually used during execution
    system_prompt = session._get_system_prompt() if hasattr(session, '_get_system_prompt') else (session.config.system_prompt or "")
    logger.info(f"[prompt-context] system_prompt length: {len(system_prompt)}, has _get_system_prompt: {hasattr(session, '_get_system_prompt')}")

    # Get active agent
    active_agent = None
    if hasattr(session, "agent_manager"):
        agent_name = session.agent_manager.active_agent_name
        if agent_name:
            agent = session.agent_manager.get_agent(agent_name)
            active_agent = {
                "name": agent_name,
                "prompt": agent.prompt if agent else "",
            }

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
        "active_agent": active_agent,
        "active_skills": active_skills,
    }


@router.put("/{session_id}/system-prompt", dependencies=[Depends(require_write("system_prompt"))])
async def update_system_prompt(
    session_id: str,
    body: dict,
    user_id: CurrentUserId,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict:
    """Update the session's system prompt.

    Updates only this session's prompt — does not affect global config or other sessions.

    Args:
        session_id: Session ID
        body: Dict with system_prompt field
        user_id: Authenticated user ID
        session_manager: Injected session manager

    Returns:
        Status and updated system_prompt

    Raises:
        404: Session not found
    """
    managed = session_manager.get_session_or_none(session_id)
    if not managed or managed.user_id != user_id:
        raise HTTPException(status_code=404, detail="Session not found")

    new_prompt = body.get("system_prompt", "")

    # Update session-local prompt only
    managed.session.config.system_prompt = new_prompt
    managed.session_prompt = new_prompt

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
        user_id: Authenticated user ID
        session_manager: Injected session manager

    Returns:
        Dict with:
            - role: Optional dict with name, description, similarity
            - skills: List of dicts with name, description, similarity
            - role_source: "skill" or "query" indicating how role was selected
    """
    managed = session_manager.get_session_or_none(session_id)
    if not managed or managed.user_id != user_id:
        raise HTTPException(status_code=404, detail="Session not found")

    query = body.get("query", "")
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query is required")

    session = managed.session
    context = session.get_dynamic_context(query)

    return {
        "agent": context.get("agent"),
        "skills": context.get("skills"),
        "agent_source": context.get("agent_source"),
    }


@router.post("/{session_id}/share", response_model=ShareSessionResponse)
async def share_session(
    session_id: str,
    body: ShareSessionRequest,
    user_id: CurrentUserId,
    request: Request,
    session_manager: SessionManager = Depends(get_session_manager),
) -> ShareSessionResponse:
    """Share a session with another user by email.

    Adds the email to the session ACL and sends an invite email with a deep link.
    Only the session owner can share.
    """
    managed = session_manager.get_session_or_none(session_id)
    if not managed:
        raise HTTPException(status_code=404, detail="Session not found")
    if managed.user_id != user_id:
        raise HTTPException(status_code=403, detail="Only the session owner can share")

    email = body.email.strip().lower()

    # Store email in shared_with ACL
    if managed.session.datastore:
        managed.session.datastore.add_shared_user(email)

    # Build share URL
    config = request.app.state.config
    server_config = request.app.state.server_config
    base_url = getattr(server_config, 'base_url', '') or ''
    if not base_url:
        # Derive from request
        base_url = str(request.base_url).rstrip('/')
    share_url = f"{base_url}/s/{session_id}"

    # Send invite email if email is configured
    if hasattr(config, 'email') and config.email:
        from constat.email import EmailSender, markdown_to_html
        sender = EmailSender(config.email)
        summary = managed.session.datastore.get_session_meta("summary") if managed.session.datastore else None
        subject = f"A Constat session has been shared with you"
        body_md = f"""# Session Shared With You

A user has shared a Constat session with you.

{f"**Summary:** {summary}" if summary else ""}

[Open Session]({share_url})

Click the link above to join the session. You may need to sign in first.
"""
        html_body = markdown_to_html(body_md)
        sender.send(to=email, subject=subject, body=html_body, html=True)

    return ShareSessionResponse(status="shared", share_url=share_url)


@router.get("/{session_id}/shares")
async def get_shares(
    session_id: str,
    user_id: CurrentUserId,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict:
    """Get the list of users a session is shared with. Owner only."""
    managed = session_manager.get_session_or_none(session_id)
    if not managed:
        raise HTTPException(status_code=404, detail="Session not found")
    if managed.user_id != user_id:
        raise HTTPException(status_code=403, detail="Only the session owner can view shares")

    shared_with = []
    if managed.session.datastore:
        shared_with = managed.session.datastore.get_shared_users()

    return {"shared_with": shared_with}


@router.delete("/{session_id}/share/{shared_user_id}")
async def remove_share(
    session_id: str,
    shared_user_id: str,
    user_id: CurrentUserId,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict:
    """Remove a user from the session's shared list. Owner only."""
    managed = session_manager.get_session_or_none(session_id)
    if not managed:
        raise HTTPException(status_code=404, detail="Session not found")
    if managed.user_id != user_id:
        raise HTTPException(status_code=403, detail="Only the session owner can manage shares")

    if managed.session.datastore:
        managed.session.datastore.remove_shared_user(shared_user_id)

    return {"status": "removed", "user_id": shared_user_id}


@router.post("/{session_id}/public", response_model=TogglePublicResponse)
async def toggle_public(
    session_id: str,
    body: TogglePublicRequest,
    user_id: CurrentUserId,
    request: Request,
    session_manager: SessionManager = Depends(get_session_manager),
) -> TogglePublicResponse:
    """Toggle public sharing for a session. Owner only.

    When public is True, anyone with the link can view the session read-only.
    Session IDs are UUIDs (unguessable) so no additional token is needed.
    """
    managed = session_manager.get_session_or_none(session_id)
    if not managed:
        raise HTTPException(status_code=404, detail="Session not found")
    if managed.user_id != user_id:
        raise HTTPException(status_code=403, detail="Only the session owner can toggle public sharing")

    if managed.session.datastore:
        managed.session.datastore.set_public(body.public)

    # Build share URL
    server_config = request.app.state.server_config
    base_url = getattr(server_config, 'base_url', '') or ''
    if not base_url:
        base_url = str(request.base_url).rstrip('/')
    share_url = f"{base_url}/s/{session_id}"

    return TogglePublicResponse(
        status="updated",
        public=body.public,
        share_url=share_url,
    )
