# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Session manager for the Constat API server.

Manages server-side Session instances, tracking lifecycle, timeout, and cleanup.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Callable, Optional, Any

from constat.api.impl import ConstatAPIImpl
from constat.core.config import Config
from constat.core.tiered_config import ResolvedConfig, TieredConfigLoader
from constat.server.config import ServerConfig
from constat.server.models import EventType, SessionStatus
from constat.session import Session, SessionConfig
from constat.storage.facts import FactStore
from constat.storage.learnings import LearningStore

logger = logging.getLogger(__name__)


@dataclass
class ManagedSession:
    """A server-managed Session with metadata."""

    session_id: str
    session: Session
    api: ConstatAPIImpl  # Clean API wrapper over session
    user_id: str
    created_at: datetime
    last_activity: datetime
    status: SessionStatus = SessionStatus.IDLE
    current_query: Optional[str] = None
    execution_id: Optional[str] = None

    # History session ID (timestamp-based, e.g., '2026-02-05_160420_697412')
    # This is different from session_id which is the server/client UUID
    _history_session_id: Optional[str] = None

    # Active domain filenames (e.g., ['sales-analytics.yaml', 'hr-reporting.yaml'])
    active_domains: list[str] = field(default_factory=list)

    # Resolved tiered config (rebuilt on domain/source changes)
    resolved_config: Optional[ResolvedConfig] = None

    # Dynamic resources (databases, APIs, file refs) added during the session
    _dynamic_dbs: list[dict[str, Any]] = field(default_factory=list)
    _dynamic_apis: list[dict[str, Any]] = field(default_factory=list)
    _file_refs: list[dict[str, Any]] = field(default_factory=list)

    # Event queue for WebSocket bridging (sync Session events -> async WebSocket)
    event_queue: asyncio.Queue = field(default_factory=asyncio.Queue)

    # Approval event for blocking on plan approval
    approval_event: Optional[asyncio.Event] = None
    approval_response: Optional[dict] = None

    # Clarification event for blocking on user clarification
    clarification_event: Optional[asyncio.Event] = None
    clarification_response: Optional[dict] = None

    @property
    def history_session_id(self) -> Optional[str]:
        """Get the session ID used for history storage (timestamp-based)."""
        return self._history_session_id

    def touch(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.now(timezone.utc)

    def is_expired(self, timeout_minutes: int) -> bool:
        """Check if session has exceeded timeout."""
        expiry = self.last_activity + timedelta(minutes=timeout_minutes)
        return datetime.now(timezone.utc) > expiry

    def has_database(self, db_name: str) -> bool:
        """Check if a database exists in any source (config, domain, dynamic, schema_manager)."""
        # Config databases
        if db_name in self.session.config.databases:
            return True
        # Domain databases
        for domain_filename in self.active_domains:
            domain = self.session.config.load_domain(domain_filename)
            if domain and db_name in domain.databases:
                return True
        # Dynamic databases
        if any(db["name"] == db_name for db in self._dynamic_dbs):
            return True
        # Schema manager (covers all loaded databases)
        sm = self.session.schema_manager
        if sm and (db_name in sm.connections or db_name in sm.file_connections
                   or db_name in getattr(sm, 'nosql_connections', {})):
            return True
        return False

    def get_database_connection(self, db_name: str) -> Any:
        """Get a usable connection for a database. Returns None if unavailable."""
        sm = self.session.schema_manager
        if sm:
            try:
                return sm.get_connection(db_name)
            except KeyError:
                pass
        return None

    def get_all_database_names(self) -> set[str]:
        """Return union of all database names from every source."""
        names: set[str] = set()
        names.update(self.session.config.databases.keys())
        for domain_filename in self.active_domains:
            domain = self.session.config.load_domain(domain_filename)
            if domain:
                names.update(domain.databases.keys())
        names.update(db["name"] for db in self._dynamic_dbs)
        sm = self.session.schema_manager
        if sm:
            names.update(sm.connections.keys())
            names.update(sm.file_connections.keys())
            names.update(getattr(sm, 'nosql_connections', {}).keys())
        return names

    def save_resources(self) -> None:
        """Save dynamic resources (dbs, file_refs, domains) to disk."""
        from constat.storage.history import SessionHistory

        history_id = self.history_session_id
        if not history_id:
            return

        history = SessionHistory(user_id=self.user_id)

        # Gather resources
        resources = {
            "dynamic_dbs": self._dynamic_dbs,
            "dynamic_apis": self._dynamic_apis,
            "file_refs": self._file_refs,
            "active_domains": self.active_domains or [],
        }

        # Save to state file
        state = history.load_state(history_id) or {}
        state["resources"] = resources
        history.save_state(history_id, state)
        logger.debug(f"Saved session resources: {len(resources['dynamic_dbs'])} dbs, {len(resources['dynamic_apis'])} apis, {len(resources['file_refs'])} refs")

    def restore_resources(self) -> None:
        """Restore dynamic resources from disk."""
        from constat.storage.history import SessionHistory

        history_id = self.history_session_id
        if not history_id:
            return

        history = SessionHistory(user_id=self.user_id)
        state = history.load_state(history_id)
        if not state or "resources" not in state:
            return

        resources = state["resources"]

        # Restore to managed session
        self._dynamic_dbs = resources.get("dynamic_dbs", [])
        self._dynamic_apis = resources.get("dynamic_apis", [])
        self._file_refs = resources.get("file_refs", [])
        self.active_domains = resources.get("active_domains", [])

        logger.info(f"Restored session resources: {len(self._dynamic_dbs)} dbs, {len(self._dynamic_apis)} apis, {len(self._file_refs)} refs, {len(self.active_domains)} domains")

        # Re-add databases to schema_manager
        if self._dynamic_dbs and self.session.schema_manager:
            from constat.core.config import DatabaseConfig
            for db in self._dynamic_dbs:
                try:
                    db_config = DatabaseConfig(
                        type=db.get("type", "csv"),
                        uri=db.get("uri", ""),
                        description=db.get("description", ""),
                    )
                    self.session.schema_manager.add_database_dynamic(db["name"], db_config)
                    logger.debug(f"Restored database: {db['name']}")
                except Exception as e:
                    logger.warning(f"Failed to restore database {db['name']}: {e}")

        # Re-add APIs to session resources
        if self._dynamic_apis and self.session.resources:
            for api in self._dynamic_apis:
                try:
                    self.session.resources.add_api(
                        name=api["name"],
                        description=api.get("description", ""),
                        api_type=api.get("type", "rest"),
                        source="session",
                    )
                    logger.debug(f"Restored API: {api['name']}")
                except Exception as e:
                    logger.warning(f"Failed to restore API {api['name']}: {e}")


class SessionManager:
    """Manages server-side Session instances.

    Thread-safe management of Session lifecycle:
    - Creating new sessions
    - Retrieving existing sessions
    - Cleanup of expired sessions
    - Enforcing max concurrent session limits

    Note: Sessions are stored in-memory. For production deployments,
    consider adding Redis-based session storage.
    """

    def __init__(self, config: Config, server_config: ServerConfig):
        """Initialize the session manager.

        Args:
            config: Main Constat configuration
            server_config: Server-specific configuration
        """
        self._config = config
        self._server_config = server_config
        self._sessions: dict[str, ManagedSession] = {}
        self._lock = Lock()
        self._cleanup_task: Optional[asyncio.Task] = None

    def create_session(self, session_id: str, user_id: str = "default") -> str:
        """Create or restore a Session instance.

        If session_id exists in history (on disk), restores it with all artifacts,
        tables, and context. Otherwise, creates a new session.

        Args:
            session_id: Session ID provided by client
            user_id: User ID for session ownership

        Returns:
            Session ID for the new/restored session

        Raises:
            RuntimeError: If max concurrent sessions limit is reached
        """
        from constat.storage.history import SessionHistory

        with self._lock:
            # Check session limit
            if len(self._sessions) >= self._server_config.max_concurrent_sessions:
                raise RuntimeError(
                    f"Maximum concurrent sessions ({self._server_config.max_concurrent_sessions}) reached"
                )

            # Check if this session exists in history (can be restored)
            history = SessionHistory(user_id=user_id)
            history_session_id = history.find_session_by_server_id(session_id)
            historical_session = history.get_session(history_session_id) if history_session_id else None
            is_restore = historical_session is not None

            if is_restore:
                logger.info(f"Restoring session {session_id} (history_id={history_session_id})")
            else:
                logger.debug(f"Creating new session {session_id}")

            # Create Session config with server-appropriate settings
            session_config = SessionConfig(
                verbose=False,
                require_approval=self._server_config.require_plan_approval,
                auto_approve=not self._server_config.require_plan_approval,
                ask_clarifications=True,  # Clarifications via WebSocket dialog
                skip_clarification=False,
            )

            # Create the underlying Session
            session = Session(
                config=self._config,
                session_id=session_id,
                session_config=session_config,
                user_id=user_id,
                data_dir=self._server_config.data_dir,
            )

            # If historical session exists, restore it (loads datastore, scratchpad, etc.)
            if is_restore and history_session_id:
                if session.resume(history_session_id):
                    logger.info(f"Successfully restored session {session_id} (history_id={history_session_id}) with datastore")
                else:
                    logger.warning(f"Failed to restore session {session_id} (history_id={history_session_id}), continuing as new")
            else:
                # Create session directory upfront (not waiting for first query)
                history_session_id = history.create_session(
                    config_dict=self._config.model_dump() if hasattr(self._config, 'model_dump') else {},
                    databases=[],
                    apis=[],
                    documents=[],
                    server_session_id=session_id,
                )
                logger.info(f"Created session directory: {history_session_id} (server_id={session_id})")

            # Create stores for API
            fact_store = FactStore(user_id=user_id)
            learning_store = LearningStore(user_id=user_id)

            # Load persisted facts for this user
            fact_store.load_into_session(session)

            # Create API wrapper
            api = ConstatAPIImpl(
                session=session,
                fact_store=fact_store,
                learning_store=learning_store,
            )

            now = datetime.now(timezone.utc)
            managed = ManagedSession(
                session_id=session_id,
                session=session,
                api=api,
                user_id=user_id,
                created_at=now,
                last_activity=now,
                _history_session_id=history_session_id,
            )

            self._sessions[session_id] = managed

            # Restore dynamic resources (dbs, file_refs, domains) if this is a restore
            if is_restore:
                managed.restore_resources()

            logger.info(f"Created session {session_id} for user {user_id}")

        # Build initial resolved config (tier 1 + user tier; domains added later)
        self.resolve_config(session_id)

        # NER is NOT started here — the route handler calls _run_entity_extraction
        # after domain loading completes, so schema entities are available for
        # pattern matching. Starting async NER here caused a race condition where
        # set_schema_entities (during domain loading) cleared all chunk_entity
        # links and the background thread's extraction was lost.

        return session_id

    @staticmethod
    def _build_session_overrides(managed: ManagedSession) -> dict:
        """Build session-tier overrides from dynamic resources."""
        overrides: dict = {}
        if managed._dynamic_dbs:
            overrides["databases"] = {
                db["name"]: {
                    "type": db.get("type", "sql"),
                    "uri": db.get("uri", ""),
                    "description": db.get("description", ""),
                }
                for db in managed._dynamic_dbs
            }
        if managed._dynamic_apis:
            overrides["apis"] = {
                api["name"]: {
                    "type": api.get("type", "rest"),
                    "url": api.get("base_url", ""),
                    "description": api.get("description", ""),
                }
                for api in managed._dynamic_apis
            }
        if managed._file_refs:
            overrides["documents"] = {
                ref["name"]: {
                    "type": "file",
                    "path": ref.get("uri", ""),
                    "description": ref.get("description", ""),
                }
                for ref in managed._file_refs
            }
        return overrides

    def resolve_config(self, session_id: str) -> Optional[ResolvedConfig]:
        """Build or rebuild the tiered ResolvedConfig for a session.

        Merges system, system-domain, user, user-domain, and session tiers.
        Stores result on both ManagedSession and Session.

        Args:
            session_id: Session ID to resolve config for

        Returns:
            ResolvedConfig or None if session not found
        """
        with self._lock:
            if session_id not in self._sessions:
                logger.warning(f"Cannot resolve config: session {session_id} not found")
                return None
            managed = self._sessions[session_id]

        session_overrides = self._build_session_overrides(managed)
        loader = TieredConfigLoader(
            config=self._config,
            user_id=managed.user_id,
            base_dir=self._server_config.data_dir,
            domain_names=managed.active_domains or [],
            session_overrides=session_overrides,
        )
        resolved = loader.resolve()
        managed.resolved_config = resolved
        managed.session.resolved_config = resolved
        logger.info(f"Resolved tiered config for session {session_id}: "
                     f"{len(resolved.sources.databases)} dbs, "
                     f"{len(resolved.sources.apis)} apis, "
                     f"{len(resolved.glossary)} glossary, "
                     f"domains={resolved.active_domains}")
        return resolved

    def _run_entity_extraction(self, session_id: str, session: Session) -> None:
        """Run NER for session's visible documents.

        Creates chunk-entity links scoped to this session's entity catalog.

        Args:
            session_id: Server session ID for storing links
            session: Session with doc_tools and schema/api entity info
        """
        if not session.doc_tools:
            logger.debug(f"Session {session_id}: no doc_tools, skipping entity extraction")
            return

        # Get domain IDs (config + active) and session database names
        domain_ids = list(session.config.domains.keys()) if session.config.domains else []
        session_db_names = []
        if hasattr(self, '_sessions') and session_id in self._sessions:
            managed = self._sessions[session_id]
            # Merge active domains (may include dynamically activated ones)
            for p in (managed.active_domains or []):
                if p not in domain_ids:
                    domain_ids.append(p)
            # Get names of dynamically added databases (include their columns in entities)
            session_db_names = [db["name"] for db in managed._dynamic_dbs]

        # Get session's entity catalog
        # Include columns only for session-added databases (their columns are meaningful)
        # Config databases often have generic column names ("id", "name", "date")
        schema_entities = list(session.schema_manager.get_entity_names(
            include_columns_for_dbs=session_db_names
        ))
        api_entities = list(session._get_api_entity_names())

        # Collect glossary + relationship terms for NER business_terms
        from constat.catalog.glossary_builder import get_glossary_terms_for_ner, get_relationship_terms_for_ner
        business_terms: list[str] = []
        if session.config.glossary:
            business_terms.extend(get_glossary_terms_for_ner(session.config.glossary))
        if session.config.relationships:
            business_terms.extend(get_relationship_terms_for_ner(session.config.relationships))

        logger.info(f"Session {session_id}: running NER with {len(schema_entities)} schema, {len(api_entities)} API, {len(business_terms)} business entities")

        # Fingerprint caching — skip NER if scope unchanged
        from constat.discovery.ner_fingerprint import compute_ner_fingerprint, should_skip_ner, update_ner_fingerprint
        chunk_ids = []
        if hasattr(session.doc_tools, '_vector_store') and session.doc_tools._vector_store:
            try:
                chunk_ids = session.doc_tools._vector_store.get_all_chunk_ids(session_id=session_id)
            except Exception:
                pass
        fingerprint = compute_ner_fingerprint(chunk_ids, schema_entities, api_entities, business_terms)
        if should_skip_ner(session_id, fingerprint):
            return

        # Run entity extraction
        try:
            # Clear existing entity links before re-extraction (handles db add/remove)
            if hasattr(session.doc_tools, '_vector_store') and session.doc_tools._vector_store:
                logger.info(f"Session {session_id}: clearing existing entities and links")
                session.doc_tools._vector_store.clear_session_entities(session_id)
                logger.info(f"Session {session_id}: cleared existing entity links")
            else:
                logger.warning(f"Session {session_id}: no vector_store to clear entities from")

            logger.info(f"Session {session_id}: running extract_entities_for_session with domain_ids={domain_ids}, {len(schema_entities)} schema entities")
            if schema_entities:
                logger.debug(f"Session {session_id}: sample schema_entities: {schema_entities[:10]}")
            link_count = session.doc_tools.extract_entities_for_session(
                session_id=session_id,
                domain_ids=domain_ids,
                schema_entities=schema_entities,
                api_entities=api_entities,
                business_terms=business_terms or None,
            )
            if link_count and link_count > 0:
                logger.info(f"Session {session_id}: created {link_count} entity links")
            else:
                logger.warning(f"Session {session_id}: NO entity links created (link_count={link_count})")
            # Cache fingerprint on successful extraction
            update_ner_fingerprint(session_id, fingerprint)
        except Exception as e:
            logger.exception(f"Session {session_id}: entity extraction failed: {e}")

    def refresh_entities(self, session_id: str) -> None:
        """Refresh entity extraction for a session (synchronous).

        Call this after dynamically adding or removing databases to update
        the session's entity catalog.

        Args:
            session_id: Session ID to refresh
        """
        logger.info(f"refresh_entities({session_id}): starting")
        with self._lock:
            if session_id not in self._sessions:
                logger.warning(f"Cannot refresh entities: session {session_id} not found")
                return
            managed = self._sessions[session_id]
            logger.info(f"refresh_entities({session_id}): calling _run_entity_extraction")
            self._run_entity_extraction(session_id, managed.session)
            logger.info(f"refresh_entities({session_id}): complete")

    def refresh_entities_async(self, session_id: str) -> None:
        """Refresh entity extraction in a background thread.

        Non-blocking — returns immediately and pushes ENTITY_REBUILD_START
        and ENTITY_REBUILD_COMPLETE events via the session's WebSocket queue.

        Args:
            session_id: Session ID to refresh
        """
        with self._lock:
            if session_id not in self._sessions:
                logger.warning(f"Cannot refresh entities async: session {session_id} not found")
                return
            managed = self._sessions[session_id]

        def _run():
            import time
            t0 = time.time()
            try:
                self._push_event(managed, EventType.ENTITY_REBUILD_START, {
                    "session_id": session_id,
                })
                self._run_entity_extraction(session_id, managed.session)
                duration_ms = int((time.time() - t0) * 1000)
                self._push_event(managed, EventType.ENTITY_REBUILD_COMPLETE, {
                    "session_id": session_id,
                    "duration_ms": duration_ms,
                })
                logger.info(f"refresh_entities_async({session_id}): complete in {duration_ms}ms")
            except Exception as e:
                logger.exception(f"refresh_entities_async({session_id}): failed: {e}")

        import threading
        thread = threading.Thread(target=_run, name=f"entity-rebuild-{session_id}", daemon=True)
        thread.start()

    def _run_glossary_generation(self, session_id: str, managed: "ManagedSession") -> None:
        """Run LLM glossary generation after entity extraction.

        Args:
            session_id: Server session ID
            managed: ManagedSession instance
        """
        import time

        session = managed.session
        if not session.doc_tools or not hasattr(session.doc_tools, '_vector_store'):
            return

        vector_store = session.doc_tools._vector_store
        if not vector_store:
            return

        # Need a router for LLM calls
        if not hasattr(session, 'router') or not session.router:
            logger.debug(f"Session {session_id}: no router available, skipping glossary generation")
            return

        try:
            t0 = time.time()
            self._push_event(managed, EventType.GLOSSARY_REBUILD_START, {
                "session_id": session_id,
            })

            from constat.discovery.glossary_generator import generate_glossary

            def on_batch(batch_terms):
                term_dicts = []
                for t in batch_terms:
                    term_dicts.append({
                        "name": t.name,
                        "display_name": t.display_name,
                        "definition": t.definition,
                        "domain": t.domain,
                        "parent_id": t.parent_id,
                        "aliases": t.aliases or [],
                        "semantic_type": t.semantic_type,
                        "status": t.status,
                        "provenance": t.provenance,
                        "glossary_status": "defined",
                        "connected_resources": [],
                    })
                self._push_event(managed, EventType.GLOSSARY_TERMS_ADDED, {"terms": term_dicts})

            def on_progress(stage: str, pct: int):
                self._push_event(managed, EventType.GLOSSARY_GENERATION_PROGRESS, {
                    "stage": stage, "percent": pct,
                })

            active_domains = getattr(managed, "active_domains", []) or []
            terms = generate_glossary(
                session_id=session_id,
                vector_store=vector_store,
                router=session.router,
                active_domains=active_domains,
                on_batch_complete=on_batch,
                on_progress=on_progress,
            )

            # Reconcile alias entities (rename "platinum" → "platinum tier")
            from constat.discovery.glossary_generator import reconcile_alias_entities
            reconcile_alias_entities(session_id, vector_store)

            # Embed generated terms as glossary chunks
            if terms:
                on_progress("Embedding terms", 78)
                active_domains = getattr(managed, "active_domains", []) or []
                self._embed_glossary_terms(terms, session_id, vector_store, session, active_domains)

            # Relationship extraction (4 phases: FK, SVO, LLM refinement, glossary inference)
            self._run_svo_extraction(session_id, managed, vector_store, on_progress=on_progress)

            on_progress("Complete", 100)
            duration_ms = int((time.time() - t0) * 1000)
            self._push_event(managed, EventType.GLOSSARY_REBUILD_COMPLETE, {
                "session_id": session_id,
                "terms_count": len(terms),
                "duration_ms": duration_ms,
            })
            logger.info(f"Glossary generation for {session_id}: {len(terms)} terms in {duration_ms}ms")
        except Exception as e:
            logger.exception(f"Glossary generation for {session_id} failed: {e}")

    @staticmethod
    def _embed_glossary_terms(
        terms: list,
        session_id: str,
        vector_store,
        session,
        domain_ids: list[str] | None = None,
    ) -> None:
        """Embed glossary terms as searchable chunks."""
        from constat.catalog.glossary_builder import glossary_term_to_chunk
        from constat.discovery.glossary_generator import resolve_physical_resources

        chunks = []
        for term in terms:
            resources = resolve_physical_resources(term.name, session_id, vector_store, domain_ids=domain_ids)
            entity_sources = []
            for r in resources:
                for s in r.get("sources", []):
                    entity_sources.append(f"{s.get('document_name', '')} ({s.get('source', '')})")
            chunk = glossary_term_to_chunk(term, entity_sources)
            chunks.append(chunk)

        if chunks and session.doc_tools:
            try:
                # Use the doc_tools model to encode and vector store to add
                doc_tools = session.doc_tools
                if hasattr(doc_tools, '_model') and doc_tools._model:
                    texts = [c.content for c in chunks]
                    embeddings = doc_tools._model.encode(texts, normalize_embeddings=True)
                    vector_store.add_chunks(chunks, embeddings, source="document")
                    logger.info(f"Embedded {len(chunks)} glossary term chunks")
            except Exception as e:
                logger.warning(f"Failed to embed glossary chunks: {e}")

    def _run_svo_extraction(
        self,
        session_id: str,
        managed: "ManagedSession",
        vector_store,
        on_progress: Callable[[str, int], None] | None = None,
    ) -> None:
        """Run four-phase relationship extraction after glossary generation.

        Phase 0: FK relationships (from schema foreign keys)
        Phase 1: SpaCy SVO (optional — graceful skip if no spaCy)
        Phase 2: LLM refinement (if router available)
        Phase 3: Glossary-informed LLM inference (cross-cutting relationships)
        """
        session = managed.session

        def on_batch(batch_rels):
            rel_dicts = [
                {
                    "subject_name": r.subject_name,
                    "verb": r.verb,
                    "object_name": r.object_name,
                    "sentence": r.sentence,
                    "confidence": r.confidence,
                    "verb_category": r.verb_category,
                }
                for r in batch_rels
            ]
            self._push_event(managed, EventType.RELATIONSHIPS_EXTRACTED, {
                "relationships": rel_dicts,
            })

        # Phase 0: FK relationships
        if on_progress:
            on_progress("Extracting FK relationships", 85)
        try:
            from constat.discovery.relationship_extractor import store_fk_relationships
            glossary_terms = vector_store.list_glossary_terms(session_id)
            if glossary_terms and session.schema_manager:
                fk_rels = store_fk_relationships(
                    session_id=session_id,
                    glossary_terms=glossary_terms,
                    schema_manager=session.schema_manager,
                    vector_store=vector_store,
                    on_batch=on_batch,
                )
                logger.info(f"Phase 0 FK relationships for {session_id}: {len(fk_rels)} relationships")
        except Exception as e:
            logger.exception(f"Phase 0 FK relationships for {session_id} failed: {e}")

        # Phase 1: spaCy SVO (optional)
        if on_progress:
            on_progress("Extracting text relationships", 88)
        svo_rels = []
        try:
            from constat.discovery.entity_extractor import get_nlp
            from constat.discovery.relationship_extractor import extract_svo_relationships
            try:
                nlp = get_nlp()
            except Exception as e:
                logger.debug(f"Session {session_id}: spaCy model not available ({e}), skipping Phase 1 SVO extraction")
                nlp = None

            if nlp:
                svo_rels = extract_svo_relationships(
                    session_id=session_id,
                    vector_store=vector_store,
                    nlp=nlp,
                    on_batch=on_batch,
                )
                logger.info(f"Phase 1 SVO extraction for {session_id}: {len(svo_rels)} relationships")
        except Exception as e:
            logger.exception(f"Phase 1 SVO extraction for {session_id} failed: {e}")

        # Phase 2: LLM refinement (if router available)
        if on_progress:
            on_progress("Refining relationships", 92)
        if hasattr(session, 'router') and session.router:
            try:
                from constat.discovery.relationship_extractor import (
                    get_co_occurring_pairs,
                    refine_relationships_with_llm,
                )
                co_pairs = get_co_occurring_pairs(session_id, vector_store)
                if co_pairs:
                    llm_rels = refine_relationships_with_llm(
                        session_id=session_id,
                        vector_store=vector_store,
                        router=session.router,
                        svo_candidates=svo_rels,
                        co_occurring_pairs=co_pairs,
                        on_batch=on_batch,
                    )
                    logger.info(f"Phase 2 LLM refinement for {session_id}: {len(llm_rels)} relationships")
            except Exception as e:
                logger.exception(f"Phase 2 LLM refinement for {session_id} failed: {e}")
        else:
            logger.debug(f"Session {session_id}: no router available, skipping Phase 2 LLM refinement")

        # Phase 3: Glossary-informed LLM inference
        if on_progress:
            on_progress("Inferring relationships", 96)
        if hasattr(session, 'router') and session.router:
            try:
                from constat.discovery.relationship_extractor import infer_glossary_relationships
                glossary_rels = infer_glossary_relationships(
                    session_id=session_id,
                    vector_store=vector_store,
                    router=session.router,
                    on_batch=on_batch,
                )
                logger.info(f"Phase 3 glossary inference for {session_id}: {len(glossary_rels)} relationships")
            except Exception as e:
                logger.exception(f"Phase 3 glossary inference for {session_id} failed: {e}")

        # Final: deduplicate — keep only best relationship per entity pair
        try:
            from constat.discovery.relationship_extractor import deduplicate_relationships
            removed = deduplicate_relationships(session_id, vector_store)
            if removed:
                logger.info(f"Relationship dedup for {session_id}: removed {removed} duplicates")
        except Exception as e:
            logger.exception(f"Relationship dedup for {session_id} failed: {e}")

    @staticmethod
    def _push_event(managed: "ManagedSession", event_type: EventType, data: dict) -> None:
        """Push an event to a managed session's WebSocket queue."""
        from constat.server.models import StepEventWS
        try:
            ws_event = StepEventWS(
                event_type=event_type,
                session_id=managed.session_id,
                step_number=0,
                timestamp=datetime.now(timezone.utc),
                data=data,
            )
            managed.event_queue.put_nowait(ws_event.model_dump(mode="json"))
        except asyncio.QueueFull:
            logger.warning(f"Event queue full for session {managed.session_id}, dropping {event_type}")

    def get_session(self, session_id: str) -> ManagedSession:
        """Get a managed session by ID.

        Args:
            session_id: Session ID to retrieve

        Returns:
            ManagedSession instance

        Raises:
            KeyError: If session not found
        """
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")
            managed = self._sessions[session_id]
            managed.touch()
            return managed

    def get_session_or_none(self, session_id: str) -> Optional[ManagedSession]:
        """Get a managed session by ID, returning None if not found.

        Args:
            session_id: Session ID to retrieve

        Returns:
            ManagedSession instance or None
        """
        try:
            return self.get_session(session_id)
        except KeyError:
            return None

    def list_sessions(self, user_id: Optional[str] = None) -> list[ManagedSession]:
        """List all managed sessions.

        Args:
            user_id: Optional filter by user ID

        Returns:
            List of ManagedSession instances
        """
        with self._lock:
            sessions = list(self._sessions.values())
            if user_id:
                sessions = [s for s in sessions if s.user_id == user_id]
            return sessions

    def delete_session(self, session_id: str) -> bool:
        """Delete and cleanup a session.

        Args:
            session_id: Session ID to delete

        Returns:
            True if session was deleted, False if not found
        """
        with self._lock:
            if session_id not in self._sessions:
                return False

            managed = self._sessions.pop(session_id)

            # Cleanup the underlying session
            try:
                # Cancel any running execution
                if hasattr(managed.session, "_cancelled"):
                    managed.session._cancelled = True

                # Close datastore if exists
                if managed.session.datastore:
                    managed.session.datastore.close()
            except Exception as e:
                logger.warning(f"Error cleaning up session {session_id}: {e}")

            logger.info(f"Deleted session {session_id}")
            return True

    def cleanup_expired(self) -> int:
        """Remove sessions that have exceeded the timeout.

        Returns:
            Number of sessions cleaned up
        """
        timeout = self._server_config.session_timeout_minutes
        expired_ids = []

        with self._lock:
            for session_id, managed in self._sessions.items():
                if managed.is_expired(timeout):
                    expired_ids.append(session_id)

        # Delete expired sessions outside the lock
        for session_id in expired_ids:
            self.delete_session(session_id)

        if expired_ids:
            logger.info(f"Cleaned up {len(expired_ids)} expired sessions")

        return len(expired_ids)

    async def start_cleanup_task(self, interval_seconds: int = 60) -> None:
        """Start the periodic cleanup background task.

        Args:
            interval_seconds: Interval between cleanup runs
        """
        if self._cleanup_task is not None:
            return

        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(interval_seconds)
                    self.cleanup_expired()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in cleanup task: {e}")

        self._cleanup_task = asyncio.create_task(cleanup_loop())
        logger.info(f"Started session cleanup task (interval: {interval_seconds}s)")

    async def stop_cleanup_task(self) -> None:
        """Stop the periodic cleanup background task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
            logger.info("Stopped session cleanup task")

    def update_status(self, session_id: str, status: SessionStatus) -> None:
        """Update session status.

        Args:
            session_id: Session ID to update
            status: New status
        """
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id].status = status
                self._sessions[session_id].touch()

    def set_current_query(self, session_id: str, query: Optional[str]) -> None:
        """Set the current query being processed.

        Args:
            session_id: Session ID to update
            query: Query text or None to clear
        """
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id].current_query = query
                self._sessions[session_id].touch()

    def get_stats(self) -> dict:
        """Get session manager statistics.

        Returns:
            Dict with session counts and status breakdown
        """
        with self._lock:
            total = len(self._sessions)
            by_status = {}
            for managed in self._sessions.values():
                status = managed.status.value
                by_status[status] = by_status.get(status, 0) + 1

            return {
                "total_sessions": total,
                "max_sessions": self._server_config.max_concurrent_sessions,
                "by_status": by_status,
            }

