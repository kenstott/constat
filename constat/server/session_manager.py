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
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Optional

from constat.core.config import Config
from constat.api.impl import ConstatAPIImpl
from constat.server.config import ServerConfig
from constat.server.models import SessionStatus
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

    # Active project filenames (e.g., ['sales-analytics.yaml', 'hr-reporting.yaml'])
    active_projects: list[str] = field(default_factory=list)

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
        """Get the session ID used for history storage."""
        return self.session.session_id if self.session else None

    def touch(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.now(timezone.utc)

    def is_expired(self, timeout_minutes: int) -> bool:
        """Check if session has exceeded timeout."""
        expiry = self.last_activity + timedelta(minutes=timeout_minutes)
        return datetime.now(timezone.utc) > expiry


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

    def create_session(self, user_id: str = "default") -> str:
        """Create a new Session instance.

        Args:
            user_id: User ID for session ownership

        Returns:
            Session ID for the new session

        Raises:
            RuntimeError: If max concurrent sessions limit is reached
        """
        with self._lock:
            # Check session limit
            if len(self._sessions) >= self._server_config.max_concurrent_sessions:
                raise RuntimeError(
                    f"Maximum concurrent sessions ({self._server_config.max_concurrent_sessions}) reached"
                )

            session_id = str(uuid.uuid4())
            logger.debug(f"Creating session {session_id} for user {user_id}")

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
                session_config=session_config,
                user_id=user_id,
                data_dir=self._server_config.data_dir,
            )
            # Set the server UUID for reverse lookup from history
            session.server_session_id = session_id

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
            )

            self._sessions[session_id] = managed

            # Run NER for session's visible documents (base + loaded projects)
            # This creates chunk-entity links scoped to this session
            self._run_entity_extraction(session_id, session)

            logger.info(f"Created session {session_id} for user {user_id}")

            return session_id

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

        # Get session's entity catalog
        schema_entities = list(session.schema_manager.get_entity_names())
        api_entities = list(session._get_api_entity_names())
        logger.info(f"Session {session_id}: running NER with {len(schema_entities)} schema entities, {len(api_entities)} API entities")
        logger.debug(f"Session {session_id}: schema_entities={schema_entities[:20]}")

        # Get active project IDs (empty on fresh session)
        project_ids = []
        if hasattr(self, '_sessions') and session_id in self._sessions:
            managed = self._sessions[session_id]
            project_ids = managed.active_projects or []

        # Run entity extraction
        try:
            link_count = session.doc_tools.extract_entities_for_session(
                session_id=session_id,
                project_ids=project_ids,
                schema_entities=schema_entities,
                api_entities=api_entities,
            )
            if link_count and link_count > 0:
                logger.debug(f"Session {session_id}: created {link_count} entity links")
        except Exception as e:
            logger.warning(f"Session {session_id}: entity extraction failed: {e}")

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

