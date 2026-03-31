# Copyright (c) 2025 Kenneth Stott
# Canary: 156faf7c-8663-4a76-8058-5707f6cfff61
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""GraphQL resolvers for session CRUD, sharing, and domain management."""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Optional

import strawberry

from constat.server.graphql.session_context import GqlInfo as Info
from constat.server.graphql.types import (
    SessionListType,
    SessionStatusEnum,
    SessionType,
    ShareResponseType,
    TogglePublicResponseType,
)
from constat.server.models import SessionStatus
from constat.server.session_manager import ManagedSession

logger = logging.getLogger(__name__)


def _managed_to_session_type(managed: ManagedSession) -> SessionType:
    """Convert ManagedSession to SessionType."""
    tables_count = 0
    artifacts_count = 0
    summary = None

    if managed.session.datastore:
        try:
            tables_count = len(managed.session.datastore.list_tables())
        except (KeyError, ValueError, OSError):
            pass
        try:
            artifacts_count = len(managed.session.datastore.list_artifacts())
        except (KeyError, ValueError, OSError):
            pass
        try:
            summary = managed.session.datastore.get_session_meta("summary")
        except (KeyError, ValueError, OSError):
            pass

    if not summary and managed.session.history:
        try:
            history_session_id = managed.session.history.find_session_by_server_id(managed.session_id)
            if history_session_id:
                hist = managed.session.history.get_session(history_session_id)
                if hist:
                    summary = hist.summary
        except (KeyError, ValueError, OSError):
            pass

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

    return SessionType(
        session_id=managed.session_id,
        user_id=managed.user_id,
        status=SessionStatusEnum(managed.status.value if isinstance(managed.status, SessionStatus) else managed.status),
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


@strawberry.type
class Query:
    @strawberry.field
    async def sessions(self, info: Info) -> SessionListType:
        sm = info.context.session_manager
        user_id = info.context.user_id

        effective_user_id = user_id if user_id != "default" else "default"

        in_memory = sm.list_sessions(user_id=effective_user_id)

        # Include sessions shared with this user
        all_sessions = sm.list_sessions()
        for s in all_sessions:
            if s.user_id != effective_user_id and s.session.datastore:
                try:
                    if effective_user_id in s.session.datastore.get_shared_users():
                        in_memory.append(s)
                except (KeyError, ValueError, OSError):
                    pass

        in_memory_ids = {s.session_id for s in in_memory}
        responses = [_managed_to_session_type(s) for s in in_memory]

        # Historical sessions from disk
        try:
            from constat.storage.history import SessionHistory

            history = SessionHistory(user_id=effective_user_id)
            historical = history.list_sessions(limit=50)

            for hist in historical:
                if not hist.server_session_id or hist.server_session_id in in_memory_ids:
                    continue
                try:
                    created_at = datetime.fromisoformat(hist.created_at.replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    created_at = datetime.now(timezone.utc)

                responses.append(SessionType(
                    session_id=hist.server_session_id,
                    user_id=hist.user_id or effective_user_id,
                    status=SessionStatusEnum.IDLE,
                    created_at=created_at,
                    last_activity=created_at,
                    current_query=hist.summary,
                    summary=hist.summary,
                    tables_count=0,
                    artifacts_count=0,
                ))
        except Exception as e:
            logger.warning(f"Failed to load historical sessions: {e}")

        responses.sort(key=lambda s: s.last_activity, reverse=True)
        return SessionListType(sessions=responses, total=len(responses))

    @strawberry.field
    async def session(self, info: Info, session_id: str) -> Optional[SessionType]:
        sm = info.context.session_manager
        managed = sm.get_session_or_none(session_id)
        if not managed:
            return None
        return _managed_to_session_type(managed)

    @strawberry.field
    async def session_shares(self, info: Info, session_id: str) -> list[str]:
        sm = info.context.session_manager
        user_id = info.context.user_id
        managed = sm.get_session_or_none(session_id)
        if not managed:
            raise ValueError("Session not found")
        if managed.user_id != user_id:
            raise ValueError("Only the session owner can view shares")
        shared_with: list[str] = []
        if managed.session.datastore:
            shared_with = managed.session.datastore.get_shared_users()
        return shared_with

    @strawberry.field
    async def active_domains(self, info: Info, session_id: str) -> list[str]:
        sm = info.context.session_manager
        managed = sm.get_session_or_none(session_id)
        if not managed:
            raise ValueError("Session not found")
        return managed.active_domains


@strawberry.type
class Mutation:
    @strawberry.mutation
    async def create_session(
        self, info: Info, session_id: str, user_id: Optional[str] = None,
    ) -> SessionType:
        sm = info.context.session_manager
        auth_user_id = info.context.user_id
        effective_user_id = auth_user_id if auth_user_id != "default" else (user_id or "default")

        # Check for existing session (reconnect case)
        existing = sm.get_session_or_none(session_id)
        if existing:
            shared_with: list[str] = []
            if existing.session.datastore:
                try:
                    shared_with = existing.session.datastore.get_shared_users()
                except (KeyError, ValueError, OSError):
                    pass
            if existing.user_id != effective_user_id and effective_user_id not in shared_with:
                raise ValueError("Not authorized to access this session")
            from constat.server.routes.learnings import _ensure_user_domain_config
            _ensure_user_domain_config(effective_user_id, existing.session.config)
            existing.touch()
            return _managed_to_session_type(existing)

        # Create new session
        _session_id = sm.create_session(session_id=session_id, user_id=effective_user_id)

        def _init_session():
            from constat.server.models import EventType
            from constat.server.routes.sessions import (
                _apply_resolved_source_overrides,
                _index_user_documents,
                _load_domains_into_session,
            )
            from constat.server.user_preferences import get_selected_domains

            _managed = sm.get_session(_session_id)
            try:
                from constat.server.routes.learnings import _ensure_user_domain_config
                _ensure_user_domain_config(effective_user_id, _managed.session.config)

                _managed.session_prompt = _managed.session.config.system_prompt

                preferred_domains = get_selected_domains(effective_user_id)
                resolved = None
                if preferred_domains:
                    loaded, conflicts = _load_domains_into_session(_managed, preferred_domains)
                    if loaded:
                        resolved = sm.resolve_config(_session_id)
                        if resolved:
                            _apply_resolved_source_overrides(_managed, resolved, skip_documents=True)
                            if resolved.system_prompt:
                                _managed.session.config.system_prompt = resolved.system_prompt
                                _managed.session_prompt = resolved.system_prompt

                if 'user' not in _managed.active_domains:
                    _managed.active_domains.append('user')

                _managed._init_complete = True

                sm._push_event(
                    _managed, EventType.SESSION_READY,
                    {"session_id": _session_id, "active_domains": _managed.active_domains},
                )

                sm.refresh_entities_async(_session_id)

                if preferred_domains and resolved and resolved.sources.documents:
                    threading.Thread(
                        target=_index_user_documents,
                        args=(sm, _managed, resolved),
                        name=f"doc-index-{session_id[:8]}",
                        daemon=True,
                    ).start()
            except Exception as e:
                logger.exception(f"[create_session] init failed for {_session_id}: {e}")

        threading.Thread(target=_init_session, name=f"session-init-{session_id[:8]}", daemon=True).start()

        return SessionType(
            session_id=session_id,
            user_id=effective_user_id,
            status=SessionStatusEnum.IDLE,
            created_at=datetime.now(timezone.utc),
            last_activity=datetime.now(timezone.utc),
        )

    @strawberry.mutation
    async def delete_session(self, info: Info, session_id: str) -> bool:
        sm = info.context.session_manager
        if not sm.delete_session(session_id):
            raise ValueError(f"Session not found: {session_id}")
        return True

    @strawberry.mutation
    async def toggle_public_sharing(
        self, info: Info, session_id: str, public: bool,
    ) -> TogglePublicResponseType:
        sm = info.context.session_manager
        user_id = info.context.user_id
        managed = sm.get_session_or_none(session_id)
        if not managed:
            raise ValueError("Session not found")
        if managed.user_id != user_id:
            raise ValueError("Only the session owner can toggle public sharing")

        if managed.session.datastore:
            managed.session.datastore.set_public(public)

        server_config = info.context.server_config
        base_url = getattr(server_config, 'base_url', '') or ''
        share_url = f"{base_url}/s/{session_id}" if base_url else f"/s/{session_id}"

        return TogglePublicResponseType(
            status="updated",
            public=public,
            share_url=share_url,
        )

    @strawberry.mutation
    async def share_session(
        self, info: Info, session_id: str, email: str,
    ) -> ShareResponseType:
        sm = info.context.session_manager
        user_id = info.context.user_id
        managed = sm.get_session_or_none(session_id)
        if not managed:
            raise ValueError("Session not found")
        if managed.user_id != user_id:
            raise ValueError("Only the session owner can share")

        email = email.strip().lower()

        if managed.session.datastore:
            managed.session.datastore.add_shared_user(email)

        server_config = info.context.server_config
        base_url = getattr(server_config, 'base_url', '') or ''
        share_url = f"{base_url}/s/{session_id}" if base_url else f"/s/{session_id}"

        return ShareResponseType(status="shared", share_url=share_url)

    @strawberry.mutation
    async def remove_share(
        self, info: Info, session_id: str, shared_user_id: str,
    ) -> bool:
        sm = info.context.session_manager
        user_id = info.context.user_id
        managed = sm.get_session_or_none(session_id)
        if not managed:
            raise ValueError("Session not found")
        if managed.user_id != user_id:
            raise ValueError("Only the session owner can manage shares")

        if managed.session.datastore:
            managed.session.datastore.remove_shared_user(shared_user_id)

        return True

    @strawberry.mutation
    async def reset_context(self, info: Info, session_id: str) -> bool:
        sm = info.context.session_manager
        managed = sm.get_session_or_none(session_id)
        if not managed:
            raise ValueError("Session not found")

        if managed.api:
            managed.api.reset_context()

        from constat.storage.history import SessionHistory
        history = SessionHistory(user_id=managed.user_id or "default")
        history.save_messages_by_server_id(session_id, [])

        return True

    @strawberry.mutation
    async def set_active_domains(
        self, info: Info, session_id: str, domains: list[str],
    ) -> list[str]:
        from constat.core.api import EntityManager
        from constat.server.routes.sessions import (
            _apply_resolved_source_overrides,
            _load_domains_into_session,
        )
        from constat.server.user_preferences import set_selected_domains

        sm = info.context.session_manager
        user_id = info.context.user_id
        managed = sm.get_session_or_none(session_id)
        if not managed:
            raise ValueError("Session not found")

        domain_filenames = domains

        # Verify all domains exist (skip synthetic nodes)
        synthetic = {'root', 'user'}
        config = managed.session.config
        for filename in domain_filenames:
            if filename in synthetic:
                continue
            domain = config.load_domain(filename)
            if not domain:
                raise ValueError(f"Domain not found: {filename}")

        # Incremental entity updates
        old_domains = set(managed.active_domains or [])
        new_domains = set(domain_filenames)

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

        # Clear existing domain state before loading new domains
        managed.session.clear_domain_apis()
        managed._domain_databases = set()
        managed.active_domains = []

        # Load real domains (skip synthetic nodes)
        real_filenames = [f for f in domain_filenames if f not in synthetic]
        synthetic_active = [f for f in domain_filenames if f in synthetic]
        loaded, conflicts = _load_domains_into_session(managed, real_filenames)

        if synthetic_active:
            managed.active_domains = synthetic_active + managed.active_domains

        if conflicts:
            raise ValueError(f"Conflicting data source names: {conflicts}")

        # Re-resolve tiered config
        resolved = sm.resolve_config(session_id)
        if resolved and resolved.system_prompt:
            managed.session.config.system_prompt = resolved.system_prompt
            managed.session_prompt = resolved.system_prompt
        if resolved:
            _apply_resolved_source_overrides(managed, resolved)

        # Save preferences
        effective_user_id = user_id if user_id != "default" else managed.user_id
        set_selected_domains(effective_user_id, real_filenames)

        return managed.active_domains
