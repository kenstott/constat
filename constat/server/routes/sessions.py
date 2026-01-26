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

from fastapi import APIRouter, Depends, Request

from constat.server.models import (
    SessionCreate,
    SessionResponse,
    SessionListResponse,
    SessionStatus,
)
from constat.server.session_manager import SessionManager, ManagedSession
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

    return SessionResponse(
        session_id=managed.session_id,
        user_id=managed.user_id,
        status=managed.status,
        created_at=managed.created_at,
        last_activity=managed.last_activity,
        current_query=managed.current_query,
        tables_count=tables_count,
        artifacts_count=artifacts_count,
    )


@router.post("", response_model=SessionResponse)
async def create_session(
    body: SessionCreate,
    session_manager: SessionManager = Depends(get_session_manager),
) -> SessionResponse:
    """Create a new session.

    Creates a new Constat session for the specified user.
    The session can then be used for query execution via the query endpoints.

    Returns:
        Session details including the session ID
    """
    session_id = session_manager.create_session(user_id=body.user_id)
    managed = session_manager.get_session(session_id)
    return _session_to_response(managed)


@router.get("", response_model=SessionListResponse)
async def list_sessions(
    user_id: Optional[str] = None,
    session_manager: SessionManager = Depends(get_session_manager),
) -> SessionListResponse:
    """List all sessions (in-memory + historical from disk).

    Optionally filter by user ID.

    Args:
        user_id: Optional user ID filter

    Returns:
        List of sessions, newest first
    """
    # Get in-memory sessions
    in_memory = session_manager.list_sessions(user_id=user_id)
    in_memory_ids = {s.session_id for s in in_memory}

    # Convert in-memory sessions to responses
    responses = [_session_to_response(s) for s in in_memory]

    # Get historical sessions from disk (not already in memory)
    try:
        history = SessionHistory(user_id=user_id or "default")
        historical = history.list_sessions(limit=50)

        for hist in historical:
            if hist.session_id not in in_memory_ids:
                # Convert historical session to response format
                try:
                    created_at = datetime.fromisoformat(hist.created_at.replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    created_at = datetime.now(timezone.utc)

                responses.append(SessionResponse(
                    session_id=hist.session_id,
                    user_id=hist.user_id or user_id or "default",
                    status=SessionStatus.IDLE,  # Historical sessions are idle
                    created_at=created_at,
                    last_activity=created_at,  # Use created_at as last activity for historical
                    current_query=hist.summary,
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

    # Get messages from session history
    history = SessionHistory(user_id=managed.session.session_config.user_id or "default")
    history_session_id = managed.history_session_id

    if history_session_id:
        messages = history.load_messages(history_session_id)
    else:
        messages = []

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

    # Save messages to session history
    history = SessionHistory(user_id=managed.session.session_config.user_id or "default")
    history_session_id = managed.history_session_id

    if history_session_id:
        history.save_messages(history_session_id, messages)

    return {"status": "saved", "count": len(messages)}
