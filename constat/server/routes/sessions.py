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
from typing import Optional

from fastapi import APIRouter, Depends, Request

from constat.server.models import (
    SessionCreate,
    SessionResponse,
    SessionListResponse,
    SessionStatus,
)
from constat.server.session_manager import SessionManager, ManagedSession

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
    """List all sessions.

    Optionally filter by user ID.

    Args:
        user_id: Optional user ID filter

    Returns:
        List of sessions
    """
    sessions = session_manager.list_sessions(user_id=user_id)
    return SessionListResponse(
        sessions=[_session_to_response(s) for s in sessions],
        total=len(sessions),
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
