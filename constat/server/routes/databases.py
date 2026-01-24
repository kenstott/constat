# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Dynamic database connection REST endpoints."""

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from constat.server.models import (
    DatabaseAddRequest,
    DatabaseTestResponse,
    SessionDatabaseInfo,
    SessionDatabaseListResponse,
)
from constat.server.session_manager import ManagedSession, SessionManager

logger = logging.getLogger(__name__)

router = APIRouter()


def get_session_manager(request: Request) -> SessionManager:
    """Dependency to get session manager from app state."""
    return request.app.state.session_manager


def _get_dynamic_dbs(managed: ManagedSession) -> list[dict[str, Any]]:
    """Get dynamically added databases from session."""
    if not hasattr(managed, "_dynamic_dbs"):
        managed._dynamic_dbs = []
    return managed._dynamic_dbs


@router.post("/{session_id}/databases", response_model=SessionDatabaseInfo)
async def add_database(
    session_id: str,
    body: DatabaseAddRequest,
    session_manager: SessionManager = Depends(get_session_manager),
) -> SessionDatabaseInfo:
    """Add a database connection to the session.

    This wraps the Session.add_database() method for dynamically
    connecting to databases during a session.

    Args:
        session_id: Session ID
        body: Database add request

    Returns:
        Database connection information

    Raises:
        404: Session not found
        400: Invalid database configuration
    """
    managed = session_manager.get_session(session_id)

    # Determine the URI
    uri = body.uri
    if body.file_id:
        # Look up file URI from uploaded files
        from constat.server.routes.files import _get_uploaded_files
        files = _get_uploaded_files(session_id)
        file_info = next((f for f in files if f["id"] == body.file_id), None)
        if not file_info:
            raise HTTPException(status_code=400, detail=f"File not found: {body.file_id}")
        # Convert file:// URI to path for database connection
        uri = file_info["file_uri"]

    if not uri:
        raise HTTPException(status_code=400, detail="Either uri or file_id is required")

    # Try to add database via session
    connected = False
    table_count = 0
    dialect = None

    try:
        if hasattr(managed.session, "add_database"):
            managed.session.add_database(
                name=body.name,
                uri=uri,
                db_type=body.type,
                description=body.description,
            )
            connected = True

            # Get table count
            if managed.session.schema_manager:
                tables = managed.session.schema_manager.get_tables_for_db(body.name)
                table_count = len(tables)

        # Detect dialect from URI
        if "postgresql" in uri or "postgres" in uri:
            dialect = "postgresql"
        elif "mysql" in uri:
            dialect = "mysql"
        elif "sqlite" in uri:
            dialect = "sqlite"
        elif "duckdb" in uri:
            dialect = "duckdb"
        elif "mssql" in uri:
            dialect = "mssql"

    except Exception as e:
        logger.error(f"Error adding database: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    # Track in managed session
    now = datetime.now(timezone.utc)
    dynamic_dbs = _get_dynamic_dbs(managed)
    db_info = {
        "name": body.name,
        "type": body.type,
        "dialect": dialect,
        "description": body.description,
        "uri": uri,
        "connected": connected,
        "table_count": table_count,
        "added_at": now.isoformat(),
        "is_dynamic": True,
        "file_id": body.file_id,
    }
    dynamic_dbs.append(db_info)

    return SessionDatabaseInfo(
        name=body.name,
        type=body.type,
        dialect=dialect,
        description=body.description,
        connected=connected,
        table_count=table_count,
        added_at=now,
        is_dynamic=True,
        file_id=body.file_id,
    )


@router.get("/{session_id}/databases", response_model=SessionDatabaseListResponse)
async def list_databases(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> SessionDatabaseListResponse:
    """List all databases available to the session.

    Includes both config-defined and dynamically added databases.

    Args:
        session_id: Session ID

    Returns:
        List of databases

    Raises:
        404: Session not found
    """
    managed = session_manager.get_session(session_id)
    databases = []

    # Add config databases
    for name, db_config in managed.session.config.databases.items():
        connected = False
        table_count = 0

        try:
            if managed.session.schema_manager:
                tables = managed.session.schema_manager.get_tables_for_db(name)
                table_count = len(tables)
                connected = True
        except Exception:
            pass

        databases.append(SessionDatabaseInfo(
            name=name,
            type=db_config.type,
            dialect=None,
            description=db_config.description,
            connected=connected,
            table_count=table_count,
            added_at=managed.created_at,
            is_dynamic=False,
            file_id=None,
        ))

    # Add dynamic databases
    dynamic_dbs = _get_dynamic_dbs(managed)
    for db in dynamic_dbs:
        databases.append(SessionDatabaseInfo(
            name=db["name"],
            type=db["type"],
            dialect=db.get("dialect"),
            description=db.get("description"),
            connected=db.get("connected", False),
            table_count=db.get("table_count", 0),
            added_at=datetime.fromisoformat(db["added_at"]),
            is_dynamic=True,
            file_id=db.get("file_id"),
        ))

    return SessionDatabaseListResponse(databases=databases)


@router.delete("/{session_id}/databases/{db_name}")
async def remove_database(
    session_id: str,
    db_name: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict:
    """Remove a dynamically added database.

    Note: Config-defined databases cannot be removed.

    Args:
        session_id: Session ID
        db_name: Database name

    Returns:
        Deletion confirmation

    Raises:
        404: Session or database not found
        400: Cannot remove config database
    """
    managed = session_manager.get_session(session_id)

    # Check if it's a config database
    if db_name in managed.session.config.databases:
        raise HTTPException(
            status_code=400,
            detail="Cannot remove config-defined database"
        )

    # Remove from dynamic databases
    dynamic_dbs = _get_dynamic_dbs(managed)
    original_len = len(dynamic_dbs)
    managed._dynamic_dbs = [db for db in dynamic_dbs if db["name"] != db_name]

    if len(managed._dynamic_dbs) == original_len:
        raise HTTPException(status_code=404, detail=f"Database not found: {db_name}")

    # Try to remove from session's schema manager
    try:
        if hasattr(managed.session, "remove_database"):
            managed.session.remove_database(db_name)
    except Exception as e:
        logger.warning(f"Error removing database from session: {e}")

    return {
        "status": "deleted",
        "name": db_name,
    }


@router.post("/{session_id}/databases/{db_name}/test", response_model=DatabaseTestResponse)
async def test_database_connection(
    session_id: str,
    db_name: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> DatabaseTestResponse:
    """Test a database connection.

    Args:
        session_id: Session ID
        db_name: Database name

    Returns:
        Connection test results

    Raises:
        404: Session or database not found
    """
    managed = session_manager.get_session(session_id)

    # Find the database
    db_config = managed.session.config.databases.get(db_name)
    dynamic_dbs = _get_dynamic_dbs(managed)
    dynamic_db = next((db for db in dynamic_dbs if db["name"] == db_name), None)

    if not db_config and not dynamic_db:
        raise HTTPException(status_code=404, detail=f"Database not found: {db_name}")

    # Test connection
    connected = False
    table_count = 0
    error = None

    try:
        if managed.session.schema_manager:
            tables = managed.session.schema_manager.get_tables_for_db(db_name)
            table_count = len(tables)
            connected = True
    except Exception as e:
        error = str(e)

    return DatabaseTestResponse(
        name=db_name,
        connected=connected,
        table_count=table_count,
        error=error,
    )