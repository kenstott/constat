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
    SessionApiInfo,
    SessionDatabaseInfo,
    SessionDatabaseListResponse,
    SessionDataSourcesResponse,
    SessionDocumentInfo,
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

    # Validate file type for file-based databases
    uri_lower = uri.lower()
    if uri_lower.startswith("file://"):
        file_path = uri[7:]  # Strip file:// prefix
    else:
        file_path = uri

    # Block xlsx as database (multi-sheet complexity)
    if file_path.endswith('.xlsx'):
        raise HTTPException(
            status_code=400,
            detail="Excel files (.xlsx) cannot be added as databases due to multi-sheet complexity. "
                   "Use 'Add Document' to index for search, or convert to CSV/Parquet."
        )

    # Validate JSON structure for JSON files
    if file_path.endswith('.json'):
        import json
        from pathlib import Path
        json_path = Path(file_path)
        if json_path.exists():
            try:
                data = json.loads(json_path.read_text())
                if not isinstance(data, list):
                    raise HTTPException(
                        status_code=400,
                        detail="JSON file must contain an array of objects to be used as a database"
                    )
                if data and not isinstance(data[0], dict):
                    raise HTTPException(
                        status_code=400,
                        detail="JSON array must contain objects (not primitives) to be used as a database"
                    )
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=400, detail=f"Invalid JSON file: {e}")

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

    Includes config-defined, project-defined, and dynamically added databases.

    Args:
        session_id: Session ID

    Returns:
        List of databases

    Raises:
        404: Session not found
    """
    managed = session_manager.get_session(session_id)
    databases = []
    seen_names = set()

    # Add config databases (from global config)
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
            source="config",
        ))
        seen_names.add(name)

    # Add project databases (from all active projects)
    for project_filename in managed.active_projects:
        project = managed.session.config.load_project(project_filename)
        if project:
            for name, db_config in project.databases.items():
                if name in seen_names:
                    continue  # Skip duplicates (conflicts checked at project selection)

                # Check if this database was loaded into the session
                connected = name in getattr(managed, "_project_databases", set())
                table_count = 0
                if connected and managed.session.schema_manager:
                    try:
                        tables = managed.session.schema_manager.get_tables_for_db(name)
                        table_count = len(tables)
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
                    source=project_filename,
                ))
                seen_names.add(name)

    # Add dynamic databases (session-added)
    dynamic_dbs = _get_dynamic_dbs(managed)
    for db in dynamic_dbs:
        if db["name"] in seen_names:
            continue  # Skip duplicates

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
            source="session",
        ))

    return SessionDatabaseListResponse(databases=databases)


@router.get("/{session_id}/sources", response_model=SessionDataSourcesResponse)
async def list_data_sources(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> SessionDataSourcesResponse:
    """List all data sources available to the session.

    Returns databases, APIs, and documents from config, active projects,
    and session-added sources.

    Args:
        session_id: Session ID

    Returns:
        Combined list of all data sources

    Raises:
        404: Session not found
    """
    managed = session_manager.get_session(session_id)
    config = managed.session.config

    # Get databases (reuse existing logic)
    db_response = await list_databases(session_id, session_manager)
    databases = db_response.databases

    # Collect APIs
    apis: list[SessionApiInfo] = []
    seen_apis: set[str] = set()

    # Config APIs
    for name, api_config in config.apis.items():
        apis.append(SessionApiInfo(
            name=name,
            type=api_config.type,
            description=api_config.description,
            base_url=api_config.url,
            connected=True,  # Assume connected if in config
            from_config=True,
            source="config",
        ))
        seen_apis.add(name)

    # Project APIs
    for project_filename in managed.active_projects:
        project = config.load_project(project_filename)
        if project:
            for name, api_config in project.apis.items():
                if name in seen_apis:
                    continue  # Skip duplicates (conflicts checked at selection)
                apis.append(SessionApiInfo(
                    name=name,
                    type=api_config.type,
                    description=api_config.description,
                    base_url=api_config.url,
                    connected=True,
                    from_config=False,
                    source=project_filename,
                ))
                seen_apis.add(name)

    # Collect Documents
    documents: list[SessionDocumentInfo] = []
    seen_docs: set[str] = set()

    # Config documents
    for name, doc_config in config.documents.items():
        documents.append(SessionDocumentInfo(
            name=name,
            type=doc_config.type,
            description=doc_config.description,
            path=doc_config.path,
            indexed=True,  # Assume indexed if in config
            from_config=True,
            source="config",
        ))
        seen_docs.add(name)

    # Project documents
    for project_filename in managed.active_projects:
        project = config.load_project(project_filename)
        if project:
            for name, doc_config in project.documents.items():
                if name in seen_docs:
                    continue  # Skip duplicates
                documents.append(SessionDocumentInfo(
                    name=name,
                    type=doc_config.type,
                    description=doc_config.description,
                    path=doc_config.path,
                    indexed=True,
                    from_config=False,
                    source=project_filename,
                ))
                seen_docs.add(name)

    # Session-added file refs (documents)
    try:
        from constat.server.routes.files import _get_file_refs
        file_refs = _get_file_refs(managed)
        for ref in file_refs:
            if ref["name"] in seen_docs:
                continue
            documents.append(SessionDocumentInfo(
                name=ref["name"],
                type=ref.get("uri", "").split(".")[-1] if ref.get("uri") else None,
                description=ref.get("description"),
                path=ref.get("uri"),
                indexed=True,
                source="session",
                from_config=False,
            ))
    except Exception:
        pass  # File refs might not exist

    return SessionDataSourcesResponse(
        databases=databases,
        apis=apis,
        documents=documents,
    )


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