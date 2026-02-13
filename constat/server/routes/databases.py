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
    ApiAddRequest,
    DatabaseAddRequest,
    DatabaseTestResponse,
    SessionApiInfo,
    SessionDatabaseInfo,
    SessionDatabaseListResponse,
    SessionDataSourcesResponse,
    SessionDocumentInfo,
)
from constat.server.session_manager import SessionManager

logger = logging.getLogger(__name__)

router = APIRouter()


def get_session_manager(request: Request) -> SessionManager:
    """Dependency to get session manager from app state."""
    return request.app.state.session_manager



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
        session_manager: Injected session manager

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
        from constat.server.routes.files import _get_uploaded_files_for_session
        files = _get_uploaded_files_for_session(managed)
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

    # Detect actual file type from extension for file-based sources
    # This ensures schema_manager recognizes it as a file source
    effective_type = body.type
    file_extensions = {'.csv': 'csv', '.tsv': 'csv', '.parquet': 'parquet', '.json': 'json', '.jsonl': 'jsonl'}
    for ext, ftype in file_extensions.items():
        if file_path.lower().endswith(ext):
            effective_type = ftype
            break

    try:
        if hasattr(managed.session, "add_database"):
            managed.session.add_database(
                name=body.name,
                uri=uri,
                db_type=effective_type,
                description=body.description,
            )
            connected = True

            # Add to schema_manager for entity extraction
            if managed.session.schema_manager:
                from constat.core.config import DatabaseConfig
                # Use 'path' for file-based sources, 'uri' for SQL databases
                is_file_source = effective_type in ('csv', 'json', 'jsonl', 'parquet', 'arrow', 'feather')
                db_config = DatabaseConfig(
                    type=effective_type,
                    path=uri if is_file_source else None,
                    uri=uri if not is_file_source else None,
                    description=body.description,
                )
                managed.session.schema_manager.add_database_dynamic(body.name, db_config)

                # Count tables for this db in metadata_cache
                table_count = sum(1 for k in managed.session.schema_manager.metadata_cache if k.startswith(f"{body.name}."))

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
    dynamic_dbs = managed._dynamic_dbs
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

    # Refresh entities in background (non-blocking)
    if connected:
        session_manager.refresh_entities_async(session_id)

    # Persist resources for session restoration
    managed.save_resources()

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
        session_manager: Injected session manager

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
    dynamic_dbs = managed._dynamic_dbs
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
        session_manager: Injected session manager

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

    # Dynamic APIs (session-added)
    dynamic_apis = managed._dynamic_apis
    for api in dynamic_apis:
        if api["name"] in seen_apis:
            continue  # Skip duplicates
        apis.append(SessionApiInfo(
            name=api["name"],
            type=api.get("type"),
            description=api.get("description"),
            base_url=api.get("base_url"),
            connected=api.get("connected", True),
            from_config=False,
            source="session",
            is_dynamic=True,
        ))

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
    for ref in managed._file_refs:
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
        session_manager: Injected session manager

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

    # Check if it's a project database
    for project_filename in managed.active_projects:
        project = managed.session.config.load_project(project_filename)
        if project and db_name in project.databases:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot remove project-defined database (from {project_filename})"
            )

    # Find the database to get its file path before removing
    dynamic_dbs = managed._dynamic_dbs
    db_to_remove = next((db for db in dynamic_dbs if db["name"] == db_name), None)

    if not db_to_remove:
        raise HTTPException(status_code=404, detail=f"Database not found: {db_name}")

    # Get file path for deletion (if it's a file-based database)
    file_path = None
    uri = db_to_remove.get("uri", "")
    if uri and not uri.startswith(("postgresql", "mysql", "sqlite", "mssql", "mongodb")):
        # It's a file path - strip file:// prefix if present
        file_path = uri[7:] if uri.startswith("file://") else uri

    # Remove from dynamic databases list
    logger.info(f"remove_database({db_name}): removing from _dynamic_dbs (had {len(dynamic_dbs)} entries)")
    managed._dynamic_dbs = [db for db in dynamic_dbs if db["name"] != db_name]

    # Remove from session_databases
    if db_name in managed.session.session_databases:
        del managed.session.session_databases[db_name]
        logger.info(f"remove_database({db_name}): removed from session_databases")

    # Remove from schema_manager (metadata, connections, and chunks)
    if managed.session.schema_manager:
        result = managed.session.schema_manager.remove_database_dynamic(db_name)
        logger.info(f"remove_database({db_name}): schema_manager.remove_database_dynamic returned {result}")
    else:
        logger.warning(f"remove_database({db_name}): no schema_manager!")

    # Delete the file from disk if it exists
    file_deleted = False
    if file_path:
        from pathlib import Path
        fp = Path(file_path)
        if fp.exists():
            try:
                fp.unlink()
                file_deleted = True
                logger.info(f"Deleted file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to delete file {file_path}: {e}")

    # Refresh entities in background (non-blocking)
    session_manager.refresh_entities_async(session_id)

    # Persist resources for session restoration
    managed.save_resources()
    logger.info(f"remove_database({db_name}): deletion complete")

    return {
        "status": "deleted",
        "name": db_name,
        "file_deleted": file_deleted,
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

    if not managed.has_database(db_name):
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


@router.get("/{session_id}/databases/{db_name}/tables/{table_name}/preview")
async def preview_database_table(
    session_id: str,
    db_name: str,
    table_name: str,
    page: int = 1,
    page_size: int = 100,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    """Preview data from a source database table.

    Args:
        session_id: Session ID
        db_name: Database name
        table_name: Table name
        page: Page number (1-indexed)
        page_size: Number of rows per page
        session_manager: Injected session manager

    Returns:
        Table data with columns, rows, and pagination info

    Raises:
        404: Session, database, or table not found
        500: Query execution error
    """
    import pandas as pd

    managed = session_manager.get_session(session_id)

    if not managed.has_database(db_name):
        raise HTTPException(status_code=404, detail=f"Database not found: {db_name}")

    db_connection = managed.get_database_connection(db_name)
    if not db_connection:
        raise HTTPException(
            status_code=404,
            detail=f"Database '{db_name}' is not connected. Try reconnecting.",
        )

    try:
        # Calculate offset
        offset = (page - 1) * page_size

        # File-based sources use DuckDB; SQL sources use pd.read_sql
        from constat.catalog.file.connector import FileConnector
        if isinstance(db_connection, FileConnector):
            import duckdb
            conn = duckdb.connect(":memory:")
            file_path = db_connection.path
            ft = db_connection.file_type.value  # csv, json, parquet, etc.
            read_fn = {
                'csv': f"read_csv_auto('{file_path}')",
                'tsv': f"read_csv_auto('{file_path}', delim='\\t')",
                'json': f"read_json_auto('{file_path}')",
                'jsonl': f"read_json_auto('{file_path}', format='newline_delimited')",
                'parquet': f"read_parquet('{file_path}')",
                'arrow': f"read_parquet('{file_path}')",
                'feather': f"read_parquet('{file_path}')",
            }.get(ft, f"read_csv_auto('{file_path}')")
            df = conn.execute(f"SELECT * FROM {read_fn} LIMIT {page_size} OFFSET {offset}").df()
            total_rows = conn.execute(f"SELECT COUNT(*) as cnt FROM {read_fn}").fetchone()[0]
            conn.close()
        else:
            query = f'SELECT * FROM "{table_name}" LIMIT {page_size} OFFSET {offset}'
            df = pd.read_sql(query, db_connection)
            count_query = f'SELECT COUNT(*) as cnt FROM "{table_name}"'
            count_df = pd.read_sql(count_query, db_connection)
            # noinspection PyTypeChecker
            total_rows = int(count_df.iloc[0]["cnt"])

        # Convert to response format
        columns = list(df.columns)
        data = df.to_dict(orient="records")

        return {
            "database": db_name,
            "table_name": table_name,
            "columns": columns,
            "data": data,
            "page": page,
            "page_size": page_size,
            "total_rows": total_rows,
            "has_more": offset + len(data) < total_rows,
        }

    except Exception as e:
        logger.error(f"Error previewing table {db_name}.{table_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error querying table: {e}")


# =============================================================================
# API Routes
# =============================================================================


@router.post("/{session_id}/apis", response_model=SessionApiInfo)
async def add_api(
    session_id: str,
    body: ApiAddRequest,
    session_manager: SessionManager = Depends(get_session_manager),
) -> SessionApiInfo:
    """Add an API connection to the session.

    Args:
        session_id: Session ID
        body: API add request
        session_manager: Injected session manager

    Returns:
        Information about the added API

    Raises:
        400: API already exists
        404: Session not found
    """
    managed = session_manager.get_session(session_id)

    # Check for duplicate name
    dynamic_apis = managed._dynamic_apis
    if any(api["name"] == body.name for api in dynamic_apis):
        raise HTTPException(
            status_code=400,
            detail=f"API already exists: {body.name}"
        )

    # Check config APIs
    if body.name in managed.session.config.apis:
        raise HTTPException(
            status_code=400,
            detail=f"API already exists in config: {body.name}"
        )

    # Track in managed session
    now = datetime.now(timezone.utc)
    api_info = {
        "name": body.name,
        "type": body.type,
        "base_url": body.base_url,
        "description": body.description,
        "auth_type": body.auth_type,
        "auth_header": body.auth_header,
        "connected": True,  # Assume connected, will be validated on use
        "added_at": now.isoformat(),
        "is_dynamic": True,
    }
    dynamic_apis.append(api_info)

    # Add to session resources
    managed.session.resources.add_api(
        name=body.name,
        description=body.description or "",
        api_type=body.type,
        source="session",
    )

    # Introspect API and build chunks/embeddings for semantic search
    if managed.session.api_schema_manager:
        from constat.core.config import APIConfig
        api_config = APIConfig(
            type=body.type,
            url=body.base_url,
            description=body.description or "",
        )
        if body.auth_type and body.auth_header:
            api_config.headers = {body.auth_header: ""}  # Placeholder, actual token set at request time
        managed.session.api_schema_manager.add_api_dynamic(body.name, api_config)

    # Refresh entities in background (non-blocking)
    session_manager.refresh_entities_async(session_id)

    # Persist resources for session restoration
    managed.save_resources()

    logger.info(f"Added dynamic API: {body.name} to session {session_id}")

    return SessionApiInfo(
        name=body.name,
        type=body.type,
        description=body.description,
        base_url=body.base_url,
        connected=True,
        from_config=False,
        source="session",
        is_dynamic=True,
    )


@router.delete("/{session_id}/apis/{api_name}")
async def remove_api(
    session_id: str,
    api_name: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    """Remove a dynamically added API from the session.

    Args:
        session_id: Session ID
        api_name: API name to remove
        session_manager: Injected session manager

    Returns:
        Status message

    Raises:
        400: Cannot remove config-defined API
        404: Session or API not found
    """
    managed = session_manager.get_session(session_id)

    # Check if it's a config API
    if api_name in managed.session.config.apis:
        raise HTTPException(
            status_code=400,
            detail="Cannot remove config-defined API"
        )

    # Find the API in dynamic APIs
    dynamic_apis = managed._dynamic_apis
    api_to_remove = next((api for api in dynamic_apis if api["name"] == api_name), None)

    if not api_to_remove:
        raise HTTPException(status_code=404, detail=f"API not found: {api_name}")

    # Remove from dynamic APIs list
    logger.info(f"remove_api({api_name}): removing from _dynamic_apis")
    managed._dynamic_apis = [api for api in dynamic_apis if api["name"] != api_name]

    # Remove from session resources
    managed.session.resources.remove_api(api_name)

    # Refresh entities in background (non-blocking)
    session_manager.refresh_entities_async(session_id)

    # Persist resources for session restoration
    managed.save_resources()
    logger.info(f"remove_api({api_name}): deletion complete")

    return {
        "status": "deleted",
        "name": api_name,
    }