# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Schema discovery REST endpoints."""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from constat.core.config import Config
from constat.server.models import (
    DatabaseInfo,
    SchemaOverviewResponse,
    TableSchemaResponse,
)
from constat.server.session_manager import SessionManager

logger = logging.getLogger(__name__)

router = APIRouter()


def get_config(request: Request) -> Config:
    """Dependency to get config from app state."""
    return request.app.state.config


def get_session_manager(request: Request) -> SessionManager:
    """Dependency to get session manager from app state."""
    return request.app.state.session_manager


@router.get("", response_model=SchemaOverviewResponse)
async def get_schema_overview(
    config: Config = Depends(get_config),
) -> SchemaOverviewResponse:
    """Get overview of all configured data sources.

    Returns information about configured databases, APIs, and documents
    without requiring an active session.

    Returns:
        Schema overview with database/API/document info
    """
    databases = []
    for name, db_config in config.databases.items():
        databases.append(
            DatabaseInfo(
                name=name,
                description=db_config.description,
                table_count=0,  # Would need schema manager to get actual count
                type=db_config.type,
            )
        )

    return SchemaOverviewResponse(
        databases=databases,
        apis=list(config.apis.keys()),
        documents=list(config.documents.keys()),
    )


@router.get("/databases/{database_name}/tables")
async def list_database_tables(
    database_name: str,
    session_id: str = Query(description="Session ID to use for schema access"),
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    """List tables in a specific database.

    Requires an active session to access the schema manager.

    Args:
        database_name: Name of the database
        session_id: Session ID for schema access
        session_manager: Injected session manager

    Returns:
        List of tables in the database

    Raises:
        404: Session or database not found
    """
    managed = session_manager.get_session(session_id)

    if not managed.has_database(database_name):
        raise HTTPException(status_code=404, detail=f"Database not found: {database_name}")

    try:
        tables = managed.session.schema_manager.get_tables_for_db(database_name)

        return {
            "database": database_name,
            "tables": [
                {
                    "name": t.name,
                    "row_count": t.row_count,
                    "column_count": len(t.columns),
                }
                for t in tables
            ],
        }
    except Exception as e:
        logger.error(f"Error listing tables: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing tables for {database_name}: {e}")


@router.get("/databases/{database_name}/tables/{table_name}", response_model=TableSchemaResponse)
async def get_table_schema(
    database_name: str,
    table_name: str,
    session_id: str = Query(description="Session ID to use for schema access"),
    session_manager: SessionManager = Depends(get_session_manager),
) -> TableSchemaResponse:
    """Get detailed schema for a specific table.

    Requires an active session to access the schema manager.

    Args:
        database_name: Name of the database
        table_name: Name of the table
        session_id: Session ID for schema access
        session_manager: Injected session manager

    Returns:
        Detailed table schema including columns and relationships

    Raises:
        404: Session, database, or table not found
    """
    managed = session_manager.get_session(session_id)

    try:
        full_name = f"{database_name}.{table_name}"
        table_meta = managed.session.schema_manager.get_table_schema(full_name)

        if not table_meta:
            raise HTTPException(status_code=404, detail=f"Table not found: {full_name}")

        return TableSchemaResponse(
            database=table_meta.database,
            table_name=table_meta.table_name,
            columns=[
                {
                    "name": col.name,
                    "type": col.data_type,
                    "nullable": col.nullable,
                    "primary_key": col.primary_key,
                    "sample_values": col.sample_values,
                }
                for col in table_meta.columns
            ],
            row_count=table_meta.row_count,
            relationships=[
                {
                    "from_column": fk.from_column,
                    "to_table": fk.to_table,
                    "to_column": fk.to_column,
                }
                for fk in table_meta.relationships
            ],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting table schema: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search")
async def search_schema(
    query: str = Query(description="Search query"),
    session_id: str = Query(description="Session ID to use for schema access"),
    limit: int = Query(default=10, ge=1, le=50, description="Maximum results"),
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    """Search for relevant tables across all databases.

    Uses semantic search to find tables matching the query.

    Args:
        query: Natural language search query
        session_id: Session ID for schema access
        limit: Maximum number of results
        session_manager: Injected session manager

    Returns:
        List of matching tables with relevance scores

    Raises:
        404: Session not found
    """
    managed = session_manager.get_session(session_id)

    try:
        results = managed.session.schema_manager.find_relevant_tables(
            query, top_k=limit
        )

        return {
            "query": query,
            "results": [
                {
                    "database": r["database"],
                    "table": r["full_name"],
                    "summary": r.get("summary", ""),
                    "relevance": r.get("relevance", 0),
                }
                for r in results
            ],
        }

    except Exception as e:
        logger.error(f"Error searching schema: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/apis/{api_name}")
async def get_api_schema(
    api_name: str,
    session_id: str = Query(description="Session ID to use for schema access"),
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    """Get schema overview for a configured API.

    Args:
        api_name: Name of the API
        session_id: Session ID for schema access
        session_manager: Injected session manager

    Returns:
        API schema overview including endpoints/queries

    Raises:
        404: Session or API not found
    """
    managed = session_manager.get_session(session_id)

    try:
        # Check config APIs and project APIs
        api_config = managed.session.config.apis.get(api_name)
        if not api_config:
            for project_filename in managed.active_projects:
                project = managed.session.config.load_project(project_filename)
                if project and api_name in project.apis:
                    api_config = project.apis[api_name]
                    break
        if not api_config:
            raise HTTPException(status_code=404, detail=f"API not found: {api_name}")

        # Get endpoints from api_schema_manager
        endpoints_meta = managed.session.api_schema_manager.get_api_schema(api_name)

        return {
            "name": api_name,
            "type": api_config.type,
            "description": api_config.description,
            "endpoints": [
                {
                    "name": ep.endpoint_name,
                    "kind": ep.api_type,
                    "return_type": ep.return_type,
                    "description": ep.description,
                    "http_method": ep.http_method,
                    "http_path": ep.http_path,
                    "fields": [
                        {
                            "name": f.name,
                            "type": f.type,
                            "description": f.description,
                            "is_required": f.is_required,
                        }
                        for f in ep.fields
                    ],
                }
                for ep in endpoints_meta
            ],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting API schema: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/refresh")
async def refresh_schema(
    session_id: str = Query(description="Session ID to use for schema refresh"),
    force_full: bool = Query(default=False, description="Force full refresh"),
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    """Refresh schema metadata.

    Re-introspects all databases, APIs, and documents to update
    the cached schema information.

    Args:
        session_id: Session ID for schema refresh
        force_full: Whether to force full rebuild
        session_manager: Injected session manager

    Returns:
        Refresh statistics

    Raises:
        404: Session not found
    """
    managed = session_manager.get_session(session_id)

    try:
        stats = managed.session.refresh_metadata(force_full=force_full)
        return {
            "status": "refreshed",
            "stats": stats,
        }

    except Exception as e:
        logger.error(f"Error refreshing schema: {e}")
        raise HTTPException(status_code=500, detail=str(e))
