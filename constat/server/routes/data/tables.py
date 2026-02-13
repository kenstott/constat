# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Table data endpoints."""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response

from constat.server.models import (
    TableDataResponse,
    TableInfo,
    TableListResponse,
    TableVersionInfo,
    TableVersionsResponse,
)
from constat.server.routes.data import get_session_manager, _sanitize_df_for_json
from constat.server.session_manager import SessionManager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/{session_id}/tables", response_model=TableListResponse)
async def list_tables(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> TableListResponse:
    """List all tables in the session's datastore.

    Args:
        session_id: Session ID

    Returns:
        List of tables with metadata

    Raises:
        404: Session not found
    """
    managed = session_manager.get_session(session_id)

    if not managed.session.datastore:
        return TableListResponse(tables=[])

    try:
        tables = managed.session.datastore.list_tables()
        starred_tables = set(managed.session.datastore.get_starred_tables())
        unstarred_tables = set(managed.session.datastore.get_state("_unstarred_tables") or [])

        # Internal tables that should be hidden from the UI
        internal_tables = {"execution_history", "_facts", "_metadata"}

        result_tables = []
        for t in tables:
            table_name = t["name"]

            # Skip internal tables (underscore prefix or known internal names)
            if table_name.startswith("_") or table_name in internal_tables:
                continue

            # Unified starred logic (same as list_artifacts)
            is_published = t.get("is_published", False)
            is_final_step = t.get("is_final_step", False)
            has_data = t.get("row_count", 0) > 0

            if table_name in starred_tables:
                is_starred = True
            elif table_name in unstarred_tables:
                is_starred = False
            else:
                # Auto-star tables that are published or from final step with data
                is_starred = is_published or (is_final_step and has_data)

            result_tables.append(
                TableInfo(
                    name=table_name,
                    row_count=t.get("row_count", 0),
                    step_number=t.get("step_number", 0),
                    columns=t.get("columns", []),
                    is_starred=is_starred,
                    version=t.get("version", 1),
                    version_count=t.get("version_count", 1),
                )
            )

        return TableListResponse(tables=result_tables)
    except Exception as e:
        logger.error(f"Error listing tables: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}/tables/{table_name}/versions", response_model=TableVersionsResponse)
async def get_table_versions(
    session_id: str,
    table_name: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> TableVersionsResponse:
    """Get version history for a table."""
    managed = session_manager.get_session(session_id)

    if not managed.session.datastore:
        raise HTTPException(status_code=404, detail="No datastore for this session")

    versions = managed.session.datastore.get_table_versions(table_name)
    if not versions:
        raise HTTPException(status_code=404, detail=f"Table not found: {table_name}")

    return TableVersionsResponse(
        name=table_name,
        current_version=versions[0]["version"] if versions else 1,
        versions=[
            TableVersionInfo(
                version=v["version"],
                step_number=v.get("step_number"),
                row_count=v.get("row_count", 0),
                created_at=v.get("created_at"),
            )
            for v in versions
        ],
    )


@router.get("/{session_id}/tables/{table_name}/version/{version}", response_model=TableDataResponse)
async def get_table_version_data(
    session_id: str,
    table_name: str,
    version: int,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=100, ge=1, le=1000),
    session_manager: SessionManager = Depends(get_session_manager),
) -> TableDataResponse:
    """Get data for a specific version of a table."""
    managed = session_manager.get_session(session_id)

    if not managed.session.datastore:
        raise HTTPException(status_code=404, detail="No datastore for this session")

    df = managed.session.datastore.load_table_version(table_name, version)
    if df is None:
        raise HTTPException(status_code=404, detail=f"Table version not found: {table_name} v{version}")

    total_rows = len(df)
    start = (page - 1) * page_size
    end = start + page_size
    page_df = df.iloc[start:end]

    return TableDataResponse(
        name=table_name,
        columns=list(df.columns),
        data=_sanitize_df_for_json(page_df),
        total_rows=total_rows,
        page=page,
        page_size=page_size,
        has_more=end < total_rows,
    )


@router.get("/{session_id}/tables/{table_name}", response_model=TableDataResponse)
async def get_table_data(
    session_id: str,
    table_name: str,
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(default=100, ge=1, le=1000, description="Rows per page"),
    session_manager: SessionManager = Depends(get_session_manager),
) -> TableDataResponse:
    """Get table data with pagination.

    Args:
        session_id: Session ID
        table_name: Table name to retrieve
        page: Page number (1-indexed)
        page_size: Number of rows per page

    Returns:
        Table data with pagination info

    Raises:
        404: Session or table not found
    """
    managed = session_manager.get_session(session_id)

    if not managed.session.datastore:
        raise HTTPException(status_code=404, detail="No datastore for this session")

    try:
        # Load the full DataFrame
        df = managed.session.datastore.load_dataframe(table_name)
        if df is None:
            raise HTTPException(status_code=404, detail=f"Table not found: {table_name}")

        total_rows = len(df)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size

        # Slice for pagination
        page_df = df.iloc[start_idx:end_idx]

        data = _sanitize_df_for_json(page_df)

        return TableDataResponse(
            name=table_name,
            columns=list(df.columns),
            data=data,
            total_rows=total_rows,
            page=page,
            page_size=page_size,
            has_more=end_idx < total_rows,
        )

    except Exception as e:
        logger.error(f"Error getting table data: {e}")
        raise HTTPException(status_code=404, detail=f"Table not found: {table_name}")


@router.get("/{session_id}/tables/{table_name}/download")
async def download_table(
    session_id: str,
    table_name: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> Response:
    """Download table data as CSV.

    Args:
        session_id: Session ID
        table_name: Table name to download

    Returns:
        CSV file response

    Raises:
        404: Session or table not found
    """
    managed = session_manager.get_session(session_id)

    if not managed.session.datastore:
        raise HTTPException(status_code=404, detail="No datastore for this session")

    try:
        df = managed.session.datastore.load_dataframe(table_name)
        if df is None:
            raise HTTPException(status_code=404, detail=f"Table not found: {table_name}")

        csv_content = df.to_csv(index=False)

        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="{table_name}.csv"'
            },
        )

    except Exception as e:
        logger.error(f"Error downloading table: {e}")
        raise HTTPException(status_code=404, detail=f"Table not found: {table_name}")


@router.post("/{session_id}/tables/{table_name}/star")
async def toggle_table_star(
    session_id: str,
    table_name: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    """Toggle a table's starred status.

    Uses unified logic: toggles between starred and unstarred state.
    Tracks explicit user actions to override auto-star defaults.

    Args:
        session_id: Session ID
        table_name: Table name

    Returns:
        New starred status

    Raises:
        404: Session or table not found
    """
    managed = session_manager.get_session(session_id)

    try:
        if not managed.session.datastore:
            raise HTTPException(status_code=404, detail="No datastore")

        # Verify table exists
        tables = managed.session.datastore.list_tables()
        table_info = next((t for t in tables if t["name"] == table_name), None)
        if not table_info:
            raise HTTPException(status_code=404, detail=f"Table not found: {table_name}")

        # Get current state
        starred_tables = set(managed.session.datastore.get_starred_tables())
        unstarred_tables = set(managed.session.datastore.get_state("_unstarred_tables") or [])

        # Determine current starred state (same logic as list_artifacts)
        is_published = table_info.get("is_published", False)
        is_final_step = table_info.get("is_final_step", False)
        has_data = table_info.get("row_count", 0) > 0

        if table_name in starred_tables:
            current_starred = True
        elif table_name in unstarred_tables:
            current_starred = False
        else:
            # Auto-star logic
            current_starred = is_published or (is_final_step and has_data)

        # Toggle to the opposite state
        new_starred = not current_starred

        # Update starred/unstarred tracking
        if new_starred:
            # Add to starred, remove from unstarred
            starred_tables.add(table_name)
            unstarred_tables.discard(table_name)
        else:
            # Remove from starred, add to unstarred
            starred_tables.discard(table_name)
            unstarred_tables.add(table_name)

        # Persist changes
        managed.session.datastore.set_starred_tables(list(starred_tables))
        managed.session.datastore.set_state("_unstarred_tables", list(unstarred_tables))

        return {"table_name": table_name, "is_starred": new_starred}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error toggling table star: {e}")
        raise HTTPException(status_code=500, detail=str(e))
