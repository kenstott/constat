# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Public read-only routes for shared sessions (no auth required)."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from constat.server.models import (
    ArtifactContentResponse,
    ArtifactInfo,
    ArtifactListResponse,
    TableDataResponse,
    TableInfo,
    TableListResponse,
)
from constat.server.session_manager import SessionManager, ManagedSession

logger = logging.getLogger(__name__)

router = APIRouter()


def get_session_manager(request: Request) -> SessionManager:
    """Dependency to get session manager from app state."""
    return request.app.state.session_manager


def _get_public_session(
    session_id: str,
    session_manager: SessionManager,
) -> ManagedSession:
    """Validate session exists and is public. Returns 404 otherwise (no info leakage)."""
    managed = session_manager.get_session_or_none(session_id)
    if not managed:
        raise HTTPException(status_code=404, detail="Not found")

    if not managed.session.datastore or not managed.session.datastore.is_public():
        raise HTTPException(status_code=404, detail="Not found")

    return managed


@router.get("/{session_id}")
async def public_session_summary(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict:
    """Get public session summary."""
    managed = _get_public_session(session_id, session_manager)

    summary = None
    try:
        summary = managed.session.datastore.get_session_meta("summary")
    except (KeyError, ValueError, OSError):
        pass

    return {
        "session_id": managed.session_id,
        "summary": summary,
        "status": managed.status.value if hasattr(managed.status, 'value') else str(managed.status),
    }


@router.get("/{session_id}/messages")
async def public_messages(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict:
    """Get conversation messages for a public session."""
    managed = _get_public_session(session_id, session_manager)

    from constat.storage.history import SessionHistory
    history = SessionHistory(user_id=managed.user_id or "default")
    messages = history.load_messages_by_server_id(session_id)

    return {"messages": messages}


@router.get("/{session_id}/artifacts", response_model=ArtifactListResponse)
async def public_artifacts(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> ArtifactListResponse:
    """List artifacts for a public session."""
    managed = _get_public_session(session_id, session_manager)

    if not managed.session.datastore:
        return ArtifactListResponse(artifacts=[])

    artifacts = managed.session.datastore.list_artifacts()
    artifact_list = []
    for a in artifacts:
        full = managed.session.datastore.get_artifact_by_id(a["id"])
        metadata = full.metadata if full else None
        artifact_list.append(
            ArtifactInfo(
                id=a["id"],
                name=a["name"],
                artifact_type=a["type"],
                step_number=a.get("step_number", 0),
                title=a.get("title"),
                description=a.get("description"),
                mime_type=a.get("content_type") or "application/octet-stream",
                created_at=a.get("created_at"),
                is_key_result=False,
                is_starred=False,
                metadata=metadata,
                version=a.get("version", 1),
                version_count=a.get("version_count", 1),
            )
        )

    return ArtifactListResponse(artifacts=artifact_list)


@router.get("/{session_id}/artifacts/{artifact_id}", response_model=ArtifactContentResponse)
async def public_artifact_content(
    session_id: str,
    artifact_id: int,
    session_manager: SessionManager = Depends(get_session_manager),
) -> ArtifactContentResponse:
    """Get artifact content for a public session."""
    managed = _get_public_session(session_id, session_manager)

    if not managed.session.datastore:
        raise HTTPException(status_code=404, detail="Not found")

    artifact = managed.session.datastore.get_artifact_by_id(artifact_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Not found")

    return ArtifactContentResponse(
        id=artifact.id,
        name=artifact.name,
        artifact_type=artifact.artifact_type.value,
        content=artifact.content,
        mime_type=artifact.mime_type,
        is_binary=artifact.is_binary,
    )


@router.get("/{session_id}/tables", response_model=TableListResponse)
async def public_tables(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> TableListResponse:
    """List tables for a public session."""
    managed = _get_public_session(session_id, session_manager)

    if not managed.session.datastore:
        return TableListResponse(tables=[])

    tables = managed.session.datastore.list_tables()
    table_list = []
    for t in tables:
        name = t["name"]
        if name.startswith("_"):
            continue
        schema = managed.session.datastore.get_table_schema(name)
        columns = [c["name"] for c in schema] if schema else []
        table_list.append(
            TableInfo(
                name=name,
                row_count=t.get("row_count", 0),
                step_number=t.get("step_number", 0),
                columns=columns,
                is_starred=False,
                version=t.get("version", 1),
                version_count=t.get("version_count", 1),
            )
        )

    return TableListResponse(tables=table_list)


@router.get("/{session_id}/tables/{table_name}")
async def public_table_data(
    session_id: str,
    table_name: str,
    page: int = 1,
    page_size: int = 100,
    session_manager: SessionManager = Depends(get_session_manager),
) -> TableDataResponse:
    """Get table data for a public session."""
    managed = _get_public_session(session_id, session_manager)

    if not managed.session.datastore:
        raise HTTPException(status_code=404, detail="Not found")

    df = managed.session.datastore.load_dataframe(table_name)
    if df is None:
        raise HTTPException(status_code=404, detail="Not found")

    total_rows = len(df)
    start = (page - 1) * page_size
    end = start + page_size
    page_df = df.iloc[start:end]

    return TableDataResponse(
        name=table_name,
        columns=list(page_df.columns),
        data=page_df.to_dict(orient="records"),
        total_rows=total_rows,
        page=page,
        page_size=page_size,
        has_more=end < total_rows,
    )


@router.get("/{session_id}/proof-facts")
async def public_proof_facts(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict:
    """Get proof facts for a public session."""
    managed = _get_public_session(session_id, session_manager)

    from constat.storage.history import SessionHistory
    history = SessionHistory(user_id=managed.user_id or "default")
    facts, summary = history.load_proof_facts_by_server_id(session_id)

    return {"facts": facts, "summary": summary}
