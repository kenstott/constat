# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Artifact data endpoints."""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from constat.server.models import (
    ArtifactContentResponse,
    ArtifactInfo,
    ArtifactListResponse,
    ArtifactVersionInfo,
    ArtifactVersionsResponse,
)
from constat.server.routes.data import get_session_manager
from constat.server.session_manager import SessionManager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/{session_id}/artifacts", response_model=ArtifactListResponse)
async def list_artifacts(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> ArtifactListResponse:
    """List all artifacts in the session.

    Args:
        session_id: Session ID
        session_manager: Injected session manager

    Returns:
        List of artifacts with metadata

    Raises:
        404: Session not found
    """
    managed = session_manager.get_session(session_id)

    if not managed.session.datastore:
        return ArtifactListResponse(artifacts=[])

    try:
        artifacts = managed.session.datastore.list_artifacts()
        tables = managed.session.datastore.list_tables()

        # Determine which artifacts are key results.
        # Key results: visualizations (unless explicitly unstarred) OR user-starred
        visualization_types = {'chart', 'plotly', 'svg', 'png', 'jpeg', 'html', 'image', 'vega', 'markdown', 'md'}
        # Code types are explicitly excluded from key results (unless user-starred)
        code_types = {'code', 'python', 'sql', 'script', 'text', 'output', 'error'}

        def get_starred_and_key_result(artifact_dict: dict) -> tuple[bool, bool]:
            """Get is_starred and is_key_result for an artifact.

            Uses a unified flag: is_starred is the single source of truth.
            - If user explicitly set is_starred, use that value
            - Otherwise, auto-determine: visualizations are starred by default

            Returns:
                (is_starred, is_key_result) tuple - both will have the same value
            """
            inner_artifact_obj = managed.session.datastore.get_artifact_by_id(artifact_dict["id"])
            metadata = inner_artifact_obj.metadata if inner_artifact_obj else {}
            artifact_type = artifact_dict.get("type", "").lower()

            # starred: single source of truth for starred state
            if "is_starred" in metadata:
                # User has explicitly set starred status - use that
                starred = metadata["is_starred"]
                logger.debug(f"[artifact_key_result] {artifact_dict['name']} type={artifact_type}: is_starred={starred} (from metadata)")
            elif artifact_type in code_types:
                # Code is NEVER starred by default
                starred = False
            elif artifact_type in visualization_types:
                # Visualizations are starred by default
                starred = True
                logger.debug(f"[artifact_key_result] id={artifact_dict['id']} {artifact_dict['name']} type={artifact_type}: is_starred=True (visualization)")
            else:
                starred = False
                logger.debug(f"[artifact_key_result] {artifact_dict['name']} type={artifact_type}: is_starred=False (default)")

            # key_result matches starred (unified behavior)
            key_result = starred

            return starred, key_result

        # Build artifact list
        artifact_list = []
        for artifact_item in artifacts:
            is_starred, is_key_result = get_starred_and_key_result(artifact_item)
            # Get full artifact to access metadata
            full_artifact = managed.session.datastore.get_artifact_by_id(artifact_item["id"])
            artifact_metadata = full_artifact.metadata if full_artifact else None
            artifact_list.append(
                ArtifactInfo(
                    id=artifact_item["id"],
                    name=artifact_item["name"],
                    artifact_type=artifact_item["type"],
                    step_number=artifact_item.get("step_number", 0),
                    title=artifact_item.get("title"),
                    description=artifact_item.get("description"),
                    mime_type=artifact_item.get("content_type") or "application/octet-stream",
                    created_at=artifact_item.get("created_at"),
                    is_key_result=is_key_result,
                    is_starred=is_starred,
                    metadata=artifact_metadata,
                    version=artifact_item.get("version", 1),
                    version_count=artifact_item.get("version_count", 1),
                )
            )

        # Add consequential tables as virtual artifacts
        # A table is consequential if it's published, from the final step, or starred
        if tables:
            starred_tables = set(managed.session.datastore.get_starred_tables())
            # Track tables that user has explicitly unstarred
            unstarred_tables = set(managed.session.datastore.get_state("_unstarred_tables") or [])
            for t in tables:
                table_name = t["name"]
                # Skip internal tables
                if table_name.startswith("_"):
                    continue
                # Check if table is published or from final step (from registry metadata)
                # noinspection DuplicatedCode
                is_published = t.get("is_published", False)
                is_final_step = t.get("is_final_step", False)
                # Tables with substantial data are consequential
                has_data = t.get("row_count", 0) > 0

                # Unified starred logic:
                # - If user explicitly starred it, table_starred=True
                # - If user explicitly unstarred it, table_starred=False
                # - Otherwise, auto-star if published or final step with data
                if table_name in starred_tables:
                    table_starred = True
                elif table_name in unstarred_tables:
                    table_starred = False
                else:
                    # Auto-star tables that are published or from final step with data
                    table_starred = is_published or (is_final_step and has_data)

                # Determine if table should appear in artifacts list
                # Tables appear if starred (including auto-starred) or explicitly starred
                should_include = table_starred or table_name in starred_tables

                if should_include:
                    # Create a virtual artifact entry for this table
                    # Use negative IDs to distinguish from real artifacts
                    virtual_id = -hash(table_name) % 1000000
                    # table_starred and is_key_result are unified (same value)
                    artifact_list.append(
                        ArtifactInfo(
                            id=virtual_id,
                            name=table_name,
                            artifact_type="table",
                            step_number=t.get("step_number", 0),
                            title=f"Table: {table_name}",
                            description=f"{t.get('row_count', 0)} rows",
                            mime_type="application/x-dataframe",
                            created_at=None,
                            is_key_result=table_starred,
                            is_starred=table_starred,
                        )
                    )

        # Log key result artifacts for debugging
        key_results = [a for a in artifact_list if a.is_key_result]
        logger.debug(f"[list_artifacts] Returning {len(artifact_list)} artifacts (including {len([a for a in artifact_list if a.artifact_type == 'table'])} tables)")
        logger.debug(f"[list_artifacts] Key results: {[(a.id, a.name, a.artifact_type, a.is_key_result) for a in key_results]}")
        return ArtifactListResponse(artifacts=artifact_list)
    except Exception as e:
        logger.error(f"Error listing artifacts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}/artifacts/{artifact_id}", response_model=ArtifactContentResponse)
async def get_artifact(
    session_id: str,
    artifact_id: int,
    session_manager: SessionManager = Depends(get_session_manager),
) -> ArtifactContentResponse:
    """Get artifact content by ID.

    Handles both real artifacts and virtual table artifacts.
    Virtual table IDs are computed as: -hash(table_name) % 1000000

    Args:
        session_id: Session ID
        artifact_id: Artifact ID (real artifact ID or virtual table ID)
        session_manager: Injected session manager

    Returns:
        Artifact with content

    Raises:
        404: Session or artifact not found
    """
    managed = session_manager.get_session(session_id)

    if not managed.session.datastore:
        raise HTTPException(status_code=404, detail="No datastore for this session")

    try:
        # First try to get a real artifact
        artifact = managed.session.datastore.get_artifact_by_id(artifact_id)

        if artifact:
            return ArtifactContentResponse(
                id=artifact.id,
                name=artifact.name,
                artifact_type=artifact.artifact_type.value,
                content=artifact.content,
                mime_type=artifact.mime_type,
                is_binary=artifact.is_binary,
            )

        # Not a real artifact - check if it's a virtual table ID
        tables = managed.session.datastore.list_tables()
        for t in tables:
            table_name = t["name"]
            virtual_id = -hash(table_name) % 1000000
            if virtual_id == artifact_id:
                # Found matching table - return as artifact content
                import json
                table_data = managed.session.datastore.get_table_data(table_name)
                if table_data is not None:
                    # Convert DataFrame to JSON for artifact content
                    content = table_data.to_json(orient="records", date_format="iso")
                    return ArtifactContentResponse(
                        id=artifact_id,
                        name=table_name,
                        artifact_type="table",
                        content=content,
                        mime_type="application/json",
                        is_binary=False,
                    )

        raise HTTPException(status_code=404, detail=f"Artifact not found: {artifact_id}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting artifact: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}/artifacts/{artifact_id}/versions", response_model=ArtifactVersionsResponse)
async def get_artifact_versions(
    session_id: str,
    artifact_id: int,
    session_manager: SessionManager = Depends(get_session_manager),
) -> ArtifactVersionsResponse:
    """Get version history for an artifact.

    Args:
        session_id: Session ID
        artifact_id: Artifact ID (used to look up the artifact name)
        session_manager: Injected session manager

    Returns:
        Version history (all versions of the same-named artifact)

    Raises:
        404: Session or artifact not found
    """
    managed = session_manager.get_session(session_id)

    if not managed.session.datastore:
        raise HTTPException(status_code=404, detail="No datastore for this session")

    artifact = managed.session.datastore.get_artifact_by_id(artifact_id)
    if not artifact:
        raise HTTPException(status_code=404, detail=f"Artifact not found: {artifact_id}")

    versions = managed.session.datastore.get_artifact_versions(artifact.name)
    return ArtifactVersionsResponse(
        name=artifact.name,
        current_version=versions[0]["version"] if versions else 1,
        versions=[
            ArtifactVersionInfo(
                id=v["id"],
                version=v["version"],
                step_number=v["step_number"],
                attempt=v["attempt"],
                created_at=v.get("created_at"),
            )
            for v in versions
        ],
    )


@router.get("/{session_id}/artifacts/{artifact_id}/download")
async def download_artifact_file(
    session_id: str,
    artifact_id: int,
    session_manager: SessionManager = Depends(get_session_manager),
) -> Response:
    """Download the original file for an artifact.

    For artifacts that are HTML previews of binary files (xlsx, docx, pdf),
    this returns the original file. For other artifacts, returns the content.

    Args:
        session_id: Session ID
        artifact_id: Artifact ID
        session_manager: Injected session manager

    Returns:
        File response with appropriate content type

    Raises:
        404: Session or artifact not found
    """
    from pathlib import Path

    managed = session_manager.get_session(session_id)

    if not managed.session.datastore:
        raise HTTPException(status_code=404, detail="No datastore for this session")

    try:
        artifact = managed.session.datastore.get_artifact_by_id(artifact_id)

        if not artifact:
            raise HTTPException(status_code=404, detail=f"Artifact not found: {artifact_id}")

        # Check if this artifact has a source file (converted binary like xlsx, docx)
        file_path = artifact.metadata.get("file_path") if artifact.metadata else None
        source_type = artifact.metadata.get("source_type") if artifact.metadata else None

        if file_path and source_type:
            # Return the original binary file
            path = Path(file_path)
            if not path.exists():
                raise HTTPException(status_code=404, detail=f"Original file not found: {path.name}")

            content = path.read_bytes()

            # Determine MIME type
            mime_types = {
                "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "xls": "application/vnd.ms-excel",
                "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "doc": "application/msword",
                "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                "ppt": "application/vnd.ms-powerpoint",
                "pdf": "application/pdf",
            }
            mime_type = mime_types.get(source_type, "application/octet-stream")

            return Response(
                content=content,
                media_type=mime_type,
                headers={
                    "Content-Disposition": f'attachment; filename="{path.name}"',
                },
            )

        # No source file - return the artifact content directly
        content = artifact.content.encode() if isinstance(artifact.content, str) else artifact.content

        # Determine extension from artifact type
        ext_map = {
            "markdown": "md",
            "md": "md",
            "html": "html",
            "json": "json",
            "text": "txt",
        }
        ext = ext_map.get(artifact.artifact_type.value if hasattr(artifact.artifact_type, 'value') else artifact.artifact_type, "txt")
        filename = f"{artifact.name}.{ext}"

        return Response(
            content=content,
            media_type=artifact.mime_type or "text/plain",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading artifact: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{session_id}/artifacts/{artifact_id}/star")
async def toggle_artifact_star(
    session_id: str,
    artifact_id: int,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    """Toggle an artifact's starred/key-result status.

    Args:
        session_id: Session ID
        artifact_id: Artifact ID
        session_manager: Injected session manager

    Returns:
        New starred status

    Raises:
        404: Session or artifact not found
    """
    managed = session_manager.get_session(session_id)

    try:
        if not managed.session.datastore:
            raise HTTPException(status_code=404, detail="No datastore")

        artifact = managed.session.datastore.get_artifact_by_id(artifact_id)
        if not artifact:
            raise HTTPException(status_code=404, detail=f"Artifact not found: {artifact_id}")

        # Toggle is_starred in metadata
        current = artifact.metadata.get("is_starred", False)
        new_status = not current
        managed.session.datastore.update_artifact_metadata(artifact_id, {"is_starred": new_status})

        return {"artifact_id": artifact_id, "is_starred": new_status}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error toggling artifact star: {e}")
        raise HTTPException(status_code=500, detail=str(e))
