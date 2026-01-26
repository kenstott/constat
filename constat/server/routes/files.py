# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""File upload and file reference REST endpoints."""

import base64
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from constat.server.models import (
    FileRefInfo,
    FileRefListResponse,
    FileRefRequest,
    FileUploadRequest,
    UploadedFileInfo,
    UploadedFileListResponse,
)
from constat.server.session_manager import ManagedSession, SessionManager

logger = logging.getLogger(__name__)

router = APIRouter()

# Base directory for uploaded files
UPLOAD_BASE_DIR = Path.home() / ".constat" / "sessions"


def get_session_manager(request: Request) -> SessionManager:
    """Dependency to get session manager from app state."""
    return request.app.state.session_manager


def _get_upload_dir(session_id: str) -> Path:
    """Get the upload directory for a session."""
    upload_dir = UPLOAD_BASE_DIR / session_id / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


def _get_uploaded_files(session_id: str) -> list[dict[str, Any]]:
    """Get list of uploaded files for a session from the metadata file."""
    upload_dir = _get_upload_dir(session_id)
    meta_file = upload_dir / "_metadata.json"

    if not meta_file.exists():
        return []

    import json
    with open(meta_file) as f:
        return json.load(f)


def _save_uploaded_files(session_id: str, files: list[dict[str, Any]]) -> None:
    """Save uploaded files metadata."""
    upload_dir = _get_upload_dir(session_id)
    meta_file = upload_dir / "_metadata.json"

    import json
    with open(meta_file, "w") as f:
        json.dump(files, f, default=str)


# ============================================================================
# File Upload Endpoints
# ============================================================================


@router.post("/{session_id}/files", response_model=UploadedFileInfo)
async def upload_file_multipart(
    session_id: str,
    file: UploadFile = File(...),
    session_manager: SessionManager = Depends(get_session_manager),
) -> UploadedFileInfo:
    """Upload a file via multipart/form-data.

    Args:
        session_id: Session ID
        file: The file to upload

    Returns:
        Information about the uploaded file including file:// URI

    Raises:
        404: Session not found
    """
    managed = session_manager.get_session(session_id)

    # Generate file ID and path
    file_id = f"f_{uuid.uuid4().hex[:12]}"
    upload_dir = _get_upload_dir(session_id)

    # Sanitize filename
    safe_filename = os.path.basename(file.filename or "upload")
    file_path = upload_dir / f"{file_id}_{safe_filename}"

    # Write file
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    # Create file info
    now = datetime.now(timezone.utc)
    file_info = {
        "id": file_id,
        "filename": safe_filename,
        "file_path": str(file_path),
        "file_uri": f"file://{file_path}",
        "size_bytes": len(content),
        "content_type": file.content_type or "application/octet-stream",
        "uploaded_at": now.isoformat(),
    }

    # Update metadata
    files = _get_uploaded_files(session_id)
    files.append(file_info)
    _save_uploaded_files(session_id, files)

    return UploadedFileInfo(
        id=file_id,
        filename=safe_filename,
        file_uri=file_info["file_uri"],
        size_bytes=file_info["size_bytes"],
        content_type=file_info["content_type"],
        uploaded_at=now,
    )


@router.post("/{session_id}/files/data-uri", response_model=UploadedFileInfo)
async def upload_file_data_uri(
    session_id: str,
    body: FileUploadRequest,
    session_manager: SessionManager = Depends(get_session_manager),
) -> UploadedFileInfo:
    """Upload a file via data URI.

    Args:
        session_id: Session ID
        body: File upload request with base64 data

    Returns:
        Information about the uploaded file including file:// URI

    Raises:
        404: Session not found
        400: Invalid data format
    """
    managed = session_manager.get_session(session_id)

    # Parse data URI or raw base64
    data = body.data
    if data.startswith("data:"):
        # Parse data URI: data:mime/type;base64,content
        try:
            _, encoded = data.split(",", 1)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid data URI format")
    else:
        encoded = data

    # Decode base64
    try:
        content = base64.b64decode(encoded)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 data: {e}")

    # Generate file ID and path
    file_id = f"f_{uuid.uuid4().hex[:12]}"
    upload_dir = _get_upload_dir(session_id)

    # Sanitize filename
    safe_filename = os.path.basename(body.filename)
    file_path = upload_dir / f"{file_id}_{safe_filename}"

    # Write file
    with open(file_path, "wb") as f:
        f.write(content)

    # Create file info
    now = datetime.now(timezone.utc)
    file_info = {
        "id": file_id,
        "filename": safe_filename,
        "file_path": str(file_path),
        "file_uri": f"file://{file_path}",
        "size_bytes": len(content),
        "content_type": body.content_type,
        "uploaded_at": now.isoformat(),
    }

    # Update metadata
    files = _get_uploaded_files(session_id)
    files.append(file_info)
    _save_uploaded_files(session_id, files)

    return UploadedFileInfo(
        id=file_id,
        filename=safe_filename,
        file_uri=file_info["file_uri"],
        size_bytes=file_info["size_bytes"],
        content_type=file_info["content_type"],
        uploaded_at=now,
    )


@router.get("/{session_id}/files", response_model=UploadedFileListResponse)
async def list_uploaded_files(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> UploadedFileListResponse:
    """List all uploaded files for a session.

    Args:
        session_id: Session ID

    Returns:
        List of uploaded files

    Raises:
        404: Session not found
    """
    managed = session_manager.get_session(session_id)
    files = _get_uploaded_files(session_id)

    return UploadedFileListResponse(
        files=[
            UploadedFileInfo(
                id=f["id"],
                filename=f["filename"],
                file_uri=f["file_uri"],
                size_bytes=f["size_bytes"],
                content_type=f["content_type"],
                uploaded_at=datetime.fromisoformat(f["uploaded_at"]),
            )
            for f in files
        ]
    )


@router.get("/{session_id}/files/{file_id}")
async def download_file(
    session_id: str,
    file_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> FileResponse:
    """Download an uploaded file.

    Args:
        session_id: Session ID
        file_id: File ID

    Returns:
        File content

    Raises:
        404: Session or file not found
    """
    managed = session_manager.get_session(session_id)
    files = _get_uploaded_files(session_id)

    # Find the file
    file_info = next((f for f in files if f["id"] == file_id), None)
    if not file_info:
        raise HTTPException(status_code=404, detail=f"File not found: {file_id}")

    file_path = Path(file_info["file_path"])
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found on disk: {file_id}")

    return FileResponse(
        path=file_path,
        filename=file_info["filename"],
        media_type=file_info["content_type"],
    )


@router.delete("/{session_id}/files/{file_id}")
async def delete_file(
    session_id: str,
    file_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict:
    """Delete an uploaded file.

    Args:
        session_id: Session ID
        file_id: File ID

    Returns:
        Deletion confirmation

    Raises:
        404: Session or file not found
    """
    managed = session_manager.get_session(session_id)
    files = _get_uploaded_files(session_id)

    # Find and remove the file
    file_info = next((f for f in files if f["id"] == file_id), None)
    if not file_info:
        raise HTTPException(status_code=404, detail=f"File not found: {file_id}")

    # Delete file from disk
    file_path = Path(file_info["file_path"])
    if file_path.exists():
        file_path.unlink()

    # Update metadata
    files = [f for f in files if f["id"] != file_id]
    _save_uploaded_files(session_id, files)

    return {
        "status": "deleted",
        "file_id": file_id,
    }


# ============================================================================
# File Reference Endpoints
# ============================================================================


def _get_file_refs(managed: ManagedSession) -> list[dict[str, Any]]:
    """Get file references from session."""
    # File refs are stored on the session object
    if not hasattr(managed, "_file_refs"):
        managed._file_refs = []
    return managed._file_refs


@router.post("/{session_id}/file-refs", response_model=FileRefInfo)
async def add_file_reference(
    session_id: str,
    body: FileRefRequest,
    session_manager: SessionManager = Depends(get_session_manager),
) -> FileRefInfo:
    """Add a file reference to the session.

    This wraps the Session.add_file() method for adding references
    to external files or URLs.

    Args:
        session_id: Session ID
        body: File reference request

    Returns:
        File reference information

    Raises:
        404: Session not found
    """
    managed = session_manager.get_session(session_id)

    # Call session.add_file if available
    try:
        managed.session.add_file(
            name=body.name,
            uri=body.uri,
            auth=body.auth,
            description=body.description,
        )
    except AttributeError:
        # Session doesn't have add_file method - that's OK
        pass
    except Exception as e:
        logger.warning(f"Error adding file reference: {e}")

    # Track in managed session
    now = datetime.now(timezone.utc)
    file_refs = _get_file_refs(managed)
    file_refs.append({
        "name": body.name,
        "uri": body.uri,
        "has_auth": body.auth is not None,
        "description": body.description,
        "added_at": now.isoformat(),
    })

    return FileRefInfo(
        name=body.name,
        uri=body.uri,
        has_auth=body.auth is not None,
        description=body.description,
        added_at=now,
    )


@router.get("/{session_id}/file-refs", response_model=FileRefListResponse)
async def list_file_references(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> FileRefListResponse:
    """List all file references for a session.

    Args:
        session_id: Session ID

    Returns:
        List of file references

    Raises:
        404: Session not found
    """
    managed = session_manager.get_session(session_id)
    file_refs = _get_file_refs(managed)

    return FileRefListResponse(
        file_refs=[
            FileRefInfo(
                name=ref["name"],
                uri=ref["uri"],
                has_auth=ref["has_auth"],
                description=ref.get("description"),
                added_at=datetime.fromisoformat(ref["added_at"]),
            )
            for ref in file_refs
        ]
    )


@router.delete("/{session_id}/file-refs/{name}")
async def delete_file_reference(
    session_id: str,
    name: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict:
    """Delete a file reference.

    Args:
        session_id: Session ID
        name: File reference name

    Returns:
        Deletion confirmation

    Raises:
        404: Session or file reference not found
    """
    managed = session_manager.get_session(session_id)
    file_refs = _get_file_refs(managed)

    # Find and remove
    original_len = len(file_refs)
    managed._file_refs = [ref for ref in file_refs if ref["name"] != name]

    if len(managed._file_refs) == original_len:
        raise HTTPException(status_code=404, detail=f"File reference not found: {name}")

    return {
        "status": "deleted",
        "name": name,
    }


# ============================================================================
# Document Upload Endpoints (for file picker)
# ============================================================================


@router.post("/{session_id}/documents/upload")
async def upload_documents(
    session_id: str,
    files: list[UploadFile] = File(...),
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict:
    """Upload and index documents from local files.

    Accepts multiple files via multipart form data. Files are saved to the
    session's upload directory and indexed for search (which extracts entities).

    Supported formats: .md, .txt, .pdf, .docx, .html, .htm

    Args:
        session_id: Session ID
        files: List of files to upload

    Returns:
        Upload results including indexed document names

    Raises:
        404: Session not found
        400: No valid documents provided
    """
    managed = session_manager.get_session(session_id)

    # Supported document extensions
    doc_extensions = {'.md', '.txt', '.pdf', '.docx', '.html', '.htm'}

    upload_dir = _get_upload_dir(session_id)
    results = []

    for file in files:
        if not file.filename:
            continue

        # Check extension
        suffix = Path(file.filename).suffix.lower()
        if suffix not in doc_extensions:
            results.append({
                "filename": file.filename,
                "status": "skipped",
                "reason": f"Unsupported format: {suffix}",
            })
            continue

        try:
            # Save file to upload directory
            safe_name = Path(file.filename).name  # Strip any path components
            file_path = upload_dir / safe_name

            # Handle duplicate names
            counter = 1
            original_stem = file_path.stem
            while file_path.exists():
                file_path = upload_dir / f"{original_stem}_{counter}{suffix}"
                counter += 1

            # Write file
            content = await file.read()
            file_path.write_bytes(content)

            # Create a name for the document (filename without extension)
            doc_name = file_path.stem

            # Index the document using session.add_file
            # This triggers entity extraction via doc_tools
            uri = f"file://{file_path}"
            managed.session.add_file(
                name=doc_name,
                uri=uri,
                description=f"Uploaded document: {file.filename}",
            )

            # Also track as a file reference
            now = datetime.now(timezone.utc)
            file_refs = _get_file_refs(managed)
            file_refs.append({
                "name": doc_name,
                "uri": uri,
                "has_auth": False,
                "description": f"Uploaded: {file.filename}",
                "added_at": now.isoformat(),
            })

            results.append({
                "filename": file.filename,
                "name": doc_name,
                "status": "indexed",
                "path": str(file_path),
            })

        except Exception as e:
            logger.error(f"Error uploading {file.filename}: {e}")
            results.append({
                "filename": file.filename,
                "status": "error",
                "reason": str(e),
            })

    if not results:
        raise HTTPException(status_code=400, detail="No files provided")

    indexed_count = sum(1 for r in results if r.get("status") == "indexed")

    return {
        "status": "success",
        "indexed_count": indexed_count,
        "total_files": len(files),
        "results": results,
    }