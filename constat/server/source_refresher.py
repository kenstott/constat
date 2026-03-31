# Copyright (c) 2025 Kenneth Stott
# Canary: f4573fe1-5994-4c2f-858c-9f24438294b6
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Background source refresh for IMAP, HTTP, and file document sources."""

import asyncio
import hashlib
import logging
import os
import urllib.request
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from constat.server.session_manager import ManagedSession, SessionManager

logger = logging.getLogger(__name__)


def _needs_refresh(file_ref: dict, default_interval: int) -> bool:
    """Check whether a file_ref is due for refresh.

    Args:
        file_ref: File reference dict from ManagedSession._file_refs
        default_interval: Default refresh interval in seconds

    Returns:
        True if the source should be refreshed
    """
    doc_config = file_ref.get("document_config")
    if not doc_config:
        return False

    # Check auto_refresh flag
    if not doc_config.get("auto_refresh", True):
        return False

    last_refreshed = file_ref.get("last_refreshed")
    if not last_refreshed:
        return True

    interval = doc_config.get("refresh_interval") or default_interval
    last_dt = datetime.fromisoformat(last_refreshed)
    elapsed = (datetime.now(timezone.utc) - last_dt).total_seconds()
    return elapsed >= interval


def _refresh_imap_source(
    managed: "ManagedSession", file_ref: dict, name: str
) -> tuple[bool, str, int]:
    """Refresh an IMAP email source by re-fetching since last_refreshed.

    Args:
        managed: The managed session
        file_ref: File reference dict
        name: Document name

    Returns:
        (success, message, new_chunk_count)
    """
    from constat.core.config import DocumentConfig

    session = managed.session
    doc_config_data = dict(file_ref["document_config"])

    # Set since to last_refreshed to get only new messages
    last_refreshed = file_ref.get("last_refreshed")
    if last_refreshed:
        doc_config_data["since"] = last_refreshed

    logger.info("[IMAP_REFRESH] Refreshing %s (since=%s, mailbox=%s)",
                name, last_refreshed, doc_config_data.get("mailbox", "INBOX"))

    doc_config = DocumentConfig(**doc_config_data)

    # Delete old chunks for this document
    if session.doc_tools and session.doc_tools._vector_store:
        session.doc_tools._vector_store.delete_resource_chunks(
            managed.session_id, "document", name
        )

    success, msg = session.doc_tools.add_document_from_config(
        name, doc_config, session_id=managed.session_id,
    )

    # Count chunks (approximate from message)
    new_chunks = 0
    if success and msg:
        # Try to extract chunk count from message
        import re
        m = re.search(r"(\d+)\s*chunk", msg)
        if m:
            new_chunks = int(m.group(1))

    logger.info("[IMAP_REFRESH] %s: success=%s chunks=%d msg=%s",
                name, success, new_chunks, msg)
    return success, msg, new_chunks


def _refresh_http_source(
    managed: "ManagedSession", file_ref: dict, name: str
) -> tuple[bool, str, int]:
    """Refresh an HTTP document source using content hash comparison.

    Args:
        managed: The managed session
        file_ref: File reference dict
        name: Document name

    Returns:
        (success, message, new_chunk_count)
    """
    from constat.core.config import DocumentConfig

    session = managed.session
    doc_config_data = dict(file_ref["document_config"])
    url = doc_config_data.get("url", "")

    # Fetch content and compute hash
    req = urllib.request.Request(url)
    for key, val in doc_config_data.get("headers", {}).items():
        req.add_header(key, val)

    with urllib.request.urlopen(req, timeout=30) as resp:
        content = resp.read()

    new_hash = hashlib.sha256(content).hexdigest()
    old_hash = file_ref.get("content_hash")

    if new_hash == old_hash:
        return True, "Content unchanged", 0

    # Content changed — delete old chunks and re-index
    doc_config = DocumentConfig(**doc_config_data)

    if session.doc_tools and session.doc_tools._vector_store:
        session.doc_tools._vector_store.delete_resource_chunks(
            managed.session_id, "document", name
        )

    success, msg = session.doc_tools.add_document_from_config(
        name, doc_config, session_id=managed.session_id,
    )

    if success:
        file_ref["content_hash"] = new_hash

    new_chunks = 0
    if success and msg:
        import re
        m = re.search(r"(\d+)\s*chunk", msg)
        if m:
            new_chunks = int(m.group(1))

    return success, msg, new_chunks


def _refresh_file_source(
    managed: "ManagedSession", file_ref: dict, name: str
) -> tuple[bool, str, int]:
    """Refresh a local file source using mtime+size comparison.

    Args:
        managed: The managed session
        file_ref: File reference dict
        name: Document name

    Returns:
        (success, message, new_chunk_count)
    """
    from constat.core.config import DocumentConfig

    session = managed.session
    doc_config_data = dict(file_ref["document_config"])
    path = doc_config_data.get("path", "")

    if not path or not os.path.exists(path):
        return False, f"File not found: {path}", 0

    stat = os.stat(path)
    new_stat_key = f"{stat.st_mtime}:{stat.st_size}"
    old_stat_key = file_ref.get("file_stat_key")

    if new_stat_key == old_stat_key:
        return True, "File unchanged", 0

    # File changed — delete old chunks and re-index
    doc_config = DocumentConfig(**doc_config_data)

    if session.doc_tools and session.doc_tools._vector_store:
        session.doc_tools._vector_store.delete_resource_chunks(
            managed.session_id, "document", name
        )

    success, msg = session.doc_tools.add_document_from_config(
        name, doc_config, session_id=managed.session_id,
    )

    if success:
        file_ref["file_stat_key"] = new_stat_key

    new_chunks = 0
    if success and msg:
        import re
        m = re.search(r"(\d+)\s*chunk", msg)
        if m:
            new_chunks = int(m.group(1))

    return success, msg, new_chunks


def _classify_source(file_ref: dict) -> str | None:
    """Classify a file_ref as 'imap', 'http', 'file', or None (unsupported)."""
    doc_config = file_ref.get("document_config")
    if not doc_config:
        return None

    url = doc_config.get("url", "")
    path = doc_config.get("path", "")

    if url:
        lower = url.lower()
        if lower.startswith("imap://") or lower.startswith("imaps://"):
            return "imap"
        if lower.startswith("http://") or lower.startswith("https://"):
            return "http"
    if path:
        return "file"

    return None


def refresh_session_sources(
    managed: "ManagedSession",
    sm: "SessionManager",
    interval: int,
) -> int:
    """Refresh all eligible sources for a session.

    Args:
        managed: The managed session
        sm: Session manager (for pushing events)
        interval: Default refresh interval in seconds

    Returns:
        Number of sources refreshed
    """
    from constat.server.models import EventType

    refreshed = 0
    session = managed.session

    if not session or not hasattr(session, "doc_tools") or not session.doc_tools:
        return 0

    for file_ref in managed._file_refs:
        name = file_ref.get("name", "")

        if not _needs_refresh(file_ref, interval):
            continue

        source_type = _classify_source(file_ref)
        if not source_type:
            continue

        try:
            if source_type == "imap":
                success, msg, count = _refresh_imap_source(managed, file_ref, name)
            elif source_type == "http":
                success, msg, count = _refresh_http_source(managed, file_ref, name)
            elif source_type == "file":
                success, msg, count = _refresh_file_source(managed, file_ref, name)
            else:
                continue

            now = datetime.now(timezone.utc).isoformat()
            file_ref["last_refreshed"] = now

            if success:
                refreshed += 1
                if count > 0:
                    logger.info(f"Refreshed {name}: {count} new chunks")
                    sm._push_event(managed, EventType.SOURCE_REFRESH_COMPLETE, {
                        "session_id": managed.session_id,
                        "source": name,
                        "new_chunks": count,
                        "message": msg,
                    })
            else:
                logger.warning(f"Failed to refresh {name}: {msg}")
                sm._push_event(managed, EventType.SOURCE_REFRESH_ERROR, {
                    "session_id": managed.session_id,
                    "source": name,
                    "error": msg,
                })

        except Exception as e:
            logger.exception(f"Error refreshing source {name}: {e}")
            sm._push_event(managed, EventType.SOURCE_REFRESH_ERROR, {
                "session_id": managed.session_id,
                "source": name,
                "error": str(e),
            })

    # Persist updated timestamps
    if refreshed > 0:
        managed.save_resources()

    return refreshed


async def source_refresh_loop(sm: "SessionManager", interval: int) -> None:
    """Async loop that periodically refreshes sources across all sessions.

    Args:
        sm: Session manager
        interval: Refresh interval in seconds
    """
    first_run = True
    while True:
        try:
            if first_run:
                # Short delay on first run to let sessions initialize
                await asyncio.sleep(30)
                first_run = False
            else:
                await asyncio.sleep(interval)

            sessions = sm.list_sessions()
            for managed in sessions:
                if not managed._file_refs:
                    continue
                try:
                    await asyncio.to_thread(
                        refresh_session_sources, managed, sm, interval
                    )
                except Exception as e:
                    logger.error(f"Source refresh error for {managed.session_id}: {e}")

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Source refresh loop error: {e}")
