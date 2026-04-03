# Copyright (c) 2025 Kenneth Stott
# Canary: ce0e2b18-ef3c-45ba-98c1-0ab31ee2d1e6
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Session history and artifact storage for review, debugging, and resumption."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Artifact:
    """A single artifact from a query execution."""
    id: str
    query_id: int
    artifact_type: str  # code, output, error, tool_call
    content: str
    attempt: int = 1
    metadata: dict = field(default_factory=dict)


@dataclass
class QueryRecord:
    """Record of a single query execution."""
    query_id: int
    timestamp: str
    question: str
    success: bool
    attempts: int
    duration_ms: int
    answer: Optional[str] = None
    error: Optional[str] = None


@dataclass
class SessionSummary:
    """Summary info for listing sessions."""
    session_id: str
    created_at: str
    databases: list[str]
    status: str
    total_queries: int
    total_duration_ms: int
    user_id: Optional[str] = None
    summary: Optional[str] = None  # Brief description (usually first query)
    apis: Optional[list[str]] = None  # API names used
    documents: Optional[list[str]] = None  # Document names used
    files: Optional[list[str]] = None  # Session-added file names
    server_session_id: Optional[str] = None  # Server-side session ID for deduplication

    def __post_init__(self):
        """Initialize optional lists."""
        if self.apis is None:
            self.apis = []
        if self.documents is None:
            self.documents = []
        if self.files is None:
            self.files = []


@dataclass
class SessionDetail:
    """Full session detail including queries."""
    session_id: str
    created_at: str
    config_hash: str
    databases: list[str]
    status: str
    total_queries: int
    total_duration_ms: int
    queries: list[QueryRecord]
    summary: Optional[str] = None
    user_id: Optional[str] = None


from constat.storage.history_files import FileSessionHistory

# Backward-compatible alias
SessionHistory = FileSessionHistory


def create_session_history(
    storage_dir: Optional[Path] = None,
    user_id: Optional[str] = None,
    db=None,
) -> FileSessionHistory:
    """Factory for session history backends.

    Args:
        storage_dir: Base directory for file-based storage.
        user_id: User ID for scoped storage.
        db: DuckDB connection. When provided, returns DuckDBSessionHistory.

    Returns:
        A session history instance.
    """
    if db is not None:
        # TODO: implement DuckDBSessionHistory in history_duckdb.py
        raise NotImplementedError("DuckDB session history backend not yet available")
    return FileSessionHistory(storage_dir=storage_dir, user_id=user_id)
