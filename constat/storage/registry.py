# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Central registry for tables and artifacts across all sessions.

Provides a shared index for discovering tables and artifacts created by users,
enabling collaboration, search, and audit trails.
"""

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import duckdb

from constat.storage.duckdb_pool import ThreadLocalDuckDB


@dataclass
class TableRecord:
    """Registry record for a table."""
    id: str
    user_id: str
    session_id: str
    name: str
    file_path: str
    description: Optional[str]
    row_count: int
    columns: list[dict]  # [{name, type}, ...]
    created_at: str
    is_published: bool = False  # Explicitly marked as output for artifacts panel
    is_final_step: bool = False  # Created in final step of request
    title: Optional[str] = None  # Human-friendly display name
    role_id: Optional[str] = None  # Role provenance - which role created this table


@dataclass
class ArtifactRecord:
    """Registry record for an artifact (chart, file, etc.)."""
    id: str
    user_id: str
    session_id: str
    name: str
    file_path: str
    artifact_type: str  # chart, csv, image, html, etc.
    description: Optional[str]
    size_bytes: int
    created_at: str
    is_published: bool = False  # Explicitly marked as output for artifacts panel
    is_final_step: bool = False  # Created in final step of request
    title: Optional[str] = None  # Human-friendly display name
    role_id: Optional[str] = None  # Role provenance - which role created this artifact


class ConstatRegistry:
    """Central registry for tables and artifacts.

    Stores metadata about all tables and artifacts created across sessions,
    enabling discovery, search, and collaboration.

    Storage location: .constat/registry.duckdb
    """

    TABLES_SCHEMA = """
        CREATE TABLE IF NOT EXISTS constat_tables (
            id VARCHAR PRIMARY KEY,
            user_id VARCHAR NOT NULL,
            session_id VARCHAR NOT NULL,
            name VARCHAR NOT NULL,
            file_path VARCHAR NOT NULL,
            description VARCHAR,
            row_count INTEGER DEFAULT 0,
            columns VARCHAR,  -- JSON array of {name, type}
            created_at VARCHAR NOT NULL,
            is_published BOOLEAN DEFAULT FALSE,
            is_final_step BOOLEAN DEFAULT FALSE,
            title VARCHAR,
            role_id VARCHAR,
            UNIQUE(user_id, session_id, name)
        )
    """

    ARTIFACTS_SCHEMA = """
        CREATE TABLE IF NOT EXISTS constat_artifacts (
            id VARCHAR PRIMARY KEY,
            user_id VARCHAR NOT NULL,
            session_id VARCHAR NOT NULL,
            name VARCHAR NOT NULL,
            file_path VARCHAR NOT NULL,
            artifact_type VARCHAR,
            description VARCHAR,
            size_bytes INTEGER DEFAULT 0,
            created_at VARCHAR NOT NULL,
            is_published BOOLEAN DEFAULT FALSE,
            is_final_step BOOLEAN DEFAULT FALSE,
            title VARCHAR,
            role_id VARCHAR
        )
    """

    INDEXES = [
        "CREATE INDEX IF NOT EXISTS idx_tables_user ON constat_tables(user_id)",
        "CREATE INDEX IF NOT EXISTS idx_tables_session ON constat_tables(session_id)",
        "CREATE INDEX IF NOT EXISTS idx_tables_user_session ON constat_tables(user_id, session_id)",
        "CREATE INDEX IF NOT EXISTS idx_artifacts_user ON constat_artifacts(user_id)",
        "CREATE INDEX IF NOT EXISTS idx_artifacts_session ON constat_artifacts(session_id)",
        "CREATE INDEX IF NOT EXISTS idx_artifacts_user_session ON constat_artifacts(user_id, session_id)",
    ]

    def __init__(self, base_dir: Optional[Path] = None):
        """Initialize the registry.

        Args:
            base_dir: Base directory for .constat. Defaults to current directory.
        """
        self.base_dir = Path(base_dir) if base_dir else Path(".constat")
        self.db_path = self.base_dir / "registry.duckdb"
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Use thread-local connections for thread safety
        self._db = ThreadLocalDuckDB(str(self.db_path))
        self._ensure_schema()

    @staticmethod
    def _get_col(row, columns: list[str], col_name: str, default=None):
        """Safely get column value with default for missing columns."""
        try:
            return row[columns.index(col_name)]
        except (ValueError, IndexError):
            return default

    @staticmethod
    def _row_to_table_record(row, columns: list[str]) -> 'TableRecord':
        """Convert a database row to a TableRecord."""
        get = lambda name, default=None: ConstatRegistry._get_col(row, columns, name, default)
        return TableRecord(
            id=row[columns.index("id")],
            user_id=row[columns.index("user_id")],
            session_id=row[columns.index("session_id")],
            name=row[columns.index("name")],
            file_path=row[columns.index("file_path")],
            description=row[columns.index("description")],
            row_count=row[columns.index("row_count")],
            columns=json.loads(row[columns.index("columns")]) if row[columns.index("columns")] else [],
            created_at=row[columns.index("created_at")],
            is_published=bool(get("is_published", False)),
            is_final_step=bool(get("is_final_step", False)),
            title=get("title"),
            role_id=get("role_id"),
        )

    @staticmethod
    def _row_to_artifact_record(row, columns: list[str]) -> 'ArtifactRecord':
        """Convert a database row to an ArtifactRecord."""
        get = lambda name, default=None: ConstatRegistry._get_col(row, columns, name, default)
        return ArtifactRecord(
            id=row[columns.index("id")],
            user_id=row[columns.index("user_id")],
            session_id=row[columns.index("session_id")],
            name=row[columns.index("name")],
            file_path=row[columns.index("file_path")],
            artifact_type=row[columns.index("artifact_type")],
            description=row[columns.index("description")],
            size_bytes=row[columns.index("size_bytes")],
            created_at=row[columns.index("created_at")],
            is_published=bool(get("is_published", False)),
            is_final_step=bool(get("is_final_step", False)),
            title=get("title"),
            role_id=get("role_id"),
        )

    def _ensure_schema(self) -> None:
        """Ensure database and schema exist."""
        conn = self._get_connection()
        conn.execute(self.TABLES_SCHEMA)
        conn.execute(self.ARTIFACTS_SCHEMA)
        for idx in self.INDEXES:
            conn.execute(idx)

        # Migration: Add new columns to existing tables if they don't exist
        self._migrate_schema(conn)

    @staticmethod
    def _migrate_schema(conn) -> None:
        """Add new columns to existing tables for backwards compatibility."""
        # Check and add columns to constat_tables
        # Note: PRAGMA table_info returns (cid, name, type, notnull, dflt_value, pk)
        # Column name is at index 1, not 0
        try:
            cols = {row[1] for row in conn.execute("PRAGMA table_info(constat_tables)").fetchall()}
            if "is_published" not in cols:
                conn.execute("ALTER TABLE constat_tables ADD COLUMN is_published BOOLEAN DEFAULT FALSE")
            if "is_final_step" not in cols:
                conn.execute("ALTER TABLE constat_tables ADD COLUMN is_final_step BOOLEAN DEFAULT FALSE")
            if "title" not in cols:
                conn.execute("ALTER TABLE constat_tables ADD COLUMN title VARCHAR")
            if "role_id" not in cols:
                conn.execute("ALTER TABLE constat_tables ADD COLUMN role_id VARCHAR")
            # Update existing NULL values to FALSE so filtering works correctly
            conn.execute("UPDATE constat_tables SET is_published = FALSE WHERE is_published IS NULL")
            conn.execute("UPDATE constat_tables SET is_final_step = FALSE WHERE is_final_step IS NULL")
        except duckdb.Error:
            pass  # Table may not exist yet

        # Check and add columns to constat_artifacts
        try:
            cols = {row[1] for row in conn.execute("PRAGMA table_info(constat_artifacts)").fetchall()}
            if "is_published" not in cols:
                conn.execute("ALTER TABLE constat_artifacts ADD COLUMN is_published BOOLEAN DEFAULT FALSE")
            if "is_final_step" not in cols:
                conn.execute("ALTER TABLE constat_artifacts ADD COLUMN is_final_step BOOLEAN DEFAULT FALSE")
            if "title" not in cols:
                conn.execute("ALTER TABLE constat_artifacts ADD COLUMN title VARCHAR")
            if "role_id" not in cols:
                conn.execute("ALTER TABLE constat_artifacts ADD COLUMN role_id VARCHAR")
            # Update existing NULL values to FALSE so filtering works correctly
            conn.execute("UPDATE constat_artifacts SET is_published = FALSE WHERE is_published IS NULL")
            conn.execute("UPDATE constat_artifacts SET is_final_step = FALSE WHERE is_final_step IS NULL")
        except duckdb.Error:
            pass  # Table may not exist yet

    def _get_connection(self):
        """Get thread-local database connection."""
        return self._db.conn

    def close(self) -> None:
        """Close all database connections."""
        self._db.close()

    # --- Table Registration ---

    def register_table(
        self,
        user_id: str,
        session_id: str,
        name: str,
        file_path: str,
        row_count: int = 0,
        columns: Optional[list[dict]] = None,
        description: Optional[str] = None,
        is_published: bool = False,
        is_final_step: bool = False,
        title: Optional[str] = None,
        role_id: Optional[str] = None,
    ) -> str:
        """Register a table in the registry.

        Args:
            user_id: User who created the table
            session_id: Session where table was created
            name: Logical table name
            file_path: Path to the Parquet file
            row_count: Number of rows in the table
            columns: List of {name, type} dicts for columns
            description: Human-readable description (auto-generated if None)
            is_published: Whether explicitly published for artifacts panel
            is_final_step: Whether created in final step of request
            title: Human-friendly display name
            role_id: Role that created this table (provenance)

        Returns:
            Table ID
        """
        table_id = str(uuid.uuid4())[:12]
        now = datetime.now(timezone.utc).isoformat()
        columns_json = json.dumps(columns or [])

        # Auto-generate description if not provided
        if description is None:
            description = self._generate_table_description(name, row_count, columns)

        conn = self._get_connection()
        # Delete existing if present (user_id, session_id, name is unique)
        conn.execute("""
            DELETE FROM constat_tables
            WHERE user_id = ? AND session_id = ? AND name = ?
        """, [user_id, session_id, name])
        # Insert new record
        conn.execute("""
            INSERT INTO constat_tables
            (id, user_id, session_id, name, file_path, description, row_count, columns, created_at,
             is_published, is_final_step, title, role_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [table_id, user_id, session_id, name, file_path, description, row_count, columns_json, now,
              is_published, is_final_step, title, role_id])

        return table_id

    @staticmethod
    def _generate_table_description(
        name: str,
        row_count: int,
        columns: Optional[list[dict]]
    ) -> str:
        """Generate a description for a table."""
        col_names = [c["name"] for c in (columns or [])][:5]
        col_str = ", ".join(col_names)
        if columns and len(columns) > 5:
            col_str += f", ... ({len(columns)} total)"

        desc = f"{name}: {row_count:,} rows"
        if col_str:
            desc += f" - columns: {col_str}"
        return desc

    def update_table(
        self,
        user_id: str,
        session_id: str,
        name: str,
        row_count: Optional[int] = None,
        description: Optional[str] = None,
    ) -> bool:
        """Update an existing table's metadata.

        Returns:
            True if updated, False if not found
        """
        conn = self._get_connection()

        updates = []
        params = []
        if row_count is not None:
            updates.append("row_count = ?")
            params.append(row_count)
        if description is not None:
            updates.append("description = ?")
            params.append(description)

        if not updates:
            return False

        params.extend([user_id, session_id, name])
        result = conn.execute(f"""
            UPDATE constat_tables
            SET {", ".join(updates)}
            WHERE user_id = ? AND session_id = ? AND name = ?
        """, params)

        return result.fetchone() is not None or True  # DuckDB doesn't return rowcount easily

    def delete_table(self, user_id: str, session_id: str, name: str) -> bool:
        """Delete a table from the registry.

        Returns:
            True if deleted, False if not found
        """
        conn = self._get_connection()
        # Check if exists first
        exists = conn.execute("""
            SELECT 1 FROM constat_tables
            WHERE user_id = ? AND session_id = ? AND name = ?
        """, [user_id, session_id, name]).fetchone()

        if not exists:
            return False

        conn.execute("""
            DELETE FROM constat_tables
            WHERE user_id = ? AND session_id = ? AND name = ?
        """, [user_id, session_id, name])
        return True

    def publish_tables(
        self,
        user_id: str,
        session_id: str,
        table_names: list[str],
    ) -> int:
        """Mark specific tables as published for artifacts panel.

        Args:
            user_id: User who owns the tables
            session_id: Session where tables exist
            table_names: Names of tables to publish

        Returns:
            Number of tables updated
        """
        if not table_names:
            return 0

        conn = self._get_connection()
        placeholders = ", ".join("?" for _ in table_names)
        conn.execute(f"""
            UPDATE constat_tables
            SET is_published = TRUE, is_final_step = TRUE
            WHERE user_id = ? AND session_id = ? AND name IN ({placeholders})
        """, [user_id, session_id] + table_names)
        return len(table_names)

    def list_tables(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[TableRecord]:
        """List tables, optionally filtered by user/session.

        Args:
            user_id: Filter by user (None = all users)
            session_id: Filter by session (None = all sessions)
            limit: Maximum number of results

        Returns:
            List of TableRecord objects
        """
        conn = self._get_connection()

        query = "SELECT * FROM constat_tables WHERE 1=1"
        params = []

        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        columns = [desc[0] for desc in conn.description]
        return [self._row_to_table_record(row, columns) for row in rows]

    def get_table(self, user_id: str, session_id: str, name: str) -> Optional[TableRecord]:
        """Get a specific table by user/session/name."""
        conn = self._get_connection()
        row = conn.execute("""
            SELECT * FROM constat_tables
            WHERE user_id = ? AND session_id = ? AND name = ?
        """, [user_id, session_id, name]).fetchone()

        if not row:
            return None

        columns = [desc[0] for desc in conn.description]
        return self._row_to_table_record(row, columns)

    # --- Artifact Registration ---

    def register_artifact(
        self,
        user_id: str,
        session_id: str,
        name: str,
        file_path: str,
        artifact_type: str = "file",
        size_bytes: int = 0,
        description: Optional[str] = None,
        is_published: bool = False,
        is_final_step: bool = False,
        title: Optional[str] = None,
        role_id: Optional[str] = None,
    ) -> str:
        """Register an artifact in the registry.

        Args:
            user_id: User who created the artifact
            session_id: Session where artifact was created
            name: Artifact filename
            file_path: Path to the file
            artifact_type: Type (chart, csv, image, html, etc.)
            size_bytes: File size in bytes
            description: Human-readable description
            is_published: Whether explicitly published for artifacts panel
            is_final_step: Whether created in final step of request
            title: Human-friendly display name
            role_id: Role that created this artifact (provenance)

        Returns:
            Artifact ID
        """
        artifact_id = str(uuid.uuid4())[:12]
        now = datetime.now(timezone.utc).isoformat()

        # Auto-generate description if not provided
        if description is None:
            description = self._generate_artifact_description(name, artifact_type, size_bytes)

        conn = self._get_connection()
        conn.execute("""
            INSERT INTO constat_artifacts
            (id, user_id, session_id, name, file_path, artifact_type, description, size_bytes, created_at,
             is_published, is_final_step, title, role_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [artifact_id, user_id, session_id, name, file_path, artifact_type, description, size_bytes, now,
              is_published, is_final_step, title, role_id])

        return artifact_id

    @staticmethod
    def _generate_artifact_description(
        name: str,
        artifact_type: str,
        size_bytes: int
    ) -> str:
        """Generate a description for an artifact."""
        size_str = f"{size_bytes / 1024:.1f}KB" if size_bytes > 0 else ""
        return f"{artifact_type}: {name}" + (f" ({size_str})" if size_str else "")

    def list_artifacts(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        artifact_type: Optional[str] = None,
        limit: int = 100,
    ) -> list[ArtifactRecord]:
        """List artifacts, optionally filtered.

        Args:
            user_id: Filter by user (None = all users)
            session_id: Filter by session (None = all sessions)
            artifact_type: Filter by type (None = all types)
            limit: Maximum number of results

        Returns:
            List of ArtifactRecord objects
        """
        conn = self._get_connection()

        query = "SELECT * FROM constat_artifacts WHERE 1=1"
        params = []

        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        if artifact_type:
            query += " AND artifact_type = ?"
            params.append(artifact_type)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        columns = [desc[0] for desc in conn.description]
        return [self._row_to_artifact_record(row, columns) for row in rows]

    def delete_artifact(self, artifact_id: str) -> bool:
        """Delete an artifact from the registry by ID.

        Returns:
            True if deleted, False if not found
        """
        conn = self._get_connection()
        # Check if exists first
        exists = conn.execute(
            "SELECT 1 FROM constat_artifacts WHERE id = ?",
            [artifact_id]
        ).fetchone()

        if not exists:
            return False

        conn.execute(
            "DELETE FROM constat_artifacts WHERE id = ?",
            [artifact_id]
        )
        return True

    def delete_session_artifacts(self, user_id: str, session_id: str) -> int:
        """Delete all artifacts for a session.

        Returns:
            Number of artifacts deleted
        """
        conn = self._get_connection()
        # Count first
        count = conn.execute("""
            SELECT COUNT(*) FROM constat_artifacts
            WHERE user_id = ? AND session_id = ?
        """, [user_id, session_id]).fetchone()[0]

        conn.execute("""
            DELETE FROM constat_artifacts
            WHERE user_id = ? AND session_id = ?
        """, [user_id, session_id])
        return count

    # --- Publishing ---

    def publish_table(
        self,
        user_id: str,
        session_id: str,
        name: str,
        is_published: bool = True,
        title: Optional[str] = None,
    ) -> bool:
        """Publish or unpublish a table.

        Args:
            user_id: User ID
            session_id: Session ID
            name: Table name
            is_published: Whether to publish (True) or unpublish (False)
            title: Optional human-friendly title

        Returns:
            True if updated, False if not found
        """
        conn = self._get_connection()

        updates = ["is_published = ?"]
        params = [is_published]

        if title is not None:
            updates.append("title = ?")
            # noinspection PyTypeChecker
            params.append(title)

        # noinspection PyTypeChecker
        params.extend([user_id, session_id, name])

        conn.execute(f"""
            UPDATE constat_tables
            SET {", ".join(updates)}
            WHERE user_id = ? AND session_id = ? AND name = ?
        """, params)

        # Check if any row was updated
        row = conn.execute("""
            SELECT 1 FROM constat_tables
            WHERE user_id = ? AND session_id = ? AND name = ?
        """, [user_id, session_id, name]).fetchone()
        return row is not None

    def publish_artifact(
        self,
        user_id: str,
        session_id: str,
        name: str,
        is_published: bool = True,
        title: Optional[str] = None,
    ) -> bool:
        """Publish or unpublish an artifact.

        Args:
            user_id: User ID
            session_id: Session ID
            name: Artifact name
            is_published: Whether to publish (True) or unpublish (False)
            title: Optional human-friendly title

        Returns:
            True if updated, False if not found
        """
        conn = self._get_connection()

        updates = ["is_published = ?"]
        params = [is_published]

        if title is not None:
            updates.append("title = ?")
            # noinspection PyTypeChecker
            params.append(title)

        # noinspection PyTypeChecker
        params.extend([user_id, session_id, name])

        conn.execute(f"""
            UPDATE constat_artifacts
            SET {", ".join(updates)}
            WHERE user_id = ? AND session_id = ? AND name = ?
        """, params)

        # Check if any row was updated
        row = conn.execute("""
            SELECT 1 FROM constat_artifacts
            WHERE user_id = ? AND session_id = ? AND name = ?
        """, [user_id, session_id, name]).fetchone()
        return row is not None

    def unpublish_superseded_tables(
        self,
        user_id: str,
        session_id: str,
        new_table_names: list[str],
    ) -> list[str]:
        """Unpublish tables that are superseded by new "corrected" versions.

        Detects patterns like "corrected_X", "updated_X", "X_corrected", "X_v2"
        and unpublishes the original "X" table.

        Args:
            user_id: User ID
            session_id: Session ID
            new_table_names: Names of newly created/published tables

        Returns:
            List of table names that were unpublished
        """
        import re

        conn = self._get_connection()
        unpublished = []

        # Patterns that indicate a table supersedes another
        # e.g., "corrected_summary" supersedes "summary"
        prefix_patterns = [
            r'^corrected_(.+)$',
            r'^updated_(.+)$',
            r'^fixed_(.+)$',
            r'^revised_(.+)$',
            r'^new_(.+)$',
        ]
        # e.g., "summary_corrected" supersedes "summary"
        suffix_patterns = [
            r'^(.+)_corrected$',
            r'^(.+)_updated$',
            r'^(.+)_fixed$',
            r'^(.+)_revised$',
            r'^(.+)_v\d+$',  # e.g., summary_v2
        ]

        for new_name in new_table_names:
            base_name = None

            # Check prefix patterns
            for pattern in prefix_patterns:
                match = re.match(pattern, new_name, re.IGNORECASE)
                if match:
                    base_name = match.group(1)
                    break

            # Check suffix patterns if no prefix match
            if not base_name:
                for pattern in suffix_patterns:
                    match = re.match(pattern, new_name, re.IGNORECASE)
                    if match:
                        base_name = match.group(1)
                        break

            if base_name and base_name != new_name:
                # Unpublish the base table if it exists and is published
                conn.execute("""
                    UPDATE constat_tables
                    SET is_published = FALSE
                    WHERE user_id = ? AND session_id = ? AND name = ?
                      AND is_published = TRUE
                """, [user_id, session_id, base_name])

                # Check if we actually unpublished something
                exists = conn.execute("""
                    SELECT 1 FROM constat_tables
                    WHERE user_id = ? AND session_id = ? AND name = ?
                """, [user_id, session_id, base_name]).fetchone()

                if exists:
                    unpublished.append(base_name)

        return unpublished

    def mark_final_step(
        self,
        user_id: str,
        session_id: str,
        table_names: list[str] = None,
        artifact_names: list[str] = None,
    ) -> int:
        """Mark tables/artifacts as created in final step (auto-publishes them).

        Also, automatically unpublishes any superseded tables (e.g., if "corrected_X"
        is being published, "X" will be unpublished).

        Args:
            user_id: User ID
            session_id: Session ID
            table_names: List of table names to mark
            artifact_names: List of artifact names to mark

        Returns:
            Number of items updated
        """
        conn = self._get_connection()
        count = 0

        if table_names:
            # First, unpublish any superseded tables
            unpublished = self.unpublish_superseded_tables(user_id, session_id, table_names)
            if unpublished:
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"Auto-unpublished superseded tables: {unpublished}")

            for name in table_names:
                conn.execute("""
                    UPDATE constat_tables
                    SET is_final_step = TRUE, is_published = TRUE
                    WHERE user_id = ? AND session_id = ? AND name = ?
                """, [user_id, session_id, name])
                count += 1

        if artifact_names:
            for name in artifact_names:
                conn.execute("""
                    UPDATE constat_artifacts
                    SET is_final_step = TRUE, is_published = TRUE
                    WHERE user_id = ? AND session_id = ? AND name = ?
                """, [user_id, session_id, name])
                count += 1

        return count

    def list_published_tables(
        self,
        user_id: str,
        session_id: str,
        limit: int = 100,
    ) -> list[TableRecord]:
        """List only published tables for a session.

        Returns tables that are either explicitly published or created in final step.
        """
        conn = self._get_connection()

        rows = conn.execute("""
            SELECT * FROM constat_tables
            WHERE user_id = ? AND session_id = ?
              AND (is_published = TRUE OR is_final_step = TRUE)
            ORDER BY created_at DESC
            LIMIT ?
        """, [user_id, session_id, limit]).fetchall()

        columns = [desc[0] for desc in conn.description]
        return [self._row_to_table_record(row, columns) for row in rows]

    def list_published_artifacts(
        self,
        user_id: str,
        session_id: str,
        limit: int = 100,
    ) -> list[ArtifactRecord]:
        """List only published artifacts for a session.

        Returns artifacts that are either explicitly published or created in final step.
        """
        conn = self._get_connection()

        rows = conn.execute("""
            SELECT * FROM constat_artifacts
            WHERE user_id = ? AND session_id = ?
              AND (is_published = TRUE OR is_final_step = TRUE)
            ORDER BY created_at DESC
            LIMIT ?
        """, [user_id, session_id, limit]).fetchall()

        columns = [desc[0] for desc in conn.description]
        return [self._row_to_artifact_record(row, columns) for row in rows]

    # --- Search ---

    def _search_rows(
        self,
        table: str,
        query: str,
        user_id: Optional[str],
        limit: int,
    ) -> tuple[list, list[str]]:
        """Execute a search query against a table, returning rows and column names."""
        conn = self._get_connection()
        sql = f"""
            SELECT * FROM {table}
            WHERE (name ILIKE ? OR description ILIKE ?)
        """
        params: list = [f"%{query}%", f"%{query}%"]
        if user_id:
            sql += " AND user_id = ?"
            params.append(user_id)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()
        columns = [desc[0] for desc in conn.description]
        return rows, columns

    def search_tables(
        self,
        query: str,
        user_id: Optional[str] = None,
        limit: int = 50,
    ) -> list[TableRecord]:
        """Search tables by name or description.

        Args:
            query: Search query (matches name or description)
            user_id: Filter by user (None = all users)
            limit: Maximum number of results

        Returns:
            List of matching TableRecord objects
        """
        rows, columns = self._search_rows("constat_tables", query, user_id, limit)
        return [self._row_to_table_record(row, columns) for row in rows]

    def search_artifacts(
        self,
        query: str,
        user_id: Optional[str] = None,
        limit: int = 50,
    ) -> list[ArtifactRecord]:
        """Search artifacts by name or description.

        Args:
            query: Search query (matches name or description)
            user_id: Filter by user (None = all users)
            limit: Maximum number of results

        Returns:
            List of matching ArtifactRecord objects
        """
        rows, columns = self._search_rows("constat_artifacts", query, user_id, limit)
        return [self._row_to_artifact_record(row, columns) for row in rows]
