# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Native DuckDB session store — data federation layer.

Single persistent DuckDB file replaces SQLite-via-SQLAlchemy (metadata) +
Parquet files (tables) + in-memory DuckDB (query engine).

All session data lives in one ``session.duckdb``:
- User tables (CREATE TABLE via Arrow zero-copy)
- SQL-derivable intermediates (CREATE VIEW)
- Attached source DBs (ATTACH ... TYPE SQLITE READ_ONLY)
- File views (CSV/JSON/Parquet/Iceberg via read_*_auto)
- Session metadata (_constat_* internal tables)

PG SQL → DuckDB transpilation via ``constat.catalog.sql_transpiler``.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Optional, Union

import duckdb
import numpy as np
import pandas as pd

from constat.core.models import Artifact, ArtifactType, ARTIFACT_MIME_TYPES
from constat.storage.registry import ConstatRegistry

logger = logging.getLogger(__name__)

# Valid table name pattern
_VALID_TABLE_NAME = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')
_VERSION_BACKUP = re.compile(r'^(.+)__v(\d+)$')

# File format → DuckDB reader function
_FILE_READERS = {
    "csv": "read_csv_auto",
    "json": "read_json_auto",
    "jsonl": "read_json_auto",
    "ndjson": "read_json_auto",
    "parquet": "read_parquet",
    "iceberg": "iceberg_scan",
    "arrow": "read_parquet",
}

_INTERNAL_TABLES = frozenset({
    "_constat_state",
    "_constat_table_registry",
    "_constat_scratchpad",
    "_constat_artifacts",
    "_constat_session",
    "_constat_plan_steps",
})


def _validate_table_name(name: str) -> str:
    if not name:
        raise ValueError("Table name cannot be empty")
    if not _VALID_TABLE_NAME.match(name):
        raise ValueError(
            f"Invalid table name '{name}': must contain only alphanumeric characters "
            "and underscores, and must start with a letter or underscore"
        )
    if len(name) > 255:
        raise ValueError(f"Table name '{name}' exceeds maximum length of 255 characters")
    return name


def _json_serializer(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif hasattr(obj, 'to_dict'):
        return obj.to_dict()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class DuckDBSessionStore:
    """Native DuckDB session store — single persistent file per session.

    Replaces DataStore (SQLAlchemy/SQLite) + RegistryAwareDataStore (Parquet +
    in-memory DuckDB) with a single persistent DuckDB connection.

    Data tables are stored as native DuckDB tables (Arrow zero-copy from
    DataFrames). SQL intermediates can be lazy views. Attached source DBs
    and file views enable data federation through a single query namespace.
    """

    INTERNAL_TABLES = _INTERNAL_TABLES

    def __init__(
        self,
        db_path: Path,
        registry: ConstatRegistry,
        user_id: str,
        session_id: str,
    ):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._registry = registry
        self._user_id = user_id
        self._session_id = session_id
        self._lock = threading.RLock()

        self._conn: Optional[duckdb.DuckDBPyConnection] = duckdb.connect(str(self._db_path))
        self._init_metadata_tables()

        # Auto-import legacy Parquet files if tables/ directory exists
        legacy_dir = self._db_path.parent / "tables"
        if legacy_dir.is_dir():
            self._import_legacy_parquet(legacy_dir)

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    @contextmanager
    def _locked_conn(self) -> Generator[duckdb.DuckDBPyConnection, None, None]:
        with self._lock:
            if self._conn is None:
                raise RuntimeError("DuckDBSessionStore is closed")
            yield self._conn

    # ------------------------------------------------------------------
    # Schema init
    # ------------------------------------------------------------------

    def _init_metadata_tables(self) -> None:
        with self._locked_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS _constat_state (
                    key VARCHAR PRIMARY KEY,
                    value_json VARCHAR,
                    step_number INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS _constat_table_registry (
                    table_name VARCHAR PRIMARY KEY,
                    step_number INTEGER,
                    row_count INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    description VARCHAR,
                    role_id VARCHAR,
                    version INTEGER DEFAULT 1
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS _constat_scratchpad (
                    step_number INTEGER PRIMARY KEY,
                    goal VARCHAR,
                    narrative VARCHAR,
                    tables_created VARCHAR,
                    code VARCHAR,
                    user_query VARCHAR,
                    objective_index INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS _constat_artifacts (
                    id INTEGER PRIMARY KEY,
                    name VARCHAR,
                    step_number INTEGER,
                    attempt INTEGER,
                    artifact_type VARCHAR,
                    content_type VARCHAR,
                    content VARCHAR,
                    title VARCHAR,
                    description VARCHAR,
                    metadata_json VARCHAR,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    role_id VARCHAR,
                    version INTEGER DEFAULT 1
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS _constat_session (
                    key VARCHAR PRIMARY KEY,
                    value VARCHAR,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS _constat_plan_steps (
                    step_number INTEGER PRIMARY KEY,
                    goal VARCHAR,
                    expected_inputs VARCHAR,
                    expected_outputs VARCHAR,
                    status VARCHAR,
                    code VARCHAR,
                    error VARCHAR,
                    attempts INTEGER DEFAULT 0,
                    duration_ms INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP
                )
            """)

    # ------------------------------------------------------------------
    # Legacy Parquet import
    # ------------------------------------------------------------------

    def _import_legacy_parquet(self, tables_dir: Path) -> None:
        """One-time import: read Parquet files into native DuckDB tables."""
        with self._locked_conn() as conn:
            for pf in tables_dir.glob("*.parquet"):
                name = pf.stem
                if _VERSION_BACKUP.match(name):
                    continue
                try:
                    existing = conn.execute(
                        "SELECT COUNT(*) FROM duckdb_tables() "
                        "WHERE table_name = ? AND schema_name = 'main' "
                        "AND database_name = current_database()",
                        [name],
                    ).fetchone()[0]
                    if existing:
                        continue
                    conn.execute(
                        f"CREATE TABLE {_validate_table_name(name)} AS "
                        f"SELECT * FROM read_parquet('{pf}')"
                    )
                    logger.info(f"Imported legacy Parquet table: {name}")
                except (duckdb.Error, ValueError) as e:
                    logger.warning(f"Failed to import legacy Parquet {pf.name}: {e}")

    # ------------------------------------------------------------------
    # Table operations
    # ------------------------------------------------------------------

    def save_dataframe(
        self,
        name: str,
        df: pd.DataFrame,
        step_number: int = 0,
        description: str = "",
        is_final_step: bool = False,
        is_published: bool = False,
        role_id: Optional[str] = None,
    ) -> None:
        if isinstance(df, (list, dict)):
            df = pd.DataFrame(df)

        if df.empty or len(df.columns) == 0:
            return

        # Auto-convert to view if data is an unmodified SELECT result
        source_sql = getattr(df, 'attrs', {}).get('_source_sql')
        if (
            source_sql
            and source_sql.strip().upper().startswith('SELECT')
            and list(df.columns) == df.attrs.get('_source_columns', [])
            and len(df) == df.attrs.get('_source_len', -1)
        ):
            logger.info(f"Auto-converting save_dataframe('{name}') to create_view (data from query)")
            # source_sql is already DuckDB SQL (post-transpilation from query())
            self._create_view_impl(
                _validate_table_name(name), source_sql, step_number=step_number,
                description=description, is_final_step=is_final_step, is_published=is_published,
            )
            return

        validated = _validate_table_name(name)

        with self._locked_conn() as conn:
            # Archive previous version
            existing = conn.execute(
                "SELECT version FROM _constat_table_registry WHERE table_name = ?",
                [validated],
            ).fetchone()

            new_version = 1
            if existing:
                current_version = existing[0] or 1
                new_version = current_version + 1
                backup = f"{validated}__v{current_version}"
                try:
                    _validate_table_name(backup)
                    conn.execute(f"CREATE TABLE {backup} AS SELECT * FROM {validated}")
                except (duckdb.Error, ValueError):
                    pass

            # Drop existing (could be TABLE or VIEW).
            # DuckDB raises CatalogException if type doesn't match, so try both.
            try:
                conn.execute(f"DROP TABLE IF EXISTS {validated}")
            except duckdb.CatalogException:
                conn.execute(f"DROP VIEW IF EXISTS {validated}")

            # Arrow zero-copy: register df, create table, unregister
            conn.register("_tmp_df", df)
            conn.execute(f"CREATE TABLE {validated} AS SELECT * FROM _tmp_df")
            conn.unregister("_tmp_df")

            # Update internal registry
            self._upsert_registry(conn, validated, step_number, len(df), description, role_id, new_version)

        # Update central registry
        columns = [
            {"name": str(col), "type": str(df[col].dtype)}
            for col in df.columns
        ]
        self._registry.register_table(
            user_id=self._user_id,
            session_id=self._session_id,
            name=name,
            file_path=str(self._db_path),
            row_count=len(df),
            columns=columns,
            description=description or None,
            step_number=step_number,
            is_final_step=is_final_step,
            is_published=is_published or is_final_step,
            role_id=role_id,
        )

    def _upsert_registry(
        self, conn: duckdb.DuckDBPyConnection,
        name: str, step_number: int, row_count: int,
        description: str, role_id: Optional[str], version: int,
    ) -> None:
        conn.execute("""
            INSERT INTO _constat_table_registry (table_name, step_number, row_count, description, role_id, version)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (table_name) DO UPDATE SET
                step_number = excluded.step_number,
                row_count = excluded.row_count,
                description = excluded.description,
                role_id = excluded.role_id,
                version = excluded.version
        """, [name, step_number, row_count, description, role_id, version])

    def save_artifact(self, name: str, content, artifact_type: str = None, **kwargs) -> None:
        """Save a non-tabular artifact (markdown, JSON, image bytes, etc.).

        Mirrors _DataStore.save_artifact for skill script compatibility.
        Delegates to add_artifact with type detection.
        """
        import base64
        if artifact_type is None:
            if isinstance(content, bytes):
                artifact_type = "png"
            elif isinstance(content, dict):
                artifact_type = "json"
            else:
                artifact_type = "markdown"
        if isinstance(content, dict):
            content = json.dumps(content)
        elif isinstance(content, bytes):
            content = base64.b64encode(content).decode()
        self.add_artifact(
            step_number=kwargs.get("step_number", 0),
            attempt=kwargs.get("attempt", 1),
            artifact_type=artifact_type,
            content=content,
            name=name,
            title=kwargs.get("title"),
            description=kwargs.get("description"),
        )

    def load_dataframe(self, name: str) -> Optional[pd.DataFrame]:
        validated = _validate_table_name(name)
        with self._locked_conn() as conn:
            try:
                return conn.execute(f"SELECT * FROM {validated}").fetchdf()
            except duckdb.Error as e:
                if "Contents of view were altered" in str(e):
                    # Stale view — upstream schema changed. Recreate to re-bind columns.
                    df = self._refresh_stale_view(conn, validated)
                    if df is not None:
                        return df
                logger.warning(f"load_dataframe('{name}') failed: {e}")
                return None
            except ValueError as e:
                logger.warning(f"load_dataframe('{name}') failed: {e}")
                return None

    def _refresh_stale_view(self, conn: duckdb.DuckDBPyConnection, name: str) -> Optional[pd.DataFrame]:
        """Recreate a stale view whose upstream schema changed, then re-query."""
        try:
            row = conn.execute(
                "SELECT sql FROM duckdb_views() "
                "WHERE database_name = current_database() AND schema_name = 'main' "
                "AND view_name = ?", [name]
            ).fetchone()
            if not row or not row[0]:
                return None
            view_ddl = row[0]
            conn.execute(f"DROP VIEW IF EXISTS {name}")
            conn.execute(view_ddl)
            logger.info(f"Refreshed stale view '{name}'")
            return conn.execute(f"SELECT * FROM {name}").fetchdf()
        except duckdb.Error as e:
            logger.warning(f"Failed to refresh stale view '{name}': {e}")
            return None

    def get_table_data(self, name: str) -> Optional[pd.DataFrame]:
        return self.load_dataframe(name)

    def query(self, sql: str) -> pd.DataFrame:
        from constat.catalog.sql_transpiler import transpile_sql
        duckdb_sql = transpile_sql(sql, target_dialect="duckdb", source_dialect="postgres")
        with self._locked_conn() as conn:
            df = conn.execute(duckdb_sql).fetchdf()
        # Tag result so save_dataframe can auto-convert to view
        # Store DuckDB SQL (post-transpilation) to avoid double transpilation
        df.attrs["_source_sql"] = duckdb_sql
        df.attrs["_source_columns"] = list(df.columns)
        df.attrs["_source_len"] = len(df)
        return df

    @property
    def _artifacts(self) -> dict[str, any]:
        """Return all saved artifacts: tables as DataFrames, others as native types."""
        result = {}
        # Tables → DataFrame
        for t in self.list_tables():
            df = self.load_dataframe(t["name"])
            if df is not None:
                result[t["name"]] = df
        # Non-table artifacts → str | dict | bytes
        for a in self.list_artifacts(include_content=True):
            name = a.get("name", "")
            atype = a.get("artifact_type", a.get("type", ""))
            content = a.get("content", "")
            if atype in ("code", "error", "output"):
                continue  # internal execution artifacts
            if atype == "json":
                try:
                    result[name] = json.loads(content)
                except (json.JSONDecodeError, TypeError):
                    result[name] = content
            elif atype in ("png", "jpeg", "svg"):
                import base64
                try:
                    result[name] = base64.b64decode(content)
                except Exception:
                    result[name] = content
            else:
                # markdown, text, html, chart, mermaid, etc.
                result[name] = content
        return result

    @property
    def _files(self) -> dict[str, any]:
        """Alias for _artifacts. Skill script compat."""
        return self._artifacts

    def list_tables(self) -> list[dict]:
        tables = self._registry.list_tables(
            user_id=self._user_id,
            session_id=self._session_id,
        )
        # Identify views: anything NOT in duckdb_tables() is a view.
        # Using duckdb_tables() instead of information_schema because
        # information_schema fails with DuckDB 1.4.x when SQLite DBs are ATTACHed.
        with self._locked_conn() as conn:
            real_table_names = {
                row[0]
                for row in conn.execute(
                    "SELECT table_name FROM duckdb_tables() "
                    "WHERE database_name = current_database() "
                    "AND schema_name = 'main' AND NOT internal"
                ).fetchall()
            }
            view_names = {
                t.name for t in tables
                if t.name not in real_table_names
            }
        return [
            {
                "name": t.name,
                "step_number": t.step_number,
                "row_count": t.row_count,
                "created_at": t.created_at,
                "description": t.description,
                "is_published": t.is_published,
                "is_final_step": t.is_final_step,
                "version": self._version_count(t.name),
                "version_count": self._version_count(t.name),
                "is_view": t.name in view_names,
            }
            for t in tables
        ]

    def _version_count(self, name: str) -> int:
        with self._locked_conn() as conn:
            result = conn.execute(
                "SELECT version FROM _constat_table_registry WHERE table_name = ?",
                [name],
            ).fetchone()
        return (result[0] or 1) if result else 1

    def get_table_schema(self, name: str) -> Optional[list[dict]]:
        with self._locked_conn() as conn:
            try:
                rows = conn.execute(f"DESCRIBE {_validate_table_name(name)}").fetchall()
                return [
                    {"name": row[0], "type": row[1], "nullable": True}
                    for row in rows if len(row) >= 2
                ]
            except (duckdb.Error, ValueError):
                return None

    def table_exists(self, name: str) -> bool:
        validated = _validate_table_name(name)
        with self._locked_conn() as conn:
            try:
                conn.execute(f"SELECT 1 FROM {validated} LIMIT 0")
                return True
            except duckdb.Error:
                return False

    def drop_table(self, name: str) -> bool:
        try:
            validated = _validate_table_name(name)
        except ValueError:
            return False

        with self._locked_conn() as conn:
            result = conn.execute(
                "SELECT version FROM _constat_table_registry WHERE table_name = ?",
                [validated],
            ).fetchone()

            try:
                conn.execute(f"DROP TABLE IF EXISTS {validated}")
            except duckdb.CatalogException:
                conn.execute(f"DROP VIEW IF EXISTS {validated}")

            if result and result[0]:
                for v in range(1, result[0]):
                    backup = f"{validated}__v{v}"
                    try:
                        _validate_table_name(backup)
                        conn.execute(f"DROP TABLE IF EXISTS {backup}")
                    except ValueError:
                        pass

            conn.execute(
                "DELETE FROM _constat_table_registry WHERE table_name = ?",
                [validated],
            )

        self._registry.delete_table(self._user_id, self._session_id, name)
        return True

    def get_table_versions(self, name: str) -> list[dict]:
        with self._locked_conn() as conn:
            result = conn.execute(
                "SELECT version, row_count FROM _constat_table_registry WHERE table_name = ?",
                [name],
            ).fetchone()

            if not result:
                return []

            current_version = result[0] or 1
            versions = []
            versions.append({
                "version": current_version,
                "row_count": result[1],
                "is_current": True,
            })

            for v in range(current_version - 1, 0, -1):
                backup = f"{name}__v{v}"
                try:
                    row_count = conn.execute(
                        f"SELECT COUNT(*) FROM {_validate_table_name(backup)}"
                    ).fetchone()[0]
                except (duckdb.Error, ValueError):
                    row_count = 0
                versions.append({
                    "version": v,
                    "row_count": row_count,
                    "is_current": False,
                })

        return versions

    def load_table_version(self, name: str, version: int) -> Optional[pd.DataFrame]:
        with self._locked_conn() as conn:
            result = conn.execute(
                "SELECT version FROM _constat_table_registry WHERE table_name = ?",
                [name],
            ).fetchone()
            if not result:
                return None

            current_version = result[0] or 1
            if version == current_version:
                return self.load_dataframe(name)

            backup = f"{name}__v{version}"
            return self.load_dataframe(backup)

    def update_table_step_number(self, name: str, step_number: int) -> None:
        with self._locked_conn() as conn:
            conn.execute(
                "UPDATE _constat_table_registry SET step_number = ? WHERE table_name = ?",
                [step_number, name],
            )
        reg_conn = self._registry._get_connection()
        reg_conn.execute("""
            UPDATE constat_tables SET step_number = ?
            WHERE user_id = ? AND session_id = ? AND name = ?
        """, [step_number, self._user_id, self._session_id, name])

    def publish_tables(self, table_names: list[str]) -> int:
        return self._registry.publish_tables(
            user_id=self._user_id,
            session_id=self._session_id,
            table_names=table_names,
        )

    # ------------------------------------------------------------------
    # Federation: views, attach, file registration
    # ------------------------------------------------------------------

    def create_view_raw(self, name: str, duckdb_sql: str, step_number: int = 0, description: str = "") -> None:
        """Create a lazy SQL view from DuckDB-dialect SQL (no transpilation)."""
        self._create_view_impl(_validate_table_name(name), duckdb_sql, step_number, description)

    def _create_view_impl(
        self, validated_name: str, duckdb_sql: str, step_number: int = 0, description: str = "",
        is_final_step: bool = False, is_published: bool = False,
    ) -> None:
        """Shared impl: create view from DuckDB-dialect SQL."""
        with self._locked_conn() as conn:
            # Drop existing TABLE if present — DuckDB can't replace a TABLE with a VIEW
            try:
                conn.execute(f"DROP TABLE IF EXISTS {validated_name}")
            except duckdb.CatalogException:
                pass
            conn.execute(f"CREATE OR REPLACE VIEW {validated_name} AS {duckdb_sql}")

            # Probe row count and columns from the view
            try:
                row_count = conn.execute(f"SELECT COUNT(*) FROM {validated_name}").fetchone()[0]
                col_info = conn.execute(f"DESCRIBE {validated_name}").fetchall()
                columns = [{"name": row[0], "type": row[1]} for row in col_info]
            except Exception as e:
                logger.warning(f"View probe failed for '{validated_name}': {e}")
                row_count = 0
                columns = []

            self._upsert_registry(conn, validated_name, step_number, row_count, description, None, 1)

        # Register in central registry
        self._registry.register_table(
            user_id=self._user_id,
            session_id=self._session_id,
            name=validated_name,
            file_path=str(self._db_path),
            row_count=row_count,
            columns=columns,
            description=description or None,
            step_number=step_number,
            is_final_step=is_final_step,
            is_published=is_published or is_final_step,
        )

    def create_view(
        self, name: str, pg_sql: str, step_number: int = 0, description: str = "",
        is_final_step: bool = False, is_published: bool = False,
    ) -> None:
        """Create a lazy SQL view. PG SQL is transpiled to DuckDB."""
        from constat.catalog.sql_transpiler import transpile_sql
        validated = _validate_table_name(name)
        duckdb_sql = transpile_sql(pg_sql, target_dialect="duckdb", source_dialect="postgres")
        self._create_view_impl(
            validated, duckdb_sql, step_number, description,
            is_final_step=is_final_step, is_published=is_published,
        )

    def attach(self, name: str, path: str, db_type: str = "sqlite", read_only: bool = True) -> None:
        """ATTACH a source SQLite/DuckDB file. Tables queryable as schema.table.

        Idempotent — skips if already attached.
        """
        validated = _validate_table_name(name)
        with self._locked_conn() as conn:
            # Check if already attached
            attached = [row[0] for row in conn.execute("SHOW DATABASES").fetchall()]
            if validated in attached:
                return
            parts = []
            if db_type:
                parts.append(f"TYPE {db_type}")
            if read_only:
                parts.append("READ_ONLY true")
            options = ", ".join(parts)
            conn.execute(f"ATTACH '{path}' AS {validated} ({options})")
            logger.info(f"Attached {db_type} database: {validated} -> {path}")

    def register_file(self, name: str, path: str, fmt: str) -> None:
        """CREATE VIEW over CSV/JSON/Parquet/Iceberg file. Supports s3://, https://."""
        validated = _validate_table_name(name)
        reader = _FILE_READERS.get(fmt)
        if not reader:
            raise ValueError(f"Unsupported file format: {fmt}. Supported: {list(_FILE_READERS.keys())}")
        with self._locked_conn() as conn:
            conn.execute(f"CREATE OR REPLACE VIEW {validated} AS SELECT * FROM {reader}('{path}')")

    def list_attached_databases(self) -> list[dict]:
        """List all attached databases (including main)."""
        with self._locked_conn() as conn:
            rows = conn.execute("SHOW DATABASES").fetchall()
            return [{"name": row[0]} for row in rows]

    def get_ddl(self) -> str:
        """Return full DDL of the session store: attached DBs, tables, views.

        Uses duckdb_tables() instead of information_schema to avoid DuckDB 1.4.x
        bug where information_schema fails when SQLite databases are ATTACHed.
        """
        lines: list[str] = []
        with self._locked_conn() as conn:
            # Attached databases — list tables via duckdb_tables()
            dbs = conn.execute("SHOW DATABASES").fetchall()
            current_db = conn.execute("SELECT current_database()").fetchone()[0]
            for row in dbs:
                db_name = row[0]
                if db_name == "memory" or db_name == current_db:
                    continue
                lines.append(f"-- Attached: {db_name}")
                try:
                    tables = conn.execute(
                        "SELECT table_name FROM duckdb_tables() "
                        "WHERE database_name = ? AND schema_name = 'main'",
                        [db_name],
                    ).fetchall()
                    for (tbl,) in tables:
                        try:
                            cols = conn.execute(f"DESCRIBE {db_name}.{tbl}").fetchall()
                            col_defs = ", ".join(f"{c[0]} {c[1]}" for c in cols)
                            lines.append(f"--   {db_name}.{tbl} ({col_defs})")
                        except duckdb.Error:
                            lines.append(f"--   {db_name}.{tbl}")
                except duckdb.Error:
                    pass
                lines.append("")

            # User tables (exclude _constat_* internals)
            real_tables = conn.execute(
                "SELECT table_name FROM duckdb_tables() "
                "WHERE database_name = current_database() AND schema_name = 'main' "
                "AND NOT internal AND table_name NOT LIKE '_constat_%' "
                "ORDER BY table_name",
            ).fetchall()
            for (tbl_name,) in real_tables:
                if _VERSION_BACKUP.match(tbl_name):
                    continue
                try:
                    cols = conn.execute(f"DESCRIBE {tbl_name}").fetchall()
                    col_defs = ", ".join(f"{c[0]} {c[1]}" for c in cols)
                    row_count = conn.execute(f"SELECT COUNT(*) FROM {tbl_name}").fetchone()[0]
                    lines.append(f"CREATE TABLE {tbl_name} ({col_defs}); -- {row_count} rows")
                except duckdb.Error:
                    lines.append(f"-- TABLE {tbl_name} (error reading schema)")
                lines.append("")

            # Views: registered names that are not in duckdb_tables()
            real_table_names = {r[0] for r in real_tables}
            registry_names = [
                r[0] for r in conn.execute(
                    "SELECT table_name FROM _constat_table_registry "
                    "WHERE table_name NOT LIKE '_constat_%' ORDER BY table_name"
                ).fetchall()
            ]
            for vname in registry_names:
                if vname in real_table_names or _VERSION_BACKUP.match(vname):
                    continue
                # This is likely a view
                try:
                    cols = conn.execute(f"DESCRIBE {vname}").fetchall()
                    col_defs = ", ".join(f"{c[0]} {c[1]}" for c in cols)
                    lines.append(f"CREATE VIEW {vname} AS ...; -- columns: {col_defs}")
                except duckdb.Error:
                    lines.append(f"CREATE VIEW {vname} AS ...;")
                lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # State operations
    # ------------------------------------------------------------------

    def set_state(self, key: str, value: Any, step_number: int = 0) -> None:
        value_json = json.dumps(value, default=_json_serializer)
        with self._locked_conn() as conn:
            conn.execute("""
                INSERT INTO _constat_state (key, value_json, step_number)
                VALUES (?, ?, ?)
                ON CONFLICT (key) DO UPDATE SET
                    value_json = excluded.value_json,
                    step_number = excluded.step_number
            """, [key, value_json, step_number])

    def get_state(self, key: str) -> Optional[Any]:
        with self._locked_conn() as conn:
            result = conn.execute(
                "SELECT value_json FROM _constat_state WHERE key = ?", [key]
            ).fetchone()
        if result:
            return json.loads(result[0])
        return None

    def get_all_state(self) -> dict[str, Any]:
        with self._locked_conn() as conn:
            rows = conn.execute("SELECT key, value_json FROM _constat_state").fetchall()
        return {key: json.loads(val) for key, val in rows}

    def set_starred_tables(self, table_names: list[str]) -> None:
        self.set_state("_starred_tables", table_names)

    def get_starred_tables(self) -> list[str]:
        return self.get_state("_starred_tables") or []

    def toggle_table_star(self, table_name: str) -> bool:
        starred = self.get_starred_tables()
        if table_name in starred:
            starred.remove(table_name)
            is_starred = False
        else:
            starred.append(table_name)
            is_starred = True
        self.set_starred_tables(starred)
        return is_starred

    def get_shared_users(self) -> list[str]:
        return self.get_state("_shared_with") or []

    def add_shared_user(self, user_id: str) -> None:
        shared = self.get_shared_users()
        if user_id not in shared:
            shared.append(user_id)
            self.set_state("_shared_with", shared)

    def remove_shared_user(self, user_id: str) -> None:
        shared = self.get_shared_users()
        if user_id in shared:
            shared.remove(user_id)
            self.set_state("_shared_with", shared)

    def is_public(self) -> bool:
        return self.get_state("_public") is True

    def set_public(self, public: bool) -> None:
        self.set_state("_public", public)

    def set_query_intent(self, query_text: str, intents: list[dict], is_followup: bool = False) -> None:
        import datetime
        history = self.get_state("_intent_history") or []
        history.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "query": query_text[:500],
            "is_followup": is_followup,
            "intents": intents,
        })
        if len(history) > 50:
            history = history[-50:]
        self.set_state("_intent_history", history)

    def get_intent_history(self) -> list[dict]:
        return self.get_state("_intent_history") or []

    # ------------------------------------------------------------------
    # Scratchpad
    # ------------------------------------------------------------------

    def add_scratchpad_entry(
        self,
        step_number: int,
        goal: str,
        narrative: str,
        tables_created: Optional[list[str]] = None,
        code: Optional[str] = None,
        user_query: Optional[str] = None,
        objective_index: Optional[int] = None,
    ) -> None:
        tables_str = ",".join(tables_created) if tables_created else ""
        with self._locked_conn() as conn:
            conn.execute("""
                INSERT INTO _constat_scratchpad
                    (step_number, goal, narrative, tables_created, code, user_query, objective_index)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (step_number) DO UPDATE SET
                    goal = excluded.goal,
                    narrative = excluded.narrative,
                    tables_created = excluded.tables_created,
                    code = excluded.code,
                    user_query = excluded.user_query,
                    objective_index = excluded.objective_index
            """, [step_number, goal, narrative, tables_str, code or "", user_query, objective_index])

    def get_scratchpad_entry(self, step_number: int) -> Optional[dict]:
        with self._locked_conn() as conn:
            result = conn.execute(
                "SELECT goal, narrative, tables_created, code, user_query, objective_index "
                "FROM _constat_scratchpad WHERE step_number = ?",
                [step_number],
            ).fetchone()
        if result:
            return {
                "step_number": step_number,
                "goal": result[0],
                "narrative": result[1],
                "tables_created": result[2].split(",") if result[2] else [],
                "code": result[3] or "",
                "user_query": result[4] or "",
                "objective_index": result[5],
            }
        return None

    def get_scratchpad(self) -> list[dict]:
        with self._locked_conn() as conn:
            rows = conn.execute("""
                SELECT step_number, goal, narrative, tables_created, code, user_query, objective_index
                FROM _constat_scratchpad ORDER BY step_number
            """).fetchall()
        return [
            {
                "step_number": row[0],
                "goal": row[1],
                "narrative": row[2],
                "tables_created": row[3].split(",") if row[3] else [],
                "code": row[4] or "",
                "user_query": row[5] or "",
                "objective_index": row[6],
            }
            for row in rows
        ]

    def get_scratchpad_as_markdown(self) -> str:
        entries = self.get_scratchpad()
        if not entries:
            return "(no previous steps)"
        parts = []
        current_query = None
        for entry in entries:
            # Group entries by user query so the LLM can distinguish objectives
            query = entry.get('user_query') or ''
            if query and query != current_query:
                current_query = query
                parts.append(f"---\n**Objective:** {query}")
            section = f"## Step {entry['step_number']}: {entry['goal']}\n{entry['narrative']}"
            if entry['tables_created']:
                table_parts = []
                for tname in entry['tables_created']:
                    schema = self.get_table_schema(tname)
                    if schema:
                        cols = ", ".join(f"{c['name']} ({c['type']})" for c in schema)
                        table_parts.append(f"`{tname}` — columns: {cols}")
                    else:
                        table_parts.append(f"`{tname}`")
                section += f"\n\n**Tables created:**\n" + "\n".join(f"- {t}" for t in table_parts)
            parts.append(section)
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------

    def add_artifact(
        self,
        step_number: int,
        attempt: int,
        artifact_type: str,
        content: str,
        name: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        content_type: Optional[str] = None,
        metadata: Optional[dict] = None,
        role_id: Optional[str] = None,
    ) -> int:
        metadata_json = json.dumps(metadata) if metadata else None
        with self._locked_conn() as conn:
            result = conn.execute(
                "SELECT COALESCE(MAX(id), 0) + 1 FROM _constat_artifacts"
            ).fetchone()
            artifact_id = result[0]

            if name is None:
                name = f"artifact_{artifact_id}_{artifact_type}"

            version = 1
            if name and not name.startswith("artifact_"):
                vr = conn.execute(
                    "SELECT MAX(version) FROM _constat_artifacts WHERE name = ?",
                    [name],
                ).fetchone()
                if vr[0] is not None:
                    version = vr[0] + 1

            conn.execute("""
                INSERT INTO _constat_artifacts
                    (id, name, step_number, attempt, artifact_type, content_type,
                     content, title, description, metadata_json, role_id, version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [artifact_id, name, step_number, attempt, artifact_type,
                  content_type, content, title, description, metadata_json,
                  role_id, version])

        return artifact_id

    def save_rich_artifact(
        self,
        name: str,
        artifact_type: Union[ArtifactType, str],
        content: Union[str, bytes],
        step_number: int = 0,
        attempt: int = 1,
        title: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[dict] = None,
        role_id: Optional[str] = None,
    ) -> Artifact:
        if isinstance(artifact_type, str):
            try:
                artifact_type_enum = ArtifactType(artifact_type)
                type_str = artifact_type_enum.value
                ct = ARTIFACT_MIME_TYPES.get(artifact_type_enum)
            except ValueError:
                artifact_type_enum = ArtifactType.TEXT
                type_str = artifact_type
                ct = None
        else:
            artifact_type_enum = artifact_type
            type_str = artifact_type.value
            ct = ARTIFACT_MIME_TYPES.get(artifact_type)

        artifact_id = self.add_artifact(
            step_number=step_number, attempt=attempt, artifact_type=type_str,
            content=content, name=name, title=title, description=description,
            content_type=ct, metadata=metadata, role_id=role_id,
        )

        return Artifact(
            id=artifact_id, name=name, artifact_type=artifact_type_enum,
            content=content, step_number=step_number, attempt=attempt,
            title=title, description=description, metadata=metadata or {},
        )

    def save_chart(self, name: str, spec: dict, step_number: int = 0,
                   title: Optional[str] = None, chart_type: str = "vega-lite") -> Artifact:
        artifact_type = ArtifactType.PLOTLY if chart_type == "plotly" else ArtifactType.CHART
        return self.save_rich_artifact(
            name=name, artifact_type=artifact_type,
            content=json.dumps(spec), step_number=step_number,
            title=title, metadata={"chart_type": chart_type},
        )

    def save_html(self, name: str, html_content: str, step_number: int = 0,
                  title: Optional[str] = None, description: Optional[str] = None) -> Artifact:
        return self.save_rich_artifact(
            name=name, artifact_type=ArtifactType.HTML,
            content=html_content, step_number=step_number,
            title=title, description=description,
        )

    def save_diagram(self, name: str, diagram_code: str, diagram_format: str = "mermaid",
                     step_number: int = 0, title: Optional[str] = None) -> Artifact:
        type_map = {"mermaid": ArtifactType.MERMAID, "graphviz": ArtifactType.GRAPHVIZ, "dot": ArtifactType.GRAPHVIZ}
        artifact_type = type_map.get(diagram_format, ArtifactType.DIAGRAM)
        return self.save_rich_artifact(
            name=name, artifact_type=artifact_type,
            content=diagram_code, step_number=step_number,
            title=title, metadata={"format": diagram_format},
        )

    def save_image(self, name: str, image_data: Union[str, bytes], image_format: str = "png",
                   step_number: int = 0, title: Optional[str] = None,
                   width: Optional[int] = None, height: Optional[int] = None) -> Artifact:
        type_map = {"png": ArtifactType.PNG, "svg": ArtifactType.SVG, "jpeg": ArtifactType.JPEG, "jpg": ArtifactType.JPEG}
        artifact_type = type_map.get(image_format, ArtifactType.PNG)
        meta = {}
        if width:
            meta["width"] = width
        if height:
            meta["height"] = height
        return self.save_rich_artifact(
            name=name, artifact_type=artifact_type,
            content=image_data, step_number=step_number,
            title=title, metadata=meta if meta else None,
        )

    @staticmethod
    def _row_to_artifact(row: tuple) -> Artifact:
        type_str = row[4]
        try:
            artifact_type = ArtifactType(type_str)
        except ValueError:
            artifact_type = ArtifactType.TEXT
        metadata = {}
        if row[9]:
            try:
                metadata = json.loads(row[9])
            except json.JSONDecodeError:
                pass
        return Artifact(
            id=row[0], name=row[1] or f"artifact_{row[0]}",
            step_number=row[2], attempt=row[3],
            artifact_type=artifact_type, content_type=row[5],
            content=row[6], title=row[7], description=row[8],
            metadata=metadata,
            created_at=str(row[10]) if row[10] else None,
            role_id=row[11] if len(row) > 11 else None,
        )

    def get_artifacts(self, step_number: Optional[int] = None, artifact_type: Optional[str] = None) -> list[Artifact]:
        query = (
            "SELECT id, name, step_number, attempt, artifact_type, content_type,"
            " content, title, description, metadata_json, created_at, role_id"
            " FROM _constat_artifacts"
        )
        params = []
        conditions = []
        if step_number is not None:
            conditions.append("step_number = ?")
            params.append(step_number)
        if artifact_type is not None:
            conditions.append("artifact_type = ?")
            params.append(artifact_type)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY step_number, attempt, id"

        with self._locked_conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_artifact(row) for row in rows]

    def list_artifacts(self, include_content: bool = False) -> list[dict]:
        if include_content:
            with self._locked_conn() as conn:
                rows = conn.execute("""
                    SELECT id, name, step_number, attempt, artifact_type, content_type,
                           content, title, description, metadata_json, created_at, role_id
                    FROM _constat_artifacts
                    WHERE (name, version) IN (
                        SELECT name, MAX(version) FROM _constat_artifacts GROUP BY name
                    )
                    ORDER BY step_number, attempt, id
                """).fetchall()
            return [self._row_to_artifact(row).to_dict() for row in rows]
        else:
            with self._locked_conn() as conn:
                rows = conn.execute("""
                    SELECT a.id, a.name, a.step_number, a.attempt, a.artifact_type, a.content_type,
                           LENGTH(a.content) as content_length, a.title, a.description,
                           a.created_at, a.role_id, a.version,
                           vc.version_count
                    FROM _constat_artifacts a
                    JOIN (
                        SELECT name, MAX(version) as max_version, COUNT(*) as version_count
                        FROM _constat_artifacts GROUP BY name
                    ) vc ON a.name = vc.name AND a.version = vc.max_version
                    ORDER BY a.step_number, a.attempt, a.id
                """).fetchall()
            return [
                {
                    "id": row[0], "name": row[1], "step_number": row[2],
                    "attempt": row[3], "type": row[4], "content_type": row[5],
                    "content_length": row[6], "title": row[7], "description": row[8],
                    "created_at": str(row[9]) if row[9] else None,
                    "role_id": row[10], "version": row[11], "version_count": row[12],
                }
                for row in rows
            ]

    def get_artifact_by_name(self, name: str) -> Optional[Artifact]:
        with self._locked_conn() as conn:
            result = conn.execute("""
                SELECT id, name, step_number, attempt, artifact_type, content_type,
                       content, title, description, metadata_json, created_at, role_id
                FROM _constat_artifacts WHERE name = ?
                ORDER BY version DESC LIMIT 1
            """, [name]).fetchone()
        return self._row_to_artifact(result) if result else None

    def get_artifact_by_id(self, artifact_id: int) -> Optional[Artifact]:
        with self._locked_conn() as conn:
            result = conn.execute("""
                SELECT id, name, step_number, attempt, artifact_type, content_type,
                       content, title, description, metadata_json, created_at, role_id
                FROM _constat_artifacts WHERE id = ?
            """, [artifact_id]).fetchone()
        return self._row_to_artifact(result) if result else None

    def get_artifact_versions(self, name: str) -> list[dict]:
        with self._locked_conn() as conn:
            rows = conn.execute("""
                SELECT id, version, step_number, attempt, created_at
                FROM _constat_artifacts WHERE name = ?
                ORDER BY version DESC
            """, [name]).fetchall()
        return [
            {"id": row[0], "version": row[1], "step_number": row[2],
             "attempt": row[3], "created_at": str(row[4]) if row[4] else None}
            for row in rows
        ]

    def get_artifacts_by_type(self, artifact_type: Union[ArtifactType, str]) -> list[Artifact]:
        type_str = artifact_type.value if isinstance(artifact_type, ArtifactType) else artifact_type
        return self.get_artifacts(artifact_type=type_str)

    def update_artifact_metadata(self, artifact_id: int, metadata_updates: dict) -> bool:
        with self._locked_conn() as conn:
            result = conn.execute(
                "SELECT metadata_json FROM _constat_artifacts WHERE id = ?",
                [artifact_id],
            ).fetchone()
            if not result:
                return False
            existing = {}
            if result[0]:
                try:
                    existing = json.loads(result[0])
                except json.JSONDecodeError:
                    pass
            existing.update(metadata_updates)
            conn.execute(
                "UPDATE _constat_artifacts SET metadata_json = ? WHERE id = ?",
                [json.dumps(existing), artifact_id],
            )
        return True

    def delete_artifact(self, artifact_id: int) -> bool:
        with self._locked_conn() as conn:
            result = conn.execute(
                "SELECT name FROM _constat_artifacts WHERE id = ?",
                [artifact_id],
            ).fetchone()
            if not result:
                return False
            conn.execute(
                "DELETE FROM _constat_artifacts WHERE name = ?",
                [result[0]],
            )
        return True

    # ------------------------------------------------------------------
    # Session metadata
    # ------------------------------------------------------------------

    def set_session_meta(self, key: str, value: str) -> None:
        with self._locked_conn() as conn:
            conn.execute("""
                INSERT INTO _constat_session (key, value)
                VALUES (?, ?)
                ON CONFLICT (key) DO UPDATE SET value = excluded.value
            """, [key, value])

    def get_session_meta(self, key: str) -> Optional[str]:
        with self._locked_conn() as conn:
            result = conn.execute(
                "SELECT value FROM _constat_session WHERE key = ?", [key]
            ).fetchone()
        return result[0] if result else None

    def get_all_session_meta(self) -> dict[str, str]:
        with self._locked_conn() as conn:
            rows = conn.execute("SELECT key, value FROM _constat_session").fetchall()
        return {key: value for key, value in rows}

    # ------------------------------------------------------------------
    # Plan steps
    # ------------------------------------------------------------------

    def save_plan_step(
        self,
        step_number: int,
        goal: str,
        expected_inputs: Optional[list[str]] = None,
        expected_outputs: Optional[list[str]] = None,
        status: str = "pending",
    ) -> None:
        with self._locked_conn() as conn:
            conn.execute("""
                INSERT INTO _constat_plan_steps
                    (step_number, goal, expected_inputs, expected_outputs, status)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (step_number) DO UPDATE SET
                    goal = excluded.goal,
                    expected_inputs = excluded.expected_inputs,
                    expected_outputs = excluded.expected_outputs,
                    status = excluded.status
            """, [
                step_number, goal,
                ",".join(expected_inputs) if expected_inputs else "",
                ",".join(expected_outputs) if expected_outputs else "",
                status,
            ])

    def update_plan_step(
        self,
        step_number: int,
        status: Optional[str] = None,
        code: Optional[str] = None,
        error: Optional[str] = None,
        attempts: Optional[int] = None,
        duration_ms: Optional[int] = None,
    ) -> None:
        updates = {}
        if status is not None:
            updates["status"] = status
        if code is not None:
            updates["code"] = code
        if error is not None:
            updates["error"] = error
        if attempts is not None:
            updates["attempts"] = attempts
        if duration_ms is not None:
            updates["duration_ms"] = duration_ms
        if not updates:
            return

        set_parts = [f"{k} = ?" for k in updates.keys()]
        params = list(updates.values())
        if status in ("completed", "failed"):
            set_parts.append("completed_at = CURRENT_TIMESTAMP")

        params.append(step_number)
        with self._locked_conn() as conn:
            conn.execute(
                f"UPDATE _constat_plan_steps SET {', '.join(set_parts)} WHERE step_number = ?",
                params,
            )

    def get_plan_steps(self) -> list[dict]:
        with self._locked_conn() as conn:
            rows = conn.execute("""
                SELECT step_number, goal, expected_inputs, expected_outputs,
                       status, code, error, attempts, duration_ms, created_at, completed_at
                FROM _constat_plan_steps ORDER BY step_number
            """).fetchall()
        return [
            {
                "step_number": row[0], "goal": row[1],
                "expected_inputs": row[2].split(",") if row[2] else [],
                "expected_outputs": row[3].split(",") if row[3] else [],
                "status": row[4], "code": row[5], "error": row[6],
                "attempts": row[7], "duration_ms": row[8],
                "created_at": str(row[9]) if row[9] else None,
                "completed_at": str(row[10]) if row[10] else None,
            }
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Execution history
    # ------------------------------------------------------------------

    def get_execution_history_table(self, include_all_attempts: bool = True) -> Optional[pd.DataFrame]:
        scratchpad_entries = self.get_scratchpad()
        if not scratchpad_entries:
            return None

        code_artifacts = self.get_artifacts(artifact_type="code")
        output_artifacts = self.get_artifacts(artifact_type="output")
        error_artifacts = self.get_artifacts(artifact_type="error")

        code_by_sa = {(a.step_number, a.attempt): a for a in code_artifacts}
        output_by_sa = {(a.step_number, a.attempt): a for a in output_artifacts}
        error_by_sa = {(a.step_number, a.attempt): a for a in error_artifacts}
        scratchpad_by_step = {e["step_number"]: e for e in scratchpad_entries}

        all_attempts = set(code_by_sa.keys()) | set(output_by_sa.keys()) | set(error_by_sa.keys())

        if not include_all_attempts:
            latest = {}
            for step, attempt in all_attempts:
                if step not in latest or attempt > latest[step]:
                    latest[step] = attempt
            all_attempts = {(s, a) for s, a in all_attempts if a == latest.get(s)}

        history = []
        for step_num, attempt in sorted(all_attempts):
            entry = scratchpad_by_step.get(step_num, {})
            code = code_by_sa.get((step_num, attempt))
            output = output_by_sa.get((step_num, attempt))
            error = error_by_sa.get((step_num, attempt))
            success = output is not None and error is None
            history.append({
                "step_number": step_num, "attempt": attempt, "success": success,
                "goal": entry.get("goal"),
                "narrative": entry.get("narrative") if success else None,
                "code": code.content if code else None,
                "output": output.content if output else None,
                "error": error.content if error else None,
                "tables_created": ", ".join(entry.get("tables_created", [])) or None if success else None,
            })

        return pd.DataFrame(history)

    def ensure_execution_history_table(self) -> bool:
        df = self.get_execution_history_table()
        if df is None or df.empty:
            return False
        self.save_dataframe("execution_history", df, step_number=0,
                            description="Execution history with code and outputs")
        return True

    # ------------------------------------------------------------------
    # Bulk clear methods (for context.py compaction)
    # ------------------------------------------------------------------

    def clear_state_before_step(self, cutoff_step: int) -> int:
        with self._locked_conn() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM _constat_state WHERE step_number < ?",
                [cutoff_step],
            ).fetchone()[0]
            conn.execute(
                "DELETE FROM _constat_state WHERE step_number < ?",
                [cutoff_step],
            )
        return count

    def clear_scratchpad(self) -> None:
        with self._locked_conn() as conn:
            conn.execute("DELETE FROM _constat_scratchpad")

    def clear_artifacts(self) -> None:
        with self._locked_conn() as conn:
            conn.execute("DELETE FROM _constat_artifacts")

    def clear_plan_steps(self) -> None:
        with self._locked_conn() as conn:
            conn.execute("DELETE FROM _constat_plan_steps")

    def clear_state(self) -> None:
        with self._locked_conn() as conn:
            conn.execute("DELETE FROM _constat_state")

    # ------------------------------------------------------------------
    # Truncation / step data clearing
    # ------------------------------------------------------------------

    def clear_step_data(self, step_number: int) -> None:
        tables = self._registry.list_tables(
            user_id=self._user_id, session_id=self._session_id
        )
        for t in tables:
            if t.step_number == step_number:
                self.drop_table(t.name)
        with self._locked_conn() as conn:
            conn.execute("DELETE FROM _constat_state WHERE step_number = ?", [step_number])

    def truncate_from_step(self, from_step: int) -> list[str]:
        tables = self._registry.list_tables(
            user_id=self._user_id, session_id=self._session_id
        )
        tables_dropped = []
        for t in tables:
            if t.step_number >= from_step:
                self.drop_table(t.name)
                tables_dropped.append(t.name)

        with self._locked_conn() as conn:
            conn.execute("DELETE FROM _constat_scratchpad WHERE step_number >= ?", [from_step])
            conn.execute("DELETE FROM _constat_state WHERE step_number >= ?", [from_step])
            conn.execute("DELETE FROM _constat_artifacts WHERE step_number >= ?", [from_step])
            conn.execute("DELETE FROM _constat_plan_steps WHERE step_number >= ?", [from_step])

        return tables_dropped

    def clear_session_data(self) -> None:
        with self._locked_conn() as conn:
            conn.execute("DELETE FROM _constat_scratchpad")
            conn.execute("DELETE FROM _constat_artifacts")
            conn.execute("DELETE FROM _constat_plan_steps")
            conn.execute("DELETE FROM _constat_state")

    # ------------------------------------------------------------------
    # State export / summary
    # ------------------------------------------------------------------

    def export_state_summary(self) -> dict:
        return {
            "tables": self.list_tables(),
            "state": self.get_all_state(),
            "scratchpad": self.get_scratchpad(),
            "artifacts": self.list_artifacts(),
        }

    def get_full_session_state(self) -> dict:
        return {
            "session": self.get_all_session_meta(),
            "plan_steps": self.get_plan_steps(),
            "tables": self.list_tables(),
            "state": self.get_all_state(),
            "scratchpad": self.get_scratchpad(),
            "artifacts": self.list_artifacts(),
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def db_path(self) -> Path:
        return self._db_path

    @property
    def engine(self):
        raise AttributeError(
            "DuckDBSessionStore does not use SQLAlchemy. "
            "Use store methods (clear_state, clear_scratchpad, etc.) instead of engine.connect()."
        )

    @property
    def registry(self) -> ConstatRegistry:
        return self._registry

    @property
    def user_id(self) -> str:
        return self._user_id

    @property
    def session_id(self) -> str:
        return self._session_id
