# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Registry-aware DataStore wrapper.

Wraps the standard DataStore to add:
1. Parquet file storage for tables (portable, file:// URIs for CLI)
2. In-memory DuckDB with views over Parquet (fast queries)
3. Central registry integration for discovery (/tables, /artifacts commands)

The underlying DataStore (SQLite) still handles:
- Scratchpad entries
- Session metadata
- Plan steps
- Internal artifacts (code, stdout, errors)
- State variables

Architecture:
- tables/*.parquet: Actual table data (portable, self-documenting)
- In-memory DuckDB: Query engine with views over Parquet files
- datastore.db: Session metadata, scratchpad, artifacts, state
- registry.db: Central registry for cross-session discovery

Parquet files are the source of truth. DuckDB views are created dynamically
on startup by scanning the tables directory.
"""

from pathlib import Path
from typing import Any, Optional

import duckdb
import pandas as pd

from constat.storage.datastore import DataStore
from constat.storage.registry import ConstatRegistry


class RegistryAwareDataStore:
    """
    DataStore wrapper that adds Parquet storage with in-memory DuckDB.

    Tables are saved as Parquet files (portable, file:// URIs for CLI).
    In-memory DuckDB creates views over these files for efficient queries.
    All tables are registered in the central registry for discovery.

    Parquet files are the source of truth. On startup, we scan the tables
    directory and create DuckDB views for each .parquet file found.

    Usage:
        datastore = DataStore(db_path=session_dir / "datastore.db")
        registry = ConstatRegistry(base_dir=Path(".constat"))

        store = RegistryAwareDataStore(
            datastore=datastore,
            registry=registry,
            user_id="default",
            session_id="2024-01-01_120000_abcd1234",
            tables_dir=session_dir / "tables",
        )

        # Tables saved to Parquet + registered, DuckDB creates views
        store.save_dataframe("customers", df)

        # Fast SQL queries via in-memory DuckDB views over Parquet
        result = store.query("SELECT * FROM customers WHERE country = 'US'")
    """

    def __init__(
        self,
        datastore: DataStore,
        registry: ConstatRegistry,
        user_id: str,
        session_id: str,
        tables_dir: Path,
    ):
        """
        Initialize the registry-aware datastore.

        Args:
            datastore: Underlying DataStore instance (for metadata/artifacts)
            registry: Central registry for tracking
            user_id: User ID for scoping
            session_id: Session ID for this session
            tables_dir: Directory for Parquet files
        """
        self._datastore = datastore
        self._registry = registry
        self._user_id = user_id
        self._session_id = session_id
        self._tables_dir = Path(tables_dir)
        self._tables_dir.mkdir(parents=True, exist_ok=True)

        # In-memory DuckDB for queries over Parquet files
        self._duckdb: Optional[duckdb.DuckDBPyConnection] = None

        # Initialize DuckDB and create views from existing Parquet files
        self._init_duckdb()

    def _init_duckdb(self) -> None:
        """Initialize in-memory DuckDB and create views for existing Parquet files."""
        self._duckdb = duckdb.connect(":memory:")

        # Scan tables directory and create views for all Parquet files
        self._load_parquet_files()

    def _load_parquet_files(self) -> None:
        """Scan tables directory and create views for all Parquet files."""
        conn = self._get_duckdb()

        for parquet_file in self._tables_dir.glob("*.parquet"):
            table_name = parquet_file.stem  # filename without extension
            try:
                conn.execute(f"""
                    CREATE OR REPLACE VIEW {table_name} AS
                    SELECT * FROM read_parquet('{parquet_file}')
                """)
            except Exception:
                pass  # Skip if view creation fails

    def _get_duckdb(self) -> duckdb.DuckDBPyConnection:
        """Get DuckDB connection, reinitializing if needed."""
        if self._duckdb is None:
            self._init_duckdb()
        return self._duckdb

    def _parquet_path(self, name: str) -> Path:
        """Get path to Parquet file for a table."""
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        return self._tables_dir / f"{safe_name}.parquet"

    # --- Table Operations (Parquet + DuckDB views + Registry) ---

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
        """
        Save a DataFrame to Parquet with DuckDB view and registry.

        Args:
            name: Table name
            df: DataFrame to save
            step_number: Which step created this table
            description: Human-readable description
            is_final_step: Whether this is created in the final execution step
            is_published: Whether explicitly published for artifacts panel
            role_id: Role that created this table (provenance)
        """
        if df.empty or len(df.columns) == 0:
            return

        conn = self._get_duckdb()
        parquet_path = self._parquet_path(name)

        # Save to Parquet using DuckDB (efficient writer)
        conn.execute(f"DROP VIEW IF EXISTS {name}")
        conn.execute("CREATE TABLE _temp_export AS SELECT * FROM df")
        conn.execute(f"COPY _temp_export TO '{parquet_path}' (FORMAT PARQUET)")
        conn.execute("DROP TABLE _temp_export")

        # Create view over the Parquet file
        conn.execute(f"""
            CREATE OR REPLACE VIEW {name} AS
            SELECT * FROM read_parquet('{parquet_path}')
        """)

        # Build column info for central registry
        columns = [
            {"name": col, "type": str(df[col].dtype)}
            for col in df.columns
        ]

        # Register in central registry (source of truth for metadata)
        self._registry.register_table(
            user_id=self._user_id,
            session_id=self._session_id,
            name=name,
            file_path=str(parquet_path),
            row_count=len(df),
            columns=columns,
            description=description or None,
            is_final_step=is_final_step,
            is_published=is_published or is_final_step,  # Final step tables are auto-published
            role_id=role_id,
        )

    def drop_table(self, name: str) -> bool:
        """Drop a table (view, Parquet, and registry entries)."""
        conn = self._get_duckdb()

        try:
            conn.execute(f"DROP VIEW IF EXISTS {name}")
        except Exception:
            pass

        # Remove Parquet file
        parquet_path = self._parquet_path(name)
        if parquet_path.exists():
            parquet_path.unlink()

        # Remove from central registry
        self._registry.delete_table(self._user_id, self._session_id, name)

        return True

    def load_dataframe(self, name: str) -> Optional[pd.DataFrame]:
        """Load a table as a DataFrame (via DuckDB view over Parquet)."""
        conn = self._get_duckdb()
        try:
            return conn.execute(f"SELECT * FROM {name}").fetchdf()
        except Exception:
            return None

    def query(self, sql: str) -> pd.DataFrame:
        """Execute SQL query over DuckDB views (which read from Parquet)."""
        conn = self._get_duckdb()
        return conn.execute(sql).fetchdf()

    def list_tables(self) -> list[dict]:
        """List all user tables with metadata from the central registry."""
        tables = self._registry.list_tables(
            user_id=self._user_id,
            session_id=self._session_id,
        )

        return [
            {
                "name": t.name,
                "step_number": 0,  # Not tracked in registry currently
                "row_count": t.row_count,
                "created_at": t.created_at,
                "description": t.description,
                "is_published": t.is_published,
                "is_final_step": t.is_final_step,
            }
            for t in tables
        ]

    def get_table_schema(self, name: str) -> Optional[list[dict]]:
        """Get schema for a table from DuckDB."""
        conn = self._get_duckdb()
        try:
            rows = conn.execute(f"DESCRIBE {name}").fetchall()
            return [
                {
                    "name": row[0],
                    "type": row[1],
                    "nullable": True,
                }
                for row in rows
            ]
        except Exception:
            return None

    def table_exists(self, name: str) -> bool:
        """Check if a table exists (Parquet file present)."""
        return self._parquet_path(name).exists()

    # --- State Operations (delegated to SQLite DataStore) ---

    def set_state(self, key: str, value: Any, step_number: int = 0) -> None:
        """Save a state variable."""
        self._datastore.set_state(key, value, step_number)

    def get_state(self, key: str) -> Optional[Any]:
        """Get a state variable."""
        return self._datastore.get_state(key)

    def get_all_state(self) -> dict[str, Any]:
        """Get all state variables."""
        return self._datastore.get_all_state()

    # --- Starred Tables Operations (delegated) ---

    def set_starred_tables(self, table_names: list[str]) -> None:
        """Set the list of starred table names."""
        self._datastore.set_starred_tables(table_names)

    def get_starred_tables(self) -> list[str]:
        """Get the list of starred table names."""
        return self._datastore.get_starred_tables()

    def toggle_table_star(self, table_name: str) -> bool:
        """Toggle a table's starred status."""
        return self._datastore.toggle_table_star(table_name)

    # --- Scratchpad Operations (delegated) ---

    def add_scratchpad_entry(
        self,
        step_number: int,
        goal: str,
        narrative: str,
        tables_created: Optional[list[str]] = None,
        code: Optional[str] = None,
    ) -> None:
        """Add a scratchpad entry."""
        self._datastore.add_scratchpad_entry(step_number, goal, narrative, tables_created, code)

    def get_scratchpad(self) -> list[dict]:
        """Get all scratchpad entries."""
        return self._datastore.get_scratchpad()

    def get_scratchpad_entry(self, step_number: int) -> Optional[dict]:
        """Get a specific scratchpad entry."""
        return self._datastore.get_scratchpad_entry(step_number)

    def get_scratchpad_as_markdown(self) -> str:
        """Get scratchpad as markdown."""
        return self._datastore.get_scratchpad_as_markdown()

    # --- Artifact Operations (delegated) ---

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
        """Add an artifact to the catalog."""
        return self._datastore.add_artifact(
            step_number, attempt, artifact_type, content,
            name, title, description, content_type, metadata, role_id
        )

    def get_artifacts(self, step_number: Optional[int] = None, artifact_type: Optional[str] = None):
        """Get artifacts."""
        return self._datastore.get_artifacts(step_number, artifact_type)

    def list_artifacts(self, include_content: bool = False) -> list[dict]:
        """List artifact metadata."""
        return self._datastore.list_artifacts(include_content)

    def get_artifact_by_name(self, name: str):
        """Get artifact by name."""
        return self._datastore.get_artifact_by_name(name)

    def get_artifact_by_id(self, artifact_id: int):
        """Get artifact by ID."""
        return self._datastore.get_artifact_by_id(artifact_id)

    def update_artifact_metadata(self, artifact_id: int, metadata_updates: dict) -> bool:
        """Update an artifact's metadata."""
        return self._datastore.update_artifact_metadata(artifact_id, metadata_updates)

    def save_rich_artifact(self, *args, **kwargs):
        """Save a rich artifact."""
        return self._datastore.save_rich_artifact(*args, **kwargs)

    def save_chart(self, *args, **kwargs):
        """Save a chart."""
        return self._datastore.save_chart(*args, **kwargs)

    def save_html(self, *args, **kwargs):
        """Save HTML."""
        return self._datastore.save_html(*args, **kwargs)

    def save_diagram(self, *args, **kwargs):
        """Save a diagram."""
        return self._datastore.save_diagram(*args, **kwargs)

    def save_image(self, *args, **kwargs):
        """Save an image."""
        return self._datastore.save_image(*args, **kwargs)

    # --- Session Metadata (delegated) ---

    def set_session_meta(self, key: str, value: str) -> None:
        """Set session metadata."""
        self._datastore.set_session_meta(key, value)

    def get_session_meta(self, key: str) -> Optional[str]:
        """Get session metadata."""
        return self._datastore.get_session_meta(key)

    def get_all_session_meta(self) -> dict[str, str]:
        """Get all session metadata."""
        return self._datastore.get_all_session_meta()

    # --- Plan Steps (delegated) ---

    def save_plan_step(self, *args, **kwargs) -> None:
        """Save a plan step."""
        self._datastore.save_plan_step(*args, **kwargs)

    def update_plan_step(self, *args, **kwargs) -> None:
        """Update a plan step."""
        self._datastore.update_plan_step(*args, **kwargs)

    def get_plan_steps(self) -> list[dict]:
        """Get all plan steps."""
        return self._datastore.get_plan_steps()

    # --- Execution History ---

    def get_execution_history_table(self):
        """Get execution history as a DataFrame."""
        return self._datastore.get_execution_history_table()

    def ensure_execution_history_table(self) -> bool:
        """Ensure execution history table exists.

        Creates/updates the execution_history table.
        """
        df = self._datastore.get_execution_history_table()
        if df is None or df.empty:
            return False

        self.save_dataframe("execution_history", df, step_number=0,
                          description="Execution history with code and outputs")
        return True

    def publish_tables(self, table_names: list[str]) -> int:
        """Mark specific tables as published for artifacts panel.

        Args:
            table_names: Names of tables to publish

        Returns:
            Number of tables updated
        """
        return self._registry.publish_tables(
            user_id=self._user_id,
            session_id=self._session_id,
            table_names=table_names,
        )

    # --- Utility ---

    def export_state_summary(self) -> dict:
        """Export state summary."""
        return {
            "tables": self.list_tables(),
            "state": self.get_all_state(),
            "scratchpad": self.get_scratchpad(),
            "artifacts": self._datastore.list_artifacts(),
        }

    def get_full_session_state(self) -> dict:
        """Get full session state for UI restoration."""
        return {
            "session": self.get_all_session_meta(),
            "plan_steps": self.get_plan_steps(),
            "tables": self.list_tables(),
            "state": self.get_all_state(),
            "scratchpad": self.get_scratchpad(),
            "artifacts": self._datastore.list_artifacts(),
        }

    def clear_step_data(self, step_number: int) -> None:
        """Clear data from a specific step.

        Note: Table cleanup by step_number is not supported with the current
        architecture (registry doesn't track step_number). Only clears
        state variables from the underlying datastore.
        """
        self._datastore.clear_step_data(step_number)

    def clear_session_data(self) -> None:
        """Clear all session data (tables, artifacts, scratchpad, state)."""
        self._datastore.clear_session_data()

    def close(self) -> None:
        """Close both DuckDB and underlying datastore."""
        if self._duckdb is not None:
            self._duckdb.close()
            self._duckdb = None
        self._datastore.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    # --- Properties for compatibility ---

    @property
    def db_path(self):
        """Get underlying SQLite database path (for metadata)."""
        return self._datastore.db_path

    @property
    def engine(self):
        """Get underlying SQLAlchemy engine (for metadata)."""
        return self._datastore.engine

    @property
    def registry(self) -> ConstatRegistry:
        """Get the registry instance."""
        return self._registry

    @property
    def user_id(self) -> str:
        """Get the user ID."""
        return self._user_id

    @property
    def session_id(self) -> str:
        """Get the session ID."""
        return self._session_id
