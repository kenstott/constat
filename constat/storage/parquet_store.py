# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Parquet-based datastore for session tables.

Stores tables as Parquet files in the session directory, using DuckDB
as an in-memory query engine. Integrates with the central registry for
discovery and collaboration.

Benefits:
- No dynamic table creation in enterprise databases
- Filesystem isolation prevents naming collisions
- DuckDB provides fast analytical queries over Parquet
- Tables are portable and inspectable (standard format)
- Works identically for file-based and server deployments
"""

import json
from pathlib import Path
from typing import Any, Optional

import duckdb
import numpy as np
import pandas as pd

from constat.storage.registry import ConstatRegistry


class ParquetDataStore:
    """
    Parquet-based datastore for session tables.

    Tables are stored as Parquet files in:
        .constat/<user_id>/sessions/<session_id>/tables/<name>.parquet

    DuckDB is used as an in-memory query engine over these files.
    The central registry tracks all tables for discovery.

    Internal state (scratchpad, session metadata) is stored in a
    lightweight SQLite file within the session directory.
    """

    def __init__(
        self,
        user_id: str,
        session_id: str,
        base_dir: Optional[Path] = None,
        registry: Optional[ConstatRegistry] = None,
    ):
        """
        Initialize the Parquet datastore.

        Args:
            user_id: User ID for scoping
            session_id: Session ID
            base_dir: Base .constat directory. Defaults to ".constat"
            registry: Shared registry instance. Creates new one if None.
        """
        self.user_id = user_id
        self.session_id = session_id
        self.base_dir = Path(base_dir) if base_dir else Path(".constat")

        # Session directory structure
        self.session_dir = self.base_dir / user_id / "sessions" / session_id
        self.tables_dir = self.session_dir / "tables"
        self.state_file = self.session_dir / "state.json"

        # Ensure directories exist
        self.tables_dir.mkdir(parents=True, exist_ok=True)

        # Registry for tracking tables/artifacts
        self.registry = registry or ConstatRegistry(base_dir=self.base_dir)

        # DuckDB connection for queries (in-memory)
        self._duckdb: Optional[duckdb.DuckDBPyConnection] = None

        # In-memory state
        self._state: dict[str, Any] = {}
        self._scratchpad: list[dict] = []

        # Load existing state if resuming
        self._load_state()

    def _get_duckdb(self) -> duckdb.DuckDBPyConnection:
        """Get or create DuckDB connection."""
        if self._duckdb is None:
            self._duckdb = duckdb.connect(":memory:")
        return self._duckdb

    def _parquet_path(self, name: str) -> Path:
        """Get path to Parquet file for a table."""
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        return self.tables_dir / f"{safe_name}.parquet"

    def _load_state(self) -> None:
        """Load state from disk if exists."""
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                self._state = data.get("state", {})
                self._scratchpad = data.get("scratchpad", [])
            except (json.JSONDecodeError, OSError):
                pass

    def _save_state(self) -> None:
        """Save state to disk."""
        data = {
            "state": self._state,
            "scratchpad": self._scratchpad,
        }
        self.state_file.write_text(json.dumps(data, indent=2, default=self._json_serializer))

    @staticmethod
    def _json_serializer(obj: Any) -> Any:
        """Custom JSON serializer for numpy types."""
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
        elif hasattr(obj, "to_dict"):
            return obj.to_dict()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    # --- Table Operations ---

    def save_dataframe(
        self,
        name: str,
        df: pd.DataFrame,
        step_number: int = 0,
        description: str = "",
    ) -> None:
        """
        Save a DataFrame as a Parquet file.

        Args:
            name: Table name
            df: DataFrame to save
            step_number: Which step created this table
            description: Human-readable description
        """
        if df.empty or len(df.columns) == 0:
            return

        # Save as Parquet
        parquet_path = self._parquet_path(name)
        df.to_parquet(parquet_path, index=False)

        # Build column info
        columns = [
            {"name": col, "type": str(df[col].dtype)}
            for col in df.columns
        ]

        # Register in central registry
        self.registry.register_table(
            user_id=self.user_id,
            session_id=self.session_id,
            name=name,
            file_path=str(parquet_path),
            row_count=len(df),
            columns=columns,
            description=description or None,
        )

        # Clear DuckDB cache so it picks up the new file
        if self._duckdb:
            try:
                self._duckdb.execute(f"DROP VIEW IF EXISTS {name}")
            except Exception:
                pass

    def load_dataframe(self, name: str) -> Optional[pd.DataFrame]:
        """
        Load a table as a DataFrame.

        Args:
            name: Table name

        Returns:
            DataFrame or None if table doesn't exist
        """
        parquet_path = self._parquet_path(name)
        if not parquet_path.exists():
            return None

        try:
            return pd.read_parquet(parquet_path)
        except Exception:
            return None

    def query(self, sql: str) -> pd.DataFrame:
        """
        Execute SQL query over Parquet tables.

        DuckDB can query Parquet files directly. Tables are automatically
        available by name.

        Args:
            sql: SQL query

        Returns:
            Query results as DataFrame
        """
        conn = self._get_duckdb()

        # Register all Parquet files as views
        for parquet_file in self.tables_dir.glob("*.parquet"):
            table_name = parquet_file.stem
            try:
                conn.execute(f"""
                    CREATE OR REPLACE VIEW {table_name} AS
                    SELECT * FROM read_parquet('{parquet_file}')
                """)
            except Exception:
                pass

        # Execute query
        result = conn.execute(sql).fetchdf()
        return result

    def table_exists(self, name: str) -> bool:
        """Check if a table exists."""
        return self._parquet_path(name).exists()

    def list_tables(self) -> list[dict]:
        """
        List all tables in this session.

        Returns:
            List of dicts with name, row_count, description
        """
        tables = self.registry.list_tables(
            user_id=self.user_id,
            session_id=self.session_id,
        )

        return [
            {
                "name": t.name,
                "row_count": t.row_count,
                "description": t.description,
                "columns": t.columns,
            }
            for t in tables
        ]

    def drop_table(self, name: str) -> bool:
        """
        Drop a table.

        Args:
            name: Table name

        Returns:
            True if dropped, False if not found
        """
        parquet_path = self._parquet_path(name)
        if not parquet_path.exists():
            return False

        # Remove file
        parquet_path.unlink()

        # Remove from registry
        self.registry.delete_table(self.user_id, self.session_id, name)

        # Clear from DuckDB cache
        if self._duckdb:
            try:
                self._duckdb.execute(f"DROP VIEW IF EXISTS {name}")
            except Exception:
                pass

        return True

    # --- State Operations ---

    def set_state(self, key: str, value: Any, step_number: int = 0) -> None:
        """Save a state variable."""
        self._state[key] = {
            "value": value,
            "step_number": step_number,
        }
        self._save_state()

    def get_state(self, key: str) -> Optional[Any]:
        """Get a state variable."""
        entry = self._state.get(key)
        return entry["value"] if entry else None

    def get_all_state(self) -> dict[str, Any]:
        """Get all state variables."""
        return {k: v["value"] for k, v in self._state.items()}

    def clear_state(self) -> None:
        """Clear all state."""
        self._state.clear()
        self._save_state()

    # --- Scratchpad Operations ---

    def record_step(
        self,
        step_number: int,
        goal: str,
        code: str = "",
        output: str = "",
        tables_created: Optional[list[str]] = None,
        error: str = "",
        success: bool = True,
    ) -> None:
        """Record a step in the scratchpad."""
        self._scratchpad.append({
            "step_number": step_number,
            "goal": goal,
            "code": code,
            "output": output,
            "tables_created": tables_created or [],
            "error": error,
            "success": success,
        })
        self._save_state()

    def get_scratchpad(self) -> list[dict]:
        """Get all scratchpad entries."""
        return list(self._scratchpad)

    def get_step(self, step_number: int) -> Optional[dict]:
        """Get a specific step from the scratchpad."""
        for entry in self._scratchpad:
            if entry["step_number"] == step_number:
                return entry
        return None

    def clear_scratchpad(self) -> None:
        """Clear the scratchpad."""
        self._scratchpad.clear()
        self._save_state()

    # --- Cleanup ---

    def close(self) -> None:
        """Close connections."""
        if self._duckdb:
            self._duckdb.close()
            self._duckdb = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
