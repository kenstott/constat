"""SQLAlchemy-based datastore for persistent intermediate storage.

Provides persistent storage for:
- Tables created during step execution
- State variables shared between steps
- Session artifacts for history/resumption
- Rich artifacts (charts, HTML, diagrams, images)

Supports multiple backends via SQLAlchemy:
- DuckDB (default): duckdb:///path/to/file.duckdb or duckdb:///:memory:
- PostgreSQL: postgresql://user:pass@host:port/db
- SQLite: sqlite:///path/to/file.db or sqlite:///:memory:
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    Integer,
    String,
    Text,
    DateTime,
    inspect,
    text,
    func,
)
from sqlalchemy.engine import Engine
from sqlalchemy.dialects import postgresql, sqlite

from constat.core.models import Artifact, ArtifactType, ARTIFACT_MIME_TYPES


class DataStore:
    """
    SQLAlchemy-based persistent datastore for session state.

    Each session gets its own database that persists:
    - Tables created by execution steps
    - State variables (serialized as JSON)
    - Metadata about what each step produced

    This enables:
    - State sharing between steps via tables
    - Session resumption with full data context
    - History inspection with actual data (not just logs)

    Supports multiple backends:
    - DuckDB (default for local): duckdb:///file.duckdb
    - PostgreSQL (for production): postgresql://user:pass@host/db
    - SQLite: sqlite:///file.db

    Usage:
        # DuckDB (default)
        store = DataStore(db_path="/path/to/session.duckdb")

        # PostgreSQL
        store = DataStore(uri="postgresql://user:pass@localhost/constat_session_123")

        # SQLite
        store = DataStore(uri="sqlite:///session.db")
    """

    # Internal table names
    INTERNAL_TABLES = {
        "_constat_state",
        "_constat_table_registry",
        "_constat_scratchpad",
        "_constat_artifacts",
        "_constat_session",
        "_constat_plan_steps",
    }

    # Valid table name pattern: alphanumeric and underscores, must start with letter or underscore
    _VALID_TABLE_NAME_PATTERN = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')

    @classmethod
    def _validate_table_name(cls, name: str) -> str:
        """Validate and return a safe table name.

        Prevents SQL injection by ensuring table names only contain safe characters.

        Args:
            name: Table name to validate

        Returns:
            The validated table name

        Raises:
            ValueError: If table name contains invalid characters
        """
        if not name:
            raise ValueError("Table name cannot be empty")
        if not cls._VALID_TABLE_NAME_PATTERN.match(name):
            raise ValueError(
                f"Invalid table name '{name}': must contain only alphanumeric characters "
                "and underscores, and must start with a letter or underscore"
            )
        if len(name) > 255:
            raise ValueError(f"Table name '{name}' exceeds maximum length of 255 characters")
        return name

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        uri: Optional[str] = None,
    ):
        """
        Initialize the datastore.

        Args:
            db_path: Path to database file. Creates SQLite database at path.
                     If None and no uri, uses in-memory SQLite.
            uri: SQLAlchemy connection URI. Takes precedence over db_path.
                 Examples:
                   - sqlite:///path/to/file.db
                   - postgresql://user:pass@host/db
                   - duckdb:///path/to/file.duckdb (requires duckdb-engine)
        """
        if uri:
            self.uri = uri
        elif db_path:
            path = Path(db_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            # Use SQLite as default (always available)
            # For .duckdb extension, try DuckDB first
            if str(path).endswith('.duckdb'):
                try:
                    # Check if duckdb-engine is available
                    import duckdb_engine  # noqa
                    self.uri = f"duckdb:///{path}"
                except ImportError:
                    # Fall back to SQLite
                    sqlite_path = str(path).replace('.duckdb', '.db')
                    self.uri = f"sqlite:///{sqlite_path}"
            else:
                self.uri = f"sqlite:///{path}"
        else:
            self.uri = "sqlite:///:memory:"

        self.db_path = Path(db_path) if db_path else None
        self.engine = create_engine(self.uri)
        self.metadata = MetaData()

        self._init_metadata_tables()

    def _init_metadata_tables(self) -> None:
        """Create internal metadata tables."""
        with self.engine.begin() as conn:
            # State variables table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS _constat_state (
                    key VARCHAR(255) PRIMARY KEY,
                    value_json TEXT,
                    step_number INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))

            # Table registry
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS _constat_table_registry (
                    table_name VARCHAR(255) PRIMARY KEY,
                    step_number INTEGER,
                    row_count INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    description TEXT
                )
            """))

            # Scratchpad - narrative per step
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS _constat_scratchpad (
                    step_number INTEGER PRIMARY KEY,
                    goal TEXT,
                    narrative TEXT,
                    tables_created TEXT,
                    code TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))

            # Artifact catalog
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS _constat_artifacts (
                    id INTEGER PRIMARY KEY,
                    name VARCHAR(255),
                    step_number INTEGER,
                    attempt INTEGER,
                    artifact_type VARCHAR(50),
                    content_type VARCHAR(100),
                    content TEXT,
                    title VARCHAR(255),
                    description TEXT,
                    metadata_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))

            # Session metadata
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS _constat_session (
                    key VARCHAR(255) PRIMARY KEY,
                    value TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))

            # Plan steps
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS _constat_plan_steps (
                    step_number INTEGER PRIMARY KEY,
                    goal TEXT,
                    expected_inputs TEXT,
                    expected_outputs TEXT,
                    status VARCHAR(50),
                    code TEXT,
                    error TEXT,
                    attempts INTEGER DEFAULT 0,
                    duration_ms INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP
                )
            """))

    def _execute(self, sql: str, params: Optional[list] = None):
        """Execute a SQL statement with optional parameters."""
        with self.engine.begin() as conn:
            if params:
                # Convert to named parameters for SQLAlchemy
                result = conn.execute(text(sql), self._to_named_params(sql, params))
            else:
                result = conn.execute(text(sql))
            return result

    def _to_named_params(self, sql: str, params: list) -> dict:
        """Convert positional params to named params."""
        # Replace ? with :p0, :p1, etc.
        named_sql = sql
        param_dict = {}
        for i, param in enumerate(params):
            named_sql = named_sql.replace("?", f":p{i}", 1)
            param_dict[f"p{i}"] = param
        return param_dict

    @staticmethod
    def _json_serializer(obj: Any) -> Any:
        """Custom JSON serializer for numpy types and other non-standard types."""
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

    def _upsert(self, table: str, key_col: str, key_val: Any, data: dict) -> None:
        """Insert or update a row."""
        with self.engine.begin() as conn:
            # Check if row exists
            result = conn.execute(
                text(f"SELECT 1 FROM {table} WHERE {key_col} = :key"),
                {"key": key_val}
            ).fetchone()

            if result:
                # Update
                set_clause = ", ".join(f"{k} = :{k}" for k in data.keys())
                conn.execute(
                    text(f"UPDATE {table} SET {set_clause} WHERE {key_col} = :key"),
                    {**data, "key": key_val}
                )
            else:
                # Insert
                data[key_col] = key_val
                cols = ", ".join(data.keys())
                vals = ", ".join(f":{k}" for k in data.keys())
                conn.execute(
                    text(f"INSERT INTO {table} ({cols}) VALUES ({vals})"),
                    data
                )

    def _serialize_complex_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Serialize dict/list columns to JSON strings for SQLite compatibility."""
        df = df.copy()
        for col in df.columns:
            # Check if any value in the column is a dict or list
            sample = df[col].dropna().head(1)
            if len(sample) > 0:
                val = sample.iloc[0]
                if isinstance(val, (dict, list)):
                    df[col] = df[col].apply(
                        lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x
                    )
        return df

    def save_dataframe(
        self,
        name: str,
        df: pd.DataFrame,
        step_number: int = 0,
        description: str = "",
    ) -> None:
        """
        Save a pandas DataFrame as a table.

        Args:
            name: Table name
            df: DataFrame to save
            step_number: Which step created this table
            description: Human-readable description
        """
        # Cannot save DataFrame with no columns - would generate invalid SQL
        if len(df.columns) == 0:
            return

        # Serialize dict/list columns to JSON for SQLite compatibility
        df = self._serialize_complex_columns(df)

        # Use pandas to_sql for cross-database compatibility
        df.to_sql(name, self.engine, if_exists="replace", index=False)

        # Update registry
        self._upsert(
            "_constat_table_registry",
            "table_name",
            name,
            {
                "step_number": step_number,
                "row_count": len(df),
                "description": description,
            }
        )

    def load_dataframe(self, name: str) -> Optional[pd.DataFrame]:
        """
        Load a table as a pandas DataFrame.

        Args:
            name: Table name

        Returns:
            DataFrame or None if table doesn't exist
        """
        try:
            # Validate table name to prevent SQL injection
            validated_name = self._validate_table_name(name)
            # Use read_sql_query instead of read_sql_table to avoid
            # duckdb-engine compatibility issues with SQLAlchemy reflection
            return pd.read_sql_query(f"SELECT * FROM {validated_name}", self.engine)
        except ValueError as e:
            logger.warning(f"Invalid table name '{name}': {e}")
            return None
        except Exception as e:
            logger.debug(f"Failed to load table '{name}': {e}")
            return None

    def query(self, sql: str) -> pd.DataFrame:
        """
        Execute SQL query and return results as DataFrame.

        Args:
            sql: SQL query

        Returns:
            Query results as DataFrame
        """
        return pd.read_sql_query(sql, self.engine)

    def set_state(self, key: str, value: Any, step_number: int = 0) -> None:
        """
        Save a state variable.

        Args:
            key: Variable name
            value: Value (must be JSON-serializable)
            step_number: Which step set this variable
        """
        value_json = json.dumps(value, default=self._json_serializer)
        self._upsert(
            "_constat_state",
            "key",
            key,
            {"value_json": value_json, "step_number": step_number}
        )

    def get_state(self, key: str) -> Optional[Any]:
        """
        Get a state variable.

        Args:
            key: Variable name

        Returns:
            Value or None if not found
        """
        with self.engine.connect() as conn:
            result = conn.execute(
                text("SELECT value_json FROM _constat_state WHERE key = :key"),
                {"key": key}
            ).fetchone()

        if result:
            return json.loads(result[0])
        return None

    def get_all_state(self) -> dict[str, Any]:
        """Get all state variables as a dictionary."""
        with self.engine.connect() as conn:
            rows = conn.execute(
                text("SELECT key, value_json FROM _constat_state")
            ).fetchall()

        return {key: json.loads(value_json) for key, value_json in rows}

    def set_query_intent(
        self,
        query_text: str,
        intents: list[dict],
        is_followup: bool = False,
    ) -> None:
        """
        Store intent classification for debugging and analysis.

        Args:
            query_text: The user's query text
            intents: List of detected intents, each with 'intent', 'confidence', 'patterns'
            is_followup: Whether this is a follow-up query
        """
        import datetime

        # Get existing intent history or create new
        history = self.get_state("_intent_history") or []

        history.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "query": query_text[:500],  # Truncate long queries
            "is_followup": is_followup,
            "intents": intents,
        })

        # Keep last 50 intent records
        if len(history) > 50:
            history = history[-50:]

        self.set_state("_intent_history", history)

    def get_intent_history(self) -> list[dict]:
        """Get the intent classification history for this session."""
        return self.get_state("_intent_history") or []

    def list_tables(self) -> list[dict]:
        """
        List all user tables with metadata.

        Returns:
            List of table info dicts
        """
        with self.engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT table_name, step_number, row_count, created_at, description
                FROM _constat_table_registry
                ORDER BY step_number, table_name
            """)).fetchall()

        return [
            {
                "name": row[0],
                "step_number": row[1],
                "row_count": row[2],
                "created_at": str(row[3]) if row[3] else None,
                "description": row[4],
            }
            for row in rows
        ]

    def get_table_schema(self, name: str) -> Optional[list[dict]]:
        """
        Get schema for a table.

        Args:
            name: Table name

        Returns:
            List of column info dicts or None if table doesn't exist
        """
        try:
            # Validate table name to prevent SQL injection
            validated_name = self._validate_table_name(name)
            inspector = inspect(self.engine)
            columns = inspector.get_columns(validated_name)
            return [
                {
                    "name": col["name"],
                    "type": str(col["type"]),
                    "nullable": col.get("nullable", True),
                }
                for col in columns
            ]
        except ValueError as e:
            logger.warning(f"Invalid table name '{name}': {e}")
            return None
        except Exception as e:
            logger.debug(f"Failed to get schema for table '{name}': {e}")
            return None

    def drop_table(self, name: str) -> bool:
        """
        Drop a table.

        Args:
            name: Table name

        Returns:
            True if dropped, False if didn't exist
        """
        try:
            # Validate table name to prevent SQL injection
            validated_name = self._validate_table_name(name)
            with self.engine.begin() as conn:
                conn.execute(text(f"DROP TABLE IF EXISTS {validated_name}"))
                conn.execute(
                    text("DELETE FROM _constat_table_registry WHERE table_name = :name"),
                    {"name": validated_name}
                )
            return True
        except ValueError as e:
            logger.warning(f"Invalid table name '{name}': {e}")
            return False
        except Exception as e:
            logger.debug(f"Failed to drop table '{name}': {e}")
            return False

    def clear_step_data(self, step_number: int) -> None:
        """
        Clear all data created by a specific step.

        Useful when retrying or revising a step.
        """
        # Get tables created by this step
        with self.engine.connect() as conn:
            tables = conn.execute(
                text("SELECT table_name FROM _constat_table_registry WHERE step_number = :step"),
                {"step": step_number}
            ).fetchall()

        for (table_name,) in tables:
            self.drop_table(table_name)

        # Clear state variables from this step
        with self.engine.begin() as conn:
            conn.execute(
                text("DELETE FROM _constat_state WHERE step_number = :step"),
                {"step": step_number}
            )

    def export_state_summary(self) -> dict:
        """
        Export a summary of datastore state for context.

        Returns:
            Dict with tables, state variables, and their metadata
        """
        return {
            "tables": self.list_tables(),
            "state": self.get_all_state(),
            "scratchpad": self.get_scratchpad(),
            "artifacts": self.list_artifacts(),
        }

    # --- Scratchpad methods ---

    def add_scratchpad_entry(
        self,
        step_number: int,
        goal: str,
        narrative: str,
        tables_created: Optional[list[str]] = None,
        code: Optional[str] = None,
    ) -> None:
        """
        Add or update a scratchpad entry for a step.

        Args:
            step_number: Step number
            goal: Goal of the step
            narrative: Step result narrative (printed output)
            tables_created: List of table names created by this step
            code: Generated Python code for this step (for replay)
        """
        tables_str = ",".join(tables_created) if tables_created else ""
        self._upsert(
            "_constat_scratchpad",
            "step_number",
            step_number,
            {"goal": goal, "narrative": narrative, "tables_created": tables_str, "code": code or ""}
        )

    def get_scratchpad_entry(self, step_number: int) -> Optional[dict]:
        """Get scratchpad entry for a step."""
        with self.engine.connect() as conn:
            result = conn.execute(
                text("SELECT goal, narrative, tables_created, code FROM _constat_scratchpad WHERE step_number = :step"),
                {"step": step_number}
            ).fetchone()

        if result:
            return {
                "step_number": step_number,
                "goal": result[0],
                "narrative": result[1],
                "tables_created": result[2].split(",") if result[2] else [],
                "code": result[3] or "",
            }
        return None

    def get_scratchpad(self) -> list[dict]:
        """Get all scratchpad entries in order."""
        with self.engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT step_number, goal, narrative, tables_created, code
                FROM _constat_scratchpad
                ORDER BY step_number
            """)).fetchall()

        return [
            {
                "step_number": row[0],
                "goal": row[1],
                "narrative": row[2],
                "tables_created": row[3].split(",") if row[3] else [],
                "code": row[4] or "",
            }
            for row in rows
        ]

    def get_scratchpad_as_markdown(self) -> str:
        """Get scratchpad as markdown for LLM context."""
        entries = self.get_scratchpad()
        if not entries:
            return "(no previous steps)"

        parts = []
        for entry in entries:
            section = f"## Step {entry['step_number']}: {entry['goal']}\n{entry['narrative']}"
            if entry['tables_created']:
                section += f"\n\n**Tables created:** {', '.join(entry['tables_created'])}"
            parts.append(section)

        return "\n\n".join(parts)

    def get_execution_history_table(self, include_all_attempts: bool = True) -> Optional["pd.DataFrame"]:
        """
        Get execution history as a queryable DataFrame.

        Returns a table with step_number, attempt, goal, code, output, error for each execution.
        By default includes ALL attempts (both successful and failed) to show what didn't work
        alongside what did work.

        Args:
            include_all_attempts: If True (default), include all attempts including failures.
                                  If False, only include the final successful attempt per step.

        Returns:
            DataFrame with execution history, or None if no history
        """
        import pandas as pd

        # Get scratchpad entries for step info (one per step)
        scratchpad_entries = self.get_scratchpad()
        if not scratchpad_entries:
            return None

        # Get all artifacts
        code_artifacts = self.get_artifacts(artifact_type="code")
        output_artifacts = self.get_artifacts(artifact_type="output")
        error_artifacts = self.get_artifacts(artifact_type="error")

        # Build lookup by (step, attempt)
        code_by_step_attempt = {}
        for artifact in code_artifacts:
            key = (artifact.step_number, artifact.attempt)
            code_by_step_attempt[key] = artifact

        output_by_step_attempt = {}
        for artifact in output_artifacts:
            key = (artifact.step_number, artifact.attempt)
            output_by_step_attempt[key] = artifact

        error_by_step_attempt = {}
        for artifact in error_artifacts:
            key = (artifact.step_number, artifact.attempt)
            error_by_step_attempt[key] = artifact

        # Build lookup by step -> scratchpad entry
        scratchpad_by_step = {entry["step_number"]: entry for entry in scratchpad_entries}

        # Get all unique (step, attempt) combinations
        all_attempts = set(code_by_step_attempt.keys())
        all_attempts.update(output_by_step_attempt.keys())
        all_attempts.update(error_by_step_attempt.keys())

        if not include_all_attempts:
            # Filter to only latest attempt per step
            latest_by_step = {}
            for step, attempt in all_attempts:
                if step not in latest_by_step or attempt > latest_by_step[step]:
                    latest_by_step[step] = attempt
            all_attempts = {(step, attempt) for step, attempt in all_attempts
                           if attempt == latest_by_step.get(step)}

        # Build history records
        history = []
        for step_num, attempt in sorted(all_attempts):
            entry = scratchpad_by_step.get(step_num, {})
            code = code_by_step_attempt.get((step_num, attempt))
            output = output_by_step_attempt.get((step_num, attempt))
            error = error_by_step_attempt.get((step_num, attempt))

            # Determine success: has output and no error
            success = output is not None and error is None

            history.append({
                "step_number": step_num,
                "attempt": attempt,
                "success": success,
                "goal": entry.get("goal"),
                "narrative": entry.get("narrative") if success else None,
                "code": code.content if code else None,
                "output": output.content if output else None,
                "error": error.content if error else None,
                "tables_created": ", ".join(entry.get("tables_created", [])) or None if success else None,
            })

        return pd.DataFrame(history)

    def ensure_execution_history_table(self) -> bool:
        """
        Ensure execution history is available as a queryable table.

        Creates/updates the `execution_history` table with current session's
        step info and code. This table can be queried via SQL.

        Returns:
            True if table was created/updated, False if no history
        """
        df = self.get_execution_history_table()
        if df is None or df.empty:
            return False

        # Store as a regular table (not internal)
        self.save_dataframe("execution_history", df, step_number=0, description="Execution history with code and outputs")
        return True

    # --- Artifact catalog methods ---

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
    ) -> int:
        """
        Add an artifact (code, output, error, or rich content) to the catalog.

        Args:
            step_number: Step number
            attempt: Attempt number within the step
            artifact_type: Type of artifact (code, output, error, html, svg, etc.)
            content: Artifact content
            name: Unique name for the artifact (auto-generated if not provided)
            title: Human-readable title for display
            description: Description of the artifact
            content_type: MIME type override
            metadata: Additional metadata (JSON-serializable)

        Returns:
            Artifact ID
        """
        # Get next ID
        with self.engine.connect() as conn:
            result = conn.execute(
                text("SELECT COALESCE(MAX(id), 0) + 1 FROM _constat_artifacts")
            ).fetchone()
            artifact_id = result[0]

        # Auto-generate name if not provided
        if name is None:
            name = f"artifact_{artifact_id}_{artifact_type}"

        # Serialize metadata
        metadata_json = json.dumps(metadata) if metadata else None

        with self.engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO _constat_artifacts
                    (id, name, step_number, attempt, artifact_type, content_type, content, title, description, metadata_json)
                    VALUES (:id, :name, :step_number, :attempt, :artifact_type, :content_type, :content, :title, :description, :metadata_json)
                """),
                {
                    "id": artifact_id,
                    "name": name,
                    "step_number": step_number,
                    "attempt": attempt,
                    "artifact_type": artifact_type,
                    "content_type": content_type,
                    "content": content,
                    "title": title,
                    "description": description,
                    "metadata_json": metadata_json,
                }
            )

        return artifact_id

    def save_rich_artifact(
        self,
        name: str,
        artifact_type: Union[ArtifactType, str],
        content: str,
        step_number: int = 0,
        attempt: int = 1,
        title: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> Artifact:
        """
        Save a rich artifact (chart, HTML, diagram, image, etc.).

        This is the preferred method for saving artifacts from step code.

        Args:
            name: Unique name for the artifact
            artifact_type: Type of artifact (ArtifactType enum or string)
            content: Artifact content (base64 for binary types)
            step_number: Step number that created this artifact
            attempt: Attempt number within the step
            title: Human-readable title for display
            description: Description of the artifact
            metadata: Additional metadata (e.g., chart config, dimensions)

        Returns:
            Artifact object with assigned ID
        """
        # Convert string to enum if needed
        if isinstance(artifact_type, str):
            try:
                artifact_type = ArtifactType(artifact_type)
            except ValueError:
                type_str = artifact_type
                content_type = None
        else:
            type_str = artifact_type.value
            content_type = ARTIFACT_MIME_TYPES.get(artifact_type)

        artifact_id = self.add_artifact(
            step_number=step_number,
            attempt=attempt,
            artifact_type=type_str if isinstance(artifact_type, str) else artifact_type.value,
            content=content,
            name=name,
            title=title,
            description=description,
            content_type=content_type if 'content_type' in dir() else ARTIFACT_MIME_TYPES.get(artifact_type) if isinstance(artifact_type, ArtifactType) else None,
            metadata=metadata,
        )

        return Artifact(
            id=artifact_id,
            name=name,
            artifact_type=artifact_type if isinstance(artifact_type, ArtifactType) else ArtifactType.TEXT,
            content=content,
            step_number=step_number,
            attempt=attempt,
            title=title,
            description=description,
            metadata=metadata or {},
        )

    def save_chart(
        self,
        name: str,
        spec: dict,
        step_number: int = 0,
        title: Optional[str] = None,
        chart_type: str = "vega-lite",
    ) -> Artifact:
        """
        Save a chart specification (Vega-Lite, Plotly, etc.).

        Args:
            name: Unique name for the chart
            spec: Chart specification as a dictionary
            step_number: Step number that created this chart
            title: Human-readable title
            chart_type: Type of chart spec (vega-lite, plotly)

        Returns:
            Artifact object
        """
        artifact_type = ArtifactType.PLOTLY if chart_type == "plotly" else ArtifactType.CHART
        return self.save_rich_artifact(
            name=name,
            artifact_type=artifact_type,
            content=json.dumps(spec),
            step_number=step_number,
            title=title,
            metadata={"chart_type": chart_type},
        )

    def save_html(
        self,
        name: str,
        html_content: str,
        step_number: int = 0,
        title: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Artifact:
        """
        Save HTML content.

        Args:
            name: Unique name for the artifact
            html_content: HTML string
            step_number: Step number that created this
            title: Human-readable title
            description: Description

        Returns:
            Artifact object
        """
        return self.save_rich_artifact(
            name=name,
            artifact_type=ArtifactType.HTML,
            content=html_content,
            step_number=step_number,
            title=title,
            description=description,
        )

    def save_diagram(
        self,
        name: str,
        diagram_code: str,
        diagram_format: str = "mermaid",
        step_number: int = 0,
        title: Optional[str] = None,
    ) -> Artifact:
        """
        Save a diagram (Mermaid, Graphviz, etc.).

        Args:
            name: Unique name for the diagram
            diagram_code: Diagram source code
            diagram_format: Format of the diagram (mermaid, graphviz, plantuml)
            step_number: Step number
            title: Human-readable title

        Returns:
            Artifact object
        """
        type_map = {
            "mermaid": ArtifactType.MERMAID,
            "graphviz": ArtifactType.GRAPHVIZ,
            "dot": ArtifactType.GRAPHVIZ,
        }
        artifact_type = type_map.get(diagram_format, ArtifactType.DIAGRAM)

        return self.save_rich_artifact(
            name=name,
            artifact_type=artifact_type,
            content=diagram_code,
            step_number=step_number,
            title=title,
            metadata={"format": diagram_format},
        )

    def save_image(
        self,
        name: str,
        image_data: str,
        image_format: str = "png",
        step_number: int = 0,
        title: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> Artifact:
        """
        Save an image (PNG, SVG, JPEG).

        Args:
            name: Unique name for the image
            image_data: Image data (base64 for PNG/JPEG, XML for SVG)
            image_format: Format of the image (png, svg, jpeg)
            step_number: Step number
            title: Human-readable title
            width: Image width in pixels
            height: Image height in pixels

        Returns:
            Artifact object
        """
        type_map = {
            "png": ArtifactType.PNG,
            "svg": ArtifactType.SVG,
            "jpeg": ArtifactType.JPEG,
            "jpg": ArtifactType.JPEG,
        }
        artifact_type = type_map.get(image_format, ArtifactType.PNG)

        metadata = {}
        if width:
            metadata["width"] = width
        if height:
            metadata["height"] = height

        return self.save_rich_artifact(
            name=name,
            artifact_type=artifact_type,
            content=image_data,
            step_number=step_number,
            title=title,
            metadata=metadata if metadata else None,
        )

    def get_artifact_by_name(self, name: str) -> Optional[Artifact]:
        """
        Get an artifact by its name.

        Args:
            name: Artifact name

        Returns:
            Artifact object or None if not found
        """
        with self.engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT id, name, step_number, attempt, artifact_type, content_type,
                           content, title, description, metadata_json, created_at
                    FROM _constat_artifacts
                    WHERE name = :name
                """),
                {"name": name}
            ).fetchone()

        if result:
            return self._row_to_artifact(result)
        return None

    def get_artifact_by_id(self, artifact_id: int) -> Optional[Artifact]:
        """
        Get an artifact by its ID.

        Args:
            artifact_id: Artifact ID

        Returns:
            Artifact object or None if not found
        """
        with self.engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT id, name, step_number, attempt, artifact_type, content_type,
                           content, title, description, metadata_json, created_at
                    FROM _constat_artifacts
                    WHERE id = :id
                """),
                {"id": artifact_id}
            ).fetchone()

        if result:
            return self._row_to_artifact(result)
        return None

    def _row_to_artifact(self, row: tuple) -> Artifact:
        """Convert a database row to an Artifact object."""
        # Try to convert artifact_type string to enum
        type_str = row[4]
        try:
            artifact_type = ArtifactType(type_str)
        except ValueError:
            artifact_type = ArtifactType.TEXT

        # Parse metadata
        metadata = {}
        if row[9]:
            try:
                metadata = json.loads(row[9])
            except json.JSONDecodeError:
                pass

        return Artifact(
            id=row[0],
            name=row[1] or f"artifact_{row[0]}",
            step_number=row[2],
            attempt=row[3],
            artifact_type=artifact_type,
            content_type=row[5],
            content=row[6],
            title=row[7],
            description=row[8],
            metadata=metadata,
            created_at=str(row[10]) if row[10] else None,
        )

    def get_artifacts(self, step_number: Optional[int] = None, artifact_type: Optional[str] = None) -> list[Artifact]:
        """
        Get artifacts, optionally filtered by step or type.

        Args:
            step_number: Filter by step number (None for all)
            artifact_type: Filter by artifact type (None for all)

        Returns:
            List of Artifact objects
        """
        query = """
            SELECT id, name, step_number, attempt, artifact_type, content_type,
                   content, title, description, metadata_json, created_at
            FROM _constat_artifacts
        """
        params = {}
        conditions = []

        if step_number is not None:
            conditions.append("step_number = :step_number")
            params["step_number"] = step_number

        if artifact_type is not None:
            conditions.append("artifact_type = :artifact_type")
            params["artifact_type"] = artifact_type

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY step_number, attempt, id"

        with self.engine.connect() as conn:
            rows = conn.execute(text(query), params).fetchall()

        return [self._row_to_artifact(row) for row in rows]

    def list_artifacts(self, include_content: bool = False) -> list[dict]:
        """
        List artifact metadata.

        Args:
            include_content: If True, include content in results

        Returns:
            List of artifact metadata dicts
        """
        if include_content:
            with self.engine.connect() as conn:
                rows = conn.execute(text("""
                    SELECT id, name, step_number, attempt, artifact_type, content_type,
                           content, title, description, metadata_json, created_at
                    FROM _constat_artifacts
                    ORDER BY step_number, attempt, id
                """)).fetchall()

            return [self._row_to_artifact(row).to_dict() for row in rows]
        else:
            with self.engine.connect() as conn:
                rows = conn.execute(text("""
                    SELECT id, name, step_number, attempt, artifact_type, content_type,
                           LENGTH(content) as content_length, title, description, created_at
                    FROM _constat_artifacts
                    ORDER BY step_number, attempt, id
                """)).fetchall()

            return [
                {
                    "id": row[0],
                    "name": row[1],
                    "step_number": row[2],
                    "attempt": row[3],
                    "type": row[4],
                    "content_type": row[5],
                    "content_length": row[6],
                    "title": row[7],
                    "description": row[8],
                    "created_at": str(row[9]) if row[9] else None,
                }
                for row in rows
            ]

    def get_artifacts_by_type(self, artifact_type: Union[ArtifactType, str]) -> list[Artifact]:
        """
        Get all artifacts of a specific type.

        Args:
            artifact_type: The type of artifacts to retrieve

        Returns:
            List of Artifact objects
        """
        type_str = artifact_type.value if isinstance(artifact_type, ArtifactType) else artifact_type
        return self.get_artifacts(artifact_type=type_str)

    # --- Session metadata methods ---

    def set_session_meta(self, key: str, value: str) -> None:
        """Set a session metadata value (problem, status, etc.)."""
        self._upsert("_constat_session", "key", key, {"value": value})

    def get_session_meta(self, key: str) -> Optional[str]:
        """Get a session metadata value."""
        with self.engine.connect() as conn:
            result = conn.execute(
                text("SELECT value FROM _constat_session WHERE key = :key"),
                {"key": key}
            ).fetchone()
        return result[0] if result else None

    def get_all_session_meta(self) -> dict[str, str]:
        """Get all session metadata."""
        with self.engine.connect() as conn:
            rows = conn.execute(
                text("SELECT key, value FROM _constat_session")
            ).fetchall()
        return {key: value for key, value in rows}

    # --- Plan step methods ---

    def save_plan_step(
        self,
        step_number: int,
        goal: str,
        expected_inputs: Optional[list[str]] = None,
        expected_outputs: Optional[list[str]] = None,
        status: str = "pending",
    ) -> None:
        """Save a plan step."""
        self._upsert(
            "_constat_plan_steps",
            "step_number",
            step_number,
            {
                "goal": goal,
                "expected_inputs": ",".join(expected_inputs) if expected_inputs else "",
                "expected_outputs": ",".join(expected_outputs) if expected_outputs else "",
                "status": status,
            }
        )

    def update_plan_step(
        self,
        step_number: int,
        status: Optional[str] = None,
        code: Optional[str] = None,
        error: Optional[str] = None,
        attempts: Optional[int] = None,
        duration_ms: Optional[int] = None,
    ) -> None:
        """Update a plan step's execution state."""
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

        # Build and execute update query
        set_clause = ", ".join(f"{k} = :{k}" for k in updates.keys())
        if status in ("completed", "failed"):
            set_clause += ", completed_at = CURRENT_TIMESTAMP"

        updates["step_number"] = step_number

        with self.engine.begin() as conn:
            conn.execute(
                text(f"UPDATE _constat_plan_steps SET {set_clause} WHERE step_number = :step_number"),
                updates
            )

    def get_plan_steps(self) -> list[dict]:
        """Get all plan steps for UI restoration."""
        with self.engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT step_number, goal, expected_inputs, expected_outputs,
                       status, code, error, attempts, duration_ms, created_at, completed_at
                FROM _constat_plan_steps
                ORDER BY step_number
            """)).fetchall()

        return [
            {
                "step_number": row[0],
                "goal": row[1],
                "expected_inputs": row[2].split(",") if row[2] else [],
                "expected_outputs": row[3].split(",") if row[3] else [],
                "status": row[4],
                "code": row[5],
                "error": row[6],
                "attempts": row[7],
                "duration_ms": row[8],
                "created_at": str(row[9]) if row[9] else None,
                "completed_at": str(row[10]) if row[10] else None,
            }
            for row in rows
        ]

    def get_full_session_state(self) -> dict:
        """
        Get complete session state for UI restoration.

        Returns everything needed to fully restore the UI:
        - Session metadata (problem, status)
        - Plan steps with status and code
        - Tables created
        - State variables
        - Scratchpad entries
        - Artifact summaries
        """
        return {
            "session": self.get_all_session_meta(),
            "plan_steps": self.get_plan_steps(),
            "tables": self.list_tables(),
            "state": self.get_all_state(),
            "scratchpad": self.get_scratchpad(),
            "artifacts": self.list_artifacts(),
        }

    def close(self) -> None:
        """Close the database connection."""
        self.engine.dispose()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
