# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""SQL transpilation wrapper for cross-dialect compatibility.

Wraps SQLAlchemy connections to automatically transpile SQL from a canonical
dialect (PostgreSQL) to the target database's dialect using SQLGlot.

This allows LLM-generated code to use PostgreSQL syntax universally, with
automatic translation to SQLite, MySQL, DuckDB, etc.
"""

import logging
from typing import Union

import pandas as pd

logger = logging.getLogger(__name__)

# SQLGlot dialect mapping from SQLAlchemy driver names
DIALECT_MAP = {
    # SQLAlchemy driver -> SQLGlot dialect
    "sqlite": "sqlite",
    "postgresql": "postgres",
    "postgres": "postgres",
    "psycopg2": "postgres",
    "mysql": "mysql",
    "mysqldb": "mysql",
    "pymysql": "mysql",
    "mariadb": "mysql",
    "duckdb": "duckdb",
    "snowflake": "snowflake",
    "bigquery": "bigquery",
    "redshift": "redshift",
    "presto": "presto",
    "trino": "trino",
    "mssql": "tsql",
    "pyodbc": "tsql",
    "oracle": "oracle",
    "clickhouse": "clickhouse",
    "databricks": "databricks",
    "spark": "spark",
    "hive": "hive",
    "athena": "athena",
}

# Canonical dialect - LLM generates SQL in this dialect
CANONICAL_DIALECT = "postgres"

# Custom SQLite mappings for functions SQLGlot doesn't handle well
# Maps regex pattern -> replacement function
SQLITE_CUSTOM_MAPPINGS = [
    # TIMESTAMP_TRUNC(col, MONTH) -> DATE(col, 'start of month')
    (
        r"TIMESTAMP_TRUNC\s*\(\s*([^,]+)\s*,\s*MONTH\s*\)",
        lambda m: f"DATE({m.group(1)}, 'start of month')"
    ),
    # TIMESTAMP_TRUNC(col, YEAR) -> DATE(col, 'start of year')
    (
        r"TIMESTAMP_TRUNC\s*\(\s*([^,]+)\s*,\s*YEAR\s*\)",
        lambda m: f"DATE({m.group(1)}, 'start of year')"
    ),
    # TIMESTAMP_TRUNC(col, DAY) -> DATE(col)
    (
        r"TIMESTAMP_TRUNC\s*\(\s*([^,]+)\s*,\s*DAY\s*\)",
        lambda m: f"DATE({m.group(1)})"
    ),
    # TIMESTAMP_TRUNC(col, WEEK) -> DATE(col, 'weekday 0', '-6 days')
    (
        r"TIMESTAMP_TRUNC\s*\(\s*([^,]+)\s*,\s*WEEK\s*\)",
        lambda m: f"DATE({m.group(1)}, 'weekday 0', '-6 days')"
    ),
    # TIMESTAMP_TRUNC(col, QUARTER) -> DATE(col, 'start of month', '-' || ((CAST(STRFTIME('%m', col) AS INTEGER) - 1) % 3) || ' months')
    # Simplified: just truncate to month for now
    (
        r"TIMESTAMP_TRUNC\s*\(\s*([^,]+)\s*,\s*QUARTER\s*\)",
        lambda m: f"DATE({m.group(1)}, 'start of month', '-' || ((CAST(STRFTIME('%m', {m.group(1)}) AS INTEGER) - 1) % 3) || ' months')"
    ),
    # TIMESTAMP_TRUNC(col, HOUR) -> DATETIME(col, 'start of hour') - but SQLite doesn't have this
    # Use: STRFTIME('%Y-%m-%d %H:00:00', col)
    (
        r"TIMESTAMP_TRUNC\s*\(\s*([^,]+)\s*,\s*HOUR\s*\)",
        lambda m: f"STRFTIME('%Y-%m-%d %H:00:00', {m.group(1)})"
    ),
    # EXTRACT(EPOCH FROM col) -> UNIXEPOCH(col) for SQLite 3.38+, or (JULIANDAY(col) - 2440587.5) * 86400
    (
        r"EXTRACT\s*\(\s*EPOCH\s+FROM\s+([^)]+)\)",
        lambda m: f"UNIXEPOCH({m.group(1)})"
    ),
    # DATE_ADD / DATE_SUB with INTERVAL - SQLite uses DATE(col, '+N units')
    (
        r"DATE_ADD\s*\(\s*([^,]+)\s*,\s*INTERVAL\s+(\d+)\s+(DAY|MONTH|YEAR)S?\s*\)",
        lambda m: f"DATE({m.group(1)}, '+{m.group(2)} {m.group(3).lower()}s')"
    ),
    (
        r"DATE_SUB\s*\(\s*([^,]+)\s*,\s*INTERVAL\s+(\d+)\s+(DAY|MONTH|YEAR)S?\s*\)",
        lambda m: f"DATE({m.group(1)}, '-{m.group(2)} {m.group(3).lower()}s')"
    ),
]


def _apply_sqlite_mappings(sql: str) -> str:
    """Apply custom SQLite function mappings.

    SQLGlot doesn't always produce valid SQLite for certain functions.
    This applies regex-based fixes for known issues.
    """
    import re

    result = sql
    for pattern, replacement in SQLITE_CUSTOM_MAPPINGS:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    return result


def _apply_custom_mappings(sql: str, dialect: str) -> str:
    """Apply dialect-specific custom mappings after SQLGlot transpilation."""
    if dialect == "sqlite":
        return _apply_sqlite_mappings(sql)
    # Add other dialect-specific mappings here as needed
    return sql


def detect_dialect(engine) -> str:
    """Detect SQLGlot dialect from SQLAlchemy engine.

    Args:
        engine: SQLAlchemy Engine object

    Returns:
        SQLGlot dialect name (e.g., 'sqlite', 'postgres', 'mysql')
    """
    try:
        # Get the dialect name from the engine
        dialect_name = engine.dialect.name.lower()

        # Map to SQLGlot dialect
        sqlglot_dialect = DIALECT_MAP.get(dialect_name)

        if sqlglot_dialect:
            return sqlglot_dialect

        # Try driver name as fallback
        driver = getattr(engine.dialect, 'driver', None)
        if driver:
            sqlglot_dialect = DIALECT_MAP.get(driver.lower())
            if sqlglot_dialect:
                return sqlglot_dialect

        logger.warning(f"Unknown dialect '{dialect_name}', using as-is")
        return dialect_name

    except Exception as e:
        logger.warning(f"Failed to detect dialect: {e}, defaulting to postgres")
        return "postgres"


def transpile_sql(
    sql: str,
    target_dialect: str,
    source_dialect: str = CANONICAL_DIALECT,
) -> str:
    """Transpile SQL from source dialect to target dialect.

    Args:
        sql: SQL query string
        target_dialect: Target SQLGlot dialect (e.g., 'sqlite', 'mysql')
        source_dialect: Source dialect (default: postgres)

    Returns:
        Transpiled SQL string

    Note:
        If transpilation fails, returns original SQL with a warning.
        Custom mappings are applied after SQLGlot for dialect-specific fixes.
    """
    # Skip transpilation if source == target
    if source_dialect == target_dialect:
        return sql

    try:
        import sqlglot

        # Transpile the query using SQLGlot
        result = sqlglot.transpile(
            sql,
            read=source_dialect,
            write=target_dialect,
            pretty=False,  # Keep compact for execution
        )

        if result:
            transpiled = result[0]

            # Apply custom dialect-specific mappings (fixes SQLGlot gaps)
            transpiled = _apply_custom_mappings(transpiled, target_dialect)

            if transpiled != sql:
                logger.debug(f"Transpiled SQL: {sql[:100]}... -> {transpiled[:100]}...")
            return transpiled

        return sql

    except ImportError:
        logger.warning("sqlglot not installed, skipping transpilation")
        return sql

    except Exception as e:
        # Log warning but don't fail - return original SQL
        logger.warning(f"SQL transpilation failed: {e}. Using original SQL.")
        return sql


class TranspilingConnection:
    """Wrapper around SQLAlchemy Engine that auto-transpiles SQL.

    Intercepts SQL queries and transpiles them from PostgreSQL syntax
    to the target database's dialect before execution.

    Usage:
        # Wrap an existing engine
        engine = create_engine("sqlite:///mydb.db")
        db = TranspilingConnection(engine)

        # LLM generates PostgreSQL-style SQL
        df = pd.read_sql("SELECT DATE_TRUNC('month', created_at) FROM orders", db)
        # Automatically transpiled to: SELECT strftime('%Y-%m-01', created_at) FROM orders

    The wrapper is transparent to pandas - it implements the connection
    protocol that pd.read_sql() expects.
    """

    def __init__(self, engine, source_dialect: str = CANONICAL_DIALECT):
        """Initialize the transpiling connection.

        Args:
            engine: SQLAlchemy Engine object
            source_dialect: Dialect of incoming SQL (default: postgres)
        """
        self._engine = engine
        self._source_dialect = source_dialect
        self._target_dialect = detect_dialect(engine)
        self._transpilation_enabled = self._source_dialect != self._target_dialect

        if self._transpilation_enabled:
            logger.info(
                f"SQL transpilation enabled: {source_dialect} -> {self._target_dialect}"
            )

    @property
    def engine(self):
        """Get the underlying SQLAlchemy engine."""
        return self._engine

    @property
    def target_dialect(self) -> str:
        """Get the target database dialect."""
        return self._target_dialect

    @property
    def source_dialect(self) -> str:
        """Get the expected source SQL dialect."""
        return self._source_dialect

    def transpile(self, sql: str) -> str:
        """Transpile SQL from source to target dialect.

        Args:
            sql: SQL query in source dialect

        Returns:
            SQL query in target dialect
        """
        if not self._transpilation_enabled:
            return sql
        return transpile_sql(sql, self._target_dialect, self._source_dialect)

    # -------------------------------------------------------------------------
    # SQLAlchemy Engine interface (passthrough)
    # These methods allow the wrapper to be used anywhere an Engine is expected
    # -------------------------------------------------------------------------

    def connect(self):
        """Get a connection from the engine."""
        return self._engine.connect()

    def begin(self):
        """Begin a transaction."""
        return self._engine.begin()

    def execute(self, sql, *args, **kwargs):
        """Execute SQL (with transpilation).

        Note: In SQLAlchemy 2.0, you should use connection.execute() instead.
        This is provided for compatibility.
        """
        transpiled = self.transpile(str(sql))
        with self._engine.connect() as conn:
            from sqlalchemy import text
            return conn.execute(text(transpiled), *args, **kwargs)

    @property
    def dialect(self):
        """Get the engine's dialect."""
        return self._engine.dialect

    @property
    def url(self):
        """Get the engine's URL."""
        return self._engine.url

    @property
    def name(self):
        """Get the engine's name."""
        return self._engine.name

    @property
    def pool(self):
        """Get the engine's connection pool."""
        return self._engine.pool

    def dispose(self):
        """Dispose of the connection pool."""
        return self._engine.dispose()

    def raw_connection(self):
        """Get a raw DBAPI connection."""
        return self._engine.raw_connection()

    def cursor(self):
        """Get a DBAPI cursor - needed for pandas read_sql compatibility."""
        return self.raw_connection().cursor()

    def get_execution_options(self):
        """Get execution options."""
        return self._engine.get_execution_options()

    def __repr__(self):
        return (
            f"TranspilingConnection("
            f"engine={self._engine!r}, "
            f"transpile={self._source_dialect}->{self._target_dialect})"
        )

    # -------------------------------------------------------------------------
    # Pandas compatibility
    # pd.read_sql() checks for these attributes/methods
    # -------------------------------------------------------------------------

    def rollback(self):
        """Rollback - required by pandas when read_sql fails."""
        # TranspilingConnection wraps an Engine, not a Connection.
        # Engines don't have rollback, but pandas calls it on error.
        # No-op is safe because each read_sql opens its own connection.
        pass

    def commit(self):
        """Commit - required by some pandas code paths."""
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, *args):
        """Context manager exit."""
        pass


def read_sql_transpiled(
    sql: str,
    con: Union[TranspilingConnection, "Engine"],
    **kwargs,
) -> pd.DataFrame:
    """Execute SQL with automatic transpilation.

    Drop-in replacement for pd.read_sql() that handles transpilation.

    Args:
        sql: SQL query (in PostgreSQL dialect)
        con: TranspilingConnection or SQLAlchemy Engine
        **kwargs: Additional arguments passed to pd.read_sql()

    Returns:
        pandas DataFrame with query results
    """
    if isinstance(con, TranspilingConnection):
        transpiled_sql = con.transpile(sql)
        return pd.read_sql(transpiled_sql, con.engine, **kwargs)
    else:
        # Fallback: no transpilation if not wrapped
        return pd.read_sql(sql, con, **kwargs)


def create_sql_helper(con: Union[TranspilingConnection, "Engine"]):
    """Create a sql() helper function for a database connection.

    Returns a function that can be called with SQL queries and returns DataFrames.
    Automatically transpiles SQL from PostgreSQL to the target dialect.

    Usage:
        sql = create_sql_helper(db_inventory)
        df = sql("SELECT DATE_TRUNC('month', created_at) FROM orders")

    Args:
        con: TranspilingConnection or SQLAlchemy Engine

    Returns:
        A function sql(query) -> pd.DataFrame
    """
    def sql(query: str, **kwargs) -> pd.DataFrame:
        return read_sql_transpiled(query, con, **kwargs)
    return sql


# Convenience function for testing
def test_transpilation():
    """Test transpilation with example queries."""
    test_cases = [
        # (source_sql, target_dialect, expected_pattern)
        ("SELECT DATE_TRUNC('month', created_at) FROM orders", "sqlite", "start of month"),
        ("SELECT DATE_TRUNC('year', created_at) FROM orders", "sqlite", "start of year"),
        ("SELECT DATE_TRUNC('day', created_at) FROM orders", "sqlite", "DATE("),
        ("SELECT NOW() FROM users", "sqlite", "CURRENT_TIMESTAMP"),
        ("SELECT 'a' || 'b'", "mysql", "CONCAT"),
        ("SELECT x::int FROM t", "mysql", "CAST"),
        ("SELECT EXTRACT(EPOCH FROM created_at) FROM orders", "sqlite", "UNIXEPOCH"),
    ]

    print("SQL Transpilation Test Cases:")
    print("-" * 60)

    all_passed = True
    for sql, target, expected in test_cases:
        result = transpile_sql(sql, target, "postgres")
        status = "OK" if expected.lower() in result.lower() else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"[{status}] {target}: {sql}")
        print(f"      -> {result}")
        print()

    return all_passed
