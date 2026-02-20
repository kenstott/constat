# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Thread-safe DuckDB connection management.

Uses a single shared connection with per-thread cursors. DuckDB cursors
created from a single connection share the same database instance and
can be used concurrently from different threads.

Usage:
    pool = DuckDBConnectionPool("/path/to/db.duckdb")

    # Get a cursor for the current thread
    with pool.connection() as conn:
        conn.execute("SELECT * FROM table")

    # Or use directly
    conn = pool.get_connection()
    conn.execute("SELECT * FROM table")

The pool automatically:
- Creates one cursor per thread (thread-local)
- Reuses cursors within the same thread
- Handles cleanup on close()
"""

import logging
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Generator

import duckdb

logger = logging.getLogger(__name__)


class DuckDBConnectionPool:
    """Thread-safe DuckDB connection pool using a single shared connection.

    A single duckdb.connect() is created at init. Each thread gets a cursor
    from that connection via conn.cursor(). Cursors share the database state
    (tables, extensions, ATTACHed databases) and can execute concurrently.

    Attributes:
        _db_path: Path to the DuckDB database file
        _read_only: Whether the connection is read-only
    """

    def __init__(
        self,
        db_path: str | Path,
        read_only: bool = False,
        config: Optional[dict] = None,
    ):
        """Initialize the connection pool.

        Args:
            db_path: Path to DuckDB database file (or ":memory:" for in-memory)
            read_only: Open connection in read-only mode
            config: Optional DuckDB configuration dict
        """
        self._db_path = str(db_path)
        self._read_only = read_only
        self._config = config or {}
        self._local = threading.local()
        self._cursors: dict[int, duckdb.DuckDBPyConnection] = {}
        self._lock = threading.Lock()
        self._closed = False

        # Single shared connection — all cursors derive from this
        self._shared_conn = duckdb.connect(
            self._db_path,
            read_only=self._read_only,
            config=self._config,
        )

    @property
    def db_path(self) -> str:
        """Get the database path."""
        return self._db_path

    def get_connection(self) -> duckdb.DuckDBPyConnection:
        """Get or create a cursor for the current thread.

        Returns:
            DuckDB cursor for the current thread

        Raises:
            RuntimeError: If the pool has been closed
        """
        if self._closed:
            raise RuntimeError("Connection pool has been closed")

        thread_id = threading.get_ident()

        # Check thread-local cache first (fast path)
        cursor = getattr(self._local, 'connection', None)
        if cursor is not None:
            try:
                # Verify cursor is still valid
                cursor.execute("SELECT 1")
                return cursor
            except duckdb.Error:
                # Cursor is dead, remove it
                logger.debug(f"Thread {thread_id}: cursor dead, creating new one")
                self._remove_connection(thread_id)

        # Create new cursor from shared connection (slow path)
        with self._lock:
            if self._closed:
                raise RuntimeError("Connection pool has been closed")

            cursor = self._shared_conn.cursor()
            self._local.connection = cursor
            self._cursors[thread_id] = cursor
            logger.debug(f"Thread {thread_id}: created new DuckDB cursor")
            return cursor

    def _remove_connection(self, thread_id: int) -> None:
        """Remove a cursor from tracking."""
        with self._lock:
            if thread_id in self._cursors:
                try:
                    self._cursors[thread_id].close()
                except duckdb.Error:
                    pass
                del self._cursors[thread_id]
            if hasattr(self._local, 'connection'):
                delattr(self._local, 'connection')

    @contextmanager
    def connection(self) -> Generator[duckdb.DuckDBPyConnection, None, None]:
        """Context manager for getting a thread-local cursor.

        Yields:
            DuckDB cursor for the current thread

        Example:
            with pool.connection() as conn:
                result = conn.execute("SELECT * FROM table").fetchall()
        """
        yield self.get_connection()

    def close(self) -> None:
        """Close all cursors and the shared connection.

        After calling close(), the pool cannot be used again.
        """
        with self._lock:
            self._closed = True
            for thread_id, cursor in list(self._cursors.items()):
                try:
                    cursor.close()
                    logger.debug(f"Thread {thread_id}: closed DuckDB cursor")
                except Exception as e:
                    logger.warning(f"Error closing cursor for thread {thread_id}: {e}")
            self._cursors.clear()
            try:
                self._shared_conn.close()
            except Exception as e:
                logger.warning(f"Error closing shared DuckDB connection: {e}")

    def __enter__(self) -> "DuckDBConnectionPool":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    @property
    def active_connections(self) -> int:
        """Get the number of active cursors."""
        with self._lock:
            return len(self._cursors)

    # Backward compat: _connections used by ThreadLocalDuckDB.run_on_all_connections
    @property
    def _connections(self) -> dict[int, duckdb.DuckDBPyConnection]:
        return self._cursors


class ThreadLocalDuckDB:
    """Simpler interface for single-database thread-local cursors.

    This is a convenience wrapper that provides a connection property
    that automatically returns the correct cursor for the current thread.

    Usage:
        db = ThreadLocalDuckDB("/path/to/db.duckdb")
        db.conn.execute("SELECT * FROM table")  # Thread-safe

        # Or with the pool directly
        with db.pool.connection() as conn:
            conn.execute("...")
    """

    def __init__(
        self,
        db_path: str | Path,
        read_only: bool = False,
        config: Optional[dict] = None,
        init_sql: Optional[list[str]] = None,
    ):
        """Initialize thread-local DuckDB wrapper.

        Args:
            db_path: Path to DuckDB database file
            read_only: Open connection in read-only mode
            config: Optional DuckDB configuration dict
            init_sql: SQL statements to run on the shared connection
        """
        self._pool = DuckDBConnectionPool(db_path, read_only, config)
        self._init_sql = init_sql or []
        self._initialized_threads: set[int] = set()
        self._init_lock = threading.Lock()

        # Run init_sql on the shared connection (extensions, ATTACHes, etc.)
        for sql in self._init_sql:
            try:
                self._pool._shared_conn.execute(sql)
            except Exception as e:
                logger.debug(f"Init SQL failed (may be expected): {e}")

    @property
    def pool(self) -> DuckDBConnectionPool:
        """Get the underlying connection pool."""
        return self._pool

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        """Get the cursor for the current thread."""
        return self._pool.get_connection()

    def execute(self, sql: str, params=None):
        """Execute SQL on the current thread's cursor."""
        if params:
            return self.conn.execute(sql, params)
        return self.conn.execute(sql)

    def add_init_sql(self, sql: str) -> None:
        """Run SQL on the shared connection (visible to all cursors)."""
        with self._init_lock:
            self._init_sql.append(sql)
        # Execute on shared connection — all cursors see the result
        try:
            self._pool._shared_conn.execute(sql)
        except Exception as e:
            logger.debug(f"add_init_sql failed: {e}")

    def remove_init_sql(self, predicate) -> None:
        """Remove init_sql entries where predicate(sql) is True."""
        with self._init_lock:
            self._init_sql = [s for s in self._init_sql if not predicate(s)]

    def run_on_all_connections(self, sql: str) -> None:
        """Execute SQL on all cursors and the shared connection.

        Cursors hold their own file handles for ATTACHed databases,
        so operations like DETACH must run on cursors too.
        """
        for tid, cursor in list(self._pool._cursors.items()):
            try:
                cursor.execute(sql)
            except Exception:
                pass
        try:
            self._pool._shared_conn.execute(sql)
        except Exception as e:
            logger.debug(f"run_on_all_connections failed: {e}")

    def close(self) -> None:
        """Close all cursors and the shared connection."""
        self._pool.close()

    def __enter__(self) -> "ThreadLocalDuckDB":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
