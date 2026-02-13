# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Thread-safe DuckDB connection management.

DuckDB connections are NOT thread-safe. This module provides thread-local
connection pooling to ensure each thread has its own connection to the
database file.

Usage:
    pool = DuckDBConnectionPool("/path/to/db.duckdb")

    # Get a connection for the current thread
    with pool.connection() as conn:
        conn.execute("SELECT * FROM table")

    # Or use directly
    conn = pool.get_connection()
    try:
        conn.execute("SELECT * FROM table")
    finally:
        # Connection stays open for thread reuse
        pass

The pool automatically:
- Creates one connection per thread (thread-local)
- Reuses connections within the same thread
- Handles connection cleanup on close()
"""

import logging
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Generator

import duckdb

logger = logging.getLogger(__name__)


class DuckDBConnectionPool:
    """Thread-safe DuckDB connection pool using thread-local storage.

    Each thread gets its own connection to the database file. Connections
    are reused within the same thread for efficiency.

    Attributes:
        _db_path: Path to the DuckDB database file
        _read_only: Whether connections should be read-only
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
            read_only: Open connections in read-only mode
            config: Optional DuckDB configuration dict
        """
        self._db_path = str(db_path)
        self._read_only = read_only
        self._config = config or {}
        self._local = threading.local()
        self._connections: dict[int, duckdb.DuckDBPyConnection] = {}
        self._lock = threading.Lock()
        self._closed = False

    @property
    def db_path(self) -> str:
        """Get the database path."""
        return self._db_path

    def get_connection(self) -> duckdb.DuckDBPyConnection:
        """Get or create a connection for the current thread.

        Returns:
            DuckDB connection for the current thread

        Raises:
            RuntimeError: If the pool has been closed
        """
        if self._closed:
            raise RuntimeError("Connection pool has been closed")

        thread_id = threading.get_ident()

        # Check thread-local cache first (fast path)
        conn = getattr(self._local, 'connection', None)
        if conn is not None:
            try:
                # Verify connection is still valid
                conn.execute("SELECT 1")
                return conn
            except Exception:
                # Connection is dead, remove it
                logger.debug(f"Thread {thread_id}: connection dead, creating new one")
                self._remove_connection(thread_id)

        # Create new connection (slow path)
        with self._lock:
            if self._closed:
                raise RuntimeError("Connection pool has been closed")

            conn = self._create_connection()
            self._local.connection = conn
            self._connections[thread_id] = conn
            logger.debug(f"Thread {thread_id}: created new DuckDB connection")
            return conn

    def _create_connection(self) -> duckdb.DuckDBPyConnection:
        """Create a new DuckDB connection."""
        return duckdb.connect(
            self._db_path,
            read_only=self._read_only,
            config=self._config,
        )

    def _remove_connection(self, thread_id: int) -> None:
        """Remove a connection from tracking."""
        with self._lock:
            if thread_id in self._connections:
                try:
                    self._connections[thread_id].close()
                except Exception:
                    pass
                del self._connections[thread_id]
            if hasattr(self._local, 'connection'):
                delattr(self._local, 'connection')

    @contextmanager
    def connection(self) -> Generator[duckdb.DuckDBPyConnection, None, None]:
        """Context manager for getting a thread-local connection.

        Yields:
            DuckDB connection for the current thread

        Example:
            with pool.connection() as conn:
                result = conn.execute("SELECT * FROM table").fetchall()
        """
        yield self.get_connection()

    def close(self) -> None:
        """Close all connections in the pool.

        After calling close(), the pool cannot be used again.
        """
        with self._lock:
            self._closed = True
            for thread_id, conn in list(self._connections.items()):
                try:
                    conn.close()
                    logger.debug(f"Thread {thread_id}: closed DuckDB connection")
                except Exception as e:
                    logger.warning(f"Error closing connection for thread {thread_id}: {e}")
            self._connections.clear()

    def __enter__(self) -> "DuckDBConnectionPool":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    @property
    def active_connections(self) -> int:
        """Get the number of active connections."""
        with self._lock:
            return len(self._connections)


class ThreadLocalDuckDB:
    """Simpler interface for single-database thread-local connections.

    This is a convenience wrapper that provides a connection property
    that automatically returns the correct connection for the current thread.

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
            read_only: Open connections in read-only mode
            config: Optional DuckDB configuration dict
            init_sql: SQL statements to run on each new connection
        """
        self._pool = DuckDBConnectionPool(db_path, read_only, config)
        self._init_sql = init_sql or []
        self._initialized_threads: set[int] = set()
        self._init_lock = threading.Lock()

    @property
    def pool(self) -> DuckDBConnectionPool:
        """Get the underlying connection pool."""
        return self._pool

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        """Get the connection for the current thread.

        Runs init_sql on first access for each thread.
        """
        conn = self._pool.get_connection()
        thread_id = threading.get_ident()

        # Run init SQL for new threads
        if thread_id not in self._initialized_threads:
            with self._init_lock:
                if thread_id not in self._initialized_threads:
                    for sql in self._init_sql:
                        try:
                            conn.execute(sql)
                        except Exception as e:
                            logger.debug(f"Init SQL failed (may be expected): {e}")
                    self._initialized_threads.add(thread_id)

        return conn

    def execute(self, sql: str, params=None):
        """Execute SQL on the current thread's connection."""
        if params:
            return self.conn.execute(sql, params)
        return self.conn.execute(sql)

    def close(self) -> None:
        """Close all connections."""
        self._pool.close()

    def __enter__(self) -> "ThreadLocalDuckDB":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()