# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Thread-safe DuckDB connection management.

Uses a single shared connection protected by a reentrant lock.
All threads serialize access through the lock — DuckDB's C++ engine
is not safe for concurrent cursor access from a single connection.

Usage:
    pool = DuckDBConnectionPool("/path/to/db.duckdb")

    # Get the shared connection (caller must not hold it across awaits)
    conn = pool.get_connection()
    conn.execute("SELECT * FROM table")

    # Or use the context manager
    with pool.connection() as conn:
        conn.execute("SELECT * FROM table")
"""

import atexit
import logging
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Generator

import weakref

import duckdb

logger = logging.getLogger(__name__)

# Track all live pools so they can be closed before process exit
_all_pools: weakref.WeakSet["DuckDBConnectionPool"] = weakref.WeakSet()


def close_all_pools() -> None:
    """Close all live DuckDBConnectionPool instances.

    Call before process exit to avoid SIGABRT from DuckDB's C++ destructors.
    """
    for pool in list(_all_pools):
        try:
            pool.close()
        except Exception:
            pass


# Release DuckDB file locks on interpreter exit (crash, SIGTERM, etc.)
atexit.register(close_all_pools)


def _try_kill_orphan_lock_holder(error_msg: str) -> None:
    """Parse PID from DuckDB lock error and kill if it's an orphaned child process.

    sentence-transformers/torch can spawn multiprocessing children that inherit
    the DuckDB file handle and survive parent process termination.
    """
    import os
    import re
    import signal

    match = re.search(r"\(PID (\d+)\)", error_msg)
    if not match:
        return
    pid = int(match.group(1))
    if pid == os.getpid():
        return
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return
    except PermissionError:
        return

    try:
        with open(f"/proc/{pid}/cmdline", "r") as f:
            cmdline = f.read()
    except FileNotFoundError:
        import subprocess
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True, text=True, timeout=5,
        )
        cmdline = result.stdout.strip()

    if "multiprocessing" in cmdline or "resource_tracker" in cmdline:
        logger.warning(f"Killing orphaned multiprocessing child PID {pid}")
        try:
            os.kill(pid, signal.SIGKILL)
            import time
            time.sleep(0.5)
        except OSError:
            pass


class _PendingResult:
    """Holds RLock from execute() until a terminal fetch completes.

    DuckDB's execute() returns the connection for chaining — cursor state
    lives on the connection object.  The lock must span execute→fetch to
    prevent another thread's execute from clobbering the cursor between
    the two calls.

    For fire-and-forget calls (DDL / INSERT with no fetch), CPython's
    reference counting invokes __del__ immediately when the temporary
    _PendingResult goes out of scope, releasing the lock.
    """

    __slots__ = ("_conn", "_lock", "_released")

    def __init__(self, conn: duckdb.DuckDBPyConnection, lock: threading.RLock):
        self._conn = conn
        self._lock = lock  # Lock is HELD — acquired by caller
        self._released = False

    def _release(self):
        if not self._released:
            self._released = True
            self._lock.release()

    def fetchall(self):
        try:
            return self._conn.fetchall()
        finally:
            self._release()

    def fetchone(self):
        try:
            return self._conn.fetchone()
        finally:
            self._release()

    def fetchdf(self):
        try:
            return self._conn.fetchdf()
        finally:
            self._release()

    def df(self):
        try:
            return self._conn.df()
        finally:
            self._release()

    def fetchnumpy(self):
        try:
            return self._conn.fetchnumpy()
        finally:
            self._release()

    @property
    def description(self):
        return self._conn.description

    @property
    def rowcount(self):
        return self._conn.rowcount

    def __del__(self):
        self._release()

    def __getattr__(self, name):
        return getattr(self._conn, name)


class _LockedConnection:
    """Proxy that serializes execute+fetch cycles through a lock.

    execute() acquires the lock and returns a _PendingResult that holds
    it until a terminal fetch operation (fetchall, fetchone, etc.) or
    until the _PendingResult is garbage-collected.
    """

    __slots__ = ("_conn", "_lock")

    def __init__(self, conn: duckdb.DuckDBPyConnection, lock: threading.RLock):
        self._conn = conn
        self._lock = lock

    def execute(self, *args, **kwargs):
        self._lock.acquire()
        try:
            self._conn.execute(*args, **kwargs)
            return _PendingResult(self._conn, self._lock)
        except:
            self._lock.release()
            raise

    def executemany(self, *args, **kwargs):
        with self._lock:
            self._conn.executemany(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._conn, name)


class DuckDBConnectionPool:
    """Thread-safe DuckDB access using a single serialized connection.

    One duckdb.connect() is created at init. All threads share this
    single connection protected by a reentrant lock. The lock serializes
    all access — DuckDB's C++ engine crashes on concurrent cursor access.

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
        self._db_path = str(db_path)
        self._read_only = read_only
        self._config = config or {}
        self._closed = False
        self._lock = threading.RLock()

        # Single connection — retry on lock conflict
        import time
        last_err = None
        for attempt in range(5):
            try:
                self._shared_conn = duckdb.connect(
                    self._db_path,
                    read_only=self._read_only,
                    config=self._config,
                )
                last_err = None
                break
            except duckdb.IOException as e:
                if "Could not set lock" in str(e) and attempt < 4:
                    last_err = e
                    _try_kill_orphan_lock_holder(str(e))
                    logger.warning(f"DuckDB lock conflict, retrying in {attempt + 1}s...")
                    time.sleep(attempt + 1)
                else:
                    raise
        if last_err is not None:
            raise last_err
        _all_pools.add(self)

    @property
    def db_path(self) -> str:
        return self._db_path

    def get_connection(self) -> "_LockedConnection":
        """Get the single shared connection wrapped in a locking proxy.

        Returns:
            Locked proxy that serializes execute/executemany calls

        Raises:
            RuntimeError: If the pool has been closed
        """
        if self._closed:
            raise RuntimeError("Connection pool has been closed")
        return _LockedConnection(self._shared_conn, self._lock)

    @contextmanager
    def connection(self) -> Generator[duckdb.DuckDBPyConnection, None, None]:
        """Context manager for the shared connection (holds lock for duration)."""
        with self._lock:
            yield self.get_connection()

    def close(self) -> None:
        """Close the shared connection."""
        if self._closed:
            return
        self._closed = True
        try:
            self._shared_conn.close()
        except Exception as e:
            logger.warning(f"Error closing DuckDB connection: {e}")

    def __enter__(self) -> "DuckDBConnectionPool":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    @property
    def active_connections(self) -> int:
        return 0 if self._closed else 1

    # Backward compat
    @property
    def _connections(self) -> dict[int, duckdb.DuckDBPyConnection]:
        return {}


class ThreadLocalDuckDB:
    """Wrapper providing a single DuckDB connection with init_sql support.

    Despite the name (kept for backward compatibility), this now uses
    a single shared connection — no per-thread cursors.

    Usage:
        db = ThreadLocalDuckDB("/path/to/db.duckdb")
        db.conn.execute("SELECT * FROM table")
    """

    def __init__(
        self,
        db_path: str | Path,
        read_only: bool = False,
        config: Optional[dict] = None,
        init_sql: Optional[list[str]] = None,
    ):
        self._pool = DuckDBConnectionPool(db_path, read_only, config)
        self._init_sql = init_sql or []
        self._init_lock = threading.Lock()

        # Run init_sql on the shared connection
        conn = self._pool.get_connection()
        for sql in self._init_sql:
            try:
                conn.execute(sql)
            except Exception as e:
                logger.debug(f"Init SQL failed (may be expected): {e}")

    @property
    def pool(self) -> DuckDBConnectionPool:
        return self._pool

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        """Get the single shared connection."""
        return self._pool.get_connection()

    def execute(self, sql: str, params=None):
        if params:
            return self.conn.execute(sql, params)
        return self.conn.execute(sql)

    def add_init_sql(self, sql: str) -> None:
        """Run SQL on the shared connection."""
        with self._init_lock:
            self._init_sql.append(sql)
        try:
            self._pool.get_connection().execute(sql)
        except Exception as e:
            logger.debug(f"add_init_sql failed: {e}")

    def remove_init_sql(self, predicate) -> None:
        with self._init_lock:
            self._init_sql = [s for s in self._init_sql if not predicate(s)]

    def run_on_all_connections(self, sql: str) -> None:
        """Execute SQL on the shared connection.

        Kept for backward compat — with a single connection,
        this just runs on the shared connection.
        """
        try:
            self._pool.get_connection().execute(sql)
        except Exception as e:
            logger.debug(f"run_on_all_connections failed: {e}")

    def close(self) -> None:
        self._pool.close()

    def __enter__(self) -> "ThreadLocalDuckDB":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
