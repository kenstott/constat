# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""DuckDB-backed bug queue for tracking test failures and resolutions."""

from __future__ import annotations

import json
import logging
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

import duckdb

logger = logging.getLogger(__name__)

_VALID_STATUSES = frozenset({"open", "in_progress", "resolved", "escalated"})
_VALID_PRIORITIES = frozenset({"p0", "p1", "p2"})
_AUTO_ESCALATE_THRESHOLD = 3


class BugQueue:
    """DuckDB-backed CRUD for the test failure bug queue."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn: duckdb.DuckDBPyConnection = duckdb.connect(str(self._db_path))
        self._init_tables()

    def _init_tables(self) -> None:
        with self._locked_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS bug_queue (
                    id              VARCHAR PRIMARY KEY,
                    test_name       VARCHAR NOT NULL,
                    test_file       VARCHAR,
                    module          VARCHAR,
                    error_type      VARCHAR NOT NULL,
                    error_message   VARCHAR,
                    traceback       VARCHAR,
                    expected        VARCHAR,  -- JSON
                    actual          VARCHAR,  -- JSON
                    reproduction_cmd VARCHAR,
                    commit_hash     VARCHAR,
                    environment     VARCHAR,  -- JSON
                    status          VARCHAR NOT NULL DEFAULT 'open',
                    priority        VARCHAR NOT NULL DEFAULT 'p1',
                    assigned_to     VARCHAR,
                    attempt_count   INTEGER NOT NULL DEFAULT 1,
                    resolution      VARCHAR,
                    resolution_commit VARCHAR,
                    tags            VARCHAR,  -- JSON
                    created_at      TIMESTAMP NOT NULL,
                    updated_at      TIMESTAMP NOT NULL,
                    resolved_at     TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS uq_bug_dedup
                ON bug_queue (test_name, commit_hash, error_type)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS bug_transitions (
                    id          VARCHAR PRIMARY KEY,
                    bug_id      VARCHAR NOT NULL REFERENCES bug_queue(id),
                    from_status VARCHAR,
                    to_status   VARCHAR NOT NULL,
                    agent_id    VARCHAR,
                    note        VARCHAR,
                    created_at  TIMESTAMP NOT NULL
                )
            """)

    @contextmanager
    def _locked_conn(self) -> Generator[duckdb.DuckDBPyConnection, None, None]:
        with self._lock:
            if self._conn is None:
                raise RuntimeError("BugQueue connection is closed")
            yield self._conn

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def upsert_bug(
        self,
        *,
        test_name: str,
        error_type: str,
        commit_hash: str,
        test_file: str | None = None,
        module: str | None = None,
        error_message: str | None = None,
        traceback: str | None = None,
        expected: Any = None,
        actual: Any = None,
        reproduction_cmd: str | None = None,
        environment: dict[str, Any] | None = None,
        priority: str = "p1",
        tags: list[str] | None = None,
    ) -> str:
        """Idempotent insert/update. Dedup on (test_name, commit_hash, error_type).

        Increments attempt_count on existing open bugs. Auto-escalates if
        attempt_count >= _AUTO_ESCALATE_THRESHOLD.

        Returns the bug ID.
        """
        if priority not in _VALID_PRIORITIES:
            raise ValueError(f"Invalid priority '{priority}', must be one of {_VALID_PRIORITIES}")

        now = datetime.now(timezone.utc)
        bug_id = str(uuid.uuid4())

        with self._locked_conn() as conn:
            # Check for existing bug with same dedup key
            existing = conn.execute(
                """
                SELECT id, attempt_count, status
                FROM bug_queue
                WHERE test_name = ? AND commit_hash = ? AND error_type = ?
                """,
                [test_name, commit_hash, error_type],
            ).fetchone()

            if existing:
                existing_id, attempt_count, status = existing
                new_count = attempt_count + 1
                new_status = status
                if new_count >= _AUTO_ESCALATE_THRESHOLD and status == "open":
                    new_status = "escalated"
                conn.execute(
                    """
                    UPDATE bug_queue
                    SET attempt_count = ?,
                        status = ?,
                        error_message = COALESCE(?, error_message),
                        traceback = COALESCE(?, traceback),
                        updated_at = ?
                    WHERE id = ?
                    """,
                    [new_count, new_status, error_message, traceback, now, existing_id],
                )
                if new_status != status:
                    self._record_transition(
                        conn, existing_id, status, new_status, None, "auto-escalated"
                    )
                return existing_id

            conn.execute(
                """
                INSERT INTO bug_queue (
                    id, test_name, test_file, module, error_type, error_message,
                    traceback, expected, actual, reproduction_cmd, commit_hash,
                    environment, status, priority, attempt_count, tags,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, 1, ?, ?, ?)
                """,
                [
                    bug_id,
                    test_name,
                    test_file,
                    module,
                    error_type,
                    error_message,
                    traceback,
                    json.dumps(expected) if expected is not None else None,
                    json.dumps(actual) if actual is not None else None,
                    reproduction_cmd,
                    commit_hash,
                    json.dumps(environment) if environment else None,
                    priority,
                    json.dumps(tags) if tags else None,
                    now,
                    now,
                ],
            )
            self._record_transition(conn, bug_id, None, "open", None, "auto-filed")
            return bug_id

    def claim_bug(self, bug_id: str, agent_id: str) -> None:
        """Set status=in_progress and assigned_to for a bug."""
        with self._locked_conn() as conn:
            row = conn.execute(
                "SELECT status FROM bug_queue WHERE id = ?", [bug_id]
            ).fetchone()
            if row is None:
                raise KeyError(f"Bug {bug_id} not found")
            old_status = row[0]
            now = datetime.now(timezone.utc)
            conn.execute(
                """
                UPDATE bug_queue
                SET status = 'in_progress', assigned_to = ?, updated_at = ?
                WHERE id = ?
                """,
                [agent_id, now, bug_id],
            )
            self._record_transition(conn, bug_id, old_status, "in_progress", agent_id, "claimed")

    def resolve_bug(
        self, bug_id: str, resolution: str, resolution_commit: str | None = None
    ) -> None:
        """Mark a bug as resolved."""
        with self._locked_conn() as conn:
            row = conn.execute(
                "SELECT status, assigned_to FROM bug_queue WHERE id = ?", [bug_id]
            ).fetchone()
            if row is None:
                raise KeyError(f"Bug {bug_id} not found")
            old_status, agent_id = row
            now = datetime.now(timezone.utc)
            conn.execute(
                """
                UPDATE bug_queue
                SET status = 'resolved', resolution = ?, resolution_commit = ?,
                    resolved_at = ?, updated_at = ?
                WHERE id = ?
                """,
                [resolution, resolution_commit, now, now, bug_id],
            )
            self._record_transition(
                conn, bug_id, old_status, "resolved", agent_id, resolution
            )

    def transition(
        self,
        bug_id: str,
        to_status: str,
        agent_id: str | None = None,
        note: str | None = None,
    ) -> None:
        """Transition a bug to a new status."""
        if to_status not in _VALID_STATUSES:
            raise ValueError(f"Invalid status '{to_status}', must be one of {_VALID_STATUSES}")
        with self._locked_conn() as conn:
            row = conn.execute(
                "SELECT status FROM bug_queue WHERE id = ?", [bug_id]
            ).fetchone()
            if row is None:
                raise KeyError(f"Bug {bug_id} not found")
            old_status = row[0]
            now = datetime.now(timezone.utc)
            updates = {"status": to_status, "updated_at": now}
            if to_status == "resolved":
                updates["resolved_at"] = now
            conn.execute(
                """
                UPDATE bug_queue
                SET status = ?, updated_at = ?, resolved_at = CASE WHEN ? = 'resolved' THEN ? ELSE resolved_at END
                WHERE id = ?
                """,
                [to_status, now, to_status, now, bug_id],
            )
            self._record_transition(conn, bug_id, old_status, to_status, agent_id, note)

    def get_bug(self, bug_id: str) -> dict[str, Any]:
        """Fetch a single bug by ID."""
        with self._locked_conn() as conn:
            row = conn.execute(
                "SELECT * FROM bug_queue WHERE id = ?", [bug_id]
            ).fetchone()
            if row is None:
                raise KeyError(f"Bug {bug_id} not found")
            columns = [desc[0] for desc in conn.description]
            return self._row_to_dict(columns, row)

    def list_bugs(
        self,
        *,
        status: str | None = None,
        priority: str | None = None,
        module: str | None = None,
    ) -> list[dict[str, Any]]:
        """Query bugs with optional filters."""
        clauses: list[str] = []
        params: list[Any] = []
        if status is not None:
            if status not in _VALID_STATUSES:
                raise ValueError(f"Invalid status '{status}'")
            clauses.append("status = ?")
            params.append(status)
        if priority is not None:
            if priority not in _VALID_PRIORITIES:
                raise ValueError(f"Invalid priority '{priority}'")
            clauses.append("priority = ?")
            params.append(priority)
        if module is not None:
            clauses.append("module = ?")
            params.append(module)

        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._locked_conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM bug_queue{where} ORDER BY created_at DESC",  # noqa: S608
                params,
            ).fetchall()
            columns = [desc[0] for desc in conn.description]
            return [self._row_to_dict(columns, r) for r in rows]

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def failure_hotspots(self) -> list[dict[str, Any]]:
        """Modules/files with the most open bugs."""
        with self._locked_conn() as conn:
            rows = conn.execute(
                """
                SELECT module, test_file, COUNT(*) as bug_count,
                       SUM(attempt_count) as total_attempts
                FROM bug_queue
                WHERE status IN ('open', 'escalated')
                GROUP BY module, test_file
                ORDER BY bug_count DESC
                """
            ).fetchall()
            columns = [desc[0] for desc in conn.description]
            return [dict(zip(columns, r)) for r in rows]

    def flaky_tests(self) -> list[dict[str, Any]]:
        """Tests with high attempt counts (likely flaky)."""
        with self._locked_conn() as conn:
            rows = conn.execute(
                """
                SELECT test_name, test_file, module, attempt_count, status
                FROM bug_queue
                WHERE attempt_count >= 2
                ORDER BY attempt_count DESC
                """
            ).fetchall()
            columns = [desc[0] for desc in conn.description]
            return [dict(zip(columns, r)) for r in rows]

    def agent_effectiveness(self) -> list[dict[str, Any]]:
        """Resolution stats per agent."""
        with self._locked_conn() as conn:
            rows = conn.execute(
                """
                SELECT assigned_to,
                       COUNT(*) FILTER (WHERE status = 'resolved') as resolved,
                       COUNT(*) FILTER (WHERE status = 'in_progress') as in_progress,
                       COUNT(*) FILTER (WHERE status = 'escalated') as escalated
                FROM bug_queue
                WHERE assigned_to IS NOT NULL
                GROUP BY assigned_to
                ORDER BY resolved DESC
                """
            ).fetchall()
            columns = [desc[0] for desc in conn.description]
            return [dict(zip(columns, r)) for r in rows]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _record_transition(
        self,
        conn: duckdb.DuckDBPyConnection,
        bug_id: str,
        from_status: str | None,
        to_status: str,
        agent_id: str | None,
        note: str | None,
    ) -> None:
        conn.execute(
            """
            INSERT INTO bug_transitions (id, bug_id, from_status, to_status, agent_id, note, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [str(uuid.uuid4()), bug_id, from_status, to_status, agent_id, note,
             datetime.now(timezone.utc)],
        )

    @staticmethod
    def _row_to_dict(columns: list[str], row: tuple) -> dict[str, Any]:
        d = dict(zip(columns, row))
        for json_field in ("expected", "actual", "environment", "tags"):
            if d.get(json_field) is not None:
                d[json_field] = json.loads(d[json_field])
        return d

    def close(self) -> None:
        """Close the DuckDB connection."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
