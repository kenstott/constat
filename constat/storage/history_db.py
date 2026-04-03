# Copyright (c) 2025 Kenneth Stott
# Canary: ce0e2b18-ef3c-45ba-98c1-0ab31ee2d1e6
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""DuckDB-backed session history implementation."""

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from constat.storage.duckdb_pool import ThreadLocalDuckDB
from constat.storage.history import (
    Artifact,
    QueryRecord,
    SessionDetail,
    SessionSummary,
)


class DuckDBSessionHistory:
    """DuckDB-backed session history with the same public API as FileSessionHistory."""

    _SESSIONS_DDL = """
    CREATE TABLE IF NOT EXISTS sessions (
        session_id VARCHAR PRIMARY KEY,
        created_at TIMESTAMP NOT NULL,
        completed_at TIMESTAMP,
        config_hash VARCHAR,
        status VARCHAR DEFAULT 'running',
        user_id VARCHAR,
        server_session_id VARCHAR,
        databases TEXT,
        apis TEXT,
        documents TEXT,
        files TEXT,
        summary TEXT,
        total_queries INTEGER DEFAULT 0,
        total_duration_ms INTEGER DEFAULT 0
    )
    """

    _QUERIES_DDL = """
    CREATE TABLE IF NOT EXISTS queries (
        id VARCHAR PRIMARY KEY,
        session_id VARCHAR NOT NULL,
        query_id INTEGER,
        timestamp TIMESTAMP NOT NULL,
        question TEXT,
        success BOOLEAN,
        attempts INTEGER DEFAULT 0,
        duration_ms INTEGER DEFAULT 0,
        answer TEXT,
        error TEXT,
        type VARCHAR DEFAULT 'query',
        attempt_history TEXT
    )
    """

    _PLAN_ITERATIONS_DDL = """
    CREATE TABLE IF NOT EXISTS plan_iterations (
        session_id VARCHAR NOT NULL,
        iteration INTEGER NOT NULL,
        raw_response TEXT,
        parsed_plan TEXT,
        reasoning TEXT,
        approval TEXT,
        PRIMARY KEY (session_id, iteration)
    )
    """

    _STEPS_DDL = """
    CREATE TABLE IF NOT EXISTS steps (
        session_id VARCHAR NOT NULL,
        step_number INTEGER NOT NULL,
        goal TEXT,
        code TEXT,
        output TEXT,
        error TEXT,
        prompt TEXT,
        model VARCHAR,
        timestamp TIMESTAMP,
        PRIMARY KEY (session_id, step_number)
    )
    """

    _INFERENCES_DDL = """
    CREATE TABLE IF NOT EXISTS inferences (
        session_id VARCHAR NOT NULL,
        inference_id VARCHAR NOT NULL,
        name TEXT,
        operation TEXT,
        code TEXT,
        attempt INTEGER DEFAULT 1,
        output TEXT,
        error TEXT,
        prompt TEXT,
        model VARCHAR,
        timestamp TIMESTAMP,
        PRIMARY KEY (session_id, inference_id, attempt)
    )
    """

    _INFERENCE_PREMISES_DDL = """
    CREATE TABLE IF NOT EXISTS inference_premises (
        session_id VARCHAR NOT NULL,
        premise_id VARCHAR NOT NULL,
        name TEXT,
        value TEXT,
        source TEXT,
        description TEXT,
        PRIMARY KEY (session_id, premise_id)
    )
    """

    _SESSION_STATE_DDL = (
        "CREATE TABLE IF NOT EXISTS session_state "
        "(session_id VARCHAR PRIMARY KEY, state TEXT)"
    )

    _SESSION_MESSAGES_DDL = (
        "CREATE TABLE IF NOT EXISTS session_messages "
        "(session_id VARCHAR PRIMARY KEY, messages TEXT)"
    )

    _PROOF_FACTS_DDL = (
        "CREATE TABLE IF NOT EXISTS proof_facts "
        "(server_session_id VARCHAR PRIMARY KEY, facts TEXT, summary TEXT)"
    )

    def __init__(self, db: ThreadLocalDuckDB, user_id: str = "default"):
        self._db = db
        self.user_id = user_id
        self._tables_ensured = False

    def _ensure_tables(self) -> None:
        """Create tables if they don't exist (idempotent)."""
        if self._tables_ensured:
            return
        self._db.execute(self._SESSIONS_DDL)
        self._db.execute(self._QUERIES_DDL)
        self._db.execute(self._PLAN_ITERATIONS_DDL)
        self._db.execute(self._STEPS_DDL)
        self._db.execute(self._INFERENCES_DDL)
        self._db.execute(self._INFERENCE_PREMISES_DDL)
        self._db.execute(self._SESSION_STATE_DDL)
        self._db.execute(self._SESSION_MESSAGES_DDL)
        self._db.execute(self._PROOF_FACTS_DDL)
        self._tables_ensured = True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_session_id() -> str:
        """Generate a unique session ID with full timestamp."""
        now = datetime.now(timezone.utc)
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        microseconds = f"{now.microsecond:06d}"
        return f"{timestamp}_{microseconds}"

    @staticmethod
    def _hash_config(config_dict: dict) -> str:
        config_str = json.dumps(config_dict, sort_keys=True)
        return f"sha256:{hashlib.sha256(config_str.encode()).hexdigest()[:16]}"

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def create_session(
        self,
        config_dict: dict,
        databases: list[str],
        apis: Optional[list[str]] = None,
        documents: Optional[list[str]] = None,
        server_session_id: Optional[str] = None,
    ) -> str:
        self._ensure_tables()
        session_id = self._generate_session_id()
        now = datetime.now(timezone.utc)

        self._db.execute(
            """
            INSERT INTO sessions (
                session_id, created_at, config_hash, status, user_id,
                server_session_id, databases, apis, documents, files,
                summary, total_queries, total_duration_ms
            ) VALUES (?, ?, ?, 'running', ?, ?, ?, ?, ?, '[]', NULL, 0, 0)
            """,
            [
                session_id,
                now,
                self._hash_config(config_dict),
                self.user_id,
                server_session_id,
                json.dumps(databases),
                json.dumps(apis or []),
                json.dumps(documents or []),
            ],
        )
        return session_id

    def record_query(
        self,
        session_id: str,
        question: str,
        success: bool,
        attempts: int,
        duration_ms: int,
        answer: Optional[str] = None,
        error: Optional[str] = None,
        attempt_history: Optional[list[dict]] = None,
    ) -> int:
        self._ensure_tables()

        # Read current totals
        row = self._db.execute(
            "SELECT total_queries, total_duration_ms, summary "
            "FROM sessions WHERE session_id = ?",
            [session_id],
        ).fetchone()
        if row is None:
            return 0

        total_queries, total_duration_ms, summary = row
        query_id = total_queries + 1

        # Auto-set summary from first query
        if summary is None and question:
            s = question[:100].strip()
            if len(question) > 100:
                s += "..."
            summary = s

        # Update session totals
        self._db.execute(
            "UPDATE sessions SET total_queries = ?, total_duration_ms = ?, "
            "summary = ? WHERE session_id = ?",
            [query_id, total_duration_ms + duration_ms, summary, session_id],
        )

        # Insert query record
        self._db.execute(
            """
            INSERT INTO queries (
                id, session_id, query_id, timestamp, question, success,
                attempts, duration_ms, answer, error, type, attempt_history
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'query', ?)
            """,
            [
                str(uuid.uuid4()),
                session_id,
                query_id,
                datetime.now(timezone.utc),
                question,
                success,
                attempts,
                duration_ms,
                answer if answer else None,
                error if error else None,
                json.dumps(attempt_history) if attempt_history else None,
            ],
        )
        return query_id

    def log_user_input(
        self,
        session_id: str,
        text: str,
        input_type: str = "query",
    ) -> None:
        self._ensure_tables()
        self._db.execute(
            """
            INSERT INTO queries (
                id, session_id, query_id, timestamp, question, success,
                attempts, duration_ms, type
            ) VALUES (?, ?, NULL, ?, ?, NULL, 0, 0, ?)
            """,
            [
                str(uuid.uuid4()),
                session_id,
                datetime.now(timezone.utc),
                text,
                input_type,
            ],
        )

    def complete_session(self, session_id: str, status: str = "completed") -> None:
        self._ensure_tables()
        self._db.execute(
            "UPDATE sessions SET status = ?, completed_at = ? WHERE session_id = ?",
            [status, datetime.now(timezone.utc), session_id],
        )

    def update_resources(
        self,
        session_id: str,
        databases: list[str],
        apis: list[str],
        documents: list[str],
    ) -> None:
        self._ensure_tables()
        self._db.execute(
            "UPDATE sessions SET databases = ?, apis = ?, documents = ? "
            "WHERE session_id = ?",
            [json.dumps(databases), json.dumps(apis), json.dumps(documents), session_id],
        )

    def update_summary(self, session_id: str, summary: str) -> None:
        self._ensure_tables()
        self._db.execute(
            "UPDATE sessions SET summary = ? WHERE session_id = ?",
            [summary, session_id],
        )

    # ------------------------------------------------------------------
    # Plan data
    # ------------------------------------------------------------------

    def save_plan_data(
        self,
        session_id: str,
        *,
        raw_response: str | None = None,
        parsed_plan: dict | None = None,
        reasoning: str | None = None,
        approval_decision: str | None = None,
        user_feedback: str | None = None,
        edited_steps: list | None = None,
        iteration: int = 0,
    ) -> None:
        self._ensure_tables()

        # Build approval JSON if decision provided
        approval_json = None
        if approval_decision is not None:
            approval_json = json.dumps({
                "decision": approval_decision,
                "feedback": user_feedback,
                "edited_steps": edited_steps,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        # Check if row exists
        existing = self._db.execute(
            "SELECT 1 FROM plan_iterations WHERE session_id = ? AND iteration = ?",
            [session_id, iteration],
        ).fetchone()

        if existing:
            # Build SET clause dynamically for non-None fields
            sets = []
            params = []
            if raw_response is not None:
                sets.append("raw_response = ?")
                params.append(raw_response)
            if parsed_plan is not None:
                sets.append("parsed_plan = ?")
                params.append(json.dumps(parsed_plan))
            if reasoning is not None:
                sets.append("reasoning = ?")
                params.append(reasoning)
            if approval_json is not None:
                sets.append("approval = ?")
                params.append(approval_json)
            if sets:
                params.extend([session_id, iteration])
                self._db.execute(
                    f"UPDATE plan_iterations SET {', '.join(sets)} "
                    f"WHERE session_id = ? AND iteration = ?",
                    params,
                )
        else:
            self._db.execute(
                """
                INSERT INTO plan_iterations (
                    session_id, iteration, raw_response, parsed_plan,
                    reasoning, approval
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    session_id,
                    iteration,
                    raw_response,
                    json.dumps(parsed_plan) if parsed_plan is not None else None,
                    reasoning,
                    approval_json,
                ],
            )

    # ------------------------------------------------------------------
    # Step codes
    # ------------------------------------------------------------------

    def save_step_code(
        self,
        session_id: str,
        step_number: int,
        goal: str,
        code: str,
        output: Optional[str] = None,
        error: Optional[str] = None,
        prompt: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self._ensure_tables()

        existing = self._db.execute(
            "SELECT 1 FROM steps WHERE session_id = ? AND step_number = ?",
            [session_id, step_number],
        ).fetchone()

        if existing:
            self._db.execute(
                """
                UPDATE steps SET goal = ?, code = ?, output = ?, error = ?,
                    prompt = ?, model = ?, timestamp = ?
                WHERE session_id = ? AND step_number = ?
                """,
                [
                    goal, code, output, error, prompt, model,
                    datetime.now(timezone.utc), session_id, step_number,
                ],
            )
        else:
            self._db.execute(
                """
                INSERT INTO steps (
                    session_id, step_number, goal, code, output, error,
                    prompt, model, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    session_id, step_number, goal, code, output, error,
                    prompt, model, datetime.now(timezone.utc),
                ],
            )

    def list_step_codes(self, session_id: str) -> list[dict]:
        self._ensure_tables()
        rows = self._db.execute(
            "SELECT step_number, goal, code, prompt, model "
            "FROM steps WHERE session_id = ? ORDER BY step_number",
            [session_id],
        ).fetchall()

        result = []
        for step_number, goal, code, prompt, model in rows:
            entry = {
                "step_number": step_number,
                "goal": goal or "",
                "code": code or "",
                "prompt": prompt or "",
            }
            if model:
                entry["model"] = model
            result.append(entry)
        return result

    # ------------------------------------------------------------------
    # Inference codes
    # ------------------------------------------------------------------

    def clear_inferences(self, session_id: str) -> None:
        self._ensure_tables()
        self._db.execute(
            "DELETE FROM inferences WHERE session_id = ?", [session_id]
        )
        self._db.execute(
            "DELETE FROM inference_premises WHERE session_id = ?", [session_id]
        )

    def save_inference_code(
        self,
        session_id: str,
        inference_id: str,
        name: str,
        operation: str,
        code: str,
        attempt: int = 1,
        output: Optional[str] = None,
        error: Optional[str] = None,
        prompt: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self._ensure_tables()

        existing = self._db.execute(
            "SELECT 1 FROM inferences "
            "WHERE session_id = ? AND inference_id = ? AND attempt = ?",
            [session_id, inference_id, attempt],
        ).fetchone()

        if existing:
            self._db.execute(
                """
                UPDATE inferences SET name = ?, operation = ?, code = ?,
                    output = ?, error = ?, prompt = ?, model = ?, timestamp = ?
                WHERE session_id = ? AND inference_id = ? AND attempt = ?
                """,
                [
                    name, operation, code, output, error, prompt, model,
                    datetime.now(timezone.utc),
                    session_id, inference_id, attempt,
                ],
            )
        else:
            self._db.execute(
                """
                INSERT INTO inferences (
                    session_id, inference_id, name, operation, code, attempt,
                    output, error, prompt, model, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    session_id, inference_id, name, operation, code, attempt,
                    output, error, prompt, model, datetime.now(timezone.utc),
                ],
            )

    def list_inference_codes(self, session_id: str) -> list[dict]:
        self._ensure_tables()

        # Get latest attempt per inference_id
        rows = self._db.execute(
            """
            SELECT i.inference_id, i.name, i.operation, i.code,
                   i.attempt, i.prompt, i.model
            FROM inferences i
            INNER JOIN (
                SELECT session_id, inference_id, MAX(attempt) AS max_attempt
                FROM inferences
                WHERE session_id = ?
                GROUP BY session_id, inference_id
            ) latest ON i.session_id = latest.session_id
                AND i.inference_id = latest.inference_id
                AND i.attempt = latest.max_attempt
            WHERE i.session_id = ?
            ORDER BY i.inference_id
            """,
            [session_id, session_id],
        ).fetchall()

        result = []
        for inference_id, name, operation, code, attempt, prompt, model in rows:
            entry = {
                "inference_id": inference_id,
                "name": name or "",
                "operation": operation or "",
                "code": code or "",
                "attempt": attempt,
                "prompt": prompt or "",
            }
            if model:
                entry["model"] = model
            result.append(entry)
        return result

    # ------------------------------------------------------------------
    # Inference premises
    # ------------------------------------------------------------------

    def save_inference_premise(
        self,
        session_id: str,
        premise_id: str,
        name: str,
        value: Any,
        source: str,
        description: str = "",
    ) -> None:
        self._ensure_tables()

        existing = self._db.execute(
            "SELECT 1 FROM inference_premises "
            "WHERE session_id = ? AND premise_id = ?",
            [session_id, premise_id],
        ).fetchone()

        if existing:
            self._db.execute(
                """
                UPDATE inference_premises SET name = ?, value = ?,
                    source = ?, description = ?
                WHERE session_id = ? AND premise_id = ?
                """,
                [name, str(value)[:500], source, description, session_id, premise_id],
            )
        else:
            self._db.execute(
                """
                INSERT INTO inference_premises (
                    session_id, premise_id, name, value, source, description
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [session_id, premise_id, name, str(value)[:500], source, description],
            )

    def list_inference_premises(self, session_id: str) -> list[dict]:
        self._ensure_tables()
        rows = self._db.execute(
            "SELECT premise_id, name, value, source, description "
            "FROM inference_premises WHERE session_id = ? ORDER BY premise_id",
            [session_id],
        ).fetchall()

        return [
            {
                "premise_id": premise_id,
                "name": name or "",
                "value": value or "",
                "source": source or "",
                "description": description or "",
            }
            for premise_id, name, value, source, description in rows
        ]

    # ------------------------------------------------------------------
    # State / messages / proof facts
    # ------------------------------------------------------------------

    def save_state(self, session_id: str, state: dict) -> None:
        self._ensure_tables()
        existing = self._db.execute(
            "SELECT 1 FROM session_state WHERE session_id = ?", [session_id]
        ).fetchone()

        if existing:
            self._db.execute(
                "UPDATE session_state SET state = ? WHERE session_id = ?",
                [json.dumps(state), session_id],
            )
        else:
            self._db.execute(
                "INSERT INTO session_state (session_id, state) VALUES (?, ?)",
                [session_id, json.dumps(state)],
            )

    def load_state(self, session_id: str) -> Optional[dict]:
        self._ensure_tables()
        row = self._db.execute(
            "SELECT state FROM session_state WHERE session_id = ?", [session_id]
        ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def save_messages(self, session_id: str, messages: list[dict]) -> None:
        self._ensure_tables()
        existing = self._db.execute(
            "SELECT 1 FROM session_messages WHERE session_id = ?", [session_id]
        ).fetchone()

        if existing:
            self._db.execute(
                "UPDATE session_messages SET messages = ? WHERE session_id = ?",
                [json.dumps(messages), session_id],
            )
        else:
            self._db.execute(
                "INSERT INTO session_messages (session_id, messages) VALUES (?, ?)",
                [session_id, json.dumps(messages)],
            )

    def load_messages(self, session_id: str) -> list[dict]:
        self._ensure_tables()
        row = self._db.execute(
            "SELECT messages FROM session_messages WHERE session_id = ?", [session_id]
        ).fetchone()
        if row is None:
            return []
        return json.loads(row[0])

    def save_messages_by_server_id(
        self, server_session_id: str, messages: list[dict]
    ) -> None:
        history_session_id = self.find_session_by_server_id(server_session_id)
        if not history_session_id:
            raise ValueError(f"No session found for server_id={server_session_id}")
        self.save_messages(history_session_id, messages)

    def load_messages_by_server_id(self, server_session_id: str) -> list[dict]:
        history_session_id = self.find_session_by_server_id(server_session_id)
        if not history_session_id:
            return []
        return self.load_messages(history_session_id)

    def save_proof_facts_by_server_id(
        self, server_session_id: str, facts: list[dict], summary: str | None = None
    ) -> None:
        self._ensure_tables()
        existing = self._db.execute(
            "SELECT 1 FROM proof_facts WHERE server_session_id = ?",
            [server_session_id],
        ).fetchone()

        if existing:
            self._db.execute(
                "UPDATE proof_facts SET facts = ?, summary = ? "
                "WHERE server_session_id = ?",
                [json.dumps(facts), summary, server_session_id],
            )
        else:
            self._db.execute(
                "INSERT INTO proof_facts (server_session_id, facts, summary) "
                "VALUES (?, ?, ?)",
                [server_session_id, json.dumps(facts), summary],
            )

    def load_proof_facts_by_server_id(
        self, server_session_id: str
    ) -> tuple[list[dict], str | None]:
        self._ensure_tables()
        row = self._db.execute(
            "SELECT facts, summary FROM proof_facts WHERE server_session_id = ?",
            [server_session_id],
        ).fetchone()
        if row is None:
            return [], None
        return json.loads(row[0]), row[1]

    # ------------------------------------------------------------------
    # Listing / retrieval
    # ------------------------------------------------------------------

    def list_sessions(self, limit: int = 20) -> list[SessionSummary]:
        self._ensure_tables()
        rows = self._db.execute(
            """
            SELECT session_id, created_at, databases, status, total_queries,
                   total_duration_ms, user_id, summary, apis, documents, files,
                   server_session_id
            FROM sessions
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            [self.user_id, limit],
        ).fetchall()

        result = []
        for row in rows:
            (
                session_id, created_at, databases, status, total_queries,
                total_duration_ms, user_id, summary, apis, documents, files,
                server_session_id,
            ) = row
            result.append(SessionSummary(
                session_id=session_id,
                created_at=created_at.isoformat() if created_at else "",
                databases=json.loads(databases) if databases else [],
                status=status or "unknown",
                total_queries=total_queries or 0,
                total_duration_ms=total_duration_ms or 0,
                user_id=user_id,
                summary=summary,
                apis=json.loads(apis) if apis else [],
                documents=json.loads(documents) if documents else [],
                files=json.loads(files) if files else [],
                server_session_id=server_session_id,
            ))
        return result

    def get_session(self, session_id: str) -> Optional[SessionDetail]:
        self._ensure_tables()
        row = self._db.execute(
            """
            SELECT session_id, created_at, config_hash, databases, status,
                   total_queries, total_duration_ms, summary, user_id
            FROM sessions WHERE session_id = ?
            """,
            [session_id],
        ).fetchone()

        if row is None:
            return None

        (
            sid, created_at, config_hash, databases, status,
            total_queries, total_duration_ms, summary, user_id,
        ) = row

        # Fetch query records (type='query' with a query_id)
        query_rows = self._db.execute(
            """
            SELECT query_id, timestamp, question, success, attempts,
                   duration_ms, answer, error
            FROM queries
            WHERE session_id = ? AND type = 'query' AND query_id IS NOT NULL
            ORDER BY query_id
            """,
            [session_id],
        ).fetchall()

        queries = []
        for qrow in query_rows:
            qid, ts, question, success, att, dur, ans, err = qrow
            queries.append(QueryRecord(
                query_id=qid,
                timestamp=ts.isoformat() if ts else "",
                question=question or "",
                success=bool(success),
                attempts=att or 0,
                duration_ms=dur or 0,
                answer=ans,
                error=err,
            ))

        return SessionDetail(
            session_id=sid,
            created_at=created_at.isoformat() if created_at else "",
            config_hash=config_hash or "",
            databases=json.loads(databases) if databases else [],
            status=status or "unknown",
            total_queries=total_queries or 0,
            total_duration_ms=total_duration_ms or 0,
            queries=queries,
            summary=summary,
            user_id=user_id,
        )

    def get_artifacts(self, session_id: str, query_id: int) -> list[Artifact]:
        self._ensure_tables()

        # Get attempt_history from the query record
        row = self._db.execute(
            "SELECT attempt_history FROM queries "
            "WHERE session_id = ? AND query_id = ? AND type = 'query'",
            [session_id, query_id],
        ).fetchone()

        if row is None or row[0] is None:
            return []

        attempt_history = json.loads(row[0])
        artifacts = []
        for attempt in attempt_history:
            attempt_num = attempt.get("attempt", 1)
            prefix = f"{query_id:03d}_{attempt_num:02d}"

            if attempt.get("code"):
                artifacts.append(Artifact(
                    id=f"{prefix}_code",
                    query_id=query_id,
                    artifact_type="code",
                    content=attempt["code"],
                    attempt=attempt_num,
                ))
            if attempt.get("stdout"):
                artifacts.append(Artifact(
                    id=f"{prefix}_output",
                    query_id=query_id,
                    artifact_type="output",
                    content=attempt["stdout"],
                    attempt=attempt_num,
                ))
            if attempt.get("error"):
                artifacts.append(Artifact(
                    id=f"{prefix}_error",
                    query_id=query_id,
                    artifact_type="error",
                    content=attempt["error"],
                    attempt=attempt_num,
                ))

        return artifacts

    def find_session_by_server_id(self, server_session_id: str) -> Optional[str]:
        self._ensure_tables()
        row = self._db.execute(
            "SELECT session_id FROM sessions WHERE server_session_id = ?",
            [server_session_id],
        ).fetchone()
        if row is None:
            return None
        return row[0]

    def delete_session(self, session_id: str) -> bool:
        self._ensure_tables()

        # Check existence first
        row = self._db.execute(
            "SELECT 1 FROM sessions WHERE session_id = ?", [session_id]
        ).fetchone()
        if row is None:
            return False

        # Delete from all tables
        for table in (
            "queries", "plan_iterations", "steps", "inferences",
            "inference_premises", "session_state", "session_messages",
        ):
            self._db.execute(
                f"DELETE FROM {table} WHERE session_id = ?", [session_id]
            )

        # proof_facts uses server_session_id, look it up first
        srv_row = self._db.execute(
            "SELECT server_session_id FROM sessions WHERE session_id = ?",
            [session_id],
        ).fetchone()
        if srv_row and srv_row[0]:
            self._db.execute(
                "DELETE FROM proof_facts WHERE server_session_id = ?", [srv_row[0]]
            )

        self._db.execute(
            "DELETE FROM sessions WHERE session_id = ?", [session_id]
        )
        return True
