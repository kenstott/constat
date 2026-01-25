# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Session history and artifact storage for review, debugging, and resumption."""

import hashlib
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


@dataclass
class Artifact:
    """A single artifact from a query execution."""
    id: str
    query_id: int
    artifact_type: str  # code, output, error, tool_call
    content: str
    attempt: int = 1
    metadata: dict = field(default_factory=dict)


@dataclass
class QueryRecord:
    """Record of a single query execution."""
    query_id: int
    timestamp: str
    question: str
    success: bool
    attempts: int
    duration_ms: int
    answer: Optional[str] = None
    error: Optional[str] = None


@dataclass
class SessionSummary:
    """Summary info for listing sessions."""
    session_id: str
    created_at: str
    databases: list[str]
    status: str
    total_queries: int
    total_duration_ms: int
    user_id: Optional[str] = None
    summary: Optional[str] = None  # Brief description (usually first query)
    apis: Optional[list[str]] = None  # API names used
    documents: Optional[list[str]] = None  # Document names used
    files: Optional[list[str]] = None  # Session-added file names

    def __post_init__(self):
        """Initialize optional lists."""
        if self.apis is None:
            self.apis = []
        if self.documents is None:
            self.documents = []
        if self.files is None:
            self.files = []


@dataclass
class SessionDetail:
    """Full session detail including queries."""
    session_id: str
    created_at: str
    config_hash: str
    databases: list[str]
    status: str
    total_queries: int
    total_duration_ms: int
    queries: list[QueryRecord]


class SessionHistory:
    """
    Persist complete session state for review, debugging, and resumption.

    Storage structure (user-scoped):
        .constat/
        ├── <user_id>/
        │   ├── sessions/
        │   │   ├── 2024-01-15_143022_abc123/
        │   │   │   ├── session.json       # Metadata, config, timestamps
        │   │   │   ├── queries.jsonl      # All queries in order
        │   │   │   ├── artifacts/
        │   │   │   │   ├── 001_code.py    # Generated code
        │   │   │   │   ├── 001_output.txt # Execution output
        │   │   │   │   └── ...
        │   │   │   └── state.json         # Final state (for resumption)
    """

    def __init__(
        self,
        storage_dir: Optional[Path] = None,
        user_id: Optional[str] = None,
    ):
        """
        Initialize session history storage.

        Args:
            storage_dir: Base directory for storage. Defaults to .constat/
            user_id: User ID for user-scoped storage. If provided, sessions are
                    stored under <storage_dir>/<user_id>/sessions/
        """
        if storage_dir is None:
            storage_dir = Path(".constat")

        self.base_dir = Path(storage_dir)
        self.user_id = user_id or "default"

        # User-scoped storage directory
        self.storage_dir = self.base_dir / self.user_id / "sessions"

    def _ensure_dir(self, path: Path) -> None:
        """Ensure directory exists."""
        path.mkdir(parents=True, exist_ok=True)

    def _generate_session_id(self) -> str:
        """Generate a unique session ID with full timestamp (sortable)."""
        now = datetime.now(timezone.utc)
        # Format: YYYY-MM-DD_HHMMSS_mmm (milliseconds for uniqueness and sortability)
        timestamp = now.strftime("%Y-%m-%d_%H%M%S")
        milliseconds = f"{now.microsecond // 1000:03d}"
        return f"{timestamp}_{milliseconds}"

    def _hash_config(self, config_dict: dict) -> str:
        """Generate hash of config for change detection."""
        config_str = json.dumps(config_dict, sort_keys=True)
        return f"sha256:{hashlib.sha256(config_str.encode()).hexdigest()[:16]}"

    def _session_dir(self, session_id: str) -> Path:
        """Get the directory for a session."""
        return self.storage_dir / session_id

    def _logs_dir(self, session_id: str) -> Path:
        """Get the logs directory for execution exhaust (code, stdout, errors)."""
        return self._session_dir(session_id) / "logs"

    def get_user_base_dir(self) -> Path:
        """Get the base directory for this user (for outputs, etc.)."""
        return self.base_dir / self.user_id

    def create_session(
        self,
        config_dict: dict,
        databases: list[str],
        apis: Optional[list[str]] = None,
        documents: Optional[list[str]] = None,
        server_session_id: Optional[str] = None,
    ) -> str:
        """
        Create a new session.

        Args:
            config_dict: Configuration dictionary for hash generation
            databases: List of database names in this session
            apis: List of API names in this session
            documents: List of document names in this session
            server_session_id: Optional server-side UUID for reverse lookup

        Returns:
            session_id: Unique identifier for this session
        """
        session_id = self._generate_session_id()
        session_dir = self._session_dir(session_id)
        self._ensure_dir(session_dir)
        self._ensure_dir(self._logs_dir(session_id))

        # Write session metadata
        metadata = {
            "session_id": session_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config_hash": self._hash_config(config_dict),
            "databases": databases,
            "apis": apis or [],
            "documents": documents or [],
            "files": [],  # Session-added files (updated during session)
            "status": "running",
            "total_queries": 0,
            "total_duration_ms": 0,
            "user_id": self.user_id,
            "summary": None,
            "server_session_id": server_session_id,  # For reverse lookup from API
        }

        with open(session_dir / "session.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Create empty queries file
        (session_dir / "queries.jsonl").touch()

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
        """
        Record a completed query with all artifacts.

        Args:
            session_id: Session to record to
            question: The user's question
            success: Whether query succeeded
            attempts: Number of attempts made
            duration_ms: Total duration in milliseconds
            answer: The answer if successful
            error: Error message if failed
            attempt_history: List of attempt details with code/errors

        Returns:
            query_id: Sequential ID of this query
        """
        session_dir = self._session_dir(session_id)
        session_file = session_dir / "session.json"
        queries_file = session_dir / "queries.jsonl"

        # Read current session metadata
        with open(session_file) as f:
            metadata = json.load(f)

        # Increment query count
        query_id = metadata["total_queries"] + 1
        metadata["total_queries"] = query_id
        metadata["total_duration_ms"] += duration_ms

        # Set summary from first query if not already set
        if metadata.get("summary") is None and question:
            # Truncate to first 100 chars for summary
            summary = question[:100].strip()
            if len(question) > 100:
                summary += "..."
            metadata["summary"] = summary

        # Create query record
        query_record = {
            "query_id": query_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question": question,
            "success": success,
            "attempts": attempts,
            "duration_ms": duration_ms,
        }
        if answer:
            query_record["answer"] = answer
        if error:
            query_record["error"] = error

        # Append to queries file
        with open(queries_file, "a") as f:
            f.write(json.dumps(query_record) + "\n")

        # Save artifacts from attempt history
        if attempt_history:
            logs_dir = self._logs_dir(session_id)
            for attempt in attempt_history:
                attempt_num = attempt.get("attempt", 1)
                prefix = f"{query_id:03d}_{attempt_num:02d}"

                # Save code
                if attempt.get("code"):
                    with open(logs_dir / f"{prefix}_code.py", "w") as f:
                        f.write(attempt["code"])

                # Save output
                if attempt.get("stdout"):
                    with open(logs_dir / f"{prefix}_output.txt", "w") as f:
                        f.write(attempt["stdout"])

                # Save error
                if attempt.get("error"):
                    with open(logs_dir / f"{prefix}_error.txt", "w") as f:
                        f.write(attempt["error"])

        # Update session metadata
        with open(session_file, "w") as f:
            json.dump(metadata, f, indent=2)

        return query_id

    def complete_session(self, session_id: str, status: str = "completed") -> None:
        """
        Mark a session as completed.

        Args:
            session_id: Session to complete
            status: Final status (completed, failed, interrupted)
        """
        session_file = self._session_dir(session_id) / "session.json"

        with open(session_file) as f:
            metadata = json.load(f)

        metadata["status"] = status
        metadata["completed_at"] = datetime.now(timezone.utc).isoformat()

        with open(session_file, "w") as f:
            json.dump(metadata, f, indent=2)

    def list_sessions(self, limit: int = 20) -> list[SessionSummary]:
        """
        List recent sessions for this user with summary info.

        Args:
            limit: Maximum number of sessions to return

        Returns:
            List of session summaries, newest first
        """
        if not self.storage_dir.exists():
            return []

        sessions = []

        # List sessions in user's directory, sorted by name (timestamp) descending
        session_dirs = sorted(
            [d for d in self.storage_dir.iterdir() if d.is_dir()],
            key=lambda x: x.name,
            reverse=True,
        )

        for session_dir in session_dirs:
            session_file = session_dir / "session.json"
            if not session_file.exists():
                continue

            try:
                with open(session_file) as f:
                    metadata = json.load(f)

                sessions.append(SessionSummary(
                    session_id=metadata["session_id"],
                    created_at=metadata["created_at"],
                    databases=metadata.get("databases", []),
                    status=metadata.get("status", "unknown"),
                    total_queries=metadata.get("total_queries", 0),
                    total_duration_ms=metadata.get("total_duration_ms", 0),
                    user_id=metadata.get("user_id"),
                    summary=metadata.get("summary"),
                    apis=metadata.get("apis", []),
                    documents=metadata.get("documents", []),
                    files=metadata.get("files", []),
                ))

                if len(sessions) >= limit:
                    break
            except (json.JSONDecodeError, KeyError):
                continue

        return sessions

    def get_session(self, session_id: str) -> Optional[SessionDetail]:
        """
        Get full session detail including all queries.

        Args:
            session_id: Session to retrieve

        Returns:
            SessionDetail or None if not found
        """
        session_dir = self._session_dir(session_id)
        session_file = session_dir / "session.json"
        queries_file = session_dir / "queries.jsonl"

        if not session_file.exists():
            return None

        with open(session_file) as f:
            metadata = json.load(f)

        queries = []
        if queries_file.exists():
            with open(queries_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        q = json.loads(line)
                        queries.append(QueryRecord(
                            query_id=q["query_id"],
                            timestamp=q["timestamp"],
                            question=q["question"],
                            success=q["success"],
                            attempts=q["attempts"],
                            duration_ms=q["duration_ms"],
                            answer=q.get("answer"),
                            error=q.get("error"),
                        ))

        return SessionDetail(
            session_id=metadata["session_id"],
            created_at=metadata["created_at"],
            config_hash=metadata.get("config_hash", ""),
            databases=metadata.get("databases", []),
            status=metadata.get("status", "unknown"),
            total_queries=metadata.get("total_queries", 0),
            total_duration_ms=metadata.get("total_duration_ms", 0),
            queries=queries,
        )

    def get_artifacts(self, session_id: str, query_id: int) -> list[Artifact]:
        """
        Get artifacts for a specific query.

        Args:
            session_id: Session ID
            query_id: Query ID within the session

        Returns:
            List of artifacts for this query
        """
        logs_dir = self._logs_dir(session_id)
        if not logs_dir.exists():
            return []

        prefix = f"{query_id:03d}_"
        artifacts = []

        for artifact_file in sorted(logs_dir.glob(f"{prefix}*")):
            # Parse filename: 001_01_code.py -> query_id=1, attempt=1, type=code
            parts = artifact_file.stem.split("_")
            if len(parts) >= 3:
                attempt = int(parts[1])
                artifact_type = parts[2]
            else:
                attempt = 1
                artifact_type = parts[1] if len(parts) >= 2 else "unknown"

            with open(artifact_file) as f:
                content = f.read()

            artifacts.append(Artifact(
                id=artifact_file.stem,
                query_id=query_id,
                artifact_type=artifact_type,
                content=content,
                attempt=attempt,
            ))

        return artifacts

    def _steps_dir(self, session_id: str) -> Path:
        """Get the steps directory for step code storage."""
        return self._session_dir(session_id) / "steps"

    def save_step_code(
        self,
        session_id: str,
        step_number: int,
        goal: str,
        code: str,
        output: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        """
        Save code for a specific execution step.

        Args:
            session_id: Session ID
            step_number: Step number in the plan
            goal: Description of what this step accomplishes
            code: Python code that was executed
            output: Standard output from execution (optional)
            error: Error message if execution failed (optional)
        """
        steps_dir = self._steps_dir(session_id)
        self._ensure_dir(steps_dir)

        # Save step metadata
        step_data = {
            "step_number": step_number,
            "goal": goal,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Save code file
        code_file = steps_dir / f"step_{step_number:03d}_code.py"
        with open(code_file, "w") as f:
            # Add goal as a comment at the top
            f.write(f'# Step {step_number}: {goal}\n\n')
            f.write(code)

        # Save output if present
        if output:
            output_file = steps_dir / f"step_{step_number:03d}_output.txt"
            with open(output_file, "w") as f:
                f.write(output)

        # Save error if present
        if error:
            error_file = steps_dir / f"step_{step_number:03d}_error.txt"
            with open(error_file, "w") as f:
                f.write(error)

        # Update step index
        index_file = steps_dir / "index.jsonl"
        with open(index_file, "a") as f:
            f.write(json.dumps(step_data) + "\n")

    def list_step_codes(self, session_id: str) -> list[dict]:
        """
        List all step codes for a session.

        Args:
            session_id: Session ID

        Returns:
            List of step info dicts with step_number, goal, code
        """
        steps_dir = self._steps_dir(session_id)
        if not steps_dir.exists():
            return []

        steps = []
        index_file = steps_dir / "index.jsonl"

        if index_file.exists():
            with open(index_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        step_data = json.loads(line)
                        step_number = step_data["step_number"]
                        code_file = steps_dir / f"step_{step_number:03d}_code.py"

                        if code_file.exists():
                            with open(code_file) as cf:
                                code = cf.read()
                            steps.append({
                                "step_number": step_number,
                                "goal": step_data.get("goal", ""),
                                "code": code,
                            })

        return sorted(steps, key=lambda x: x["step_number"])

    def find_session_by_server_id(self, server_session_id: str) -> Optional[str]:
        """
        Find a history session ID by its server UUID.

        This is used when the server restarts and loses the in-memory
        mapping between server UUIDs and history session IDs.

        Args:
            server_session_id: The server-side UUID

        Returns:
            History session ID (timestamp-based) or None if not found
        """
        if not self.storage_dir.exists():
            return None

        # Search through recent sessions (most recent first for efficiency)
        session_dirs = sorted(
            [d for d in self.storage_dir.iterdir() if d.is_dir()],
            key=lambda x: x.name,
            reverse=True,
        )

        for session_dir in session_dirs[:50]:  # Only check last 50 sessions
            session_file = session_dir / "session.json"
            if not session_file.exists():
                continue

            try:
                with open(session_file) as f:
                    metadata = json.load(f)

                if metadata.get("server_session_id") == server_session_id:
                    return metadata.get("session_id")
            except (json.JSONDecodeError, KeyError):
                continue

        return None

    def update_summary(self, session_id: str, summary: str) -> None:
        """
        Update the session summary (LLM-generated).

        Args:
            session_id: Session ID
            summary: New summary text
        """
        session_file = self._session_dir(session_id) / "session.json"
        if not session_file.exists():
            return

        with open(session_file) as f:
            metadata = json.load(f)

        metadata["summary"] = summary

        with open(session_file, "w") as f:
            json.dump(metadata, f, indent=2)

    def save_state(self, session_id: str, state: dict) -> None:
        """
        Save session state for resumption.

        Args:
            session_id: Session to save state for
            state: State dictionary to persist
        """
        state_file = self._session_dir(session_id) / "state.json"
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)

    def load_state(self, session_id: str) -> Optional[dict]:
        """
        Load session state for resumption.

        Args:
            session_id: Session to load state from

        Returns:
            State dictionary or None if not found
        """
        state_file = self._session_dir(session_id) / "state.json"
        if not state_file.exists():
            return None

        with open(state_file) as f:
            return json.load(f)

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and all its artifacts.

        Args:
            session_id: Session to delete

        Returns:
            True if deleted, False if not found
        """
        import shutil

        session_dir = self._session_dir(session_id)
        if not session_dir.exists():
            return False

        shutil.rmtree(session_dir)
        return True
