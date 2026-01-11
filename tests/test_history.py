"""Tests for SessionHistory."""

import json
import pytest
import tempfile
from pathlib import Path

from constat.storage.history import SessionHistory


@pytest.fixture
def temp_storage():
    """Create a temporary storage directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "sessions"


@pytest.fixture
def history(temp_storage):
    """Create a SessionHistory instance with temp storage."""
    return SessionHistory(storage_dir=temp_storage)


class TestSessionHistory:
    """Tests for SessionHistory."""

    def test_create_session(self, history):
        """Test creating a session."""
        session_id = history.create_session(
            config_dict={"test": "config"},
            databases=["db1", "db2"],
        )

        assert session_id is not None
        assert len(session_id) > 0

        # Session dir should exist
        session_dir = history._session_dir(session_id)
        assert session_dir.exists()
        assert (session_dir / "session.json").exists()
        assert (session_dir / "queries.jsonl").exists()
        assert (session_dir / "artifacts").is_dir()

    def test_record_query(self, history):
        """Test recording a query."""
        session_id = history.create_session(
            config_dict={},
            databases=["test_db"],
        )

        query_id = history.record_query(
            session_id=session_id,
            question="What is the total?",
            success=True,
            attempts=2,
            duration_ms=1500,
            answer="The total is $1000",
        )

        assert query_id == 1

        # Verify query was recorded
        session = history.get_session(session_id)
        assert session.total_queries == 1
        assert len(session.queries) == 1
        assert session.queries[0].question == "What is the total?"
        assert session.queries[0].success

    def test_record_multiple_queries(self, history):
        """Test recording multiple queries."""
        session_id = history.create_session(config_dict={}, databases=[])

        q1 = history.record_query(session_id, "Q1", True, 1, 100)
        q2 = history.record_query(session_id, "Q2", False, 3, 500, error="Failed")
        q3 = history.record_query(session_id, "Q3", True, 1, 200, answer="Answer")

        assert q1 == 1
        assert q2 == 2
        assert q3 == 3

        session = history.get_session(session_id)
        assert session.total_queries == 3
        assert session.total_duration_ms == 800

    def test_record_query_with_artifacts(self, history):
        """Test recording query with attempt history/artifacts."""
        session_id = history.create_session(config_dict={}, databases=[])

        attempt_history = [
            {"attempt": 1, "code": "print('hello')", "stdout": "hello", "error": None},
            {"attempt": 2, "code": "print('world')", "stdout": "world", "error": None},
        ]

        history.record_query(
            session_id=session_id,
            question="Test",
            success=True,
            attempts=2,
            duration_ms=100,
            attempt_history=attempt_history,
        )

        # Check artifacts were saved
        artifacts = history.get_artifacts(session_id, query_id=1)
        assert len(artifacts) >= 2

        # Should have code and output files
        types = {a.artifact_type for a in artifacts}
        assert "code" in types
        assert "output" in types

    def test_complete_session(self, history):
        """Test completing a session."""
        session_id = history.create_session(config_dict={}, databases=[])

        history.complete_session(session_id, status="completed")

        session = history.get_session(session_id)
        assert session.status == "completed"

    def test_list_sessions(self, history):
        """Test listing sessions."""
        # Create multiple sessions
        s1 = history.create_session(config_dict={}, databases=["db1"])
        s2 = history.create_session(config_dict={}, databases=["db2"])
        s3 = history.create_session(config_dict={}, databases=["db3"])

        sessions = history.list_sessions(limit=10)

        assert len(sessions) == 3
        # All sessions should be present
        session_ids = {s.session_id for s in sessions}
        assert s1 in session_ids
        assert s2 in session_ids
        assert s3 in session_ids

    def test_list_sessions_with_limit(self, history):
        """Test listing sessions with limit."""
        for i in range(5):
            history.create_session(config_dict={}, databases=[f"db{i}"])

        sessions = history.list_sessions(limit=2)
        assert len(sessions) == 2

    def test_get_session(self, history):
        """Test getting session details."""
        session_id = history.create_session(
            config_dict={"key": "value"},
            databases=["chinook", "northwind"],
        )

        session = history.get_session(session_id)

        assert session is not None
        assert session.session_id == session_id
        assert session.databases == ["chinook", "northwind"]
        assert session.status == "running"

    def test_get_nonexistent_session(self, history):
        """Test getting nonexistent session returns None."""
        session = history.get_session("nonexistent")
        assert session is None

    def test_save_and_load_state(self, history):
        """Test saving and loading session state."""
        session_id = history.create_session(config_dict={}, databases=[])

        state = {
            "plan": {"steps": [1, 2, 3]},
            "scratchpad": "Some notes",
            "variables": {"total": 1000},
        }

        history.save_state(session_id, state)
        loaded = history.load_state(session_id)

        assert loaded == state

    def test_load_nonexistent_state(self, history):
        """Test loading nonexistent state returns None."""
        session_id = history.create_session(config_dict={}, databases=[])
        loaded = history.load_state(session_id)
        assert loaded is None

    def test_delete_session(self, history):
        """Test deleting a session."""
        session_id = history.create_session(config_dict={}, databases=[])

        assert history.get_session(session_id) is not None

        deleted = history.delete_session(session_id)
        assert deleted

        assert history.get_session(session_id) is None

    def test_delete_nonexistent_session(self, history):
        """Test deleting nonexistent session returns False."""
        deleted = history.delete_session("nonexistent")
        assert not deleted

    def test_session_id_format(self, history):
        """Test session ID format includes timestamp."""
        session_id = history.create_session(config_dict={}, databases=[])

        # Format: YYYY-MM-DD_HHMMSS_xxxxxxxx
        parts = session_id.split("_")
        assert len(parts) == 3
        assert len(parts[0]) == 10  # Date
        assert len(parts[1]) == 6   # Time
        assert len(parts[2]) == 8   # UUID suffix
