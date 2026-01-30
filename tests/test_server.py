# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tests for the Constat API server.

Tests cover:
- Session management (CRUD operations)
- Query submission and execution
- Data endpoints (tables, artifacts, facts)
- Schema discovery endpoints
- WebSocket connections
"""

import asyncio
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch, PropertyMock

from fastapi.testclient import TestClient

from constat.core.config import Config
from constat.server.app import create_app
from constat.server.config import ServerConfig
from constat.server.models import SessionStatus, EventType
from constat.server.session_manager import SessionManager, ManagedSession


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def clear_env_overrides(monkeypatch):
    """Clear environment variables that would override ServerConfig defaults.

    This ensures test fixtures control the config values, not .env files.
    """
    monkeypatch.delenv("AUTH_DISABLED", raising=False)
    monkeypatch.delenv("FIREBASE_PROJECT_ID", raising=False)


@pytest.fixture
def server_config():
    """Create a test server configuration."""
    return ServerConfig(
        host="127.0.0.1",
        port=8000,
        cors_origins=["http://localhost:3000"],
        session_timeout_minutes=60,
        max_concurrent_sessions=5,
        require_plan_approval=False,
        auth_disabled=True,  # Disable auth for tests
    )


@pytest.fixture
def minimal_config():
    """Create a minimal Constat configuration for testing."""
    return Config()


def create_mock_session():
    """Create a mock Session object."""
    mock = MagicMock()
    mock._event_handlers = []
    mock._cancelled = False
    mock.plan = None
    mock.datastore = None
    mock.fact_resolver = MagicMock()
    mock.fact_resolver.get_all_facts.return_value = {}
    mock.scratchpad = MagicMock()
    mock.scratchpad.get_recent_context.return_value = ""
    mock.schema_manager = MagicMock()
    mock.api_schema_manager = MagicMock()
    mock.config = Config()
    mock._execution_context = MagicMock()

    def on_event(handler):
        mock._event_handlers.append(handler)
    mock.on_event = on_event

    return mock


def create_mock_api():
    """Create a mock ConstatAPIImpl object."""
    mock = MagicMock()
    mock.set_approval_callback = MagicMock()
    mock.solve = MagicMock()
    mock.follow_up = MagicMock()
    mock.get_facts = MagicMock(return_value={})
    mock.get_learnings = MagicMock(return_value=[])
    return mock


@pytest.fixture
def mock_session_class():
    """Fixture that patches Session class for all tests that need it."""
    with patch('constat.server.session_manager.Session') as mock_cls:
        mock_cls.return_value = create_mock_session()
        yield mock_cls


@pytest.fixture
def app_with_mock(minimal_config, server_config, mock_session_class):
    """Create a FastAPI test application with mocked Session."""
    return create_app(minimal_config, server_config)


@pytest.fixture
def client_with_mock(app_with_mock):
    """Create a test client with mocked Session."""
    return TestClient(app_with_mock)


@pytest.fixture
def session_manager_with_mock(minimal_config, server_config, mock_session_class):
    """Create a session manager with mocked Session for unit tests."""
    return SessionManager(minimal_config, server_config)


# ============================================================================
# ServerConfig Tests
# ============================================================================


class TestServerConfig:
    """Tests for ServerConfig model."""

    def test_default_values(self, monkeypatch):
        """Test default configuration values (isolated from env vars)."""
        # Clear env vars that would override defaults
        monkeypatch.delenv("AUTH_DISABLED", raising=False)
        monkeypatch.delenv("FIREBASE_PROJECT_ID", raising=False)

        config = ServerConfig()
        assert config.host == "127.0.0.1"
        assert config.port == 8000
        assert "http://localhost:5173" in config.cors_origins
        assert config.session_timeout_minutes == 60
        assert config.max_concurrent_sessions == 10
        assert config.require_plan_approval is True  # Actual default is True
        assert config.auth_disabled is True  # Actual default is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ServerConfig(
            host="0.0.0.0",
            port=9000,
            cors_origins=["http://example.com"],
            session_timeout_minutes=30,
            max_concurrent_sessions=20,
            require_plan_approval=True,
        )
        assert config.host == "0.0.0.0"
        assert config.port == 9000
        assert config.cors_origins == ["http://example.com"]
        assert config.session_timeout_minutes == 30
        assert config.max_concurrent_sessions == 20
        assert config.require_plan_approval is True


# ============================================================================
# SessionManager Tests
# ============================================================================


class TestSessionManager:
    """Tests for SessionManager."""

    def test_create_session(self, session_manager_with_mock):
        """Test creating a new session."""
        session_id = session_manager_with_mock.create_session(user_id="test_user")

        assert session_id is not None
        assert len(session_id) == 36  # UUID format

        managed = session_manager_with_mock.get_session(session_id)
        assert managed.user_id == "test_user"
        assert managed.status == SessionStatus.IDLE
        assert managed.session is not None

    def test_create_session_max_limit(self, minimal_config, mock_session_class):
        """Test that session creation fails when limit is reached."""
        server_config = ServerConfig(max_concurrent_sessions=2)
        manager = SessionManager(minimal_config, server_config)

        # Create two sessions
        manager.create_session(user_id="user1")
        manager.create_session(user_id="user2")

        # Third session should fail
        with pytest.raises(RuntimeError, match="Maximum concurrent sessions"):
            manager.create_session(user_id="user3")

    def test_get_session_not_found(self, session_manager_with_mock):
        """Test getting a non-existent session."""
        with pytest.raises(KeyError, match="Session not found"):
            session_manager_with_mock.get_session("nonexistent-id")

    def test_get_session_or_none(self, session_manager_with_mock):
        """Test get_session_or_none returns None for missing session."""
        result = session_manager_with_mock.get_session_or_none("nonexistent-id")
        assert result is None

    def test_list_sessions(self, session_manager_with_mock):
        """Test listing all sessions."""
        session_manager_with_mock.create_session(user_id="user1")
        session_manager_with_mock.create_session(user_id="user2")
        session_manager_with_mock.create_session(user_id="user1")

        # List all
        all_sessions = session_manager_with_mock.list_sessions()
        assert len(all_sessions) == 3

        # Filter by user
        user1_sessions = session_manager_with_mock.list_sessions(user_id="user1")
        assert len(user1_sessions) == 2
        assert all(s.user_id == "user1" for s in user1_sessions)

    def test_delete_session(self, session_manager_with_mock):
        """Test deleting a session."""
        session_id = session_manager_with_mock.create_session(user_id="test")

        assert session_manager_with_mock.delete_session(session_id) is True
        assert session_manager_with_mock.get_session_or_none(session_id) is None

    def test_delete_session_not_found(self, session_manager_with_mock):
        """Test deleting a non-existent session."""
        result = session_manager_with_mock.delete_session("nonexistent-id")
        assert result is False

    def test_session_expiry(self, minimal_config, mock_session_class):
        """Test session expiry detection."""
        server_config = ServerConfig(session_timeout_minutes=0)  # Immediate expiry
        manager = SessionManager(minimal_config, server_config)

        session_id = manager.create_session()
        managed = manager.get_session(session_id)

        # Should be expired immediately
        assert managed.is_expired(0) is True

    def test_cleanup_expired(self, minimal_config, mock_session_class):
        """Test cleanup of expired sessions."""
        server_config = ServerConfig(session_timeout_minutes=0)
        manager = SessionManager(minimal_config, server_config)

        manager.create_session()
        manager.create_session()

        # Should cleanup both
        count = manager.cleanup_expired()
        assert count == 2
        assert len(manager.list_sessions()) == 0

    def test_update_status(self, session_manager_with_mock):
        """Test updating session status."""
        session_id = session_manager_with_mock.create_session()
        session_manager_with_mock.update_status(session_id, SessionStatus.EXECUTING)

        managed = session_manager_with_mock.get_session(session_id)
        assert managed.status == SessionStatus.EXECUTING

    def test_get_stats(self, session_manager_with_mock):
        """Test getting session statistics."""
        session_manager_with_mock.create_session()
        session_id2 = session_manager_with_mock.create_session()
        session_manager_with_mock.update_status(session_id2, SessionStatus.EXECUTING)

        stats = session_manager_with_mock.get_stats()
        assert stats["total_sessions"] == 2
        assert stats["max_sessions"] == 5
        assert "idle" in stats["by_status"]
        assert "executing" in stats["by_status"]


# ============================================================================
# ManagedSession Tests
# ============================================================================


class TestManagedSession:
    """Tests for ManagedSession dataclass."""

    def test_touch_updates_last_activity(self):
        """Test that touch() updates last_activity."""
        mock_session = create_mock_session()
        mock_api = create_mock_api()
        now = datetime.now(timezone.utc)
        managed = ManagedSession(
            session_id="test-id",
            session=mock_session,
            api=mock_api,
            user_id="test",
            created_at=now - timedelta(hours=1),
            last_activity=now - timedelta(hours=1),
        )

        old_activity = managed.last_activity
        managed.touch()
        assert managed.last_activity > old_activity

    def test_is_expired_false(self):
        """Test is_expired returns False when not expired."""
        mock_session = create_mock_session()
        mock_api = create_mock_api()
        now = datetime.now(timezone.utc)
        managed = ManagedSession(
            session_id="test-id",
            session=mock_session,
            api=mock_api,
            user_id="test",
            created_at=now,
            last_activity=now,
        )

        assert managed.is_expired(60) is False

    def test_is_expired_true(self):
        """Test is_expired returns True when expired."""
        mock_session = create_mock_session()
        mock_api = create_mock_api()
        now = datetime.now(timezone.utc)
        managed = ManagedSession(
            session_id="test-id",
            session=mock_session,
            api=mock_api,
            user_id="test",
            created_at=now - timedelta(hours=2),
            last_activity=now - timedelta(hours=2),
        )

        assert managed.is_expired(60) is True


# ============================================================================
# Health Endpoint Tests
# ============================================================================


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check(self, client_with_mock):
        """Test health check returns OK status."""
        response = client_with_mock.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "sessions" in data


# ============================================================================
# Session Endpoints Tests
# ============================================================================


class TestSessionEndpoints:
    """Tests for session management endpoints."""

    def test_create_session(self, client_with_mock):
        """Test creating a session via API."""
        response = client_with_mock.post(
            "/api/sessions",
            json={"user_id": "test_user"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert data["user_id"] == "test_user"
        assert data["status"] == "idle"

    def test_create_session_default_user(self, client_with_mock):
        """Test creating a session with default user ID."""
        response = client_with_mock.post("/api/sessions", json={})

        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "default"

    def test_list_sessions(self, client_with_mock):
        """Test listing sessions via API."""
        # Create sessions with default user (when auth disabled, no user_id = "default")
        client_with_mock.post("/api/sessions", json={})
        client_with_mock.post("/api/sessions", json={})

        # List sessions for default user (no filter)
        response = client_with_mock.get("/api/sessions")

        assert response.status_code == 200
        data = response.json()
        assert len(data["sessions"]) >= 2
        assert data["total"] >= 2

    def test_list_sessions_filter_by_user(self, client_with_mock):
        """Test filtering sessions by user ID."""
        client_with_mock.post("/api/sessions", json={"user_id": "filter_test"})

        response = client_with_mock.get("/api/sessions?user_id=filter_test")

        assert response.status_code == 200
        data = response.json()
        assert all(s["user_id"] == "filter_test" for s in data["sessions"])

    def test_get_session(self, client_with_mock):
        """Test getting a single session."""
        # Create a session
        create_response = client_with_mock.post("/api/sessions", json={})
        session_id = create_response.json()["session_id"]

        response = client_with_mock.get(f"/api/sessions/{session_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == session_id

    def test_get_session_not_found(self, client_with_mock):
        """Test getting a non-existent session."""
        response = client_with_mock.get("/api/sessions/nonexistent-id")

        assert response.status_code == 404
        data = response.json()
        assert data["error"] == "not_found"

    def test_delete_session(self, client_with_mock):
        """Test deleting a session."""
        # Create a session
        create_response = client_with_mock.post("/api/sessions", json={})
        session_id = create_response.json()["session_id"]

        response = client_with_mock.delete(f"/api/sessions/{session_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "deleted"

        # Verify it's gone
        get_response = client_with_mock.get(f"/api/sessions/{session_id}")
        assert get_response.status_code == 404

    def test_delete_session_not_found(self, client_with_mock):
        """Test deleting a non-existent session."""
        response = client_with_mock.delete("/api/sessions/nonexistent-id")

        assert response.status_code == 404


# ============================================================================
# Query Endpoints Tests
# ============================================================================


class TestQueryEndpoints:
    """Tests for query execution endpoints."""

    def test_submit_query(self, client_with_mock):
        """Test submitting a query."""
        # Create a session first
        create_response = client_with_mock.post("/api/sessions", json={})
        session_id = create_response.json()["session_id"]

        response = client_with_mock.post(
            f"/api/sessions/{session_id}/query",
            json={"problem": "What is 2 + 2?"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "execution_id" in data
        assert data["status"] == "started"

    def test_submit_query_session_not_found(self, client_with_mock):
        """Test submitting query to non-existent session."""
        response = client_with_mock.post(
            "/api/sessions/nonexistent-id/query",
            json={"problem": "test"}
        )

        assert response.status_code == 404

    def test_cancel_execution(self, client_with_mock):
        """Test cancelling execution."""
        # Create a session
        create_response = client_with_mock.post("/api/sessions", json={})
        session_id = create_response.json()["session_id"]

        response = client_with_mock.post(f"/api/sessions/{session_id}/cancel")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "cancelling"

    def test_get_plan_no_plan(self, client_with_mock):
        """Test getting plan when no plan exists."""
        # Create a session
        create_response = client_with_mock.post("/api/sessions", json={})
        session_id = create_response.json()["session_id"]

        response = client_with_mock.get(f"/api/sessions/{session_id}/plan")

        assert response.status_code == 404
        data = response.json()
        assert "No plan exists" in data["detail"]


# ============================================================================
# Data Endpoints Tests
# ============================================================================


class TestDataEndpoints:
    """Tests for data access endpoints."""

    def test_list_tables_empty(self, client_with_mock):
        """Test listing tables when none exist."""
        # Create a session
        create_response = client_with_mock.post("/api/sessions", json={})
        session_id = create_response.json()["session_id"]

        response = client_with_mock.get(f"/api/sessions/{session_id}/tables")

        assert response.status_code == 200
        data = response.json()
        assert data["tables"] == []

    def test_list_artifacts_empty(self, client_with_mock):
        """Test listing artifacts when none exist."""
        # Create a session
        create_response = client_with_mock.post("/api/sessions", json={})
        session_id = create_response.json()["session_id"]

        response = client_with_mock.get(f"/api/sessions/{session_id}/artifacts")

        assert response.status_code == 200
        data = response.json()
        assert data["artifacts"] == []

    def test_list_facts(self, client_with_mock):
        """Test listing facts."""
        # Create a session
        create_response = client_with_mock.post("/api/sessions", json={})
        session_id = create_response.json()["session_id"]

        response = client_with_mock.get(f"/api/sessions/{session_id}/facts")

        assert response.status_code == 200
        data = response.json()
        assert "facts" in data

    def test_get_proof_tree(self, client_with_mock):
        """Test getting proof tree."""
        # Create a session
        create_response = client_with_mock.post("/api/sessions", json={})
        session_id = create_response.json()["session_id"]

        response = client_with_mock.get(f"/api/sessions/{session_id}/proof-tree")

        assert response.status_code == 200
        data = response.json()
        assert "facts" in data

    def test_get_output(self, client_with_mock):
        """Test getting session output."""
        # Create a session
        create_response = client_with_mock.post("/api/sessions", json={})
        session_id = create_response.json()["session_id"]

        response = client_with_mock.get(f"/api/sessions/{session_id}/output")

        assert response.status_code == 200
        data = response.json()
        assert "output" in data


# ============================================================================
# Schema Endpoints Tests
# ============================================================================


class TestSchemaEndpoints:
    """Tests for schema discovery endpoints."""

    def test_get_schema_overview(self, client_with_mock):
        """Test getting schema overview."""
        response = client_with_mock.get("/api/schema")

        assert response.status_code == 200
        data = response.json()
        assert "databases" in data
        assert "apis" in data
        assert "documents" in data


# ============================================================================
# WebSocket Tests
# ============================================================================


class TestWebSocket:
    """Tests for WebSocket functionality."""

    def test_websocket_connect(self, client_with_mock):
        """Test WebSocket connection."""
        # Create a session
        create_response = client_with_mock.post("/api/sessions", json={})
        session_id = create_response.json()["session_id"]

        with client_with_mock.websocket_connect(f"/api/sessions/{session_id}/ws") as websocket:
            # Connection successful if we get here
            pass

    def test_websocket_send_cancel_command(self, client_with_mock):
        """Test sending cancel command via WebSocket."""
        # Create a session
        create_response = client_with_mock.post("/api/sessions", json={})
        session_id = create_response.json()["session_id"]

        with client_with_mock.websocket_connect(f"/api/sessions/{session_id}/ws") as websocket:
            # First, consume the welcome event sent on connection
            welcome = websocket.receive_json()
            assert welcome["type"] == "event"
            assert welcome["payload"]["event_type"] == "welcome"

            # Send cancel command
            websocket.send_json({"action": "cancel"})

            # Should receive acknowledgment
            response = websocket.receive_json()
            assert response["type"] == "ack"
            assert response["payload"]["action"] == "cancel"

    def test_websocket_send_unknown_command(self, client_with_mock):
        """Test sending unknown command via WebSocket."""
        # Create a session
        create_response = client_with_mock.post("/api/sessions", json={})
        session_id = create_response.json()["session_id"]

        with client_with_mock.websocket_connect(f"/api/sessions/{session_id}/ws") as websocket:
            # First, consume the welcome event sent on connection
            welcome = websocket.receive_json()
            assert welcome["type"] == "event"
            assert welcome["payload"]["event_type"] == "welcome"

            # Send unknown command
            websocket.send_json({"action": "unknown_action"})

            # Should receive error
            response = websocket.receive_json()
            assert response["type"] == "error"
            assert "Unknown action" in response["payload"]["message"]


# ============================================================================
# Model Tests
# ============================================================================


class TestModels:
    """Tests for Pydantic models."""

    def test_session_status_enum(self):
        """Test SessionStatus enum values."""
        assert SessionStatus.IDLE.value == "idle"
        assert SessionStatus.PLANNING.value == "planning"
        assert SessionStatus.EXECUTING.value == "executing"
        assert SessionStatus.COMPLETED.value == "completed"

    def test_event_type_enum(self):
        """Test EventType enum values."""
        assert EventType.STEP_START.value == "step_start"
        assert EventType.QUERY_COMPLETE.value == "query_complete"
        assert EventType.PLANNING_START.value == "planning_start"


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_invalid_json(self, client_with_mock):
        """Test handling of invalid JSON."""
        response = client_with_mock.post(
            "/api/sessions",
            content="invalid json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422

    def test_missing_required_field(self, client_with_mock):
        """Test handling of missing required fields."""
        # Create a session first
        create_response = client_with_mock.post("/api/sessions", json={})
        session_id = create_response.json()["session_id"]

        # Query without problem field
        response = client_with_mock.post(
            f"/api/sessions/{session_id}/query",
            json={}
        )

        assert response.status_code == 422


# ============================================================================
# CORS Tests
# ============================================================================


class TestCORS:
    """Tests for CORS configuration."""

    def test_cors_headers(self, client_with_mock):
        """Test CORS headers are present."""
        response = client_with_mock.options(
            "/api/sessions",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            }
        )

        # CORS preflight should succeed
        assert response.status_code == 200


# ============================================================================
# Integration Tests
# ============================================================================


class TestFileEndpoints:
    """Tests for file upload endpoints."""

    def test_list_files_empty(self, client_with_mock):
        """Test listing files when none uploaded."""
        # Create a session
        create_response = client_with_mock.post("/api/sessions", json={})
        session_id = create_response.json()["session_id"]

        response = client_with_mock.get(f"/api/sessions/{session_id}/files")

        assert response.status_code == 200
        data = response.json()
        assert data["files"] == []

    def test_list_file_refs_empty(self, client_with_mock):
        """Test listing file references when none exist."""
        # Create a session
        create_response = client_with_mock.post("/api/sessions", json={})
        session_id = create_response.json()["session_id"]

        response = client_with_mock.get(f"/api/sessions/{session_id}/file-refs")

        assert response.status_code == 200
        data = response.json()
        assert data["file_refs"] == []

    def test_add_file_reference(self, client_with_mock):
        """Test adding a file reference."""
        # Create a session
        create_response = client_with_mock.post("/api/sessions", json={})
        session_id = create_response.json()["session_id"]

        response = client_with_mock.post(
            f"/api/sessions/{session_id}/file-refs",
            json={
                "name": "test_file",
                "uri": "https://example.com/test.csv",
                "description": "Test file",
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "test_file"
        assert data["uri"] == "https://example.com/test.csv"
        assert data["has_auth"] is False


class TestDatabaseEndpoints:
    """Tests for database connection endpoints."""

    def test_list_databases(self, client_with_mock):
        """Test listing databases."""
        # Create a session
        create_response = client_with_mock.post("/api/sessions", json={})
        session_id = create_response.json()["session_id"]

        response = client_with_mock.get(f"/api/sessions/{session_id}/databases")

        assert response.status_code == 200
        data = response.json()
        assert "databases" in data


class TestLearningsEndpoints:
    """Tests for learnings endpoints."""

    def test_list_learnings(self, client_with_mock):
        """Test listing learnings."""
        response = client_with_mock.get("/api/learnings")

        assert response.status_code == 200
        data = response.json()
        assert "learnings" in data

    def test_add_learning(self, client_with_mock):
        """Test adding a learning."""
        response = client_with_mock.post(
            "/api/learnings",
            json={
                "content": "Test learning content",
                "category": "user_correction",
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["content"] == "Test learning content"
        assert data["category"] == "user_correction"

    def test_get_config(self, client_with_mock):
        """Test getting sanitized config."""
        response = client_with_mock.get("/api/config")

        assert response.status_code == 200
        data = response.json()
        assert "databases" in data
        assert "llm_provider" in data
        assert "llm_model" in data


class TestEntityEndpoints:
    """Tests for entity endpoints."""

    def test_list_entities(self, client_with_mock):
        """Test listing entities."""
        # Create a session
        create_response = client_with_mock.post("/api/sessions", json={})
        session_id = create_response.json()["session_id"]

        response = client_with_mock.get(f"/api/sessions/{session_id}/entities")

        assert response.status_code == 200
        data = response.json()
        assert "entities" in data


class TestIntegration:
    """Integration tests for full workflows."""

    def test_session_lifecycle(self, client_with_mock):
        """Test complete session lifecycle."""
        # 1. Create session
        create_response = client_with_mock.post("/api/sessions", json={"user_id": "integration_test"})
        assert create_response.status_code == 200
        session_id = create_response.json()["session_id"]

        # 2. Get session
        get_response = client_with_mock.get(f"/api/sessions/{session_id}")
        assert get_response.status_code == 200
        assert get_response.json()["status"] == "idle"

        # 3. List sessions (should include our session)
        list_response = client_with_mock.get("/api/sessions?user_id=integration_test")
        assert list_response.status_code == 200
        assert any(s["session_id"] == session_id for s in list_response.json()["sessions"])

        # 4. Get tables (empty initially)
        tables_response = client_with_mock.get(f"/api/sessions/{session_id}/tables")
        assert tables_response.status_code == 200
        assert tables_response.json()["tables"] == []

        # 5. Delete session
        delete_response = client_with_mock.delete(f"/api/sessions/{session_id}")
        assert delete_response.status_code == 200

        # 6. Verify deleted
        get_deleted_response = client_with_mock.get(f"/api/sessions/{session_id}")
        assert get_deleted_response.status_code == 404

    def test_health_and_schema(self, client_with_mock):
        """Test health and schema endpoints work together."""
        # Health check
        health_response = client_with_mock.get("/health")
        assert health_response.status_code == 200
        assert health_response.json()["status"] == "ok"

        # Schema overview
        schema_response = client_with_mock.get("/api/schema")
        assert schema_response.status_code == 200
        assert "databases" in schema_response.json()
