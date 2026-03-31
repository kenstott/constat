# Copyright (c) 2025 Kenneth Stott
# Canary: e81ab566-008c-4f9f-b382-ff09c9248309
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tests for diff generator registry and heartbeat processing."""

import threading
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from constat.server.diff_generators import EntityDiffGenerator


class TestEntityDiffGenerator:
    """Tests for EntityDiffGenerator.should_run logic."""

    def _make_managed(self, file_refs=None, ner_cancelled=None):
        managed = MagicMock()
        managed._file_refs = file_refs or []
        managed._ner_cancelled = ner_cancelled or threading.Event()
        managed._ner_thread = None
        managed.session_id = "test-session"
        return managed

    def test_first_heartbeat_with_docs_runs(self):
        gen = EntityDiffGenerator()
        managed = self._make_managed(file_refs=[{"name": "doc1", "added_at": "2026-01-01T00:00:00"}])
        assert gen.should_run(managed, since=None) is True

    def test_first_heartbeat_no_docs_skips(self):
        gen = EntityDiffGenerator()
        managed = self._make_managed(file_refs=[])
        assert gen.should_run(managed, since=None) is False

    def test_subsequent_heartbeat_with_new_ref(self):
        gen = EntityDiffGenerator()
        managed = self._make_managed(file_refs=[
            {"name": "doc1", "added_at": "2026-03-24T10:00:00"},
        ])
        assert gen.should_run(managed, since="2026-03-24T09:00:00") is True

    def test_subsequent_heartbeat_no_changes(self):
        gen = EntityDiffGenerator()
        managed = self._make_managed(file_refs=[
            {"name": "doc1", "added_at": "2026-03-24T08:00:00"},
        ])
        assert gen.should_run(managed, since="2026-03-24T09:00:00") is False

    def test_subsequent_heartbeat_last_refreshed(self):
        gen = EntityDiffGenerator()
        managed = self._make_managed(file_refs=[
            {"name": "doc1", "added_at": "2026-03-24T08:00:00", "last_refreshed": "2026-03-24T10:00:00"},
        ])
        assert gen.should_run(managed, since="2026-03-24T09:00:00") is True

    def test_run_calls_refresh(self):
        gen = EntityDiffGenerator()
        managed = self._make_managed()
        sm = MagicMock()
        gen.run(managed, sm, since=None)
        sm.refresh_entities_async.assert_called_once_with("test-session")


class TestProcessHeartbeat:
    """Tests for SessionManager.process_heartbeat."""

    def test_heartbeat_unknown_session(self):
        """process_heartbeat on unknown session returns timestamp without error."""
        from constat.server.session_manager import SessionManager
        sm = MagicMock(spec=SessionManager)
        sm._sessions = {}
        sm._active_connections = {}
        sm._diff_generators = []
        # Call the real method
        server_time = SessionManager.process_heartbeat(sm, "nonexistent", None)
        assert server_time  # ISO string

    def test_heartbeat_runs_generators(self):
        """process_heartbeat runs eligible generators."""
        from constat.server.session_manager import SessionManager
        managed = MagicMock()
        managed.touch = MagicMock()

        sm = MagicMock(spec=SessionManager)
        sm._sessions = {"s1": managed}
        sm._active_connections = {"s1": MagicMock()}
        sm.update_heartbeat = SessionManager.update_heartbeat.__get__(sm)

        gen = MagicMock()
        gen.name = "test"
        gen.should_run.return_value = True
        sm._diff_generators = [gen]

        server_time = SessionManager.process_heartbeat(sm, "s1", None)

        managed.touch.assert_called_once()
        gen.should_run.assert_called_once_with(managed, None)
        gen.run.assert_called_once_with(managed, sm, None)
        assert server_time

    def test_heartbeat_skips_when_should_run_false(self):
        """process_heartbeat skips generators that return should_run=False."""
        from constat.server.session_manager import SessionManager
        managed = MagicMock()
        managed.touch = MagicMock()

        sm = MagicMock(spec=SessionManager)
        sm._sessions = {"s1": managed}
        sm._active_connections = {"s1": MagicMock()}
        sm.update_heartbeat = SessionManager.update_heartbeat.__get__(sm)

        gen = MagicMock()
        gen.name = "test"
        gen.should_run.return_value = False
        sm._diff_generators = [gen]

        SessionManager.process_heartbeat(sm, "s1", "2026-03-24T09:00:00")
        gen.run.assert_not_called()


class TestActiveConnections:
    """Tests for connection tracking."""

    def test_register_and_list(self):
        from constat.server.session_manager import SessionManager
        sm = MagicMock(spec=SessionManager)
        sm._active_connections = {}
        SessionManager.register_connection(sm, "s1", "user1", "127.0.0.1:5000")
        connections = SessionManager.get_active_connections(sm)
        assert len(connections) == 1
        assert connections[0]["session_id"] == "s1"
        assert connections[0]["user_id"] == "user1"

    def test_unregister(self):
        from constat.server.session_manager import SessionManager
        sm = MagicMock(spec=SessionManager)
        sm._active_connections = {}
        SessionManager.register_connection(sm, "s1", "user1", "127.0.0.1:5000")
        SessionManager.unregister_connection(sm, "s1")
        assert SessionManager.get_active_connections(sm) == []

    def test_update_heartbeat(self):
        from constat.server.session_manager import SessionManager
        sm = MagicMock(spec=SessionManager)
        sm._active_connections = {}
        SessionManager.register_connection(sm, "s1", "user1", "127.0.0.1:5000")
        old_hb = sm._active_connections["s1"].last_heartbeat
        import time
        time.sleep(0.01)
        SessionManager.update_heartbeat(sm, "s1")
        new_hb = sm._active_connections["s1"].last_heartbeat
        assert new_hb >= old_hb


class TestGeneratorCooldown:
    """Tests for per-generator interval_seconds cooldown."""

    def test_cooldown_skips_within_interval(self):
        from constat.server.diff_generators import DiffGenerator

        class SlowGen(DiffGenerator):
            name = "slow"
            interval_seconds = 60

            def _should_run(self, managed, since):
                return True

            def _run(self, managed, session_manager, since):
                pass

        gen = SlowGen()
        managed = MagicMock()
        managed.session_id = "s1"

        # First call — should run (no previous run)
        assert gen.should_run(managed, None) is True
        gen.run(managed, MagicMock(), None)

        # Immediate second call — cooldown blocks it
        assert gen.should_run(managed, None) is False

    def test_zero_interval_always_eligible(self):
        gen = EntityDiffGenerator()
        managed = MagicMock()
        managed.session_id = "s1"
        managed._file_refs = [{"name": "d", "added_at": "2026-01-01T00:00:00"}]
        managed._ner_cancelled = threading.Event()
        managed._ner_thread = None

        assert gen.should_run(managed, None) is True
        gen.run(managed, MagicMock(), None)
        # Still eligible immediately (interval_seconds=0)
        assert gen.should_run(managed, None) is True
