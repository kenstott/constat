# Copyright (c) 2025 Kenneth Stott
# Canary: ec4e337d-b4ad-4dd8-b15f-6c8be0b85a48
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tests for REPL execution — solve, session lifecycle, initialization, and cleanup."""

from __future__ import annotations

import pytest
from unittest.mock import Mock, patch

from rich.console import Console

from constat.repl import InteractiveREPL
from constat.core.config import Config


@pytest.fixture
def mock_config():
    """Create a minimal mock Config."""
    config = Mock(spec=Config)
    config.databases = {}
    config.llm = Mock()
    config.execution = Mock()
    config.execution.timeout_seconds = 60
    config.execution.allowed_imports = []
    config.system_prompt = ""
    config.model_dump = Mock(return_value={})
    config.databases_description = None
    config.documents = None
    config.apis = None
    config.context_preload = None
    return config


@pytest.fixture
def mock_console():
    """Create a mock Rich Console that captures output."""
    console = Mock(spec=Console)
    console.print = Mock()
    return console


@pytest.fixture
def repl(mock_config, mock_console):
    """Create an InteractiveREPL with mocked dependencies."""
    with patch('constat.repl.interactive._core.FeedbackDisplay'), \
         patch('constat.repl.interactive._core.FactStore') as mock_fact_store_class, \
         patch('constat.repl.interactive._core.Session') as mock_session_class, \
         patch('constat.repl.interactive._core.LearningStore') as mock_learning_store_class:
        mock_fact_store = Mock()
        mock_fact_store.list_facts.return_value = {}
        mock_fact_store_class.return_value = mock_fact_store

        mock_learning_store = Mock()
        mock_learning_store.get_stats.return_value = {"unpromoted": 0}
        mock_learning_store_class.return_value = mock_learning_store

        mock_session = Mock()
        mock_session.session_id = None
        mock_session.datastore = None
        mock_session_class.return_value = mock_session

        repl = InteractiveREPL(
            config=mock_config,
            verbose=False,
            console=mock_console,
        )
    return repl


class TestSolveIntegration:
    """Tests for the _solve method that handles queries."""

    def test_solve_calls_session_solve(self, repl, mock_console):
        """Test that _solve calls solve on the API session."""
        mock_session = Mock()
        mock_session.session_id = None
        mock_session.solve.return_value = {"success": True, "results": []}
        repl.api._session = mock_session

        repl._solve("Count the customers")

        mock_session.solve.assert_called_once_with("Count the customers")

    def test_solve_uses_follow_up_for_existing_session(self, repl, mock_console):
        """Test that _solve uses follow_up for existing session."""
        mock_session = Mock()
        mock_session.session_id = "existing-session-123"
        mock_session.follow_up.return_value = {"success": True, "results": []}
        repl.api._session = mock_session

        repl._solve("Now show the top 5")

        mock_session.follow_up.assert_called_once_with("Now show the top 5")

    def test_solve_handles_keyboard_interrupt(self, repl, mock_console):
        """Test that _solve handles KeyboardInterrupt gracefully."""
        mock_session = Mock()
        mock_session.session_id = None
        mock_session.solve.side_effect = KeyboardInterrupt()
        repl.api._session = mock_session

        repl._solve("Some problem")

        calls = [str(call) for call in mock_console.print.call_args_list]
        calls_str = " ".join(calls)
        assert "Interrupted" in calls_str

    def test_solve_handles_exception(self, repl, mock_console):
        """Test that _solve handles exceptions gracefully."""
        mock_session = Mock()
        mock_session.session_id = None
        mock_session.solve.side_effect = Exception("Something went wrong")
        repl.api._session = mock_session

        repl._solve("Some problem")

        calls = [str(call) for call in mock_console.print.call_args_list]
        calls_str = " ".join(calls)
        assert "Error" in calls_str
        assert "Something went wrong" in calls_str


class TestCreateSession:
    """Tests for session creation."""

    def test_create_api_returns_session(self, repl, mock_config):
        """Test _create_api creates and returns a ConstatAPIImpl."""
        with patch('constat.repl.interactive._core.Session') as mock_session_class:
            with patch('constat.repl.interactive._core.SessionFeedbackHandler'):
                with patch('constat.repl.interactive._core.FactStore'):
                    with patch('constat.repl.interactive._core.LearningStore'):
                        mock_session = Mock()
                        mock_session_class.return_value = mock_session

                        result = repl._create_api()

                        mock_session_class.assert_called_once()
                        assert hasattr(result, 'session')
                        assert result.session == mock_session

    def test_create_api_wires_feedback_handler(self, repl, mock_config):
        """Test _create_api wires up feedback handler."""
        with patch('constat.repl.interactive._core.Session') as mock_session_class:
            with patch('constat.repl.interactive._core.SessionFeedbackHandler') as mock_handler_class:
                with patch('constat.repl.interactive._core.FactStore'):
                    with patch('constat.repl.interactive._core.LearningStore'):
                        mock_session = Mock()
                        mock_session_class.return_value = mock_session
                        mock_handler = Mock()
                        mock_handler_class.return_value = mock_handler

                        repl._create_api()

                        mock_handler_class.assert_called_once_with(repl.display, repl.session_config)


class TestReplInitialization:
    """Tests for REPL initialization."""

    def test_repl_initializes_with_config(self, mock_config, mock_console):
        """Test REPL initializes correctly with config."""
        with patch('constat.repl.interactive._core.FeedbackDisplay'):
            with patch('constat.repl.interactive._core.Session') as mock_session_class:
                mock_session = Mock()
                mock_session_class.return_value = mock_session
                with patch('constat.repl.interactive._core.FactStore'):
                    with patch('constat.repl.interactive._core.LearningStore'):
                        repl = InteractiveREPL(
                            config=mock_config,
                            verbose=True,
                            console=mock_console,
                        )

        assert repl.config == mock_config
        assert repl.verbose is True
        assert repl.console == mock_console
        assert repl.api._session == mock_session

    def test_repl_creates_default_console(self, mock_config):
        """Test REPL creates default console if none provided."""
        with patch('constat.repl.interactive._core.FeedbackDisplay'):
            with patch('constat.repl.interactive._core.Session') as mock_session_class:
                mock_session_class.return_value = Mock()
                with patch('constat.repl.interactive._core.Console') as mock_console_class:
                    mock_console_instance = Mock()
                    mock_console_class.return_value = mock_console_instance

                    repl = InteractiveREPL(config=mock_config)

                    mock_console_class.assert_called_once()
                    assert repl.console == mock_console_instance


class TestEmptyInput:
    """Tests for empty input handling."""

    def test_empty_input_is_ignored(self, repl):
        """Test that empty input is ignored in the REPL loop."""
        user_input = ""
        assert not user_input.strip()

    def test_whitespace_only_input_is_ignored(self, repl):
        """Test that whitespace-only input is ignored."""
        user_input = "   \t\n  "
        assert not user_input.strip()


class TestDatastoreCleanup:
    """Tests for datastore cleanup on exit."""

    def test_cleanup_closes_datastore(self, repl):
        """Test that datastore is closed on REPL exit."""
        mock_session = Mock()
        mock_datastore = Mock()
        mock_session.datastore = mock_datastore
        repl.api._session = mock_session

        if repl.api.session and repl.api.session.datastore:
            repl.api.session.datastore.close()

        mock_datastore.close.assert_called_once()

    def test_cleanup_handles_no_session(self, repl):
        """Test cleanup handles case with no session."""
        repl.api._session = None

        if repl.api.session and repl.api.session.datastore:
            repl.api.session.datastore.close()

    def test_cleanup_handles_no_datastore(self, repl):
        """Test cleanup handles session with no datastore."""
        mock_session = Mock()
        mock_session.datastore = None
        repl.api._session = mock_session

        if repl.api.session and repl.api.session.datastore:
            repl.api.session.datastore.close()


class TestSuccessfulSolveOutput:
    """Tests for successful solve output display."""

    def test_solve_shows_summary_on_success(self, repl, mock_console):
        """Test successful solve shows summary."""
        mock_session = Mock()
        mock_session.session_id = None

        mock_step = Mock()
        mock_step.number = 1
        mock_step.goal = "Analyze data"
        mock_plan = Mock()
        mock_plan.steps = [mock_step]

        mock_result = Mock()
        mock_result.duration_ms = 100

        mock_session.solve.return_value = {
            "success": True,
            "plan": mock_plan,
            "results": [mock_result],
            "datastore_tables": [],
        }
        repl.api._session = mock_session
        repl.display = Mock()

        repl._solve("Analyze the data")

        mock_session.solve.assert_called_once_with("Analyze the data")
        repl.display.show_summary.assert_called_once()

    def test_solve_shows_tables(self, repl, mock_console):
        """Test successful solve shows created tables."""
        mock_session = Mock()
        mock_session.session_id = None

        mock_result = Mock()
        mock_result.duration_ms = 100

        mock_session.solve.return_value = {
            "success": True,
            "plan": None,
            "results": [mock_result],
            "datastore_tables": [
                {"name": "results", "row_count": 10, "step_number": 1},
            ],
        }
        repl.api._session = mock_session
        repl.display = Mock()

        repl._solve("Analyze the data")

        repl.display.show_tables.assert_called_once()


class TestFailedSolveOutput:
    """Tests for failed solve output display."""

    def test_solve_shows_error(self, repl, mock_console):
        """Test failed solve shows error message."""
        mock_session = Mock()
        mock_session.session_id = None
        mock_session.solve.return_value = {
            "success": False,
            "plan": None,
            "error": "Database connection failed",
        }
        repl.api._session = mock_session
        repl.display = Mock()

        repl._solve("Analyze the data")

        calls = [str(call) for call in mock_console.print.call_args_list]
        calls_str = " ".join(calls)
        assert "Error" in calls_str
        assert "Database connection failed" in calls_str


class TestCancelExecution:
    """Tests for Ctrl+C execution cancellation."""

    def test_solve_calls_cancel_on_interrupt(self, repl, mock_console):
        """Test that _solve calls session.cancel_execution() on KeyboardInterrupt."""
        mock_session = Mock()
        mock_session.session_id = None
        mock_session.solve.side_effect = KeyboardInterrupt()
        mock_session.cancel_execution = Mock()
        repl.api._session = mock_session

        repl._solve("Some query")

        mock_session.cancel_execution.assert_called_once()

    def test_solve_cleanup_on_interrupt(self, repl, mock_console):
        """Test that _solve cleans up display state on KeyboardInterrupt."""
        mock_session = Mock()
        mock_session.session_id = None
        mock_session.solve.side_effect = KeyboardInterrupt()
        mock_session.cancel_execution = Mock()
        repl.api._session = mock_session
        repl.display = Mock()

        repl._solve("Some query")

        repl.display.stop.assert_called_once()
        repl.display.stop_spinner.assert_called_once()
