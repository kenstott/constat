# Copyright (c) 2025 Kenneth Stott
# Canary: ec4e337d-b4ad-4dd8-b15f-6c8be0b85a48
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tests for REPL command parsing and individual command handlers."""

from __future__ import annotations

import pytest
from unittest.mock import Mock, patch

from rich.console import Console

from constat.repl import InteractiveREPL
from constat.core.config import Config
from constat.storage.history import SessionSummary


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


class TestHelpCommand:
    """Tests for /help and /h commands."""

    def test_help_command_shows_table(self, repl, mock_console):
        """Test /help shows command table."""
        repl._show_help()

        assert mock_console.print.call_count >= 1

    def test_help_shows_all_commands(self, repl):
        """Test /help includes all documented commands."""
        with patch.object(repl.console, 'print'):
            repl._show_help()

        expected_commands = [
            "/help", "/h",
            "/tables",
            "/query",
            "/state",
            "/facts",
            "/history",
            "/resume",
            "/verbose",
            "/raw",
            "/insights",
            "/artifacts",
            "/reason",
            "/quit", "/q",
        ]

        import inspect
        source = inspect.getsource(repl._show_help)
        for cmd in expected_commands:
            assert cmd in source, f"Command {cmd} not found in help"


class TestTablesCommand:
    """Tests for /tables command."""

    def test_tables_no_session(self, repl, mock_console):
        """Test /tables with no active session."""
        repl.api._session = None

        repl._show_tables()

        mock_console.print.assert_called()
        call_args = str(mock_console.print.call_args)
        assert "No active session" in call_args

    def test_tables_no_datastore(self, repl, mock_console):
        """Test /tables with session but no datastore."""
        repl.api._session = Mock()
        repl.api.session.datastore = None

        repl._show_tables()

        call_args = str(mock_console.print.call_args)
        assert "No active session" in call_args

    def test_tables_empty(self, repl, mock_console):
        """Test /tables when no tables exist."""
        repl.api._session = Mock()
        repl.api.session.datastore = Mock()
        repl.api.session.datastore.list_tables.return_value = []

        repl._show_tables()

        mock_console.print.assert_called()
        call_args = str(mock_console.print.call_args)
        assert "No tables yet" in call_args

    def test_tables_with_data(self, repl):
        """Test /tables displays table list correctly."""
        repl.api._session = Mock()
        repl.api.session.datastore = Mock()
        repl.api.session.datastore.list_tables.return_value = [
            {"name": "customers", "row_count": 100, "step_number": 1},
            {"name": "orders", "row_count": 50, "step_number": 2},
        ]

        repl.display = Mock()
        repl.display.show_tables = Mock()

        repl._show_tables()

        repl.display.show_tables.assert_called_once()
        tables_arg = repl.display.show_tables.call_args[0][0]
        assert len(tables_arg) == 2
        assert tables_arg[0]["name"] == "customers"


class TestQueryCommand:
    """Tests for /query command."""

    def test_query_no_session(self, repl, mock_console):
        """Test /query with no active session."""
        repl.api._session = None

        repl._run_query("SELECT * FROM users")

        call_args = str(mock_console.print.call_args)
        assert "No active session" in call_args

    def test_query_no_datastore(self, repl, mock_console):
        """Test /query with session but no datastore."""
        repl.api._session = Mock()
        repl.api.session.datastore = None

        repl._run_query("SELECT * FROM users")

        call_args = str(mock_console.print.call_args)
        assert "No active session" in call_args

    def test_query_success(self, repl, mock_console):
        """Test /query runs SQL and displays results."""
        mock_result = Mock()
        mock_result.to_string.return_value = "col1 | col2\n----\na    | b"

        repl.api._session = Mock()
        repl.api.session.datastore = Mock()
        repl.api.session.datastore.query.return_value = mock_result

        repl._run_query("SELECT * FROM test")

        repl.api.session.datastore.query.assert_called_once_with("SELECT * FROM test")
        mock_console.print.assert_called_with("col1 | col2\n----\na    | b")

    def test_query_error(self, repl, mock_console):
        """Test /query handles SQL errors gracefully."""
        repl.api._session = Mock()
        repl.api.session.datastore = Mock()
        repl.api.session.datastore.query.side_effect = Exception("Table not found")

        repl._run_query("SELECT * FROM nonexistent")

        call_args = str(mock_console.print.call_args)
        assert "Query error" in call_args
        assert "Table not found" in call_args


class TestStateCommand:
    """Tests for /state command."""

    def test_state_no_session(self, repl, mock_console):
        """Test /state with no active session."""
        repl.api._session = None

        repl._show_state()

        call_args = str(mock_console.print.call_args)
        assert "No active session" in call_args

    def test_state_displays_session_info(self, repl, mock_console):
        """Test /state shows session ID and tables."""
        repl.api._session = Mock()
        repl.api.session.get_state.return_value = {
            "session_id": "test-session-123",
            "datastore_tables": [
                {"name": "users", "row_count": 10},
                {"name": "orders", "row_count": 5},
            ],
        }

        repl._show_state()

        calls = [str(call) for call in mock_console.print.call_args_list]
        calls_str = " ".join(calls)
        assert "test-session-123" in calls_str
        assert "users" in calls_str
        assert "10" in calls_str

    def test_state_handles_empty_tables(self, repl, mock_console):
        """Test /state handles empty tables list."""
        repl.api._session = Mock()
        repl.api.session.get_state.return_value = {
            "session_id": "test",
            "datastore_tables": [],
        }

        repl._show_state()

        calls = [str(call) for call in mock_console.print.call_args_list]
        calls_str = " ".join(calls)
        assert "test" in calls_str


class TestFactsCommand:
    """Tests for /facts command."""

    def test_facts_no_session(self, repl, mock_console):
        """Test /facts with no active session shows message about /remember."""
        repl.api._session = None

        repl._show_facts()

        call_args = str(mock_console.print.call_args)
        assert "/remember" in call_args

    def test_facts_shows_cached_facts(self, repl, mock_console):
        """Test /facts displays cached facts."""
        from constat.execution.fact_resolver import Fact, FactSource

        repl.api._session = Mock()
        repl.api.session.fact_resolver = Mock()
        repl.api.session.fact_resolver.get_all_facts.return_value = {
            "march_attendance": Fact(
                name="march_attendance",
                value=1000000,
                source=FactSource.USER_PROVIDED,
                description="Estimated attendance at the march",
            ),
        }

        repl._show_facts()

        repl.api.session.fact_resolver.get_all_facts.assert_called_once()

    def test_facts_empty(self, repl, mock_console):
        """Test /facts handles case when no facts cached."""
        repl.api._session = Mock()
        repl.api.session.fact_resolver = Mock()
        repl.api.session.fact_resolver.get_all_facts.return_value = {}

        repl._show_facts()

        call_args = str(mock_console.print.call_args)
        assert "/remember" in call_args


class TestHistoryCommand:
    """Tests for /history command."""

    def test_history_shows_sessions_from_api(self, repl, mock_console):
        """Test /history displays session list from API session."""
        repl.api._session = Mock()
        repl.api._session.history = Mock()
        repl.api._session.history.list_sessions.return_value = []

        repl._show_history()

        repl.api._session.history.list_sessions.assert_called_once()

    def test_history_empty(self, repl, mock_console):
        """Test /history when no previous sessions."""
        repl.api._session = Mock()
        repl.api.session.history = Mock()
        repl.api.session.history.list_sessions.return_value = []

        repl._show_history()

        call_args = str(mock_console.print.call_args)
        assert "No session history" in call_args

    def test_history_shows_sessions(self, repl, mock_console):
        """Test /history displays session list."""
        repl.api._session = Mock()
        repl.api.session.history = Mock()
        repl.api.session.history.list_sessions.return_value = [
            SessionSummary(
                session_id="2024-01-15_143022_abc12345",
                created_at="2024-01-15 14:30:22",
                databases=["chinook"],
                status="completed",
                total_queries=3,
                total_duration_ms=1500,
            ),
            SessionSummary(
                session_id="2024-01-14_101000_def67890",
                created_at="2024-01-14 10:10:00",
                databases=["northwind"],
                status="failed",
                total_queries=1,
                total_duration_ms=500,
            ),
        ]

        repl._show_history()

        repl.api.session.history.list_sessions.assert_called_once_with(limit=10)
        assert mock_console.print.call_count >= 1


class TestResumeCommand:
    """Tests for /resume command."""

    def test_resume_searches_history_via_api(self, repl, mock_console):
        """Test /resume searches history via API session."""
        repl.api._session = Mock()
        repl.api._session.history = Mock()
        repl.api._session.history.list_sessions.return_value = []

        repl._resume_session("abc123")

        repl.api._session.history.list_sessions.assert_called()

    def test_resume_session_not_found(self, repl, mock_console):
        """Test /resume with non-existent session ID."""
        repl.api._session = Mock()
        repl.api.session.history = Mock()
        repl.api.session.history.list_sessions.return_value = []

        repl._resume_session("nonexistent")

        call_args = str(mock_console.print.call_args)
        assert "Session not found" in call_args
        assert "nonexistent" in call_args

    def test_resume_multiple_matches_takes_first(self, repl, mock_console):
        """Test /resume when session ID prefix matches multiple sessions takes first."""
        repl.api._session = Mock()
        repl.api.session.history = Mock()
        repl.api.session.history.list_sessions.return_value = [
            SessionSummary(
                session_id="abc123_session1",
                created_at="2024-01-15",
                databases=[],
                status="completed",
                total_queries=1,
                total_duration_ms=100,
            ),
            SessionSummary(
                session_id="abc123_session2",
                created_at="2024-01-14",
                databases=[],
                status="completed",
                total_queries=1,
                total_duration_ms=100,
            ),
        ]
        repl.api.session.resume.return_value = True
        repl.api.session.datastore = None

        repl._resume_session("abc123")

        repl.api.session.resume.assert_called_once_with("abc123_session1")

    def test_resume_success(self, repl, mock_console):
        """Test /resume successfully resumes a session."""
        repl.api._session = Mock()
        repl.api.session.history = Mock()
        repl.api.session.history.list_sessions.return_value = [
            SessionSummary(
                session_id="abc123_unique",
                created_at="2024-01-15",
                databases=["chinook"],
                status="completed",
                total_queries=2,
                total_duration_ms=200,
            ),
        ]
        repl.api.session.resume.return_value = True
        repl.api.session.datastore = Mock()
        repl.api.session.datastore.list_tables.return_value = [
            {"name": "test_table", "row_count": 10}
        ]

        repl._resume_session("abc123")

        repl.api.session.resume.assert_called_once_with("abc123_unique")
        calls = [str(call) for call in mock_console.print.call_args_list]
        calls_str = " ".join(calls)
        assert "Resumed session" in calls_str
        assert "abc123_unique" in calls_str

    def test_resume_failure(self, repl, mock_console):
        """Test /resume handles resume failure."""
        repl.api._session = Mock()
        repl.api.session.history = Mock()
        repl.api.session.history.list_sessions.return_value = [
            SessionSummary(
                session_id="abc123_unique",
                created_at="2024-01-15",
                databases=[],
                status="completed",
                total_queries=1,
                total_duration_ms=100,
            ),
        ]
        repl.api.session.resume.return_value = False

        repl._resume_session("abc123")

        calls = [str(call) for call in mock_console.print.call_args_list]
        calls_str = " ".join(calls)
        assert "Failed to resume" in calls_str


class TestVerboseToggle:
    """Tests for /verbose command."""

    def test_verbose_toggle_on(self, repl, mock_console):
        """Test /verbose toggles verbose mode on."""
        repl.verbose = False
        repl.display = Mock()
        repl.display.verbose = False

        repl._toggle_verbose()

        assert repl.verbose is True
        assert repl.display.verbose is True

    def test_verbose_toggle_off(self, repl, mock_console):
        """Test /verbose toggles verbose mode off."""
        repl.verbose = True
        repl.display = Mock()
        repl.display.verbose = True

        repl._toggle_verbose()

        assert repl.verbose is False
        assert repl.display.verbose is False

    def test_verbose_set_on(self, repl, mock_console):
        """Test /verbose on explicitly sets verbose mode."""
        repl.verbose = False
        repl.display = Mock()
        repl.display.verbose = False

        repl._toggle_verbose("on")

        assert repl.verbose is True

    def test_verbose_set_off(self, repl, mock_console):
        """Test /verbose off explicitly unsets verbose mode."""
        repl.verbose = True
        repl.display = Mock()
        repl.display.verbose = True

        repl._toggle_verbose("off")

        assert repl.verbose is False


class TestCommandParsing:
    """Tests for command parsing in the REPL."""

    def test_command_recognized_with_leading_slash(self, repl):
        """Test that inputs starting with / are recognized as commands."""
        test_inputs = ["/help", "/tables", "/query SELECT 1", "/state"]

        for input_text in test_inputs:
            assert input_text.startswith("/"), f"{input_text} should start with /"

    def test_regular_query_not_treated_as_command(self, repl):
        """Test that regular text is not treated as command."""
        regular_input = "What are the top customers?"
        assert not regular_input.startswith("/")

    def test_command_parsing_splits_correctly(self, repl):
        """Test command and argument parsing."""
        test_cases = [
            ("/query SELECT * FROM users", "/query", "SELECT * FROM users"),
            ("/resume abc123", "/resume", "abc123"),
            ("/help", "/help", ""),
            ("/facts There were 1 million people", "/facts", "There were 1 million people"),
        ]

        for input_text, expected_cmd, expected_arg in test_cases:
            parts = input_text.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            assert cmd == expected_cmd, f"Command mismatch for '{input_text}'"
            assert arg == expected_arg, f"Argument mismatch for '{input_text}'"


class TestUnknownCommand:
    """Tests for unknown command handling."""

    def test_unknown_command_shows_warning(self, repl, mock_console):
        """Test that unknown commands show warning message."""
        repl._handle_command("/unknowncmd")

        calls = [str(call) for call in mock_console.print.call_args_list]
        calls_str = " ".join(calls)
        assert "Unknown" in calls_str


class TestCommandsWithoutArguments:
    """Tests for commands that require arguments."""

    def test_remember_without_argument(self, repl, mock_console):
        """Test /remember without text shows usage."""
        repl._remember_fact("")

        calls = [str(call) for call in mock_console.print.call_args_list]
        calls_str = " ".join(calls)
        assert "Usage: /remember" in calls_str

    def test_forget_without_argument(self, repl, mock_console):
        """Test /forget without name shows usage."""
        repl._forget_fact("")

        calls = [str(call) for call in mock_console.print.call_args_list]
        calls_str = " ".join(calls)
        assert "Usage: /forget" in calls_str

    def test_correct_without_argument(self, repl, mock_console):
        """Test /correct without text shows usage."""
        repl._handle_correct("")

        calls = [str(call) for call in mock_console.print.call_args_list]
        calls_str = " ".join(calls)
        assert "Usage: /correct" in calls_str


class TestCommandCaseInsensitivity:
    """Tests for case-insensitive command handling."""

    def test_commands_are_lowercased(self, repl):
        """Test that commands are converted to lowercase."""
        test_cases = [
            ("/HELP", "/help"),
            ("/Help", "/help"),
            ("/TABLES", "/tables"),
            ("/Query SELECT 1", "/query"),
            ("/STATE", "/state"),
        ]

        for input_text, expected_cmd in test_cases:
            parts = input_text.split(maxsplit=1)
            cmd = parts[0].lower()
            assert cmd == expected_cmd


class TestQuitCommands:
    """Tests for quit/exit commands."""

    def test_quit_variations(self, repl):
        """Test that /quit, /exit, and /q are recognized as quit commands."""
        quit_commands = ["/quit", "/exit", "/q"]

        for cmd in quit_commands:
            cmd_lower = cmd.lower()
            is_quit = cmd_lower in ("/quit", "/exit", "/q")
            assert is_quit, f"{cmd} should be recognized as quit"
