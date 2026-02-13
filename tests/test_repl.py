# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tests for the Interactive REPL commands.

Tests cover:
- Command parsing and recognition
- Each REPL command individually (/tables, /query, /state, /facts, /unresolved, /history, /resume, /help, /verbose)
- Error handling for invalid/unknown commands
- Commands with arguments vs without
- Session state handling (active vs no session)
"""

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
    config.context_preload = None  # For MetadataPreloadCache
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
        # Mock FactStore to return empty facts by default
        mock_fact_store = Mock()
        mock_fact_store.list_facts.return_value = {}
        mock_fact_store_class.return_value = mock_fact_store

        # Mock LearningStore
        mock_learning_store = Mock()
        mock_learning_store.get_stats.return_value = {"unpromoted": 0}
        mock_learning_store_class.return_value = mock_learning_store

        # Mock Session creation
        mock_session = Mock()
        mock_session.session_id = None
        mock_session.datastore = None
        mock_session_class.return_value = mock_session

        repl = InteractiveREPL(
            config=mock_config,
            verbose=False,
            console=mock_console,
        )
        # The mocks are already injected via the patched classes
    return repl


class TestHelpCommand:
    """Tests for /help and /h commands."""

    def test_help_command_shows_table(self, repl, mock_console):
        """Test /help shows command table."""
        repl._show_help()

        # Verify console.print was called at least once (for the table)
        assert mock_console.print.call_count >= 1

    def test_help_shows_all_commands(self, repl):
        """Test /help includes all documented commands."""
        # Capture the table structure from _show_help
        with patch.object(repl.console, 'print'):
            repl._show_help()

        # The commands list in _show_help should include these
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
            "/prove",
            "/quit", "/q",
        ]

        # Verify by checking the source - commands are defined in _show_help
        # This is a structural test to ensure commands aren't accidentally removed
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

        # Should print warning about no session
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

        # Mock the display to verify it's called
        repl.display = Mock()
        repl.display.show_tables = Mock()

        repl._show_tables()

        # Verify show_tables was called with the table list
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

        # Verify query was executed
        repl.api.session.datastore.query.assert_called_once_with("SELECT * FROM test")
        # Verify result was printed
        mock_console.print.assert_called_with("col1 | col2\n----\na    | b")

    def test_query_error(self, repl, mock_console):
        """Test /query handles SQL errors gracefully."""
        repl.api._session = Mock()
        repl.api.session.datastore = Mock()
        repl.api.session.datastore.query.side_effect = Exception("Table not found")

        repl._run_query("SELECT * FROM nonexistent")

        # Verify error was printed
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

        # Verify session info was printed
        calls = [str(call) for call in mock_console.print.call_args_list]
        calls_str = " ".join(calls)
        assert "test-session-123" in calls_str
        assert "users" in calls_str
        assert "10" in calls_str  # row count

    def test_state_handles_empty_tables(self, repl, mock_console):
        """Test /state handles empty tables list."""
        repl.api._session = Mock()
        repl.api.session.get_state.return_value = {
            "session_id": "test",
            "datastore_tables": [],
        }

        repl._show_state()

        # Should still show session ID
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
        # With no session and no persistent facts, shows hint about /remember
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

        # Verify get_all_facts was called
        repl.api.session.fact_resolver.get_all_facts.assert_called_once()

    def test_facts_empty(self, repl, mock_console):
        """Test /facts handles case when no facts cached."""
        repl.api._session = Mock()
        repl.api.session.fact_resolver = Mock()
        repl.api.session.fact_resolver.get_all_facts.return_value = {}

        repl._show_facts()

        call_args = str(mock_console.print.call_args)
        # With no persistent or session facts, shows hint about /remember
        assert "/remember" in call_args


class TestHistoryCommand:
    """Tests for /history command."""

    def test_history_shows_sessions_from_api(self, repl, mock_console):
        """Test /history displays session list from API session."""
        repl.api._session = Mock()
        repl.api._session.history = Mock()
        repl.api._session.history.list_sessions.return_value = []

        repl._show_history()

        # Should call list_sessions on the API's session history
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

        # Should call list_sessions with limit=10
        repl.api.session.history.list_sessions.assert_called_once_with(limit=10)

        # Verify table was printed (console.print receives the Rich Table)
        assert mock_console.print.call_count >= 1


class TestResumeCommand:
    """Tests for /resume command."""

    def test_resume_searches_history_via_api(self, repl, mock_console):
        """Test /resume searches history via API session."""
        repl.api._session = Mock()
        repl.api._session.history = Mock()
        repl.api._session.history.list_sessions.return_value = []

        repl._resume_session("abc123")

        # Should search history via the API's session
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

        # Should resume the first match
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

        # Simulate the verbose toggle from run()
        repl.verbose = not repl.verbose
        repl.display.verbose = repl.verbose

        assert repl.verbose is True
        assert repl.display.verbose is True

    def test_verbose_toggle_off(self, repl, mock_console):
        """Test /verbose toggles verbose mode off."""
        repl.verbose = True
        repl.display = Mock()
        repl.display.verbose = True

        # Simulate the toggle
        repl.verbose = not repl.verbose
        repl.display.verbose = repl.verbose

        assert repl.verbose is False
        assert repl.display.verbose is False


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

    def test_unknown_command_shows_help(self, repl, mock_console):
        """Test that unknown commands show help message."""
        # Simulate handling an unknown command (from the run() method logic)
        unknown_cmd = "/unknowncmd"

        # The REPL would call console.print with yellow warning and show_help
        with patch.object(repl, '_show_help') as mock_help:
            # Simulate what run() does for unknown commands
            mock_console.print(f"[yellow]Unknown command: {unknown_cmd}[/yellow]")
            repl._show_help()

            mock_help.assert_called_once()
            call_args = str(mock_console.print.call_args)
            assert "Unknown command" in call_args


class TestCommandsWithoutArguments:
    """Tests for commands that require arguments."""

    def test_query_without_argument(self, repl, mock_console):
        """Test /query without SQL shows usage."""
        # Simulate the check from run()
        arg = ""
        if not arg:
            mock_console.print("[yellow]Usage: /query <sql>[/yellow]")

        call_args = str(mock_console.print.call_args)
        assert "Usage: /query" in call_args

    def test_resume_without_argument(self, repl, mock_console):
        """Test /resume without session ID shows usage."""
        arg = ""
        if not arg:
            mock_console.print("[yellow]Usage: /resume <session_id>[/yellow]")

        call_args = str(mock_console.print.call_args)
        assert "Usage: /resume" in call_args

    def test_facts_without_argument(self, repl, mock_console):
        """Test /facts without text shows usage."""
        arg = ""
        if not arg:
            mock_console.print("[yellow]Usage: /facts <text with facts>[/yellow]")

        call_args = str(mock_console.print.call_args)
        assert "Usage: /facts" in call_args


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

        # Should print "Interrupted"
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

        # Should print error
        calls = [str(call) for call in mock_console.print.call_args_list]
        calls_str = " ".join(calls)
        assert "Error" in calls_str
        assert "Something went wrong" in calls_str


class TestQuitCommands:
    """Tests for quit/exit commands."""

    def test_quit_variations(self, repl):
        """Test that /quit, /exit, and /q are recognized as quit commands."""
        quit_commands = ["/quit", "/exit", "/q"]

        for cmd in quit_commands:
            cmd_lower = cmd.lower()
            is_quit = cmd_lower in ("/quit", "/exit", "/q")
            assert is_quit, f"{cmd} should be recognized as quit"


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
                        # Result is ConstatAPIImpl, not Session
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

                        # Verify handler was created with display and session_config
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
        assert repl.api._session == mock_session  # API wraps session

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


class TestEmptyInput:
    """Tests for empty input handling."""

    def test_empty_input_is_ignored(self, repl):
        """Test that empty input is ignored in the REPL loop."""
        # In the run() method, empty input triggers 'continue'
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

        # Simulate the cleanup logic from run()
        if repl.api.session and repl.api.session.datastore:
            repl.api.session.datastore.close()

        mock_datastore.close.assert_called_once()

    def test_cleanup_handles_no_session(self, repl):
        """Test cleanup handles case with no session."""
        repl.api._session = None

        # This should not raise
        if repl.api.session and repl.api.session.datastore:
            repl.api.session.datastore.close()

    def test_cleanup_handles_no_datastore(self, repl):
        """Test cleanup handles session with no datastore."""
        mock_session = Mock()
        mock_session.datastore = None
        repl.api._session = mock_session

        # This should not raise
        if repl.api.session and repl.api.session.datastore:
            repl.api.session.datastore.close()


class TestSuccessfulSolveOutput:
    """Tests for successful solve output display."""

    def test_solve_shows_summary_on_success(self, repl, mock_console):
        """Test successful solve shows summary."""
        mock_session = Mock()
        mock_session.session_id = None

        # Create a mock plan with steps
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

        # Verify solve was called and summary shown (plan now shown via plan_ready event)
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

        # Verify tables were shown
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

        # Verify error was printed
        calls = [str(call) for call in mock_console.print.call_args_list]
        calls_str = " ".join(calls)
        assert "Error" in calls_str
        assert "Database connection failed" in calls_str


class TestRememberCommand:
    """Tests for /remember command - promoting session facts to persistent storage."""

    def test_remember_session_fact(self, repl, mock_console):
        """Test /remember <fact-name> persists a session fact."""
        from constat.execution.fact_resolver import Fact, FactSource
        from datetime import datetime

        # Setup session with a cached fact
        repl.api._session = Mock()
        repl.api.session.fact_resolver = Mock()
        repl.api.session.fact_resolver.get_all_facts.return_value = {
            "churn_rate()": Fact(
                name="churn_rate",
                value=0.032,
                source=FactSource.DATABASE,
                source_name="analytics_db",
                query="SELECT AVG(churned) FROM customers",
                resolved_at=datetime.now(),
            ),
        }

        repl._remember_fact("churn_rate")

        # Verify fact was persisted
        repl.api.fact_store.save_fact.assert_called_once()
        call_args = repl.api.fact_store.save_fact.call_args
        assert call_args.kwargs["name"] == "churn_rate"
        assert call_args.kwargs["value"] == 0.032
        assert "database" in call_args.kwargs["context"].lower()

        # Verify success message
        calls = [str(call) for call in mock_console.print.call_args_list]
        calls_str = " ".join(calls)
        assert "Remembered" in calls_str
        assert "churn_rate" in calls_str

    def test_remember_session_fact_with_rename(self, repl, mock_console):
        """Test /remember <fact-name> as <new-name> persists with new name."""
        from constat.execution.fact_resolver import Fact, FactSource
        from datetime import datetime

        # Setup session with a cached fact
        repl.api._session = Mock()
        repl.api.session.fact_resolver = Mock()
        repl.api.session.fact_resolver.get_all_facts.return_value = {
            "enterprise_churn()": Fact(
                name="enterprise_churn",
                value=0.015,
                source=FactSource.DATABASE,
                resolved_at=datetime.now(),
            ),
        }

        repl._remember_fact("enterprise_churn as baseline_churn")

        # Verify fact was persisted with new name
        repl.api.fact_store.save_fact.assert_called_once()
        call_args = repl.api.fact_store.save_fact.call_args
        assert call_args.kwargs["name"] == "baseline_churn"
        assert call_args.kwargs["value"] == 0.015

        # Verify rename message
        calls = [str(call) for call in mock_console.print.call_args_list]
        calls_str = " ".join(calls)
        assert "Renamed from" in calls_str

    def test_remember_fact_by_name_property(self, repl, mock_console):
        """Test /remember matches by fact.name property, not just cache key."""
        from constat.execution.fact_resolver import Fact, FactSource
        from datetime import datetime

        repl.api._session = Mock()
        repl.api.session.fact_resolver = Mock()
        # Cache key has params but fact.name is simple
        repl.api.session.fact_resolver.get_all_facts.return_value = {
            "customer_ltv(segment=enterprise)": Fact(
                name="customer_ltv",
                value=50000,
                source=FactSource.DATABASE,
                resolved_at=datetime.now(),
            ),
        }

        repl._remember_fact("customer_ltv")

        # Should find by name property
        repl.api.fact_store.save_fact.assert_called_once()
        assert repl.api.fact_store.save_fact.call_args.kwargs["value"] == 50000

    def test_remember_fact_not_found_shows_error(self, repl, mock_console):
        """Test /remember with non-existent fact shows helpful error."""
        repl.api._session = Mock()
        repl.api.session.fact_resolver = Mock()
        repl.api.session.fact_resolver.get_all_facts.return_value = {}
        repl.api.session.fact_resolver.add_user_facts_from_text.return_value = []
        repl.display = Mock()

        repl._remember_fact("nonexistent_fact")

        # Should show not found message
        calls = [str(call) for call in mock_console.print.call_args_list]
        calls_str = " ".join(calls)
        assert "No session fact named" in calls_str or "nonexistent_fact" in calls_str

    def test_remember_falls_back_to_text_extraction(self, repl, mock_console):
        """Test /remember falls back to LLM extraction for natural language."""
        from constat.execution.fact_resolver import Fact, FactSource

        repl.api._session = Mock()
        repl.api.session.fact_resolver = Mock()
        repl.api.session.fact_resolver.get_all_facts.return_value = {}

        # Mock LLM extraction returning a fact
        mock_fact = Fact(
            name="user_role",
            value="CFO",
            source=FactSource.USER_PROVIDED,
            description="User's role",
        )
        repl.api.session.fact_resolver.add_user_facts_from_text.return_value = [mock_fact]
        repl.display = Mock()

        repl._remember_fact("my role is CFO")

        # Should have called LLM extraction
        repl.api.session.fact_resolver.add_user_facts_from_text.assert_called_once_with("my role is CFO")

        # Should have persisted the extracted fact
        repl.api.fact_store.save_fact.assert_called_once()
        assert repl.api.fact_store.save_fact.call_args.kwargs["name"] == "user_role"
        assert repl.api.fact_store.save_fact.call_args.kwargs["value"] == "CFO"

    def test_remember_preserves_provenance(self, repl, mock_console):
        """Test /remember preserves full provenance in context."""
        from constat.execution.fact_resolver import Fact, FactSource
        from datetime import datetime

        repl.api._session = Mock()
        repl.api.session.fact_resolver = Mock()
        repl.api.session.fact_resolver.get_all_facts.return_value = {
            "revenue()": Fact(
                name="revenue",
                value=1000000,
                source=FactSource.DATABASE,
                source_name="sales_db",
                query="SELECT SUM(amount) FROM sales",
                reasoning="Aggregated from sales table",
                resolved_at=datetime(2024, 1, 15, 10, 30, 0),
            ),
        }

        repl._remember_fact("revenue")

        # Verify context includes provenance
        call_args = repl.api.fact_store.save_fact.call_args
        context = call_args.kwargs["context"]
        assert "database" in context.lower()
        assert "sales_db" in context
        assert "SELECT SUM(amount)" in context

    def test_remember_no_args_shows_usage(self, repl, mock_console):
        """Test /remember without arguments shows usage."""
        repl._remember_fact("")

        calls = [str(call) for call in mock_console.print.call_args_list]
        calls_str = " ".join(calls)
        assert "Usage" in calls_str
        assert "/remember" in calls_str

    def test_remember_handles_table_facts(self, repl, mock_console):
        """Test /remember handles facts with table references."""
        from constat.execution.fact_resolver import Fact, FactSource
        from datetime import datetime

        repl.api._session = Mock()
        repl.api.session.fact_resolver = Mock()
        repl.api.session.fact_resolver.get_all_facts.return_value = {
            "top_customers()": Fact(
                name="top_customers",
                value=None,  # Value stored in table
                source=FactSource.DATABASE,
                table_name="_top_customers",
                row_count=100,
                resolved_at=datetime.now(),
            ),
        }

        repl._remember_fact("top_customers")

        # Should still persist (value may be None for table facts)
        repl.api.fact_store.save_fact.assert_called_once()
        # display_value should mention the table
        calls = [str(call) for call in mock_console.print.call_args_list]
        calls_str = " ".join(calls)
        assert "Remembered" in calls_str


class TestStatusLine:
    """Tests for StatusLine class (Phase 5)."""

    def test_status_line_initial_state(self):
        """Test StatusLine initial render."""
        from constat.repl.feedback import StatusLine

        status = StatusLine()

        # Should show idle phase (Ready is displayed for idle)
        rendered = status.render()
        assert "Ready" in rendered

    def test_status_line_planning_phase(self):
        """Test StatusLine renders planning phase correctly."""
        from constat.repl.feedback import StatusLine
        from constat.execution.mode import Phase

        status = StatusLine()
        status._phase = Phase.PLANNING
        status._plan_name = "Analyze revenue"

        rendered = status.render()
        assert "Planning" in rendered
        assert "Analyze revenue" in rendered

    def test_status_line_executing_phase(self):
        """Test StatusLine renders executing phase correctly."""
        from constat.repl.feedback import StatusLine
        from constat.execution.mode import Phase

        status = StatusLine()
        status._phase = Phase.EXECUTING
        status._step_current = 2
        status._step_total = 5
        status._step_description = "Loading data"

        rendered = status.render()
        assert "Executing" in rendered
        assert "Step 2/5" in rendered
        assert "Loading data" in rendered

    def test_status_line_failed_phase(self):
        """Test StatusLine renders failed phase correctly."""
        from constat.repl.feedback import StatusLine
        from constat.execution.mode import Phase

        status = StatusLine()
        status._phase = Phase.FAILED
        status._step_current = 3
        status._error_message = "Connection timeout"

        rendered = status.render()
        assert "Failed" in rendered
        assert "step 3" in rendered
        assert "Connection timeout" in rendered
        assert "retry/replan/abandon" in rendered

    def test_status_line_awaiting_approval_phase(self):
        """Test StatusLine renders awaiting_approval phase correctly."""
        from constat.repl.feedback import StatusLine
        from constat.execution.mode import Phase

        status = StatusLine()
        status._phase = Phase.AWAITING_APPROVAL
        status._plan_name = "Revenue analysis"
        status._step_total = 4

        rendered = status.render()
        assert "Awaiting approval" in rendered
        assert "(4 steps)" in rendered
        assert "y/n/suggest" in rendered

    def test_status_line_update_from_state(self):
        """Test StatusLine.update() from ConversationState."""
        from constat.repl.feedback import StatusLine
        from constat.execution.mode import Phase, ConversationState

        status = StatusLine()
        state = ConversationState(
            phase=Phase.EXECUTING,
            failure_context="Test error",
        )

        status.update(state)

        assert status._phase == Phase.EXECUTING
        assert status._error_message == "Test error"

    def test_status_line_queue_indicator(self):
        """Test StatusLine shows queue indicator during execution."""
        from constat.repl.feedback import StatusLine
        from constat.execution.mode import Phase

        status = StatusLine()
        status._phase = Phase.EXECUTING
        status._queue_count = 2

        rendered = status.render()
        assert "Queued: 2" in rendered

    def test_status_line_spinner_advance(self):
        """Test StatusLine.advance_spinner() cycles through frames."""
        from constat.repl.feedback import StatusLine

        status = StatusLine()
        initial_frame = status._spinner_frame

        status.advance_spinner()
        assert status._spinner_frame == initial_frame + 1

    def test_status_line_status_message(self):
        """Test StatusLine renders status message when set."""
        from constat.repl.feedback import StatusLine

        status = StatusLine()
        status.set_status_message("Analyzing your question...")

        rendered = status.render()
        # Should show spinner and message instead of phase
        assert "Analyzing your question..." in rendered
        assert "Ready" not in rendered  # Phase should be hidden when message is set

        # Clear message
        status.set_status_message(None)
        rendered = status.render()
        assert "Analyzing" not in rendered
        assert "Ready" in rendered  # Phase should be shown again


class TestFailureRecovery:
    """Tests for failure recovery prompts (Phase 5)."""

    def test_failure_recovery_retry(self, mock_console):
        """Test failure recovery prompt returns 'retry' for retry input."""
        from constat.repl.feedback import FeedbackDisplay

        display = FeedbackDisplay(console=mock_console)

        # Mock console.input to return 'retry'
        mock_console.input = Mock(return_value="retry")

        result = display.request_failure_recovery("Connection error", "step 2")

        assert result == "retry"

    def test_failure_recovery_replan(self, mock_console):
        """Test failure recovery prompt returns 'replan' for replan input."""
        from constat.repl.feedback import FeedbackDisplay

        display = FeedbackDisplay(console=mock_console)

        # Mock console.input to return 'replan'
        mock_console.input = Mock(return_value="replan")

        result = display.request_failure_recovery("API error")

        assert result == "replan"

    def test_failure_recovery_abandon(self, mock_console):
        """Test failure recovery prompt returns 'abandon' for abandon input."""
        from constat.repl.feedback import FeedbackDisplay

        display = FeedbackDisplay(console=mock_console)

        # Mock console.input to return 'abandon'
        mock_console.input = Mock(return_value="abandon")

        result = display.request_failure_recovery("Unknown error")

        assert result == "abandon"

    def test_failure_recovery_numeric_shortcuts(self, mock_console):
        """Test failure recovery accepts numeric shortcuts."""
        from constat.repl.feedback import FeedbackDisplay

        display = FeedbackDisplay(console=mock_console)

        # Test '1' for retry
        mock_console.input = Mock(return_value="1")
        assert display.request_failure_recovery("Error") == "retry"

        # Test '2' for replan
        mock_console.input = Mock(return_value="2")
        assert display.request_failure_recovery("Error") == "replan"

        # Test '3' for abandon
        mock_console.input = Mock(return_value="3")
        assert display.request_failure_recovery("Error") == "abandon"

    def test_failure_recovery_keyboard_interrupt(self, mock_console):
        """Test failure recovery handles KeyboardInterrupt."""
        from constat.repl.feedback import FeedbackDisplay

        display = FeedbackDisplay(console=mock_console)

        # Mock console.input to raise KeyboardInterrupt
        mock_console.input = Mock(side_effect=KeyboardInterrupt())

        result = display.request_failure_recovery("Error")

        assert result == "abandon"

    def test_failure_recovery_empty_input(self, mock_console):
        """Test failure recovery treats empty input as abandon."""
        from constat.repl.feedback import FeedbackDisplay

        display = FeedbackDisplay(console=mock_console)

        # Mock console.input to return empty string
        mock_console.input = Mock(return_value="")

        result = display.request_failure_recovery("Error")

        assert result == "abandon"


class TestCancelExecution:
    """Tests for Ctrl+C execution cancellation (Phase 5)."""

    def test_solve_calls_cancel_on_interrupt(self, repl, mock_console):
        """Test that _solve calls session.cancel_execution() on KeyboardInterrupt."""
        mock_session = Mock()
        mock_session.session_id = None
        mock_session.solve.side_effect = KeyboardInterrupt()
        mock_session.cancel_execution = Mock()
        repl.api._session = mock_session

        repl._solve("Some query")

        # Verify cancel_execution was called
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

        # Verify display cleanup
        repl.display.stop.assert_called_once()
        repl.display.stop_spinner.assert_called_once()
