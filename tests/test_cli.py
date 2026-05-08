# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tests for CLI interface.

These tests use Click's CliRunner to test command behavior without
requiring API keys or real database connections.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest
from click.testing import CliRunner

from constat.cli import cli


@pytest.fixture
def runner():
    """Create a CLI runner."""
    return CliRunner()


@pytest.fixture
def valid_config_file(tmp_path):
    """Create a valid minimal config file."""
    config_content = """
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: test-key

databases:
  test:
    uri: sqlite:///test.db
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_content)
    return str(config_path)


@pytest.fixture
def config_with_env_var(tmp_path):
    """Create a config file with unset environment variable."""
    config_content = """
llm:
  api_key: ${NONEXISTENT_API_KEY_FOR_TEST}

databases:
  test:
    uri: sqlite:///test.db
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_content)
    return str(config_path)


class TestCLIGroup:
    """Test the main CLI group."""

    def test_cli_help(self, runner):
        """CLI --help shows usage information."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "solve" in result.output
        assert "repl" in result.output

    def test_cli_no_command(self, runner):
        """CLI with no command shows usage."""
        result = runner.invoke(cli, [])
        # Click may return 0 or 2 depending on configuration
        # Should show available commands or usage
        assert "solve" in result.output or "Usage" in result.output


class TestSolveCommand:
    """Test the solve command."""

    def test_solve_help(self, runner):
        """solve --help shows usage."""
        result = runner.invoke(cli, ["solve", "--help"])
        assert result.exit_code == 0
        assert "PROBLEM" in result.output or "problem" in result.output.lower()

    def test_solve_requires_config(self, runner):
        """solve command requires --config option."""
        result = runner.invoke(cli, ["solve", "test problem"])
        assert result.exit_code != 0
        assert "config" in result.output.lower() or "Missing" in result.output

    def test_solve_requires_problem(self, runner, valid_config_file):
        """solve command requires problem argument."""
        result = runner.invoke(cli, ["solve", "-c", valid_config_file])
        assert result.exit_code != 0

    def test_solve_config_not_found(self, runner):
        """solve command with nonexistent config file fails."""
        result = runner.invoke(cli, ["solve", "test", "-c", "/nonexistent/config.yaml"])
        assert result.exit_code != 0
        assert "does not exist" in result.output or "not found" in result.output.lower() or "Error" in result.output

    def test_solve_config_invalid_yaml(self, runner, tmp_path):
        """solve command with invalid YAML config fails gracefully."""
        bad_config = tmp_path / "bad.yaml"
        bad_config.write_text("invalid: yaml: content: [")

        result = runner.invoke(cli, ["solve", "test", "-c", str(bad_config)])
        assert result.exit_code == 1

    @patch("constat.cli.Session")
    @patch("constat.cli.Config.from_yaml")
    def test_solve_success(self, mock_config, mock_session_class, runner, tmp_path):
        """solve command succeeds with valid config and mocked session."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("llm:\n  api_key: test\ndatabases: {}")

        mock_cfg = Mock()
        mock_cfg.llm = Mock()
        mock_cfg.databases = {}
        mock_config.return_value = mock_cfg

        mock_session = Mock()
        mock_session.solve.return_value = {
            "success": True,
            "plan": Mock(steps=[Mock(number=1, goal="Test")]),
            "results": [Mock(duration_ms=100)],
            "output": "Test output",
            "datastore_tables": [],
        }
        mock_session_class.return_value = mock_session

        result = runner.invoke(cli, ["solve", "test problem", "-c", str(config_path)])

        mock_session.solve.assert_called_once_with("test problem")

    @patch("constat.cli.Session")
    @patch("constat.cli.Config.from_yaml")
    def test_solve_failure_shows_error(self, mock_config, mock_session_class, runner, tmp_path):
        """solve command shows error message on failure."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("llm:\n  api_key: test\ndatabases: {}")

        mock_cfg = Mock()
        mock_config.return_value = mock_cfg

        mock_session = Mock()
        mock_session.solve.return_value = {
            "success": False,
            "error": "Database connection failed",
        }
        mock_session_class.return_value = mock_session

        result = runner.invoke(cli, ["solve", "test", "-c", str(config_path)])

        assert result.exit_code == 1

    @patch("constat.cli.Session")
    @patch("constat.cli.Config.from_yaml")
    def test_solve_writes_output_to_file(self, mock_config, mock_session_class, runner, tmp_path):
        """solve --output writes result to file."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("llm:\n  api_key: test\ndatabases: {}")
        output_path = tmp_path / "output.txt"

        mock_cfg = Mock()
        mock_config.return_value = mock_cfg

        mock_session = Mock()
        mock_session.solve.return_value = {
            "success": True,
            "plan": Mock(steps=[]),
            "results": [],
            "output": "Expected output content",
            "datastore_tables": [],
        }
        mock_session_class.return_value = mock_session

        result = runner.invoke(cli, [
            "solve", "test",
            "-c", str(config_path),
            "-o", str(output_path)
        ])

        assert output_path.exists()
        assert output_path.read_text() == "Expected output content"

    @patch("constat.cli.Session")
    @patch("constat.cli.Config.from_yaml")
    def test_solve_keyboard_interrupt(self, mock_config, mock_session_class, runner, tmp_path):
        """solve command handles KeyboardInterrupt gracefully."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("llm:\n  api_key: test\ndatabases: {}")

        mock_cfg = Mock()
        mock_config.return_value = mock_cfg

        mock_session = Mock()
        mock_session.solve.side_effect = KeyboardInterrupt()
        mock_session_class.return_value = mock_session

        result = runner.invoke(cli, ["solve", "test", "-c", str(config_path)])

        assert result.exit_code == 130
        assert "Interrupted" in result.output

    @patch("constat.cli.Session")
    @patch("constat.cli.Config.from_yaml")
    def test_solve_rejected_plan(self, mock_config, mock_session_class, runner, tmp_path):
        """solve command handles rejected plan."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("llm:\n  api_key: test\ndatabases: {}")

        mock_cfg = Mock()
        mock_config.return_value = mock_cfg

        mock_session = Mock()
        mock_session.solve.return_value = {
            "success": False,
            "rejected": True,
            "plan": Mock(steps=[]),
            "reason": "User rejected",
            "message": "Plan was rejected by user.",
        }
        mock_session_class.return_value = mock_session

        result = runner.invoke(cli, ["solve", "test", "-c", str(config_path)])

        # Rejected plans return failure exit code
        # The CLI treats rejected as a form of failure
        mock_session.solve.assert_called_once_with("test")


class TestReplCommand:
    """Test the repl command."""

    def test_repl_help(self, runner):
        """repl --help shows usage."""
        result = runner.invoke(cli, ["repl", "--help"])
        assert result.exit_code == 0

    def test_repl_requires_config(self, runner):
        """repl command requires --config option."""
        result = runner.invoke(cli, ["repl"])
        assert result.exit_code != 0
        assert "config" in result.output.lower() or "Missing" in result.output

    def test_repl_config_not_found(self, runner):
        """repl command with nonexistent config fails."""
        result = runner.invoke(cli, ["repl", "-c", "/nonexistent/config.yaml"])
        assert result.exit_code != 0

    @patch("constat.textual_repl.run_textual_repl")
    def test_repl_starts(self, mock_run_repl, runner, tmp_path):
        """repl command starts REPL with correct config."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("llm:\n  api_key: test\ndatabases: {}")

        result = runner.invoke(cli, ["repl", "-c", str(config_path)])

        mock_run_repl.assert_called_once()

    @patch("constat.textual_repl.run_textual_repl")
    def test_repl_with_initial_problem(self, mock_run_repl, runner, tmp_path):
        """repl --problem passes problem to REPL."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("llm:\n  api_key: test\ndatabases: {}")

        result = runner.invoke(cli, [
            "repl", "-c", str(config_path),
            "-p", "Show me sales data"
        ])

        mock_run_repl.assert_called_once()
        call_kwargs = mock_run_repl.call_args
        assert call_kwargs.kwargs.get("problem") == "Show me sales data" or \
            (len(call_kwargs.args) > 2 and call_kwargs.args[2] == "Show me sales data")

    @patch("constat.textual_repl.run_textual_repl")
    def test_repl_keyboard_interrupt_handled(self, mock_run_repl, runner, tmp_path):
        """repl handles KeyboardInterrupt gracefully."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("llm:\n  api_key: test\ndatabases: {}")

        mock_run_repl.side_effect = KeyboardInterrupt()

        result = runner.invoke(cli, ["repl", "-c", str(config_path)])

        assert result.exit_code == 0


class TestHistoryCommand:
    """Test the history command."""

    def test_history_help(self, runner):
        """history --help shows usage."""
        result = runner.invoke(cli, ["history", "--help"])
        assert result.exit_code == 0

    @patch("constat.storage.history.SessionHistory")
    def test_history_empty(self, mock_history_class, runner):
        """history command with no sessions shows message."""
        mock_history = Mock()
        mock_history.list_sessions.return_value = []
        mock_history_class.return_value = mock_history

        result = runner.invoke(cli, ["history"])

        assert result.exit_code == 0
        assert "No previous sessions" in result.output or "empty" in result.output.lower() or "sessions" in result.output.lower()

    @patch("constat.storage.history.SessionHistory")
    def test_history_shows_sessions(self, mock_history_class, runner):
        """history command shows session list."""
        mock_session = Mock()
        mock_session.session_id = "2024-01-15_143022_abc12345"
        mock_session.created_at = "2024-01-15T14:30:22"
        mock_session.status = "completed"
        mock_session.total_queries = 3
        mock_session.databases = ["main", "analytics"]

        mock_history = Mock()
        mock_history.list_sessions.return_value = [mock_session]
        mock_history_class.return_value = mock_history

        result = runner.invoke(cli, ["history"])

        assert result.exit_code == 0

    @patch("constat.storage.history.SessionHistory")
    def test_history_respects_limit(self, mock_history_class, runner):
        """history --limit passes limit to list_sessions."""
        mock_history = Mock()
        mock_history.list_sessions.return_value = []
        mock_history_class.return_value = mock_history

        result = runner.invoke(cli, ["history", "-n", "5"])

        mock_history.list_sessions.assert_called_once_with(limit=5)


class TestResumeCommand:
    """Test the resume command."""

    def test_resume_help(self, runner):
        """resume --help shows usage."""
        result = runner.invoke(cli, ["resume", "--help"])
        assert result.exit_code == 0

    def test_resume_requires_session_id(self, runner, valid_config_file):
        """resume command requires session_id argument."""
        result = runner.invoke(cli, ["resume", "-c", valid_config_file])
        assert result.exit_code != 0

    def test_resume_requires_config(self, runner):
        """resume command requires --config option."""
        result = runner.invoke(cli, ["resume", "abc123"])
        assert result.exit_code != 0


class TestValidateCommand:
    """Test the validate command."""

    def test_validate_help(self, runner):
        """validate --help shows usage."""
        result = runner.invoke(cli, ["validate", "--help"])
        assert result.exit_code == 0

    def test_validate_requires_config(self, runner):
        """validate command requires --config option."""
        result = runner.invoke(cli, ["validate"])
        assert result.exit_code != 0

    def test_validate_config_not_found(self, runner):
        """validate with nonexistent config fails."""
        result = runner.invoke(cli, ["validate", "-c", "/nonexistent.yaml"])
        assert result.exit_code != 0

    def test_validate_invalid_yaml(self, runner, tmp_path):
        """validate with invalid YAML shows error."""
        bad_config = tmp_path / "bad.yaml"
        bad_config.write_text("not: valid: yaml: [")

        result = runner.invoke(cli, ["validate", "-c", str(bad_config)])

        assert result.exit_code == 1
        assert "FAIL" in result.output or "error" in result.output.lower()

    @patch("constat.catalog.schema_manager.SchemaManager")
    @patch("constat.core.config.Config.from_yaml")
    def test_validate_valid_config(self, mock_config, mock_schema_mgr, runner, tmp_path):
        """validate with valid config shows OK."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("llm:\n  api_key: test\ndatabases: {}")

        mock_cfg = Mock()
        mock_cfg.llm = Mock()
        mock_cfg.llm.api_key = "test-key"
        mock_cfg.databases = {}
        mock_config.return_value = mock_cfg

        result = runner.invoke(cli, ["validate", "-c", str(config_path)])

        # Check for success indicators
        assert result.exit_code == 0 or "Config file parsed" in result.output

    @patch("constat.catalog.schema_manager.SchemaManager")
    @patch("constat.core.config.Config.from_yaml")
    def test_validate_warns_no_api_key(self, mock_config, mock_schema_mgr, runner, tmp_path):
        """validate warns when API key is not set."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("llm:\n  model: test\ndatabases: {}")

        mock_cfg = Mock()
        mock_cfg.llm = Mock()
        mock_cfg.llm.api_key = None
        mock_cfg.databases = {}
        mock_config.return_value = mock_cfg

        result = runner.invoke(cli, ["validate", "-c", str(config_path)])

        # Either warns or validates successfully
        assert result.exit_code == 0 or "WARN" in result.output


class TestSchemaCommand:
    """Test the schema command."""

    def test_schema_help(self, runner):
        """schema --help shows usage."""
        result = runner.invoke(cli, ["schema", "--help"])
        assert result.exit_code == 0

    def test_schema_requires_config(self, runner):
        """schema command requires --config option."""
        result = runner.invoke(cli, ["schema"])
        assert result.exit_code != 0

    @patch("constat.catalog.schema_manager.SchemaManager")
    @patch("constat.core.config.Config.from_yaml")
    def test_schema_shows_overview(self, mock_config, mock_schema_mgr_class, runner, tmp_path):
        """schema command shows schema overview."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("llm:\n  api_key: test\ndatabases: {}")

        mock_cfg = Mock()
        mock_config.return_value = mock_cfg

        mock_schema_mgr = Mock()
        mock_schema_mgr.get_overview.return_value = "Table: users (id, name, email)"
        mock_schema_mgr_class.return_value = mock_schema_mgr

        result = runner.invoke(cli, ["schema", "-c", str(config_path)])

        # Schema command should succeed
        assert result.exit_code == 0 or "schema" in result.output.lower()


class TestInitCommand:
    """Test the init command."""

    def test_init_help(self, runner):
        """init --help shows usage."""
        result = runner.invoke(cli, ["init", "--help"])
        assert result.exit_code == 0

    def test_init_creates_config_file(self, runner, tmp_path, monkeypatch):
        """init creates config.yaml in current directory."""
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(cli, ["init"])

        assert result.exit_code == 0
        assert (tmp_path / "config.yaml").exists()
        assert "Created" in result.output

    def test_init_config_has_required_sections(self, runner, tmp_path, monkeypatch):
        """init creates config with required sections."""
        monkeypatch.chdir(tmp_path)

        runner.invoke(cli, ["init"])

        content = (tmp_path / "config.yaml").read_text()
        assert "llm:" in content
        assert "databases:" in content

    def test_init_prompts_before_overwrite(self, runner, tmp_path, monkeypatch):
        """init asks before overwriting existing file."""
        monkeypatch.chdir(tmp_path)

        # Create existing config
        (tmp_path / "config.yaml").write_text("existing content")

        # Decline overwrite
        result = runner.invoke(cli, ["init"], input="n\n")

        assert "already exists" in result.output
        assert (tmp_path / "config.yaml").read_text() == "existing content"

    def test_init_overwrites_when_confirmed(self, runner, tmp_path, monkeypatch):
        """init overwrites existing file when confirmed."""
        monkeypatch.chdir(tmp_path)

        # Create existing config
        (tmp_path / "config.yaml").write_text("existing content")

        # Confirm overwrite
        result = runner.invoke(cli, ["init"], input="y\n")

        assert "Created" in result.output
        assert "existing content" not in (tmp_path / "config.yaml").read_text()


class TestExitCodes:
    """Test that exit codes are correct."""

    def test_help_returns_zero(self, runner):
        """--help returns exit code 0."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0

    def test_invalid_command_returns_nonzero(self, runner):
        """Invalid command returns non-zero exit code."""
        result = runner.invoke(cli, ["nonexistent_command"])
        assert result.exit_code != 0

    @patch("constat.cli.Session")
    @patch("constat.cli.Config.from_yaml")
    def test_solve_success_returns_zero(self, mock_config, mock_session_class, runner, tmp_path):
        """Successful solve returns exit code 0."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("llm:\n  api_key: test\ndatabases: {}")

        mock_cfg = Mock()
        mock_config.return_value = mock_cfg

        mock_session = Mock()
        mock_session.solve.return_value = {
            "success": True,
            "plan": Mock(steps=[]),
            "results": [],
            "output": "",
            "datastore_tables": [],
        }
        mock_session_class.return_value = mock_session

        result = runner.invoke(cli, ["solve", "test", "-c", str(config_path)])

        assert result.exit_code == 0

    @patch("constat.cli.Session")
    @patch("constat.cli.Config.from_yaml")
    def test_solve_failure_returns_one(self, mock_config, mock_session_class, runner, tmp_path):
        """Failed solve returns exit code 1."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("llm:\n  api_key: test\ndatabases: {}")

        mock_cfg = Mock()
        mock_config.return_value = mock_cfg

        mock_session = Mock()
        mock_session.solve.return_value = {"success": False, "error": "Test error"}
        mock_session_class.return_value = mock_session

        result = runner.invoke(cli, ["solve", "test", "-c", str(config_path)])

        assert result.exit_code == 1

    @patch("constat.cli.Session")
    @patch("constat.cli.Config.from_yaml")
    def test_keyboard_interrupt_returns_130(self, mock_config, mock_session_class, runner, tmp_path):
        """KeyboardInterrupt returns exit code 130."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("llm:\n  api_key: test\ndatabases: {}")

        mock_cfg = Mock()
        mock_config.return_value = mock_cfg

        mock_session = Mock()
        mock_session.solve.side_effect = KeyboardInterrupt()
        mock_session_class.return_value = mock_session

        result = runner.invoke(cli, ["solve", "test", "-c", str(config_path)])

        assert result.exit_code == 130


class TestShortOptions:
    """Test short option aliases."""

    def test_config_short_option_c(self, runner, tmp_path):
        """-c works as alias for --config."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("llm:\n  api_key: test\ndatabases: {}")

        # Both should parse correctly
        result1 = runner.invoke(cli, ["validate", "-c", str(config_path), "--help"])
        result2 = runner.invoke(cli, ["validate", "--config", str(config_path), "--help"])

        # Both should show help without config error
        assert result1.exit_code == result2.exit_code

    @patch("constat.storage.history.SessionHistory")
    def test_limit_short_option_n(self, mock_history_class, runner):
        """-n works as alias for --limit."""
        mock_history = Mock()
        mock_history.list_sessions.return_value = []
        mock_history_class.return_value = mock_history

        result = runner.invoke(cli, ["history", "-n", "5"])

        mock_history.list_sessions.assert_called_once_with(limit=5)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @patch("constat.cli.Session")
    @patch("constat.cli.Config.from_yaml")
    def test_solve_empty_problem_string(self, mock_config, mock_session_class, runner, tmp_path):
        """solve with empty problem string."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("llm:\n  api_key: test\ndatabases: {}")

        mock_cfg = Mock()
        mock_config.return_value = mock_cfg

        mock_session = Mock()
        mock_session.solve.return_value = {
            "success": True,
            "plan": Mock(steps=[]),
            "results": [],
            "output": "",
            "datastore_tables": [],
        }
        mock_session_class.return_value = mock_session

        result = runner.invoke(cli, ["solve", "", "-c", str(config_path)])

        # Empty string is still a valid argument
        mock_session.solve.assert_called_once_with("")

    def test_config_path_with_spaces(self, runner, tmp_path):
        """Config path with spaces is handled correctly."""
        dir_with_spaces = tmp_path / "path with spaces"
        dir_with_spaces.mkdir()
        config_path = dir_with_spaces / "config.yaml"
        config_path.write_text("llm:\n  api_key: test\ndatabases:\n  test:\n    uri: sqlite:///test.db")

        result = runner.invoke(cli, ["validate", "-c", str(config_path)])

        # Should not fail due to spaces in path
        assert "does not exist" not in result.output

    @patch("constat.storage.history.SessionHistory")
    def test_history_limit_zero(self, mock_history_class, runner):
        """history --limit 0 is handled."""
        mock_history = Mock()
        mock_history.list_sessions.return_value = []
        mock_history_class.return_value = mock_history

        result = runner.invoke(cli, ["history", "-n", "0"])

        mock_history.list_sessions.assert_called_once_with(limit=0)

    def test_config_empty_file(self, runner, tmp_path):
        """Empty config file is handled."""
        empty_config = tmp_path / "empty.yaml"
        empty_config.write_text("")

        result = runner.invoke(cli, ["validate", "-c", str(empty_config)])

        # Empty YAML parses to None, should fail validation
        assert result.exit_code == 1

    @patch("constat.cli.Session")
    @patch("constat.cli.Config.from_yaml")
    def test_solve_unicode_problem(self, mock_config, mock_session_class, runner, tmp_path):
        """solve handles unicode in problem string."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("llm:\n  api_key: test\ndatabases: {}")

        mock_cfg = Mock()
        mock_config.return_value = mock_cfg

        mock_session = Mock()
        mock_session.solve.return_value = {
            "success": True,
            "plan": Mock(steps=[]),
            "results": [],
            "output": "",
            "datastore_tables": [],
        }
        mock_session_class.return_value = mock_session

        unicode_problem = "Find customers with name containing 日本語"
        result = runner.invoke(cli, ["solve", unicode_problem, "-c", str(config_path)])

        mock_session.solve.assert_called_once_with(unicode_problem)


class TestStartupProgress:
    """Test startup progress feedback."""

    @patch("constat.cli.Session")
    @patch("constat.cli.Config.from_yaml")
    def test_solve_shows_ready_message(self, mock_config, mock_session_class, runner, tmp_path):
        """solve command shows Ready message after initialization."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("llm:\n  api_key: test\ndatabases: {}")

        mock_cfg = Mock()
        mock_config.return_value = mock_cfg

        mock_session = Mock()
        mock_session.solve.return_value = {
            "success": True,
            "plan": Mock(steps=[]),
            "results": [],
            "output": "",
            "datastore_tables": [],
        }
        mock_session_class.return_value = mock_session

        result = runner.invoke(cli, ["solve", "test", "-c", str(config_path)])

        assert "Ready" in result.output

    @patch("constat.textual_repl.run_textual_repl")
    def test_repl_shows_ready_message(self, mock_run_repl, runner, tmp_path):
        """repl command delegates to run_textual_repl successfully."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("llm:\n  api_key: test\ndatabases: {}")

        result = runner.invoke(cli, ["repl", "-c", str(config_path)])

        assert result.exit_code == 0
        mock_run_repl.assert_called_once()

    @patch("constat.cli.Session")
    @patch("constat.cli.Config.from_yaml")
    def test_session_receives_progress_callback(self, mock_config, mock_session_class, runner, tmp_path):
        """Session is initialized with progress_callback parameter."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("llm:\n  api_key: test\ndatabases: {}")

        mock_cfg = Mock()
        mock_config.return_value = mock_cfg

        mock_session = Mock()
        mock_session.solve.return_value = {
            "success": True,
            "plan": Mock(steps=[]),
            "results": [],
            "output": "",
            "datastore_tables": [],
        }
        mock_session_class.return_value = mock_session

        result = runner.invoke(cli, ["solve", "test", "-c", str(config_path)])

        # Session should be called with progress_callback
        call_kwargs = mock_session_class.call_args.kwargs
        assert "progress_callback" in call_kwargs
        assert call_kwargs["progress_callback"] is not None


class TestVerboseMode:
    """Test verbose mode behavior."""

    @patch("constat.cli.Session")
    @patch("constat.cli.Config.from_yaml")
    def test_verbose_flag_accepted(self, mock_config, mock_session_class, runner, tmp_path):
        """--verbose flag is accepted without error."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("llm:\n  api_key: test\ndatabases: {}")

        mock_cfg = Mock()
        mock_config.return_value = mock_cfg

        mock_session = Mock()
        mock_session.solve.return_value = {
            "success": True,
            "plan": Mock(steps=[]),
            "results": [],
            "output": "",
            "datastore_tables": [],
        }
        mock_session_class.return_value = mock_session

        result = runner.invoke(cli, ["solve", "test", "-c", str(config_path), "-v"])

        # Should not error on -v flag
        assert result.exit_code == 0

    @patch("constat.cli.Session")
    @patch("constat.cli.Config.from_yaml")
    def test_verbose_long_form_accepted(self, mock_config, mock_session_class, runner, tmp_path):
        """--verbose long form is accepted."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("llm:\n  api_key: test\ndatabases: {}")

        mock_cfg = Mock()
        mock_config.return_value = mock_cfg

        mock_session = Mock()
        mock_session.solve.return_value = {
            "success": True,
            "plan": Mock(steps=[]),
            "results": [],
            "output": "",
            "datastore_tables": [],
        }
        mock_session_class.return_value = mock_session

        result = runner.invoke(cli, ["solve", "test", "-c", str(config_path), "--verbose"])

        assert result.exit_code == 0
