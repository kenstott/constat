# Copyright (c) 2025 Kenneth Stott
#
# Tests for the commands module.

"""Tests for the shared command module."""

import pytest
from unittest.mock import Mock

from constat.commands import (
    HELP_COMMANDS,
    get_help_markdown,
    CommandContext,
    TableResult,
    ListResult,
    TextResult,
    KeyValueResult,
    HelpResult,
    ErrorResult,
    get_command,
    is_command,
    parse_command,
    render_markdown,
)
from constat.commands.help import help_command


class TestCommandRegistry:
    """Tests for command registry functions."""

    def test_is_command_with_slash(self):
        """Commands starting with / are recognized."""
        assert is_command("/help")
        assert is_command("/tables")
        assert is_command("  /show users")  # with leading whitespace

    def test_is_command_without_slash(self):
        """Text not starting with / is not a command."""
        assert not is_command("help")
        assert not is_command("what tables are there?")
        assert not is_command("")

    def test_parse_command_simple(self):
        """Parse simple commands."""
        assert parse_command("/help") == ("/help", "")
        assert parse_command("/tables") == ("/tables", "")

    def test_parse_command_with_args(self):
        """Parse commands with arguments."""
        assert parse_command("/show users") == ("/show", "users")
        assert parse_command("/query SELECT * FROM users") == ("/query", "SELECT * FROM users")

    def test_get_command_exists(self):
        """Get handler for existing command."""
        handler = get_command("/help")
        assert handler is not None
        handler = get_command("/tables")
        assert handler is not None

    def test_get_command_not_exists(self):
        """Get handler for non-existent command returns None."""
        assert get_command("/nonexistent") is None

    def test_get_command_normalizes(self):
        """Command lookup normalizes case and adds slash."""
        # Without slash
        assert get_command("help") is not None
        # Uppercase
        assert get_command("/HELP") is not None


class TestHelpCommands:
    """Tests for help command and HELP_COMMANDS."""

    def test_help_commands_not_empty(self):
        """HELP_COMMANDS should have entries."""
        assert len(HELP_COMMANDS) > 0

    def test_help_commands_structure(self):
        """Each entry in HELP_COMMANDS is (command, description, category)."""
        for entry in HELP_COMMANDS:
            assert len(entry) == 3
            cmd, desc, cat = entry
            assert cmd.startswith("/")
            assert len(desc) > 0
            assert len(cat) > 0

    def test_get_help_markdown(self):
        """get_help_markdown returns formatted markdown."""
        md = get_help_markdown()
        assert "**Available Commands:**" in md
        assert "/help" in md
        assert "Tips:" in md

    def test_help_command_returns_result(self):
        """help_command returns HelpResult."""
        mock_session = Mock()
        ctx = CommandContext(session=mock_session, args="")
        result = help_command(ctx)
        assert isinstance(result, HelpResult)
        assert result.success
        assert len(result.commands) > 0


class TestCommandResults:
    """Tests for command result types."""

    def test_table_result_is_empty(self):
        """TableResult.is_empty works correctly."""
        empty = TableResult(columns=["a", "b"], rows=[])
        assert empty.is_empty

        non_empty = TableResult(columns=["a", "b"], rows=[["1", "2"]])
        assert not non_empty.is_empty

    def test_list_result_is_empty(self):
        """ListResult.is_empty works correctly."""
        empty = ListResult(items=[])
        assert empty.is_empty

        non_empty = ListResult(items=[{"name": "test"}])
        assert not non_empty.is_empty

    def test_error_result_has_success_false(self):
        """ErrorResult has success=False by default."""
        error = ErrorResult(error="Something went wrong")
        assert not error.success


class TestRenderers:
    """Tests for command result renderers."""

    def test_render_table_markdown(self):
        """Render TableResult as markdown."""
        result = TableResult(
            title="Test Table",
            columns=["Name", "Value"],
            rows=[["foo", "1"], ["bar", "2"]],
        )
        md = render_markdown(result)
        assert "**Test Table**" in md
        assert "| Name | Value |" in md
        assert "| foo | 1 |" in md

    def test_render_list_markdown(self):
        """Render ListResult as markdown."""
        result = ListResult(
            title="Test List",
            items=[{"name": "item1", "desc": "description"}],
        )
        md = render_markdown(result)
        assert "**Test List**" in md
        assert "**item1**" in md

    def test_render_error_markdown(self):
        """Render ErrorResult as markdown."""
        result = ErrorResult(error="Test error", details="More info")
        md = render_markdown(result)
        assert "**Error:**" in md
        assert "Test error" in md
        assert "More info" in md

    def test_render_help_markdown(self):
        """Render HelpResult as markdown."""
        result = HelpResult(
            commands=[("/test", "Test command", "Testing")],
            tips=["Tip 1"],
        )
        md = render_markdown(result)
        assert "**Available Commands:**" in md
        assert "`/test`" in md
        assert "Tip 1" in md
