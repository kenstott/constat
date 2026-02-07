# Copyright (c) 2025 Kenneth Stott
#
# Command module - shared command handlers for REPL and UI.
#
# Commands return structured data that can be rendered differently
# by each interface (Rich tables for REPL, Markdown/JSON for UI).

"""Shared command handlers for REPL and UI."""

from constat.commands.base import (
    CommandResult,
    CommandContext,
    TableResult,
    ListResult,
    TextResult,
    KeyValueResult,
    HelpResult,
    ErrorResult,
)
from constat.commands.data import (
    tables_command,
    show_command,
    query_command,
    code_command,
    artifacts_command,
    export_command,
    download_code_command,
)
from constat.commands.help import (
    HELP_COMMANDS,
    help_command,
    get_help_markdown,
)
from constat.commands.registry import (
    COMMANDS,
    get_command,
    execute_command,
    is_command,
    parse_command,
)
from constat.commands.renderers import (
    render_markdown,
    render_rich,
)
from constat.commands.session_cmds import (
    state_command,
    reset_command,
    facts_command,
    context_command,
    preferences_command,
    role_command,
    roles_command,
    role_create_command,
    role_edit_command,
    role_delete_command,
    role_draft_command,
    skill_command,
    skills_command,
    skill_create_command,
    skill_edit_command,
    skill_delete_command,
    skill_deactivate_command,
    skill_draft_command,
)
from constat.commands.sources import (
    databases_command,
    apis_command,
    documents_command,
    files_command,
)

__all__ = [
    # Base types
    "CommandResult",
    "CommandContext",
    "TableResult",
    "ListResult",
    "TextResult",
    "KeyValueResult",
    "HelpResult",
    "ErrorResult",
    # Registry
    "COMMANDS",
    "get_command",
    "execute_command",
    "is_command",
    "parse_command",
    # Help
    "HELP_COMMANDS",
    "help_command",
    "get_help_markdown",
    # Data commands
    "tables_command",
    "show_command",
    "query_command",
    "code_command",
    "artifacts_command",
    "export_command",
    "download_code_command",
    # Session commands
    "state_command",
    "reset_command",
    "facts_command",
    "context_command",
    "preferences_command",
    # Role commands
    "role_command",
    "roles_command",
    "role_create_command",
    "role_edit_command",
    "role_delete_command",
    "role_draft_command",
    # Skill commands
    "skill_command",
    "skills_command",
    "skill_create_command",
    "skill_edit_command",
    "skill_delete_command",
    "skill_deactivate_command",
    "skill_draft_command",
    # Source commands
    "databases_command",
    "apis_command",
    "documents_command",
    "files_command",
    # Renderers
    "render_markdown",
    "render_rich",
]
