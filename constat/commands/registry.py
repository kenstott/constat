# Copyright (c) 2025 Kenneth Stott
#
# Command registry - maps command names to handlers.

"""Command registry and execution."""

from __future__ import annotations

from typing import Callable, Optional, TYPE_CHECKING

from constat.commands.base import CommandContext, CommandResult, ErrorResult

if TYPE_CHECKING:
    from constat.session import Session

# Import command handlers
from constat.commands.help import help_command
from constat.commands.data import (
    tables_command,
    show_command,
    query_command,
    code_command,
    artifacts_command,
    export_command,
    download_code_command,
)
from constat.commands.session_cmds import (
    state_command,
    reset_command,
    facts_command,
    context_command,
    preferences_command,
    learnings_command,
    rule_command,
    rule_edit_command,
    rule_delete_command,
    correct_command,
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
    prove_command,
)
from constat.commands.sources import (
    databases_command,
    apis_command,
    documents_command,
    files_command,
    discover_command,
)


# Command registry: maps command name(s) to handler
# Each entry: (aliases, handler, description)
COMMANDS: list[tuple[tuple[str, ...], Callable[[CommandContext], CommandResult], str]] = [
    # Help
    (("/help", "/h"), help_command, "Show help information"),
    # Data exploration
    (("/tables",), tables_command, "List available tables"),
    (("/show",), show_command, "Show table contents"),
    (("/query",), query_command, "Run SQL query"),
    (("/code",), code_command, "Show generated code"),
    (("/artifacts",), artifacts_command, "List artifacts"),
    (("/export",), export_command, "Export table to file"),
    (("/download-code",), download_code_command, "Download code as Python script"),
    # Session management
    (("/state", "/status"), state_command, "Show session state"),
    (("/reset",), reset_command, "Reset session"),
    (("/facts",), facts_command, "Show cached facts"),
    (("/context",), context_command, "Show context usage"),
    (("/preferences",), preferences_command, "Show preferences"),
    # Data sources
    (("/databases",), databases_command, "List databases"),
    (("/apis",), apis_command, "List APIs"),
    (("/documents", "/docs"), documents_command, "List documents"),
    (("/files",), files_command, "List data files"),
    (("/discover",), discover_command, "Semantic search across all data sources"),
    # Learnings & Rules
    (("/learnings",), learnings_command, "Show learnings and rules"),
    (("/rule",), rule_command, "Add a new rule"),
    (("/rule-edit",), rule_edit_command, "Edit an existing rule"),
    (("/rule-delete",), rule_delete_command, "Delete a rule"),
    (("/correct",), correct_command, "Record a correction"),
    # Roles
    (("/role",), role_command, "Set or show current role"),
    (("/roles",), roles_command, "List available roles"),
    (("/role-create",), role_create_command, "Create a new role"),
    (("/role-edit",), role_edit_command, "Edit a role's prompt"),
    (("/role-delete",), role_delete_command, "Delete a role"),
    (("/role-draft",), role_draft_command, "Draft a role using AI"),
    # Skills
    (("/skill",), skill_command, "Show or activate a skill"),
    (("/skills",), skills_command, "List available skills"),
    (("/skill-create",), skill_create_command, "Create a new skill"),
    (("/skill-edit",), skill_edit_command, "Edit a skill's content"),
    (("/skill-delete",), skill_delete_command, "Delete a skill"),
    (("/skill-deactivate",), skill_deactivate_command, "Deactivate a skill"),
    (("/skill-draft",), skill_draft_command, "Draft a skill using AI"),
    # Proof/verification
    (("/prove",), prove_command, "Verify conversation with auditable proof"),
]

# Build lookup dict for fast access
_COMMAND_MAP: dict[str, Callable[[CommandContext], CommandResult]] = {}
for aliases, handler, _ in COMMANDS:
    for alias in aliases:
        _COMMAND_MAP[alias] = handler


def get_command(name: str) -> Optional[Callable[[CommandContext], CommandResult]]:
    """Get command handler by name.

    Args:
        name: Command name (with or without leading /)

    Returns:
        Command handler function or None if not found
    """
    # Normalize name
    if not name.startswith("/"):
        name = "/" + name
    name = name.lower()

    return _COMMAND_MAP.get(name)


def execute_command(
    session: "Session",
    command: str,
    args: str = "",
) -> CommandResult:
    """Execute a command by name.

    Args:
        session: The session context
        command: Command name (e.g., "/tables" or "tables")
        args: Command arguments

    Returns:
        CommandResult from the handler
    """
    handler = get_command(command)

    if handler is None:
        return ErrorResult(
            error=f"Unknown command: {command}",
            details="Type /help for available commands.",
        )

    ctx = CommandContext(session=session, args=args)

    try:
        return handler(ctx)
    except Exception as e:
        return ErrorResult(
            error=f"Command error: {e}",
            details=str(e),
        )


def is_command(text: str) -> bool:
    """Check if text starts with a command."""
    return text.strip().startswith("/")


def parse_command(text: str) -> tuple[str, str]:
    """Parse command text into command name and arguments.

    Args:
        text: Full command text (e.g., "/show users")

    Returns:
        Tuple of (command_name, arguments)
    """
    text = text.strip()
    if not text.startswith("/"):
        return ("", text)

    parts = text.split(maxsplit=1)
    command = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    return (command, args)
