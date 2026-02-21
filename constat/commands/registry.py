# Copyright (c) 2025 Kenneth Stott
#
# Command registry - maps command names to handlers.

"""Command registry and execution."""

from __future__ import annotations

from typing import Callable, Optional, TYPE_CHECKING

from constat.commands.base import CommandContext, CommandResult, ErrorResult
from constat.prompts import load_yaml

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
    agent_command,
    agents_command,
    agent_create_command,
    agent_edit_command,
    agent_delete_command,
    agent_draft_command,
    skill_command,
    skills_command,
    skill_create_command,
    skill_edit_command,
    skill_delete_command,
    skill_deactivate_command,
    skill_draft_command,
    skill_download_command,
    prove_command,
)
from constat.commands.sources import (
    databases_command,
    apis_command,
    documents_command,
    files_command,
    discover_command,
)


# Command descriptions loaded from YAML for i18n support
_cmd_desc = load_yaml("help_strings.yaml")["command_descriptions"]

# Command registry: maps command name(s) to handler
# Each entry: (aliases, handler, description)
COMMANDS: list[tuple[tuple[str, ...], Callable[[CommandContext], CommandResult], str]] = [
    # Help
    (("/help", "/h"), help_command, _cmd_desc["help"]),
    # Data exploration
    (("/tables",), tables_command, _cmd_desc["tables"]),
    (("/show",), show_command, _cmd_desc["show"]),
    (("/query",), query_command, _cmd_desc["query"]),
    (("/code",), code_command, _cmd_desc["code"]),
    (("/artifacts",), artifacts_command, _cmd_desc["artifacts"]),
    (("/export",), export_command, _cmd_desc["export"]),
    (("/download-code",), download_code_command, _cmd_desc["download-code"]),
    # Session management
    (("/state", "/status"), state_command, _cmd_desc["state"]),
    (("/reset",), reset_command, _cmd_desc["reset"]),
    (("/facts",), facts_command, _cmd_desc["facts"]),
    (("/context",), context_command, _cmd_desc["context"]),
    (("/preferences",), preferences_command, _cmd_desc["preferences"]),
    # Data sources
    (("/databases",), databases_command, _cmd_desc["databases"]),
    (("/apis",), apis_command, _cmd_desc["apis"]),
    (("/documents", "/docs"), documents_command, _cmd_desc["documents"]),
    (("/files",), files_command, _cmd_desc["files"]),
    (("/discover",), discover_command, _cmd_desc["discover"]),
    # Learnings & Rules
    (("/learnings",), learnings_command, _cmd_desc["learnings"]),
    (("/rule",), rule_command, _cmd_desc["rule"]),
    (("/rule-edit",), rule_edit_command, _cmd_desc["rule-edit"]),
    (("/rule-delete",), rule_delete_command, _cmd_desc["rule-delete"]),
    (("/correct",), correct_command, _cmd_desc["correct"]),
    # Agents
    (("/agent",), agent_command, _cmd_desc["agent"]),
    (("/agents",), agents_command, _cmd_desc["agents"]),
    (("/agent-create",), agent_create_command, _cmd_desc["agent-create"]),
    (("/agent-edit",), agent_edit_command, _cmd_desc["agent-edit"]),
    (("/agent-delete",), agent_delete_command, _cmd_desc["agent-delete"]),
    (("/agent-draft",), agent_draft_command, _cmd_desc["agent-draft"]),
    # Skills
    (("/skill",), skill_command, _cmd_desc["skill"]),
    (("/skills",), skills_command, _cmd_desc["skills"]),
    (("/skill-create",), skill_create_command, _cmd_desc["skill-create"]),
    (("/skill-edit",), skill_edit_command, _cmd_desc["skill-edit"]),
    (("/skill-delete",), skill_delete_command, _cmd_desc["skill-delete"]),
    (("/skill-deactivate",), skill_deactivate_command, _cmd_desc["skill-deactivate"]),
    (("/skill-draft",), skill_draft_command, _cmd_desc["skill-draft"]),
    (("/skill-download",), skill_download_command, _cmd_desc["skill-download"]),
    # Proof/verification
    (("/prove",), prove_command, _cmd_desc["prove"]),
]

# Build lookup dict for fast access
_COMMAND_MAP: dict[str, Callable[[CommandContext], CommandResult]] = {}
for aliases, cmd_handler, _ in COMMANDS:
    for alias in aliases:
        _COMMAND_MAP[alias] = cmd_handler


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
        return "", text

    parts = text.split(maxsplit=1)
    command = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    return command, args
