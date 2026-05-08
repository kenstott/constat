# Copyright (c) 2025 Kenneth Stott
#
# Base types for command results.

"""Base types and interfaces for commands."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from constat.session import Session


@dataclass
class CommandContext:
    """Context passed to command handlers.

    Provides access to session state without tight coupling.
    """
    session: Session
    args: str = ""  # Command arguments (text after command name)

    @property
    def has_datastore(self) -> bool:
        return self.session.datastore is not None

    @property
    def has_plan(self) -> bool:
        return self.session.plan is not None


@dataclass
class CommandResult:
    """Base result from a command."""
    success: bool = True
    message: Optional[str] = None  # Optional status message


@dataclass
class TextResult(CommandResult):
    """Result containing text/markdown content."""
    content: str = ""
    format: str = "markdown"  # "markdown", "plain", "code"


@dataclass
class TableResult(CommandResult):
    """Result containing tabular data."""
    title: Optional[str] = None
    columns: list[str] = field(default_factory=list)
    rows: list[list[Any]] = field(default_factory=list)
    footer: Optional[str] = None

    @property
    def is_empty(self) -> bool:
        return len(self.rows) == 0


@dataclass
class ListResult(CommandResult):
    """Result containing a list of items."""
    title: Optional[str] = None
    items: list[dict[str, Any]] = field(default_factory=list)
    empty_message: str = "No items found."

    @property
    def is_empty(self) -> bool:
        return len(self.items) == 0


@dataclass
class KeyValueResult(CommandResult):
    """Result containing key-value pairs."""
    title: Optional[str] = None
    pairs: dict[str, Any] = field(default_factory=dict)
    sections: list[tuple[str, dict[str, Any]]] = field(default_factory=list)  # Grouped pairs


@dataclass
class ErrorResult(CommandResult):
    """Result indicating an error."""
    success: bool = False
    error: str = ""
    details: Optional[str] = None


@dataclass
class HelpResult(CommandResult):
    """Result containing help information."""
    commands: list[tuple[str, str, str]] = field(default_factory=list)  # (cmd, desc, category)
    shortcuts: list[tuple[str, str]] = field(default_factory=list)  # (key, action)
    tips: list[str] = field(default_factory=list)


class CommandHandler(Protocol):
    """Protocol for command handlers."""

    def __call__(self, ctx: CommandContext) -> CommandResult:
        """Execute the command and return a result."""
        ...
