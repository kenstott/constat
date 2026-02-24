# Copyright (c) 2025 Kenneth Stott
#
# Renderers for command results - convert to Rich or Markdown.

"""Renderers for command results."""

from __future__ import annotations

from typing import Any

from constat.commands.base import (
    CommandResult,
    TableResult,
    ListResult,
    TextResult,
    KeyValueResult,
    JsonResult,
    ErrorResult,
    HelpResult,
)


def render_markdown(result: CommandResult) -> str:
    """Render a command result as Markdown.

    Args:
        result: Command result to render

    Returns:
        Markdown string
    """
    if isinstance(result, ErrorResult):
        return f"**Error:** {result.error}" + (f"\n\n{result.details}" if result.details else "")

    if isinstance(result, TextResult):
        return result.content

    if isinstance(result, TableResult):
        return _render_table_markdown(result)

    if isinstance(result, ListResult):
        return _render_list_markdown(result)

    if isinstance(result, KeyValueResult):
        return _render_keyvalue_markdown(result)

    if isinstance(result, JsonResult):
        return _render_json_markdown(result)

    if isinstance(result, HelpResult):
        return _render_help_markdown(result)

    # Fallback
    if result.message:
        return result.message
    return "Command completed."


def _render_table_markdown(result: TableResult) -> str:
    """Render TableResult as Markdown table."""
    lines = []

    if result.title:
        lines.append(f"**{result.title}**")
        lines.append("")

    if result.is_empty:
        lines.append(result.footer or "No data.")
        return "\n".join(lines)

    # Header
    lines.append("| " + " | ".join(str(c) for c in result.columns) + " |")
    lines.append("| " + " | ".join("---" for _ in result.columns) + " |")

    # Rows
    for row in result.rows:
        cells = [str(cell) if cell is not None else "" for cell in row]
        lines.append("| " + " | ".join(cells) + " |")

    if result.footer:
        lines.append("")
        lines.append(f"*{result.footer}*")

    return "\n".join(lines)


def _render_list_markdown(result: ListResult) -> str:
    """Render ListResult as Markdown list."""
    lines = []

    if result.title:
        lines.append(f"**{result.title}**")
        lines.append("")

    if result.is_empty:
        lines.append(result.empty_message)
        return "\n".join(lines)

    for item in result.items:
        # Render each item based on its structure
        if isinstance(item, dict):
            name = item.get("name", item.get("id", "Item"))
            lines.append(f"- **{name}**")
            for key, value in item.items():
                if key not in ("name", "id") and value:
                    lines.append(f"  - {key}: {value}")
        else:
            lines.append(f"- {item}")

    return "\n".join(lines)


def _render_keyvalue_markdown(result: KeyValueResult) -> str:
    """Render KeyValueResult as Markdown."""
    lines = []

    if result.title:
        lines.append(f"**{result.title}**")
        lines.append("")

    # Main pairs
    for key, value in result.pairs.items():
        lines.append(f"- **{key}:** {value}")

    # Sections
    for section_name, section_pairs in result.sections:
        lines.append("")
        lines.append(f"**{section_name}:**")
        for key, value in section_pairs.items():
            lines.append(f"- {key}: {value}")

    return "\n".join(lines)


def _render_json_markdown(result: JsonResult) -> str:
    """Render JsonResult as Markdown code block."""
    import json
    lines = []
    if result.title:
        lines.append(f"**{result.title}**")
        lines.append("")
    lines.append("```json")
    lines.append(json.dumps(result.data, indent=2, default=str))
    lines.append("```")
    if result.footer:
        lines.append("")
        lines.append(f"*{result.footer}*")
    return "\n".join(lines)


def _render_help_markdown(result: HelpResult) -> str:
    """Render HelpResult as Markdown."""
    # Group commands by category
    categories: dict[str, list[tuple[str, str]]] = {}
    for cmd, desc, cat in result.commands:
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((cmd, desc))

    # noinspection DuplicatedCode
    lines = ["**Available Commands:**", ""]

    for category, commands in categories.items():
        lines.append(f"**{category}:**")
        lines.append("| Command | Description |")
        lines.append("|---------|-------------|")
        for cmd, desc in commands:
            lines.append(f"| `{cmd}` | {desc} |")
        lines.append("")

    if result.tips:
        lines.append("**Tips:**")
        for tip in result.tips:
            lines.append(f"- {tip}")

    return "\n".join(lines)


# Rich rendering (for REPL)
def render_rich(result: CommandResult) -> Any:
    """Render a command result as Rich objects.

    Args:
        result: Command result to render

    Returns:
        Rich renderable (Table, Text, etc.) or list of renderables
    """
    try:
        from rich.table import Table
        from rich.text import Text
        from rich.panel import Panel
    except ImportError:
        # Fallback to markdown if Rich not available
        return render_markdown(result)

    if isinstance(result, ErrorResult):
        return Text(f"Error: {result.error}", style="red")

    if isinstance(result, TextResult):
        return Text(result.content)

    if isinstance(result, TableResult):
        return _render_table_rich(result)

    if isinstance(result, ListResult):
        return _render_list_rich(result)

    if isinstance(result, KeyValueResult):
        return _render_keyvalue_rich(result)

    if isinstance(result, JsonResult):
        return _render_json_rich(result)

    if isinstance(result, HelpResult):
        return _render_help_rich(result)

    # Fallback
    if result.message:
        return Text(result.message)
    return Text("Command completed.", style="dim")


def _render_table_rich(result: TableResult) -> Any:
    """Render TableResult as Rich Table."""
    from rich.table import Table
    from rich.text import Text

    if result.is_empty:
        return Text(result.footer or "No data.", style="dim")

    table = Table(title=result.title, show_header=True, box=None)

    for col in result.columns:
        table.add_column(str(col))

    for row in result.rows:
        table.add_row(*[str(cell) if cell is not None else "" for cell in row])

    return table


def _render_list_rich(result: ListResult) -> Any:
    """Render ListResult as Rich output."""
    from rich.text import Text
    from rich.console import Group

    if result.is_empty:
        return Text(result.empty_message, style="dim")

    lines = []
    if result.title:
        lines.append(Text(result.title, style="bold"))
        lines.append(Text(""))

    for item in result.items:
        if isinstance(item, dict):
            name = item.get("name", item.get("id", "Item"))
            lines.append(Text(f"  {name}", style="cyan"))
            for key, value in item.items():
                if key not in ("name", "id") and value:
                    lines.append(Text(f"    {key}: {value}", style="dim"))
        else:
            lines.append(Text(f"  - {item}"))

    return Group(*lines)


def _render_keyvalue_rich(result: KeyValueResult) -> Any:
    """Render KeyValueResult as Rich output."""
    from rich.table import Table

    table = Table(title=result.title, show_header=False, box=None)
    table.add_column("Key", style="cyan")
    table.add_column("Value")

    for key, value in result.pairs.items():
        table.add_row(key, str(value))

    return table


def _render_json_rich(result: JsonResult) -> Any:
    """Render JsonResult as Rich Syntax."""
    import json
    from rich.syntax import Syntax
    from rich.text import Text
    from rich.console import Group

    parts = []
    if result.title:
        parts.append(Text(result.title, style="bold"))
    json_str = json.dumps(result.data, indent=2, default=str)
    parts.append(Syntax(json_str, "json", theme="monokai", word_wrap=True))
    if result.footer:
        parts.append(Text(result.footer, style="dim"))
    return Group(*parts)


def _render_help_rich(result: HelpResult) -> Any:
    """Render HelpResult as Rich Tables."""
    from rich.table import Table
    from rich.console import Group

    # Commands table
    cmd_table = Table(title="Commands", show_header=True, box=None)
    cmd_table.add_column("Command", style="cyan")
    cmd_table.add_column("Description")

    for cmd, desc, _cat in result.commands:
        cmd_table.add_row(cmd, desc)

    tables = [cmd_table]

    # Shortcuts table if present
    if result.shortcuts:
        from rich.text import Text
        # noinspection PyTypeChecker
        tables.append(Text(""))

        shortcut_table = Table(title="Keyboard Shortcuts", show_header=True, box=None)
        shortcut_table.add_column("Key", style="cyan")
        shortcut_table.add_column("Action")

        for key, action in result.shortcuts:
            shortcut_table.add_row(key, action)

        tables.append(shortcut_table)

    return Group(*tables)
