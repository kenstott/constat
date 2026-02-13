# Copyright (c) 2025 Kenneth Stott
#
# Help command - centralized command documentation.

"""Help command and command documentation."""

from constat.commands.base import CommandContext, HelpResult
from constat.prompts import load_yaml

# Help content loaded from YAML for i18n support
_help = load_yaml("help_strings.yaml")

# noinspection PyTypeChecker
HELP_COMMANDS: list[tuple[str, str, str]] = [tuple(cmd) for cmd in _help["help_commands"]]
# noinspection PyTypeChecker
KEYBOARD_SHORTCUTS: list[tuple[str, str]] = [tuple(s) for s in _help["keyboard_shortcuts"]]
HELP_TIPS: list[str] = _help["help_tips"]


def help_command(_ctx: CommandContext) -> HelpResult:
    """Return help information."""
    return HelpResult(
        success=True,
        commands=HELP_COMMANDS,
        shortcuts=KEYBOARD_SHORTCUTS,
        tips=HELP_TIPS,
    )


def get_help_markdown() -> str:
    """Generate help text as markdown for UI display."""
    categories: dict[str, list[tuple[str, str]]] = {}
    for cmd, desc, cat in HELP_COMMANDS:
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((cmd, desc))

    lines = ["**Available Commands:**", ""]
    for category, commands in categories.items():
        lines.append(f"**{category}:**")
        lines.append("| Command | Description |")
        lines.append("|---------|-------------|")
        for cmd, desc in commands:
            lines.append(f"| `{cmd}` | {desc} |")
        lines.append("")

    lines.append("**Tips:**")
    for tip in HELP_TIPS:
        lines.append(f"- {tip}")

    return "\n".join(lines)
