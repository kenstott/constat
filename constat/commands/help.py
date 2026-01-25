# Copyright (c) 2025 Kenneth Stott
#
# Help command - centralized command documentation.

"""Help command and command documentation."""

from constat.commands.base import CommandContext, HelpResult


# Centralized help commands - single source of truth for REPL and UI
# Each tuple: (command, description, category)
HELP_COMMANDS: list[tuple[str, str, str]] = [
    # Data Exploration
    ("/help, /h", "Show this help message", "Data Exploration"),
    ("/tables", "List available tables", "Data Exploration"),
    ("/show <table>", "Show table contents", "Data Exploration"),
    ("/export <table> [file]", "Export table to CSV or XLSX", "Data Exploration"),
    ("/query <sql>", "Run SQL query on datastore", "Data Exploration"),
    ("/code [step]", "Show generated code (all or specific step)", "Data Exploration"),
    ("/download-code [file]", "Download all code as Python script", "Data Exploration"),
    ("/artifacts [all]", "Show artifacts (use 'all' for intermediate)", "Data Exploration"),
    # Session Management
    ("/state", "Show session state", "Session"),
    ("/reset", "Clear session state and start fresh", "Session"),
    ("/redo [instruction]", "Retry last query (with optional modifications)", "Session"),
    ("/update, /refresh", "Refresh metadata and rebuild cache", "Session"),
    ("/context", "Show context size and token usage", "Session"),
    ("/compact", "Compact context to reduce token usage", "Session"),
    # Facts & Memory
    ("/facts", "Show cached facts from this session", "Facts"),
    ("/remember <fact>", "Persist a session fact", "Facts"),
    ("/forget <name>", "Forget a remembered fact", "Facts"),
    # Plans & History
    ("/save <name>", "Save current plan for replay", "Plans"),
    ("/share <name>", "Save plan as shared (all users)", "Plans"),
    ("/plans", "List saved plans", "Plans"),
    ("/replay <name>", "Replay a saved plan", "Plans"),
    ("/history, /sessions", "List recent sessions", "Plans"),
    ("/resume <id>", "Resume a previous session", "Plans"),
    # Data Sources
    ("/databases", "List configured databases", "Data Sources"),
    ("/database <uri> [name]", "Add a database to this session", "Data Sources"),
    ("/apis", "List configured APIs", "Data Sources"),
    ("/api <spec_url> [name]", "Add an API to this session", "Data Sources"),
    ("/documents, /docs", "List all documents", "Data Sources"),
    ("/files", "List all data files", "Data Sources"),
    ("/doc <path> [name] [desc]", "Add a document to this session", "Data Sources"),
    # Preferences
    ("/verbose [on|off]", "Toggle verbose mode", "Preferences"),
    ("/raw [on|off]", "Toggle raw output display", "Preferences"),
    ("/insights [on|off]", "Toggle insight synthesis", "Preferences"),
    ("/preferences", "Show current preferences", "Preferences"),
    ("/user [name]", "Show or set current user", "Preferences"),
    # Analysis
    ("/discover [scope] <query>", "Search data sources (database|api|document)", "Analysis"),
    ("/summarize <target>", "Summarize plan|session|facts|<table>", "Analysis"),
    ("/prove", "Verify conversation claims with auditable proof", "Analysis"),
    ("/correct <text>", "Record a correction for future reference", "Analysis"),
    ("/learnings", "Show learnings and rules", "Analysis"),
    ("/compact-learnings", "Promote similar learnings into rules", "Analysis"),
    # Exit
    ("/quit, /q", "Exit the session", "Exit"),
]

# Keyboard shortcuts (REPL-specific but included for completeness)
KEYBOARD_SHORTCUTS: list[tuple[str, str]] = [
    ("Ctrl+Left", "Shrink side panel"),
    ("Ctrl+Right", "Expand side panel"),
    ("Up/Down", "Navigate command history"),
    ("Ctrl+C / Esc", "Cancel current operation"),
    ("Ctrl+D", "Exit"),
]

HELP_TIPS: list[str] = [
    "Ask questions naturally to analyze your data",
    "Use numbered suggestions (1, 2, 3...) as shortcuts",
    'Use "why?" or "explain" to drill down into results',
]


def help_command(ctx: CommandContext) -> HelpResult:
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
