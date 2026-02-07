# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Interactive REPL for refinement loop."""

from pathlib import Path
from typing import Optional

# prompt_toolkit for input with status bar at bottom and auto-completion
from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.styles import Style as PTStyle
# Rich for output (tables, panels, syntax highlighting)
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table

from constat.api.impl import ConstatAPIImpl
from constat.core.config import Config
from constat.execution.mode import Mode
from constat.repl.feedback import FeedbackDisplay, SessionFeedbackHandler
from constat.session import Session, SessionConfig
from constat.storage.facts import FactStore
from constat.storage.learnings import LearningStore, LearningCategory, LearningSource
from constat.visualization.output import clear_pending_outputs, get_pending_outputs

# Vera's personality - imported from shared source
from constat.messages import RELIABLE_ADJECTIVES, HONEST_ADJECTIVES, get_vera_adjectives


# Commands available in the REPL
REPL_COMMANDS = [
    "/help", "/h", "/tables", "/show", "/query", "/code", "/state",
    "/update", "/refresh", "/reset", "/redo", "/user", "/save", "/share", "/sharewith",
    "/plans", "/replay", "/history", "/sessions", "/resume", "/restore",
    "/context", "/compact", "/facts", "/remember", "/forget",
    "/verbose", "/raw", "/insights", "/preferences", "/artifacts",
    "/database", "/databases", "/db", "/file", "/files",
    "/correct", "/learnings", "/compact-learnings", "/forget-learning",
    "/audit", "/summarize", "/prove",
    "/quit", "/exit", "/q"
]

# Use readline for tab completion (works reliably across terminals)
import os
import sys


# Tab completion uses prompt_toolkit's WordCompleter (set up in __init__)


class InteractiveREPL:
    """Interactive Read-Eval-Print Loop for Constat sessions."""

    def __init__(
        self,
        config: Config,
        verbose: bool = False,
        console: Optional[Console] = None,
        progress_callback: Optional[callable] = None,
        user_id: str = "default",
        auto_resume: bool = False,
    ):
        self.config = config
        self.verbose = verbose
        self.console = console or Console()
        self.display = FeedbackDisplay(console=self.console, verbose=verbose)
        self.progress_callback = progress_callback
        self.session_config = SessionConfig(verbose=verbose)
        self.user_id = user_id
        self.auto_resume = auto_resume
        self.last_problem = ""  # Track last problem for /save
        self.suggestions: list[str] = []  # Follow-up suggestions

        # Create the API (single interface for all operations)
        self.api: ConstatAPIImpl = self._create_api()
        # Note: auto-compact is called in run() after spinner stops

        # Style for prompt_toolkit status bar - dark background with light text
        self._prompt_style = PTStyle.from_dict({
            'bottom-toolbar': 'bg:#1a1a1a #888888',
            'bottom-toolbar.text': '#888888',
        })

    def _get_suggestion_context(self) -> dict:
        """Provide context for typeahead suggestions."""
        context = {"tables": [], "columns": [], "plans": []}

        # Get table names from datastore
        if self.api.session and self.api.session.datastore:
            tables = self.api.session.datastore.list_tables()
            context["tables"] = [t["name"] for t in tables]

        # Get saved plan names
        try:
            plans = Session.list_saved_plans(user_id=self.user_id)
            context["plans"] = [p["name"] for p in plans]
        except Exception:
            pass

        return context

    def _apply_display_overrides(self, overrides) -> dict:
        """Apply display overrides and return original values for restoration.

        Args:
            overrides: DisplayOverrides from detect_display_overrides()

        Returns:
            Dict of original values to restore after query (for single_turn only)
        """
        original = {}

        # Apply persistent changes first (these stick)
        for setting, value in overrides.persistent.items():
            if setting == "verbose":
                self.verbose = value
                self.display.verbose = value
                self.console.print(f"[dim]Verbose: {'on' if value else 'off'} (persistent)[/dim]")
            elif setting == "raw":
                self.session_config.show_raw_output = value
                self.console.print(f"[dim]Raw output: {'on' if value else 'off'} (persistent)[/dim]")
            elif setting == "insights":
                self.session_config.enable_insights = value
                self.console.print(f"[dim]Insights: {'on' if value else 'off'} (persistent)[/dim]")

        # Apply single-turn overrides (save originals for restoration)
        for setting, value in overrides.single_turn.items():
            # Skip if persistent already set the same value
            if setting in overrides.persistent:
                continue

            if setting == "verbose":
                original["verbose"] = self.verbose
                self.verbose = value
                self.display.verbose = value
            elif setting == "raw":
                original["raw"] = self.session_config.show_raw_output
                self.session_config.show_raw_output = value
            elif setting == "insights":
                original["insights"] = self.session_config.enable_insights
                self.session_config.enable_insights = value

        return original

    def _restore_display_settings(self, original: dict) -> None:
        """Restore display settings after single-turn override."""
        if "verbose" in original:
            self.verbose = original["verbose"]
            self.display.verbose = original["verbose"]
        if "raw" in original:
            self.session_config.show_raw_output = original["raw"]
        if "insights" in original:
            self.session_config.enable_insights = original["insights"]

    def _display_outputs(self) -> None:
        """Display any pending outputs from artifact saves."""
        outputs = get_pending_outputs()
        if not outputs:
            return

        self.console.print()
        self.console.print("[bold]Outputs:[/bold]")
        for output in outputs:
            file_uri = output["file_uri"]
            desc = output.get("description", "")
            file_type = output.get("type", "")
            type_hint = f" [dim]({file_type})[/dim]" if file_type else ""
            self.console.print(f"  [cyan]{desc}[/cyan]{type_hint}")
            self.console.print(f"    {file_uri}")

    def _create_api(self, new_session: bool = False) -> ConstatAPIImpl:
        """Create a new API instance with callbacks configured.

        Args:
            new_session: If True, always create a new session ID.
                        If False, reuse stored session ID if available.
        """
        from constat.storage.session_store import SessionStore
        session_store = SessionStore(user_id=self.user_id)
        if new_session:
            session_id = session_store.create_new()
        else:
            session_id = session_store.get_or_create()

        # Create session
        session = Session(
            self.config,
            session_id=session_id,
            session_config=self.session_config,
            progress_callback=self.progress_callback,
            user_id=self.user_id,
        )

        # Create stores
        fact_store = FactStore(user_id=self.user_id)
        learning_store = LearningStore(user_id=self.user_id)

        # Load persistent facts
        fact_store.load_into_session(session)

        # Create API
        api = ConstatAPIImpl(
            session=session,
            fact_store=fact_store,
            learning_store=learning_store,
        )

        # Wire up feedback handler
        handler = SessionFeedbackHandler(self.display, self.session_config)
        api.on_event(lambda event_type, data: handler.handle_event(
            type('Event', (), {'event_type': event_type, 'data': data})()
        ))

        # Wire up approval callback
        api.set_approval_callback(self.display.request_plan_approval)

        # Wire up clarification callback
        api.set_clarification_callback(self.display.request_clarification)

        # Initialize status line with default mode
        if self.session_config.default_mode:
            self.display.update_status_line(mode=self.session_config.default_mode)

        return api

    def _get_bottom_toolbar(self):
        """Get the status bar text for the bottom toolbar as HTML.

        Returns a two-line toolbar with:
        1. A horizontal rule
        2. The status bar with mode, status, and stats
        """
        import html as html_module
        import shutil

        status_bar = self.display._status_bar
        status_line = status_bar.status_line

        # Build status text based on current state
        mode = status_line._mode
        phase = status_line._phase
        status_msg = status_line._status_message

        # Mode badge
        if mode == Mode.PROOF:
            mode_html = '<style bg="ansiyellow" fg="ansiblack"><b> PROOF </b></style>'
        else:
            mode_html = '<style bg="ansicyan" fg="ansiblack"><b> EXPLORE </b></style>'

        # Phase/status text
        if status_msg:
            phase_text = html_module.escape(status_msg)
        elif phase.value == "idle":
            phase_text = 'ready'
        elif phase.value == "planning":
            plan = status_line._plan_name or ""
            if plan:
                plan = plan[:40] + "..." if len(plan) > 40 else plan
                phase_text = f'planning: {html_module.escape(plan)}'
            else:
                phase_text = 'planning...'
        elif phase.value == "executing":
            step = status_line._step_current
            total = status_line._step_total
            desc = status_line._step_description or ""
            if desc:
                desc = desc[:30] + "..." if len(desc) > 30 else desc
                phase_text = f'executing step {step}/{total}: {html_module.escape(desc)}'
            else:
                phase_text = f'executing step {step}/{total}'
        elif phase.value == "failed":
            err = status_line._error_message or "error"
            err = err[:40] + "..." if len(err) > 40 else err
            phase_text = f'failed: {html_module.escape(err)}'
        else:
            phase_text = phase.value

        # Stats - get current counts
        tables_count = status_bar._tables_count
        facts_count = status_bar._facts_count

        # Build full status bar with stats on right
        stats_html = f'<style fg="ansigray">tables:{tables_count} facts:{facts_count}</style>'

        # Get terminal width for rule
        terminal_width = shutil.get_terminal_size().columns
        rule_line = '─' * terminal_width

        # Return two-line toolbar: rule + status bar
        # Rule uses gray foreground, explicitly set dark background to match toolbar
        return HTML(f'<style fg="ansigray" bg="#333333">{rule_line}</style>\n{mode_html} {phase_text}  {stats_html}')

    def _get_completer(self) -> WordCompleter:
        """Build a completer with commands and dynamic context."""
        words = list(REPL_COMMANDS)

        # Add table names if session has datastore
        if self.api.session and self.api.session.datastore:
            try:
                tables = self.api.session.datastore.list_tables()
                for t in tables:
                    words.append(t["name"])
            except Exception:
                pass

        # Add saved plan names
        try:
            plans = Session.list_saved_plans(user_id=self.user_id)
            for p in plans:
                words.append(p["name"])
        except Exception:
            pass

        return WordCompleter(words, ignore_case=True)

    def _get_input(self) -> str:
        """Get user input with status bar at bottom.

        Uses Rich for output (header), prompt_toolkit for input with bottom toolbar.
        """
        # Print YOU header
        self.console.print()
        self.console.print(Rule("[bold green]YOU[/bold green]", align="right"))

        # prompt_toolkit input with auto-completion and status bar
        result = pt_prompt(
            "> ",
            completer=self._get_completer(),
            complete_while_typing=True,
            bottom_toolbar=self._get_bottom_toolbar,
            style=self._prompt_style,
        )
        return result.strip()

    def _show_help(self) -> None:
        """Show available commands."""
        table = Table(title="Commands", show_header=True, box=None)
        table.add_column("Command", style="cyan")
        table.add_column("Description")
        commands = [
            ("/help, /h", "Show this help message"),
            ("/tables", "List available tables"),
            ("/show <table>", "Show table contents"),
            ("/export <table> [file]", "Export table to CSV or XLSX"),
            ("/query <sql>", "Run SQL query on datastore"),
            ("/code [step]", "Show generated code (all or specific step)"),
            ("/state", "Show session state"),
            ("/update, /refresh", "Refresh metadata and rebuild preload cache"),
            ("/reset", "Clear session state and start fresh"),
            ("/redo [instruction]", "Retry last query (optionally with modifications)"),
            ("/user [name]", "Show or set current user"),
            ("/save <name>", "Save current plan for replay"),
            ("/share <name>", "Save plan as shared (all users)"),
            ("/sharewith <user> <name>", "Share plan with specific user"),
            ("/plans", "List saved plans"),
            ("/replay <name>", "Replay a saved plan"),
            ("/history, /sessions", "List your recent sessions"),
            ("/resume, /restore <id>", "Resume a previous session"),
            ("/context", "Show context size and token usage"),
            ("/compact", "Compact context to reduce token usage"),
            ("/facts", "Show cached facts from this session"),
            ("/remember <fact> [as name]", "Persist a session fact or extract from text"),
            ("/forget <name>", "Forget a remembered fact by name"),
            ("/verbose [on|off]", "Toggle or set verbose mode (step details)"),
            ("/raw [on|off]", "Toggle or set raw output display"),
            ("/insights [on|off]", "Toggle or set insight synthesis"),
            ("/preferences", "Show current preferences"),
            ("/artifacts", "Show saved artifacts with file:// URIs"),
            ("/database, /databases, /db", "List or manage databases"),
            ("/database save <n> <t> <uri>", "Save database bookmark"),
            ("/database delete <name>", "Delete database bookmark"),
            ("/database <name>", "Use bookmarked database"),
            ("/file, /files", "List all data files"),
            ("/file save <n> <uri>", "Save file bookmark"),
            ("/file delete <name>", "Delete file bookmark"),
            ("/file <name>", "Use bookmarked file"),
            ("/correct <text>", "Record a correction for future reference"),
            ("/learnings [category]", "Show learnings and rules"),
            ("/compact-learnings", "Compact learnings into rules"),
            ("/forget-learning <id>", "Delete a learning by ID"),
            ("/audit", "Re-derive last result with full audit trail"),
            ("/summarize <target>", "Summarize plan|session|facts|<table>"),
            ("/prove", "Verify conversation claims with auditable proof"),
            ("/quit, /q", "Exit"),
        ]
        for cmd, desc in commands:
            table.add_row(cmd, desc)
        self.console.print(table)

    def _show_tables(self) -> None:
        """Show tables in current session with file:// URIs for Parquet files."""
        if not self.api.session or not self.api.session.session_id:
            self.console.print("[yellow]No active session.[/yellow]")
            return

        # Try registry first for Parquet file paths
        try:
            from constat.storage.registry import ConstatRegistry
            registry = ConstatRegistry()
            tables = registry.list_tables(user_id=self.user_id, session_id=self.api.session.session_id)
            registry.close()

            if not tables:
                self.console.print("[dim]No tables yet.[/dim]")
                return

            self.console.print(f"\n[bold]Tables[/bold] ({len(tables)})")
            for t in tables:
                role_suffix = f" [blue]@{t.role_id}[/blue]" if getattr(t, "role_id", None) else ""
                self.console.print(f"  [cyan]{t.name}[/cyan] [dim]({t.row_count} rows)[/dim]{role_suffix}")
                # Show file:// URI for the Parquet file
                file_path = Path(t.file_path)
                if file_path.exists():
                    file_uri = file_path.resolve().as_uri()
                    self.console.print(f"    {file_uri}")

        except Exception:
            # Fall back to datastore
            if not self.api.session.datastore:
                self.console.print("[yellow]No active session.[/yellow]")
                return
            tables = self.api.session.datastore.list_tables()
            if not tables:
                self.console.print("[dim]No tables yet.[/dim]")
                return
            self.display.show_tables(tables, force_show=True)

    def _export_table(self, arg: str) -> None:
        """Export a table to CSV or XLSX file.

        Usage:
            /export <table>              - Export to table.csv in current dir
            /export <table> <filename>   - Export to specified file (.csv or .xlsx)
            /export _facts               - Export facts table
        """
        if not arg.strip():
            self.console.print("[yellow]Usage: /export <table> [filename][/yellow]")
            self.console.print("[dim]Example: /export orders orders.csv[/dim]")
            self.console.print("[dim]Example: /export orders report.xlsx[/dim]")
            self.console.print("[dim]Example: /export _facts[/dim]")
            return

        parts = arg.strip().split(maxsplit=1)
        table_name = parts[0]
        filename = parts[1] if len(parts) > 1 else f"{table_name}.csv"

        # Determine format from extension
        ext = Path(filename).suffix.lower()
        if ext not in (".csv", ".xlsx"):
            self.console.print(f"[yellow]Unsupported format: {ext}. Use .csv or .xlsx[/yellow]")
            return

        try:
            # Handle special _facts table
            if table_name == "_facts":
                if not self.api.session:
                    self.console.print("[yellow]No active session.[/yellow]")
                    return
                df = self.api.session.fact_resolver.get_facts_as_dataframe()
                if df.empty:
                    self.console.print("[dim]No facts to export.[/dim]")
                    return
            else:
                # Get table from datastore
                if not self.api.session or not self.api.session.datastore:
                    self.console.print("[yellow]No active session.[/yellow]")
                    return

                tables = self.api.session.datastore.list_tables()
                table_names = [t['name'] for t in tables]
                if table_name not in table_names:
                    self.console.print(f"[yellow]Table '{table_name}' not found.[/yellow]")
                    self.console.print(f"[dim]Available: {', '.join(table_names) or '(none)'}[/dim]")
                    return

                df = self.api.session.datastore.query(f"SELECT * FROM {table_name}")

            # Export to file
            output_path = Path(filename).resolve()
            if ext == ".csv":
                df.to_csv(output_path, index=False)
            else:  # .xlsx
                df.to_excel(output_path, index=False)

            self.console.print(f"[green]Exported {len(df)} rows to:[/green]")
            self.console.print(f"  {output_path.as_uri()}")

        except Exception as e:
            self.console.print(f"[red]Export failed:[/red] {e}")

    def _show_artifacts(self) -> None:
        """Show session artifacts: tables (Parquet) and saved files from registry."""
        if not self.api.session or not self.api.session.session_id:
            self.console.print("[yellow]No active session.[/yellow]")
            return

        has_artifacts = False
        session_id = self.api.session.session_id

        # Try to get registry (may not exist yet)
        try:
            from constat.storage.registry import ConstatRegistry
            registry = ConstatRegistry()

            # Show tables from registry (with file:// URIs for Parquet files)
            tables = registry.list_tables(user_id=self.user_id, session_id=session_id)
            if tables:
                has_artifacts = True
                self.console.print(f"\n[bold]Tables[/bold] ({len(tables)})")
                for t in tables:
                    role_suffix = f" [blue]@{t.role_id}[/blue]" if getattr(t, "role_id", None) else ""
                    self.console.print(f"  [cyan]{t.name}[/cyan] [dim]({t.row_count} rows)[/dim]{role_suffix}")
                    if t.description:
                        self.console.print(f"    {t.description}")
                    # Show file:// URI for the Parquet file
                    file_path = Path(t.file_path)
                    if file_path.exists():
                        file_uri = file_path.resolve().as_uri()
                        self.console.print(f"    {file_uri}")

            # Show artifacts from registry (charts, files, etc.)
            artifacts = registry.list_artifacts(user_id=self.user_id, session_id=session_id)
            if artifacts:
                has_artifacts = True
                self.console.print(f"\n[bold]Files[/bold] ({len(artifacts)})")
                for a in artifacts[:20]:
                    file_path = Path(a.file_path)
                    if file_path.exists():
                        file_uri = file_path.resolve().as_uri()
                        size_str = f"{a.size_bytes / 1024:.1f}KB" if a.size_bytes else ""
                        role_suffix = f" [blue]@{a.role_id}[/blue]" if getattr(a, "role_id", None) else ""
                        self.console.print(f"  [cyan]{a.name}[/cyan] [dim]({a.artifact_type}) {size_str}[/dim]{role_suffix}")
                        if a.description:
                            self.console.print(f"    {a.description}")
                        self.console.print(f"    {file_uri}")

                if len(artifacts) > 20:
                    self.console.print(f"\n[dim]... and {len(artifacts) - 20} more[/dim]")

            registry.close()

        except Exception:
            # Fall back to datastore if registry not available
            if self.api.session.datastore:
                tables = self.api.session.datastore.list_tables()
                if tables:
                    has_artifacts = True
                    self.console.print(f"\n[bold]Tables[/bold] ({len(tables)})")
                    for t in tables:
                        role_suffix = f" [blue]@{t.get('role_id')}[/blue]" if t.get("role_id") else ""
                        self.console.print(f"  [cyan]{t['name']}[/cyan] [dim]({t['row_count']} rows)[/dim]{role_suffix}")

        if not has_artifacts:
            self.console.print("[dim]No artifacts in this session.[/dim]")

    def _handle_database(self, arg: str) -> None:
        """Handle /database command variants.

        /database                           - List all databases
        /database save <name> <type> <uri> [--desc "..."]  - Save bookmark
        /database delete <name>             - Delete bookmark
        /database <name>                    - Use bookmark in session
        /database <name> <type> <uri> [--desc "..."]  - Add for this session only
        """
        from constat.storage.bookmarks import BookmarkStore

        if not arg:
            # List all databases
            self._show_databases()
            return

        parts = arg.split()
        subcommand = parts[0].lower()

        if subcommand == "save" and len(parts) >= 4:
            # /database save <name> <type> <uri> [--desc "..."]
            name, db_type, uri = parts[1], parts[2], parts[3]
            description = self._extract_flag(arg, "--desc") or ""
            bookmarks = BookmarkStore()
            bookmarks.save_database(name, db_type, uri, description)
            self.console.print(f"[green]Saved database bookmark:[/green] {name}")

        elif subcommand == "delete" and len(parts) >= 2:
            # /database delete <name>
            name = parts[1]
            bookmarks = BookmarkStore()
            if bookmarks.delete_database(name):
                self.console.print(f"[green]Deleted database bookmark:[/green] {name}")
            else:
                self.console.print(f"[yellow]Bookmark not found:[/yellow] {name}")

        elif len(parts) == 1:
            # /database <name> - Use bookmark
            name = parts[0]
            bookmarks = BookmarkStore()
            bm = bookmarks.get_database(name)
            if bm:
                if self.api.session:
                    self.api.session.add_database(name, bm["type"], bm["uri"], bm["description"])
                    self.console.print(f"[green]Added database to session:[/green] {name} ({bm['type']})")
                else:
                    self.console.print("[yellow]Start a session first by asking a question.[/yellow]")
            else:
                self.console.print(f"[yellow]Bookmark not found:[/yellow] {name}")

        elif len(parts) >= 3:
            # /database <name> <type> <uri> [--desc "..."] - Session-only
            name, db_type, uri = parts[0], parts[1], parts[2]
            description = self._extract_flag(arg, "--desc") or ""
            if self.api.session:
                self.api.session.add_database(name, db_type, uri, description)
                self.console.print(f"[green]Added database to session:[/green] {name} ({db_type})")
            else:
                self.console.print("[yellow]Start a session first by asking a question.[/yellow]")

        else:
            self.console.print("[yellow]Usage: /database [save|delete] <name> [<type> <uri>] [--desc \"...\"][/yellow]")

    def _handle_file(self, arg: str) -> None:
        """Handle /file command variants.

        /file                               - List all files
        /file save <name> <uri> [--auth "..."] [--desc "..."]  - Save bookmark
        /file delete <name>                 - Delete bookmark
        /file <name>                        - Use bookmark in session
        /file <name> <uri> [--auth "..."] [--desc "..."]  - Add for this session only
        """
        from constat.storage.bookmarks import BookmarkStore

        if not arg:
            # List all files
            self._show_files()
            return

        parts = arg.split()
        subcommand = parts[0].lower()

        if subcommand == "save" and len(parts) >= 3:
            # /file save <name> <uri> [--auth "..."] [--desc "..."]
            name, uri = parts[1], parts[2]
            auth = self._extract_flag(arg, "--auth") or ""
            description = self._extract_flag(arg, "--desc") or ""
            bookmarks = BookmarkStore()
            bookmarks.save_file(name, uri, description, auth)
            self.console.print(f"[green]Saved file bookmark:[/green] {name}")

        elif subcommand == "delete" and len(parts) >= 2:
            # /file delete <name>
            name = parts[1]
            bookmarks = BookmarkStore()
            if bookmarks.delete_file(name):
                self.console.print(f"[green]Deleted file bookmark:[/green] {name}")
            else:
                self.console.print(f"[yellow]Bookmark not found:[/yellow] {name}")

        elif len(parts) == 1:
            # /file <name> - Use bookmark
            name = parts[0]
            bookmarks = BookmarkStore()
            bm = bookmarks.get_file(name)
            if bm:
                if self.api.session:
                    self.api.session.add_file(name, bm["uri"], bm.get("auth", ""), bm["description"])
                    self.console.print(f"[green]Added file to session:[/green] {name}")
                else:
                    self.console.print("[yellow]Start a session first by asking a question.[/yellow]")
            else:
                self.console.print(f"[yellow]Bookmark not found:[/yellow] {name}")

        elif len(parts) >= 2:
            # /file <name> <uri> [--auth "..."] [--desc "..."] - Session-only
            name, uri = parts[0], parts[1]
            auth = self._extract_flag(arg, "--auth") or ""
            description = self._extract_flag(arg, "--desc") or ""
            if self.api.session:
                self.api.session.add_file(name, uri, auth, description)
                self.console.print(f"[green]Added file to session:[/green] {name}")
            else:
                self.console.print("[yellow]Start a session first by asking a question.[/yellow]")

        else:
            self.console.print("[yellow]Usage: /file [save|delete] <name> [<uri>] [--auth \"...\"] [--desc \"...\"][/yellow]")

    def _extract_flag(self, text: str, flag: str) -> Optional[str]:
        """Extract a flag value from command text.

        Handles both --flag "quoted value" and --flag value formats.
        """
        import re

        # Try quoted value first
        pattern = rf'{flag}\s+"([^"]*)"'
        match = re.search(pattern, text)
        if match:
            return match.group(1)

        # Try unquoted value (next word after flag)
        pattern = rf'{flag}\s+(\S+)'
        match = re.search(pattern, text)
        if match:
            return match.group(1)

        return None

    def _show_databases(self) -> None:
        """Show all databases (config + bookmarks + session)."""
        from constat.storage.bookmarks import BookmarkStore

        # Group databases by source
        config_dbs = {}
        bookmark_dbs = {}
        session_dbs = {}

        # Get from session if available
        if self.api.session:
            all_dbs = self.api.session.get_all_databases()
            for name, db in all_dbs.items():
                if db["source"] == "config":
                    config_dbs[name] = db
                elif db["source"] == "bookmark":
                    bookmark_dbs[name] = db
                elif db["source"] == "session":
                    session_dbs[name] = db
        else:
            # No session - show config and bookmarks only
            if self.config and self.config.databases:
                for name, db_config in self.config.databases.items():
                    config_dbs[name] = {
                        "type": db_config.type or "sql",
                        "uri": db_config.uri or db_config.path or "",
                        "description": db_config.description or "",
                    }
            bookmarks = BookmarkStore()
            bookmark_dbs = bookmarks.list_databases()

        has_any = config_dbs or bookmark_dbs or session_dbs
        if not has_any:
            self.console.print("[dim]No databases configured.[/dim]")
            return

        if config_dbs:
            self.console.print(f"\n[bold]Config Databases[/bold] ({len(config_dbs)})")
            for name, db in config_dbs.items():
                uri_display = self._mask_credentials(db["uri"])
                self.console.print(f"  [cyan]{name}[/cyan] [dim]({db['type']})[/dim]")
                if db["description"]:
                    self.console.print(f"    {db['description']}")
                self.console.print(f"    [dim]{uri_display}[/dim]")

        if bookmark_dbs:
            self.console.print(f"\n[bold]Bookmarked Databases[/bold] ({len(bookmark_dbs)})")
            for name, db in bookmark_dbs.items():
                uri_display = self._mask_credentials(db["uri"])
                self.console.print(f"  [cyan]{name}[/cyan] [dim]({db['type']})[/dim]")
                if db["description"]:
                    self.console.print(f"    {db['description']}")
                self.console.print(f"    [dim]{uri_display}[/dim]")

        if session_dbs:
            self.console.print(f"\n[bold]Session Databases[/bold] ({len(session_dbs)})")
            for name, db in session_dbs.items():
                uri_display = self._mask_credentials(db["uri"])
                self.console.print(f"  [cyan]{name}[/cyan] [dim]({db['type']})[/dim]")
                if db["description"]:
                    self.console.print(f"    {db['description']}")
                self.console.print(f"    [dim]{uri_display}[/dim]")

    def _show_files(self) -> None:
        """Show all files (config docs + file sources + bookmarks + session)."""
        from constat.storage.bookmarks import BookmarkStore

        # Group files by source
        config_files = {}
        bookmark_files = {}
        session_files = {}

        # Get from session if available
        if self.api.session:
            all_files = self.api.session.get_all_files()
            for name, f in all_files.items():
                if f["source"] == "config":
                    config_files[name] = f
                elif f["source"] == "bookmark":
                    bookmark_files[name] = f
                elif f["source"] == "session":
                    session_files[name] = f
        else:
            # No session - show config and bookmarks only
            if self.config:
                # Documents
                if self.config.documents:
                    for name, doc in self.config.documents.items():
                        uri = ""
                        if doc.path:
                            uri = f"file://{doc.path}"
                        elif doc.url:
                            uri = doc.url
                        config_files[name] = {
                            "uri": uri,
                            "description": doc.description or "",
                            "file_type": "document",
                        }
                # File-type databases
                for name, db in self.config.databases.items():
                    if db.type in ("csv", "json", "jsonl", "parquet", "arrow", "feather"):
                        path = db.path or db.uri or ""
                        config_files[name] = {
                            "uri": f"file://{path}" if not path.startswith(("file://", "http")) else path,
                            "description": db.description or "",
                            "file_type": db.type,
                        }
            bookmarks = BookmarkStore()
            bookmark_files = bookmarks.list_files()

        has_any = config_files or bookmark_files or session_files
        if not has_any:
            self.console.print("[dim]No files configured.[/dim]")
            return

        if config_files:
            self.console.print(f"\n[bold]Config Files[/bold] ({len(config_files)})")
            for name, f in config_files.items():
                file_type = f.get("file_type", "file")
                self.console.print(f"  [cyan]{name}[/cyan] [dim]({file_type})[/dim]")
                if f.get("description"):
                    self.console.print(f"    {f['description']}")
                self.console.print(f"    [dim]{f['uri']}[/dim]")

        if bookmark_files:
            self.console.print(f"\n[bold]Bookmarked Files[/bold] ({len(bookmark_files)})")
            for name, f in bookmark_files.items():
                auth_status = " [auth]" if f.get("auth") else ""
                self.console.print(f"  [cyan]{name}[/cyan]{auth_status}")
                if f.get("description"):
                    self.console.print(f"    {f['description']}")
                self.console.print(f"    [dim]{f['uri']}[/dim]")

        if session_files:
            self.console.print(f"\n[bold]Session Files[/bold] ({len(session_files)})")
            for name, f in session_files.items():
                auth_status = " [auth]" if f.get("auth") else ""
                self.console.print(f"  [cyan]{name}[/cyan]{auth_status}")
                if f.get("description"):
                    self.console.print(f"    {f['description']}")
                self.console.print(f"    [dim]{f['uri']}[/dim]")

    def _mask_credentials(self, uri: str) -> str:
        """Mask credentials in a URI for display."""
        import re
        # Mask password in URIs like postgresql://user:password@host
        return re.sub(r'://([^:]+):([^@]+)@', r'://\1:***@', uri)

    # -------------------------------------------------------------------------
    # Learning/Correction Commands
    # -------------------------------------------------------------------------

    def _handle_correct(self, arg: str) -> None:
        """Handle /correct <text> - explicit user correction."""
        if not arg.strip():
            self.console.print("[yellow]Usage: /correct <correction>[/yellow]")
            self.console.print("[dim]Example: /correct 'active users' means users logged in within 30 days[/dim]")
            return

        self.api.learning_store.save_learning(
            category=LearningCategory.USER_CORRECTION,
            context={
                "previous_question": self.last_problem,
                "correction_text": arg,
            },
            correction=arg,
            source=LearningSource.EXPLICIT_COMMAND,
        )
        self.console.print(f"[green]Learned:[/green] {arg[:60]}{'...' if len(arg) > 60 else ''}")

        # Auto-compact if threshold reached
        self._maybe_auto_compact()

    def _show_learnings(self, arg: str = "") -> None:
        """Handle /learnings [category] - show learnings and rules."""
        # Parse optional category filter
        category = None
        if arg.strip():
            try:
                category = LearningCategory(arg.strip().lower())
            except ValueError:
                self.console.print(f"[yellow]Unknown category: {arg}[/yellow]")
                self.console.print("[dim]Valid: user_correction, api_error, codegen_error, nl_correction[/dim]")

        # Show rules first (compacted, high-value)
        rules = self.api.learning_store.list_rules(category=category)
        if rules:
            self.console.print(f"\n[bold]Rules[/bold] ({len(rules)})")
            for r in rules[:10]:
                conf = r.get("confidence", 0) * 100
                applied = r.get("applied_count", 0)
                self.console.print(f"  [{conf:.0f}%] {r['summary'][:60]} [dim](applied {applied}x)[/dim]")

        # Show pending raw learnings
        raw = self.api.learning_store.list_raw_learnings(category=category, limit=20)
        pending = [l for l in raw if not l.get("promoted_to")]
        if pending:
            self.console.print(f"\n[bold]Pending Learnings[/bold] ({len(pending)})")
            for l in pending[:10]:
                cat = l.get("category", "")[:10]
                lid = l.get("id", "")[:12]
                self.console.print(f"  [dim]{lid}[/dim] [{cat}] {l['correction'][:50]}...")

        # Show stats
        stats = self.api.learning_store.get_stats()
        if stats.get("total_raw", 0) > 0 or stats.get("total_rules", 0) > 0:
            self.console.print(
                f"\n[dim]Total: {stats.get('unpromoted', 0)} pending, "
                f"{stats.get('total_rules', 0)} rules, "
                f"{stats.get('total_archived', 0)} archived[/dim]"
            )
        else:
            self.console.print("[dim]No learnings yet. Use /correct to add corrections.[/dim]")

    def _compact_learnings(self) -> None:
        """Handle /compact-learnings - trigger compaction."""
        from constat.learning.compactor import LearningCompactor

        if not self.api.session:
            self.console.print("[yellow]Start a session first by asking a question.[/yellow]")
            return

        stats = self.api.learning_store.get_stats()
        if stats.get("unpromoted", 0) < 2:
            self.console.print("[dim]Not enough learnings to compact (need at least 2).[/dim]")
            return

        self.display.start_spinner("Analyzing learnings for patterns...")
        try:
            # Get LLM from session's router
            llm = self.api.session.router._get_provider(self.api.session.router.models["planning"])
            compactor = LearningCompactor(self.api.learning_store, llm)
            result = compactor.compact()

            self.display.stop_spinner()
            self.console.print(f"[green]Compaction complete:[/green]")
            self.console.print(f"  Rules created: {result.rules_created}")
            self.console.print(f"  Learnings archived: {result.learnings_archived}")
            if result.skipped_low_confidence > 0:
                self.console.print(f"  [dim]Skipped (low confidence): {result.skipped_low_confidence}[/dim]")
            if result.errors:
                for err in result.errors:
                    self.console.print(f"  [yellow]Warning: {err}[/yellow]")
        except Exception as e:
            self.display.stop_spinner()
            self.console.print(f"[red]Error:[/red] {e}")

    def _maybe_auto_compact(self) -> None:
        """Check if auto-compaction should trigger and run it."""
        stats = self.api.learning_store.get_stats()
        unpromoted = stats.get("unpromoted", 0)

        # Use API method for auto-compaction
        result = self.api.maybe_auto_compact()

        if result is not None:
            # Compaction was triggered
            if result.rules_created > 0:
                self.console.print(
                    f"[green]Created {result.rules_created} rules from "
                    f"{result.learnings_archived} learnings[/green]"
                )
            else:
                self.console.print(
                    f"[dim]No new rules created (reviewed {unpromoted} learnings)[/dim]"
                )

    def _forget_learning(self, learning_id: str) -> None:
        """Handle /forget-learning <id> - delete a learning."""
        learning_id = learning_id.strip()
        if self.api.learning_store.delete_learning(learning_id):
            self.console.print(f"[green]Deleted learning:[/green] {learning_id}")
        else:
            # Try deleting as a rule
            if self.api.learning_store.delete_rule(learning_id):
                self.console.print(f"[green]Deleted rule:[/green] {learning_id}")
            else:
                self.console.print(f"[yellow]Not found:[/yellow] {learning_id}")
                self.console.print("[dim]Use /learnings to see IDs[/dim]")

    def _handle_audit(self) -> None:
        """Handle /audit command - re-derive last result with full audit trail."""
        if not self.api.session:
            self.console.print("[yellow]No active session. Ask a question first.[/yellow]")
            return

        self.display.start_spinner("Re-deriving with full audit trail...")
        try:
            result = self.api.session.audit()
            self.display.stop_spinner()

            if result.get("success"):
                # Display the audit output
                output = result.get("output", "")
                if output:
                    self.console.print()
                    self.console.print(Panel(
                        output,
                        title="[bold]Audit Result[/bold]",
                        border_style="green",
                    ))

                # Show verification status if available
                verification = result.get("verification")
                if verification:
                    status = verification.get("verified", False)
                    msg = verification.get("message", "")
                    if status:
                        self.console.print(f"\n[bold green]Verified:[/bold green] {msg}")
                    else:
                        self.console.print(f"\n[bold yellow]Discrepancy:[/bold yellow] {msg}")

                # Show suggestions
                suggestions = result.get("suggestions", [])
                if suggestions:
                    self.display.show_suggestions(suggestions)
            else:
                error = result.get("error", "Unknown error")
                self.console.print(f"[red]Audit failed:[/red] {error}")

        except Exception as e:
            self.display.stop_spinner()
            self.console.print(f"[red]Error during audit:[/red] {e}")

    def _handle_prove(self) -> None:
        """Handle /prove command - verify conversation claims with auditable proof.

        This command:
        1. Collects claims from the conversation history
        2. Generates an auditable proof for each claim
        3. Reports any discrepancies with original answers
        """
        if not self.api.session:
            self.console.print("[yellow]No active session. Ask questions first, then use /prove to verify.[/yellow]")
            return

        if not self.api.session.session_id:
            self.console.print("[yellow]No conversation to prove. Ask questions first.[/yellow]")
            return

        self.console.print("[cyan]Generating auditable proof for conversation claims...[/cyan]")
        self.display.start_spinner("Analyzing conversation...")

        try:
            result = self.api.session.prove_conversation()

            self.display.stop_spinner()

            if result.get("error"):
                self.console.print(f"[red]Error:[/red] {result['error']}")
                return

            if result.get("no_claims"):
                self.console.print("[yellow]No verifiable claims found in conversation.[/yellow]")
                self.console.print("[dim]Try asking data-related questions first, then use /prove.[/dim]")
                return

            # Display proof results
            claims = result.get("claims", [])
            self.console.print(f"\n[bold]Verified {len(claims)} claim(s):[/bold]\n")

            for i, claim in enumerate(claims, 1):
                status = "[green]VERIFIED[/green]" if claim.get("verified") else "[red]UNVERIFIED[/red]"
                self.console.print(f"  {i}. {status} {claim.get('claim', '')}")
                if claim.get("proof"):
                    self.console.print(f"     [dim]Proof: {claim['proof'][:100]}...[/dim]")
                if claim.get("discrepancy"):
                    self.console.print(f"     [yellow]Discrepancy: {claim['discrepancy']}[/yellow]")

        except Exception as e:
            self.display.stop_spinner()
            self.console.print(f"[red]Error during proof generation:[/red] {e}")

    def _handle_summarize(self, arg: str) -> None:
        """Handle /summarize command - generate LLM summary of plan, session, facts, or table.

        Usage:
            /summarize plan     - Summarize the current execution plan
            /summarize session  - Summarize the current session state
            /summarize facts    - Summarize all cached facts
            /summarize <table>  - Summarize a specific table's contents
        """
        if not arg.strip():
            self.console.print("[yellow]Usage: /summarize plan|session|facts|<table_name>[/yellow]")
            self.console.print("[dim]Examples:[/dim]")
            self.console.print("  /summarize plan     - Summarize execution plan")
            self.console.print("  /summarize session  - Summarize session state")
            self.console.print("  /summarize facts    - Summarize cached facts")
            self.console.print("  /summarize orders   - Summarize 'orders' table")
            return

        target = arg.strip().lower()
        self.display.start_spinner(f"Generating summary of {target}...")

        try:
            if target == "plan":
                result = self.api.summarize_plan()
            elif target == "session":
                result = self.api.summarize_session()
            elif target == "facts":
                result = self.api.summarize_facts()
            else:
                # Assume it's a table name
                result = self.api.summarize_table(arg.strip())

            self.display.stop_spinner()

            if result.success and result.summary:
                self.console.print()
                self.console.print(Panel(
                    result.summary,
                    title=f"[bold]Summary: {target}[/bold]",
                    border_style="cyan",
                ))
            elif result.error:
                self.console.print(f"[yellow]{result.error}[/yellow]")
            else:
                self.console.print(f"[yellow]No data to summarize for '{target}'[/yellow]")

        except Exception as e:
            self.display.stop_spinner()
            self.console.print(f"[red]Error generating summary:[/red] {e}")

    def _save_nl_correction(self, correction, full_text: str) -> None:
        """Save an NL-detected correction as a learning.

        Args:
            correction: CorrectionDetection from detect_nl_correction()
            full_text: The original user input text
        """
        self.api.learning_store.save_learning(
            category=LearningCategory.NL_CORRECTION,
            context={
                "match_type": correction.correction_type,
                "matched_text": correction.matched_text,
                "full_text": full_text,
                "previous_question": self.last_problem,
            },
            correction=full_text,
            source=LearningSource.NL_DETECTION,
        )
        # Auto-compact if threshold reached
        self._maybe_auto_compact()

    def _run_query(self, sql: str) -> None:
        """Run SQL query on datastore."""
        if not self.api.session or not self.api.session.datastore:
            self.console.print("[yellow]No active session.[/yellow]")
            return
        try:
            result = self.api.session.datastore.query(sql)
            self.console.print(result.to_string())
        except Exception as e:
            self.console.print(f"[red]Query error:[/red] {e}")

    def _show_state(self) -> None:
        """Show current session state."""
        if not self.api.session:
            self.console.print("[yellow]No active session.[/yellow]")
            return
        state = self.api.session.get_state()
        self.console.print(f"\n[bold]Session:[/bold] {state['session_id']}")
        if state['datastore_tables']:
            self.console.print("[bold]Tables:[/bold]")
            for t in state['datastore_tables']:
                self.console.print(f"  - {t['name']} ({t['row_count']} rows)")

    def _refresh_metadata(self) -> None:
        """Refresh database metadata, documents, and preload cache."""
        if not self.api.session:
            self.console.print("[yellow]No active session.[/yellow]")
            return
        self.display.start_spinner("Refreshing metadata and documents...")
        try:
            stats = self.api.session.refresh_metadata()
            self.display.stop_spinner()

            # Build status message
            parts = ["[green]Refreshed:[/green]"]

            # Preloaded tables
            if stats.get("preloaded_tables", 0) > 0:
                parts.append(f"{stats['preloaded_tables']} tables preloaded")

            # Document changes
            doc_stats = stats.get("documents", {})
            if doc_stats:
                doc_parts = []
                if doc_stats.get("added", 0) > 0:
                    doc_parts.append(f"{doc_stats['added']} added")
                if doc_stats.get("updated", 0) > 0:
                    doc_parts.append(f"{doc_stats['updated']} updated")
                if doc_stats.get("removed", 0) > 0:
                    doc_parts.append(f"{doc_stats['removed']} removed")
                if doc_stats.get("unchanged", 0) > 0:
                    doc_parts.append(f"{doc_stats['unchanged']} unchanged")
                if doc_parts:
                    parts.append(f"docs: {', '.join(doc_parts)}")

            self.console.print(" ".join(parts) if len(parts) > 1 else "[green]Metadata refreshed.[/green]")
        except Exception as e:
            self.display.stop_spinner()
            self.console.print(f"[red]Error refreshing metadata:[/red] {e}")

    def _reset_session(self) -> None:
        """Reset session state and start a new session."""
        # Reset display state
        self.display.reset()

        # Create a fresh session with new session ID
        self.api = self._create_api(new_session=True)

        # Clear REPL state
        self.last_problem = ""
        self.suggestions = []

        session_id = self.api.session.session_id
        self.console.print(f"[green]New session: {session_id[:8]}...[/green]")

    def _handle_redo(self, arg: str) -> None:
        """Handle /redo command - retry last query with optional modifications.

        Args:
            arg: Optional modification instruction (e.g., "only use Q1 data")
        """
        if not self.last_problem:
            self.console.print("[yellow]No previous query to redo.[/yellow]")
            return

        if not self.api.session or not self.api.session.session_id:
            self.console.print("[yellow]No active session. Use the original query.[/yellow]")
            return

        # Build the redo query
        if arg:
            # Redo with modification instruction
            redo_query = f"redo. {arg}"
            self.console.print(f"[dim]Retrying with: {arg}[/dim]")
        else:
            # Simple retry (for transient errors)
            redo_query = "redo"
            self.console.print("[dim]Retrying last query...[/dim]")

        # Execute as a follow-up (will be detected as redo intent)
        self._solve(redo_query)

    def _show_context(self) -> None:
        """Show context size and token usage statistics."""
        if not self.api.session:
            self.console.print("[yellow]No active session.[/yellow]")
            return

        stats = self.api.session.get_context_stats()
        if not stats:
            self.console.print("[yellow]No datastore available.[/yellow]")
            return

        # Build display
        from rich.panel import Panel as RichPanel

        # Color based on status
        if stats.is_critical:
            title_style = "bold red"
            status = "CRITICAL - Consider using /compact"
        elif stats.is_warning:
            title_style = "bold yellow"
            status = "WARNING - Context growing large"
        else:
            title_style = "bold green"
            status = "OK"

        content = stats.summary()
        content += f"\n\nStatus: {status}"

        self.console.print(RichPanel(
            content,
            title="[bold]Context Size[/bold]",
            title_align="left",
            border_style=title_style,
        ))

    def _compact_context(self) -> None:
        """Compact context to reduce token usage."""
        if not self.api.session:
            self.console.print("[yellow]No active session.[/yellow]")
            return

        # Show before stats
        stats_before = self.api.session.get_context_stats()
        if not stats_before:
            self.console.print("[yellow]No datastore available.[/yellow]")
            return

        self.console.print(f"[dim]Before compaction: ~{stats_before.total_tokens:,} tokens[/dim]")

        # Perform compaction
        self.display.start_spinner("Compacting context...")
        try:
            result = self.api.session.compact_context(
                summarize_scratchpad=True,
                sample_tables=True,
                clear_old_state=False,  # Conservative by default
                keep_recent_steps=3,
            )
            self.display.stop_spinner()

            if result:
                self.console.print(f"[green]{result.message}[/green]")
                self.console.print(result.summary())
            else:
                self.console.print("[yellow]Compaction returned no result.[/yellow]")

        except Exception as e:
            self.display.stop_spinner()
            self.console.print(f"[red]Error during compaction:[/red] {e}")

    def _show_facts(self) -> None:
        """Show both persistent and session facts."""
        # Get persistent facts
        persistent_facts = self.api.fact_store.list_facts()

        # Get session facts
        session_facts = {}
        if self.api.session:
            session_facts = self.api.session.fact_resolver.get_all_facts()

        if not persistent_facts and not session_facts:
            self.console.print("[dim]No facts stored.[/dim]")
            self.console.print("[dim]Use /remember to save facts that persist across sessions.[/dim]")
            return

        table = Table(title="Facts", show_header=True, box=None)
        table.add_column("Name", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Description", style="dim")
        table.add_column("Source", style="dim")
        table.add_column("Role", style="blue")

        # Show persistent facts first
        for name, fact_data in persistent_facts.items():
            value = fact_data.get("value", "")
            desc = fact_data.get("description", "")
            role_id = fact_data.get("role_id") or ""
            table.add_row(name, str(value), desc, "[bold]persistent[/bold]", role_id)

        # Show session facts (skip if same name exists in persistent)
        for name, fact in session_facts.items():
            if name in persistent_facts:
                continue  # Skip - persistent version shown above

            desc = fact.description or ""
            role_id = getattr(fact, "role_id", None) or ""
            # Build specific source info with source_name
            if fact.source_name:
                source_info = f"{fact.source.value}:{fact.source_name}"
                if fact.api_endpoint:
                    source_info += f" ({fact.api_endpoint})"
            elif fact.api_endpoint:
                endpoint_preview = fact.api_endpoint[:50] + "..." if len(fact.api_endpoint) > 50 else fact.api_endpoint
                source_info = f"api: {endpoint_preview}"
            elif fact.query:
                query_preview = fact.query[:50] + "..." if len(fact.query) > 50 else fact.query
                source_info = f"SQL: {query_preview}"
            elif fact.rule_name:
                source_info = f"rule: {fact.rule_name}"
            elif fact.reasoning and fact.source.value == "user_provided":
                source_info = fact.reasoning[:40] + "..." if len(fact.reasoning) > 40 else fact.reasoning
            else:
                source_info = f"session:{fact.source.value}"
            table.add_row(name, fact.display_value, desc, source_info, role_id)

        self.console.print(table)

        # Sync facts to datastore as "_facts" table for SQL queries
        if self.api.session and self.api.session.datastore:
            try:
                facts_df = self.api.session.fact_resolver.get_facts_as_dataframe()
                if not facts_df.empty:
                    self.api.session.datastore.save_dataframe("_facts", facts_df)
                    self.console.print("[dim]Facts synced to _facts table (queryable via SQL)[/dim]")
            except Exception:
                pass  # Silent fail - facts table is optional

    def _remember_fact(self, fact_text: str) -> None:
        """Remember a fact persistently (survives across sessions).

        Supports two modes:
        1. Promote session fact: /remember <fact-name> [as <new-name>]
           Looks up a resolved fact from the current session and persists it.
        2. Extract from text: /remember my role is CFO
           Parses natural language to extract and persist a new fact.
        """
        import re

        if not fact_text.strip():
            self.console.print("[yellow]Usage: /remember <fact>[/yellow]")
            self.console.print("[dim]Examples:[/dim]")
            self.console.print("[dim]  /remember enterprise_churn_rate    - persist a session fact[/dim]")
            self.console.print("[dim]  /remember churn_rate as baseline   - persist with new name[/dim]")
            self.console.print("[dim]  /remember my role is CFO           - extract from text[/dim]")
            return

        # Check if this is a session fact reference with optional rename
        # Pattern: <fact_name> [as <new_name>]
        session_fact_match = re.match(r'^(\S+)(?:\s+as\s+(\S+))?$', fact_text.strip())

        if session_fact_match and self.api.session:
            fact_name = session_fact_match.group(1)
            new_name = session_fact_match.group(2)  # May be None

            # Try to find this fact in the session's resolver cache
            session_facts = self.api.session.fact_resolver.get_all_facts()

            # Look for exact match or match with empty params (e.g., "fact_name()")
            matching_fact = None
            matching_key = None

            for key, fact in session_facts.items():
                # Match by cache key (e.g., "churn_rate()" or "customer_ltv(id=ACME)")
                if key == fact_name or key == f"{fact_name}()":
                    matching_fact = fact
                    matching_key = key
                    break
                # Also match by fact.name property
                if fact.name == fact_name:
                    matching_fact = fact
                    matching_key = key
                    break

            if matching_fact:
                # Persist this session fact
                persist_name = new_name if new_name else matching_fact.name

                # Build context string from provenance
                context_parts = [f"Source: {matching_fact.source.value}"]
                if matching_fact.source_name:
                    context_parts.append(f"From: {matching_fact.source_name}")
                if matching_fact.query:
                    context_parts.append(f"Query: {matching_fact.query}")
                if matching_fact.reasoning:
                    context_parts.append(f"Reasoning: {matching_fact.reasoning}")
                if matching_fact.api_endpoint:
                    context_parts.append(f"API: {matching_fact.api_endpoint}")
                if matching_fact.resolved_at:
                    context_parts.append(f"Resolved: {matching_fact.resolved_at.isoformat()}")
                context = "\n".join(context_parts)

                # Build description from fact metadata
                description = matching_fact.description or f"Persisted from session (originally: {matching_key})"

                self.api.fact_store.save_fact(
                    name=persist_name,
                    value=matching_fact.value,
                    description=description,
                    context=context,
                )

                self.console.print(f"[green]Remembered:[/green] {persist_name} = {matching_fact.display_value}")
                self.console.print(f"[dim]Source: {matching_fact.source.value}[/dim]")
                if new_name:
                    self.console.print(f"[dim]Renamed from: {matching_fact.name}[/dim]")
                self.console.print("[dim]This fact will persist across sessions.[/dim]")
                return

        # Fall back to existing behavior: extract fact from natural language
        self.display.start_spinner("Extracting fact...")
        try:
            extracted = []
            if self.api.session:
                extracted = self.api.session.fact_resolver.add_user_facts_from_text(fact_text)
            else:
                # No session - do lightweight extraction
                extracted = self._extract_fact_without_session(fact_text)

            self.display.stop_spinner()

            if extracted:
                for fact in extracted:
                    # Save to persistent store
                    self.api.fact_store.save_fact(
                        name=fact.name if hasattr(fact, 'name') else fact['name'],
                        value=fact.value if hasattr(fact, 'value') else fact['value'],
                        description=fact.description if hasattr(fact, 'description') else fact.get('description', ''),
                        context=fact.context if hasattr(fact, 'context') else fact.get('context', ''),
                    )
                    name = fact.name if hasattr(fact, 'name') else fact['name']
                    value = fact.value if hasattr(fact, 'value') else fact['value']
                    self.console.print(f"[green]Remembered:[/green] {name} = {value}")
                    self.console.print("[dim]This fact will persist across sessions.[/dim]")
            else:
                # Check if it looked like a fact name but wasn't found
                if session_fact_match and self.api.session:
                    fact_name = session_fact_match.group(1)
                    self.console.print(f"[yellow]No session fact named '{fact_name}' found.[/yellow]")
                    self.console.print("[dim]Use /facts to see available facts, or provide natural language.[/dim]")
                else:
                    self.console.print("[yellow]Could not extract a fact from that text.[/yellow]")
                    self.console.print("[dim]Try being more explicit, e.g., 'my role is CFO'[/dim]")

        except Exception as e:
            self.display.stop_spinner()
            self.console.print(f"[red]Error:[/red] {e}")

    def _extract_fact_without_session(self, text: str) -> list[dict]:
        """Extract facts from text without an active session (lightweight)."""
        # Simple pattern matching for common fact patterns
        import re

        patterns = [
            # "my X is Y" pattern
            (r"my\s+(\w+)\s+is\s+(.+)", lambda m: {"name": f"user_{m.group(1)}", "value": m.group(2).strip(), "description": f"User's {m.group(1)}"}),
            # "I am a X" pattern
            (r"i\s+am\s+(?:a|an)\s+(.+)", lambda m: {"name": "user_role", "value": m.group(1).strip(), "description": "User's role"}),
            # "X = Y" pattern
            (r"(\w+)\s*=\s*(.+)", lambda m: {"name": m.group(1).strip(), "value": m.group(2).strip(), "description": ""}),
            # "X is Y" pattern (generic)
            (r"(\w+(?:\s+\w+)?)\s+is\s+(.+)", lambda m: {"name": m.group(1).strip().replace(" ", "_"), "value": m.group(2).strip(), "description": ""}),
        ]

        text_lower = text.lower()
        for pattern, extractor in patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                return [extractor(match)]

        return []

    def _forget_fact(self, fact_name: str) -> None:
        """Forget a fact by name (checks both persistent and session)."""
        if not fact_name.strip():
            self.console.print("[yellow]Usage: /forget <fact_name>[/yellow]")
            self.console.print("[dim]Use /facts to see fact names[/dim]")
            return

        fact_name = fact_name.strip()
        found = False

        # Check persistent facts first
        if self.api.fact_store.delete_fact(fact_name):
            self.console.print(f"[green]Forgot persistent fact:[/green] {fact_name}")
            found = True

        # Also remove from session if exists
        if self.api.session:
            facts = self.api.session.fact_resolver.get_all_facts()
            if fact_name in facts:
                self.api.session.fact_resolver._cache.pop(fact_name, None)
                if not found:
                    self.console.print(f"[green]Forgot session fact:[/green] {fact_name}")
                found = True

        if not found:
            self.console.print(f"[yellow]Fact '{fact_name}' not found.[/yellow]")
            self.console.print("[dim]Use /facts to see available facts[/dim]")

    def _toggle_verbose(self, arg: str = "") -> None:
        """Toggle or set verbose mode on/off."""
        arg_lower = arg.lower().strip()
        if arg_lower == "on":
            self.verbose = True
        elif arg_lower == "off":
            self.verbose = False
        else:
            # Toggle
            self.verbose = not self.verbose

        self.display.verbose = self.verbose
        status = "on" if self.verbose else "off"
        self.console.print(f"Verbose: [bold]{status}[/bold]")
        if self.verbose:
            self.console.print("[dim]Will show detailed step execution info[/dim]")

    def _toggle_raw(self, arg: str = "") -> None:
        """Toggle or set raw output display on/off."""
        arg_lower = arg.lower().strip()
        if arg_lower == "on":
            self.session_config.show_raw_output = True
        elif arg_lower == "off":
            self.session_config.show_raw_output = False
        else:
            # Toggle
            self.session_config.show_raw_output = not self.session_config.show_raw_output

        status = "on" if self.session_config.show_raw_output else "off"
        self.console.print(f"Raw output: [bold]{status}[/bold]")
        if self.session_config.show_raw_output:
            self.console.print("[dim]Raw step results will be shown before synthesis[/dim]")
        else:
            self.console.print("[dim]Only synthesized answer will be shown[/dim]")

    def _toggle_insights(self, arg: str = "") -> None:
        """Toggle or set insight synthesis on/off."""
        arg_lower = arg.lower().strip()
        if arg_lower == "on":
            self.session_config.enable_insights = True
        elif arg_lower == "off":
            self.session_config.enable_insights = False
        else:
            # Toggle
            self.session_config.enable_insights = not self.session_config.enable_insights

        status = "on" if self.session_config.enable_insights else "off"
        self.console.print(f"Insights: [bold]{status}[/bold]")
        if not self.session_config.enable_insights:
            self.console.print("[dim]Raw results will be shown without synthesis[/dim]")

    def _show_preferences(self) -> None:
        """Show current preferences/settings."""
        table = Table(title="Preferences", show_header=True, box=None)
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("verbose", "on" if self.verbose else "off")
        table.add_row("raw", "on" if self.session_config.show_raw_output else "off")
        table.add_row("insights", "on" if self.session_config.enable_insights else "off")
        table.add_row("user", self.user_id)

        # Show default execution mode
        default_mode = self.session_config.default_mode
        mode_display = default_mode.value if default_mode else "auto (LLM decides)"
        table.add_row("default_mode", mode_display)

        self.console.print(table)

    def _show_code(self, step_arg: str = "") -> None:
        """Show generated code for steps."""
        if not self.api.session or not self.api.session.datastore:
            self.console.print("[yellow]No active session.[/yellow]")
            return

        entries = self.api.session.datastore.get_scratchpad()
        if not entries:
            self.console.print("[dim]No steps executed yet.[/dim]")
            return

        if step_arg:
            # Show specific step
            try:
                step_num = int(step_arg)
                entry = next((e for e in entries if e["step_number"] == step_num), None)
                if not entry:
                    self.console.print(f"[yellow]Step {step_num} not found.[/yellow]")
                    return
                self.console.print(f"\n[bold]Step {step_num}:[/bold] {entry['goal']}")
                if entry["code"]:
                    self.console.print(Syntax(entry["code"], "python", theme="monokai", line_numbers=True))
                else:
                    self.console.print("[dim]No code stored for this step.[/dim]")
            except ValueError:
                self.console.print("[red]Invalid step number.[/red]")
        else:
            # Show all steps
            for entry in entries:
                self.console.print(f"\n[bold]Step {entry['step_number']}:[/bold] {entry['goal']}")
                if entry["code"]:
                    self.console.print(Syntax(entry["code"], "python", theme="monokai", line_numbers=True))
                else:
                    self.console.print("[dim]No code stored.[/dim]")

    def _show_user(self, name: str = "") -> None:
        """Show or set current user."""
        if name:
            self.user_id = name
            self.console.print(f"User set to: [bold]{self.user_id}[/bold]")
        else:
            self.console.print(f"Current user: [bold]{self.user_id}[/bold]")

    def _save_plan(self, name: str, shared: bool = False) -> None:
        """Save current plan for replay."""
        if not self.api.session or not self.api.session.datastore:
            self.console.print("[yellow]No active session.[/yellow]")
            return
        if not self.last_problem:
            self.console.print("[yellow]No problem executed yet.[/yellow]")
            return
        try:
            self.api.session.save_plan(name, self.last_problem, user_id=self.user_id, shared=shared)
            if shared:
                self.console.print(f"[green]Plan saved as shared:[/green] {name}")
            else:
                self.console.print(f"[green]Plan saved:[/green] {name}")
        except Exception as e:
            self.console.print(f"[red]Error saving plan:[/red] {e}")

    def _share_with(self, args: str) -> None:
        """Share a plan with a specific user."""
        parts = args.split(maxsplit=1)
        if len(parts) < 2:
            self.console.print("[yellow]Usage: /sharewith <user> <plan_name>[/yellow]")
            return

        target_user, plan_name = parts
        if Session.share_plan_with(plan_name, target_user, from_user=self.user_id):
            self.console.print(f"[green]Plan '{plan_name}' shared with {target_user}[/green]")
        else:
            self.console.print(f"[red]Plan '{plan_name}' not found[/red]")

    def _list_plans(self) -> None:
        """List saved plans."""
        plans = Session.list_saved_plans(user_id=self.user_id)
        if not plans:
            self.console.print("[dim]No saved plans.[/dim]")
            return

        table = Table(title="Saved Plans", show_header=True, box=None)
        table.add_column("Name", style="cyan")
        table.add_column("Problem")
        table.add_column("Steps", justify="right")
        table.add_column("Type")

        for p in plans:
            plan_type = "[dim]shared[/dim]" if p.get("shared") else "private"
            problem = p.get("problem", "")[:50]
            if len(p.get("problem", "")) > 50:
                problem += "..."
            table.add_row(p["name"], problem, str(p.get("steps", 0)), plan_type)

        self.console.print(table)

    def _show_history(self) -> None:
        """Show recent session history for current user."""
        if not self.api.session:
            self.api = self._create_api()

        sessions = self.api.session.history.list_sessions(limit=10)
        if not sessions:
            self.console.print("[dim]No session history.[/dim]")
            return

        table = Table(title=f"Sessions for {self.user_id}", show_header=True, box=None)
        table.add_column("ID", style="cyan")
        table.add_column("Started")
        table.add_column("Summary")
        table.add_column("Queries", justify="right")
        table.add_column("Status")

        for s in sessions:
            # Shorten ID for display
            short_id = s.session_id[:20] + "..." if len(s.session_id) > 20 else s.session_id
            started = s.created_at[:16] if s.created_at else "?"  # Truncate to date/time
            # Truncate summary for table display
            summary = s.summary[:40] + "..." if s.summary and len(s.summary) > 40 else (s.summary or "-")
            table.add_row(short_id, started, summary, str(s.total_queries), s.status or "?")

        self.console.print(table)
        self.console.print("[dim]Use /resume <id> or /restore <id> to continue a session[/dim]")

    def _resume_session(self, session_id: str) -> None:
        """Resume a previous session."""
        if not self.api.session:
            self.api = self._create_api()

        # Find matching session (partial ID match)
        sessions = self.api.session.history.list_sessions(limit=50)
        match = None
        for s in sessions:
            if s.session_id.startswith(session_id) or session_id in s.session_id:
                match = s.session_id
                break

        if not match:
            self.console.print(f"[red]Session not found: {session_id}[/red]")
            return

        if self.api.session.resume(match):
            self.console.print(f"[green]Resumed session:[/green] {match[:30]}...")
            # Show what's available
            tables = self.api.session.datastore.list_tables() if self.api.session.datastore else []
            if tables:
                self.console.print(f"[dim]{len(tables)} tables available - use /tables to view[/dim]")
        else:
            self.console.print(f"[red]Failed to resume session: {match}[/red]")

    def _handle_auto_resume(self) -> None:
        """Handle auto-resume from --continue flag."""
        # Create session if needed to access history
        if not self.api.session:
            self.api = self._create_api()

        # Get most recent session for this user
        sessions = self.api.session.history.list_sessions(limit=1)
        if not sessions:
            self.console.print("[dim]No previous session to resume.[/dim]")
            return

        latest = sessions[0]
        if self.api.session.resume(latest.session_id):
            self.console.print(f"[green]Resumed last session:[/green] {latest.session_id[:30]}...")
            if latest.summary:
                self.console.print(f"[dim]{latest.summary}[/dim]")
            # Show what's available
            tables = self.api.session.datastore.list_tables() if self.api.session.datastore else []
            if tables:
                self.console.print(f"[dim]{len(tables)} tables available - use /tables to view[/dim]")
            self.console.print()
        else:
            self.console.print(f"[yellow]Could not resume session {latest.session_id[:20]}...[/yellow]")

    def _replay_plan(self, name: str) -> None:
        """Replay a saved plan."""
        if not self.api.session:
            self.api = self._create_api()

        try:
            # Load the plan to get the problem
            plan_data = Session.load_saved_plan(name, user_id=self.user_id)
            self.last_problem = plan_data["problem"]
            self.display.set_problem(f"[Replay: {name}] {self.last_problem}")

            result = self.api.session.replay_saved(name, user_id=self.user_id)

            if result.get("success"):
                tables = result.get("datastore_tables", [])
                self.display.show_tables(tables, duration_ms=result.get("duration_ms", 0))
                self.display.show_summary(
                    success=True,
                    total_steps=len(result.get("results", [])),
                    duration_ms=result.get("duration_ms", 0),
                )
            else:
                self.console.print(f"[red]Error:[/red] {result.get('error', 'Unknown error')}")
        except ValueError as e:
            self.console.print(f"[red]Error:[/red] {e}")
        except Exception as e:
            self.console.print(f"[red]Error during replay:[/red] {e}")

    def _check_context_warning(self) -> None:
        """Check context size and warn if getting large."""
        if not self.api.session:
            return

        stats = self.api.session.get_context_stats()
        if not stats:
            return

        if stats.is_critical:
            self.console.print(
                "\n[bold red]Warning:[/bold red] Context size is critical "
                f"(~{stats.total_tokens:,} tokens). Use [cyan]/compact[/cyan] to reduce."
            )
        elif stats.is_warning:
            self.console.print(
                f"\n[yellow]Note:[/yellow] Context is growing (~{stats.total_tokens:,} tokens). "
                "Use [cyan]/context[/cyan] to view details."
            )

    def _handle_command(self, cmd_input: str) -> bool:
        """Handle a slash command.

        Routes core commands through session.solve() to use centralized command registry,
        keeping only REPL-specific presentation here. This ensures REPL and web UI share
        the same command logic.

        Args:
            cmd_input: The full command string (e.g., "/tables" or "/show orders")

        Returns:
            True if the REPL should exit, False otherwise.
        """
        parts = cmd_input.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        # REPL-specific: exit commands
        if cmd in ("/quit", "/exit", "/q"):
            self.console.print("[dim]Goodbye![/dim]")
            return True

        # Commands routed through session for shared core logic
        # These use the centralized command registry
        registry_commands = {
            "/help", "/h", "/tables", "/show", "/query", "/code",
            "/artifacts", "/export", "/state", "/status", "/reset",
            "/facts", "/context", "/preferences",
            "/databases", "/apis", "/documents", "/docs", "/files",
        }

        if cmd in registry_commands or (cmd == "/show" and arg) or (cmd == "/query" and arg):
            return self._run_registry_command(cmd_input)

        # REPL-specific commands with custom presentation or interactive handling
        # These are NOT in the centralized registry
        if cmd in ("/update", "/refresh"):
            self._refresh_metadata()
        elif cmd == "/redo":
            self._handle_redo(arg)
        elif cmd == "/user":
            self._show_user(arg)
        elif cmd == "/save" and arg:
            self._save_plan(arg, shared=False)
        elif cmd == "/share" and arg:
            self._save_plan(arg, shared=True)
        elif cmd == "/sharewith" and arg:
            self._share_with(arg)
        elif cmd == "/plans":
            self._list_plans()
        elif cmd == "/replay" and arg:
            self._replay_plan(arg)
        elif cmd in ("/history", "/sessions"):
            self._show_history()
        elif cmd in ("/resume", "/restore") and arg:
            self._resume_session(arg)
        elif cmd == "/compact":
            self._compact_context()
        elif cmd == "/remember" and arg:
            self._remember_fact(arg)
        elif cmd == "/forget" and arg:
            self._forget_fact(arg)
        elif cmd == "/verbose":
            self._toggle_verbose(arg)
        elif cmd == "/raw":
            self._toggle_raw(arg)
        elif cmd == "/insights":
            self._toggle_insights(arg)
        elif cmd in ("/database", "/db"):
            self._handle_database(arg)
        elif cmd == "/file":
            self._handle_file(arg)
        elif cmd == "/correct":
            self._handle_correct(arg)
        elif cmd == "/learnings":
            self._show_learnings(arg)
        elif cmd == "/compact-learnings":
            self._compact_learnings()
        elif cmd == "/forget-learning" and arg:
            self._forget_learning(arg)
        elif cmd == "/audit":
            self._handle_audit()
        elif cmd == "/summarize":
            self._handle_summarize(arg)
        elif cmd == "/prove":
            self._handle_prove()
        else:
            self.console.print(f"[yellow]Unknown: {cmd}[/yellow]")

        return False

    def _run_registry_command(self, cmd_input: str) -> bool:
        """Run a command through the session's centralized command registry.

        This ensures REPL and web UI share the same command logic.
        Handles presentation of the result in Rich format.

        Args:
            cmd_input: The full command string (e.g., "/tables" or "/show orders")

        Returns:
            False (commands don't exit the REPL)
        """
        if not self.api.session:
            self.api = self._create_api()

        try:
            # Route through session which uses the command registry
            result = self.api.session.solve(cmd_input)

            if result.get("success") is False:
                self.console.print(f"[red]{result.get('output', 'Command failed')}[/red]")
            else:
                output = result.get("output", "")
                if output:
                    # Display markdown output using Rich
                    from rich.markdown import Markdown
                    self.console.print(Markdown(output))

        except Exception as e:
            self.console.print(f"[red]Error: {e}[/red]")

        return False

    def _solve(self, problem: str) -> Optional[str]:
        """Solve a problem.

        Returns:
            Optional command string if user entered a slash command during approval,
            None otherwise.
        """
        # Detect and apply display preference overrides from NL
        overrides = self.api.detect_display_overrides(problem)
        original_settings = self._apply_display_overrides(overrides)

        # Detect NL corrections and save as learnings
        nl_correction = self.api.detect_nl_correction(problem)
        if nl_correction.detected:
            self._save_nl_correction(nl_correction, problem)
            self.console.print(f"[dim]Noted: {nl_correction.correction_type.replace('_', ' ')}[/dim]")

        # Clear any pending outputs from previous execution
        clear_pending_outputs()

        self.last_problem = problem  # Track for /save
        self.suggestions = []  # Clear previous suggestions
        self.display.set_problem(problem)

        try:
            if self.api.session.session_id:
                result = self.api.session.follow_up(problem)
            else:
                result = self.api.session.solve(problem)

            # Check if a slash command was entered during approval
            if result.get("command"):
                return result["command"]

            if result.get("meta_response"):
                self.display.show_output(result.get("output", ""))
                # Store and show suggestions from meta responses (example questions)
                self.suggestions = result.get("suggestions", [])
                if self.suggestions:
                    self.display.show_suggestions(self.suggestions)
                self.display.show_summary(success=True, total_steps=0, duration_ms=0)
            elif result.get("mode") == Mode.PROOF.value and result.get("success", True):
                # AUDITABLE mode - display full output with answer, derivation, and insights
                # Note: success=True default since older code paths may not set it explicitly
                self.display.show_output(result.get("output", ""))
                tables = result.get("datastore_tables", [])
                self.display.show_tables(tables, force_show=False)
                self.suggestions = result.get("suggestions", [])
                if self.suggestions:
                    self.display.show_suggestions(self.suggestions)
                self.display.show_summary(success=True, total_steps=0, duration_ms=0)
            elif result.get("success"):
                tables = result.get("datastore_tables", [])
                total_duration = sum(r.duration_ms for r in result.get("results", []))
                self.display.show_tables(tables, duration_ms=total_duration)
                self.display.show_summary(
                    success=True,
                    total_steps=len(result.get("results", [])),
                    duration_ms=total_duration,
                )
                # Store suggestions for shortcut handling
                self.suggestions = result.get("suggestions", [])

                # Display any outputs (artifacts) from this execution
                self._display_outputs()

                # Check context size and warn if needed
                self._check_context_warning()
            elif result.get("mode") != Mode.PROOF.value:
                # Only show generic error for non-proof modes
                # Proof mode errors are already displayed via verification_error event
                self.console.print(f"[red]Error:[/red] {result.get('error', 'Unknown error')}")
        except KeyboardInterrupt:
            # Cancel execution via session's cancel_execution() method
            # This properly signals the execution context and emits events
            if self.api.session:
                self.api.session.cancel_execution()
            # Clean up display state on interrupt
            self.display.stop()
            self.display.stop_spinner()
            # Print interrupt message on new line
            self.console.print("\n[yellow]Interrupted.[/yellow]")
        except Exception as e:
            self.display.stop()
            self.display.stop_spinner()
            self.console.print(f"[red]Error:[/red] {e}")
        finally:
            # Restore single-turn display settings
            self._restore_display_settings(original_settings)

        return None

    def run(self, initial_problem: Optional[str] = None) -> None:
        """Run the interactive REPL."""
        # Set REPL mode for output formatting
        os.environ["CONSTAT_REPL_MODE"] = "1"
        try:
            # Enable persistent status bar at bottom of terminal
            self.display.enable_status_bar()
            self._run_repl_body(initial_problem)
        finally:
            # Disable status bar and restore terminal
            self.display.disable_status_bar()
            if self.api.session and self.api.session.datastore:
                self.api.session.datastore.close()

    def _run_repl_body(self, initial_problem: Optional[str] = None) -> None:
        """Run the REPL body (banner + loop)."""
        # Flush any pending Rich output before switching to prompt_toolkit
        sys.stdout.flush()
        sys.stderr.flush()

        # Auto-compact learnings on startup (after spinner stops, before banner)
        self._maybe_auto_compact()

        # Handle auto-resume before banner
        if self.auto_resume:
            self._handle_auto_resume()

        # Welcome banner with random personality adjectives (Rich output)
        reliable_adj, honest_adj = get_vera_adjectives()
        hints = "Tab completes commands | Ctrl+C interrupts"

        # Use Rich for banner output
        self.console.print()
        self.console.print(
            f"Hi, I'm [bold]Vera[/bold], your {reliable_adj} and {honest_adj} data analyst."
        )
        self.console.print(
            "[dim]I make every effort to tell the truth and fully explain my reasoning.[/dim]"
        )
        self.console.print()
        self.console.print(
            "[dim]Powered by[/dim] [blue bold]Constat[/blue bold] "
            "[dim](Latin: \"it is established\") — Multi-Step AI Reasoning Agent[/dim]"
        )
        self.console.print(
            f"[dim]Type /help for commands, or ask a question. | {hints}[/dim]"
        )

        # Show starter suggestions if no initial problem
        if not initial_problem:
            self.console.print()
            self.console.print("[dim]Try asking:[/dim]")
            starter_suggestions = [
                "What data is available?",
                "What can you help me with?",
                "How do you reason about problems?",
                "What makes you different, Vera?",
            ]
            for i, s in enumerate(starter_suggestions, 1):
                self.console.print(f"  [dim]{i}.[/dim] [cyan]{s}[/cyan]")
            self.suggestions = starter_suggestions
            self.console.print()

        if initial_problem:
            self._solve(initial_problem)

        self._loop_body()

    def _loop_body(self) -> None:
        """The actual REPL loop body."""
        while True:
            try:
                user_input = self._get_input()

                if not user_input:
                    continue

                # Check for suggestion shortcuts
                suggestion_to_run = None
                lower_input = user_input.lower().strip()

                if self.suggestions:
                    # Number shortcuts: "1", "2", "3", etc.
                    if lower_input.isdigit():
                        idx = int(lower_input) - 1
                        if 0 <= idx < len(self.suggestions):
                            suggestion_to_run = self.suggestions[idx]
                        else:
                            self.console.print(f"[yellow]No suggestion #{lower_input}[/yellow]")
                            continue
                    # Affirmative shortcuts: accept first/only suggestion
                    elif lower_input in ("ok", "yes", "sure", "y"):
                        suggestion_to_run = self.suggestions[0]

                if suggestion_to_run:
                    passthrough_cmd = self._solve(suggestion_to_run)
                    if passthrough_cmd and self._handle_command(passthrough_cmd):
                        break
                elif user_input.startswith("/"):
                    if self._handle_command(user_input):
                        break
                else:
                    passthrough_cmd = self._solve(user_input)
                    if passthrough_cmd and self._handle_command(passthrough_cmd):
                        break

            except KeyboardInterrupt:
                self.console.print("\n[dim]Type /quit to exit.[/dim]")
            except EOFError:
                break


def run_repl(config_path: str, verbose: bool = False, problem: Optional[str] = None) -> None:
    """Run the interactive REPL."""
    config = Config.from_yaml(config_path)
    repl = InteractiveREPL(config, verbose=verbose)
    repl.run(initial_problem=problem)
