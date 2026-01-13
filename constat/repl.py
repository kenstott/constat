"""Interactive REPL for refinement loop."""

import os
import sys
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from constat.session import Session, SessionConfig
from constat.core.config import Config
from constat.feedback import FeedbackDisplay, SessionFeedbackHandler


def _is_compatible_terminal() -> bool:
    """Check if terminal is compatible with prompt_toolkit."""
    # IntelliJ/PyCharm terminal sets TERMINAL_EMULATOR
    if "TERMINAL_EMULATOR" in os.environ:
        return False
    # JetBrains IDEs also set this
    if os.environ.get("TERM_PROGRAM") == "JetBrains-JediTerm":
        return False
    # Check if we have a real TTY
    if not sys.stdin.isatty():
        return False
    return True


try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.auto_suggest import AutoSuggest, Suggestion
    from prompt_toolkit.history import InMemoryHistory
    PROMPT_TOOLKIT_AVAILABLE = _is_compatible_terminal()

    class ConstatSuggester(AutoSuggest):
        """Schema-aware auto-suggester for the Constat REPL."""

        # Static suggestions for empty input
        DEFAULT_SUGGESTIONS = [
            "What questions can you answer for me?",
            "What data sources are available?",
            "Show me a summary of the data",
        ]

        # Question patterns that benefit from schema completion
        PATTERNS = [
            ("show me ", "the {table}"),
            ("what is the ", "{column} in {table}"),
            ("list all ", "{table}"),
            ("how many ", "records in {table}"),
            ("compare ", "{table} with "),
            ("summarize ", "{table}"),
            ("filter ", "{table} where "),
        ]

        def __init__(self, context_provider=None):
            """
            Args:
                context_provider: Callable that returns dict with 'tables', 'columns', 'plans'
            """
            self._context_provider = context_provider
            self._index = 0

        def _get_context(self):
            """Get current context from provider."""
            if self._context_provider:
                try:
                    return self._context_provider()
                except Exception:
                    pass
            return {"tables": [], "columns": [], "plans": []}

        def get_suggestion(self, buffer, document):
            text = document.text

            # Only suggest when input is manageable
            if len(text) > 50:
                return None

            if not text:
                # Empty input - show default suggestion
                return Suggestion(self.DEFAULT_SUGGESTIONS[self._index % len(self.DEFAULT_SUGGESTIONS)])

            text_lower = text.lower()

            # Check for command completions
            if text.startswith("/"):
                return self._suggest_command(text)

            # Check pattern-based completions
            context = self._get_context()
            tables = context.get("tables", [])

            # Try to complete based on patterns
            for pattern, completion_template in self.PATTERNS:
                if text_lower.startswith(pattern):
                    remaining = text[len(pattern):]
                    # Suggest table names
                    for table in tables:
                        if table.lower().startswith(remaining.lower()):
                            return Suggestion(table[len(remaining):])

            # Default: match against static suggestions
            all_suggestions = self.DEFAULT_SUGGESTIONS[:]

            # Add dynamic suggestions based on context
            for table in tables[:3]:
                all_suggestions.append(f"Show me the {table}")
                all_suggestions.append(f"Summarize {table}")

            for suggestion in all_suggestions:
                if suggestion.lower().startswith(text_lower):
                    return Suggestion(suggestion[len(text):])

            return None

        def _suggest_command(self, text):
            """Suggest command completions."""
            commands = [
                "/help", "/tables", "/show ", "/query ", "/code", "/state",
                "/update", "/user ", "/save ", "/share ", "/sharewith ",
                "/plans", "/replay ", "/history", "/resume ", "/verbose", "/quit"
            ]
            text_lower = text.lower()
            for cmd in commands:
                if cmd.startswith(text_lower) and cmd != text_lower:
                    return Suggestion(cmd[len(text):])
            return None

except ImportError:
    PROMPT_TOOLKIT_AVAILABLE = False


class InteractiveREPL:
    """Interactive Read-Eval-Print Loop for Constat sessions."""

    def __init__(
        self,
        config: Config,
        verbose: bool = False,
        console: Optional[Console] = None,
        progress_callback: Optional[callable] = None,
    ):
        self.config = config
        self.verbose = verbose
        self.console = console or Console()
        self.display = FeedbackDisplay(console=self.console, verbose=verbose)
        self.progress_callback = progress_callback
        self.session: Optional[Session] = None
        self.session_config = SessionConfig(verbose=verbose)
        self.user_id = "root"  # Default user
        self.last_problem = ""  # Track last problem for /save
        self.suggestions: list[str] = []  # Follow-up suggestions

        # Setup prompt_toolkit with schema-aware suggestions
        self._prompt_session = None
        if PROMPT_TOOLKIT_AVAILABLE:
            self._prompt_session = PromptSession(
                history=InMemoryHistory(),
                auto_suggest=ConstatSuggester(context_provider=self._get_suggestion_context),
            )

    def _get_suggestion_context(self) -> dict:
        """Provide context for typeahead suggestions."""
        context = {"tables": [], "columns": [], "plans": []}

        # Get table names from datastore
        if self.session and self.session.datastore:
            tables = self.session.datastore.list_tables()
            context["tables"] = [t["name"] for t in tables]

        # Get saved plan names
        try:
            plans = Session.list_saved_plans(user_id=self.user_id)
            context["plans"] = [p["name"] for p in plans]
        except Exception:
            pass

        return context

    def _create_session(self) -> Session:
        """Create a new session with feedback handler and callbacks."""
        session = Session(
            self.config,
            session_config=self.session_config,
            progress_callback=self.progress_callback,
        )
        handler = SessionFeedbackHandler(self.display)
        session.on_event(handler.handle_event)

        # Wire up approval callback
        session.set_approval_callback(self.display.request_plan_approval)

        # Wire up clarification callback
        session.set_clarification_callback(self.display.request_clarification)

        return session

    def _get_input(self) -> str:
        """Get user input."""
        if self._prompt_session:
            return self._prompt_session.prompt("> ").strip()
        else:
            return input("> ").strip()

    def _show_help(self) -> None:
        """Show available commands."""
        table = Table(title="Commands", show_header=True, box=None)
        table.add_column("Command", style="cyan")
        table.add_column("Description")
        commands = [
            ("/help, /h", "Show this help message"),
            ("/tables", "List available tables"),
            ("/show <table>", "Show table contents"),
            ("/query <sql>", "Run SQL query on datastore"),
            ("/code [step]", "Show generated code (all or specific step)"),
            ("/state", "Show session state"),
            ("/update", "Refresh database metadata cache"),
            ("/reset", "Clear session state and start fresh"),
            ("/user [name]", "Show or set current user"),
            ("/save <name>", "Save current plan for replay"),
            ("/share <name>", "Save plan as shared (all users)"),
            ("/sharewith <user> <name>", "Share plan with specific user"),
            ("/plans", "List saved plans"),
            ("/replay <name>", "Replay a saved plan"),
            ("/history", "List recent sessions"),
            ("/resume <id>", "Resume a previous session"),
            ("/context", "Show context size and token usage"),
            ("/compact", "Compact context to reduce token usage"),
            ("/verbose", "Toggle verbose mode"),
            ("/quit, /q", "Exit"),
        ]
        for cmd, desc in commands:
            table.add_row(cmd, desc)
        self.console.print(table)

    def _show_tables(self) -> None:
        """Show tables in current session."""
        if not self.session or not self.session.datastore:
            self.console.print("[yellow]No active session.[/yellow]")
            return
        tables = self.session.datastore.list_tables()
        if not tables:
            self.console.print("[dim]No tables yet.[/dim]")
            return
        self.display.show_tables(tables, force_show=True)

    def _run_query(self, sql: str) -> None:
        """Run SQL query on datastore."""
        if not self.session or not self.session.datastore:
            self.console.print("[yellow]No active session.[/yellow]")
            return
        try:
            result = self.session.datastore.query(sql)
            self.console.print(result.to_string())
        except Exception as e:
            self.console.print(f"[red]Query error:[/red] {e}")

    def _show_state(self) -> None:
        """Show current session state."""
        if not self.session:
            self.console.print("[yellow]No active session.[/yellow]")
            return
        state = self.session.get_state()
        self.console.print(f"\n[bold]Session:[/bold] {state['session_id']}")
        if state['datastore_tables']:
            self.console.print("[bold]Tables:[/bold]")
            for t in state['datastore_tables']:
                self.console.print(f"  - {t['name']} ({t['row_count']} rows)")

    def _refresh_metadata(self) -> None:
        """Refresh database metadata and clear caches."""
        if not self.session:
            self.console.print("[yellow]No active session.[/yellow]")
            return
        self.display.start_spinner("Refreshing metadata...")
        try:
            self.session.refresh_metadata()
            self.display.stop_spinner()
            self.console.print("[green]Metadata refreshed.[/green]")
        except Exception as e:
            self.display.stop_spinner()
            self.console.print(f"[red]Error refreshing metadata:[/red] {e}")

    def _reset_session(self) -> None:
        """Reset session state and start fresh."""
        # Reset display state
        self.display.reset()

        # Create a fresh session
        self.session = self._create_session()

        # Clear REPL state
        self.last_problem = ""
        self.suggestions = []

        self.console.print("[green]Session reset. State cleared.[/green]")

    def _show_context(self) -> None:
        """Show context size and token usage statistics."""
        if not self.session:
            self.console.print("[yellow]No active session.[/yellow]")
            return

        stats = self.session.get_context_stats()
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
        if not self.session:
            self.console.print("[yellow]No active session.[/yellow]")
            return

        # Show before stats
        stats_before = self.session.get_context_stats()
        if not stats_before:
            self.console.print("[yellow]No datastore available.[/yellow]")
            return

        self.console.print(f"[dim]Before compaction: ~{stats_before.total_tokens:,} tokens[/dim]")

        # Perform compaction
        self.display.start_spinner("Compacting context...")
        try:
            result = self.session.compact_context(
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

    def _show_code(self, step_arg: str = "") -> None:
        """Show generated code for steps."""
        if not self.session or not self.session.datastore:
            self.console.print("[yellow]No active session.[/yellow]")
            return

        entries = self.session.datastore.get_scratchpad()
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
        if not self.session or not self.session.datastore:
            self.console.print("[yellow]No active session.[/yellow]")
            return
        if not self.last_problem:
            self.console.print("[yellow]No problem executed yet.[/yellow]")
            return
        try:
            self.session.save_plan(name, self.last_problem, user_id=self.user_id, shared=shared)
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
        """Show recent session history."""
        if not self.session:
            self.session = self._create_session()

        sessions = self.session.history.list_sessions(limit=10)
        if not sessions:
            self.console.print("[dim]No session history.[/dim]")
            return

        table = Table(title="Recent Sessions", show_header=True, box=None)
        table.add_column("ID", style="cyan")
        table.add_column("Started")
        table.add_column("Queries", justify="right")
        table.add_column("Status")

        for s in sessions:
            # Shorten ID for display
            short_id = s.session_id[:20] + "..." if len(s.session_id) > 20 else s.session_id
            started = s.started_at.strftime("%Y-%m-%d %H:%M") if s.started_at else "?"
            table.add_row(short_id, started, str(s.query_count), s.status or "?")

        self.console.print(table)
        self.console.print("[dim]Use /resume <id> to continue a session[/dim]")

    def _resume_session(self, session_id: str) -> None:
        """Resume a previous session."""
        if not self.session:
            self.session = self._create_session()

        # Find matching session (partial ID match)
        sessions = self.session.history.list_sessions(limit=50)
        match = None
        for s in sessions:
            if s.session_id.startswith(session_id) or session_id in s.session_id:
                match = s.session_id
                break

        if not match:
            self.console.print(f"[red]Session not found: {session_id}[/red]")
            return

        if self.session.resume(match):
            self.console.print(f"[green]Resumed session:[/green] {match[:30]}...")
            # Show what's available
            tables = self.session.datastore.list_tables() if self.session.datastore else []
            if tables:
                self.console.print(f"[dim]{len(tables)} tables available - use /tables to view[/dim]")
        else:
            self.console.print(f"[red]Failed to resume session: {match}[/red]")

    def _replay_plan(self, name: str) -> None:
        """Replay a saved plan."""
        if not self.session:
            self.session = self._create_session()

        try:
            # Load the plan to get the problem
            plan_data = Session.load_saved_plan(name, user_id=self.user_id)
            self.last_problem = plan_data["problem"]
            self.display.set_problem(f"[Replay: {name}] {self.last_problem}")

            result = self.session.replay_saved(name, user_id=self.user_id)

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
        if not self.session:
            return

        stats = self.session.get_context_stats()
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

    def _solve(self, problem: str) -> None:
        """Solve a problem."""
        if not self.session:
            self.session = self._create_session()

        self.last_problem = problem  # Track for /save
        self.suggestions = []  # Clear previous suggestions
        self.display.set_problem(problem)

        try:
            if self.session.session_id:
                result = self.session.follow_up(problem)
            else:
                result = self.session.solve(problem)

            if result.get("meta_response"):
                self.display.show_output(result.get("output", ""))
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

                # Check context size and warn if needed
                self._check_context_warning()
            else:
                self.console.print(f"[red]Error:[/red] {result.get('error', 'Unknown error')}")
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Interrupted.[/yellow]")
        except Exception as e:
            self.console.print(f"[red]Error:[/red] {e}")

    def run(self, initial_problem: Optional[str] = None) -> None:
        """Run the interactive REPL."""
        # Welcome banner
        hints = "[dim]Tab[/dim] accepts suggestion | [dim]Ctrl+C[/dim] interrupts"
        self.console.print(Panel.fit(
            "[bold blue]Constat[/bold blue] - Multi-Step AI Reasoning Engine\n"
            f"[dim]Type /help for commands, or ask a question.[/dim]\n{hints}",
            border_style="blue",
        ))

        if initial_problem:
            self._solve(initial_problem)

        while True:
            try:
                user_input = self._get_input()

                if not user_input:
                    continue

                # Check for suggestion shortcuts
                suggestion_to_run = None
                lower_input = user_input.lower().strip()

                if self.suggestions:
                    # Number shortcuts: "1", "2", "3"
                    if lower_input in ("1", "2", "3"):
                        idx = int(lower_input) - 1
                        if idx < len(self.suggestions):
                            suggestion_to_run = self.suggestions[idx]
                        else:
                            self.console.print(f"[yellow]No suggestion #{lower_input}[/yellow]")
                            continue
                    # Affirmative shortcuts: accept first/only suggestion
                    elif lower_input in ("ok", "yes", "sure", "y"):
                        suggestion_to_run = self.suggestions[0]

                if suggestion_to_run:
                    self._solve(suggestion_to_run)
                elif user_input.startswith("/"):
                    parts = user_input.split(maxsplit=1)
                    cmd = parts[0].lower()
                    arg = parts[1] if len(parts) > 1 else ""

                    if cmd in ("/quit", "/exit", "/q"):
                        self.console.print("[dim]Goodbye![/dim]")
                        break
                    elif cmd in ("/help", "/h"):
                        self._show_help()
                    elif cmd == "/tables":
                        self._show_tables()
                    elif cmd == "/show" and arg:
                        self._run_query(f"SELECT * FROM {arg}")
                    elif cmd == "/query" and arg:
                        self._run_query(arg)
                    elif cmd == "/code":
                        self._show_code(arg)
                    elif cmd == "/state":
                        self._show_state()
                    elif cmd == "/update":
                        self._refresh_metadata()
                    elif cmd == "/reset":
                        self._reset_session()
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
                    elif cmd == "/history":
                        self._show_history()
                    elif cmd == "/resume" and arg:
                        self._resume_session(arg)
                    elif cmd == "/context":
                        self._show_context()
                    elif cmd == "/compact":
                        self._compact_context()
                    elif cmd == "/verbose":
                        self.verbose = not self.verbose
                        self.display.verbose = self.verbose
                        self.console.print(f"Verbose: [bold]{'on' if self.verbose else 'off'}[/bold]")
                    else:
                        self.console.print(f"[yellow]Unknown: {cmd}[/yellow]")
                else:
                    self._solve(user_input)

            except KeyboardInterrupt:
                self.console.print("\n[dim]Type /quit to exit.[/dim]")
            except EOFError:
                break

        if self.session and self.session.datastore:
            self.session.datastore.close()


def run_repl(config_path: str, verbose: bool = False, problem: Optional[str] = None) -> None:
    """Run the interactive REPL."""
    config = Config.from_yaml(config_path)
    repl = InteractiveREPL(config, verbose=verbose)
    repl.run(initial_problem=problem)
