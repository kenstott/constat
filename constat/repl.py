"""Interactive REPL for refinement loop."""

from typing import Optional
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax

from constat.session import Session, SessionConfig
from constat.core.config import Config
from constat.feedback import FeedbackDisplay, SessionFeedbackHandler


class InteractiveREPL:
    """
    Interactive Read-Eval-Print Loop for Constat sessions.

    Provides:
    - Initial problem solving
    - Follow-up questions with context preservation
    - Session state inspection
    - Table querying
    - Session history navigation
    """

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

    def _create_session(self) -> Session:
        """Create a new session with feedback handler."""
        session = Session(
            self.config,
            session_config=self.session_config,
            progress_callback=self.progress_callback,
        )

        # Wire up feedback display
        handler = SessionFeedbackHandler(self.display)
        session.on_event(handler.handle_event)

        return session

    def _show_help(self) -> None:
        """Show available commands."""
        table = Table(title="Commands", show_header=True, box=None)
        table.add_column("Command", style="cyan")
        table.add_column("Description")

        commands = [
            ("/help, /h", "Show this help message"),
            ("/tables", "List available tables from previous steps"),
            ("/query <sql>", "Run SQL query on datastore"),
            ("/state", "Show session state (variables, tables)"),
            ("/unresolved", "Show unresolved facts (auditable mode)"),
            ("/facts <text>", "Provide facts in natural language"),
            ("/history", "List previous sessions"),
            ("/resume <id>", "Resume a previous session"),
            ("/verbose", "Toggle verbose mode"),
            ("/quit, /exit, /q", "Exit the REPL"),
        ]

        for cmd, desc in commands:
            table.add_row(cmd, desc)

        self.console.print(table)
        self.console.print("\n[dim]Or type a question to analyze your data.[/dim]\n")

    def _show_tables(self) -> None:
        """Show tables in current session."""
        if not self.session or not self.session.datastore:
            self.console.print("[yellow]No active session.[/yellow]")
            return

        tables = self.session.datastore.list_tables()
        if not tables:
            self.console.print("[dim]No tables saved yet.[/dim]")
            return

        self.display.show_tables(tables)

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

        self.console.print(f"\n[bold]Session ID:[/bold] {state['session_id']}")

        # Show tables
        if state['datastore_tables']:
            self.console.print("\n[bold]Tables:[/bold]")
            for t in state['datastore_tables']:
                self.console.print(f"  - {t['name']} ({t['row_count']} rows)")

        # Show state variables
        if state['state']:
            self.console.print("\n[bold]State Variables:[/bold]")
            for key, value in state['state'].items():
                val_str = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
                self.console.print(f"  - {key}: {val_str}")

        # Show completed steps
        if state['completed_steps']:
            self.console.print(f"\n[bold]Completed Steps:[/bold] {state['completed_steps']}")

        self.console.print()

    def _show_unresolved(self) -> None:
        """Show unresolved facts from auditable mode."""
        if not self.session:
            self.console.print("[yellow]No active session.[/yellow]")
            return

        summary = self.session.get_unresolved_summary()
        self.console.print(summary)

    def _provide_facts(self, text: str) -> None:
        """Provide facts in natural language."""
        if not self.session:
            self.console.print("[yellow]No active session.[/yellow]")
            return

        result = self.session.provide_facts(text)

        if result["extracted_facts"]:
            self.console.print("\n[green]Extracted facts:[/green]")
            for fact in result["extracted_facts"]:
                self.console.print(f"  - {fact['name']} = {fact['value']}")
                if fact.get("reasoning"):
                    self.console.print(f"    [dim]({fact['reasoning']})[/dim]")
        else:
            self.console.print("[yellow]No facts could be extracted from your input.[/yellow]")

        if result["unresolved_remaining"]:
            self.console.print(f"\n[yellow]Still unresolved: {len(result['unresolved_remaining'])} facts[/yellow]")
            self.console.print("[dim]Use /unresolved to see details[/dim]")
        else:
            self.console.print("\n[green]All facts resolved![/green]")

    def _show_history(self) -> None:
        """Show recent session history."""
        if not self.session:
            # Create temporary session for history access
            self.session = self._create_session()

        sessions = self.session.history.list_sessions(limit=10)

        if not sessions:
            self.console.print("[dim]No previous sessions.[/dim]")
            return

        table = Table(title="Recent Sessions", show_header=True)
        table.add_column("ID", style="cyan")
        table.add_column("Created")
        table.add_column("Status")
        table.add_column("Queries")

        for s in sessions:
            table.add_row(
                s.session_id[:12] + "...",
                s.created_at[:16] if s.created_at else "",
                s.status,
                str(s.total_queries),
            )

        self.console.print(table)

    def _resume_session(self, session_id: str) -> None:
        """Resume a previous session."""
        if not self.session:
            self.session = self._create_session()

        # Try to find matching session
        sessions = self.session.history.list_sessions(limit=50)
        matching = [s for s in sessions if s.session_id.startswith(session_id)]

        if not matching:
            self.console.print(f"[red]Session not found:[/red] {session_id}")
            return

        if len(matching) > 1:
            self.console.print(f"[yellow]Multiple matches. Be more specific:[/yellow]")
            for s in matching[:5]:
                self.console.print(f"  - {s.session_id}")
            return

        full_id = matching[0].session_id
        if self.session.resume(full_id):
            self.console.print(f"[green]Resumed session:[/green] {full_id}")
            self._show_state()
        else:
            self.console.print(f"[red]Failed to resume session:[/red] {full_id}")

    def _solve(self, problem: str) -> None:
        """Solve a problem (new session or follow-up)."""
        if not self.session:
            self.session = self._create_session()

        self.display.set_problem(problem)

        try:
            if self.session.session_id:
                # Follow-up on existing session
                result = self.session.follow_up(problem)
            else:
                # New session
                result = self.session.solve(problem)

            # Show plan
            if result.get("plan"):
                plan = result["plan"]
                steps = [{"number": s.number, "goal": s.goal} for s in plan.steps]
                self.display.show_plan(steps)

            # Summary is shown via events, but show tables
            if result.get("success"):
                tables = result.get("datastore_tables", [])
                if tables:
                    self.display.show_tables(tables)

                self.display.show_summary(
                    success=True,
                    total_steps=len(result.get("results", [])),
                    duration_ms=sum(r.duration_ms for r in result.get("results", [])),
                )
            else:
                self.display.show_summary(
                    success=False,
                    total_steps=len(result.get("plan", {}).get("steps", [])) if result.get("plan") else 0,
                    duration_ms=0,
                )
                self.console.print(f"\n[red]Error:[/red] {result.get('error', 'Unknown error')}")

        except KeyboardInterrupt:
            self.console.print("\n[yellow]Interrupted.[/yellow]")
        except Exception as e:
            self.console.print(f"\n[red]Error:[/red] {e}")

    def run(self, initial_problem: Optional[str] = None) -> None:
        """
        Run the interactive REPL.

        Args:
            initial_problem: Optional problem to solve immediately
        """
        self.console.print(Panel.fit(
            "[bold blue]Constat[/bold blue] - Multi-Step AI Reasoning Engine\n"
            "[dim]Type /help for commands, or ask a question.[/dim]",
            border_style="blue",
        ))

        # Solve initial problem if provided
        if initial_problem:
            self._solve(initial_problem)

        # Main REPL loop
        while True:
            try:
                prompt_text = "[bold cyan]>[/bold cyan] " if self.session and self.session.session_id else "[bold blue]>[/bold blue] "
                user_input = Prompt.ask(prompt_text).strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith("/"):
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

                    elif cmd == "/query":
                        if arg:
                            self._run_query(arg)
                        else:
                            self.console.print("[yellow]Usage: /query <sql>[/yellow]")

                    elif cmd == "/state":
                        self._show_state()

                    elif cmd == "/history":
                        self._show_history()

                    elif cmd == "/resume":
                        if arg:
                            self._resume_session(arg)
                        else:
                            self.console.print("[yellow]Usage: /resume <session_id>[/yellow]")

                    elif cmd == "/unresolved":
                        self._show_unresolved()

                    elif cmd == "/facts":
                        if arg:
                            self._provide_facts(arg)
                        else:
                            self.console.print("[yellow]Usage: /facts <text with facts>[/yellow]")
                            self.console.print("[dim]Example: /facts There were 1 million people at the march[/dim]")

                    elif cmd == "/verbose":
                        self.verbose = not self.verbose
                        self.display.verbose = self.verbose
                        self.console.print(f"Verbose mode: [bold]{'on' if self.verbose else 'off'}[/bold]")

                    else:
                        self.console.print(f"[yellow]Unknown command: {cmd}[/yellow]")
                        self._show_help()

                else:
                    # Treat as a problem/question
                    self._solve(user_input)

            except KeyboardInterrupt:
                self.console.print("\n[dim]Type /quit to exit.[/dim]")
            except EOFError:
                break

        # Cleanup
        if self.session and self.session.datastore:
            self.session.datastore.close()


def run_repl(config_path: str, verbose: bool = False, problem: Optional[str] = None) -> None:
    """
    Run the interactive REPL.

    Args:
        config_path: Path to config YAML file
        verbose: Enable verbose output
        problem: Optional initial problem to solve
    """
    config = Config.from_yaml(config_path)
    repl = InteractiveREPL(config, verbose=verbose)
    repl.run(initial_problem=problem)
