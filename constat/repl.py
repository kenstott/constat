"""Interactive REPL for refinement loop."""

from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from constat.session import Session, SessionConfig
from constat.core.config import Config
from constat.feedback import FeedbackDisplay, SessionFeedbackHandler

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.auto_suggest import AutoSuggest, Suggestion
    from prompt_toolkit.history import InMemoryHistory
    PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:
    PROMPT_TOOLKIT_AVAILABLE = False


class ConstatSuggester(AutoSuggest):
    """Auto-suggester for the Constat REPL."""

    SUGGESTIONS = [
        "What questions can you answer for me?",
        "What data sources are available?",
        "Show me a summary of the data",
        "What tables can I query?",
    ]

    def __init__(self):
        self._index = 0

    def get_suggestion(self, buffer, document):
        text = document.text

        # Only suggest when empty or short input
        if len(text) > 30:
            return None

        if not text:
            # Empty input - show first suggestion
            return Suggestion(self.SUGGESTIONS[self._index])

        # Find matching suggestion
        text_lower = text.lower()
        for suggestion in self.SUGGESTIONS:
            if suggestion.lower().startswith(text_lower):
                return Suggestion(suggestion[len(text):])

        return None


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

        # Setup prompt_toolkit
        self._prompt_session = None
        if PROMPT_TOOLKIT_AVAILABLE:
            self._prompt_session = PromptSession(
                history=InMemoryHistory(),
                auto_suggest=ConstatSuggester(),
            )

    def _create_session(self) -> Session:
        """Create a new session with feedback handler."""
        session = Session(
            self.config,
            session_config=self.session_config,
            progress_callback=self.progress_callback,
        )
        handler = SessionFeedbackHandler(self.display)
        session.on_event(handler.handle_event)
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
            ("/query <sql>", "Run SQL query on datastore"),
            ("/state", "Show session state"),
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
        self.console.print(f"\n[bold]Session:[/bold] {state['session_id']}")
        if state['datastore_tables']:
            self.console.print("[bold]Tables:[/bold]")
            for t in state['datastore_tables']:
                self.console.print(f"  - {t['name']} ({t['row_count']} rows)")

    def _solve(self, problem: str) -> None:
        """Solve a problem."""
        if not self.session:
            self.session = self._create_session()

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
                    elif cmd == "/query" and arg:
                        self._run_query(arg)
                    elif cmd == "/state":
                        self._show_state()
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
