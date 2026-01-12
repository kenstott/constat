"""Live feedback system for terminal output using rich."""

from dataclasses import dataclass, field
from typing import Optional, Callable
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID
from rich.prompt import Prompt
from rich.table import Table
from rich.syntax import Syntax
from rich.text import Text
from rich.markdown import Markdown
from rich.rule import Rule

from constat.execution.mode import (
    ExecutionMode,
    PlanApproval,
    PlanApprovalRequest,
    PlanApprovalResponse,
)


@dataclass
class StepDisplay:
    """Display state for a step."""
    number: int
    goal: str
    status: str = "pending"  # pending, running, generating, executing, completed, failed
    code: Optional[str] = None
    output: Optional[str] = None
    error: Optional[str] = None
    attempts: int = 0
    duration_ms: int = 0
    tables_created: list[str] = field(default_factory=list)


class FeedbackDisplay:
    """
    Rich-based terminal display for session execution.

    Provides real-time feedback including:
    - Plan overview with step checklist
    - Current step progress with spinner
    - Code syntax highlighting
    - Error display with retry indication
    - Timing information
    """

    def __init__(self, console: Optional[Console] = None, verbose: bool = False):
        self.console = console or Console()
        self.verbose = verbose
        self.plan_steps: list[StepDisplay] = []
        self.current_step: Optional[int] = None
        self.problem: str = ""
        self._live: Optional[Live] = None

    def start(self) -> None:
        """Start the live display."""
        self._live = Live(console=self.console, refresh_per_second=4)
        self._live.start()

    def stop(self) -> None:
        """Stop the live display."""
        if self._live:
            self._live.stop()
            self._live = None

    def set_problem(self, problem: str) -> None:
        """Set the problem being solved."""
        self.problem = problem
        self.console.print(Rule("[bold blue]CONSTAT[/bold blue]"))
        self.console.print(f"\n[bold]Problem:[/bold] {problem}\n")

    def show_plan(self, steps: list[dict]) -> None:
        """Display the execution plan."""
        self.plan_steps = [
            StepDisplay(number=s.get("number", i+1), goal=s.get("goal", ""))
            for i, s in enumerate(steps)
        ]

        self.console.print(Rule("[bold cyan]PLAN[/bold cyan]"))

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Num", style="dim", width=4)
        table.add_column("Goal")

        for step in self.plan_steps:
            table.add_row(f"{step.number}.", step.goal)

        self.console.print(table)
        self.console.print()

    def show_mode_selection(self, mode: ExecutionMode, reasoning: str) -> None:
        """Display the selected execution mode."""
        mode_style = "cyan" if mode == ExecutionMode.EXPLORATORY else "yellow"
        self.console.print(
            f"[bold]Mode:[/bold] [{mode_style}]{mode.value.upper()}[/{mode_style}]"
        )
        self.console.print(f"  [dim]{reasoning}[/dim]")
        self.console.print()

    def request_plan_approval(self, request: PlanApprovalRequest) -> PlanApprovalResponse:
        """
        Request user approval for a generated plan.

        Displays the plan with mode selection and prompts for approval.
        Returns user's decision with optional feedback.

        Args:
            request: PlanApprovalRequest with full context

        Returns:
            PlanApprovalResponse with user's decision
        """
        # Show mode selection
        self.show_mode_selection(request.mode, request.mode_reasoning)

        # Show the plan
        self.show_plan(request.steps)

        # Show reasoning if available
        if request.reasoning:
            self.console.print("[bold]Reasoning:[/bold]")
            self.console.print(Panel(request.reasoning, border_style="dim"))

        # Prompt for approval
        self.console.print(Rule("[bold]Approval Required[/bold]"))
        self.console.print(
            "[bold green][Y][/bold green]es - Execute this plan\n"
            "[bold red][N][/bold red]o  - Cancel and do not execute\n"
            "[bold yellow][S][/bold yellow]uggest - Provide feedback to improve the plan"
        )
        self.console.print()

        while True:
            choice = Prompt.ask(
                "Execute this plan?",
                choices=["y", "n", "s", "yes", "no", "suggest"],
                default="y",
            ).lower()

            if choice in ("y", "yes"):
                self.console.print("[green]Plan approved. Executing...[/green]\n")
                return PlanApprovalResponse.approve()

            elif choice in ("n", "no"):
                reason = Prompt.ask(
                    "[dim]Reason for rejection (optional)[/dim]",
                    default="",
                )
                self.console.print("[red]Plan rejected.[/red]\n")
                return PlanApprovalResponse.reject(reason if reason else None)

            elif choice in ("s", "suggest"):
                suggestion = Prompt.ask(
                    "[yellow]What changes would you suggest?[/yellow]"
                )
                if suggestion.strip():
                    self.console.print("[yellow]Incorporating feedback and replanning...[/yellow]\n")
                    return PlanApprovalResponse.suggest(suggestion)
                else:
                    self.console.print("[dim]No suggestion provided. Please try again.[/dim]")

    def show_replan_notice(self, attempt: int, max_attempts: int) -> None:
        """Show notice that we're replanning based on feedback."""
        self.console.print(
            f"[yellow]Replanning (attempt {attempt}/{max_attempts})...[/yellow]"
        )

    def step_start(self, step_number: int, goal: str) -> None:
        """Mark a step as starting."""
        self.current_step = step_number

        # Update step status
        for step in self.plan_steps:
            if step.number == step_number:
                step.status = "running"
                break

        self.console.print(Rule(f"[bold]Step {step_number}[/bold]: {goal}"))

    def step_generating(self, step_number: int, attempt: int) -> None:
        """Show code generation in progress."""
        if attempt == 1:
            self.console.print("  [dim]generating code...[/dim]", end="\r")
        else:
            self.console.print(f"  [yellow]retry #{attempt}[/yellow] generating code...", end="\r")

    def step_executing(self, step_number: int, attempt: int, code: Optional[str] = None) -> None:
        """Show code execution in progress."""
        self.console.print("  [dim]executing...[/dim]        ", end="\r")

        if self.verbose and code:
            self.console.print()
            self.console.print(Syntax(code, "python", theme="monokai", line_numbers=True))

    def step_complete(
        self,
        step_number: int,
        output: str,
        attempts: int,
        duration_ms: int,
        tables_created: Optional[list[str]] = None,
    ) -> None:
        """Mark a step as completed successfully."""
        for step in self.plan_steps:
            if step.number == step_number:
                step.status = "completed"
                step.output = output
                step.attempts = attempts
                step.duration_ms = duration_ms
                step.tables_created = tables_created or []
                break

        # Clear the progress line
        self.console.print(" " * 60, end="\r")

        # Show completion
        retry_info = f" [dim]({attempts} attempts)[/dim]" if attempts > 1 else ""
        time_info = f"[dim]{duration_ms/1000:.1f}s[/dim]"
        self.console.print(f"  [green]OK[/green] {time_info}{retry_info}")

        # Show tables created
        if tables_created:
            self.console.print(f"  [dim]tables:[/dim] {', '.join(tables_created)}")

        # Show output (truncated if long)
        if output:
            lines = output.strip().split("\n")
            if len(lines) > 5 and not self.verbose:
                preview = "\n".join(lines[:5])
                self.console.print(Panel(preview + "\n[dim]...[/dim]", border_style="dim"))
            else:
                self.console.print(Panel(output.strip(), border_style="dim"))

        self.console.print()

    def step_error(self, step_number: int, error: str, attempt: int) -> None:
        """Show a step error (before retry)."""
        self.console.print(" " * 60, end="\r")

        # Show brief error
        error_lines = error.strip().split("\n")
        brief = error_lines[-1] if error_lines else error
        self.console.print(f"  [red]Error:[/red] {brief[:80]}")

        if self.verbose:
            self.console.print(Panel(error, title="Full Error", border_style="red"))

    def step_failed(self, step_number: int, error: str, attempts: int) -> None:
        """Mark a step as permanently failed."""
        for step in self.plan_steps:
            if step.number == step_number:
                step.status = "failed"
                step.error = error
                step.attempts = attempts
                break

        self.console.print(" " * 60, end="\r")
        self.console.print(f"  [bold red]FAILED[/bold red] after {attempts} attempts")
        self.console.print(Panel(error, title="Error", border_style="red"))

    def show_summary(self, success: bool, total_steps: int, duration_ms: int) -> None:
        """Show final execution summary."""
        self.console.print(Rule())

        if success:
            self.console.print(
                f"[bold green]COMPLETE[/bold green] "
                f"({total_steps} steps, {duration_ms/1000:.1f}s total)"
            )
        else:
            completed = sum(1 for s in self.plan_steps if s.status == "completed")
            self.console.print(
                f"[bold red]FAILED[/bold red] "
                f"({completed}/{total_steps} steps completed)"
            )

    def show_tables(self, tables: list[dict]) -> None:
        """Show available tables in the datastore."""
        if not tables:
            return

        self.console.print("\n[bold]Available Tables:[/bold]")
        table = Table(show_header=True, box=None)
        table.add_column("Name", style="cyan")
        table.add_column("Rows", justify="right")
        table.add_column("From Step", justify="right")

        for t in tables:
            table.add_row(t["name"], str(t["row_count"]), str(t["step_number"]))

        self.console.print(table)

    def show_output(self, output: str) -> None:
        """Show final output."""
        self.console.print("\n[bold]Output:[/bold]")
        self.console.print(Markdown(output))


class SessionFeedbackHandler:
    """
    Event handler that bridges Session events to FeedbackDisplay.

    Usage:
        display = FeedbackDisplay(verbose=True)
        handler = SessionFeedbackHandler(display)
        session.on_event(handler.handle_event)
    """

    def __init__(self, display: FeedbackDisplay):
        self.display = display

    def handle_event(self, event) -> None:
        """Handle a StepEvent from Session."""
        event_type = event.event_type
        step_number = event.step_number
        data = event.data

        if event_type == "step_start":
            self.display.step_start(step_number, data.get("goal", ""))

        elif event_type == "generating":
            self.display.step_generating(step_number, data.get("attempt", 1))

        elif event_type == "executing":
            self.display.step_executing(
                step_number,
                data.get("attempt", 1),
                data.get("code"),
            )

        elif event_type == "step_complete":
            self.display.step_complete(
                step_number,
                data.get("stdout", ""),
                data.get("attempts", 1),
                data.get("duration_ms", 0),
                data.get("tables_created"),
            )

        elif event_type == "step_error":
            self.display.step_error(
                step_number,
                data.get("error", "Unknown error"),
                data.get("attempt", 1),
            )
