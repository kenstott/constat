"""Live feedback system for terminal output using rich."""

from dataclasses import dataclass, field
from typing import Optional, Callable
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID
from rich.prompt import Prompt
from rich.table import Table
from rich.syntax import Syntax
from rich.text import Text
from rich.markdown import Markdown
from rich.rule import Rule
import threading

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
    output_summary: Optional[str] = None  # Brief summary for display
    error: Optional[str] = None
    attempts: int = 0
    duration_ms: int = 0
    tables_created: list[str] = field(default_factory=list)
    status_message: str = ""  # Current status message for live display


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
        self._lock = threading.Lock()  # For thread-safe updates
        self._execution_started = False
        self._use_live_display = True  # Enable in-place updates
        self._spinner_progress: Optional[Progress] = None
        self._spinner_task: Optional[TaskID] = None

    def start(self) -> None:
        """Start the live display."""
        self._live = Live(console=self.console, refresh_per_second=4)
        self._live.start()

    def stop(self) -> None:
        """Stop the live display."""
        if self._live:
            self._live.stop()
            self._live = None

    def start_spinner(self, message: str) -> None:
        """Start an animated spinner with a message."""
        self.stop_spinner()  # Stop any existing spinner
        self._spinner_progress = Progress(
            SpinnerColumn("dots"),
            TextColumn("[cyan]{task.description}"),
            console=self.console,
            transient=True,  # Remove spinner when done
        )
        self._spinner_progress.start()
        self._spinner_task = self._spinner_progress.add_task(message, total=None)

    def update_spinner(self, message: str) -> None:
        """Update the spinner message."""
        if self._spinner_progress and self._spinner_task is not None:
            self._spinner_progress.update(self._spinner_task, description=message)

    def stop_spinner(self) -> None:
        """Stop the spinner."""
        if self._spinner_progress:
            self._spinner_progress.stop()
            self._spinner_progress = None
            self._spinner_task = None

    def show_discovery_start(self) -> None:
        """Show that schema/data discovery is starting."""
        self.start_spinner("Discovering available data sources...")

    def show_discovery_progress(self, source: str) -> None:
        """Update discovery progress with current source."""
        self.update_spinner(f"Discovering: {source}")

    def show_discovery_complete(self, sources_found: int) -> None:
        """Show discovery completed."""
        self.stop_spinner()
        self.console.print(f"  [dim]Found {sources_found} data source(s)[/dim]")

    def show_planning_start(self) -> None:
        """Show that planning is starting."""
        self.start_spinner("Planning analysis approach...")

    def show_planning_progress(self, stage: str) -> None:
        """Update planning progress."""
        self.update_spinner(f"Planning: {stage}")

    def show_planning_complete(self) -> None:
        """Show planning completed."""
        self.stop_spinner()

    def _build_steps_display(self) -> Group:
        """Build a renderable showing all steps' current status."""
        renderables = []

        for step in self.plan_steps:
            # Build status indicator and message
            if step.status == "pending":
                status_icon = "[dim]○[/dim]"
                status_text = f"[dim]{step.goal}[/dim]"
            elif step.status in ("running", "generating", "executing"):
                status_icon = "[yellow]●[/yellow]"
                msg = step.status_message or "working..."
                status_text = f"{step.goal}\n    [yellow]{msg}[/yellow]"
            elif step.status == "completed":
                status_icon = "[green]✓[/green]"
                time_info = f"[dim]{step.duration_ms/1000:.1f}s[/dim]"
                retry_info = f" [yellow]({step.attempts} attempts)[/yellow]" if step.attempts > 1 else ""
                if step.output_summary:
                    status_text = f"{step.goal}\n    [green]→[/green] {step.output_summary} {time_info}{retry_info}"
                else:
                    status_text = f"{step.goal} {time_info}{retry_info}"
            elif step.status == "failed":
                status_icon = "[red]✗[/red]"
                error_brief = step.error.split('\n')[-1][:60] if step.error else "Failed"
                status_text = f"{step.goal}\n    [red]{error_brief}[/red]"
            else:
                status_icon = "[dim]○[/dim]"
                status_text = step.goal

            renderables.append(Text.from_markup(f"  {status_icon} Step {step.number}: {status_text}"))

        return Group(*renderables)

    def _update_live(self) -> None:
        """Update the live display with current step states."""
        if self._live and self._use_live_display:
            with self._lock:
                self._live.update(self._build_steps_display())

    def _get_step(self, step_number: int) -> Optional[StepDisplay]:
        """Get a step by number."""
        for step in self.plan_steps:
            if step.number == step_number:
                return step
        return None

    def set_problem(self, problem: str) -> None:
        """Set the problem being solved."""
        self.problem = problem
        self.console.print(Rule("[bold blue]CONSTAT[/bold blue]", align="left"))
        self.console.print(f"\n[bold]Problem:[/bold] {problem}\n")

    def show_plan(self, steps: list[dict]) -> None:
        """Display the execution plan."""
        self.plan_steps = [
            StepDisplay(number=s.get("number", i+1), goal=s.get("goal", ""))
            for i, s in enumerate(steps)
        ]

        # Always show the plan so user knows what's coming
        self.console.print(Rule("[bold cyan]PLAN[/bold cyan]", align="left"))

        for i, s in enumerate(steps):
            step_num = s.get("number", i+1)
            goal = s.get("goal", "")
            depends_on = s.get("depends_on", [])

            # Format dependency info
            dep_str = ""
            if depends_on:
                dep_str = f" [dim](depends on {', '.join(str(d) for d in depends_on)})[/dim]"

            self.console.print(f"  [dim]{step_num}.[/dim] {goal}{dep_str}")

        self.console.print()

    def start_execution(self) -> None:
        """Start the live execution display."""
        self.console.print(Rule("[bold cyan]EXECUTING[/bold cyan]", align="left"))
        self._execution_started = True
        if self._use_live_display:
            self.start()
            self._update_live()

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
        self.console.print(Rule("[bold]Approval Required[/bold]", align="left"))
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
        step = self._get_step(step_number)
        if step:
            step.status = "running"
            step.status_message = "starting..."

        if self._live and self._use_live_display:
            self._update_live()
        else:
            # Fallback to direct printing
            total = len(self.plan_steps) if self.plan_steps else "?"
            self.console.print(f"\n[bold]Step {step_number}/{total}:[/bold] {goal}")

    def step_generating(self, step_number: int, attempt: int) -> None:
        """Show code generation in progress."""
        step = self._get_step(step_number)
        if step:
            step.status = "generating"
            step.status_message = f"retry #{attempt}..." if attempt > 1 else "working..."
            step.attempts = attempt

        if self._live and self._use_live_display:
            self._update_live()
        else:
            if attempt > 1:
                self.console.print(f"  [yellow]retry #{attempt}...[/yellow]", end="\r")
            else:
                self.console.print("  [dim]working...[/dim]", end="\r")

    def step_executing(self, step_number: int, attempt: int, code: Optional[str] = None) -> None:
        """Show code execution in progress."""
        step = self._get_step(step_number)
        if step:
            step.status = "executing"
            step.status_message = "executing..."
            step.code = code

        if self._live and self._use_live_display:
            self._update_live()
        elif self.verbose and code:
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
        # Build output summary
        output_summary = ""
        if output:
            lines = [l.strip() for l in output.strip().split("\n") if l.strip()]
            if lines:
                # Take first 2 lines max as summary for Live display
                summary_lines = lines[:2]
                output_summary = " | ".join(summary_lines)
                if len(output_summary) > 80:
                    output_summary = output_summary[:77] + "..."

        step = self._get_step(step_number)
        if step:
            step.status = "completed"
            step.output = output
            step.output_summary = output_summary
            step.attempts = attempts
            step.duration_ms = duration_ms
            step.tables_created = tables_created or []

        if self._live and self._use_live_display:
            self._update_live()
        else:
            # Fallback to direct printing
            self.console.print(" " * 60, end="\r")
            retry_info = f" [yellow]({attempts} attempts)[/yellow]" if attempts > 1 else ""
            time_info = f"[dim]{duration_ms/1000:.1f}s[/dim]"

            if output:
                lines = [l.strip() for l in output.strip().split("\n") if l.strip()]
                if lines:
                    summary_lines = lines[:3]
                    summary = "\n  ".join(summary_lines)
                    if len(lines) > 3:
                        summary += f"\n  [dim]... ({len(lines) - 3} more lines)[/dim]"
                    self.console.print(f"  [green]→[/green] {summary} {time_info}{retry_info}")
                else:
                    self.console.print(f"  [green]✓[/green] Done {time_info}{retry_info}")
            else:
                self.console.print(f"  [green]✓[/green] Done {time_info}{retry_info}")

        if self.verbose and tables_created:
            self.console.print(f"  [dim]tables:[/dim] {', '.join(tables_created)}")

    def step_error(self, step_number: int, error: str, attempt: int) -> None:
        """Show a step error (before retry)."""
        error_lines = error.strip().split("\n")
        brief = error_lines[-1] if error_lines else error

        step = self._get_step(step_number)
        if step:
            step.status_message = f"retry #{attempt}... ({brief[:40]})"

        if self._live and self._use_live_display:
            self._update_live()
        else:
            self.console.print(" " * 60, end="\r")
            self.console.print(f"  [yellow]retry...[/yellow]", end="\r")

            if self.verbose:
                self.console.print(f"  [red]Error:[/red] {brief[:80]}")
                self.console.print(Panel(error, title="Full Error", border_style="red"))

    def step_failed(self, step_number: int, error: str, attempts: int) -> None:
        """Mark a step as permanently failed."""
        step = self._get_step(step_number)
        if step:
            step.status = "failed"
            step.error = error
            step.attempts = attempts

        if self._live and self._use_live_display:
            self._update_live()
        else:
            self.console.print(" " * 60, end="\r")
            self.console.print(f"  [bold red]FAILED[/bold red] after {attempts} attempts")
            self.console.print(Panel(error, title="Error", border_style="red"))

    def show_summary(self, success: bool, total_steps: int, duration_ms: int) -> None:
        """Show final execution summary."""
        # Stop the Live display before printing summary
        self.stop()

        if not success:
            completed = sum(1 for s in self.plan_steps if s.status == "completed")
            self.console.print(
                f"\n[bold red]FAILED[/bold red] "
                f"({completed}/{total_steps} steps completed)"
            )
        # Success case: timing shown after tables hint

    def show_tables(self, tables: list[dict], duration_ms: int = 0) -> None:
        """Show available tables in the datastore."""
        if not tables:
            if duration_ms:
                self.console.print(f"\n[dim]({duration_ms/1000:.1f}s total)[/dim]")
            return

        if self.verbose:
            self.console.print("\n[bold]Available Tables:[/bold]")
            table = Table(show_header=True, box=None)
            table.add_column("Name", style="cyan")
            table.add_column("Rows", justify="right")
            table.add_column("From Step", justify="right")

            for t in tables:
                table.add_row(t["name"], str(t["row_count"]), str(t["step_number"]))

            self.console.print(table)
            if duration_ms:
                self.console.print(f"[dim]({duration_ms/1000:.1f}s total)[/dim]")
        else:
            # Compact: tables hint with timing
            time_str = f", {duration_ms/1000:.1f}s total" if duration_ms else ""
            self.console.print(f"\n[dim]({len(tables)} tables available - use /tables to view{time_str})[/dim]")

    def show_output(self, output: str) -> None:
        """Show final output."""
        self.console.print("\n[bold]Output:[/bold]")
        self.console.print(Markdown(output, justify="left"))

    def show_final_answer(self, answer: str) -> None:
        """Show the final synthesized answer prominently."""
        self.console.print()
        self.console.print(Rule("[bold green]ANSWER[/bold green]", align="left"))
        self.console.print(Markdown(answer, justify="left"))
        self.console.print()


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
        self._execution_started = False

    def handle_event(self, event) -> None:
        """Handle a StepEvent from Session."""
        event_type = event.event_type
        step_number = event.step_number
        data = event.data

        # Discovery events
        if event_type == "discovery_start":
            self.display.show_discovery_start()

        elif event_type == "discovery_progress":
            self.display.show_discovery_progress(data.get("source", ""))

        elif event_type == "discovery_complete":
            self.display.show_discovery_complete(data.get("sources_found", 0))

        # Planning events
        elif event_type == "planning_start":
            self.display.show_planning_start()

        elif event_type == "planning_progress":
            self.display.show_planning_progress(data.get("stage", ""))

        elif event_type == "planning_complete":
            self.display.show_planning_complete()

        elif event_type == "plan_ready":
            # Show plan BEFORE execution starts
            self.display.show_plan(data.get("steps", []))
            if data.get("reasoning") and self.display.verbose:
                self.display.console.print(f"[dim]Reasoning: {data['reasoning']}[/dim]\n")

        elif event_type == "step_start":
            # Start execution display on first step
            if not self._execution_started:
                self._execution_started = True
                self.display.start_execution()
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

        elif event_type == "synthesizing":
            # Stop Live display before printing synthesizing message
            self.display.stop()
            self.display.console.print(f"\n[dim]{data.get('message', 'Synthesizing...')}[/dim]")

        elif event_type == "answer_ready":
            self.display.show_final_answer(data.get("answer", ""))
