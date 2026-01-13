"""Live feedback system for terminal output using rich."""

import re
import time
from dataclasses import dataclass, field
from typing import Optional, Callable
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID
from rich.prompt import Prompt
from rich.table import Table
from rich.syntax import Syntax
from rich.text import Text
from rich.markdown import Markdown
from rich.rule import Rule
from rich.layout import Layout
from rich.columns import Columns
import threading


def _left_align_markdown(text: str) -> str:
    """Convert Markdown headers to bold text to avoid Rich's centering."""
    # Convert ## Header to **Header**
    text = re.sub(r'^#{1,6}\s+(.+)$', r'**\1**', text, flags=re.MULTILINE)
    return text

from constat.execution.mode import (
    ExecutionMode,
    PlanApproval,
    PlanApprovalRequest,
    PlanApprovalResponse,
)
from constat.session import ClarificationRequest, ClarificationResponse, ClarificationQuestion


# Spinner frames for animation
SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


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
    - Plan overview with step checklist (pinned at bottom)
    - Current step progress with spinner
    - Output streaming above pinned plan
    - Real-time elapsed timer
    - Code syntax highlighting
    - Error display with retry indication
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
        self._spinner_frame: int = 0  # For step execution animation
        self._step_number_map: dict[int, int] = {}  # Maps session step numbers to display numbers (1-indexed)

        # Animated display state
        self._start_time: Optional[float] = None
        self._output_lines: list[str] = []  # Output buffer for streaming above plan
        self._max_output_lines: int = 15  # Max lines to show above plan
        self._current_step_output: str = ""  # Current step's streaming output
        self._active_step_goal: str = ""  # Goal of currently executing step

    def start(self) -> None:
        """Start the live display."""
        # Create a wrapper that implements __rich__() so Live calls our builder on each refresh
        class AnimatedDisplayWrapper:
            def __init__(wrapper_self, display: "FeedbackDisplay"):
                wrapper_self._display = display

            def __rich__(wrapper_self) -> RenderableType:
                return wrapper_self._display._build_animated_display()

        self._display_wrapper = AnimatedDisplayWrapper(self)
        self._live = Live(
            self._display_wrapper,
            console=self.console,
            refresh_per_second=10,  # Faster for smoother animation
            transient=False,
        )
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

    def show_progress(self, message: str) -> None:
        """Show a generic progress message with spinner."""
        if self._spinner_progress:
            # Update existing spinner
            self.update_spinner(message)
        else:
            # Start new spinner
            self.start_spinner(message)

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

        # Advance spinner frame for animation
        self._spinner_frame = (self._spinner_frame + 1) % len(SPINNER_FRAMES)
        spinner_char = SPINNER_FRAMES[self._spinner_frame]

        for step in self.plan_steps:
            # Build status indicator and message
            if step.status == "pending":
                status_icon = "[dim]○[/dim]"
                status_text = f"[dim]{step.goal}[/dim]"
            elif step.status in ("running", "generating", "executing"):
                # Animated spinner for running steps
                status_icon = f"[yellow]{spinner_char}[/yellow]"
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

            display_num = self._step_number_map.get(step.number, step.number)
            renderables.append(Text.from_markup(f"  {status_icon} Step {display_num}: {status_text}"))

        return Group(*renderables)

    def _format_elapsed(self) -> str:
        """Format elapsed time as human-readable string."""
        if not self._start_time:
            return "0s"
        elapsed = time.time() - self._start_time
        if elapsed < 60:
            return f"{elapsed:.1f}s"
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        return f"{minutes}m {seconds}s"

    def _build_animated_display(self) -> RenderableType:
        """Build the animated display with output above and plan pinned below.

        Layout:
        ┌─────────────────────────────────────────┐
        │  Output from current step               │
        │  (streaming text, code, results)        │
        ├─────────────────────────────────────────┤
        │  · Step 1: Goal here...     (elapsed)   │
        │  ☐ Step 1: Goal description             │
        │  ⋯ Step 2: Currently running...         │
        │  ☐ Step 3: Pending...                   │
        │  ☑ Step 4: Completed 2.3s               │
        └─────────────────────────────────────────┘
        """
        # Advance spinner frame
        self._spinner_frame = (self._spinner_frame + 1) % len(SPINNER_FRAMES)
        spinner_char = SPINNER_FRAMES[self._spinner_frame]

        # Build output section (top)
        output_parts = []

        # Show current step's streaming output
        if self._current_step_output:
            # Truncate to max lines
            lines = self._current_step_output.split('\n')
            if len(lines) > self._max_output_lines:
                lines = lines[-self._max_output_lines:]
            output_text = '\n'.join(lines)
            output_parts.append(Text(output_text, style="dim"))

        # Build plan section (bottom) - todo-list style
        plan_parts = []

        # Header with elapsed time and active step
        elapsed = self._format_elapsed()
        if self._active_step_goal:
            header_text = f"[dim]·[/dim] [cyan]{self._active_step_goal[:50]}{'...' if len(self._active_step_goal) > 50 else ''}[/cyan] [dim]({elapsed})[/dim]"
        else:
            header_text = f"[dim]({elapsed})[/dim]"
        plan_parts.append(Text.from_markup(header_text))

        # Steps as checklist
        for step in self.plan_steps:
            display_num = self._step_number_map.get(step.number, step.number)

            if step.status == "pending":
                marker = "[dim]☐[/dim]"
                goal_style = "dim"
                suffix = ""
            elif step.status in ("running", "generating", "executing"):
                marker = f"[yellow]{spinner_char}[/yellow]"
                goal_style = "cyan"
                suffix = f" [yellow]{step.status_message or 'working...'}[/yellow]"
            elif step.status == "completed":
                marker = "[green]☑[/green]"
                goal_style = ""
                time_str = f"{step.duration_ms/1000:.1f}s"
                retry_str = f" ({step.attempts} tries)" if step.attempts > 1 else ""
                suffix = f" [dim]{time_str}{retry_str}[/dim]"
            elif step.status == "failed":
                marker = "[red]☒[/red]"
                goal_style = "red"
                suffix = ""
            else:
                marker = "[dim]☐[/dim]"
                goal_style = "dim"
                suffix = ""

            goal_text = step.goal[:60] + "..." if len(step.goal) > 60 else step.goal
            if goal_style:
                plan_parts.append(Text.from_markup(f"  {marker} [{goal_style}]{goal_text}[/{goal_style}]{suffix}"))
            else:
                plan_parts.append(Text.from_markup(f"  {marker} {goal_text}{suffix}"))

        # Combine: output on top, then separator, then plan
        all_parts = []
        if output_parts:
            all_parts.extend(output_parts)
            all_parts.append(Text(""))  # Spacer

        all_parts.extend(plan_parts)

        return Group(*all_parts)

    def _update_live(self) -> None:
        """Update the live display with current step states."""
        if self._live and self._use_live_display:
            with self._lock:
                self._live.update(self._build_animated_display())

    def _get_step(self, step_number: int) -> Optional[StepDisplay]:
        """Get a step by number."""
        for step in self.plan_steps:
            if step.number == step_number:
                return step
        return None

    def reset(self) -> None:
        """Reset all display state for a fresh start."""
        self.plan_steps = []
        self.current_step = None
        self.problem = ""
        self._step_number_map = {}
        self._execution_started = False
        self._start_time = None
        self._output_lines = []
        self._current_step_output = ""
        self._active_step_goal = ""
        self.stop()

    def set_problem(self, problem: str) -> None:
        """Set the problem being solved."""
        self.problem = problem
        self.console.print(Rule("[bold blue]CONSTAT[/bold blue]", align="left"))
        self.console.print()  # Just a blank line before plan

    def show_plan(self, steps: list[dict], is_followup: bool = False) -> None:
        """Display the execution plan.

        Args:
            steps: List of step dicts with number, goal, depends_on
            is_followup: If True, continue step numbering from previous plan
        """
        # Determine starting step number
        if is_followup and self.plan_steps:
            # Continue from last step number
            start_num = max(self._step_number_map.values()) + 1 if self._step_number_map else 1
        else:
            # New problem - start from 1
            start_num = 1
            self._step_number_map = {}
            self.plan_steps = []

        # Build mapping from session step numbers to display numbers
        for i, s in enumerate(steps):
            session_num = s.get("number", i + 1)
            self._step_number_map[session_num] = start_num + i

        # Append new steps (for follow-up) or replace (for new problem)
        new_steps = [
            StepDisplay(number=s.get("number", i+1), goal=s.get("goal", ""))
            for i, s in enumerate(steps)
        ]
        if is_followup:
            self.plan_steps.extend(new_steps)
        else:
            self.plan_steps = new_steps

        # Always show the plan so user knows what's coming
        self.console.print(Rule("[bold cyan]PLAN[/bold cyan]", align="left"))

        for i, s in enumerate(steps):
            display_num = self._step_number_map.get(s.get("number", i + 1), start_num + i)
            goal = s.get("goal", "")
            depends_on = s.get("depends_on", [])

            # Format dependency info with remapped step numbers
            dep_str = ""
            if depends_on:
                remapped_deps = [str(self._step_number_map.get(d, d)) for d in depends_on]
                dep_str = f" [dim](depends on {', '.join(remapped_deps)})[/dim]"

            self.console.print(f"  [dim]{display_num}.[/dim] {goal}{dep_str}")

        self.console.print()

    def start_execution(self) -> None:
        """Start the live execution display."""
        self.console.print(Rule("[bold cyan]EXECUTING[/bold cyan]", align="left"))
        self._execution_started = True
        self._start_time = time.time()  # Start timing
        self._current_step_output = ""  # Clear output buffer
        self._active_step_goal = ""
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

    def request_clarification(self, request: ClarificationRequest) -> ClarificationResponse:
        """
        Request clarification from user for ambiguous questions.

        Displays questions with numbered suggestions. Users can:
        - Enter a number to select a suggestion
        - Type a custom answer
        - Press Enter to skip

        Args:
            request: ClarificationRequest with questions

        Returns:
            ClarificationResponse with user's answers
        """
        # Stop any running animation before prompting for input
        self.stop_spinner()

        self.console.print()
        self.console.print(Rule("[bold cyan]Clarification Needed[/bold cyan]", align="left"))

        if request.ambiguity_reason:
            self.console.print(f"[dim]{request.ambiguity_reason}[/dim]")
            self.console.print()

        self.console.print("[bold]Please clarify[/bold] [dim](enter number, custom text, or press Enter to skip):[/dim]")
        self.console.print()

        answers = {}
        for i, question in enumerate(request.questions, 1):
            # Handle both old format (str) and new format (ClarificationQuestion)
            if isinstance(question, ClarificationQuestion):
                question_text = question.text
                suggestions = question.suggestions
            else:
                question_text = str(question)
                suggestions = []

            self.console.print(f"  [cyan]{i}.[/cyan] {question_text}")

            # Show numbered suggestions if available
            if suggestions:
                for j, suggestion in enumerate(suggestions, 1):
                    self.console.print(f"      [dim]{j})[/dim] [yellow]{suggestion}[/yellow]")

            # Get answer - show first suggestion as default hint if available
            if suggestions:
                default_hint = suggestions[0]
                answer = Prompt.ask(f"     [dim]>[/dim]", default=default_hint)
            else:
                answer = Prompt.ask("     [dim]>[/dim]", default="", show_default=False)
            answer = answer.strip()

            # Process the answer
            if answer.lower() == "skip" or not answer:
                # When skipping with no suggestions, use empty
                answers[question_text] = ""
                self.console.print(f"     [dim]Skipped (will use defaults)[/dim]")
            elif suggestions and answer == suggestions[0]:
                # User accepted the default suggestion
                answers[question_text] = answer
                self.console.print(f"     [green]Using: {answer}[/green]")
            elif answer.isdigit() and suggestions:
                # User selected a suggestion by number
                idx = int(answer) - 1
                if 0 <= idx < len(suggestions):
                    answers[question_text] = suggestions[idx]
                    self.console.print(f"     [green]Selected: {suggestions[idx]}[/green]")
                else:
                    # Invalid number, treat as custom input
                    answers[question_text] = answer
            else:
                answers[question_text] = answer

            self.console.print()

        # Check if any answers were provided
        has_answers = any(a for a in answers.values())

        # Check if user wants to skip all clarifications
        if has_answers:
            skip = Prompt.ask(
                "[dim]Press Enter to continue, or 's' to skip all[/dim]",
                default="",
                show_default=False
            ).lower()
        else:
            skip = Prompt.ask(
                "[dim]No answers provided. Press Enter to proceed anyway, or 's' to cancel[/dim]",
                default="",
                show_default=False
            ).lower()

        if skip == "s":
            self.console.print("[dim]Skipping, proceeding with original question...[/dim]")
            return ClarificationResponse(answers={}, skip=True)

        # Filter out empty answers
        non_empty_answers = {q: a for q, a in answers.items() if a}

        if non_empty_answers:
            self.console.print("[green]Proceeding with clarified question...[/green]\n")
        else:
            self.console.print("[dim]Proceeding with original question...[/dim]\n")

        return ClarificationResponse(answers=non_empty_answers, skip=False)

    def show_replan_notice(self, attempt: int, max_attempts: int) -> None:
        """Show notice that we're replanning based on feedback."""
        self.console.print(
            f"[yellow]Replanning (attempt {attempt}/{max_attempts})...[/yellow]"
        )

    def step_start(self, step_number: int, goal: str) -> None:
        """Mark a step as starting."""
        self.current_step = step_number
        self._active_step_goal = goal  # For animated header
        self._current_step_output = ""  # Clear previous step output

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
            display_num = self._step_number_map.get(step_number, step_number)
            self.console.print(f"\n[bold]Step {display_num}/{total}:[/bold] {goal}")

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
        # Update streaming output for animated display
        if output:
            self._current_step_output = output

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

        # Clear active step goal after completion
        self._active_step_goal = ""

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

    def show_tables(self, tables: list[dict], duration_ms: int = 0, force_show: bool = False) -> None:
        """Show available tables in the datastore."""
        if not tables:
            if duration_ms:
                self.console.print(f"\n[dim]({duration_ms/1000:.1f}s total)[/dim]")
            return

        if self.verbose or force_show:
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
        # Stop any running spinner first
        self.stop_spinner()
        self.console.print("\n[bold]Output:[/bold]")
        self.console.print(Markdown(_left_align_markdown(output)))

    def show_final_answer(self, answer: str) -> None:
        """Show the final synthesized answer prominently."""
        self.console.print()
        self.console.print(Rule("[bold green]ANSWER[/bold green]", align="left"))
        self.console.print(Markdown(_left_align_markdown(answer)))
        self.console.print()

    def show_suggestions(self, suggestions: list[str]) -> None:
        """Show follow-up suggestions."""
        if not suggestions:
            return

        if len(suggestions) == 1:
            self.console.print(f"[dim]Suggestion:[/dim] [cyan]{suggestions[0]}[/cyan]")
        else:
            self.console.print("[dim]Suggestions:[/dim]")
            for i, s in enumerate(suggestions, 1):
                self.console.print(f"  [dim]{i}.[/dim] [cyan]{s}[/cyan]")

    def show_facts_extracted(self, facts: list[dict], source: str) -> None:
        """Show facts that were extracted and cached.

        Args:
            facts: List of fact dicts with 'name' and 'value' keys
            source: Where facts came from ('question' or 'response')
        """
        if not facts:
            return

        # Only show response-derived facts (question facts are implicit)
        if source == "response":
            fact_strs = [f"[cyan]{f['name']}[/cyan]={f['value']}" for f in facts[:5]]
            self.console.print(f"[dim]Remembered: {', '.join(fact_strs)}[/dim]")


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

        # Generic progress events (used for early-stage operations)
        if event_type == "progress":
            self.display.show_progress(data.get("message", "Processing..."))

        # Discovery events
        elif event_type == "discovery_start":
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
            is_followup = data.get("is_followup", False)
            self.display.show_plan(data.get("steps", []), is_followup=is_followup)
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

        elif event_type == "suggestions_ready":
            self.display.show_suggestions(data.get("suggestions", []))

        elif event_type == "facts_extracted":
            facts = data.get("facts", [])
            source = data.get("source", "unknown")
            if facts:
                self.display.show_facts_extracted(facts, source)
