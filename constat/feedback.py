"""Live feedback system for terminal output using rich."""

from __future__ import annotations

import re
import sys
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
from rich.tree import Tree
import threading

from constat.proof_tree import ProofTree, NodeStatus


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
from constat.keywords import detect_mode_switch as _detect_mode_switch_str


def detect_mode_switch(text: str) -> ExecutionMode | None:
    """Detect if the user is requesting a mode switch.

    Args:
        text: User input text

    Returns:
        Target ExecutionMode if a switch is requested, None otherwise
    """
    mode_name = _detect_mode_switch_str(text)
    if mode_name is None:
        return None

    # Map mode name string to ExecutionMode enum
    mode_map = {
        "knowledge": ExecutionMode.KNOWLEDGE,
        "auditable": ExecutionMode.AUDITABLE,
        "exploratory": ExecutionMode.EXPLORATORY,
    }
    return mode_map.get(mode_name)
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


@dataclass
class PlanItem:
    """A single item (premise or inference) in the execution plan."""
    fact_id: str  # P1, P2, I1, I2, etc.
    name: str
    item_type: str  # "premise" or "inference"
    status: str = "pending"  # pending, running, resolved, failed, blocked
    value: Optional[str] = None
    error: Optional[str] = None
    confidence: float = 0.0
    dependencies: list[str] = field(default_factory=list)  # List of fact_ids this depends on


class LivePlanExecutionDisplay:
    """
    Live-updating display for plan execution.

    Shows all premises and inferences upfront with status indicators:
    - ○ pending
    - ⠋ running (animated spinner)
    - ✓ resolved
    - ✗ failed
    """

    def __init__(self, console: Console, premises: list[dict], inferences: list[dict]):
        self.console = console
        self.premises = premises
        self.inferences = inferences
        self._lock = threading.Lock()
        self._live: Optional[Live] = None
        self._spinner_frame: int = 0
        self._animation_thread: Optional[threading.Thread] = None
        self._animation_running: bool = False

        # Initialize plan items
        self.items: dict[str, PlanItem] = {}
        for p in premises:
            fact_id = p.get("id", "")
            self.items[fact_id] = PlanItem(
                fact_id=fact_id,
                name=p.get("name", fact_id),
                item_type="premise",
            )
        for inf in inferences:
            fact_id = inf.get("id", "")
            # Extract dependencies from operation (P1, P2, I1, etc.)
            operation = inf.get("operation", "")
            deps = re.findall(r'\b([PI]\d+)\b', operation)
            self.items[fact_id] = PlanItem(
                fact_id=fact_id,
                name=inf.get("name", "") or fact_id,
                item_type="inference",
                dependencies=deps,
            )

    def start(self) -> None:
        """Start the live display."""
        self._live = Live(
            self._build_display(),
            console=self.console,
            refresh_per_second=10,
            transient=False,
        )
        self._live.start()
        self._start_animation()

    def stop(self) -> None:
        """Stop the live display, ensuring cleanup even on error."""
        self._stop_animation()
        if self._live:
            try:
                self._live.stop()
            except Exception:
                pass  # Ensure we don't fail during cleanup
            finally:
                self._live = None

    def _start_animation(self) -> None:
        """Start background animation thread."""
        self._animation_running = True
        self._animation_thread = threading.Thread(
            target=self._animation_loop,
            daemon=True,
        )
        self._animation_thread.start()

    def _stop_animation(self) -> None:
        """Stop animation thread."""
        self._animation_running = False
        if self._animation_thread:
            self._animation_thread.join(timeout=0.5)
            self._animation_thread = None

    def _animation_loop(self) -> None:
        """Background loop for spinner animation."""
        while self._animation_running:
            with self._lock:
                self._spinner_frame = (self._spinner_frame + 1) % len(SPINNER_FRAMES)
                if self._live:
                    try:
                        self._live.update(self._build_display())
                    except Exception:
                        pass
            time.sleep(0.1)

    def update_status(self, fact_id: str, status: str, value: str = None,
                      error: str = None, confidence: float = 0.0) -> None:
        """Update the status of a plan item."""
        with self._lock:
            matched_fact_id = None
            if fact_id in self.items:
                item = self.items[fact_id]
                item.status = status
                if value is not None:
                    item.value = value
                if error is not None:
                    item.error = error
                item.confidence = confidence
                matched_fact_id = fact_id
            # Also check by name (for events that use names)
            else:
                for item in self.items.values():
                    if item.name == fact_id or fact_id in item.name:
                        item.status = status
                        if value is not None:
                            item.value = value
                        if error is not None:
                            item.error = error
                        item.confidence = confidence
                        matched_fact_id = item.fact_id
                        break

            # If a node failed or blocked, cascade to all dependent nodes
            if status in ("failed", "blocked") and matched_fact_id:
                failed_name = self.items[matched_fact_id].name if matched_fact_id in self.items else matched_fact_id
                self._cascade_blocked(matched_fact_id, failed_name)

    def _cascade_blocked(self, blocked_id: str, root_cause: str) -> None:
        """Recursively mark all nodes depending on blocked_id as blocked."""
        newly_blocked = []
        for item in self.items.values():
            if item.status == "pending" and blocked_id in item.dependencies:
                item.status = "blocked"
                item.error = f"blocked: {root_cause} failed"
                newly_blocked.append(item.fact_id)
        # Recursively block dependents of newly blocked nodes
        for fact_id in newly_blocked:
            self._cascade_blocked(fact_id, root_cause)

    def _get_status_indicator(self, item: PlanItem) -> str:
        """Get the status indicator for an item."""
        if item.status == "pending":
            return "[dim]○[/dim]"
        elif item.status == "running":
            return f"[cyan]{SPINNER_FRAMES[self._spinner_frame]}[/cyan]"
        elif item.status == "resolved":
            return "[green]✓[/green]"
        elif item.status == "failed":
            return "[red]✗[/red]"
        elif item.status == "blocked":
            return "[yellow]⊘[/yellow]"
        return "[dim]?[/dim]"

    def _format_value(self, item: PlanItem) -> str:
        """Format the value/result for display."""
        if item.status == "pending":
            return "[dim]pending[/dim]"
        elif item.status == "running":
            return "[cyan]resolving...[/cyan]"
        elif item.status == "resolved":
            if item.value:
                val_str = str(item.value)
                # Collapse and truncate
                val_str = val_str.replace("\n", " ")
                if len(val_str) > 50:
                    val_str = val_str[:47] + "..."
                conf_str = f" [dim]({item.confidence:.0%})[/dim]" if item.confidence else ""
                return f"[green]{val_str}[/green]{conf_str}"
            return "[green]done[/green]"
        elif item.status == "failed":
            err = item.error or "error"
            if len(err) > 40:
                err = err[:37] + "..."
            return f"[red]{err}[/red]"
        elif item.status == "blocked":
            err = item.error or "blocked by failed dependency"
            if len(err) > 50:
                err = err[:47] + "..."
            return f"[yellow]{err}[/yellow]"
        return ""

    def _build_display(self) -> Table:
        """Build the display table."""
        table = Table(
            show_header=False,
            box=None,
            padding=(0, 1),
            collapse_padding=True,
        )
        table.add_column("status", width=2)
        table.add_column("id", width=4)
        table.add_column("name", min_width=20)
        table.add_column("result", min_width=30)

        # Add premises
        for p in self.premises:
            fact_id = p.get("id", "")
            if fact_id in self.items:
                item = self.items[fact_id]
                table.add_row(
                    self._get_status_indicator(item),
                    f"[bold]{fact_id}[/bold]",
                    item.name,
                    self._format_value(item),
                )

        # Separator
        if self.premises and self.inferences:
            table.add_row("", "", "[dim]───────────────────[/dim]", "")

        # Add inferences
        for inf in self.inferences:
            fact_id = inf.get("id", "")
            if fact_id in self.items:
                item = self.items[fact_id]
                table.add_row(
                    self._get_status_indicator(item),
                    f"[bold]{fact_id}[/bold]",
                    item.name,
                    self._format_value(item),
                )

        return table


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
        self._max_output_lines: int = 12  # Max lines to show above plan
        self._completed_outputs: list[tuple[int, str]] = []  # (step_num, output) for completed steps
        self._active_step_goal: str = ""  # Goal of currently executing step

        # Background animation thread
        self._animation_thread: Optional[threading.Thread] = None
        self._animation_running: bool = False

        # Proof tree for auditable mode
        self._proof_tree: Optional[ProofTree] = None
        self._auditable_mode: bool = False
        self._proof_outputs: list[tuple[str, str]] = []  # (fact_name, output) for resolved facts

        # Live plan execution display for DAG-based execution
        self._live_plan_display: Optional[LivePlanExecutionDisplay] = None

        # Stopped flag to prevent updates after interruption
        self._stopped: bool = False

    def start_live_plan_display(self, premises: list[dict], inferences: list[dict]) -> None:
        """Start the live plan execution display showing all P/I items."""
        self._stopped = False  # Reset stopped flag when starting new display
        self.stop_spinner()  # Stop any existing spinner
        self._live_plan_display = LivePlanExecutionDisplay(
            console=self.console,
            premises=premises,
            inferences=inferences,
        )
        self._live_plan_display.start()

    def stop_live_plan_display(self) -> None:
        """Stop the live plan execution display."""
        if self._live_plan_display:
            self._live_plan_display.stop()
            self._live_plan_display = None

    def update_plan_item_status(self, fact_id: str, status: str, value: str = None,
                                 error: str = None, confidence: float = 0.0) -> None:
        """Update a plan item's status in the live display."""
        if self._stopped:
            return  # Ignore updates after stop
        if self._live_plan_display:
            self._live_plan_display.update_status(fact_id, status, value, error, confidence)

    def start_proof_tree(self, conclusion_name: str, conclusion_description: str = "") -> None:
        """Start tracking a proof tree for auditable mode."""
        self._proof_tree = ProofTree(conclusion_name, conclusion_description)
        self._auditable_mode = True
        self._proof_outputs = []

    def update_proof_resolving(self, fact_name: str, description: str = "", parent_name: str = None) -> None:
        """Mark a fact as being resolved in the proof tree."""
        if self._proof_tree:
            self._proof_tree.start_resolving(fact_name, description, parent_name=parent_name)

    def update_proof_resolved(
        self,
        fact_name: str,
        value,
        source: str = "",
        confidence: float = 1.0,
        query: str = "",
        from_cache: bool = False,
        resolution_summary: str = None,
    ) -> None:
        """Mark a fact as resolved in the proof tree."""
        if self._proof_tree:
            self._proof_tree.resolve_fact(
                fact_name,
                value,
                source=source,
                confidence=confidence,
                query=query,
                from_cache=from_cache,
                result_summary=resolution_summary,  # Show resolution method, not just value
            )
            # Store output for intermediate display - use resolution summary
            if value is not None:
                summary = resolution_summary or str(value)
                # Collapse newlines for compact single-line display
                summary = summary.replace("\n", " ").replace("  ", " ")
                if len(summary) > 100:
                    summary = summary[:97] + "..."
                self._proof_outputs.append((fact_name, summary))

    def update_proof_failed(self, fact_name: str, error: str) -> None:
        """Mark a fact as failed in the proof tree."""
        if self._proof_tree:
            self._proof_tree.fail_fact(fact_name, error)

    def get_proof_tree_renderable(self) -> Optional[Panel]:
        """Get the proof tree as a renderable for display."""
        if self._proof_tree:
            return self._proof_tree.render_with_panel()
        return None

    def stop_proof_tree(self) -> None:
        """Stop the proof tree display."""
        self._auditable_mode = False

    def start(self) -> None:
        """Start the live display."""
        self._stopped = False  # Reset stopped flag when starting new display
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
            refresh_per_second=8,  # Balanced refresh rate
            transient=False,
            vertical_overflow="visible",
        )
        self._live.start()

    def stop(self) -> None:
        """Stop all animations and live displays, ensuring cleanup even on error."""
        self._stopped = True  # Prevent further updates
        self._stop_animation_thread()
        self.stop_spinner()  # Also stop any standalone spinner
        self.stop_live_plan_display()  # Stop DAG execution display
        if self._live:
            try:
                self._live.stop()
            except Exception:
                pass  # Ensure we don't fail during cleanup
            finally:
                self._live = None

    def _ensure_live(self) -> None:
        """Ensure the live display is started if configured to use it.

        This lazy initialization ensures consistent display behavior regardless
        of which code path triggers step events.
        """
        if self._use_live_display and not self._live:
            self.start()
            self._start_animation_thread()
            if self._start_time is None:
                self._start_time = time.time()

    def _start_animation_thread(self) -> None:
        """Start background thread for smooth spinner animation."""
        if self._animation_thread and self._animation_thread.is_alive():
            return  # Already running

        self._animation_running = True
        self._animation_thread = threading.Thread(
            target=self._animation_loop,
            daemon=True,
            name="FeedbackAnimation"
        )
        self._animation_thread.start()

    def _stop_animation_thread(self) -> None:
        """Stop the background animation thread."""
        self._animation_running = False
        if self._animation_thread:
            self._animation_thread.join(timeout=0.5)
            self._animation_thread = None

    def _animation_loop(self) -> None:
        """Background loop that advances spinner and triggers refresh."""
        while self._animation_running:
            with self._lock:
                # Advance spinner frame
                self._spinner_frame = (self._spinner_frame + 1) % len(SPINNER_FRAMES)
                # Trigger Live refresh if active
                if self._live:
                    try:
                        self._live.refresh()
                    except Exception:
                        pass  # Ignore refresh errors during shutdown
            # ~10 FPS for smooth animation
            time.sleep(0.1)

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

        # Use current spinner frame (advanced by background thread)
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

        For exploratory mode:
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

        For auditable mode:
        ┌─────────────────────────────────────────┐
        │  Resolved fact outputs                   │
        │  (intermediate values)                   │
        ├─────────────────────────────────────────┤
        │  Prove: conclusion_fact                  │
        │  ├── premise_1 ✓ (database)             │
        │  ├── premise_2 ⏳ resolving...           │
        │  └── premise_3 ○ pending                 │
        └─────────────────────────────────────────┘
        """
        # Use current spinner frame (advanced by background thread)
        spinner_char = SPINNER_FRAMES[self._spinner_frame]

        # Check if we're in auditable mode with a proof tree
        if self._auditable_mode and self._proof_tree:
            return self._build_proof_tree_display(spinner_char)

        # Build output section (top) - show completed step outputs
        output_parts = []

        if self._completed_outputs:
            # Collect all output lines with step prefixes
            all_lines = []
            for step_num, output in self._completed_outputs:
                output_lines = output.split('\n')
                # Add step header for each output
                all_lines.append(f"[cyan]Step {step_num}:[/cyan]")
                for line in output_lines[:3]:  # Max 3 lines per step
                    all_lines.append(f"  {line}")
                if len(output_lines) > 3:
                    all_lines.append(f"  [dim]... ({len(output_lines) - 3} more lines)[/dim]")

            # Truncate total to max lines
            if len(all_lines) > self._max_output_lines:
                all_lines = all_lines[-self._max_output_lines:]

            for line in all_lines:
                output_parts.append(Text.from_markup(line))

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

    def _build_proof_tree_display(self, spinner_char: str) -> RenderableType:
        """Build the animated display for auditable mode with proof tree.

        Shows:
        1. Intermediate outputs from resolved facts (top)
        2. Proof tree structure with live status updates (bottom)
        """
        all_parts = []

        # Show elapsed time header
        elapsed = self._format_elapsed()
        header = Text.from_markup(f"[dim]Resolving proof... ({elapsed})[/dim]")
        all_parts.append(header)
        all_parts.append(Text(""))

        # Show the proof tree structure (with animated spinner for resolving nodes)
        # Note: Removed "Resolved Facts" section - duplicative with tree display
        if self._proof_tree:
            tree = self._proof_tree.render(spinner_char=spinner_char)
            all_parts.append(tree)

            # Show summary
            summary = self._proof_tree.get_summary()
            resolved = summary["resolved"] + summary["cached"]
            total = summary["total"]
            pending = summary["pending"]
            failed = summary["failed"]

            status_parts = []
            if resolved > 0:
                status_parts.append(f"[green]{resolved} resolved[/green]")
            if pending > 0:
                status_parts.append(f"[yellow]{pending} pending[/yellow]")
            if failed > 0:
                status_parts.append(f"[red]{failed} failed[/red]")

            if status_parts:
                all_parts.append(Text(""))
                all_parts.append(Text.from_markup(f"[dim]Progress: {' | '.join(status_parts)}[/dim]"))

        return Group(*all_parts)

    def _update_live(self) -> None:
        """Update the live display with current step states."""
        if self._live and self._use_live_display:
            with self._lock:
                # Force immediate refresh for smoother updates
                self._live.refresh()

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
        self._completed_outputs = []
        self._active_step_goal = ""
        self.stop()

    def show_user_input(self, user_input: str) -> None:
        """Display user input with YOU header (right-aligned)."""
        self.console.print()
        self.console.print(Rule("[bold green]YOU[/bold green]", align="right"))
        self.console.print(f"[white]{user_input}[/white]")

    def set_problem(self, problem: str) -> None:
        """Set the problem being solved."""
        self.problem = problem
        self.console.print()  # Blank line before plan

    def show_plan(self, steps: list[dict], is_followup: bool = False) -> None:
        """Display the execution plan.

        Args:
            steps: List of step dicts with number, goal, depends_on, type (optional)
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
        self.console.print(Rule("[bold cyan]CONSTAT[/bold cyan]", align="left"))

        # Check if this is an auditable proof structure (has type field)
        is_proof_structure = any(s.get("type") in ("premise", "inference", "conclusion") for s in steps)
        current_section = None

        for i, s in enumerate(steps):
            display_num = self._step_number_map.get(s.get("number", i + 1), start_num + i)
            goal = s.get("goal", "")
            depends_on = s.get("depends_on", [])
            step_type = s.get("type")

            # Show section headers for proof structure
            if is_proof_structure and step_type != current_section:
                current_section = step_type
                if step_type == "premise":
                    self.console.print("\n  [bold yellow]PREMISES[/bold yellow] [dim](facts to retrieve from sources)[/dim]")
                elif step_type == "inference":
                    self.console.print("\n  [bold yellow]INFERENCES[/bold yellow] [dim](facts derived from premises)[/dim]")
                elif step_type == "conclusion":
                    # Show data flow DAG before conclusion
                    self._show_data_flow_dag(steps)
                    self.console.print("\n  [bold yellow]CONCLUSION[/bold yellow]")

            # Format dependency info with remapped step numbers
            dep_str = ""
            if depends_on and not is_proof_structure:
                # Only show depends_on for non-proof plans (proof structure is implicit)
                remapped_deps = [str(self._step_number_map.get(d, d)) for d in depends_on]
                dep_str = f" [dim](depends on {', '.join(remapped_deps)})[/dim]"

            # Use fact_id (P1, I1, C) for proof structures, numeric for regular plans
            if is_proof_structure:
                fact_id = s.get("fact_id", "")
                if step_type == "conclusion":
                    fact_id = "C"
                self.console.print(f"  [dim]{fact_id}:[/dim] {goal}")
            else:
                self.console.print(f"  [dim]{display_num}.[/dim] {goal}{dep_str}")

        self.console.print()

    def _show_data_flow_dag(self, steps: list[dict]) -> None:
        """Display an ASCII data flow DAG with proper box-drawing characters.

        Args:
            steps: List of proof steps with type, fact_id, goal, etc.
        """
        import re

        try:
            import networkx as nx
            from constat.visualization.box_dag import render_dag
        except ImportError:
            # networkx not available, skip diagram
            return

        # Build NetworkX graph from steps
        G = nx.DiGraph()

        for s in steps:
            step_type = s.get("type")
            fact_id = s.get("fact_id", "")
            goal = s.get("goal", "")

            if step_type == "premise":
                G.add_node(fact_id)
            elif step_type == "inference":
                G.add_node(fact_id)
                # Extract dependencies from the operation
                inf_match = re.match(r'^(\w+)\s*=\s*(.+)', goal)
                if inf_match:
                    operation = inf_match.group(2)
                else:
                    operation = goal
                deps = re.findall(r'[PI]\d+', operation)
                for dep in deps:
                    if G.has_node(dep):
                        G.add_edge(dep, fact_id)

        if G.number_of_nodes() == 0:
            return

        # Find terminal inference and add conclusion
        inferences = [n for n in G.nodes() if n.startswith('I')]
        if inferences:
            # Find the one with no outgoing edges to other inferences
            terminal = None
            for inf in inferences:
                successors = list(G.successors(inf))
                if not any(s.startswith('I') for s in successors):
                    terminal = inf
                    break
            if terminal is None:
                terminal = inferences[-1]
            G.add_node("C")
            G.add_edge(terminal, "C")

        self.console.print("\n  [bold yellow]DATA FLOW[/bold yellow]")

        try:
            from constat.visualization.box_dag import render_dag
            diagram = render_dag(G, style='rounded')
            for line in diagram.split('\n'):
                if line.strip():
                    self.console.print(f"      [dim]{line}[/dim]")
        except Exception:
            # Fallback: just list the edges
            self.console.print("      [dim](diagram rendering failed)[/dim]")

        self.console.print()

    def start_execution(self) -> None:
        """Start the live execution display."""
        self.console.print(Rule("[bold cyan]CONSTAT[/bold cyan]", align="left"))
        self._execution_started = True
        self._start_time = time.time()  # Start timing
        self._completed_outputs = []  # Clear completed outputs buffer
        self._active_step_goal = ""
        if self._use_live_display:
            self.start()
            self._start_animation_thread()  # Start background animation
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
        # Stop any running animation/spinner before prompting for input
        self.stop_spinner()

        # Show mode selection
        self.show_mode_selection(request.mode, request.mode_reasoning)

        # Show the plan
        self.show_plan(request.steps)

        # Show reasoning if available
        if request.reasoning:
            self.console.print("[bold]Reasoning:[/bold]")
            self.console.print(Panel(request.reasoning, border_style="dim"))

        # Prompt for approval - allow direct steering input
        self.console.print()

        while True:
            try:
                response = self.console.input("[dim]Enter to execute, 'n' to cancel, 'k/a/e' to switch mode, or type changes >[/dim] ").strip()
            except (EOFError, KeyboardInterrupt):
                self.console.print("\n[red]Cancelled.[/red]")
                return PlanApprovalResponse.reject("User cancelled")

            # Empty or affirmative = approve
            if not response or response.lower() in ("y", "yes", "ok", "go", "execute"):
                # Don't print "Executing..." - proof tree display shows "Resolving proof..."
                return PlanApprovalResponse.approve()

            # Reject
            elif response.lower() in ("n", "no", "cancel", "stop"):
                self.console.print("[red]Plan rejected.[/red]\n")
                return PlanApprovalResponse.reject()

            # Slash commands - pass through to REPL for global handling
            elif response.startswith("/"):
                return PlanApprovalResponse.pass_command(response)

            # Check for mode switch request
            target_mode = detect_mode_switch(response)
            if target_mode is not None:
                # Check if already in this mode
                if target_mode == request.mode:
                    self.console.print(f"[dim]Already in {target_mode.value} mode.[/dim]")
                    continue  # Ask again
                else:
                    # Mode switch message will be shown by the mode_switch event
                    return PlanApprovalResponse.switch_mode(target_mode)

            # Anything else is steering feedback
            else:
                self.console.print("[yellow]Incorporating feedback and replanning...[/yellow]\n")
                return PlanApprovalResponse.suggest(response)

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

            # Flush all output to ensure suggestions are visible before prompt
            sys.stdout.flush()
            sys.stderr.flush()
            # Force Rich console to flush as well
            self.console.file.flush() if hasattr(self.console, 'file') else None

            # Get answer - show first suggestion as default hint if available
            try:
                if suggestions:
                    default_hint = suggestions[0]
                    # Show clear prompt line with default
                    # Print the prompt prefix separately to ensure visibility
                    self.console.print(f"     > [dim]({default_hint})[/dim]: ", end="")
                    answer = input() or default_hint
                else:
                    # Show clear prompt line for custom input
                    answer = Prompt.ask("     >", default="", show_default=False)
            except Exception:
                # Fallback to basic input if Rich Prompt fails
                print("     > ", end="", flush=True)
                answer = input()
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

        # Filter out empty answers
        non_empty_answers = {q: a for q, a in answers.items() if a}
        all_answered = len(non_empty_answers) == len(request.questions)

        # Show summary of user's clarifications with YOU header
        if non_empty_answers:
            self.console.print()
            self.console.print(Rule("[bold green]YOU[/bold green]", align="right"))
            for q, a in non_empty_answers.items():
                # Shorten question for display
                short_q = q[:50] + "..." if len(q) > 50 else q
                self.console.print(f"[dim]{short_q}:[/dim] [white]{a}[/white]")
            self.console.print()

        if all_answered:
            # All questions answered - proceed automatically
            return ClarificationResponse(answers=non_empty_answers, skip=False)
        elif non_empty_answers:
            # Some questions skipped - ask if they want to continue
            skip = Prompt.ask(
                "[yellow]Some clarifications skipped.[/yellow] [dim]Press Enter to continue anyway, or 's' to cancel[/dim]",
                default="",
                show_default=False
            ).lower()
            if skip == "s":
                self.console.print("[dim]Cancelled.[/dim]")
                return ClarificationResponse(answers={}, skip=True)
            return ClarificationResponse(answers=non_empty_answers, skip=False)
        else:
            # No answers provided at all
            skip = Prompt.ask(
                "[yellow]No clarifications provided.[/yellow] [dim]Press Enter to try anyway, or 's' to cancel[/dim]",
                default="",
                show_default=False
            ).lower()
            if skip == "s":
                self.console.print("[dim]Cancelled.[/dim]")
                return ClarificationResponse(answers={}, skip=True)
            self.console.print("[dim]Proceeding with original question...[/dim]\n")
            return ClarificationResponse(answers={}, skip=False)

    def show_replan_notice(self, attempt: int, max_attempts: int) -> None:
        """Show notice that we're replanning based on feedback."""
        self.console.print(
            f"[yellow]Replanning (attempt {attempt}/{max_attempts})...[/yellow]"
        )

    def step_start(self, step_number: int, goal: str) -> None:
        """Mark a step as starting."""
        self.current_step = step_number
        self._active_step_goal = goal  # For animated header

        # Update step status
        step = self._get_step(step_number)
        if step:
            step.status = "running"
            step.status_message = "starting..."

        # Ensure live display is initialized
        self._ensure_live()
        self._update_live()

    def step_generating(self, step_number: int, attempt: int) -> None:
        """Show code generation in progress."""
        step = self._get_step(step_number)
        if step:
            step.status = "generating"
            step.status_message = f"retry #{attempt}..." if attempt > 1 else "working..."
            step.attempts = attempt

        # Ensure live display is initialized
        self._ensure_live()
        self._update_live()

    def step_executing(self, step_number: int, attempt: int, code: Optional[str] = None) -> None:
        """Show code execution in progress."""
        step = self._get_step(step_number)
        if step:
            step.status = "executing"
            step.status_message = "executing..."
            step.code = code

        # Ensure live display is initialized
        self._ensure_live()
        self._update_live()

        # In verbose mode, also show the code being executed
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
        # Accumulate completed step output for animated display
        display_num = self._step_number_map.get(step_number, step_number)
        if output:
            self._completed_outputs.append((display_num, output.strip()))

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

        # Ensure live display is initialized and update
        self._ensure_live()
        self._update_live()

        # In verbose mode, show tables that were created
        if self.verbose and tables_created:
            self.console.print(f"  [dim]tables:[/dim] {', '.join(tables_created)}")

    def step_error(self, step_number: int, error: str, attempt: int) -> None:
        """Show a step error (before retry)."""
        error_lines = error.strip().split("\n")
        brief = error_lines[-1] if error_lines else error

        step = self._get_step(step_number)
        if step:
            step.status_message = f"retry #{attempt}... ({brief[:40]})"

        # Ensure live display is initialized and update
        self._ensure_live()
        self._update_live()

        # In verbose mode, show the full error details
        if self.verbose:
            self.console.print(f"  [red]Error:[/red] {brief[:80]}")
            self.console.print(Panel(error, title="Full Error", border_style="red"))

    def step_failed(
        self,
        step_number: int,
        error: str,
        attempts: int,
        suggestions: Optional[list] = None
    ) -> None:
        """Mark a step as permanently failed and show suggestions.

        Args:
            step_number: The step that failed
            error: Error message
            attempts: Number of attempts made
            suggestions: List of FailureSuggestion objects for alternative approaches
        """
        step = self._get_step(step_number)
        if step:
            step.status = "failed"
            step.error = error
            step.attempts = attempts

        # Ensure live display is initialized and update
        self._ensure_live()
        self._update_live()

        # Show suggestions if available
        if suggestions:
            self._show_failure_suggestions(suggestions)

    def _show_failure_suggestions(self, suggestions: list) -> None:
        """Display failure recovery suggestions to the user.

        Args:
            suggestions: List of FailureSuggestion objects
        """
        from rich.table import Table

        self.console.print()
        self.console.print("[bold yellow]Alternative approaches:[/bold yellow]")

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Option", style="cyan", width=3)
        table.add_column("Label", style="bold")
        table.add_column("Description", style="dim")

        for i, suggestion in enumerate(suggestions, 1):
            label = getattr(suggestion, 'label', str(suggestion))
            description = getattr(suggestion, 'description', '')
            table.add_row(f"{i}.", label, description)

        self.console.print(table)
        self.console.print()
        self.console.print("[dim]Enter a number to try that approach, or type your own suggestion[/dim]")

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
        # Clean up output: strip trailing whitespace and collapse multiple blank lines
        cleaned = re.sub(r'\n{3,}', '\n\n', output.rstrip())
        self.console.print()
        self.console.print(Markdown(_left_align_markdown(cleaned)))
        self.console.print()  # One blank line after output

    def show_final_answer(self, answer: str) -> None:
        """Show the final synthesized answer from Vera."""
        self.console.print()
        self.console.print(Rule("[bold blue]VERA[/bold blue]", align="left"))
        self.console.print(Markdown(_left_align_markdown(answer)))
        self.console.print()

    def show_suggestions(self, suggestions: list[str]) -> None:
        """Show follow-up suggestions."""
        if not suggestions:
            return

        # No extra blank line - caller (show_output) handles spacing
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

    def show_mode_switch(self, mode: str, keywords: list[str]) -> None:
        """Show that execution mode has been switched.

        Args:
            mode: The new execution mode
            keywords: Keywords that triggered the switch
        """
        # Skip message for explicit user requests - they know what they asked for
        if keywords == ["user request"]:
            return
        keyword_str = ", ".join(keywords) if keywords else "context"
        self.console.print()
        self.console.print(
            f"[bold cyan]Switching to {mode.upper()} mode[/bold cyan] "
            f"[dim](triggered by: {keyword_str})[/dim]"
        )


class SessionFeedbackHandler:
    """
    Event handler that bridges Session events to FeedbackDisplay.

    Usage:
        display = FeedbackDisplay(verbose=True)
        handler = SessionFeedbackHandler(display, session_config)
        session.on_event(handler.handle_event)
    """

    def __init__(self, display: FeedbackDisplay, session_config=None):
        self.display = display
        self.session_config = session_config
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

        elif event_type == "step_failed":
            # Permanent failure after all retries - show suggestions
            self.display.step_failed(
                step_number,
                data.get("error", "Unknown error"),
                data.get("attempts", 1),
                data.get("suggestions"),
            )

        elif event_type == "proof_start":
            # Start the proof tree as the live display for auditable mode
            # This replaces the table-based live plan display with a tree view
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"[PROOF_START] Starting proof tree display")
            self.display.stop_spinner()
            conclusion_fact = data.get("conclusion_fact", "answer")
            conclusion_desc = data.get("conclusion_description", "")
            self.display.start_proof_tree(conclusion_fact, conclusion_desc)
            # start_proof_tree sets _auditable_mode and _proof_tree
            logger.debug(f"[PROOF_START] _auditable_mode={self.display._auditable_mode}, _proof_tree={self.display._proof_tree is not None}")
            # Start live display with proof tree
            self.display.start()
            self.display._start_time = time.time()
            self.display._start_animation_thread()
            logger.debug(f"[PROOF_START] Live display started, _live={self.display._live is not None}")

        elif event_type == "dag_execution_start":
            # Start the live plan execution display with all P/I items
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"[DAG_EXECUTION_START] _proof_tree={self.display._proof_tree is not None}")

            premises = data.get("premises", [])
            inferences = data.get("inferences", [])

            if self.display._proof_tree:
                # Pre-build the proof tree structure from the DAG
                # Tree shows: each node's children are what it REQUIRES (dependencies)
                # Root (answer) -> terminal inference -> ... -> premises (leaves)
                logger.debug(f"[DAG_EXECUTION_START] Pre-building proof tree structure")

                # Build name -> fact_id and fact_id -> dependencies maps
                name_to_id = {}
                id_to_deps = {}

                for p in premises:
                    fact_id = p.get("id", "")
                    name = p.get("name", fact_id)
                    name_to_id[name] = fact_id
                    id_to_deps[fact_id] = []  # Premises have no dependencies

                for inf in inferences:
                    fact_id = inf.get("id", "")
                    name = inf.get("name", "") or fact_id
                    op = inf.get("operation", "")
                    name_to_id[name] = fact_id
                    # Extract dependencies from operation (P1, P2, I1, etc.)
                    import re
                    deps = re.findall(r'[PI]\d+', op)
                    id_to_deps[fact_id] = deps

                # Find terminal inference (feeds into answer)
                all_deps = set()
                for deps in id_to_deps.values():
                    all_deps.update(deps)
                inference_ids = [inf.get("id") for inf in inferences]
                terminal = None
                for iid in reversed(inference_ids):
                    if iid not in all_deps:
                        terminal = iid
                        break
                if not terminal and inference_ids:
                    terminal = inference_ids[-1]

                # Add terminal inference as child of root
                if terminal:
                    terminal_name = next((inf.get("name", inf.get("id")) for inf in inferences if inf.get("id") == terminal), terminal)
                    self.display._proof_tree.add_fact(f"{terminal}: {terminal_name}", "", parent_name="answer")
                    logger.debug(f"[DAG_EXECUTION_START] Added terminal {terminal} under root")

                # BFS from terminal to build tree (each node's children are its dependencies)
                added = {"answer", terminal}
                queue = [terminal]
                while queue:
                    current_id = queue.pop(0)
                    current_deps = id_to_deps.get(current_id, [])
                    for dep_id in current_deps:
                        if dep_id not in added:
                            # Find name for this dep
                            dep_name = None
                            for p in premises:
                                if p.get("id") == dep_id:
                                    dep_name = p.get("name", dep_id)
                                    break
                            if not dep_name:
                                for inf in inferences:
                                    if inf.get("id") == dep_id:
                                        dep_name = inf.get("name", "") or dep_id
                                        break
                            if not dep_name:
                                dep_name = dep_id

                            # Find parent in tree (what uses this dep)
                            # Current_id uses dep_id, so dep is child of current
                            current_name = None
                            for p in premises:
                                if p.get("id") == current_id:
                                    current_name = p.get("name", current_id)
                                    break
                            if not current_name:
                                for inf in inferences:
                                    if inf.get("id") == current_id:
                                        current_name = inf.get("name", "") or current_id
                                        break
                            if not current_name:
                                current_name = current_id

                            parent_key = f"{current_id}: {current_name}"
                            self.display._proof_tree.add_fact(f"{dep_id}: {dep_name}", "", parent_name=parent_key)
                            logger.debug(f"[DAG_EXECUTION_START] Added {dep_id} under {current_id}")
                            added.add(dep_id)
                            queue.append(dep_id)

                logger.debug(f"[DAG_EXECUTION_START] Proof tree pre-built with {len(added)} nodes")
                return

            logger.debug(f"[DAG_EXECUTION_START] Starting table display")
            self.display.start_live_plan_display(premises, inferences)

        elif event_type == "dag_execution_complete":
            # Update any remaining items based on success/failure
            success = data.get("success", True)
            failed_nodes = data.get("failed_nodes", [])

            if self.display._live_plan_display:
                # Mark failed nodes
                for node_name in failed_nodes:
                    self.display.update_plan_item_status(node_name, "failed", error="execution failed")

                # Mark any remaining pending/running items as resolved if success
                if success:
                    for item in self.display._live_plan_display.items.values():
                        if item.status in ("pending", "running"):
                            item.status = "resolved"
                            if not item.value:
                                item.value = "done"

                # Give display a moment to render final state
                time.sleep(0.3)

            # Note: Don't update proof tree here - failed nodes are already marked
            # by premise_resolved or inference_failed events. Adding them again
            # would create duplicates with different names (e.g., "raise_guidelines"
            # vs "P3: raise_guidelines").

            # Stop the live plan display (safe even if not started)
            self.display.stop_live_plan_display()

        elif event_type == "premise_resolving":
            # Show which fact is being resolved
            fact_name = data.get("fact_name", "?")
            description = data.get("description", "")
            step = data.get("step", 0)
            total = data.get("total", 0)

            # Extract fact_id from "P1: name" format
            fact_id = fact_name.split(":")[0].strip() if ":" in fact_name else fact_name

            # Update live plan display if active
            if self.display._live_plan_display:
                self.display.update_plan_item_status(fact_id, "running")
            # Update proof tree status (tree structure is pre-built)
            if self.display._proof_tree:
                # Just mark as resolving - node already exists from pre-build
                self.display.update_proof_resolving(fact_name, description)
            elif not self.display._live_plan_display:
                self.display.update_spinner(f"Resolving {fact_name} ({step}/{total})...")

        elif event_type == "premise_retry":
            # Show retry info inline with the current premise being resolved
            premise_id = data.get("premise_id", "?")
            premise_name = data.get("premise_name", "?")
            attempt = data.get("attempt", 2)
            max_attempts = data.get("max_attempts", 3)
            error_brief = data.get("error_brief", "")
            step = data.get("step", 0)
            total = data.get("total", 0)

            # Format: "Resolving P2 (2/4)... [retry 2/3: syntax error]"
            retry_info = f"retry {attempt}/{max_attempts}"
            if error_brief:
                # Truncate error and clean up for display
                err_display = error_brief.replace('\n', ' ')[:40]
                retry_info += f": {err_display}"

            if self.display._proof_tree:
                self.display.update_proof_resolving(premise_id, f"retrying... [{retry_info}]")
            else:
                self.display.update_spinner(f"Resolving {premise_id} ({step}/{total})... [yellow][{retry_info}][/yellow]")

        elif event_type == "premise_resolved":
            # Show resolved fact value
            fact_name = data.get("fact_name", "?")
            value = data.get("value")
            source = data.get("source", "")
            step = data.get("step", 0)
            total = data.get("total", 0)
            confidence = data.get("confidence", 1.0)
            resolution_summary = data.get("resolution_summary")
            query = data.get("query")
            error = data.get("error")

            # Extract fact_id from "P1: name" format
            fact_id = fact_name.split(":")[0].strip() if ":" in fact_name else fact_name

            # Update live plan display if active
            if self.display._live_plan_display:
                if value is not None:
                    self.display.update_plan_item_status(
                        fact_id, "resolved", value=str(value), confidence=confidence
                    )
                else:
                    self.display.update_plan_item_status(
                        fact_id, "failed", error=error or "unresolved"
                    )
            # Also update proof tree (for final proof display)
            if self.display._proof_tree:
                if value is not None:
                    from_cache = source == "cache"
                    self.display.update_proof_resolved(
                        fact_name,
                        value,
                        source=source,
                        confidence=confidence,
                        from_cache=from_cache,
                        resolution_summary=resolution_summary,
                        query=query,
                    )
                else:
                    self.display.update_proof_failed(fact_name, error or "unresolved")
            elif not self.display._live_plan_display:
                # Fallback to simple console output
                if value is not None:
                    val_str = str(value)
                    # Collapse newlines for compact single-line display
                    val_str = val_str.replace("\n", " ").replace("  ", " ")
                    if len(val_str) > 60:
                        val_str = val_str[:57] + "..."
                    conf_str = f" ({confidence:.0%})" if confidence else ""
                    self.display.console.print(f"  [green]✓[/green] {fact_name} = {val_str} [dim][{source}]{conf_str}[/dim]")
                else:
                    self.display.console.print(f"  [red]✗[/red] {fact_name} = [red]UNRESOLVED[/red] [dim]({error})[/dim]")

        elif event_type == "inference_executing":
            # Show which inference step is being executed
            inference_id = data.get("inference_id", "?")
            operation = data.get("operation", "")  # This is the inference variable name
            step = data.get("step", 0)
            total = data.get("total", 0)

            # Format inference name like premises: "I1: recent_reviews"
            inference_display = f"{inference_id}: {operation}" if operation else inference_id

            # Update live plan display if active
            if self.display._live_plan_display:
                self.display.update_plan_item_status(inference_id, "running")
            # Update proof tree status (tree structure is pre-built)
            if self.display._proof_tree:
                # Just mark as resolving - node already exists from pre-build
                self.display.update_proof_resolving(inference_display, operation)
            elif not self.display._live_plan_display:
                self.display.update_spinner(f"Executing {inference_id} ({step}/{total})...")

        elif event_type == "inference_retry":
            # Show retry info inline with the spinner (single line)
            inference_id = data.get("inference_id", "?")
            attempt = data.get("attempt", 2)
            max_attempts = data.get("max_attempts", 3)
            error_brief = data.get("error_brief", "")
            step = data.get("step", 0)
            total = data.get("total", 0)

            # Format: "Executing I2 (2/4)... [retry 2/3: 'column not found']"
            retry_info = f"retry {attempt}/{max_attempts}"
            if error_brief:
                # Truncate error and clean up for display
                err_display = error_brief.replace('\n', ' ')[:40]
                retry_info += f": {err_display}"

            if self.display._proof_tree:
                self.display.update_proof_resolving(inference_id, f"retrying... [{retry_info}]")
            else:
                self.display.update_spinner(f"Executing {inference_id} ({step}/{total})... [yellow][{retry_info}][/yellow]")

        elif event_type == "inference_complete":
            # Show completed inference step
            inference_id = data.get("inference_id", "?")
            inference_name = data.get("inference_name", "")
            result = data.get("result", "computed")
            output = data.get("output", "")
            step = data.get("step", 0)
            total = data.get("total", 0)

            # Build result summary including output if present
            result_summary = result
            if output:
                # Truncate long output
                output_preview = output[:100] + "..." if len(output) > 100 else output
                result_summary = f"{result} ({output_preview})"

            # Build display label with ID and name (like premises: "I1: recent_reviews")
            display_label = f"{inference_id}: {inference_name}" if inference_name else inference_id

            # Update live plan display if active
            if self.display._live_plan_display:
                self.display.update_plan_item_status(
                    inference_id, "resolved", value=str(result), confidence=0.9
                )
            # Also update proof tree (for final proof display)
            if self.display._proof_tree:
                self.display.update_proof_resolved(
                    display_label,  # Use full name like "I1: recent_reviews"
                    result,
                    source="derived",
                    confidence=1.0,
                    resolution_summary=output if output else None,
                )
            elif not self.display._live_plan_display:
                self.display.console.print(f"  [green]✓[/green] {display_label} = {result}")
                if output:
                    # Show captured output
                    for line in output.split("\n"):
                        self.display.console.print(f"    [dim]{line}[/dim]")

        elif event_type == "inference_failed":
            # Show failed inference step and stop animation
            # Event data may have fact_name (with full name) or inference_id
            fact_name = data.get("fact_name", "")
            inference_id = data.get("inference_id", "?")
            # Use fact_name if available (includes name like "I1: recent_reviews"), else fallback
            display_name = fact_name if fact_name else inference_id
            # Extract just the ID for live plan display
            id_only = fact_name.split(":")[0].strip() if ":" in fact_name else inference_id
            error = data.get("error", "unknown error")
            step = data.get("step", 0)
            total = data.get("total", 0)

            # Update live plan display if active
            if self.display._live_plan_display:
                self.display.update_plan_item_status(id_only, "failed", error=error)
                # Stop the live display on failure
                self.display.stop_live_plan_display()
            # Also update proof tree (for final proof display)
            if self.display._proof_tree:
                self.display.update_proof_failed(display_name, error)
                if not self.display._live_plan_display:
                    self.display.stop()
            elif not self.display._live_plan_display:
                self.display.console.print(f"  [red]✗[/red] {display_name} = [red]FAILED[/red] [dim]({error})[/dim]")
                self.display.stop()

            # Show error summary and suggestions
            self.display.console.print(f"\n[red bold]Error:[/red bold] {error[:200]}")
            self.display.console.print("\n[yellow]Suggestions:[/yellow]")
            self.display.console.print("  • Try simplifying the request")
            self.display.console.print("  • Check that the data sources have the required columns")
            self.display.console.print("  • Use '/redo' to try again with modifications")

        elif event_type == "data_warning":
            name = data.get("name", "?")
            row_count = data.get("row_count", 0)
            threshold = data.get("threshold", 0)
            data_type = data.get("type", "data")
            self.display.console.print(
                f"  [yellow]Warning:[/yellow] {name} has {row_count:,} rows "
                f"(threshold: {threshold:,})"
            )

        elif event_type == "data_sampled":
            name = data.get("name", "?")
            original = data.get("original_rows", 0)
            sampled = data.get("sampled_rows", 0)
            self.display.console.print(
                f"  [cyan]Sampled:[/cyan] {name}: {original:,} -> {sampled:,} rows"
            )

        elif event_type == "synthesizing":
            # Update root node to RESOLVED before stopping live display
            if self.display._proof_tree:
                from constat.proof_tree import NodeStatus
                # Stop animation thread first
                self.display._animation_running = False
                if self.display._animation_thread:
                    self.display._animation_thread.join(timeout=0.5)
                    self.display._animation_thread = None
                # Mark root as resolved (all premises/inferences completed)
                self.display._proof_tree.root.status = NodeStatus.RESOLVED
                # Calculate confidence as minimum of all resolved nodes with confidence
                # (answer is only as confident as its weakest premise)
                # Need to collect recursively since premises may be nested deep
                def collect_confidences(node):
                    confidences = []
                    if node.status == NodeStatus.RESOLVED and node.confidence > 0:
                        confidences.append(node.confidence)
                    for child in node.children:
                        confidences.extend(collect_confidences(child))
                    return confidences
                all_confidences = collect_confidences(self.display._proof_tree.root)
                if all_confidences:
                    self.display._proof_tree.root.confidence = min(all_confidences)
                # Force final refresh to show completed state
                if self.display._live:
                    self.display._live.refresh()
                    time.sleep(0.1)
            # Stop Live display before printing synthesizing message
            self.display.stop()
            self.display.stop_proof_tree()
            self.display.console.print(f"\n[dim]{data.get('message', 'Synthesizing...')}[/dim]")

        elif event_type == "raw_results_ready":
            # Raw results are shown immediately so user can see them while synthesis runs
            # (or if synthesis is skipped, this is the only output)
            # Check if raw output is enabled (respects session_config.show_raw_output)
            show_raw = True
            if self.session_config is not None:
                show_raw = getattr(self.session_config, 'show_raw_output', True)

            output = data.get("output", "")
            if output and show_raw:
                self.display.stop()  # Stop any spinners/live display
                self.display.console.print(f"\n[dim]─── Raw Results ───[/dim]")
                self.display.console.print(output)

        elif event_type == "answer_ready":
            self.display.show_final_answer(data.get("answer", ""))

        elif event_type == "suggestions_ready":
            self.display.show_suggestions(data.get("suggestions", []))

        elif event_type == "facts_extracted":
            facts = data.get("facts", [])
            source = data.get("source", "unknown")
            if facts:
                self.display.show_facts_extracted(facts, source)

        elif event_type == "mode_switch":
            mode = data.get("mode", "")
            keywords = data.get("matched_keywords", [])
            self.display.show_mode_switch(mode, keywords)

        elif event_type == "deriving":
            message = data.get("message", "Deriving answer...")
            self.display.start_spinner(message)

        elif event_type == "verifying":
            # Don't start spinner if proof tree is already active (it has its own display)
            if not self.display._proof_tree:
                message = data.get("message", "Verifying...")
                self.display.start_spinner(message)

        elif event_type == "verification_complete":
            # Stop proof tree display if active (don't reprint - already shown during execution)
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"[VERIFICATION_COMPLETE] proof_tree={self.display._proof_tree is not None}, live={self.display._live is not None}")
            if self.display._proof_tree:
                # Mark root node as resolved before stopping
                # Use lock to prevent race with animation thread
                from constat.proof_tree import NodeStatus
                confidence = data.get("confidence", 0.0)
                logger.debug(f"[VERIFICATION_COMPLETE] Root status BEFORE: {self.display._proof_tree.root.status}")

                # Stop animation thread first to prevent it from overwriting our update
                self.display._animation_running = False
                if self.display._animation_thread:
                    self.display._animation_thread.join(timeout=0.5)
                    self.display._animation_thread = None

                # Now update root status and refresh
                self.display._proof_tree.root.status = NodeStatus.RESOLVED
                self.display._proof_tree.root.confidence = confidence
                logger.debug(f"[VERIFICATION_COMPLETE] Root status AFTER: {self.display._proof_tree.root.status}")

                # Force final refresh
                if self.display._live:
                    logger.debug("[VERIFICATION_COMPLETE] Calling refresh()")
                    self.display._live.refresh()
                    time.sleep(0.1)  # Brief pause for render

                logger.debug("[VERIFICATION_COMPLETE] Stopping display")
                self.display.stop()
                self.display.stop_proof_tree()
            else:
                self.display.stop_spinner()

        elif event_type == "verification_error":
            # Stop proof tree display if active (don't reprint - already shown during execution)
            import logging
            logger = logging.getLogger(__name__)
            if self.display._proof_tree:
                # Mark root node as failed before stopping
                from constat.proof_tree import NodeStatus

                # Stop animation thread first to prevent it from overwriting our update
                self.display._animation_running = False
                if self.display._animation_thread:
                    self.display._animation_thread.join(timeout=0.5)
                    self.display._animation_thread = None

                # Now update root status and refresh
                self.display._proof_tree.root.status = NodeStatus.FAILED
                logger.debug(f"[VERIFICATION_ERROR] Root status set to FAILED")

                # Force final refresh
                if self.display._live:
                    self.display._live.refresh()
                    time.sleep(0.1)

                self.display.stop()
                self.display.stop_proof_tree()
            else:
                self.display.stop_spinner()
            error = data.get("error", "Unknown error")
            self.display.console.print(f"[red]Verification failed:[/red] {error}")
