# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""FeedbackDisplayCore — base class for FeedbackDisplay.

Contains __init__, status bar, lifecycle, spinner, and display builders.
"""

from __future__ import annotations

import sys
import threading
import time
from typing import Optional

# prompt_toolkit for input with status bar
from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.styles import Style as PTStyle
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.progress import Progress, TaskID
from rich.rule import Rule
from rich.text import Text

from constat.execution.mode import Phase
from constat.proof_tree import ProofTree
from constat.repl.feedback._models import SPINNER_FRAMES, StepDisplay
from constat.repl.feedback._plan_display import LivePlanExecutionDisplay
from constat.repl.feedback._status import PersistentStatusBar


class FeedbackDisplayCore:
    """
    Rich-based terminal display for session execution (core/base).

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
        self._resolved_tables: set[str] = set()  # Track unique table names for status bar

        # Live plan execution display for DAG-based execution
        self._live_plan_display: Optional[LivePlanExecutionDisplay] = None

        # Stopped flag to prevent updates after interruption
        self._stopped: bool = False

        # Persistent status bar pinned to bottom of terminal
        self._status_bar: PersistentStatusBar = PersistentStatusBar(self.console)
        self._status_bar_enabled: bool = False
        self._scroll_region_active: bool = False

    def setup_scroll_region(self) -> None:
        """Set up terminal scroll region to reserve bottom 2 lines for status bar."""
        import shutil
        if not sys.stdout.isatty():
            return
        height = shutil.get_terminal_size().lines
        # Set scroll region from line 1 to height-2 (reserve bottom 2 lines)
        sys.stdout.write(f"\033[1;{height - 2}r")
        # Move cursor to top of scroll region
        sys.stdout.write("\033[1;1H")
        sys.stdout.flush()
        self._scroll_region_active = True
        # Draw initial status bar
        self._draw_bottom_status_bar()

    def restore_scroll_region(self) -> None:
        """Restore full terminal scroll region."""
        if not sys.stdout.isatty() or not self._scroll_region_active:
            return
        # Reset scroll region to full terminal
        sys.stdout.write("\033[r")
        sys.stdout.flush()
        self._scroll_region_active = False

    def _draw_bottom_status_bar(self) -> None:
        """Draw the status bar at the fixed bottom of the terminal."""
        import shutil
        if not sys.stdout.isatty():
            return
        height = shutil.get_terminal_size().lines
        width = shutil.get_terminal_size().columns

        # Save cursor position
        sys.stdout.write("\033[s")

        # Move to bottom area (line height-1 for rule, height for status)
        sys.stdout.write(f"\033[{height - 1};1H")

        # Draw rule line
        rule_line = '─' * width
        sys.stdout.write(f"\033[90m{rule_line}\033[0m")  # Gray rule

        # Move to status line
        sys.stdout.write(f"\033[{height};1H")

        # Build status text
        status_line = self._status_bar.status_line
        status_msg = status_line._status_message
        phase = status_line._phase

        # Status text with spinner if active
        if status_msg:
            spinner_char = SPINNER_FRAMES[status_line._spinner_frame % len(SPINNER_FRAMES)]
            status_text = f"\033[36m{spinner_char} {status_msg}\033[0m"  # Cyan
        elif phase.value == "idle":
            status_text = "\033[90mready\033[0m"  # Gray
        else:
            status_text = f"\033[90m{phase.value}\033[0m"

        # Stats
        tables_count = self._status_bar._tables_count
        facts_count = self._status_bar._facts_count
        stats = f"\033[90mtables:{tables_count} facts:{facts_count}\033[0m"

        # Clear line and write status
        sys.stdout.write("\033[K")  # Clear line
        sys.stdout.write(f"{status_text}  {stats}")

        # Restore cursor position
        sys.stdout.write("\033[u")
        sys.stdout.flush()

    def enable_status_bar(self) -> None:
        """Enable the persistent status bar at the bottom of the terminal."""
        self._status_bar.enable()
        self._status_bar_enabled = True
        # Note: scroll region disabled due to compatibility issues with prompt_toolkit

    def disable_status_bar(self) -> None:
        """Disable the persistent status bar."""
        self._status_bar.disable()
        self._status_bar_enabled = False
        self._status_bar_enabled = False

    def print(self, *args, **kwargs) -> None:
        """Print content, routing through status bar if active.

        When the status bar is active, content is printed above the status line.
        Otherwise, prints directly to the console.
        """
        self._status_bar.print(*args, **kwargs)

    def get_status_line(self) -> str:
        """Get the current status line for display (fallback for non-persistent mode)."""
        return self._status_bar.status_line.render()

    def update_status_line(self, phase: Phase = None,
                           plan_name: str = None, step_current: int = None,
                           step_total: int = None, step_description: str = None,
                           error_message: str = None) -> None:
        """Update the status line/bar with new values."""
        self._status_bar.update(
            phase=phase,
            plan_name=plan_name,
            step_current=step_current,
            step_total=step_total,
            step_description=step_description,
            error_message=error_message,
        )

    def reset_status_line(self) -> None:
        """Reset status line to idle state."""
        self._status_bar.reset()

    def set_status_message(self, message: str | None) -> None:
        """Set an arbitrary status message in the status bar."""
        self._status_bar.set_status_message(message)

    def refresh_status_bar(self) -> None:
        """Refresh the status bar display."""
        if self._status_bar_enabled:
            self._status_bar.refresh()

    def _get_status_toolbar(self):
        """Get the status bar for prompt_toolkit bottom_toolbar.

        Returns a two-line toolbar with:
        1. A horizontal rule
        2. The status bar with mode, status, and stats
        """
        import shutil

        status_line = self._status_bar.status_line
        status_msg = status_line._status_message
        phase = status_line._phase

        # Status text
        if status_msg:
            status_text = status_msg
        elif phase.value == "idle":
            status_text = "ready"
        else:
            status_text = phase.value

        # Stats
        tables_count = self._status_bar._tables_count
        facts_count = self._status_bar._facts_count
        stats = f"tables:{tables_count} facts:{facts_count}"

        # Get terminal width for rule
        terminal_width = shutil.get_terminal_size().columns
        rule_line = '─' * terminal_width

        # Return two-line toolbar: rule + status bar
        # Rule uses gray foreground, explicitly set dark background to match toolbar
        return HTML(f'<style fg="ansigray" bg="#333333">{rule_line}</style>\n{status_text}  <style fg="ansigray">{stats}</style>')

    def prompt_with_status(self, prompt_text: str = "> ", default: str = "") -> str:
        """Prompt for input with status bar at bottom.

        Uses prompt_toolkit to show status bar during all input prompts.
        Falls back to console.input in non-interactive environments (tests).

        Args:
            prompt_text: The prompt string to display
            default: Default value if user presses Enter (not pre-filled)

        Returns:
            User's input string, or default if empty
        """
        # Check if we're in an interactive terminal
        import os
        if not sys.stdin.isatty() or os.environ.get("CONSTAT_TEST_MODE"):
            # Fall back to console.input in non-interactive environments
            try:
                result = self.console.input(prompt_text)
                return result.strip() if result else default
            except (EOFError, KeyboardInterrupt):
                return default

        style = PTStyle.from_dict({
            'bottom-toolbar': '#ffffff bg:#333333',
        })

        try:
            # Don't pre-fill the default - just return it if user enters nothing
            result = pt_prompt(
                prompt_text,
                bottom_toolbar=self._get_status_toolbar,
                style=style,
            )
            return result.strip() if result else default
        except (EOFError, KeyboardInterrupt):
            return default
        except Exception:
            # Fall back to console.input if prompt_toolkit fails
            try:
                result = self.console.input(prompt_text)
                return result.strip() if result else default
            except (EOFError, KeyboardInterrupt):
                return default

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
        self._resolved_tables = set()  # Reset table tracking for new proof

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

            # Update status bar facts count
            self._status_bar._facts_count += 1

    def update_proof_failed(self, fact_name: str, error: str) -> None:
        """Mark a fact as failed in the proof tree."""
        if self._proof_tree:
            self._proof_tree.fail_fact(fact_name, error)

    def get_proof_tree_renderable(self) -> Optional[RenderableType]:
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
            def __init__(wrapper_self, display: "FeedbackDisplayCore"):
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

    def _render_spinner_with_status(self, message: str) -> RenderableType:
        """Render just the status bar (spinner is now inside the status bar)."""
        # The spinner and message are now shown inside _build_status_bar_line()
        return self._build_status_bar_line()

    def start_spinner(self, message: str) -> None:
        """Start an animated spinner with a message in the status bar."""
        self.stop_spinner()  # Stop any existing spinner
        self._status_bar.set_status_message(message)
        self._current_spinner_message = message

        # Use Live display to show status bar with spinner
        self._spinner_live = Live(
            self._render_spinner_with_status(message),
            console=self.console,
            transient=True,
            refresh_per_second=10,
        )
        self._spinner_live.start()

        # Start animation thread
        self._spinner_running = True
        self._spinner_thread = threading.Thread(target=self._animate_spinner, daemon=True)
        self._spinner_thread.start()

    def _animate_spinner(self) -> None:
        """Animate the spinner in Live display mode."""
        while self._spinner_running:
            self._status_bar.status_line.advance_spinner()
            if hasattr(self, '_spinner_live') and self._spinner_live:
                try:
                    self._spinner_live.update(
                        self._render_spinner_with_status(self._current_spinner_message)
                    )
                except Exception:
                    pass
            time.sleep(0.1)

    def _animate_bottom_spinner(self) -> None:
        """Animate the spinner in the fixed bottom status bar."""
        while self._spinner_running:
            self._status_bar.status_line.advance_spinner()
            try:
                self._draw_bottom_status_bar()
            except Exception:
                pass
            time.sleep(0.1)

    def update_spinner(self, message: str) -> None:
        """Update the spinner message."""
        self._current_spinner_message = message
        self._status_bar.set_status_message(message)

        # Update Live display if running
        if hasattr(self, '_spinner_live') and self._spinner_live:
            try:
                self._spinner_live.update(self._render_spinner_with_status(message))
            except Exception:
                pass
        # Fallback to old Progress spinner
        elif self._spinner_progress and self._spinner_task is not None:
            self._spinner_progress.update(self._spinner_task, description=message)

    def stop_spinner(self) -> None:
        """Stop the spinner."""
        # Clear status bar message
        self._status_bar.set_status_message(None)

        # Stop animation thread
        if hasattr(self, '_spinner_running'):
            self._spinner_running = False

        # Stop Live display
        if hasattr(self, '_spinner_live') and self._spinner_live:
            try:
                self._spinner_live.stop()
            except Exception:
                pass
            self._spinner_live = None

        # Stop old Progress spinner
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
        3. Status bar at bottom with elapsed time
        """
        all_parts = []

        # Set the status message to include elapsed time (shown in status bar)
        elapsed = self._format_elapsed()
        self._status_bar.set_status_message(f"Resolving proof... ({elapsed})")

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

        # Add status bar at bottom (includes rule line)
        all_parts.append(self._build_status_bar_line())

        return Group(*all_parts)

    def _build_status_bar_line(self) -> RenderableType:
        """Build a Rich status bar with rule and status line.

        Returns a Group containing:
        1. A horizontal rule
        2. The status bar with spinner (if active), status, and stats
        """
        status_line = self._status_bar.status_line
        status_msg = status_line._status_message
        phase = status_line._phase

        # Status text with spinner if there's an active status message
        if status_msg:
            spinner_char = SPINNER_FRAMES[status_line._spinner_frame % len(SPINNER_FRAMES)]
            status_text = f"{spinner_char} {status_msg}"
        elif phase.value == "idle":
            status_text = "Ready to accept input"
        else:
            status_text = phase.value

        # Stats
        tables_count = self._status_bar._tables_count
        facts_count = self._status_bar._facts_count
        stats = f"tables:{tables_count} facts:{facts_count}"

        # Build the status line
        line = Text()
        line.append(status_text, style="dim" if not status_msg else "cyan")
        line.append("  ")
        line.append(stats, style="dim")

        # Return Group with rule and status line
        return Group(Rule(style="dim"), line)

    def _update_live(self) -> None:
        """Update the live display with current step states."""
        if self._live and self._use_live_display:
            with self._lock:
                # Force immediate refresh for smoother updates
                self._live.refresh()
