# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""StatusLine and PersistentStatusBar for terminal status display."""

from __future__ import annotations

from rich.console import Console

from constat.execution.mode import Phase, ConversationState
from constat.repl.feedback._models import SPINNER_FRAMES


class StatusLine:
    """
    Persistent status line showing current conversation state.

    Shows phase and contextual information:
    - idle
    - planning "Analyze revenue by region"
    - executing step 2/5 "Loading sales data"
    - failed "Database connection error" [retry/replan/abandon]
    """

    def __init__(self):
        self._phase: Phase = Phase.IDLE
        self._plan_name: str | None = None
        self._step_current: int = 0
        self._step_total: int = 0
        self._step_description: str = ""
        self._queue_count: int = 0
        self._error_message: str | None = None
        self._spinner_frame: int = 0
        self._status_message: str | None = None  # Arbitrary status message (e.g., "Analyzing...")

    def render(self) -> str:
        """Render current status line as a formatted string."""
        parts = []

        # If there's an explicit status message, show that instead of phase
        if self._status_message:
            spinner = SPINNER_FRAMES[self._spinner_frame % len(SPINNER_FRAMES)]
            parts.append(f"[cyan]{spinner}[/cyan] {self._status_message}")
            return "    ".join(parts)

        # Phase with context
        if self._phase == Phase.IDLE:
            parts.append("[dim]Ready[/dim]")

        elif self._phase == Phase.PLANNING:
            spinner = SPINNER_FRAMES[self._spinner_frame % len(SPINNER_FRAMES)]
            parts.append(f"[dim]Planning[/dim] [cyan]{spinner}[/cyan]")
            if self._plan_name:
                truncated = self._plan_name[:40] + "..." if len(self._plan_name) > 40 else self._plan_name
                parts.append(f'[dim]"{truncated}"[/dim]')

        elif self._phase == Phase.AWAITING_APPROVAL:
            parts.append("[dim]Awaiting approval[/dim]")
            if self._plan_name:
                truncated = self._plan_name[:30] + "..." if len(self._plan_name) > 30 else self._plan_name
                parts.append(f'"{truncated}"')
            if self._step_total > 0:
                parts.append(f"({self._step_total} steps)")
            parts.append("[dim][y/n/suggest][/dim]")

        elif self._phase == Phase.EXECUTING:
            spinner = SPINNER_FRAMES[self._spinner_frame % len(SPINNER_FRAMES)]
            parts.append(f"[green]Executing[/green] [cyan]{spinner}[/cyan]")
            if self._step_total > 0:
                parts.append(f"Step {self._step_current}/{self._step_total}")
            if self._step_description:
                truncated = self._step_description[:30] + "..." if len(self._step_description) > 30 else self._step_description
                parts.append(f'"{truncated}"')
            if self._queue_count > 0:
                parts.append(f"[dim][Queued: {self._queue_count}][/dim]")

        elif self._phase == Phase.FAILED:
            parts.append("[red]Failed[/red] [red]x[/red]")
            if self._step_current > 0:
                parts.append(f"step {self._step_current}")
            if self._error_message:
                truncated = self._error_message[:40] + "..." if len(self._error_message) > 40 else self._error_message
                parts.append(f'[red]"{truncated}"[/red]')
            parts.append("[dim][retry/replan/abandon][/dim]")

        return "    ".join(parts)

    def update(self, state: ConversationState) -> None:
        """Update from conversation state."""
        self._phase = state.phase
        self._error_message = state.failure_context

        # Extract plan info if available
        if state.active_plan:
            plan = state.active_plan
            if hasattr(plan, 'goal'):
                self._plan_name = plan.goal
            if hasattr(plan, 'steps'):
                self._step_total = len(plan.steps)

    def set_step_progress(self, current: int, total: int, description: str = "") -> None:
        """Update step progress during execution."""
        self._step_current = current
        self._step_total = total
        self._step_description = description

    def set_queue_count(self, count: int) -> None:
        """Update the queued intent count."""
        self._queue_count = count

    def set_status_message(self, message: str | None) -> None:
        """Set an arbitrary status message (overrides phase display when set)."""
        self._status_message = message

    def advance_spinner(self) -> None:
        """Advance the spinner animation frame."""
        self._spinner_frame += 1


class PersistentStatusBar:
    """
    Status bar state tracker.

    Tracks mode, phase, and stats. The actual display is handled by:
    - prompt_toolkit's bottom_toolbar during input
    - Rich spinner display during processing
    """

    def __init__(self, console: Console):
        self.console = console
        self._status_line = StatusLine()
        self._enabled = False
        self._tables_count = 0
        self._facts_count = 0
        self._artifacts_count = 0

    def enable(self) -> None:
        """Enable status tracking."""
        self._enabled = True

    def disable(self) -> None:
        """Disable status tracking."""
        self._enabled = False

    def refresh(self) -> None:
        """No-op - display is handled by prompt_toolkit or Rich."""
        pass

    def update(self, phase: Phase = None,
               plan_name: str = None, step_current: int = None,
               step_total: int = None, step_description: str = None,
               error_message: str = None, tables_count: int = None,
               facts_count: int = None, artifacts_count: int = None) -> None:
        """Update status bar values and refresh display."""
        if phase is not None:
            self._status_line._phase = phase
        if plan_name is not None:
            self._status_line._plan_name = plan_name
        if step_current is not None:
            self._status_line._step_current = step_current
        if step_total is not None:
            self._status_line._step_total = step_total
        if step_description is not None:
            self._status_line._step_description = step_description
        if error_message is not None:
            self._status_line._error_message = error_message
        if tables_count is not None:
            self._tables_count = tables_count
        if facts_count is not None:
            self._facts_count = facts_count
        if artifacts_count is not None:
            self._artifacts_count = artifacts_count

        self.refresh()

    def set_status_message(self, message: str | None) -> None:
        """Set an arbitrary status message and refresh."""
        self._status_line.set_status_message(message)
        self.refresh()

    def reset(self) -> None:
        """Reset status bar to idle state."""
        self._status_line._phase = Phase.IDLE
        self._status_line._plan_name = None
        self._status_line._step_current = 0
        self._status_line._step_total = 0
        self._status_line._step_description = ""
        self._status_line._error_message = None
        self._status_line._status_message = None
        self.refresh()

    @property
    def status_line(self) -> StatusLine:
        """Access the underlying status line for direct manipulation."""
        return self._status_line

    @property
    def is_active(self) -> bool:
        """Status bar is never 'active' in persistent scroll mode.

        Since we use prompt_toolkit for the status bar during input and
        Rich Live for the status bar during processing, this should always
        return False so that spinners display properly with their own
        status bar line.
        """
        return False

    def print(self, *args, **kwargs) -> None:
        """Print to console directly."""
        self.console.print(*args, **kwargs)
