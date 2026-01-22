# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Textual-based interactive REPL with persistent status bar."""

from __future__ import annotations

import asyncio
import logging
import os
import random
import re
import shutil
import sys
import threading
from pathlib import Path
from typing import Optional, Callable

logger = logging.getLogger(__name__)

from rich.console import Console, RenderableType
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.markdown import Markdown
from rich.tree import Tree

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import ScrollableContainer, Vertical, Horizontal
from textual.widgets import Static, Input, RichLog, Footer
from textual.reactive import reactive
from textual.message import Message
from textual import events, work
from textual.worker import Worker, get_current_worker
from textual.suggester import Suggester

from constat.session import Session, SessionConfig, ClarificationRequest, ClarificationResponse
from constat.execution.mode import Mode, Phase, PlanApprovalRequest, PlanApprovalResponse, PlanApproval
from constat.core.config import Config
from constat.feedback import FeedbackDisplay, SessionFeedbackHandler, StatusLine, SPINNER_FRAMES
from constat.visualization.output import clear_pending_outputs, get_pending_outputs
from constat.storage.facts import FactStore
from constat.storage.learnings import LearningStore
from constat.proof_tree import ProofTree, NodeStatus


# Vera's personality adjectives (abbreviated list for space)
RELIABLE_ADJECTIVES = [
    "dependable", "reliable", "trustworthy", "steadfast", "unwavering",
    "rock-solid", "battle-tested", "bulletproof", "laser-focused",
]

HONEST_ADJECTIVES = [
    "honest", "truthful", "candid", "forthright", "sincere",
    "straight-shooting", "no-nonsense", "radically-transparent",
]


def get_vera_adjectives() -> tuple[str, str]:
    """Return a random pair of (reliable, honest) adjectives for Vera's intro."""
    return (random.choice(RELIABLE_ADJECTIVES), random.choice(HONEST_ADJECTIVES))


def terminal_supports_hyperlinks() -> bool:
    """Check if the terminal supports OSC 8 hyperlinks.

    Returns True for terminals known to support OSC 8 hyperlinks:
    - iTerm2
    - Windows Terminal
    - Konsole
    - GNOME Terminal 3.26+
    - kitty
    - WezTerm
    - Alacritty
    - foot
    """
    # Check for specific terminal emulators via environment variables
    term_program = os.environ.get("TERM_PROGRAM", "").lower()
    term = os.environ.get("TERM", "").lower()

    # iTerm2
    if term_program == "iterm.app" or os.environ.get("ITERM_SESSION_ID"):
        return True

    # Windows Terminal
    if os.environ.get("WT_SESSION"):
        return True

    # kitty
    if term_program == "kitty" or term == "xterm-kitty":
        return True

    # WezTerm
    if term_program == "wezterm":
        return True

    # Alacritty (supports hyperlinks since 0.11)
    if term_program == "alacritty" or term == "alacritty":
        return True

    # Konsole (check for KONSOLE_VERSION)
    if os.environ.get("KONSOLE_VERSION"):
        return True

    # GNOME Terminal (check for VTE_VERSION >= 5000 which is 3.26+)
    vte_version = os.environ.get("VTE_VERSION", "")
    if vte_version:
        try:
            if int(vte_version) >= 5000:
                return True
        except ValueError:
            pass

    # foot terminal
    if term == "foot" or term_program == "foot":
        return True

    # tmux passes through hyperlinks if the outer terminal supports them
    # Check if we're in tmux and the outer terminal supports hyperlinks
    if os.environ.get("TMUX"):
        # tmux itself may or may not pass through hyperlinks
        # We'll be conservative and return False
        return False

    # Default: assume no hyperlink support
    return False


# Cache the result since it won't change during runtime
_HYPERLINKS_SUPPORTED: bool | None = None


def supports_hyperlinks() -> bool:
    """Cached check for terminal hyperlink support."""
    global _HYPERLINKS_SUPPORTED
    if _HYPERLINKS_SUPPORTED is None:
        _HYPERLINKS_SUPPORTED = terminal_supports_hyperlinks()
        logger.debug(f"[HYPERLINK] Terminal support check: {_HYPERLINKS_SUPPORTED}, TERM_PROGRAM={os.environ.get('TERM_PROGRAM')}, ITERM_SESSION_ID={os.environ.get('ITERM_SESSION_ID')}")
    return _HYPERLINKS_SUPPORTED


def make_file_link_markup(file_uri: str, style: str = "cyan underline", indent: str = "") -> str:
    """Create Textual markup for a clickable file link.

    Uses Textual's @click action syntax to make the filename clickable.
    When clicked, opens the file with the system's default application.

    Use this for writing to RichLog (which has markup=True).
    For Static widgets, use make_file_text() instead.

    Args:
        file_uri: The file:// URI to open
        style: Rich style to apply (default: "cyan underline")
        indent: Optional indentation prefix

    Returns:
        Markup string with clickable filename
    """
    if not file_uri.startswith("file://"):
        file_uri = f"file://{file_uri}"

    # Extract filename for display
    try:
        filepath = file_uri.replace("file://", "")
        filename = Path(filepath).name
    except Exception:
        filename = file_uri.split("/")[-1]

    # Escape single quotes in the path for the action parameter
    escaped_uri = file_uri.replace("'", "\\'")

    # Create clickable markup using Textual's @click syntax
    # Format: [@click=app.open_file('path')][style]text[/style][/]
    return f"{indent}[@click=app.open_file('{escaped_uri}')][{style}]{filename}[/{style}][/]"


def make_file_text(file_uri: str, style: str = "cyan underline", indent: str = "") -> Text:
    """Create a Text object for a file URI display (for Static widgets).

    For RichLog with clickable links, use make_file_link_markup() instead.

    Args:
        file_uri: The file:// URI to display
        style: Rich style to apply
        indent: Optional indentation prefix

    Returns:
        Text object with filename
    """
    if not file_uri.startswith("file://"):
        file_uri = f"file://{file_uri}"

    try:
        filepath = file_uri.replace("file://", "")
        filename = Path(filepath).name
    except Exception:
        filename = file_uri.split("/")[-1]

    return Text(f"{indent}{filename}", style=style)


class StatusBar(Static):
    """Persistent status bar widget at the bottom of the terminal."""

    phase: reactive[Phase] = reactive(Phase.IDLE)
    status_message: reactive[str | None] = reactive(None)
    tables_count: reactive[int] = reactive(0)
    facts_count: reactive[int] = reactive(0)
    spinner_frame: reactive[int] = reactive(0)
    elapsed_time: reactive[str] = reactive("")  # Timer display
    panel_ratio: reactive[str] = reactive("4:1")  # Panel ratio display
    settings_display: reactive[str] = reactive("raw:on insights:on")  # Settings display

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._timer_start: float | None = None
        self._timer_interval = None
        self._final_time: str | None = None  # Stores final time after stop

    def start_timer(self) -> None:
        """Start the elapsed time timer."""
        import time
        self._timer_start = time.time()
        self._final_time = None
        self.elapsed_time = "0.0s"
        # Update timer every 100ms
        if self._timer_interval is None:
            self._timer_interval = self.set_interval(0.1, self._update_timer)

    def stop_timer(self) -> None:
        """Stop the timer and show final time."""
        import time
        if self._timer_start is not None:
            elapsed = time.time() - self._timer_start
            self._final_time = f"{elapsed:.1f}s"
            self.elapsed_time = self._final_time
        if self._timer_interval is not None:
            self._timer_interval.stop()
            self._timer_interval = None
        self._timer_start = None

    def _update_timer(self) -> None:
        """Update the elapsed time display."""
        import time
        if self._timer_start is not None:
            elapsed = time.time() - self._timer_start
            self.elapsed_time = f"{elapsed:.1f}s"

    DEFAULT_CSS = """
    StatusBar {
        height: 2;
        dock: bottom;
        background: $surface;
    }

    StatusBar > .status-rule {
        color: $text-muted;
    }

    StatusBar > .status-content {
        color: $text;
        padding: 0 1;
    }
    """

    def render(self) -> RenderableType:
        """Render the status bar.

        Layout: <interrupt hint> <timer> <status>
        - Interrupt hint: "(Ctrl+C to interrupt)" when processing, blank otherwise
        - Timer: fixed width (10 chars) showing elapsed/final time
        - Status: remaining space for status message
        """
        content = Text()

        # Determine if processing
        is_processing = bool(self.status_message) or self.phase in (Phase.PLANNING, Phase.EXECUTING)

        # 1. Interrupt hint (shown only when processing)
        if is_processing:
            content.append(" (Ctrl+C to interrupt)", style="dim")
        else:
            content.append(" " * 22)  # Same width as hint for alignment

        # 3. Timer (fixed width: 10 chars including brackets)
        if self.elapsed_time:
            timer_str = f"[{self.elapsed_time}]"
            style = "bold cyan" if is_processing else "dim green"
            content.append(f" {timer_str:>9}", style=style)
        elif self._final_time:
            timer_str = f"[{self._final_time}]"
            content.append(f" {timer_str:>9}", style="dim green")
        else:
            content.append(" " * 10)  # Blank timer space

        # 4. Status message (remaining space)
        content.append(" ")
        if self.status_message:
            spinner = SPINNER_FRAMES[self.spinner_frame % len(SPINNER_FRAMES)]
            content.append(f"{spinner} {self.status_message}", style="cyan")
        elif self.phase == Phase.IDLE:
            content.append("ready", style="dim")
        elif self.phase == Phase.PLANNING:
            spinner = SPINNER_FRAMES[self.spinner_frame % len(SPINNER_FRAMES)]
            content.append(f"{spinner} planning", style="cyan")
        elif self.phase == Phase.EXECUTING:
            spinner = SPINNER_FRAMES[self.spinner_frame % len(SPINNER_FRAMES)]
            content.append(f"{spinner} executing", style="green")
        elif self.phase == Phase.AWAITING_APPROVAL:
            content.append("awaiting approval", style="yellow")
        elif self.phase == Phase.FAILED:
            content.append("failed", style="bold red")

        # Calculate space for right-aligned elements
        terminal_width = shutil.get_terminal_size().columns
        rule_line = "─" * terminal_width

        # Settings display and panel controls
        settings_str = f" [{self.settings_display}]"
        panel_controls = f" [◀ {self.panel_ratio} ▶]"
        right_content = settings_str + panel_controls
        right_len = len(right_content)

        # Calculate padding to right-align
        current_len = len(content.plain)
        padding_needed = max(0, terminal_width - current_len - right_len)
        content.append(" " * padding_needed)
        content.append(settings_str, style="dim")
        content.append(panel_controls, style="dim cyan")

        return Text.assemble(
            (rule_line, "dim"),
            "\n",
            content,
        )

    # Sentinel for "not provided" vs "explicitly None"
    _NOT_PROVIDED = object()

    def update_status(
        self,
        phase: Phase = None,
        status_message: str | None = _NOT_PROVIDED,
        tables_count: int = None,
        facts_count: int = None,
    ) -> None:
        """Update status bar values.

        Note: Pass status_message=None to explicitly clear the message (show phase-based status).
        Omit status_message to leave it unchanged.
        """
        if phase is not None:
            self.phase = phase
        # Use sentinel to distinguish "not provided" from "explicitly None"
        if status_message is not StatusBar._NOT_PROVIDED:
            self.status_message = status_message
        if tables_count is not None:
            self.tables_count = tables_count
        if facts_count is not None:
            self.facts_count = facts_count

    def advance_spinner(self) -> None:
        """Advance the spinner animation."""
        self.spinner_frame = (self.spinner_frame + 1) % len(SPINNER_FRAMES)
        self.refresh()

    def on_click(self, event) -> None:
        """Handle clicks on panel resize controls."""
        # Check if click is in the panel controls region (right side)
        terminal_width = shutil.get_terminal_size().columns
        controls_text = f" [◀ {self.panel_ratio} ▶]"
        controls_start = terminal_width - len(controls_text)

        # event.x is the click position
        if event.x >= controls_start:
            # Clicked in controls region
            relative_x = event.x - controls_start
            # [◀ 3:1 ▶] - ◀ is at position 1, ▶ is near the end
            if relative_x <= 3:  # Left arrow region
                self.app.action_shrink_panel()
            elif relative_x >= len(controls_text) - 3:  # Right arrow region
                self.app.action_expand_panel()


class CommandSuggester(Suggester):
    """Provides auto-complete suggestions for REPL commands."""

    # All available commands
    COMMANDS = [
        "/help", "/h",
        "/tables",
        "/show",
        "/query",
        "/facts",
        "/state",
        "/reset",
        "/redo",
        "/artifacts",
        "/code",
        "/prove", "/audit",
        "/preferences",
        "/databases", "/database", "/db",
        "/files", "/file",
        "/context",
        "/history", "/sessions",
        "/verbose",
        "/raw",
        "/insights",
        "/update", "/refresh",
        "/learnings",
        "/consolidate",
        "/compact",
        "/remember",
        "/forget",
        "/correct",
        "/save",
        "/share",
        "/plans",
        "/replay",
        "/resume",
        "/export",
        "/summarize",
        "/prove",
        "/user",
        "/quit", "/q",
    ]

    async def get_suggestion(self, value: str) -> str | None:
        """Get a suggestion for the current input value."""
        if not value or not value.startswith("/"):
            return None

        value_lower = value.lower()
        for cmd in self.COMMANDS:
            if cmd.startswith(value_lower) and cmd != value_lower:
                return cmd
        return None


class ConstatInput(Input):
    """Input widget styled for Constat REPL with history and auto-complete."""

    DEFAULT_CSS = """
    ConstatInput {
        dock: bottom;
        margin: 0;
        padding: 0 1;
        background: $surface;
        border: none;
    }

    ConstatInput:focus {
        border: none;
    }
    """

    # Maximum history size
    MAX_HISTORY = 100

    def __init__(self, *args, **kwargs) -> None:
        # Initialize with command suggester for auto-complete
        kwargs.setdefault("suggester", CommandSuggester(use_cache=False, case_sensitive=False))
        super().__init__(*args, **kwargs)
        self._history: list[str] = []
        self._history_index: int = -1  # -1 means not browsing history
        self._current_input: str = ""  # Stores current input when browsing history

    def add_to_history(self, command: str) -> None:
        """Add a command to history (called after submission)."""
        if command and command.strip():
            # Don't add duplicates of the last command
            if not self._history or self._history[-1] != command:
                self._history.append(command)
                # Trim to max size
                if len(self._history) > self.MAX_HISTORY:
                    self._history = self._history[-self.MAX_HISTORY:]
        # Reset history navigation
        self._history_index = -1
        self._current_input = ""

    def _on_key(self, event: events.Key) -> None:
        """Handle key events for history navigation."""
        if event.key == "up":
            self._navigate_history(-1)
            event.prevent_default()
            event.stop()
        elif event.key == "down":
            self._navigate_history(1)
            event.prevent_default()
            event.stop()
        else:
            # Any other key resets history navigation
            if self._history_index != -1:
                self._history_index = -1
                self._current_input = ""

    def _navigate_history(self, direction: int) -> None:
        """Navigate through command history.

        Args:
            direction: -1 for older (up), 1 for newer (down)
        """
        if not self._history:
            return

        # Save current input when starting to browse
        if self._history_index == -1 and direction == -1:
            self._current_input = self.value
            self._history_index = len(self._history)

        # Calculate new index
        new_index = self._history_index + direction

        if new_index < 0:
            # Already at oldest
            return
        elif new_index >= len(self._history):
            # Return to current input
            self._history_index = -1
            self.value = self._current_input
            self.cursor_position = len(self.value)
            return

        # Update to history entry
        self._history_index = new_index
        self.value = self._history[new_index]
        self.cursor_position = len(self.value)

    # Note: Uses Input.Submitted from parent class


class OutputLog(RichLog):
    """Scrollable output area for Rich content."""

    DEFAULT_CSS = """
    OutputLog {
        scrollbar-gutter: stable;
        padding: 0 1;
        overflow-x: auto;  /* Enable horizontal scrolling for long URIs */
    }
    """


class SidePanelContent(RichLog):
    """Panel that shows DFD during approval, proof tree during execution, artifacts after.

    Uses RichLog to support clickable file links via Textual's @click action syntax.
    Uses the actual ProofTree class from constat.proof_tree for proper hierarchical display.
    """

    DEFAULT_CSS = """
    SidePanelContent {
        width: 100%;
        height: 100%;
        scrollbar-gutter: stable;
    }
    """

    # Panel modes
    MODE_DFD = "dfd"
    MODE_PLAN = "plan"  # Plan approval display (no box)
    MODE_PROOF_TREE = "proof_tree"
    MODE_ARTIFACTS = "artifacts"
    MODE_STEPS = "steps"  # Exploratory mode step tracking

    # Spinner frames for animation
    SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    PROGRESS_BAR_CHARS = ["░", "▒", "▓", "█"]

    def __init__(self, **kwargs) -> None:
        # Initialize RichLog with markup support for clickable links
        super().__init__(highlight=False, markup=True, wrap=True, **kwargs)
        self._mode = self.MODE_DFD
        self._proof_tree: Optional[ProofTree] = None  # Use actual ProofTree class
        self._dag_lines: list[str] = []
        self._artifacts: list[dict] = []  # {type, name, description, command}
        # Exploratory mode step tracking
        self._steps: list[dict] = []  # {number, goal, status, result}
        self._current_step: int = 0
        # Animation state
        self._spinner_frame: int = 0
        self._animation_timer = None
        self._is_animating: bool = False
        self._pulse_state: bool = False  # For pulse effect

    def start_animation(self) -> None:
        """Start the spinner animation for active steps."""
        logger.debug(f"[ANIM] start_animation called, already_animating={self._is_animating}")
        if not self._is_animating:
            self._is_animating = True
            self._animation_timer = self.set_interval(0.1, self._animate_tick)
            logger.debug("[ANIM] Animation timer started")
            # Add executing class to side panel (changes border color)
            try:
                side_panel = self.app.query_one("#side-panel")
                side_panel.add_class("executing")
            except Exception:
                pass

    def stop_animation(self) -> None:
        """Stop the spinner animation."""
        if self._is_animating:
            self._is_animating = False
            if self._animation_timer:
                self._animation_timer.stop()
                self._animation_timer = None
            # Remove executing class
            try:
                side_panel = self.app.query_one("#side-panel")
                side_panel.remove_class("executing")
            except Exception:
                pass

    def _animate_tick(self) -> None:
        """Called on each animation tick to update spinner."""
        self._spinner_frame = (self._spinner_frame + 1) % len(self.SPINNER_FRAMES)
        self._pulse_state = not self._pulse_state
        # Only redraw if we're in step mode with active steps
        has_active = any(s.get("status") in ("in_progress", "executing") for s in self._steps)
        logger.debug(f"[ANIM] tick frame={self._spinner_frame}, mode={self._mode}, has_active={has_active}, steps={len(self._steps)}")
        if self._mode == self.MODE_STEPS and has_active:
            self._update_display()

    def get_spinner(self) -> str:
        """Get current spinner character."""
        return self.SPINNER_FRAMES[self._spinner_frame]

    def get_progress_bar(self, progress: float, width: int = 10) -> str:
        """Generate an animated progress bar."""
        filled = int(progress * width)
        remainder = progress * width - filled
        bar = "█" * filled
        if filled < width:
            # Animated partial fill
            partial_idx = int(remainder * len(self.PROGRESS_BAR_CHARS))
            bar += self.PROGRESS_BAR_CHARS[min(partial_idx, len(self.PROGRESS_BAR_CHARS) - 1)]
            bar += "░" * (width - filled - 1)
        return bar

    def show_dfd(self, dag_lines: list[str]) -> None:
        """Show only the DFD diagram (during approval)."""
        self._dag_lines = dag_lines
        self._mode = self.MODE_DFD
        self._update_display()

    def show_plan(self, dag_lines: list[str]) -> None:
        """Show plan for approval (no box, just content)."""
        self._dag_lines = dag_lines
        self._mode = self.MODE_PLAN
        self._update_display()

    def start_proof_tree(self, conclusion: str = "") -> None:
        """Start showing proof tree (during execution)."""
        # Create actual ProofTree instance
        self._proof_tree = ProofTree("answer", conclusion)
        self._mode = self.MODE_PROOF_TREE
        self._update_display()

    def add_fact(self, name: str, description: str = "", parent_name: str = None, dependencies: list[str] = None) -> None:
        """Add a fact to the proof tree."""
        if self._proof_tree:
            self._proof_tree.add_fact(name, description, parent_name)
            if dependencies:
                self._proof_tree.set_dependencies(name, dependencies)
            if self._mode == self.MODE_PROOF_TREE:
                self._update_display()

    def update_resolving(self, name: str, description: str = "", parent_name: str = None) -> None:
        """Mark a fact as being resolved."""
        if self._proof_tree:
            self._proof_tree.start_resolving(name, description, parent_name)
            if self._mode == self.MODE_PROOF_TREE:
                self._update_display()

    def update_resolved(self, name: str, value, source: str = "", confidence: float = 1.0,
                       result_summary: str = "", from_cache: bool = False) -> None:
        """Mark a fact as resolved."""
        if self._proof_tree:
            self._proof_tree.resolve_fact(name, value, source, confidence,
                                          result_summary=result_summary, from_cache=from_cache)
            if self._mode == self.MODE_PROOF_TREE:
                self._update_display()

    def update_failed(self, name: str, error: str) -> None:
        """Mark a fact as failed."""
        if self._proof_tree:
            self._proof_tree.fail_fact(name, error)
            if self._mode == self.MODE_PROOF_TREE:
                self._update_display()

    def reset(self) -> None:
        """Reset panel to initial state."""
        self.stop_animation()  # Stop any running animation
        self._proof_tree = None
        self._dag_lines = []
        self._artifacts = []
        self._steps = []
        self._current_step = 0
        self._mode = self.MODE_DFD  # Reset to default mode
        super().clear()  # RichLog.clear()

    def clear_panel(self) -> None:
        """Clear the panel content and state (alias for reset)."""
        self.reset()

    def show_artifacts(self, artifacts: list[dict]) -> None:
        """Show artifacts after execution completes."""
        self._artifacts = artifacts
        self._mode = self.MODE_ARTIFACTS
        self._update_display()

    # Exploratory mode step tracking
    def start_steps(self, steps: list[dict]) -> None:
        """Initialize step tracking for exploratory mode.

        Steps can have a 'completed' flag to show them as already done (for follow-ups).
        """
        import time
        self._steps = []
        for i, step in enumerate(steps):
            is_completed = step.get("completed", False)
            self._steps.append({
                "number": step.get("number", i + 1),
                "goal": step.get("goal", ""),
                "status": "complete" if is_completed else "pending",
                "result": None,
                "start_time": None,
                "elapsed": None,
                "retries": 0,
            })
        self._current_step = 0
        self._mode = self.MODE_STEPS
        self._update_display()

    def extend_steps(self, steps: list[dict]) -> None:
        """Add new steps to existing step list (for follow-up/extension queries).

        Preserves completed steps and adds new ones with renumbered step numbers.
        """
        import time
        # Calculate starting number from existing steps
        start_num = max((s["number"] for s in self._steps), default=0) + 1

        for i, step in enumerate(steps):
            self._steps.append({
                "number": start_num + i,
                "goal": step.get("goal", ""),
                "status": "pending",
                "result": None,
                "start_time": None,
                "elapsed": None,
                "retries": 0,
            })
        self._mode = self.MODE_STEPS
        self._update_display()

    def update_step_executing(self, step_num: int, retry: bool = False) -> None:
        """Mark a step as currently executing."""
        import time
        self._current_step = step_num
        for step in self._steps:
            if step["number"] == step_num:
                step["status"] = "executing"
                if retry:
                    step["retries"] = step.get("retries", 0) + 1
                else:
                    step["start_time"] = time.time()
                break
        # Start animation for executing steps
        self.start_animation()
        if self._mode == self.MODE_STEPS:
            self._update_display()

    def update_step_complete(self, step_num: int, result: str = "") -> None:
        """Mark a step as complete with optional result."""
        import time
        for step in self._steps:
            if step["number"] == step_num:
                step["status"] = "complete"
                step["result"] = result
                if step["start_time"]:
                    step["elapsed"] = time.time() - step["start_time"]
                break
        # Check if all steps are done - stop animation
        all_done = all(s.get("status") in ("complete", "failed") for s in self._steps)
        if all_done:
            self.stop_animation()
        if self._mode == self.MODE_STEPS:
            self._update_display()

    def update_step_failed(self, step_num: int, error: str = "") -> None:
        """Mark a step as failed."""
        import time
        for step in self._steps:
            if step["number"] == step_num:
                step["status"] = "failed"
                step["result"] = f"Error: {error}"
                if step["start_time"]:
                    step["elapsed"] = time.time() - step["start_time"]
                break
        # Check if all steps are done - stop animation
        all_done = all(s.get("status") in ("complete", "failed") for s in self._steps)
        if all_done:
            self.stop_animation()
        if self._mode == self.MODE_STEPS:
            self._update_display()

    def _update_display(self) -> None:
        """Update panel content based on mode."""
        # Update border title based on mode
        mode_titles = {
            self.MODE_DFD: "Data Flow",
            self.MODE_PLAN: "Plan",
            self.MODE_PROOF_TREE: "Proof Tree",
            self.MODE_ARTIFACTS: "Artifacts",
            self.MODE_STEPS: "Steps",
        }
        try:
            side_panel = self.app.query_one("#side-panel", SidePanel)
            side_panel.border_title = mode_titles.get(self._mode, "Panel")
        except Exception:
            pass

        if self._mode == self.MODE_ARTIFACTS:
            self._render_artifacts()
        elif self._mode == self.MODE_DFD:
            self._render_dfd()
        elif self._mode == self.MODE_PLAN:
            self._render_plan()
        elif self._mode == self.MODE_STEPS:
            self._render_steps()
        else:
            self._render_proof_tree()

    def _render_dfd(self) -> None:
        """Render DFD diagram or plan steps."""
        super().clear()

        if not self._dag_lines:
            content = Text()
            content.append("No data flow diagram.\n", style="dim")
            self.write(content)
            return

        # Check if this is simple step list (lines start with P/I prefix like "P P1: ...")
        first_line = self._dag_lines[0].strip() if self._dag_lines else ""
        is_step_list = first_line and (first_line.startswith("P ") or first_line.startswith("I "))

        if is_step_list:
            # Render step list with text wrapping
            import textwrap
            content_width = self.content_size.width - 3  # -3 for accurate wrapping
            if content_width <= 0:
                # Widget not laid out yet - calculate from app size
                try:
                    app = self.app
                    ratio_index = getattr(app, '_panel_ratio_index', 1)
                    ratios = getattr(app, 'PANEL_RATIOS', [(3, 1), (2, 1), (1, 1), (1, 2)])
                    output_ratio, side_ratio = ratios[ratio_index]
                    total_ratio = output_ratio + side_ratio
                    app_width = app.size.width if app.size.width > 0 else 120
                    panel_width = (app_width * side_ratio) // total_ratio
                    content_width = panel_width - 6
                except Exception as e:
                    content_width = 40  # Reasonable default
            content_width = max(20, content_width)
            for line in self._dag_lines:
                if line.strip():  # Skip empty lines
                    # Lines have prefix like "P P1: " or "I I1: " - detect and account for it
                    import re
                    prefix_match = re.match(r'^([PI]\s+[PI]\d+:\s*)', line)
                    if prefix_match:
                        prefix = prefix_match.group(1)
                        rest = line[len(prefix):]
                        first_width = content_width - len(prefix)
                        cont_width = content_width - 4  # indent for continuation
                        if len(rest) <= first_width:
                            self.write(f"[white]{line}[/white]")
                        else:
                            wrapped = textwrap.wrap(rest, width=first_width)
                            self.write(f"[white]{prefix}{wrapped[0]}[/white]")
                            remainder = rest[len(wrapped[0]):].strip()
                            if remainder:
                                cont_wrapped = textwrap.wrap(remainder, width=cont_width)
                                for cont in cont_wrapped:
                                    self.write(f"[white]    {cont}[/white]")
                    else:
                        # No prefix detected, wrap normally
                        wrapped_lines = textwrap.wrap(line, width=content_width)
                        for wrapped in wrapped_lines:
                            self.write(f"[white]{wrapped}[/white]")
        else:
            # Render actual DFD in a centered box
            from rich.panel import Panel
            from rich.align import Align

            dfd_text = Text()
            for line in self._dag_lines:
                dfd_text.append(f"{line}\n", style="white")

            panel = Panel(
                dfd_text,
                title="DATA FLOW",
                title_align="center",
                border_style="cyan",
                padding=(0, 1),
            )

            centered = Align.center(panel)
            self.write(centered)

    def _render_plan(self) -> None:
        """Render plan for approval (no box, just content)."""
        super().clear()

        if not self._dag_lines:
            content = Text()
            content.append("No plan to display.\n", style="dim")
            self.write(content)
            return

        # Render plan content directly without a box
        for line in self._dag_lines:
            if line.strip():
                self.write(Text(line, style="white"))

    def _render_artifacts(self) -> None:
        """Render artifact links with clickable file:// URIs.

        Each artifact on one line: icon + name (as link) + row count
        execution_history shown first with blank line after, then others in creation order.
        """
        super().clear()

        if not self._artifacts:
            self.write(Text("No artifacts created.", style="dim"))
        else:
            # Separate execution_history from other artifacts
            exec_history = [a for a in self._artifacts if a.get("name") == "execution_history"]
            other_artifacts = [a for a in self._artifacts if a.get("name") != "execution_history"]

            # Render execution_history first (if present)
            for artifact in exec_history:
                self._write_artifact_line(artifact)
            if exec_history:
                self.write("")  # Blank line after execution_history

            # Then render other artifacts in creation order
            for artifact in other_artifacts:
                self._write_artifact_line(artifact)

    def _write_artifact_line(self, artifact: dict) -> None:
        """Write a single artifact line with icon, name link, and row count."""
        artifact_type = artifact.get("type", "unknown")
        name = artifact.get("name", "Unknown")
        row_count = artifact.get("row_count")
        file_uri = artifact.get("file_uri", "")

        # Icon based on type
        if artifact_type == "table":
            icon = "📊"
        elif artifact_type == "chart":
            icon = "📈"
        elif artifact_type == "file":
            icon = "📄"
        else:
            icon = "📦"

        # Build one-line display using markup for clickable links
        row_suffix = f" [dim]({row_count} rows)[/dim]" if row_count is not None else ""

        if file_uri:
            # Escape single quotes in path for @click action
            escaped_uri = file_uri.replace("'", "\\'")
            # Use @click markup to make the name clickable
            line = f"{icon} [@click=app.open_file('{escaped_uri}')][cyan underline]{name}[/cyan underline][/]{row_suffix}"
        else:
            line = f"{icon} [green]{name}[/green]{row_suffix}"

        self.write(line)

    def _render_proof_tree(self) -> None:
        """Render the proof tree using the actual ProofTree class."""
        super().clear()

        if not self._proof_tree:
            self.write(Text("PROOF TREE", style="bold yellow"))
            self.write(Text("\nWaiting for resolution...", style="dim"))
            return

        # Use ProofTree's render method - it returns a proper Rich Tree
        tree = self._proof_tree.render()
        self.write(tree)

    def _render_steps(self) -> None:
        """Render exploratory mode step progress."""
        super().clear()

        if not self._steps:
            self.write(Text("Waiting for plan...", style="dim"))
            return

        for step in self._steps:
            num = step["number"]
            goal = step["goal"]
            status = step["status"]
            result = step.get("result", "")
            elapsed = step.get("elapsed")
            retries = step.get("retries", 0)

            # Status icon with animation for executing steps
            if status == "pending":
                icon = "○"
                style = "dim"
            elif status == "executing" or status == "in_progress":
                icon = self.get_spinner()  # Animated spinner
                style = "bold cyan" if self._pulse_state else "cyan"
            elif status == "complete":
                icon = "✓"
                style = "green"
            elif status == "failed":
                icon = "✗"
                style = "red"
            else:
                icon = "?"
                style = "dim"

            # Step header: icon, number, goal (full text, wraps at panel width)
            # Get content width - content_size may be 0 if not laid out yet
            import textwrap
            content_width = self.content_size.width - 3  # -3 for accurate wrapping
            if content_width <= 0:
                # Widget not laid out yet - calculate from app size
                try:
                    app = self.app
                    ratio_index = getattr(app, '_panel_ratio_index', 1)
                    ratios = getattr(app, 'PANEL_RATIOS', [(3, 1), (2, 1), (1, 1), (1, 2)])
                    output_ratio, side_ratio = ratios[ratio_index]
                    total_ratio = output_ratio + side_ratio
                    app_width = app.size.width if app.size.width > 0 else 120
                    panel_width = (app_width * side_ratio) // total_ratio
                    content_width = panel_width - 6  # border + padding + extra
                except Exception:
                    content_width = 40
            content_width = max(20, content_width)
            prefix = f"{icon} Step {num}: "
            # Wrap goal text, accounting for prefix on first line
            first_line_width = content_width - len(prefix)
            continuation_width = content_width - 2  # for "  " indent
            if len(goal) <= first_line_width:
                self.write(f"[{style}]{prefix}[bold]{goal}[/bold][/{style}]")
            else:
                # Wrap first line shorter to account for prefix
                wrapped = textwrap.wrap(goal, width=first_line_width)
                self.write(f"[{style}]{prefix}[bold]{wrapped[0]}[/bold][/{style}]")
                # Re-wrap remainder for continuation lines (slightly wider)
                remainder = goal[len(wrapped[0]):].strip()
                if remainder:
                    cont_wrapped = textwrap.wrap(remainder, width=continuation_width)
                    for continuation in cont_wrapped:
                        self.write(f"[{style}]  {continuation}[/{style}]")

            # Show timing and retries for non-pending steps
            meta_parts = []
            if elapsed is not None:
                meta_parts.append(f"{elapsed:.1f}s")
            if retries > 0:
                meta_parts.append(f"{retries} retries")
            if meta_parts:
                self.write(Text(f"   [{', '.join(meta_parts)}]", style="dim"))

            # Show result summary for complete/failed steps
            if result and status in ("complete", "failed"):
                result_display = str(result)[:60]
                if len(str(result)) > 60:
                    result_display += "..."
                result_style = "dim green" if status == "complete" else "dim red"
                self.write(Text(f"   → {result_display}", style=result_style))

            self.write("")


# Alias for backwards compatibility
ProofTreePanel = SidePanelContent


class SidePanel(ScrollableContainer):
    """Collapsible side panel for DFD and proof tree."""
    # CSS is defined in ConstatREPLApp.CSS for centralized styling
    pass


class ShowApprovalUI(Message):
    """Message to trigger showing the approval UI on the main thread."""
    pass


class ShowClarificationUI(Message):
    """Message to trigger showing the clarification UI on the main thread."""
    pass


class SolveComplete(Message):
    """Message posted when solve operation completes."""
    def __init__(self, result: dict) -> None:
        self.result = result
        super().__init__()


class ProveComplete(Message):
    """Message posted when prove operation completes."""
    def __init__(self, result: dict) -> None:
        self.result = result
        super().__init__()


class ConsolidateComplete(Message):
    """Message posted when consolidate operation completes."""
    def __init__(self, result: dict) -> None:
        self.result = result
        super().__init__()


class SessionEvent(Message):
    """Message posted when a session event occurs (proof tree, steps, etc.)."""
    def __init__(self, event) -> None:
        self.event = event
        super().__init__()


class TextualFeedbackHandler:
    """
    Event handler that updates Textual UI during session execution.

    Receives events from Session and updates the status bar and log.
    """

    def __init__(self, app: "ConstatREPLApp"):
        self.app = app
        self._proof_items: dict[str, dict] = {}  # Track proof tree items
        self._current_step = 0
        self._total_steps = 0
        self._steps_initialized = False  # Track if steps panel has been initialized

    def _get_log(self) -> OutputLog:
        """Get the output log widget."""
        return self.app.query_one("#output-log", OutputLog)

    def _get_status_bar(self) -> StatusBar:
        """Get the status bar widget."""
        return self.app.query_one("#status-bar", StatusBar)

    def _get_side_panel(self) -> SidePanel:
        """Get the side panel widget."""
        return self.app.query_one("#side-panel", SidePanel)

    def _get_panel_content(self) -> SidePanelContent:
        """Get the side panel content widget."""
        return self.app.query_one("#proof-tree-panel", SidePanelContent)

    def handle_event(self, event) -> None:
        """Handle a StepEvent from Session (called from background thread)."""
        # Post message directly - thread-safe in Textual (like memray pattern)
        self.app.post_message(SessionEvent(event))

    def _handle_event_on_main(self, event) -> None:
        """Handle event on main thread where UI updates are safe."""
        event_type = event.event_type
        data = event.data
        log = self._get_log()
        status_bar = self._get_status_bar()

        # Generic progress events (early-stage operations)
        if event_type == "progress":
            message = data.get("message", "Processing...")
            status_bar.update_status(status_message=message)

        # Clarification events
        elif event_type == "clarification_needed":
            status_bar.update_status(status_message="Clarification needed...")

        # Planning events
        elif event_type == "planning_start":
            status_bar.update_status(status_message="Planning approach...", phase=Phase.PLANNING)

        elif event_type == "planning_complete":
            status_bar.update_status(status_message="Plan complete, validating...")

        # Plan validation events
        elif event_type == "plan_validating":
            attempt = data.get("attempt", 1)
            max_attempts = data.get("max_attempts", 3)
            premises = data.get("premises_count", 0)
            inferences = data.get("inferences_count", 0)
            if attempt == 1:
                status_bar.update_status(status_message=f"Validating plan ({premises} premises, {inferences} inferences)...")
            else:
                status_bar.update_status(status_message=f"Validating plan (attempt {attempt}/{max_attempts})...")

        elif event_type == "plan_validated":
            premises = data.get("premises_count", 0)
            inferences = data.get("inferences_count", 0)
            status_bar.update_status(status_message=f"Plan validated ({premises}P, {inferences}I)")

        elif event_type == "plan_validation_failed":
            attempt = data.get("attempt", 1)
            max_attempts = data.get("max_attempts", 3)
            error_count = data.get("error_count", 0)
            error_summary = data.get("error_summary", "errors")
            will_retry = data.get("will_retry", False)

            if will_retry:
                status_bar.update_status(
                    status_message=f"Plan validation failed ({error_count} errors: {error_summary}), will retry..."
                )
                log.write(Text(f"  Plan validation failed (attempt {attempt}/{max_attempts}): {error_summary}", style="yellow"))
            else:
                status_bar.update_status(status_message=f"Plan validation failed after {max_attempts} attempts")
                log.write(Text(f"  Plan validation failed after {max_attempts} attempts: {error_summary}", style="red"))

        elif event_type == "plan_regenerating":
            attempt = data.get("attempt", 1)
            max_attempts = data.get("max_attempts", 3)
            reason = data.get("reason", "validation errors")
            fixing = data.get("fixing_errors", [])
            fixing_str = ", ".join(fixing[:3]) if fixing else reason

            status_bar.update_status(
                status_message=f"Regenerating plan (attempt {attempt}/{max_attempts}) - fixing: {fixing_str}"
            )
            log.write(Text(f"  Regenerating plan to fix: {fixing_str}", style="cyan"))

        elif event_type == "plan_ready":
            plan = data.get("plan", {})
            goal = plan.get("goal", "")
            steps = data.get("steps", [])
            is_followup = data.get("is_followup", False)

            # Use app's plan step tracking for follow-up support
            app = self.app

            # For follow-ups, merge completed steps with new steps
            if is_followup and app._completed_plan_steps:
                # Mark completed steps
                for s in app._completed_plan_steps:
                    s["completed"] = True
                # Combine: completed steps + new steps
                all_steps = app._completed_plan_steps + [
                    {"number": s.get("number", i + 1), "goal": s.get("goal", str(s)), "completed": False}
                    for i, s in enumerate(steps)
                ]
                app._plan_steps = all_steps
            else:
                # New plan - reset completed steps
                app._completed_plan_steps = []
                app._plan_steps = [
                    {"number": s.get("number", i + 1), "goal": s.get("goal", str(s)), "completed": False}
                    for i, s in enumerate(steps)
                ]

            logger.debug(f"plan_ready: set app._plan_steps with {len(app._plan_steps)} steps, resetting _steps_initialized")
            self._steps_initialized = False  # Reset for new plan
            # Don't set status_message here - the approval UI will handle it
            # Just update phase (approval callback will clear spinner)
            status_bar.update_status(phase=Phase.AWAITING_APPROVAL)
            # Show plan in log
            log.write(Text(goal, style="bold cyan"))

            # Show completed steps first (for follow-ups)
            if is_followup and app._completed_plan_steps:
                log.write(Text("  Completed:", style="dim green"))
                for s in app._completed_plan_steps:
                    log.write(Text(f"    \u2713 {s.get('number', '?')}. {s.get('goal', '')}", style="dim green"))
                log.write(Text("  New steps:", style="cyan"))

            for i, step in enumerate(steps):
                num = step.get("number", i + 1)
                log.write(Text(f"    {num}. {step.get('goal', step)}", style="dim"))

        # Proof tree events
        elif event_type in ("proof_tree_start", "proof_start"):
            # proof_start is emitted when proof execution begins
            logger.debug(f"Handling proof_start event: {data}")
            conclusion_fact = data.get("conclusion_fact", "")
            conclusion_desc = data.get("conclusion_description", "")
            # Restart timer for execution phase
            status_bar.start_timer()
            status_bar.update_status(status_message="Resolving proof tree...", phase=Phase.EXECUTING)

            # Switch to proof tree mode (replaces DFD)
            side_panel = self._get_side_panel()
            panel_content = self._get_panel_content()
            logger.debug(f"proof_start: switching to proof tree mode")
            panel_content.start_proof_tree(conclusion_desc)
            side_panel.add_class("visible")

        elif event_type == "dag_execution_start":
            # Pre-build the ENTIRE proof tree structure from premises and inferences
            # This shows all nodes upfront in the tree, then they animate as they resolve
            # Algorithm from original feedback.py: dependencies are CHILDREN of what uses them
            premises = data.get("premises", [])
            inferences = data.get("inferences", [])
            logger.debug(f"dag_execution_start: {len(premises)} premises, {len(inferences)} inferences")

            panel_content = self._get_panel_content()
            if not panel_content._proof_tree:
                logger.debug("dag_execution_start: no proof tree initialized, skipping")
                return

            # Build name -> fact_id and fact_id -> dependencies maps
            import re
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
                deps = re.findall(r'[PI]\d+', op)
                id_to_deps[fact_id] = deps
                logger.debug(f"dag_execution_start: {fact_id} deps={deps} from op={op[:50]}")

            # Find terminal inference (the one not used by any other inference)
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

            # Add terminal inference as child of root (answer)
            if terminal:
                terminal_name = next((inf.get("name", inf.get("id")) for inf in inferences if inf.get("id") == terminal), terminal)
                terminal_deps = id_to_deps.get(terminal, [])
                panel_content.add_fact(f"{terminal}: {terminal_name}", "", parent_name="answer", dependencies=terminal_deps)
                logger.debug(f"dag_execution_start: added terminal {terminal} under root, deps={terminal_deps}")

            # BFS from terminal to build tree (each node's children are its dependencies)
            added = {"answer", terminal} if terminal else {"answer"}
            queue = [terminal] if terminal else []

            while queue:
                current_id = queue.pop(0)
                current_deps = id_to_deps.get(current_id, [])

                for dep_id in current_deps:
                    if dep_id not in added:
                        # Find name for this dependency
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

                        # Find parent name (current_id uses dep_id, so dep is child of current)
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
                        # Get all dependencies for this node (to show non-visual deps)
                        node_deps = id_to_deps.get(dep_id, [])
                        panel_content.add_fact(f"{dep_id}: {dep_name}", "", parent_name=parent_key, dependencies=node_deps)
                        logger.debug(f"dag_execution_start: added {dep_id} under {current_id}, deps={node_deps}")
                        added.add(dep_id)
                        queue.append(dep_id)

            logger.debug(f"dag_execution_start: pre-built tree with {len(added)} nodes")

        elif event_type == "premise_resolving":
            # Events use fact_name and description
            fact_name = data.get("fact_name", "") or data.get("fact_id", "")
            description = data.get("description", "")
            logger.debug(f"premise_resolving: fact_name={fact_name}")
            status_bar.update_status(status_message=f"Resolving {fact_name[:40]}...")
            # Update proof tree - mark as resolving
            panel_content = self._get_panel_content()
            panel_content.update_resolving(fact_name, description)

        # SQL generation events (for database premises)
        elif event_type == "sql_generating":
            fact_name = data.get("fact_name", "")
            db = data.get("database", "")
            attempt = data.get("attempt", 1)
            max_attempts = data.get("max_attempts", 7)
            is_retry = data.get("is_retry", False)
            retry_reason = data.get("retry_reason", "")

            if is_retry:
                reason_short = retry_reason[:25] + "..." if retry_reason and len(retry_reason) > 25 else retry_reason or "error"
                status_bar.update_status(
                    status_message=f"Regenerating SQL for {fact_name[:25]}... (attempt {attempt}/{max_attempts}: {reason_short})"
                )
            else:
                status_bar.update_status(status_message=f"Generating SQL for {fact_name[:30]}... ({db})")

        elif event_type == "sql_executing":
            fact_name = data.get("fact_name", "")
            db = data.get("database", "")
            attempt = data.get("attempt", 1)
            max_attempts = data.get("max_attempts", 7)

            if attempt > 1:
                status_bar.update_status(status_message=f"Executing SQL retry {attempt}/{max_attempts} for {fact_name[:25]}...")
            else:
                status_bar.update_status(status_message=f"Executing SQL for {fact_name[:30]}...")

        elif event_type == "sql_error":
            fact_name = data.get("fact_name", "")
            error = data.get("error", "")
            attempt = data.get("attempt", 1)
            max_attempts = data.get("max_attempts", 7)
            will_retry = data.get("will_retry", False)

            error_short = error[:40] + "..." if len(error) > 40 else error
            if will_retry:
                status_bar.update_status(
                    status_message=f"SQL error for {fact_name[:20]} (attempt {attempt}/{max_attempts}), retrying..."
                )
                log.write(Text(f"  SQL attempt {attempt} failed for {fact_name}: {error_short}", style="yellow"))
            else:
                status_bar.update_status(status_message=f"SQL failed for {fact_name[:25]} after {max_attempts} attempts")
                log.write(Text(f"  SQL failed after {max_attempts} attempts: {error_short}", style="red"))

        elif event_type == "premise_resolved":
            # Events use fact_name and value
            fact_name = data.get("fact_name", "") or data.get("fact_id", "")
            value = data.get("value", "")
            source = data.get("source", "")
            confidence = data.get("confidence", 1.0)
            from_cache = source == "cache"
            logger.debug(f"premise_resolved: fact_name={fact_name}, value={str(value)[:30]}")
            # Update proof tree - mark as resolved
            panel_content = self._get_panel_content()
            panel_content.update_resolved(fact_name, value, source, confidence, from_cache=from_cache)

        elif event_type in ("inference_resolving", "inference_executing"):
            # Events use inference_id and operation
            inference_id = data.get("inference_id", "") or data.get("fact_id", "")
            operation = data.get("operation", "") or data.get("name", "")
            display_name = f"{inference_id}: {operation}" if operation else inference_id
            logger.debug(f"inference_executing: {display_name}")
            status_bar.update_status(status_message=f"Computing {display_name[:40]}...")
            # Update proof tree - mark as resolving
            panel_content = self._get_panel_content()
            panel_content.update_resolving(display_name, operation)

        elif event_type in ("inference_resolved", "inference_complete"):
            # Events use inference_id/inference_name and result
            inference_id = data.get("inference_id", "") or data.get("fact_id", "")
            inference_name = data.get("inference_name", "") or data.get("operation", "")
            # Handle result carefully - it may be a DataFrame which can't use 'or'
            result = data.get("result")
            if result is None:
                result = data.get("value", "")
            display_name = f"{inference_id}: {inference_name}" if inference_name else inference_id
            logger.debug(f"inference_complete: {display_name}, result={str(result)[:30]}")
            # Update proof tree - mark as resolved
            panel_content = self._get_panel_content()
            panel_content.update_resolved(display_name, result, source="derived", confidence=1.0)

        elif event_type == "proof_tree_complete":
            status_bar.update_status(status_message="Generating insights...", phase=Phase.EXECUTING)
            # Keep side panel visible until solve complete

        # Step execution events (for exploratory mode)
        elif event_type == "step_start":
            step_num = event.step_number or 0
            goal = data.get("goal", "")
            self._current_step = step_num
            status_bar.update_status(
                status_message=f"Step {step_num}: {goal[:40]}...",
                phase=Phase.EXECUTING
            )

            # Initialize steps panel on first step
            panel_content = self._get_panel_content()
            side_panel = self._get_side_panel()
            logger.debug(f"step_start: step={step_num}, _steps_initialized={self._steps_initialized}, _plan_steps={len(self.app._plan_steps) if self.app._plan_steps else 0}")
            if not self._steps_initialized and self.app._plan_steps:
                logger.debug(f"step_start: calling start_steps with {len(self.app._plan_steps)} steps")
                panel_content.start_steps(self.app._plan_steps)
                side_panel.add_class("visible")
                self._steps_initialized = True
            else:
                logger.debug(f"step_start: NOT calling start_steps - initialized={self._steps_initialized}, plan_steps_exist={bool(self.app._plan_steps)}")

            # Update step status in panel
            panel_content.update_step_executing(step_num)

        elif event_type == "step_complete":
            step_num = event.step_number or 0
            result = data.get("result", data.get("stdout", ""))
            # Truncate result for display
            result_summary = str(result)[:100] if result else ""
            # Don't show in main panel - side panel is sufficient

            # Update step status in panel
            panel_content = self._get_panel_content()
            panel_content.update_step_complete(step_num, result_summary)

        elif event_type == "step_failed":
            step_num = event.step_number or 0
            error = data.get("error", "Unknown error")
            log.write(Text(f"  Step {step_num} failed: {error}", style="red"))

            # Update step status in panel
            panel_content = self._get_panel_content()
            panel_content.update_step_failed(step_num, error[:100] if error else "")

        # Code generation/execution events
        elif event_type == "generating":
            step_num = event.step_number or 0
            attempt = data.get("attempt", 1)
            max_attempts = data.get("max_attempts", 10)
            is_retry = data.get("is_retry", False)
            retry_reason = data.get("retry_reason", "")

            if is_retry:
                # Show detailed retry message
                reason_short = retry_reason[:30] + "..." if retry_reason and len(retry_reason) > 30 else retry_reason
                status_bar.update_status(
                    status_message=f"Step {step_num}: Regenerating code (attempt {attempt}/{max_attempts}) - {reason_short}" if reason_short
                    else f"Step {step_num}: Regenerating code (attempt {attempt}/{max_attempts})"
                )
                # Track retry in panel
                panel_content = self._get_panel_content()
                panel_content.update_step_executing(step_num, retry=True)
            else:
                status_bar.update_status(status_message=f"Step {step_num}: Generating code...")

        elif event_type == "executing":
            step_num = event.step_number or 0
            attempt = data.get("attempt", 1)
            max_attempts = data.get("max_attempts", 10)
            is_retry = data.get("is_retry", False)
            code_lines = data.get("code_lines", 0)

            if is_retry:
                status_bar.update_status(status_message=f"Step {step_num}: Executing retry {attempt}/{max_attempts} ({code_lines} lines)")
            else:
                status_bar.update_status(status_message=f"Step {step_num}: Executing code ({code_lines} lines)")

        elif event_type == "step_error":
            step_num = event.step_number or 0
            error_type = data.get("error_type", "Error")
            attempt = data.get("attempt", 1)
            max_attempts = data.get("max_attempts", 10)
            will_retry = data.get("will_retry", False)

            if will_retry:
                next_attempt = data.get("next_attempt", attempt + 1)
                status_bar.update_status(
                    status_message=f"Step {step_num}: {error_type} on attempt {attempt}/{max_attempts}, retrying..."
                )
                log.write(Text(f"  Step {step_num} attempt {attempt} failed: {error_type}, will retry", style="yellow"))
            else:
                status_bar.update_status(status_message=f"Step {step_num}: {error_type} - max retries exceeded")
                log.write(Text(f"  Step {step_num} failed after {max_attempts} attempts: {error_type}", style="red"))

        # Synthesis events
        elif event_type == "synthesizing":
            status_bar.update_status(status_message="Generating insights...")

        elif event_type == "answer_ready":
            status_bar.update_status(status_message="Answer ready")

        elif event_type == "suggestions_ready":
            status_bar.update_status(status_message="Preparing suggestions...")

        elif event_type == "facts_extracted":
            facts = data.get("facts", [])
            count = len(facts)
            if count > 0:
                # Get fact names/keys
                fact_names = []
                for f in facts:
                    if isinstance(f, dict):
                        fact_names.append(f.get("name", f.get("key", str(f)[:20])))
                    elif hasattr(f, "name"):
                        fact_names.append(f.name)
                    else:
                        fact_names.append(str(f)[:20])
                facts_str = ", ".join(fact_names[:5])  # Limit to 5
                if count > 5:
                    facts_str += f", ... (+{count - 5} more)"
                display_msg = f"Extracted {count} facts: {facts_str}"
                logger.debug(f"on_session_event: {display_msg}")
                log.write(Text(f"  {display_msg}", style="dim green"))

        elif event_type == "correction_saved":
            correction = data.get("correction", "")[:60]
            learning_id = data.get("learning_id", "")
            status_bar.update_status(status_message="Correction saved")
            log.write(Text(f"  Saved correction: {correction}", style="dim cyan"))
            logger.debug(f"Correction saved as learning {learning_id}")

        # Proof/verification events (from /prove command)
        elif event_type == "extracting_claims":
            msg = data.get("message", "Extracting claims...")
            status_bar.update_status(status_message=msg)
            log.write(Text(f"  {msg}", style="cyan"))

        elif event_type == "verifying_claim":
            claim = data.get("claim", "")[:60]
            total = data.get("total", 1)
            step = data.get("step_number", 1)
            status_bar.update_status(status_message=f"Verifying claim {step}/{total}...")
            log.write(Text(f"  Verifying: {claim}...", style="dim"))

        elif event_type == "proof_complete":
            verified = data.get("verified_count", 0)
            total = data.get("claims_count", 0)
            confidence = data.get("confidence", 0.0)
            status_bar.update_status(status_message=f"Proof complete: {verified}/{total} verified")
            log.write(Text(f"  Proof complete: {verified}/{total} claims verified (confidence: {confidence:.0%})", style="bold"))

        elif event_type == "verification_error":
            error = data.get("error", "Unknown error")
            log.write(Text(f"  Verification error: {error[:80]}", style="dim red"))

        elif event_type == "complete":
            status_bar.update_status(status_message=None, phase=Phase.IDLE)


class ConstatREPLApp(App):
    """Textual-based REPL application with persistent status bar."""

    CSS = """
    Screen {
        layout: grid;
        grid-size: 1;
        grid-rows: 1fr auto auto;
    }

    #content-area {
        height: 100%;
        width: 100%;
    }

    #output-container {
        width: 2fr;
        height: 100%;
        border: solid $primary-darken-2;
        border-title-align: left;
        border-title-color: $text;
        border-title-background: $primary-darken-2;
        border-subtitle-align: right;
        border-subtitle-color: $text-muted;
        margin: 0 1 0 0;
    }

    #output-log {
        height: 1fr;
    }

    #side-panel {
        width: 1fr;
        background: $surface;
        display: none;
        border: solid $primary-darken-2;
        border-title-align: left;
        border-title-color: $text;
        border-title-background: $primary-darken-2;
        border-subtitle-align: right;
        border-subtitle-color: $text-muted;
    }

    #side-panel.visible {
        display: block;
    }

    #side-panel.executing {
        border: solid cyan;
    }

    #proof-tree-panel {
        height: 1fr;
        padding: 1;
    }

    /* Animated step indicator */
    .step-active {
        text-style: bold;
    }

    .step-pulse {
        text-style: bold italic;
    }

    #input-rule {
        height: 1;
        color: $success;
    }

    #input-container {
        height: 1;
    }

    #input-prompt {
        width: 2;
        color: $text;
    }

    #user-input {
        width: 1fr;
        border: none;
        height: 1;
    }

    #status-bar {
        height: 2;
        dock: bottom;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "interrupt", "Interrupt", show=False),
        Binding("escape", "interrupt", "Interrupt", show=False),
        Binding("ctrl+d", "quit", "Quit", show=False),
        Binding("ctrl+left", "shrink_panel", "Shrink panel", show=False),
        Binding("ctrl+right", "expand_panel", "Expand panel", show=False),
        Binding("ctrl+shift+o", "copy_output", "Copy output", show=False),
        Binding("ctrl+shift+p", "copy_panel", "Copy panel", show=False),
    ]

    # Panel width ratios (output_log : side_panel)
    PANEL_RATIOS = [(4, 1), (3, 1), (2, 1), (1, 1), (1, 2), (1, 3)]
    DEFAULT_RATIO_INDEX = 2  # 2:1 ratio

    def __init__(
        self,
        config: Config,
        verbose: bool = False,
        user_id: str = "default",
        initial_problem: Optional[str] = None,
        auto_resume: bool = False,
        debug: bool = False,
        session: Optional[Session] = None,
    ):
        super().__init__()
        self.config = config
        self.verbose = verbose
        self.user_id = user_id
        self.initial_problem = initial_problem
        self.auto_resume = auto_resume
        self.debug_mode = debug

        # Session state - use pre-created session if provided
        self.session_config = SessionConfig(verbose=verbose)
        self.session: Optional[Session] = session
        self.fact_store = FactStore(user_id=user_id)
        self.learning_store = LearningStore(user_id=user_id)

        # Input state
        self.suggestions: list[str] = []
        self.last_problem = ""

        # Spinner animation
        self._spinner_running = False
        self._spinner_task: Optional[asyncio.Task] = None

        # Clarification handling (thread synchronization)
        self._clarification_event = threading.Event()
        self._clarification_request: Optional[ClarificationRequest] = None
        self._clarification_response: Optional[ClarificationResponse] = None
        self._awaiting_clarification = False
        self._clarification_answers: dict[str, str] = {}
        self._current_question_idx = 0

        # Plan approval handling (thread synchronization)
        self._approval_event = threading.Event()
        self._approval_request: Optional[PlanApprovalRequest] = None
        self._approval_response: Optional[PlanApprovalResponse] = None
        self._awaiting_approval = False

        # App state
        self._app_running = True
        self._pending_result = None  # Result from worker
        self._is_solving = False  # True when a solve is in progress
        self._queued_input: list[str] = []  # Input queued while solving

        # Panel sizing state
        self._panel_ratio_index = self.DEFAULT_RATIO_INDEX

        # Plan step tracking for follow-ups
        self._plan_steps: list[dict] = []
        self._completed_plan_steps: list[dict] = []

    def compose(self) -> ComposeResult:
        """Compose the UI layout."""
        with Horizontal(id="content-area"):
            with Vertical(id="output-container"):
                yield OutputLog(id="output-log", highlight=True, markup=True, wrap=True)
            with SidePanel(id="side-panel"):
                yield ProofTreePanel(id="proof-tree-panel")
        yield Static("─" * 80, id="input-rule")
        with Horizontal(id="input-container"):
            yield Static("> ", id="input-prompt")
            yield ConstatInput(placeholder="Ask a question or type /help", id="user-input")
        yield StatusBar(id="status-bar")

    async def on_mount(self) -> None:
        """Initialize after mounting."""
        # Focus the input
        self.query_one("#user-input", Input).focus()

        # Set border titles with copy hint (shortcuts: Ctrl+Shift+O for output, Ctrl+Shift+P for panel)
        output_container = self.query_one("#output-container", Vertical)
        output_container.border_title = "Output"
        output_container.border_subtitle = "📋 ^⇧O"

        side_panel = self.query_one("#side-panel", SidePanel)
        side_panel.border_title = "Panel"
        side_panel.border_subtitle = "📋 ^⇧P"

        # Create session
        await self._create_session()

        # Initialize settings display
        self._update_settings_display()

        # Show welcome banner
        await self._show_banner()

        # Handle initial problem if provided
        if self.initial_problem:
            await self._solve(self.initial_problem)

    async def _create_session(self) -> None:
        """Verify session is ready and register feedback handler."""
        log = self.query_one("#output-log", OutputLog)

        if self.session:
            # Register feedback handler to show progress in UI
            self._feedback_handler = TextualFeedbackHandler(self)
            self.session.on_event(self._feedback_handler.handle_event)

            # Register clarification callback
            self.session.set_clarification_callback(self._handle_clarification_sync)

            # Register approval callback
            self.session.set_approval_callback(self._handle_approval_sync)

            log.write(Text("Session ready.", style="dim green"))
        else:
            log.write(Text("Error: No session provided.", style="red"))

    def _handle_clarification_sync(self, request: ClarificationRequest) -> ClarificationResponse:
        """
        Handle clarification request from session (called from worker thread).

        This blocks the worker thread until the user provides answers via the UI.
        """
        logger.debug(f"Clarification callback triggered with {len(request.questions)} questions")

        # Store request and signal main thread
        self._clarification_request = request
        self._clarification_answers = {}
        self._current_question_idx = 0
        self._awaiting_clarification = True
        self._clarification_event.clear()

        # post_message is thread-safe, no call_from_thread needed
        self.post_message(ShowClarificationUI())

        # Wait for user to provide answers (blocks worker thread)
        self._clarification_event.wait()

        # Return the response
        self._awaiting_clarification = False
        return self._clarification_response or ClarificationResponse(answers={}, skip=True)

    def _show_clarification_ui(self) -> None:
        """Show clarification UI and set focus."""
        logger.debug("_show_clarification_ui called")

        # Stop spinner
        self._spinner_running = False
        if self._spinner_task:
            self._spinner_task.cancel()
            self._spinner_task = None

        # Show the questions (this updates status bar)
        self._show_clarification_questions()

        # Focus input immediately
        input_widget = self.query_one("#user-input", Input)
        logger.debug(f"Focusing input, disabled={input_widget.disabled}")
        self.set_focus(input_widget)

        logger.debug("_show_clarification_ui complete")

    def _show_clarification_questions(self) -> None:
        """Show clarification questions in the UI (called on main thread)."""
        if not self._clarification_request:
            return

        log = self.query_one("#output-log", OutputLog)
        input_widget = self.query_one("#user-input", Input)
        status_bar = self.query_one("#status-bar", StatusBar)

        log.write("")
        log.write(Text("Clarification needed. Answer below ↓", style="bold yellow"))
        log.write(Text("  (Enter number, type answer, or press Enter to use default [1])", style="dim"))

        questions = self._clarification_request.questions
        for i, q in enumerate(questions):
            log.write(Text(f"  Q{i+1}: {q.text}", style="cyan"))
            if q.suggestions:
                for j, s in enumerate(q.suggestions, 1):
                    # Mark first suggestion as default
                    if j == 1:
                        log.write(Text(f"      {j}. {s} [default]", style="green"))
                    else:
                        log.write(Text(f"      {j}. {s}", style="dim"))

        # Show first question prompt with default hint
        if questions:
            first_q = questions[0]
            default_hint = ""
            if first_q.suggestions:
                default_hint = f" [Enter={first_q.suggestions[0][:20]}...]"
            input_widget.placeholder = f"Q1 (1-{len(first_q.suggestions)} or type){default_hint}"
            input_widget.value = ""  # Clear any existing value
            input_widget.disabled = False  # Ensure input is enabled
            status_bar.update_status(status_message=f"Clarification Q1/{len(questions)}")

    def _handle_approval_sync(self, request: PlanApprovalRequest) -> PlanApprovalResponse:
        """
        Handle plan approval request from session (called from worker thread).

        This blocks the worker thread until the user approves/rejects the plan.
        """
        self._approval_request = request
        self._awaiting_approval = True
        self._approval_event.clear()

        logger.debug("Approval callback triggered, posting ShowApprovalUI message")

        # post_message is thread-safe, no call_from_thread needed
        self.post_message(ShowApprovalUI())

        logger.debug("Message posted, now waiting on approval_event")

        # Wait for user to approve/reject (blocks worker thread)
        self._approval_event.wait()

        # Return the response
        self._awaiting_approval = False
        return self._approval_response or PlanApprovalResponse.reject(reason="Cancelled")

    def on_show_approval_ui(self, message: ShowApprovalUI) -> None:
        """Handle ShowApprovalUI message - runs on main thread."""
        logger.debug("on_show_approval_ui message handler called")
        self._show_approval_ui()

    def on_show_clarification_ui(self, message: ShowClarificationUI) -> None:
        """Handle ShowClarificationUI message - runs on main thread."""
        logger.debug("on_show_clarification_ui message handler called")
        self._show_clarification_ui()

    def on_session_event(self, message: SessionEvent) -> None:
        """Handle SessionEvent message - runs on main thread."""
        event = message.event
        event_type = event.event_type
        # Log error details for debugging
        if event_type in ("step_error", "step_failed") and event.data:
            error = event.data.get("error", "")
            error_type = event.data.get("error_type", "")
            logger.debug(f"on_session_event: {event_type} - {error_type}: {error[:500] if error else 'no error message'}")
        else:
            logger.debug(f"on_session_event: {event_type}")
        if hasattr(self, '_feedback_handler'):
            self._feedback_handler._handle_event_on_main(message.event)

    async def on_solve_complete(self, message: SolveComplete) -> None:
        """Handle SolveComplete message - runs on main thread."""
        logger.debug("on_solve_complete message handler called")
        result = message.result
        log = self.query_one("#output-log", OutputLog)
        status_bar = self.query_one("#status-bar", StatusBar)
        side_panel = self.query_one("#side-panel", SidePanel)
        panel_content = self.query_one("#proof-tree-panel", SidePanelContent)

        # Extract and store code blocks from results for /code command
        results = result.get("results", [])
        if results and self.session and self.session.datastore:
            code_blocks = []
            for i, r in enumerate(results):
                if hasattr(r, "code") and r.code:
                    code_blocks.append({
                        "step": i + 1,
                        "code": r.code,
                        "success": getattr(r, "success", True),
                    })
            if code_blocks:
                self.session.datastore.set_session_meta("code_blocks", code_blocks)
                logger.debug(f"on_solve_complete: stored {len(code_blocks)} code blocks")

        # Build artifacts list from result
        artifacts = []
        logger.debug(f"on_solve_complete: result keys = {list(result.keys())}")

        # Get table file paths from registry for file:// URIs
        table_file_paths = {}
        try:
            from constat.storage.registry import ConstatRegistry
            from pathlib import Path
            registry = ConstatRegistry()
            registry_tables = registry.list_tables(user_id=self.user_id, session_id=self.session.session_id)
            registry.close()
            for t in registry_tables:
                file_path = Path(t.file_path)
                if file_path.exists():
                    table_file_paths[t.name] = file_path.resolve().as_uri()
        except Exception as e:
            logger.debug(f"on_solve_complete: failed to get table file paths: {e}")

        # Add tables as artifacts (datastore returns them ordered by step_number - creation order)
        datastore_tables = result.get("datastore_tables", [])
        logger.debug(f"on_solve_complete: datastore_tables = {datastore_tables}")
        for table in datastore_tables:
            if isinstance(table, str):
                artifacts.append({
                    "type": "table",
                    "name": table,
                    "row_count": None,
                    "step_number": 0,
                    "created_at": "",  # Unknown for string tables
                    "command": f"/show {table}",
                    "file_uri": table_file_paths.get(table, ""),
                })
            elif isinstance(table, dict) and "name" in table:
                # Dict with name key (from datastore)
                table_name = table["name"]
                row_count = table.get("row_count")
                step_number = table.get("step_number", 0)
                created_at = table.get("created_at", "")
                artifacts.append({
                    "type": "table",
                    "name": table_name,
                    "row_count": row_count,
                    "step_number": step_number,
                    "created_at": created_at,
                    "command": f"/show {table_name}",
                    "file_uri": table_file_paths.get(table_name, ""),
                })
            elif hasattr(table, "name"):
                artifacts.append({
                    "type": "table",
                    "name": table.name,
                    "row_count": getattr(table, "row_count", None),
                    "step_number": getattr(table, "step_number", 0),
                    "created_at": getattr(table, "created_at", ""),
                    "command": f"/show {table.name}",
                    "file_uri": table_file_paths.get(table.name, ""),
                })

        # Add any visualizations/outputs (created after tables, so use high step number)
        import datetime
        max_step = max((a.get("step_number", 0) for a in artifacts), default=0)
        if result.get("visualizations"):
            for i, viz in enumerate(result.get("visualizations", [])):
                artifacts.append({
                    "type": "chart",
                    "name": viz.get("name", "Chart"),
                    "description": viz.get("description", ""),
                    "step_number": max_step + 1 + i,
                    "created_at": datetime.datetime.now().isoformat(),
                    "command": "",
                })

        # Add pending outputs (charts, md files, etc. saved during execution)
        pending = get_pending_outputs()
        logger.debug(f"on_solve_complete: pending outputs = {pending}")
        max_step = max((a.get("step_number", 0) for a in artifacts), default=0)
        for i, output in enumerate(pending):
            file_uri = output.get("file_uri", "")
            description = output.get("description", "")
            file_type = output.get("type", "file")

            # Determine artifact type from file extension or type field
            if file_type == "chart" or file_uri.endswith(('.html', '.png', '.svg')):
                artifact_type = "chart"
            elif file_uri.endswith('.md'):
                artifact_type = "file"
            else:
                artifact_type = "file"

            # Get just the filename for display
            from pathlib import Path
            filename = Path(file_uri).name if file_uri else description

            artifacts.append({
                "type": artifact_type,
                "name": filename,
                "description": description,
                "step_number": max_step + 1 + i,
                "created_at": datetime.datetime.now().isoformat(),
                "command": f"open {file_uri}" if file_uri else "",
                "file_uri": file_uri,  # Include URI for clickable links
            })

        # Add DFD artifact if it exists (created during plan approval)
        if self.session and self.session.session_id:
            try:
                from pathlib import Path
                import tempfile
                artifacts_dir = Path(tempfile.gettempdir()) / "constat_artifacts"
                dfd_path = artifacts_dir / f"{self.session.session_id}_data_flow.txt"
                if dfd_path.exists():
                    dfd_uri = dfd_path.resolve().as_uri()
                    artifacts.insert(0, {
                        "type": "file",
                        "name": "Data Flow",
                        "description": "Data flow diagram",
                        "step_number": -1,  # Show first
                        "created_at": "",
                        "command": f"open {dfd_uri}",
                        "file_uri": dfd_uri,
                    })
                    logger.debug(f"on_solve_complete: added DFD artifact {dfd_path}")
            except Exception as e:
                logger.debug(f"on_solve_complete: failed to add DFD artifact: {e}")

        # Sort artifacts by step_number and created_at to ensure creation order (older first, newer at bottom)
        artifacts.sort(key=lambda a: (a.get("step_number", 0), a.get("created_at", "")))

        # Show artifacts in side panel (or hide if none)
        logger.debug(f"on_solve_complete: total artifacts = {len(artifacts)}")
        if artifacts:
            logger.debug(f"on_solve_complete: showing {len(artifacts)} artifacts in side panel")
            for a in artifacts[:3]:  # Log first 3
                logger.debug(f"  artifact: {a}")
            panel_content.show_artifacts(artifacts)
            side_panel.add_class("visible")
        else:
            logger.debug("on_solve_complete: no artifacts, hiding side panel")
            side_panel.remove_class("visible")

        log.write(Rule("[bold blue]VERA[/bold blue]", align="left"))

        # Display result based on response type
        logger.debug(f"on_solve_complete: meta_response={result.get('meta_response')}, "
                     f"has_output={bool(result.get('output'))}, "
                     f"has_final_answer={bool(result.get('final_answer'))}, "
                     f"raw={self.session_config.show_raw_output}")
        if result.get("error"):
            log.write(Text(f"Error: {result['error']}", style="red"))
        elif result.get("meta_response"):
            output = result.get("output", "")
            if output:
                log.write(Markdown(output))
            self.suggestions = result.get("suggestions", [])
        else:
            # For normal results, respect show_raw_output and enable_insights settings:
            # - raw output = result["output"] (combined step stdout - verbose)
            # - final answer = result["final_answer"] (synthesized summary or same as raw)
            raw_output = result.get("output", "")
            final_answer = result.get("final_answer", "")
            is_synthesized = final_answer and final_answer != raw_output

            if self.session_config.show_raw_output:
                # raw=on: Show verbose step output
                if raw_output:
                    log.write(Markdown(raw_output))
                # Also show synthesis if different and insights enabled
                if is_synthesized and self.session_config.enable_insights:
                    log.write("")
                    log.write(Text("Summary:", style="bold cyan"))
                    log.write(Markdown(final_answer))
            else:
                # raw=off: Prefer synthesized summary over verbose step output
                if is_synthesized:
                    # Show synthesis (different from raw)
                    log.write(Markdown(final_answer))
                elif final_answer:
                    # final_answer exists but equals raw_output - show it
                    log.write(Markdown(final_answer))
                elif raw_output:
                    # No synthesis available but we have output - show it
                    # (raw=off preference can't be honored when no synthesis exists)
                    log.write(Markdown(raw_output))
                else:
                    log.write(Text("No output returned.", style="dim"))

            self.suggestions = result.get("suggestions", [])

        # Show suggestions if any
        if self.suggestions:
            log.write("")
            log.write(Text("You might also ask:", style="dim"))
            for i, s in enumerate(self.suggestions, 1):
                log.write(Text.assemble(
                    (f"  {i}. ", "dim"),
                    (s, "cyan"),
                ))

        # Save completed steps for follow-up plans
        if result.get("success") and self._plan_steps:
            # Mark all current steps as completed and save them
            completed = []
            for s in self._plan_steps:
                if not s.get("completed"):  # Only save new steps, not already-completed ones
                    completed.append({
                        "number": s.get("number"),
                        "goal": s.get("goal"),
                        "completed": True,
                    })
            if completed:
                self._completed_plan_steps.extend(completed)
                logger.debug(f"on_solve_complete: saved {len(completed)} completed steps, total={len(self._completed_plan_steps)}")

        # Stop spinner/timer and reset status to Ready
        await self._stop_spinner()
        status_bar.stop_timer()
        logger.debug("on_solve_complete: resetting status bar to IDLE")
        status_bar.update_status(status_message=None, phase=Phase.IDLE)
        status_bar.refresh()  # Force refresh to ensure update is visible

        # Mark solving complete and process any queued input
        self._is_solving = False
        if self._queued_input:
            next_input = self._queued_input.pop(0)
            remaining = len(self._queued_input)
            log.write(Text(f"\nProcessing queued input: {next_input}", style="cyan"))
            if remaining > 0:
                log.write(Text(f"  ({remaining} more queued)", style="dim"))
            await self._solve(next_input)

    def _show_approval_ui(self) -> None:
        """Show approval UI and set focus."""
        logger.debug(">>> _show_approval_ui ENTERED <<<")

        # Stop spinner
        self._spinner_running = False
        if self._spinner_task:
            self._spinner_task.cancel()
            self._spinner_task = None

        # Show the approval prompt (this updates status bar)
        self._show_approval_prompt()

        # Focus input immediately
        input_widget = self.query_one("#user-input", Input)
        logger.debug(f"Focusing input, disabled={input_widget.disabled}")
        self.set_focus(input_widget)

        logger.debug("_show_approval_ui complete")

    def _show_approval_prompt(self) -> None:
        """Show plan approval prompt in the UI (called on main thread).

        Shows step summary in main panel and DFD in side panel.
        """
        if not self._approval_request:
            return

        log = self.query_one("#output-log", OutputLog)
        input_widget = self.query_one("#user-input", Input)
        status_bar = self.query_one("#status-bar", StatusBar)

        request = self._approval_request

        # Show plan summary in main panel
        log.write("")
        log.write(Text("Plan ready. Approve? ↓", style="bold yellow"))

        # Show premises and inferences
        if request.steps:
            premises = [s for s in request.steps if s.get("type") == "premise"]
            inferences = [s for s in request.steps if s.get("type") == "inference"]

            if premises:
                log.write(Text(f"  Premises ({len(premises)}):", style="cyan"))
                for s in premises:
                    fact_id = s.get("fact_id", "")
                    goal = s.get("goal", "")
                    # Extract just the variable name
                    if "=" in goal:
                        var_name = goal.split("=", 1)[0].strip()
                    else:
                        var_name = goal[:50]
                    log.write(Text(f"    {fact_id}: {var_name}", style="dim"))

            if inferences:
                log.write(Text(f"  Inferences ({len(inferences)}):", style="green"))
                for s in inferences:
                    fact_id = s.get("fact_id", "")
                    goal = s.get("goal", "")
                    # Extract just the variable name
                    if "=" in goal:
                        var_name = goal.split("=", 1)[0].strip()
                    else:
                        var_name = goal[:50]
                    log.write(Text(f"    {fact_id}: {var_name}", style="dim"))

            # Show DFD in side panel
            self._show_dfd_in_side_panel(request.steps)

        log.write("")

        # Configure input for approval
        input_widget.placeholder = "[yes/Enter] Approve  [no] Reject  [or provide feedback]"
        input_widget.value = ""
        input_widget.disabled = False  # Ensure input is enabled

        # Stop timer after planning completes (will restart when execution begins)
        status_bar.stop_timer()
        status_bar.update_status(status_message=None, phase=Phase.AWAITING_APPROVAL)

    def _show_data_flow_dag(self, log: OutputLog, steps: list[dict]) -> None:
        """Display an ASCII data flow DAG."""
        try:
            import networkx as nx
            from constat.visualization.box_dag import render_dag
        except ImportError:
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

        log.write("")
        log.write(Text("  DATA FLOW:", style="bold yellow"))

        try:
            diagram = render_dag(G, style='rounded')
            for line in diagram.split('\n'):
                if line.strip():
                    log.write(Text(f"      {line}", style="dim"))
        except Exception:
            log.write(Text("      (diagram rendering failed)", style="dim"))

    def _show_dfd_in_side_panel(self, steps: list[dict]) -> None:
        """Show DFD diagram in the side panel. Always shows side panel."""
        side_panel = self.query_one("#side-panel", SidePanel)
        panel_content = self.query_one("#proof-tree-panel", SidePanelContent)

        # Always make side panel visible
        side_panel.add_class("visible")

        try:
            import networkx as nx
            from constat.visualization.box_dag import render_dag
        except ImportError:
            # Fallback: show simple step list if networkx not available
            self._show_steps_fallback(steps, panel_content)
            return

        # Calculate available panel width based on terminal width and ratio
        try:
            app_width = self.size.width
            output_ratio, side_ratio = self.PANEL_RATIOS[self._panel_ratio_index]
            total_ratio = output_ratio + side_ratio
            # Side panel width minus border/padding (approx 4 chars)
            panel_width = max(20, (app_width * side_ratio // total_ratio) - 4)
        except Exception:
            panel_width = 40  # Fallback width

        # Scale node name length based on panel width
        # Allow for ~3 nodes side by side with spacing
        max_name_len = max(6, min(12, panel_width // 4))

        # Build mapping from fact_id (P1, I1) to English variable name
        fact_to_name: dict[str, str] = {}
        for s in steps:
            fact_id = s.get("fact_id", "")
            goal = s.get("goal", "")
            # Extract English name from BEFORE the first '='
            # Premise format: "employees = ? (description) [source: xxx]"
            # Inference format: "remaining_days = P1 - P2 -- explanation"
            if "=" in goal:
                english_name = goal.split("=", 1)[0].strip()
                # Truncate to fit panel width
                if len(english_name) > max_name_len:
                    english_name = english_name[:max_name_len]
                fact_to_name[fact_id] = english_name
            elif fact_id:
                fact_to_name[fact_id] = fact_id  # Fallback to fact_id

        # Build NetworkX graph from steps using English names
        G = nx.DiGraph()

        for s in steps:
            step_type = s.get("type")
            fact_id = s.get("fact_id", "")
            goal = s.get("goal", "")
            node_name = fact_to_name.get(fact_id, fact_id)

            if step_type == "premise":
                G.add_node(node_name)
            elif step_type == "inference":
                G.add_node(node_name)
                # Extract dependencies from the operation
                inf_match = re.match(r'^(\w+)\s*=\s*(.+)', goal)
                if inf_match:
                    operation = inf_match.group(2)
                else:
                    operation = goal
                deps = re.findall(r'[PI]\d+', operation)
                for dep in deps:
                    dep_name = fact_to_name.get(dep, dep)
                    if G.has_node(dep_name):
                        G.add_edge(dep_name, node_name)

        if G.number_of_nodes() == 0:
            # Fallback: show simple step list if no graph nodes
            self._show_steps_fallback(steps, panel_content)
            return

        # Find terminal inference and add conclusion
        inference_names = [fact_to_name.get(f, f) for f in fact_to_name if f.startswith('I')]
        if inference_names:
            terminal = None
            for inf_name in inference_names:
                successors = list(G.successors(inf_name))
                if not any(s in inference_names for s in successors):
                    terminal = inf_name
                    break
            if terminal is None:
                terminal = inference_names[-1]
            G.add_node("conclusion")
            G.add_edge(terminal, "conclusion")

        try:
            diagram = render_dag(G, style='rounded', max_width=panel_width)
            dag_lines = [line for line in diagram.split('\n') if line.strip()]
            panel_content.show_plan(dag_lines)
            logger.debug(f"_show_dfd_in_side_panel: dag_lines={len(dag_lines)}, panel_width={panel_width}")

            # Save DFD as artifact file (before execution)
            if self.session and self.session.session_id:
                try:
                    from pathlib import Path
                    import tempfile
                    artifacts_dir = Path(tempfile.gettempdir()) / "constat_artifacts"
                    artifacts_dir.mkdir(exist_ok=True)
                    dfd_path = artifacts_dir / f"{self.session.session_id}_data_flow.txt"
                    dfd_path.write_text(diagram)
                    logger.debug(f"_show_dfd_in_side_panel: saved DFD artifact to {dfd_path}")
                except Exception as e:
                    logger.debug(f"_show_dfd_in_side_panel: failed to save DFD artifact: {e}")
        except Exception as e:
            logger.debug(f"_show_dfd_in_side_panel render failed: {e}")
            # Fallback: show simple step list
            self._show_steps_fallback(steps, panel_content)

    def _show_steps_fallback(self, steps: list[dict], panel_content: SidePanelContent) -> None:
        """Show simple step list when DFD can't be rendered."""
        lines = []
        for s in steps:
            fact_id = s.get("fact_id", "")
            goal = s.get("goal", "")
            step_type = s.get("type", "")
            prefix = "P" if step_type == "premise" else "I" if step_type == "inference" else "→"
            lines.append(f"{prefix} {fact_id}: {goal}")  # Full text, wraps at panel width
        # Set lines directly for plan display (no box)
        panel_content._dag_lines = lines
        panel_content._mode = panel_content.MODE_PLAN  # Use plan rendering (no box)
        panel_content._update_display()

    def _focus_input(self) -> None:
        """Focus the input widget."""
        input_widget = self.query_one("#user-input", Input)
        logger.debug(f"_focus_input called, current focus={self.focused}, input.disabled={input_widget.disabled}")
        self.set_focus(input_widget)
        logger.debug(f"_focus_input complete, new focus={self.focused}")

    async def _show_banner(self) -> None:
        """Show welcome banner."""
        log = self.query_one("#output-log", OutputLog)

        reliable_adj, honest_adj = get_vera_adjectives()

        log.write("")
        log.write(Text.assemble(
            "Hi, I'm ",
            ("Vera", "bold"),
            f", your {reliable_adj} and {honest_adj} data analyst.",
        ))
        log.write(Text(
            "I make every effort to tell the truth and fully explain my reasoning.",
            style="dim",
        ))
        log.write("")
        log.write(Text.assemble(
            ("Powered by ", "dim"),
            ("Constat", "bold blue"),
            (" (Latin: \"it is established\") — Multi-Step AI Reasoning Agent", "dim"),
        ))
        log.write(Text(
            "Type /help for commands, or ask a question. | Tab completes commands | Ctrl+C interrupts",
            style="dim",
        ))

        # Show starter suggestions
        if not self.initial_problem:
            log.write("")
            log.write(Text("Try asking:", style="dim"))
            starter_suggestions = [
                "What data is available?",
                "What can you help me with?",
                "How do you reason about problems?",
                "What makes you different, Vera?",
            ]
            for i, s in enumerate(starter_suggestions, 1):
                log.write(Text.assemble(
                    (f"  {i}. ", "dim"),
                    (s, "cyan"),
                ))
            self.suggestions = starter_suggestions
            log.write("")

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission."""
        user_input = event.value.strip()

        # Clear input and add to history
        input_widget = self.query_one("#user-input", ConstatInput)
        input_widget.value = ""

        log = self.query_one("#output-log", OutputLog)
        status_bar = self.query_one("#status-bar", StatusBar)

        logger.debug(f"Input received: '{user_input}', awaiting_approval={self._awaiting_approval}, awaiting_clarification={self._awaiting_clarification}")

        # Handle approval input (don't add to history - these are just y/n answers)
        if self._awaiting_approval and self._approval_request:
            logger.debug("Routing to approval handler")
            await self._handle_approval_answer(user_input)
            return

        # Handle clarification input (don't add to history)
        if self._awaiting_clarification and self._clarification_request:
            logger.debug("Routing to clarification handler")
            await self._handle_clarification_answer(user_input)
            return

        if not user_input:
            return

        # Add to command history (for regular commands and queries)
        input_widget.add_to_history(user_input)

        # Echo user input with rule
        log.write(Rule("[bold green]YOU[/bold green]", align="right"))
        log.write(Text(f"> {user_input}"))
        log.write("")

        # Handle input
        lower_input = user_input.lower()

        # Check for suggestion shortcuts
        if self.suggestions and lower_input.isdigit():
            idx = int(lower_input) - 1
            if 0 <= idx < len(self.suggestions):
                user_input = self.suggestions[idx]
            else:
                log.write(Text(f"No suggestion #{lower_input}", style="yellow"))
                return
        elif self.suggestions and lower_input in ("ok", "yes", "sure", "y"):
            user_input = self.suggestions[0]

        # Handle commands
        if user_input.startswith("/"):
            await self._handle_command(user_input)
        else:
            # Queue input if already solving - wait for current solve to complete
            if self._is_solving:
                self._queued_input.append(user_input)
                log.write(Text(f"  (queued - will process after current solve completes)", style="dim cyan"))
                # Don't cancel - let the current solve finish naturally
                status_bar = self.query_one("#status-bar", StatusBar)
                status_bar.update_status(status_message=f"Solving... ({len(self._queued_input)} queued)")
            else:
                await self._solve(user_input)

    async def _handle_clarification_answer(self, answer: str) -> None:
        """Handle a clarification answer from the user."""
        if not self._clarification_request:
            return

        log = self.query_one("#output-log", OutputLog)
        input_widget = self.query_one("#user-input", Input)
        status_bar = self.query_one("#status-bar", StatusBar)
        questions = self._clarification_request.questions

        current_q = questions[self._current_question_idx]
        logger.debug(f"[CLARIFICATION] Handler received: answer={answer!r}, q_idx={self._current_question_idx}, q_text={current_q.text!r}, suggestions={current_q.suggestions}")

        # Handle empty input - use default (first suggestion)
        if not answer and current_q.suggestions:
            answer = current_q.suggestions[0]
            logger.debug(f"[CLARIFICATION] Using default (empty input): {answer!r}")
            log.write(Text(f"  A{self._current_question_idx + 1}: {answer} (default)", style="dim green"))
        # Handle numbered suggestion selection
        elif answer.isdigit() and current_q.suggestions:
            idx = int(answer) - 1
            if 0 <= idx < len(current_q.suggestions):
                answer = current_q.suggestions[idx]
                logger.debug(f"[CLARIFICATION] Selected option {idx+1}: {answer!r}")
            else:
                logger.debug(f"[CLARIFICATION] Invalid index {idx+1}, keeping raw answer: {answer!r}")
            log.write(Text(f"  A{self._current_question_idx + 1}: {answer}", style="green"))
        elif answer:
            log.write(Text(f"  A{self._current_question_idx + 1}: {answer}", style="green"))

        # Store answer (use question text as key)
        if answer:
            self._clarification_answers[current_q.text] = answer
            logger.debug(f"[CLARIFICATION] Stored: {current_q.text!r} -> {answer!r}")

        # Move to next question
        self._current_question_idx += 1

        if self._current_question_idx < len(questions):
            # Show next question with default hint
            next_q = questions[self._current_question_idx]
            default_hint = ""
            if next_q.suggestions:
                default_hint = f" [Enter={next_q.suggestions[0][:15]}...]"
            input_widget.placeholder = f"Q{self._current_question_idx + 1} (1-{len(next_q.suggestions) if next_q.suggestions else 0} or type){default_hint}"
            status_bar.update_status(
                status_message=f"Clarification Q{self._current_question_idx + 1}/{len(questions)}"
            )
        else:
            # All questions answered - send response
            log.write(Text("Clarifications received, continuing...", style="dim"))
            input_widget.placeholder = "Ask a question or type /help"
            status_bar.update_status(status_message="Processing with clarifications...")

            logger.debug(f"[CLARIFICATION] Creating response with answers: {self._clarification_answers}")
            self._clarification_response = ClarificationResponse(
                answers=self._clarification_answers,
                skip=False
            )
            self._clarification_event.set()  # Unblock worker thread

    async def _handle_approval_answer(self, answer: str) -> None:
        """Handle a plan approval answer from the user."""
        log = self.query_one("#output-log", OutputLog)
        input_widget = self.query_one("#user-input", Input)
        status_bar = self.query_one("#status-bar", StatusBar)
        side_panel = self.query_one("#side-panel", SidePanel)

        lower = answer.lower() if answer else ""

        # Slash commands during approval - pass through to command handler
        if answer.startswith("/"):
            # Cancel approval state and handle command
            self._awaiting_approval = False
            self._approval_request = None
            status_bar.update_status(status_message=None, phase=Phase.IDLE)
            side_panel.remove_class("visible")
            input_widget.placeholder = "Ask a question or type /help"
            # Pass command through (signal to worker that we're abandoning this plan)
            self._approval_response = PlanApprovalResponse.pass_command(answer)
            self._approval_event.set()
            # Handle the command directly
            await self._handle_command(answer)
            return

        # Empty or 'y' or 'yes' = approve
        if not answer or lower in ('y', 'yes', 'ok', 'approve'):
            log.write(Text("Plan approved, executing...", style="green"))
            input_widget.placeholder = "Ask a question or type /help"
            # Restart timer for execution phase
            status_bar.start_timer()
            status_bar.update_status(status_message="Executing plan...", phase=Phase.EXECUTING)
            # Keep side panel visible - will show proof tree during execution

            self._approval_response = PlanApprovalResponse.approve()
            self._approval_event.set()

        # 'n' or 'no' = reject
        elif lower in ('n', 'no', 'reject', 'cancel'):
            log.write(Text("Plan rejected.", style="yellow"))
            input_widget.placeholder = "Ask a question or type /help"
            status_bar.update_status(status_message=None, phase=Phase.IDLE)
            # Hide side panel on rejection
            side_panel.remove_class("visible")

            self._approval_response = PlanApprovalResponse.reject(reason="User rejected")
            self._approval_event.set()

        # Any other input = treat as feedback/suggestion
        else:
            log.write(Text(f"Suggestion noted: {answer}", style="dim"))
            input_widget.placeholder = "Ask a question or type /help"
            # Restart timer for replanning phase
            status_bar.start_timer()
            status_bar.update_status(status_message="Replanning with feedback...", phase=Phase.PLANNING)
            # Hide side panel while replanning
            side_panel.remove_class("visible")

            self._approval_response = PlanApprovalResponse.suggest(suggestion=answer)
            self._approval_event.set()

    async def _handle_command(self, command: str) -> None:
        """Handle a slash command."""
        log = self.query_one("#output-log", OutputLog)

        cmd_parts = command.split(maxsplit=1)
        cmd = cmd_parts[0].lower()
        args = cmd_parts[1] if len(cmd_parts) > 1 else ""

        if cmd in ("/quit", "/q"):
            self.exit()
        elif cmd in ("/help", "/h"):
            await self._show_help()
        elif cmd == "/tables":
            await self._show_tables()
        elif cmd == "/show" and args:
            await self._show_table(args)
        elif cmd == "/query" and args:
            await self._run_query(args)
        elif cmd == "/facts":
            await self._show_facts()
        elif cmd == "/state":
            await self._show_state()
        elif cmd == "/reset":
            await self._reset_session()
        elif cmd == "/redo":
            await self._redo(args)
        elif cmd == "/artifacts":
            await self._show_artifacts()
        elif cmd == "/code":
            await self._show_code(args)
        elif cmd in ("/prove", "/audit"):
            await self._handle_prove()
        elif cmd == "/preferences":
            await self._show_preferences()
        elif cmd in ("/databases", "/database", "/db"):
            await self._show_databases()
        elif cmd in ("/files", "/file"):
            await self._show_files()
        elif cmd == "/context":
            await self._show_context()
        elif cmd in ("/history", "/sessions"):
            await self._show_history()
        elif cmd == "/verbose":
            await self._toggle_setting("verbose", args)
        elif cmd == "/raw":
            await self._toggle_setting("raw", args)
        elif cmd == "/insights":
            await self._toggle_setting("insights", args)
        elif cmd in ("/update", "/refresh"):
            await self._refresh_metadata()
        elif cmd == "/learnings":
            await self._show_learnings()
        elif cmd == "/consolidate":
            await self._consolidate_learnings()
        elif cmd == "/compact":
            await self._compact_context()
        elif cmd == "/remember" and args:
            await self._remember_fact(args)
        elif cmd == "/forget" and args:
            await self._forget_fact(args)
        elif cmd == "/correct" and args:
            await self._handle_correct(args)
        elif cmd == "/save" and args:
            await self._save_plan(args)
        elif cmd == "/share" and args:
            await self._save_plan(args, shared=True)
        elif cmd == "/plans":
            await self._list_plans()
        elif cmd == "/replay" and args:
            await self._replay_plan(args)
        elif cmd == "/resume" and args:
            await self._resume_session(args)
        elif cmd == "/export" and args:
            await self._export_table(args)
        elif cmd == "/summarize" and args:
            await self._handle_summarize(args)
        elif cmd == "/prove":
            await self._handle_audit()
        elif cmd == "/user":
            log.write(Text(f"Current user: {self.user_id}", style="dim"))
        elif cmd == "/discover":
            await self._discover(args)
        else:
            log.write(Text(f"Unknown command: {cmd}", style="yellow"))
            log.write(Text("Type /help for available commands.", style="dim"))

    async def _show_help(self) -> None:
        """Show help information."""
        log = self.query_one("#output-log", OutputLog)

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
            ("/update, /refresh", "Refresh metadata and rebuild cache"),
            ("/reset", "Clear session state and start fresh"),
            ("/redo [instruction]", "Retry last query (optionally with modifications)"),
            ("/user [name]", "Show or set current user"),
            ("/save <name>", "Save current plan for replay"),
            ("/share <name>", "Save plan as shared (all users)"),
            ("/plans", "List saved plans"),
            ("/replay <name>", "Replay a saved plan"),
            ("/history, /sessions", "List recent sessions"),
            ("/resume <id>", "Resume a previous session"),
            ("/context", "Show context size and token usage"),
            ("/compact", "Compact context to reduce token usage"),
            ("/facts", "Show cached facts from this session"),
            ("/remember <fact>", "Persist a session fact"),
            ("/forget <name>", "Forget a remembered fact"),
            ("/verbose [on|off]", "Toggle verbose mode"),
            ("/raw [on|off]", "Toggle raw output display"),
            ("/insights [on|off]", "Toggle insight synthesis"),
            ("/preferences", "Show current preferences"),
            ("/artifacts", "Show saved artifacts with file:// URIs"),
            ("/databases, /db", "List databases"),
            ("/files", "List all data files"),
            ("/correct <text>", "Record a correction for future reference"),
            ("/learnings", "Show learnings and rules"),
            ("/consolidate", "Promote similar learnings into rules"),
            ("/prove", "Verify conversation claims with auditable proof"),
            ("/discover [scope] <query>", "Search data sources (scope: database|api|document)"),
            ("/summarize <target>", "Summarize plan|session|facts|<table>"),
            ("/quit, /q", "Exit"),
        ]

        for cmd, desc in commands:
            table.add_row(cmd, desc)

        log.write(table)

        # Keyboard shortcuts
        log.write("")
        shortcuts_table = Table(title="Keyboard Shortcuts", show_header=True, box=None)
        shortcuts_table.add_column("Key", style="cyan")
        shortcuts_table.add_column("Action")

        shortcuts = [
            ("Ctrl+Left", "Shrink side panel"),
            ("Ctrl+Right", "Expand side panel"),
            ("Up/Down", "Navigate command history"),
            ("Ctrl+C / Esc", "Cancel current operation"),
            ("Ctrl+D", "Exit"),
        ]

        for key, action in shortcuts:
            shortcuts_table.add_row(key, action)

        log.write(shortcuts_table)

    async def _show_tables(self) -> None:
        """Show available tables with file:// URIs."""
        log = self.query_one("#output-log", OutputLog)

        if not self.session or not self.session.session_id:
            log.write(Text("No active session.", style="yellow"))
            return

        try:
            from constat.storage.registry import ConstatRegistry
            from pathlib import Path
            registry = ConstatRegistry()
            tables = registry.list_tables(user_id=self.user_id, session_id=self.session.session_id)
            registry.close()

            if not tables:
                log.write(Text("No tables yet.", style="dim"))
                return

            log.write(Text(f"Tables ({len(tables)})", style="bold"))
            for t in tables:
                log.write(Text.assemble(
                    ("  ", ""),
                    (t.name, "cyan"),
                    (f" ({t.row_count} rows)", "dim"),
                ))
                # Show clickable file link
                file_path = Path(t.file_path)
                if file_path.exists():
                    file_uri = file_path.resolve().as_uri()
                    link_markup = make_file_link_markup(file_uri, style="dim cyan underline", indent="    ")
                    log.write(link_markup)
        except Exception as e:
            log.write(Text(f"Error listing tables: {e}", style="red"))

    async def _show_table(self, table_name: str) -> None:
        """Show contents of a specific table."""
        log = self.query_one("#output-log", OutputLog)

        if not self.session or not self.session.datastore:
            log.write(Text("No active session.", style="yellow"))
            return

        try:
            df = self.session.datastore.query(f"SELECT * FROM {table_name} LIMIT 20")
            if df.empty:
                log.write(Text(f"Table '{table_name}' is empty.", style="dim"))
                return

            # Create a Rich table
            table = Table(title=f"{table_name} ({len(df)} rows shown)", show_header=True)
            for col in df.columns:
                table.add_column(str(col), style="cyan")

            for _, row in df.iterrows():
                table.add_row(*[str(v)[:50] for v in row.values])

            log.write(table)
        except Exception as e:
            log.write(Text(f"Error showing table: {e}", style="red"))

    async def _run_query(self, sql: str) -> None:
        """Run a SQL query on the datastore."""
        log = self.query_one("#output-log", OutputLog)

        if not self.session or not self.session.datastore:
            log.write(Text("No active session.", style="yellow"))
            return

        try:
            df = self.session.datastore.query(sql)
            if df.empty:
                log.write(Text("Query returned no results.", style="dim"))
                return

            table = Table(show_header=True)
            for col in df.columns:
                table.add_column(str(col), style="cyan")

            for _, row in df.head(20).iterrows():
                table.add_row(*[str(v)[:50] for v in row.values])

            log.write(table)
            if len(df) > 20:
                log.write(Text(f"... and {len(df) - 20} more rows", style="dim"))
        except Exception as e:
            log.write(Text(f"Query error: {e}", style="red"))

    async def _show_facts(self) -> None:
        """Show cached facts from this session."""
        log = self.query_one("#output-log", OutputLog)

        if not self.session or not hasattr(self.session, 'fact_resolver'):
            log.write(Text("No active session.", style="yellow"))
            return

        try:
            facts = self.session.fact_resolver.get_all_facts()
            if not facts:
                log.write(Text("No facts cached.", style="dim"))
                return

            log.write(Text(f"Cached Facts ({len(facts)})", style="bold"))
            for fact_id, fact in facts.items():
                # Fact is a dataclass, not a dict
                value = getattr(fact, 'value', None)
                confidence = getattr(fact, 'confidence', 1.0)
                source = getattr(fact, 'source', None)

                # Format value
                value_str = str(value)[:60] + "..." if len(str(value)) > 60 else str(value)

                # Show confidence as checkmark levels
                if confidence >= 0.9:
                    status = "✓"
                elif confidence >= 0.5:
                    status = "◐"
                else:
                    status = "○"

                # Source info
                source_str = f"[{source.value}]" if source and hasattr(source, 'value') else ""

                log.write(Text(f"  {status} {fact_id}: {value_str} {source_str}", style="dim"))
        except Exception as e:
            log.write(Text(f"Error showing facts: {e}", style="red"))
            logger.debug(f"_show_facts error: {e}", exc_info=True)

    async def _show_state(self) -> None:
        """Show session state."""
        log = self.query_one("#output-log", OutputLog)
        status_bar = self.query_one("#status-bar", StatusBar)

        log.write(Text("Session State", style="bold"))
        log.write(Text(f"  Phase: {status_bar.phase.value}", style="dim"))
        log.write(Text(f"  Verbose: {self.verbose}", style="dim"))
        log.write(Text(f"  User: {self.user_id}", style="dim"))
        if self.session:
            log.write(Text(f"  Session ID: {self.session.session_id}", style="dim"))
            log.write(Text(f"  Tables: {status_bar.tables_count}", style="dim"))
            log.write(Text(f"  Facts: {status_bar.facts_count}", style="dim"))

    async def _reset_session(self) -> None:
        """Clear session state."""
        log = self.query_one("#output-log", OutputLog)
        status_bar = self.query_one("#status-bar", StatusBar)
        side_panel = self.query_one("#side-panel", SidePanel)
        panel_content = self.query_one("#proof-tree-panel", ProofTreePanel)

        if self.session:
            self.session.reset_context()

        # Reset plan steps
        self._plan_steps = []
        self._completed_plan_steps = []

        # Reset feedback handler state
        if self._feedback_handler:
            self._feedback_handler._steps_initialized = False

        # Clear side panel
        panel_content.reset()
        side_panel.remove_class("visible")

        status_bar.update_status(
            phase=Phase.IDLE,
            status_message=None,
            tables_count=0,
            facts_count=0,
        )
        log.write(Text("Session reset.", style="green"))

    async def _redo(self, instruction: str = "") -> None:
        """Retry last query, optionally with modifications."""
        log = self.query_one("#output-log", OutputLog)

        if not self.last_problem:
            log.write(Text("No previous query to redo.", style="yellow"))
            return

        problem = self.last_problem
        if instruction:
            problem = f"{problem}\n\nModification: {instruction}"

        log.write(Text(f"Redoing: {self.last_problem[:50]}...", style="dim"))
        await self._solve(problem)

    async def _show_artifacts(self) -> None:
        """Show saved artifacts with file:// URIs."""
        log = self.query_one("#output-log", OutputLog)

        if not self.session:
            log.write(Text("No active session.", style="yellow"))
            return

        try:
            from constat.storage.registry import ConstatRegistry
            from pathlib import Path
            registry = ConstatRegistry()
            tables = registry.list_tables(user_id=self.user_id, session_id=self.session.session_id)
            registry.close()

            outputs = self.session.datastore.get_session_meta("pending_outputs") or []

            if not tables and not outputs:
                log.write(Text("No artifacts.", style="dim"))
                return

            log.write(Text("Artifacts", style="bold"))

            for t in tables:
                file_path = Path(t.file_path)
                if file_path.exists():
                    file_uri = file_path.resolve().as_uri()
                    log.write(Text(f"  📊 {t.name} ({t.row_count} rows)", style="cyan"))
                    link_markup = make_file_link_markup(file_uri, style="dim cyan underline", indent="     ")
                    log.write(link_markup)

            for output in outputs:
                if isinstance(output, dict):
                    name = output.get("name", "Unknown")
                    path = output.get("path", "")
                    if path:
                        file_path = Path(path)
                        if file_path.exists():
                            file_uri = file_path.resolve().as_uri()
                            log.write(Text(f"  📄 {name}", style="cyan"))
                            link_markup = make_file_link_markup(file_uri, style="dim cyan underline", indent="     ")
                            log.write(link_markup)
        except Exception as e:
            log.write(Text(f"Error showing artifacts: {e}", style="red"))

    async def _show_code(self, step_arg: str = "") -> None:
        """Show generated code from execution history."""
        log = self.query_one("#output-log", OutputLog)

        if not self.session:
            log.write(Text("No active session.", style="yellow"))
            return

        try:
            # Try execution_history table first (populated after solve)
            history_df = None
            if self.session.datastore:
                history_df = self.session.datastore.get_execution_history_table()

            if history_df is not None and len(history_df) > 0:
                # Filter to rows with code
                code_rows = history_df[history_df['code'].notna() & (history_df['code'] != '')]
                if len(code_rows) == 0:
                    log.write(Text("No code generated yet.", style="dim"))
                    return

                log.write(Text(f"Generated Code ({len(code_rows)} blocks)", style="bold"))
                for _, row in code_rows.iterrows():
                    step_num = row.get('step_number', 0)
                    if step_arg and str(step_num) != step_arg:
                        continue
                    step_goal = row.get('step_goal', '')
                    code = row.get('code', '')
                    success = row.get('success', True)

                    status = "✓" if success else "✗"
                    log.write(Text(f"\n--- Step {step_num}: {step_goal[:50]} {status} ---", style="dim"))
                    if code:
                        # Show code with syntax highlighting
                        syntax = Syntax(code, "python", theme="monokai", line_numbers=False)
                        log.write(syntax)
                return

            # Fallback to code_blocks session meta
            code_blocks = self.session.datastore.get_session_meta("code_blocks") if self.session.datastore else []
            if code_blocks:
                log.write(Text(f"Generated Code ({len(code_blocks)} blocks)", style="bold"))
                for i, block in enumerate(code_blocks):
                    if step_arg and str(i + 1) != step_arg:
                        continue
                    log.write(Text(f"\n--- Step {i + 1} ---", style="dim"))
                    code = block.get("code", "") if isinstance(block, dict) else str(block)
                    if code:
                        syntax = Syntax(code, "python", theme="monokai", line_numbers=False)
                        log.write(syntax)
                return

            log.write(Text("No code generated yet.", style="dim"))
        except Exception as e:
            log.write(Text(f"Error showing code: {e}", style="red"))
            logger.debug(f"_show_code error: {e}", exc_info=True)

    async def _handle_prove(self) -> None:
        """Handle /prove command - verify conversation claims with auditable proof."""
        log = self.query_one("#output-log", OutputLog)
        status_bar = self.query_one("#status-bar", StatusBar)

        if not self.session or not self.session.session_id:
            log.write(Text("No active session. Ask questions first, then use /prove to verify.", style="yellow"))
            return

        # Show UI feedback but DON'T block input
        # (don't set _is_solving = True so input stays available)
        status_bar.update_status(status_message="Proving claims...", phase=Phase.EXECUTING)
        status_bar.start_timer()
        await self._start_spinner()

        log.write(Text("Generating auditable proof for conversation claims...", style="cyan"))

        # Run prove in a background thread (keeps input responsive)
        prove_thread = threading.Thread(
            target=self._prove_in_thread,
            daemon=True
        )
        prove_thread.start()
        logger.debug("Prove thread started")

    def _prove_in_thread(self) -> None:
        """Run prove_conversation in a thread and post result message when done."""
        logger.debug("_prove_in_thread starting")
        try:
            result = self.session.prove_conversation()
        except Exception as e:
            result = {"error": str(e)}
            logger.debug(f"_prove_in_thread error: {e}", exc_info=True)
        logger.debug("_prove_in_thread complete, posting ProveComplete message")
        self.post_message(ProveComplete(result))

    async def on_prove_complete(self, message: "ProveComplete") -> None:
        """Handle ProveComplete message - display results and reset UI."""
        log = self.query_one("#output-log", OutputLog)
        status_bar = self.query_one("#status-bar", StatusBar)
        result = message.result

        try:
            if result.get("error"):
                log.write(Text(f"Error: {result['error']}", style="red"))
                return

            if result.get("no_claims"):
                log.write(Text("No question to prove. Ask a question first, then use /prove.", style="yellow"))
                return

            # Display proof result
            success = result.get("success", False)
            confidence = result.get("confidence", 0.0)

            log.write(Rule("[bold green]PROOF RESULT[/bold green]", align="left"))

            if success:
                log.write(Text(f"Proof completed (confidence: {confidence:.0%})", style="bold green"))
            else:
                log.write(Text("Proof could not be completed", style="bold red"))

            # Show derivation chain if available
            derivation = result.get("derivation_chain", "")
            if derivation:
                log.write(Text("\nDerivation:", style="bold"))
                log.write(Markdown(derivation))

            # Show output/answer if available
            output = result.get("output", "")
            if output:
                log.write(Text("\nResult:", style="bold"))
                log.write(Markdown(output))

            # Switch side panel to artifacts mode
            side_panel = self.query_one("#side-panel", SidePanel)
            panel_content = self.query_one("#proof-tree-panel", ProofTreePanel)

            # Build artifacts list from result (similar to on_solve_complete)
            artifacts = []
            import datetime

            # Get table file paths from registry
            table_file_paths = {}
            try:
                from constat.storage.registry import ConstatRegistry
                from pathlib import Path
                registry = ConstatRegistry()
                registry_tables = registry.list_tables(user_id=self.user_id, session_id=self.session.session_id)
                registry.close()
                for t in registry_tables:
                    file_path = Path(t.file_path)
                    if file_path.exists():
                        table_file_paths[t.name] = file_path.resolve().as_uri()
            except Exception as e:
                logger.debug(f"on_prove_complete: failed to get table file paths: {e}")

            # Add tables created during proof
            datastore_tables = result.get("datastore_tables", [])
            for table in datastore_tables:
                if isinstance(table, dict) and "name" in table:
                    table_name = table["name"]
                    artifacts.append({
                        "type": "table",
                        "name": table_name,
                        "row_count": table.get("row_count"),
                        "step_number": table.get("step_number", 0),
                        "created_at": table.get("created_at", ""),
                        "command": f"/show {table_name}",
                        "file_uri": table_file_paths.get(table_name, ""),
                    })

            # Sort and show artifacts
            artifacts.sort(key=lambda a: (a.get("step_number", 0), a.get("created_at", "")))
            if artifacts:
                panel_content.show_artifacts(artifacts)
                side_panel.add_class("visible")
            else:
                # Keep panel visible with proof tree if no artifacts
                pass

        finally:
            # Reset UI state
            await self._stop_spinner()
            status_bar.stop_timer()
            status_bar.update_status(status_message=None, phase=Phase.IDLE)

    async def _show_preferences(self) -> None:
        """Show current preferences."""
        log = self.query_one("#output-log", OutputLog)
        status_bar = self.query_one("#status-bar", StatusBar)

        log.write(Text("Preferences", style="bold"))
        log.write(Text(f"  verbose: {self.verbose}", style="dim"))
        log.write(Text(f"  user: {self.user_id}", style="dim"))

    async def _show_databases(self) -> None:
        """Show configured databases."""
        log = self.query_one("#output-log", OutputLog)

        if not self.session or not hasattr(self.session, 'schema_manager'):
            log.write(Text("No session available.", style="yellow"))
            return

        try:
            databases = self.session.config.databases
            log.write(Text(f"Databases ({len(databases)})", style="bold"))
            for db in databases:
                log.write(Text(f"  {db.name}: {db.uri[:50]}...", style="dim"))
        except Exception as e:
            log.write(Text(f"Error: {e}", style="red"))

    async def _discover(self, args: str) -> None:
        """Semantic search across data sources (databases, APIs, documents).

        Usage:
            /discover <query>                    - Search all sources
            /discover database <query>           - Search database tables/columns
            /discover api <query>                - Search API endpoints
            /discover document <query>           - Search documents
            /discover database <schema> <query>  - Search within specific schema
        """
        log = self.query_one("#output-log", OutputLog)
        status_bar = self.query_one(StatusBar)

        if not args.strip():
            log.write(Text("Usage: /discover [scope] <query>", style="yellow"))
            log.write(Text("  scope: database|api|document (optional)", style="dim"))
            log.write(Text("  query: semantic search terms", style="dim"))
            return

        # Parse arguments: [scope] [sub-scope] <query>
        parts = args.strip().split()
        source_filter = None
        parent_filter = None
        query_parts = parts

        # Check for scope prefix
        scope_map = {
            "database": "schema",
            "db": "schema",
            "databases": "schema",
            "api": "api",
            "apis": "api",
            "document": "document",
            "documents": "document",
            "doc": "document",
            "docs": "document",
        }

        if parts and parts[0].lower() in scope_map:
            source_filter = scope_map[parts[0].lower()]
            query_parts = parts[1:]

            # For database scope, check for schema sub-filter
            if source_filter == "schema" and len(query_parts) >= 2:
                # Check if next part looks like a schema name (not a search term)
                potential_schema = query_parts[0].lower()
                if self.session and hasattr(self.session, 'config'):
                    schema_names = [db.name.lower() for db in self.session.config.databases]
                    if potential_schema in schema_names:
                        parent_filter = query_parts[0]
                        query_parts = query_parts[1:]

        if not query_parts:
            log.write(Text("Please provide a search query.", style="yellow"))
            return

        query = " ".join(query_parts)
        status_bar.update_status(status_message=f"Searching: {query[:40]}...")

        try:
            # Get vector store from session
            if not self.session or not hasattr(self.session, 'catalog'):
                log.write(Text("No catalog available for search.", style="yellow"))
                return

            # Get embedding model
            from constat.embedding_loader import EmbeddingModelLoader
            model = EmbeddingModelLoader.get_instance().get_model()
            query_embedding = model.encode(query, normalize_embeddings=True)

            # Get the vector store from schema manager
            vector_store = None
            if hasattr(self.session, 'schema_manager') and self.session.schema_manager:
                vector_store = getattr(self.session.schema_manager, '_vector_store', None)

            if not vector_store:
                # Try to get from catalog
                if hasattr(self.session.catalog, '_vector_store'):
                    vector_store = self.session.catalog._vector_store

            if not vector_store:
                log.write(Text("No vector store available. Run /update to build index.", style="yellow"))
                return

            # Search catalog entities
            results = vector_store.search_catalog_entities(
                query_embedding=query_embedding,
                source=source_filter,
                limit=15,
                min_similarity=0.3,
            )

            # Filter by parent if specified
            if parent_filter:
                results = [r for r in results if r.get("parent_id", "").startswith(parent_filter)]

            if not results:
                scope_str = f" in {source_filter}" if source_filter else ""
                log.write(Text(f"No results found for '{query}'{scope_str}.", style="dim"))
                return

            # Display results
            scope_str = f" ({source_filter})" if source_filter else ""
            log.write(Text(f"Found {len(results)} matches{scope_str}:", style="bold"))

            from rich.table import Table
            table = Table(show_header=True, box=None, padding=(0, 1))
            table.add_column("Score", style="dim", width=5)
            table.add_column("Type", style="cyan", width=12)
            table.add_column("Name", style="bold")
            table.add_column("Details", style="dim")

            for r in results:
                score = f"{r['similarity']:.2f}"
                etype = r.get("type", "unknown")
                name = r.get("name", "")
                parent = r.get("parent_id", "")
                metadata = r.get("metadata", {})

                # Build details string
                details = []
                if parent:
                    details.append(f"in {parent}")
                if etype == "column" and metadata.get("dtype"):
                    details.append(metadata["dtype"])
                if etype == "api_endpoint" and metadata.get("method"):
                    details.append(metadata["method"])
                if metadata.get("description"):
                    desc = metadata["description"][:40]
                    details.append(desc)

                table.add_row(score, etype, name, " | ".join(details) if details else "")

            log.write(table)
            status_bar.update_status(status_message="Search complete")

        except Exception as e:
            log.write(Text(f"Error: {e}", style="red"))
            logger.debug(f"_discover error: {e}", exc_info=True)
            status_bar.update_status(status_message="Search failed")

    async def _show_context(self) -> None:
        """Show context size and token usage."""
        log = self.query_one("#output-log", OutputLog)

        if not self.session:
            log.write(Text("No active session.", style="yellow"))
            return

        try:
            stats = self.session.get_context_stats()
            if not stats:
                log.write(Text("No context stats available.", style="dim"))
                return

            log.write(Text("Context Usage", style="bold"))
            log.write(Text(f"  Total tokens: ~{stats.total_tokens:,}", style="dim"))

            # Show breakdown if available
            if hasattr(stats, 'scratchpad_tokens') and stats.scratchpad_tokens:
                log.write(Text(f"  Scratchpad: ~{stats.scratchpad_tokens:,} tokens", style="dim"))
            if hasattr(stats, 'tables_tokens') and stats.tables_tokens:
                log.write(Text(f"  Tables: ~{stats.tables_tokens:,} tokens", style="dim"))
            if hasattr(stats, 'state_tokens') and stats.state_tokens:
                log.write(Text(f"  State: ~{stats.state_tokens:,} tokens", style="dim"))

            # Show warning levels
            if hasattr(stats, 'is_critical') and stats.is_critical:
                log.write(Text("  ⚠️  Context is critical - consider /compact", style="yellow"))
            elif hasattr(stats, 'is_warning') and stats.is_warning:
                log.write(Text("  ⚡ Context is getting large", style="yellow"))
            else:
                log.write(Text("  ✓ Context size is healthy", style="green"))

        except Exception as e:
            log.write(Text(f"Error: {e}", style="red"))
            logger.debug(f"_show_context error: {e}", exc_info=True)

    async def _show_history(self) -> None:
        """Show recent sessions with IDs and summaries."""
        log = self.query_one("#output-log", OutputLog)

        try:
            from constat.storage.history import SessionHistory
            hist = SessionHistory(user_id=self.user_id)
            sessions = hist.list_sessions(limit=10)

            if not sessions:
                log.write(Text("No previous sessions.", style="dim"))
                return

            log.write(Text(f"Recent Sessions ({len(sessions)})", style="bold"))
            log.write("")

            for s in sessions:
                # Show session ID (enough to use with /resume)
                short_id = s.session_id[:20]
                status_style = "green" if s.status == "completed" else "yellow" if s.status == "active" else "dim"

                # Format: ID | Date | Status | Queries
                log.write(Text(f"  {short_id}", style="cyan bold"))

                # Show date and stats on same line
                date_str = s.created_at[:16] if s.created_at else "unknown"
                stats = f"    {date_str}  |  {s.status}  |  {s.total_queries} queries"
                log.write(Text(stats, style="dim"))

                # Show summary if available, otherwise try to get first query
                summary = s.summary
                if not summary:
                    # Try to get first query as summary
                    try:
                        detail = hist.get_session(s.session_id)
                        if detail and detail.queries:
                            first_q = detail.queries[0].question
                            summary = first_q[:80] + "..." if len(first_q) > 80 else first_q
                    except Exception:
                        pass

                if summary:
                    log.write(Text(f"    {summary}", style="white"))

                log.write("")  # Blank line between sessions

            log.write(Text("Use /resume <id> to continue a session (can use partial ID)", style="dim"))
        except Exception as e:
            log.write(Text(f"Error: {e}", style="red"))
            logger.debug(f"_show_history error: {e}", exc_info=True)

    async def _toggle_setting(self, setting: str, value: str = "") -> None:
        """Toggle or set a boolean setting."""
        log = self.query_one("#output-log", OutputLog)

        if setting == "verbose":
            if value.lower() in ("on", "true", "1"):
                self.verbose = True
            elif value.lower() in ("off", "false", "0"):
                self.verbose = False
            else:
                self.verbose = not self.verbose
            self.session_config.verbose = self.verbose
            log.write(Text(f"Verbose: {'on' if self.verbose else 'off'}", style="dim"))
        elif setting == "raw":
            if value.lower() in ("on", "true", "1"):
                self.session_config.show_raw_output = True
            elif value.lower() in ("off", "false", "0"):
                self.session_config.show_raw_output = False
            else:
                self.session_config.show_raw_output = not self.session_config.show_raw_output
            status = "on" if self.session_config.show_raw_output else "off"
            log.write(Text(f"Raw output: {status}", style="dim"))
            self._update_settings_display()
        elif setting == "insights":
            if value.lower() in ("on", "true", "1"):
                self.session_config.enable_insights = True
            elif value.lower() in ("off", "false", "0"):
                self.session_config.enable_insights = False
            else:
                self.session_config.enable_insights = not self.session_config.enable_insights
            status = "on" if self.session_config.enable_insights else "off"
            log.write(Text(f"Insights: {status}", style="dim"))
            self._update_settings_display()

    def _update_settings_display(self) -> None:
        """Update the status bar settings display."""
        raw_status = "on" if self.session_config.show_raw_output else "off"
        insights_status = "on" if self.session_config.enable_insights else "off"
        status_bar = self.query_one("#status-bar", StatusBar)
        status_bar.settings_display = f"raw:{raw_status} insights:{insights_status}"

    async def _refresh_metadata(self) -> None:
        """Refresh metadata and rebuild cache."""
        log = self.query_one("#output-log", OutputLog)
        status_bar = self.query_one("#status-bar", StatusBar)

        if not self.session:
            log.write(Text("No active session.", style="yellow"))
            return

        status_bar.update_status(status_message="Refreshing metadata...")
        try:
            self.session.schema_manager.refresh()
            log.write(Text("Metadata refreshed.", style="green"))
        except Exception as e:
            log.write(Text(f"Refresh failed: {e}", style="red"))
        finally:
            status_bar.update_status(status_message=None)

    async def _show_learnings(self) -> None:
        """Show learnings and rules."""
        log = self.query_one("#output-log", OutputLog)

        try:
            # Show rules first
            rules = self.learning_store.list_rules()
            if rules:
                log.write(Text(f"Rules ({len(rules)})", style="bold"))
                for r in rules[:10]:
                    conf = r.get("confidence", 0) * 100
                    applied = r.get("applied_count", 0)
                    log.write(Text(f"  [{conf:.0f}%] {r.get('summary', '')[:60]} (applied {applied}x)", style="dim"))

            # Show pending learnings (get all to show accurate count, but display limit)
            raw = self.learning_store.list_raw_learnings(limit=None)
            pending = [l for l in raw if not l.get("promoted_to")]
            if pending:
                log.write(Text(f"Pending Learnings ({len(pending)} total)", style="bold"))
                for l in pending[:10]:
                    cat = l.get("category", "")[:10]
                    lid = l.get("id", "")[:12]
                    log.write(Text(f"  {lid} [{cat}] {l.get('correction', '')[:50]}...", style="dim"))
                if len(pending) > 10:
                    log.write(Text(f"  ... and {len(pending) - 10} more", style="dim"))

            if not rules and not pending:
                log.write(Text("No learnings yet.", style="dim"))
            elif len(pending) >= 5:
                log.write(Text(f"  Tip: Use /consolidate to promote similar learnings to rules", style="dim cyan"))
        except Exception as e:
            log.write(Text(f"Error: {e}", style="red"))

    async def _consolidate_learnings(self) -> None:
        """Consolidate similar learnings into rules using LLM."""
        log = self.query_one("#output-log", OutputLog)
        status_bar = self.query_one("#status-bar", StatusBar)

        if not self.session:
            log.write(Text("No active session.", style="yellow"))
            return

        # Check if we have enough learnings before starting background work
        stats = self.learning_store.get_stats()
        unpromoted = stats.get("unpromoted", 0)

        if unpromoted < 2:
            log.write(Text(f"Not enough learnings to consolidate ({unpromoted} pending, need at least 2).", style="yellow"))
            return

        # Show UI feedback but don't block input
        status_bar.update_status(status_message="Consolidating learnings...", phase=Phase.EXECUTING)
        status_bar.start_timer()
        await self._start_spinner()

        log.write(Text(f"Analyzing {unpromoted} pending learnings...", style="dim"))

        # Run consolidation in background thread
        consolidate_thread = threading.Thread(
            target=self._consolidate_in_thread,
            daemon=True
        )
        consolidate_thread.start()
        logger.debug("Consolidate thread started")

    def _consolidate_in_thread(self) -> None:
        """Run consolidation in a thread and post result message when done."""
        logger.debug("_consolidate_in_thread starting")
        try:
            from constat.learning.compactor import LearningCompactor

            compactor = LearningCompactor(self.learning_store, self.session.llm)
            result = compactor.compact(dry_run=False)

            # Convert result to dict for message passing
            result_dict = {
                "success": True,
                "rules_created": result.rules_created,
                "rules_strengthened": result.rules_strengthened,
                "rules_merged": result.rules_merged,
                "learnings_archived": result.learnings_archived,
                "groups_found": result.groups_found,
                "errors": result.errors,
            }
        except Exception as e:
            result_dict = {"success": False, "error": str(e)}
            logger.debug(f"_consolidate_in_thread error: {e}", exc_info=True)

        logger.debug("_consolidate_in_thread complete, posting ConsolidateComplete message")
        self.post_message(ConsolidateComplete(result_dict))

    async def on_consolidate_complete(self, message: "ConsolidateComplete") -> None:
        """Handle ConsolidateComplete message - display results and reset UI."""
        log = self.query_one("#output-log", OutputLog)
        status_bar = self.query_one("#status-bar", StatusBar)
        result = message.result

        try:
            if not result.get("success"):
                log.write(Text(f"Error during consolidation: {result.get('error', 'Unknown error')}", style="red"))
                return

            rules_created = result.get("rules_created", 0)
            rules_strengthened = result.get("rules_strengthened", 0)
            rules_merged = result.get("rules_merged", 0)
            learnings_archived = result.get("learnings_archived", 0)
            groups_found = result.get("groups_found", 0)
            errors = result.get("errors", [])

            # Report what happened
            actions = []
            if rules_created > 0:
                actions.append(f"created {rules_created} new rules")
            if rules_strengthened > 0:
                actions.append(f"strengthened {rules_strengthened} existing rules")
            if rules_merged > 0:
                actions.append(f"merged {rules_merged} duplicate rules")

            if actions:
                summary = ", ".join(actions)
                if learnings_archived > 0:
                    summary += f" (from {learnings_archived} learnings)"
                log.write(Text(summary.capitalize() + ".", style="green"))
            elif groups_found > 0:
                log.write(Text(f"Found {groups_found} potential groups but none met confidence threshold.", style="yellow"))
            else:
                log.write(Text("No similar patterns found to consolidate.", style="dim"))

            if errors:
                for err in errors[:3]:
                    log.write(Text(f"  Error: {err}", style="red"))

        finally:
            # Stop spinner/timer and reset status
            await self._stop_spinner()
            status_bar.stop_timer()
            status_bar.update_status(status_message=None, phase=Phase.IDLE)

    async def _compact_context(self) -> None:
        """Compact context to reduce token usage."""
        log = self.query_one("#output-log", OutputLog)
        status_bar = self.query_one("#status-bar", StatusBar)

        if not self.session:
            log.write(Text("No active session.", style="yellow"))
            return

        status_bar.update_status(status_message="Compacting context...")
        try:
            stats_before = self.session.get_context_stats()
            if stats_before:
                log.write(Text(f"Before: ~{stats_before.total_tokens:,} tokens", style="dim"))

            result = self.session.compact_context(
                summarize_scratchpad=True,
                sample_tables=True,
                clear_old_state=False,
                keep_recent_steps=3,
            )

            if result:
                log.write(Text(f"{result.message}", style="green"))
                log.write(Text(result.summary(), style="dim"))
            else:
                log.write(Text("Compaction returned no result.", style="yellow"))
        except Exception as e:
            log.write(Text(f"Error during compaction: {e}", style="red"))
        finally:
            status_bar.update_status(status_message=None)

    async def _remember_fact(self, fact_text: str) -> None:
        """Remember a fact persistently."""
        log = self.query_one("#output-log", OutputLog)
        import re

        if not fact_text.strip():
            log.write(Text("Usage: /remember <fact>", style="yellow"))
            log.write(Text("  /remember churn_rate        - persist a session fact", style="dim"))
            log.write(Text("  /remember churn as baseline - persist with new name", style="dim"))
            return

        # Check if this is a session fact reference with optional rename
        session_fact_match = re.match(r'^(\S+)(?:\s+as\s+(\S+))?$', fact_text.strip())

        if session_fact_match and self.session:
            fact_name = session_fact_match.group(1)
            new_name = session_fact_match.group(2)

            session_facts = self.session.fact_resolver.get_all_facts()

            # Look for matching fact
            matching_fact = None
            matching_key = None
            for key, fact in session_facts.items():
                if key == fact_name or key == f"{fact_name}()":
                    matching_fact = fact
                    matching_key = key
                    break
                if hasattr(fact, 'name') and fact.name == fact_name:
                    matching_fact = fact
                    matching_key = key
                    break

            if matching_fact:
                persist_name = new_name if new_name else (matching_fact.name if hasattr(matching_fact, 'name') else fact_name)

                # Build context from provenance
                context_parts = []
                if hasattr(matching_fact, 'source'):
                    context_parts.append(f"Source: {matching_fact.source.value}")
                if hasattr(matching_fact, 'source_name') and matching_fact.source_name:
                    context_parts.append(f"From: {matching_fact.source_name}")
                context = "\n".join(context_parts)

                description = matching_fact.description if hasattr(matching_fact, 'description') else f"Persisted from session"

                self.fact_store.save_fact(
                    name=persist_name,
                    value=matching_fact.value if hasattr(matching_fact, 'value') else str(matching_fact),
                    description=description,
                    context=context,
                )

                display_value = matching_fact.display_value if hasattr(matching_fact, 'display_value') else str(matching_fact)[:50]
                log.write(Text(f"Remembered: {persist_name} = {display_value}", style="green"))
                log.write(Text("This fact will persist across sessions.", style="dim"))
                return

        # Not a session fact - treat as natural language
        log.write(Text(f"Fact '{fact_text[:30]}...' not found in session.", style="yellow"))
        log.write(Text("Use /facts to see available session facts.", style="dim"))

    async def _forget_fact(self, fact_name: str) -> None:
        """Forget a fact by name."""
        log = self.query_one("#output-log", OutputLog)

        if not fact_name.strip():
            log.write(Text("Usage: /forget <fact_name>", style="yellow"))
            return

        fact_name = fact_name.strip()
        found = False

        # Check persistent facts
        if self.fact_store.delete_fact(fact_name):
            log.write(Text(f"Forgot persistent fact: {fact_name}", style="green"))
            found = True

        # Check session facts
        if self.session:
            facts = self.session.fact_resolver.get_all_facts()
            if fact_name in facts:
                self.session.fact_resolver._cache.pop(fact_name, None)
                if not found:
                    log.write(Text(f"Forgot session fact: {fact_name}", style="green"))
                found = True

        if not found:
            log.write(Text(f"Fact '{fact_name}' not found.", style="yellow"))
            log.write(Text("Use /facts to see available facts.", style="dim"))

    async def _handle_correct(self, correction: str) -> None:
        """Handle /correct command - record user correction."""
        log = self.query_one("#output-log", OutputLog)
        from constat.learning.store import LearningCategory, LearningSource

        if not correction.strip():
            log.write(Text("Usage: /correct <correction>", style="yellow"))
            log.write(Text("  /correct 'active users' means logged in within 30 days", style="dim"))
            return

        self.learning_store.save_learning(
            category=LearningCategory.USER_CORRECTION,
            context={
                "previous_question": self.last_problem,
                "correction_text": correction,
            },
            correction=correction,
            source=LearningSource.EXPLICIT_COMMAND,
        )
        log.write(Text(f"Learned: {correction[:60]}{'...' if len(correction) > 60 else ''}", style="green"))

    async def _save_plan(self, name: str, shared: bool = False) -> None:
        """Save current plan for replay."""
        log = self.query_one("#output-log", OutputLog)

        if not self.session:
            log.write(Text("No active session.", style="yellow"))
            return
        if not self.last_problem:
            log.write(Text("No problem executed yet.", style="yellow"))
            return

        try:
            self.session.save_plan(name, self.last_problem, user_id=self.user_id, shared=shared)
            if shared:
                log.write(Text(f"Plan saved as shared: {name}", style="green"))
            else:
                log.write(Text(f"Plan saved: {name}", style="green"))
        except Exception as e:
            log.write(Text(f"Error saving plan: {e}", style="red"))

    async def _list_plans(self) -> None:
        """List saved plans."""
        log = self.query_one("#output-log", OutputLog)

        try:
            plans = Session.list_saved_plans(user_id=self.user_id)
            if not plans:
                log.write(Text("No saved plans.", style="dim"))
                return

            table = Table(title="Saved Plans", show_header=True, box=None)
            table.add_column("Name", style="cyan")
            table.add_column("Problem")
            table.add_column("Steps", justify="right")
            table.add_column("Type")

            for p in plans:
                plan_type = "shared" if p.get("shared") else "private"
                problem = p.get("problem", "")[:50]
                if len(p.get("problem", "")) > 50:
                    problem += "..."
                table.add_row(p["name"], problem, str(p.get("steps", 0)), plan_type)

            log.write(table)
        except Exception as e:
            log.write(Text(f"Error listing plans: {e}", style="red"))

    async def _replay_plan(self, name: str) -> None:
        """Replay a saved plan."""
        log = self.query_one("#output-log", OutputLog)

        if not self.session:
            log.write(Text("No active session.", style="yellow"))
            return

        try:
            plan_data = Session.load_saved_plan(name, user_id=self.user_id)
            self.last_problem = plan_data["problem"]
            log.write(Text(f"Replaying: {self.last_problem[:50]}...", style="dim"))
            await self._solve(self.last_problem)
        except Exception as e:
            log.write(Text(f"Error replaying plan: {e}", style="red"))

    async def _resume_session(self, session_id: str) -> None:
        """Resume a previous session."""
        log = self.query_one("#output-log", OutputLog)

        if not self.session:
            log.write(Text("No active session.", style="yellow"))
            return

        try:
            # Find matching session (partial ID match)
            sessions = self.session.history.list_sessions(limit=50)
            match = None
            for s in sessions:
                if s.session_id.startswith(session_id) or session_id in s.session_id:
                    match = s.session_id
                    break

            if not match:
                log.write(Text(f"Session not found: {session_id}", style="red"))
                return

            if self.session.resume(match):
                log.write(Text(f"Resumed session: {match[:30]}...", style="green"))
                tables = self.session.datastore.list_tables() if self.session.datastore else []
                if tables:
                    log.write(Text(f"{len(tables)} tables available - use /tables to view", style="dim"))
            else:
                log.write(Text(f"Failed to resume session: {match}", style="red"))
        except Exception as e:
            log.write(Text(f"Error resuming session: {e}", style="red"))

    async def _export_table(self, arg: str) -> None:
        """Export a table to CSV or XLSX file."""
        log = self.query_one("#output-log", OutputLog)
        from pathlib import Path

        if not arg.strip():
            log.write(Text("Usage: /export <table> [filename]", style="yellow"))
            log.write(Text("  /export orders           - Export to orders.csv", style="dim"))
            log.write(Text("  /export orders data.xlsx - Export to data.xlsx", style="dim"))
            return

        if not self.session or not self.session.datastore:
            log.write(Text("No active session.", style="yellow"))
            return

        parts = arg.strip().split(maxsplit=1)
        table_name = parts[0]
        filename = parts[1] if len(parts) > 1 else f"{table_name}.csv"

        ext = Path(filename).suffix.lower()
        if ext not in (".csv", ".xlsx"):
            log.write(Text(f"Unsupported format: {ext}. Use .csv or .xlsx", style="yellow"))
            return

        try:
            df = self.session.datastore.query(f"SELECT * FROM {table_name}")

            output_path = Path(filename).resolve()
            if ext == ".csv":
                df.to_csv(output_path, index=False)
            else:
                df.to_excel(output_path, index=False)

            log.write(Text(f"Exported {len(df)} rows to:", style="green"))
            link_markup = make_file_link_markup(output_path.as_uri(), style="cyan underline", indent="  ")
            log.write(link_markup)
        except Exception as e:
            log.write(Text(f"Export failed: {e}", style="red"))

    async def _handle_summarize(self, arg: str) -> None:
        """Generate LLM summary of plan, session, facts, or table."""
        log = self.query_one("#output-log", OutputLog)
        status_bar = self.query_one("#status-bar", StatusBar)

        if not arg.strip():
            log.write(Text("Usage: /summarize plan|session|facts|<table>", style="yellow"))
            return

        if not self.session:
            log.write(Text("No active session.", style="yellow"))
            return

        target = arg.strip().lower()
        status_bar.update_status(status_message=f"Summarizing {target}...")

        try:
            llm = self.session.router._get_provider(self.session.router.models["planning"])

            if target == "plan":
                if not self.session.plan:
                    log.write(Text("No plan to summarize.", style="yellow"))
                    return
                plan_text = str(self.session.plan)
                prompt = f"Summarize this execution plan concisely:\n\n{plan_text}"
            elif target == "session":
                tables = self.session.datastore.list_tables() if self.session.datastore else []
                facts = self.session.fact_resolver.get_all_facts() if hasattr(self.session, 'fact_resolver') else {}
                session_text = f"Tables: {len(tables)}, Facts: {len(facts)}"
                prompt = f"Summarize this session state:\n\n{session_text}"
            elif target == "facts":
                facts = self.session.fact_resolver.get_all_facts() if hasattr(self.session, 'fact_resolver') else {}
                if not facts:
                    log.write(Text("No facts to summarize.", style="yellow"))
                    return
                facts_text = "\n".join([f"{k}: {v}" for k, v in list(facts.items())[:20]])
                prompt = f"Summarize these facts concisely:\n\n{facts_text}"
            else:
                # Assume table name
                df = self.session.datastore.query(f"SELECT * FROM {target} LIMIT 100")
                if df.empty:
                    log.write(Text(f"Table '{target}' is empty.", style="yellow"))
                    return
                table_text = df.to_string()[:2000]
                prompt = f"Summarize this table data concisely:\n\n{table_text}"

            response = llm.complete(prompt, max_tokens=500)
            summary = response.content if hasattr(response, 'content') else str(response)

            log.write(Text(f"Summary: {target}", style="bold"))
            log.write(Text(summary, style="dim"))
        except Exception as e:
            log.write(Text(f"Error generating summary: {e}", style="red"))
        finally:
            status_bar.update_status(status_message=None)

    async def _handle_audit(self) -> None:
        """Re-derive last result with full audit trail."""
        log = self.query_one("#output-log", OutputLog)
        status_bar = self.query_one("#status-bar", StatusBar)

        if not self.session:
            log.write(Text("No active session. Ask a question first.", style="yellow"))
            return

        status_bar.update_status(status_message="Re-deriving with audit trail...")
        try:
            result = self.session.audit()

            if result.get("success"):
                output = result.get("output", "")
                if output:
                    log.write(Text("Audit Result", style="bold green"))
                    log.write(Text(output, style="dim"))

                verification = result.get("verification")
                if verification:
                    status = verification.get("verified", False)
                    msg = verification.get("message", "")
                    if status:
                        log.write(Text(f"Verified: {msg}", style="green"))
                    else:
                        log.write(Text(f"Discrepancy: {msg}", style="yellow"))
            else:
                error = result.get("error", "Unknown error")
                log.write(Text(f"Audit failed: {error}", style="red"))
        except Exception as e:
            log.write(Text(f"Error during audit: {e}", style="red"))
        finally:
            status_bar.update_status(status_message=None)

    async def _show_files(self) -> None:
        """Show data files."""
        log = self.query_one("#output-log", OutputLog)

        if not self.session:
            log.write(Text("No active session.", style="yellow"))
            return

        try:
            # Get files from config
            files = getattr(self.session.config, 'files', [])
            if not files:
                log.write(Text("No data files configured.", style="dim"))
                return

            log.write(Text(f"Data Files ({len(files)})", style="bold"))
            for f in files:
                name = f.get('name', 'unknown') if isinstance(f, dict) else str(f)
                uri = f.get('uri', '') if isinstance(f, dict) else ''
                log.write(Text(f"  {name}", style="cyan"))
                if uri:
                    log.write(Text(f"    {uri}", style="dim underline"))
        except Exception as e:
            log.write(Text(f"Error: {e}", style="red"))

    async def _solve(self, problem: str) -> None:
        """Solve a problem - starts worker thread, result comes via message."""
        log = self.query_one("#output-log", OutputLog)
        status_bar = self.query_one("#status-bar", StatusBar)

        self._is_solving = True
        self.last_problem = problem
        self.suggestions = []

        # Start spinner and timer
        status_bar.start_timer()
        status_bar.update_status(status_message="Analyzing question...")
        await self._start_spinner()

        # Run solve in a regular thread (like memray pattern)
        # Result will be posted via SolveComplete message
        solve_thread = threading.Thread(
            target=self._solve_in_thread,
            args=(problem,),
            daemon=True
        )
        solve_thread.start()
        logger.debug("Solve thread started")

    def _run_solve(self, problem: str) -> dict:
        """Run session.solve() synchronously (called from worker thread)."""
        try:
            if self.session.session_id:
                return self.session.follow_up(problem)
            else:
                return self.session.solve(problem)
        except Exception as e:
            return {"error": str(e)}

    def _solve_in_thread(self, problem: str) -> None:
        """Run solve in a thread and post result message when done."""
        logger.debug("_solve_in_thread starting")
        result = self._run_solve(problem)
        logger.debug("_solve_in_thread complete, posting SolveComplete message")
        # Post result via message system (thread-safe)
        self.post_message(SolveComplete(result))

    async def _start_spinner(self) -> None:
        """Start the spinner animation."""
        if self._spinner_running:
            return

        self._spinner_running = True
        self._spinner_task = asyncio.create_task(self._animate_spinner())

    async def _stop_spinner(self) -> None:
        """Stop the spinner animation."""
        self._spinner_running = False
        if self._spinner_task:
            self._spinner_task.cancel()
            try:
                await self._spinner_task
            except asyncio.CancelledError:
                pass
            self._spinner_task = None

    async def _animate_spinner(self) -> None:
        """Animate the spinner in the status bar."""
        status_bar = self.query_one("#status-bar", StatusBar)
        while self._spinner_running:
            status_bar.advance_spinner()
            await asyncio.sleep(0.1)

    def action_interrupt(self) -> None:
        """Handle Ctrl+C - cancel execution, stop timer and spinner."""
        # Cancel execution if session is active
        if self.session:
            self.session.cancel_execution()

        # Stop timer and spinner
        status_bar = self.query_one("#status-bar", StatusBar)
        status_bar.stop_timer()
        self._stop_spinner()

        # Reset solving state (keep queue for user to retry)
        self._is_solving = False

        # Release any waiting approval/clarification
        self._approval_event.set()
        self._clarification_event.set()

        log = self.query_one("#output-log", OutputLog)
        queued_count = len(self._queued_input)
        if queued_count > 0:
            log.write(Text(f"Interrupted. ({queued_count} queued input(s) preserved)", style="dim"))
        else:
            log.write(Text("Interrupted.", style="dim"))

    def action_quit(self) -> None:
        """Handle quit."""
        self._app_running = False
        # Release any waiting threads
        self._approval_event.set()
        self._clarification_event.set()
        self.exit()

    def action_open_file(self, file_path: str) -> None:
        """Open a file with the system's default application."""
        import subprocess
        import platform

        status_bar = self.query_one("#status-bar", StatusBar)

        try:
            # Handle file:// URI format
            if file_path.startswith("file://"):
                file_path = file_path[7:]  # Remove file:// prefix

            system = platform.system()
            if system == "Darwin":  # macOS
                subprocess.run(["open", file_path], check=True)
            elif system == "Windows":
                subprocess.run(["start", "", file_path], shell=True, check=True)
            else:  # Linux and others
                subprocess.run(["xdg-open", file_path], check=True)

            # Show in status bar (will clear on next action)
            status_bar.update_status(status_message=f"Opened: {Path(file_path).name}")
        except Exception as e:
            status_bar.update_status(status_message=f"Failed to open: {e}")

    def action_shrink_panel(self) -> None:
        """Shrink the side panel (make output log larger)."""
        if self._panel_ratio_index > 0:
            self._panel_ratio_index -= 1
            self._update_panel_sizes()

    def action_expand_panel(self) -> None:
        """Expand the side panel (make output log smaller)."""
        if self._panel_ratio_index < len(self.PANEL_RATIOS) - 1:
            self._panel_ratio_index += 1
            self._update_panel_sizes()

    def action_copy_output(self) -> None:
        """Copy output log content to clipboard."""
        try:
            output_log = self.query_one("#output-log", OutputLog)
            content = self._extract_log_content(output_log)
            if content.strip():
                self.copy_to_clipboard(content)
                status_bar = self.query_one("#status-bar", StatusBar)
                status_bar.update_status(status_message="Output copied")
        except Exception as e:
            logger.debug(f"Copy output failed: {e}")

    def action_copy_panel(self) -> None:
        """Copy side panel content to clipboard."""
        try:
            panel = self.query_one("#proof-tree-panel", ProofTreePanel)
            content = self._extract_log_content(panel)
            if content.strip():
                self.copy_to_clipboard(content)
                status_bar = self.query_one("#status-bar", StatusBar)
                status_bar.update_status(status_message="Panel copied")
        except Exception as e:
            logger.debug(f"Copy panel failed: {e}")

    def _extract_log_content(self, log_widget) -> str:
        """Extract plain text content from a RichLog widget."""
        import re
        from rich.text import Text as RichText

        if hasattr(log_widget, 'lines'):
            text_lines = []
            for line in log_widget.lines:
                if isinstance(line, RichText):
                    # Rich Text object - use .plain property
                    text_lines.append(line.plain)
                elif hasattr(line, 'plain'):
                    text_lines.append(line.plain)
                elif hasattr(line, 'text'):
                    # Some objects have .text property
                    text_lines.append(line.text)
                elif hasattr(line, '_segments'):
                    # Strip objects have _segments - extract text from each segment
                    segment_texts = []
                    for seg in line._segments:
                        if hasattr(seg, 'text'):
                            segment_texts.append(seg.text)
                        elif isinstance(seg, tuple) and len(seg) > 0:
                            segment_texts.append(str(seg[0]))
                    text_lines.append("".join(segment_texts))
                elif isinstance(line, str):
                    # String might have markup - strip it
                    plain = re.sub(r'\[/?[^\]]+\]', '', line)
                    text_lines.append(plain)
                else:
                    # Try to get segments from the object
                    try:
                        if hasattr(line, '__iter__'):
                            segment_texts = []
                            for item in line:
                                if hasattr(item, 'text'):
                                    segment_texts.append(item.text)
                            if segment_texts:
                                text_lines.append("".join(segment_texts))
                                continue
                    except Exception:
                        pass
                    # Last resort - convert to string and strip
                    text_lines.append(str(line))
            return "\n".join(text_lines)
        return ""

    def _update_panel_sizes(self) -> None:
        """Update panel widths based on current ratio."""
        output_ratio, side_ratio = self.PANEL_RATIOS[self._panel_ratio_index]

        output_container = self.query_one("#output-container", Vertical)
        side_panel = self.query_one("#side-panel", SidePanel)

        # Update styles dynamically
        output_container.styles.width = f"{output_ratio}fr"
        side_panel.styles.width = f"{side_ratio}fr"

        # Update panel ratio display in status bar
        status_bar = self.query_one("#status-bar", StatusBar)
        status_bar.panel_ratio = f"{output_ratio}:{side_ratio}"


def run_textual_repl(
    config_path: str,
    verbose: bool = False,
    problem: Optional[str] = None,
    user_id: str = "default",
    auto_resume: bool = False,
    debug: bool = False,
) -> None:
    """Run the Textual-based REPL."""
    from rich.console import Console
    console = Console()

    config = Config.from_yaml(config_path)

    # Create session BEFORE starting Textual to avoid multiprocessing lock issues
    # (SentenceTransformer uses tqdm which conflicts with Textual's event loop)
    with console.status("[bold]Initializing session...", spinner="dots"):
        session_config = SessionConfig(
            verbose=verbose,
            auto_approve=False,  # Show plan and ask for approval
            require_approval=True,  # Require user approval of plans
            ask_clarifications=True,  # Enable clarifications
            skip_clarification=False,
        )
        session = Session(
            config,
            session_config=session_config,
            user_id=user_id,
        )
    # Note: Feedback handler is registered in ConstatREPLApp.on_mount()

    app = ConstatREPLApp(
        config=config,
        verbose=verbose,
        user_id=user_id,
        initial_problem=problem,
        auto_resume=auto_resume,
        debug=debug,
        session=session,  # Pass pre-created session
    )
    app.run()


if __name__ == "__main__":
    # For testing: python -m constat.textual_repl
    import sys
    if len(sys.argv) > 1:
        run_textual_repl(sys.argv[1])
    else:
        print("Usage: python -m constat.textual_repl <config.yaml>")
