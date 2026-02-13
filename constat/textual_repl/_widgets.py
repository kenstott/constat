# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Textual widget classes and utility functions for the REPL UI."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Optional

from rich.console import RenderableType
from rich.text import Text
from textual import events
from textual.containers import ScrollableContainer
from textual.reactive import reactive
from textual.suggester import Suggester
from textual.widgets import Static, Input, RichLog

from constat.execution.mode import Phase
from constat.proof_tree import ProofTree
from constat.repl.feedback import SPINNER_FRAMES

logger = logging.getLogger(__name__)


def terminal_supports_hyperlinks() -> bool:
    """Check if the terminal supports OSC 8 hyperlinks."""
    term_program = os.environ.get("TERM_PROGRAM", "").lower()
    term = os.environ.get("TERM", "").lower()

    if term_program == "iterm.app" or os.environ.get("ITERM_SESSION_ID"):
        return True
    if os.environ.get("WT_SESSION"):
        return True
    if term_program == "kitty" or term == "xterm-kitty":
        return True
    if term_program == "wezterm":
        return True
    if term_program == "alacritty" or term == "alacritty":
        return True
    if os.environ.get("KONSOLE_VERSION"):
        return True

    vte_version = os.environ.get("VTE_VERSION", "")
    if vte_version:
        try:
            if int(vte_version) >= 5000:
                return True
        except ValueError:
            pass

    if term == "foot" or term_program == "foot":
        return True

    if os.environ.get("TMUX"):
        return False

    return False


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
    """
    if not file_uri.startswith("file://"):
        file_uri = f"file://{file_uri}"

    try:
        filepath = file_uri.replace("file://", "")
        filename = Path(filepath).name
    except Exception:
        filename = file_uri.split("/")[-1]

    escaped_uri = file_uri.replace("'", "\\'")
    return f"{indent}[@click=app.open_file('{escaped_uri}')][{style}]{filename}[/{style}][/]"


def make_file_text(file_uri: str, style: str = "cyan underline", indent: str = "") -> Text:
    """Create a Text object for a file URI display (for Static widgets).

    For RichLog with clickable links, use make_file_link_markup() instead.
    """
    if not file_uri.startswith("file://"):
        file_uri = f"file://{file_uri}"

    try:
        filepath = file_uri.replace("file://", "")
        filename = Path(filepath).name
    except Exception:
        filename = file_uri.split("/")[-1]

    return Text(f"{indent}{filename}", style=style)


def make_artifact_link_markup(
    artifact_name: str,
    artifact_type: str = "table",
    row_count: int | None = None,
    style: str = "cyan underline",
) -> str:
    """Create Textual markup for a clickable artifact reference."""
    escaped_name = artifact_name.replace("'", "\\'")
    display_text = artifact_name
    if row_count is not None:
        display_text = f"{artifact_name} ({row_count:,} rows)"
    return f"[@click=app.open_artifact('{artifact_type}', '{escaped_name}')][{style}]{display_text}[/{style}][/]"


def markdown_to_rich_markup(text: str) -> str:
    """Convert common Markdown patterns to Rich markup."""
    import re

    result = text
    result = re.sub(r'\*\*([^*]+)\*\*', r'[bold]\1[/bold]', result)
    result = re.sub(r'(?<!\*)\*([^*\n]+)\*(?!\*)', r'[italic]\1[/italic]', result)
    result = re.sub(r'^(#{1,6})\s+(.+)$', r'[bold]\2[/bold]', result, flags=re.MULTILINE)
    result = re.sub(r'^(\s*)-\s+', r'\1• ', result, flags=re.MULTILINE)
    return result


def linkify_artifact_references(
    text: str,
    tables: list[dict],
    artifacts: list[dict],
) -> str:
    """Convert artifact name references to clickable links."""
    import re

    table_map = {t['name']: t for t in tables}
    artifact_map = {a['name']: a for a in artifacts}

    desc_to_artifact = {}
    for a in artifacts:
        desc = a.get('description', '')
        if desc and len(desc) >= 10:
            desc_to_artifact[desc.lower()] = a

    # Pass 1: backtick-quoted strings
    backtick_pattern = r'`([^`]+)`(?:\s*\((\d+(?:,\d+)*)\s*rows?\))?'

    def replace_backtick_match(match):
        name = match.group(1)
        explicit_count = match.group(2)

        if name in table_map:
            table = table_map[name]
            row_count = int(explicit_count.replace(',', '')) if explicit_count else table.get('row_count')
            return make_artifact_link_markup(name, "table", row_count)

        if name in artifact_map:
            artifact = artifact_map[name]
            return make_artifact_link_markup(name, artifact.get('artifact_type', 'file'))

        for aname, artifact in artifact_map.items():
            file_path = artifact.get('file_path', '') or artifact.get('file_uri', '')
            if file_path and Path(file_path).name == name:
                return make_file_link_markup(file_path)

        name_lower = name.lower()
        for desc, artifact in desc_to_artifact.items():
            if desc == name_lower:
                file_path = artifact.get('file_path', '') or artifact.get('file_uri', '')
                if file_path:
                    return make_file_link_markup(file_path)
                return make_artifact_link_markup(artifact.get('name', name), artifact.get('artifact_type', 'file'))

        return match.group(0)

    result = re.sub(backtick_pattern, replace_backtick_match, text)

    # Pass 2: non-backticked table names followed by (N rows)
    for table_name, table in table_map.items():
        if f"'{table_name}')" in result:
            continue
        escaped_name = re.escape(table_name)
        bare_pattern = rf'\b({escaped_name})\s*\((\d+(?:,\d+)*)\s*rows?\)'

        def replace_bare_match(match, tname=table_name, _rc=table.get('row_count')):
            row_count_str = match.group(2)
            row_count = int(row_count_str.replace(',', '')) if row_count_str else _rc
            return make_artifact_link_markup(tname, "table", row_count)

        result = re.sub(bare_pattern, replace_bare_match, result)

    # Pass 3: bare distinctive table names
    for table_name, table in table_map.items():
        if f"'{table_name}')" in result:
            continue
        if '_' not in table_name or len(table_name) < 8:
            continue
        skip_names = {'the_data', 'the_table', 'new_data', 'old_data'}
        if table_name.lower() in skip_names:
            continue
        escaped_name = re.escape(table_name)
        bare_pattern = rf'\b({escaped_name})\b'

        def replace_bare_name(_match, tname=table_name, _rc=table.get('row_count')):
            return make_artifact_link_markup(tname, "table", _rc)

        result = re.sub(bare_pattern, replace_bare_name, result)

    # Pass 4: artifact descriptions
    if desc_to_artifact:
        for desc, artifact in desc_to_artifact.items():
            aname = artifact.get('name', '')
            if f"'{aname}')" in result:
                continue
            escaped_desc = re.escape(artifact.get('description', ''))
            if not escaped_desc:
                continue
            desc_pattern = rf'(?:(?:Created|Saved|Generated|Exported):\s*)?({escaped_desc})'

            def replace_desc_match(match, a=artifact):
                file_path = a.get('file_path', '') or a.get('file_uri', '')
                if file_path:
                    return make_file_link_markup(file_path)
                else:
                    return make_artifact_link_markup(a.get('name', match.group(1)), a.get('artifact_type', 'file'))

            result = re.sub(desc_pattern, replace_desc_match, result, flags=re.IGNORECASE)

    return result


class StatusBar(Static):
    """Persistent status bar widget at the bottom of the terminal."""

    phase: reactive[Phase] = reactive(Phase.IDLE)
    status_message: reactive[str | None] = reactive(None)
    tables_count: reactive[int] = reactive(0)
    facts_count: reactive[int] = reactive(0)
    spinner_frame: reactive[int] = reactive(0)
    elapsed_time: reactive[str] = reactive("")
    panel_ratio: reactive[str] = reactive("4:1")
    settings_display: reactive[str] = reactive("raw:on insights:on")
    role_display: reactive[str] = reactive("")

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._timer_start: float | None = None
        self._timer_interval = None
        self._final_time: str | None = None
        self._timer_hidden: bool = False

    def start_timer(self) -> None:
        """Start the elapsed time timer from 0."""
        import time
        self._timer_start = time.time()
        self._final_time = None
        self._timer_hidden = False
        self.elapsed_time = "0.0s"
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

    def hide_timer(self) -> None:
        """Hide the timer (used during clarifications)."""
        self.stop_timer()
        self._timer_hidden = True
        self._final_time = None
        self.elapsed_time = ""

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
        """Render the status bar."""
        content = Text()
        is_processing = bool(self.status_message) or self.phase in (Phase.PLANNING, Phase.EXECUTING)

        if is_processing:
            content.append(" (Ctrl+C to interrupt)", style="dim")
        else:
            content.append(" " * 22)

        if self.elapsed_time:
            timer_str = f"[{self.elapsed_time}]"
            style = "bold cyan" if is_processing else "dim green"
            content.append(f" {timer_str:>9}", style=style)
        elif self._final_time:
            timer_str = f"[{self._final_time}]"
            content.append(f" {timer_str:>9}", style="dim green")
        else:
            content.append(" " * 10)

        content.append(" ")
        if self.status_message:
            content.append(self.status_message, style="cyan")
        elif self.phase == Phase.IDLE:
            content.append("ready", style="dim")
        elif self.phase == Phase.PLANNING:
            content.append("planning", style="cyan")
        elif self.phase == Phase.EXECUTING:
            content.append("executing", style="green")
        elif self.phase == Phase.AWAITING_APPROVAL:
            content.append("awaiting approval", style="yellow")
        elif self.phase == Phase.FAILED:
            content.append("failed", style="bold red")

        terminal_width = shutil.get_terminal_size().columns
        rule_line = "─" * terminal_width

        role_str = f" [{self.role_display or 'no role'}]" if self.role_display or True else ""
        settings_str = f" [{self.settings_display}]"
        panel_controls = f" [◀ {self.panel_ratio} ▶]"
        right_content = role_str + settings_str + panel_controls
        right_len = len(right_content)

        current_len = len(content.plain)
        padding_needed = max(0, terminal_width - current_len - right_len)
        content.append(" " * padding_needed)
        role_style = "cyan" if self.role_display else "dim"
        content.append(role_str, style=role_style)
        content.append(settings_str, style="dim")
        content.append(panel_controls, style="dim cyan")

        return Text.assemble(
            (rule_line, "dim"),
            "\n",
            content,
        )

    _NOT_PROVIDED = object()

    def update_status(
        self,
        phase: Phase = None,
        status_message: str | None = _NOT_PROVIDED,
        tables_count: int = None,
        facts_count: int = None,
    ) -> None:
        """Update status bar values."""
        if phase is not None:
            self.phase = phase
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
        """Handle clicks on status bar elements (role selector, panel controls)."""
        terminal_width = shutil.get_terminal_size().columns

        role_str = f" [{self.role_display or 'no role'}]"
        settings_str = f" [{self.settings_display}]"
        panel_controls = f" [◀ {self.panel_ratio} ▶]"

        panel_end = terminal_width
        panel_start = panel_end - len(panel_controls)
        settings_end = panel_start
        settings_start = settings_end - len(settings_str)
        role_end = settings_start
        role_start = role_end - len(role_str)

        if event.x >= panel_start:
            relative_x = event.x - panel_start
            if relative_x <= 3:
                self.app.action_shrink_panel()
            elif relative_x >= len(panel_controls) - 3:
                self.app.action_expand_panel()
        elif role_start <= event.x < role_end:
            self.app.action_select_role()


class CommandSuggester(Suggester):
    """Provides auto-complete suggestions for REPL commands."""

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
        "/databases",
        "/database", "/db",
        "/apis",
        "/api",
        "/documents", "/docs",
        "/files", "/file",
        "/doc",
        "/context",
        "/history", "/sessions",
        "/verbose",
        "/raw",
        "/insights",
        "/update", "/refresh",
        "/learnings",
        "/consolidate", "/compact-learnings",
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

    MAX_HISTORY = 500

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("suggester", CommandSuggester(use_cache=False, case_sensitive=False))
        super().__init__(*args, **kwargs)
        self._history: list[str] = []
        self._history_index: int = -1
        self._current_input: str = ""
        self._history_file: Path | None = None

    def set_history_file(self, user_id: str) -> None:
        """Set the history file path for persistent storage."""
        self._history_file = Path(".constat") / user_id / "prompt_history.json"

    def load_history(self) -> None:
        """Load command history from file."""
        if not self._history_file or not self._history_file.exists():
            return
        try:
            import json
            with open(self._history_file, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    self._history = data[-self.MAX_HISTORY:]
                    logger.debug(f"Loaded {len(self._history)} history entries")
        except Exception as e:
            logger.debug(f"Failed to load history: {e}")

    def save_history(self) -> None:
        """Save command history to file."""
        if not self._history_file:
            return
        try:
            import json
            self._history_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._history_file, "w") as f:
                json.dump(self._history[-self.MAX_HISTORY:], f, indent=2)
            logger.debug(f"Saved {len(self._history)} history entries")
        except Exception as e:
            logger.debug(f"Failed to save history: {e}")

    def add_to_history(self, command: str) -> None:
        """Add a command to history (called after submission)."""
        if command and command.strip():
            if not self._history or self._history[-1] != command:
                self._history.append(command)
                if len(self._history) > self.MAX_HISTORY:
                    self._history = self._history[-self.MAX_HISTORY:]
                self.save_history()
        self._history_index = -1
        self._current_input = ""

    def _on_key(self, event: events.Key) -> None:
        """Handle key events for history navigation and autocomplete."""
        if event.key == "up":
            self._navigate_history(-1)
            event.prevent_default()
            event.stop()
        elif event.key == "down":
            self._navigate_history(1)
            event.prevent_default()
            event.stop()
        elif event.key == "tab":
            if self._suggestion:
                self.value = self._suggestion
                self.cursor_position = len(self.value)
                self._suggestion = ""
                event.prevent_default()
                event.stop()
        else:
            if self._history_index != -1:
                self._history_index = -1
                self._current_input = ""

    def _navigate_history(self, direction: int) -> None:
        """Navigate through command history."""
        if not self._history:
            return

        if self._history_index == -1 and direction == -1:
            self._current_input = self.value
            self._history_index = len(self._history)

        new_index = self._history_index + direction

        if new_index < 0:
            return
        elif new_index >= len(self._history):
            self._history_index = -1
            self.value = self._current_input
            self.cursor_position = len(self.value)
            return

        self._history_index = new_index
        self.value = self._history[new_index]
        self.cursor_position = len(self.value)


class OutputLog(RichLog):
    """Scrollable output area for Rich content."""

    DEFAULT_CSS = """
    OutputLog {
        scrollbar-gutter: stable;
        padding: 0 1;
        overflow-x: auto;
    }
    """


class SidePanelContent(RichLog):
    """Panel that shows DFD during approval, proof tree during execution, artifacts after."""

    DEFAULT_CSS = """
    SidePanelContent {
        width: 100%;
        height: 100%;
        scrollbar-gutter: stable;
    }
    """

    MODE_DFD = "dfd"
    MODE_PLAN = "plan"
    MODE_PROOF_TREE = "proof_tree"
    MODE_ARTIFACTS = "artifacts"
    MODE_STEPS = "steps"

    SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    PROGRESS_BAR_CHARS = ["░", "▒", "▓", "█"]

    def __init__(self, **kwargs) -> None:
        super().__init__(highlight=False, markup=True, wrap=True, **kwargs)
        self._mode = self.MODE_DFD
        self._proof_tree: Optional[ProofTree] = None
        self._dag_lines: list[str] = []
        self._artifacts: list[dict] = []
        self._steps: list[dict] = []
        self._current_step: int = 0
        self._spinner_frame: int = 0
        self._animation_timer = None
        self._is_animating: bool = False
        self._pulse_state: bool = False

    def start_animation(self) -> None:
        """Start the spinner animation for active steps."""
        if not self._is_animating:
            self._is_animating = True
            self._animation_timer = self.set_interval(0.1, self._animate_tick)
            try:
                side_panel = self.app.query_one("#side-panel")
                side_panel.add_class("executing")
            except Exception:
                pass

    # noinspection PyMethodOverriding
    def stop_animation(self) -> None:
        """Stop the spinner animation."""
        if self._is_animating:
            self._is_animating = False
            if self._animation_timer:
                self._animation_timer.stop()
                self._animation_timer = None
            try:
                side_panel = self.app.query_one("#side-panel")
                side_panel.remove_class("executing")
            except Exception:
                pass

    def _animate_tick(self) -> None:
        """Called on each animation tick to update spinner."""
        self._spinner_frame = (self._spinner_frame + 1) % len(self.SPINNER_FRAMES)
        self._pulse_state = not self._pulse_state
        has_active = any(s.get("status") in ("in_progress", "executing") for s in self._steps)
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
        self.stop_animation()
        self._proof_tree = None
        self._dag_lines = []
        self._artifacts = []
        self._steps = []
        self._current_step = 0
        self._mode = self.MODE_DFD
        super().clear()

    def clear_panel(self) -> None:
        """Clear the panel content and state (alias for reset)."""
        self.reset()

    def show_artifacts(self, artifacts: list[dict]) -> None:
        """Show artifacts after execution completes."""
        self._artifacts = artifacts
        self._mode = self.MODE_ARTIFACTS
        self._update_display()

    def start_steps(self, steps: list[dict]) -> None:
        """Initialize step tracking for exploratory mode."""
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
        """Add new steps to existing step list (for follow-up/extension queries)."""
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
        all_done = all(s.get("status") in ("complete", "failed") for s in self._steps)
        if all_done:
            self.stop_animation()
        if self._mode == self.MODE_STEPS:
            self._update_display()

    def _update_display(self) -> None:
        """Update panel content based on mode."""
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

        first_line = self._dag_lines[0].strip() if self._dag_lines else ""
        is_step_list = first_line and (first_line.startswith("P ") or first_line.startswith("I "))

        if is_step_list:
            import textwrap
            import re
            content_width = self.content_size.width - 3
            if content_width <= 0:
                try:
                    app = self.app
                    ratio_index = getattr(app, '_panel_ratio_index', 1)
                    ratios = getattr(app, 'PANEL_RATIOS', [(3, 1), (2, 1), (1, 1), (1, 2)])
                    output_ratio, side_ratio = ratios[ratio_index]
                    total_ratio = output_ratio + side_ratio
                    app_width = app.size.width if app.size.width > 0 else 120
                    panel_width = (app_width * side_ratio) // total_ratio
                    content_width = panel_width - 6
                except Exception:
                    content_width = 40
            content_width = max(20, content_width)
            for line in self._dag_lines:
                if line.strip():
                    prefix_match = re.match(r'^([PI]\s+[PI]\d+:\s*|→\s*\d*:?\s*|\d+\.\s*)', line)
                    if prefix_match:
                        prefix = prefix_match.group(1)
                        rest = line[len(prefix):]
                        cont_indent = " " * len(prefix)
                        wrapped = textwrap.wrap(rest, width=content_width - len(prefix))
                        if wrapped:
                            self.write(f"[white]{prefix}{wrapped[0]}[/white]")
                            for cont in wrapped[1:]:
                                self.write(f"[white]{cont_indent}{cont}[/white]")
                    else:
                        wrapped_lines = textwrap.wrap(line, width=content_width)
                        for wrapped in wrapped_lines:
                            self.write(f"[white]{wrapped}[/white]")
        else:
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
        """Render plan for approval (no box, just content with proper wrapping)."""
        super().clear()

        if not self._dag_lines:
            content = Text()
            content.append("No plan to display.\n", style="dim")
            self.write(content)
            return

        import textwrap
        import re

        content_width = self.content_size.width - 3
        if content_width <= 0:
            try:
                app = self.app
                ratio_index = getattr(app, '_panel_ratio_index', 1)
                ratios = getattr(app, 'PANEL_RATIOS', [(3, 1), (2, 1), (1, 1), (1, 2)])
                output_ratio, side_ratio = ratios[ratio_index]
                total_ratio = output_ratio + side_ratio
                app_width = app.size.width if app.size.width > 0 else 120
                panel_width = (app_width * side_ratio) // total_ratio
                content_width = panel_width - 6
            except Exception:
                content_width = 40
        content_width = max(20, content_width)

        for line in self._dag_lines:
            if line.strip():
                prefix_match = re.match(r'^([PI]\s+[PI]\d+:\s*|→\s*\d*:?\s*|\d+\.\s*)', line)
                if prefix_match:
                    prefix = prefix_match.group(1)
                    rest = line[len(prefix):]
                    cont_indent = " " * len(prefix)
                    rest = " ".join(rest.split())
                    wrapped = textwrap.wrap(rest, width=content_width - len(prefix), subsequent_indent="")
                    if wrapped:
                        self.write(f"[white]{prefix}{wrapped[0]}[/white]")
                        for cont in wrapped[1:]:
                            self.write(f"[white]{cont_indent}{cont}[/white]")
                else:
                    normalized = " ".join(line.split())
                    wrapped_lines = textwrap.wrap(normalized, width=content_width)
                    for wrapped in wrapped_lines:
                        self.write(f"[white]{wrapped}[/white]")

    def _render_artifacts(self) -> None:
        """Render artifact links with clickable file:// URIs."""
        super().clear()

        if not self._artifacts:
            self.write(Text("No artifacts created.", style="dim"))
        else:
            exec_history = [a for a in self._artifacts if a.get("name") == "execution_history"]
            other_artifacts = [a for a in self._artifacts if a.get("name") != "execution_history"]

            for artifact in exec_history:
                self._write_artifact_line(artifact)
            if exec_history:
                self.write("")

            for artifact in other_artifacts:
                self._write_artifact_line(artifact)

    def _write_artifact_line(self, artifact: dict) -> None:
        """Write a single artifact line with icon, name link, and row count."""
        artifact_type = artifact.get("type", "unknown")
        name = artifact.get("name", "Unknown")
        row_count = artifact.get("row_count")
        file_uri = artifact.get("file_uri", "")

        if artifact_type == "table":
            icon = "📊"
        elif artifact_type == "chart":
            icon = "📈"
        elif artifact_type == "diagram":
            icon = "🔀"
        elif artifact_type == "file":
            icon = "📄"
        else:
            icon = "📦"

        row_suffix = f" [dim]({row_count} rows)[/dim]" if row_count is not None else ""

        if file_uri:
            escaped_uri = file_uri.replace("'", "\\'")
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

            if status == "pending":
                icon = "○"
                style = "dim"
            elif status == "executing" or status == "in_progress":
                icon = self.get_spinner()
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

            import textwrap
            content_width = self.content_size.width - 3
            if content_width <= 0:
                try:
                    app = self.app
                    ratio_index = getattr(app, '_panel_ratio_index', 1)
                    ratios = getattr(app, 'PANEL_RATIOS', [(3, 1), (2, 1), (1, 1), (1, 2)])
                    output_ratio, side_ratio = ratios[ratio_index]
                    total_ratio = output_ratio + side_ratio
                    app_width = app.size.width if app.size.width > 0 else 120
                    panel_width = (app_width * side_ratio) // total_ratio
                    content_width = panel_width - 6
                except Exception:
                    content_width = 40
            content_width = max(20, content_width)
            prefix = f"{icon} Step {num}: "
            first_line_width = content_width - len(prefix)
            continuation_width = content_width - 2
            if len(goal) <= first_line_width:
                self.write(f"[{style}]{prefix}[bold]{goal}[/bold][/{style}]")
            else:
                wrapped = textwrap.wrap(goal, width=first_line_width)
                self.write(f"[{style}]{prefix}[bold]{wrapped[0]}[/bold][/{style}]")
                remainder = goal[len(wrapped[0]):].strip()
                if remainder:
                    cont_wrapped = textwrap.wrap(remainder, width=continuation_width)
                    for continuation in cont_wrapped:
                        self.write(f"[{style}]  {continuation}[/{style}]")

            meta_parts = []
            if elapsed is not None:
                meta_parts.append(f"{elapsed:.1f}s")
            if retries > 0:
                meta_parts.append(f"{retries} retries")
            if meta_parts:
                self.write(Text(f"   [{', '.join(meta_parts)}]", style="dim"))

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
    pass
