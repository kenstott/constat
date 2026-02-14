# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""ConstatREPLApp core (CSS, bindings, init, compose, lifecycle, actions, display helpers) + run_textual_repl()."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from pathlib import Path
from typing import Optional

from rich.rule import Rule
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, Horizontal
from textual.css.query import NoMatches
from textual.widgets import Static, Input

from constat.core.config import Config
from constat.execution.mode import PlanApprovalRequest, PlanApprovalResponse
from constat.messages import get_starter_suggestions, get_vera_adjectives, get_vera_tagline
from constat.session import Session, SessionConfig, ClarificationRequest, ClarificationResponse
from constat.storage.facts import FactStore
from constat.storage.learnings import LearningStore
from constat.textual_repl._commands import CommandsMixin
from constat.textual_repl._feedback import TextualFeedbackHandler
from constat.textual_repl._messages import (
    ShowApprovalUI, ShowClarificationUI, )
from constat.textual_repl._operations import OperationsMixin
from constat.textual_repl._role_screen import RoleSelectorScreen
from constat.textual_repl._widgets import (
    StatusBar, ConstatInput, OutputLog, SidePanel, ProofTreePanel,
    markdown_to_rich_markup, linkify_artifact_references,
)

logger = logging.getLogger(__name__)


class ConstatREPLApp(OperationsMixin, CommandsMixin, App):
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
        Binding("ctrl+shift+r", "select_role", "Select role", show=False),
    ]

    PANEL_RATIOS = [(4, 1), (3, 1), (2, 1), (1, 1), (1, 2), (1, 3)]
    DEFAULT_RATIO_INDEX = 2

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

        self.session_config = SessionConfig(verbose=verbose)
        self.session: Optional[Session] = session
        self.fact_store = FactStore(user_id=user_id)
        self.learning_store = LearningStore(user_id=user_id)

        self.suggestions: list[str] = []
        self.last_problem = ""

        self._spinner_running = False
        self._spinner_task: Optional[asyncio.Task] = None

        self._clarification_event = threading.Event()
        self._clarification_request: Optional[ClarificationRequest] = None
        self._clarification_response: Optional[ClarificationResponse] = None
        self._awaiting_clarification = False
        self._clarification_answers: dict[str, str] = {}
        self._current_question_idx = 0

        self._approval_event = threading.Event()
        self._approval_request: Optional[PlanApprovalRequest] = None
        self._approval_response: Optional[PlanApprovalResponse] = None
        self._awaiting_approval = False

        self._app_running = True
        self._pending_result = None
        self._is_solving = False
        self._queued_input: list[str] = []

        self._panel_ratio_index = self.DEFAULT_RATIO_INDEX

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
        input_widget = self.query_one("#user-input", ConstatInput)
        input_widget.focus()
        input_widget.set_history_file(self.user_id)
        input_widget.load_history()

        output_container = self.query_one("#output-container", Vertical)
        output_container.border_title = "Output"
        output_container.border_subtitle = "📋 ^⇧O"

        side_panel = self.query_one("#side-panel", SidePanel)
        side_panel.border_title = "Panel"
        side_panel.border_subtitle = "📋 ^⇧P"

        await self._create_session()

        self._update_settings_display()

        await self._show_banner()

        if self.initial_problem:
            await self._solve(self.initial_problem)

    def on_unmount(self) -> None:
        """Save history when app unmounts."""
        try:
            input_widget = self.query_one("#user-input", ConstatInput)
            input_widget.save_history()
        except (NoMatches, OSError):
            pass

    async def _create_session(self) -> None:
        """Verify session is ready and register feedback handler."""
        log = self.query_one("#output-log", OutputLog)

        if self.session:
            self._feedback_handler = TextualFeedbackHandler(self)
            self.session.on_event(self._feedback_handler.handle_event)

            self.session.set_clarification_callback(self._handle_clarification_sync)
            self.session.set_approval_callback(self._handle_approval_sync)

            self._update_role_display()

            log.write(Text("Session ready.", style="dim green"))
        else:
            log.write(Text("Error: No session provided.", style="red"))

    def _handle_clarification_sync(self, request: ClarificationRequest) -> ClarificationResponse:
        """Handle clarification request from session (called from worker thread)."""
        logger.debug(f"Clarification callback triggered with {len(request.questions)} questions")

        self._clarification_request = request
        self._clarification_answers = {}
        self._current_question_idx = 0
        self._awaiting_clarification = True
        self._clarification_event.clear()

        self.post_message(ShowClarificationUI())

        self._clarification_event.wait()

        self._awaiting_clarification = False
        return self._clarification_response or ClarificationResponse(answers={}, skip=True)

    def _handle_approval_sync(self, request: PlanApprovalRequest) -> PlanApprovalResponse:
        """Handle plan approval request from session (called from worker thread)."""
        self._approval_request = request
        self._awaiting_approval = True
        self._approval_event.clear()

        logger.debug("Approval callback triggered, posting ShowApprovalUI message")
        self.post_message(ShowApprovalUI())

        logger.debug("Message posted, now waiting on approval_event")
        self._approval_event.wait()

        self._awaiting_approval = False
        return self._approval_response or PlanApprovalResponse.reject(reason="Cancelled")

    def _get_artifacts_for_linkification(self) -> tuple[list[dict], list[dict]]:
        """Get current tables and artifacts for linkifying references."""
        tables = []
        artifacts = []
        seen_tables = set()

        if self.session and self.session.datastore:
            try:
                for t in self.session.datastore.list_tables():
                    name = t.get("name") if isinstance(t, dict) else t
                    if name and name not in seen_tables:
                        seen_tables.add(name)
                        tables.append({
                            "name": name,
                            "row_count": t.get("row_count") if isinstance(t, dict) else None,
                            "file_path": t.get("file_path", "") if isinstance(t, dict) else "",
                        })
            except Exception as e:
                logger.debug(f"Failed to get tables from datastore: {e}")

        try:
            from constat.storage.registry import ConstatRegistry
            registry = ConstatRegistry()

            if self.session:
                table_records = registry.list_tables(
                    user_id=self.user_id,
                    session_id=self.session.session_id,
                )
                for t in table_records:
                    if t.name not in seen_tables:
                        seen_tables.add(t.name)
                        tables.append({
                            "name": t.name,
                            "row_count": t.row_count,
                            "file_path": t.file_path,
                        })

                artifact_records = registry.list_artifacts(
                    user_id=self.user_id,
                    session_id=self.session.session_id,
                )
                artifacts = [
                    {
                        "name": a.name,
                        "artifact_type": a.artifact_type,
                        "file_path": a.file_path,
                    }
                    for a in artifact_records
                ]

            registry.close()
        except Exception as e:
            logger.debug(f"Failed to get artifacts from registry: {e}")

        return tables, artifacts

    def _write_with_artifact_links(self, log: "OutputLog", text: str) -> None:
        """Write text to log with artifact references converted to clickable links."""
        tables, artifacts = self._get_artifacts_for_linkification()

        logger.debug(f"_write_with_artifact_links: {len(tables)} tables, {len(artifacts)} artifacts")
        if tables:
            logger.debug(f"  Available tables: {[t['name'] for t in tables[:5]]}")

        rich_text = markdown_to_rich_markup(text)

        if not tables and not artifacts:
            log.write(Text.from_markup(rich_text))
            return

        linkified = linkify_artifact_references(rich_text, tables, artifacts)

        logger.debug(f"_write_with_artifact_links: links={'[@click=' in linkified}")
        log.write(Text.from_markup(linkified))

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
            get_vera_tagline(),
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

        if not self.initial_problem:
            log.write("")
            log.write(Text("Try asking:", style="dim"))
            suggestions = get_starter_suggestions()
            for i, s in enumerate(suggestions, 1):
                log.write(Text.assemble(
                    (f"  {i}. ", "dim"),
                    (s, "cyan"),
                ))
            self.suggestions = suggestions
            log.write("")

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission."""
        user_input = event.value.strip()

        input_widget = self.query_one("#user-input", ConstatInput)
        input_widget.value = ""

        log = self.query_one("#output-log", OutputLog)
        _status_bar = self.query_one("#status-bar", StatusBar)

        logger.debug(f"Input received: '{user_input}', awaiting_approval={self._awaiting_approval}, awaiting_clarification={self._awaiting_clarification}")

        if self._awaiting_approval and self._approval_request:
            logger.debug("Routing to approval handler")
            await self._handle_approval_answer(user_input)
            return

        if self._awaiting_clarification and self._clarification_request:
            logger.debug("Routing to clarification handler")
            await self._handle_clarification_answer(user_input)
            return

        if not user_input:
            return

        input_widget.add_to_history(user_input)

        log.write(Rule("[bold green]YOU[/bold green]", align="right"))
        log.write(Text(f"> {user_input}"))
        log.write("")

        lower_input = user_input.lower()

        if self.suggestions and lower_input.isdigit():
            idx = int(lower_input) - 1
            if 0 <= idx < len(self.suggestions):
                user_input = self.suggestions[idx]
            else:
                log.write(Text(f"No suggestion #{lower_input}", style="yellow"))
                return
        elif self.suggestions and lower_input in ("ok", "yes", "sure", "y"):
            user_input = self.suggestions[0]

        if user_input.startswith("/"):
            await self._handle_command(user_input)
        else:
            if self._is_solving:
                self._queued_input.append(user_input)
                log.write(Text(f"  (queued - will process after current solve completes)", style="dim cyan"))
                status_bar = self.query_one("#status-bar", StatusBar)
                status_bar.update_status(status_message=f"Solving... ({len(self._queued_input)} queued)")
            else:
                await self._solve(user_input)

    def _focus_input(self) -> None:
        """Focus the input widget."""
        input_widget = self.query_one("#user-input", Input)
        logger.debug(f"_focus_input called, current focus={self.focused}, input.disabled={input_widget.disabled}")
        self.set_focus(input_widget)
        logger.debug(f"_focus_input complete, new focus={self.focused}")

    def _update_settings_display(self) -> None:
        """Update the status bar settings display."""
        raw_status = "on" if self.session_config.show_raw_output else "off"
        insights_status = "on" if self.session_config.enable_insights else "off"
        status_bar = self.query_one("#status-bar", StatusBar)
        status_bar.settings_display = f"raw:{raw_status} insights:{insights_status}"

    def _update_role_display(self) -> None:
        """Update the status bar role display."""
        status_bar = self.query_one("#status-bar", StatusBar)
        if self.session and hasattr(self.session, "role_manager"):
            role_name = self.session.role_manager.active_role_name
            status_bar.role_display = role_name or ""
        else:
            status_bar.role_display = ""

    def action_interrupt(self) -> None:
        """Handle Ctrl+C - cancel execution, stop timer and spinner."""
        if self.session:
            self.session.cancel_execution()

        status_bar = self.query_one("#status-bar", StatusBar)
        status_bar.stop_timer()
        self._stop_spinner()

        self._is_solving = False

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
        try:
            input_widget = self.query_one("#user-input", ConstatInput)
            input_widget.save_history()
        except (NoMatches, OSError):
            pass
        self._approval_event.set()
        self._clarification_event.set()
        self.exit()

    def action_open_file(self, file_path: str) -> None:
        """Open a file with the system's default application."""
        import subprocess
        import platform

        status_bar = self.query_one("#status-bar", StatusBar)

        try:
            if file_path.startswith("file://"):
                file_path = file_path[7:]

            system = platform.system()
            if system == "Darwin":
                subprocess.run(["open", file_path], check=True)
            elif system == "Windows":
                subprocess.run(["start", "", file_path], shell=True, check=True)
            else:
                subprocess.run(["xdg-open", file_path], check=True)

            status_bar.update_status(status_message=f"Opened: {Path(file_path).name}")
        except Exception as e:
            status_bar.update_status(status_message=f"Failed to open: {e}")

    def action_open_artifact(self, artifact_type: str, artifact_name: str) -> None:
        """Open an artifact by name (table, file, chart, etc.)."""
        import subprocess
        import platform

        status_bar = self.query_one("#status-bar", StatusBar)
        log = self.query_one("#output-log", OutputLog)

        try:
            from constat.storage.registry import ConstatRegistry

            registry = ConstatRegistry()

            if artifact_type == "table":
                tables = registry.list_tables(
                    user_id=self.user_id,
                    session_id=self.session.session_id if self.session else None,
                )
                table = next((t for t in tables if t.name == artifact_name), None)

                if table and Path(table.file_path).exists():
                    file_path = table.file_path
                    system = platform.system()
                    if system == "Darwin":
                        subprocess.run(["open", file_path], check=True)
                    elif system == "Windows":
                        subprocess.run(["start", "", file_path], shell=True, check=True)
                    else:
                        subprocess.run(["xdg-open", file_path], check=True)
                    status_bar.update_status(status_message=f"Opened table: {artifact_name}")
                else:
                    if self.session and self.session.datastore:
                        df = self.session.datastore.get_table(artifact_name)
                        if df is not None:
                            log.write(Text(f"\n{artifact_name} ({len(df)} rows):", style="bold cyan"))
                            preview = df.head(10).to_string()
                            log.write(Text(preview, style="dim"))
                            if len(df) > 10:
                                log.write(Text(f"... ({len(df) - 10} more rows)", style="dim"))
                            status_bar.update_status(status_message=f"Showing: {artifact_name}")
                        else:
                            status_bar.update_status(status_message=f"Table not found: {artifact_name}")
                    else:
                        status_bar.update_status(status_message=f"Table not found: {artifact_name}")
            else:
                artifacts = registry.list_artifacts(
                    user_id=self.user_id,
                    session_id=self.session.session_id if self.session else None,
                )
                artifact = next((a for a in artifacts if a.name == artifact_name), None)

                if artifact and Path(artifact.file_path).exists():
                    file_path = artifact.file_path
                    system = platform.system()
                    if system == "Darwin":
                        subprocess.run(["open", file_path], check=True)
                    elif system == "Windows":
                        subprocess.run(["start", "", file_path], shell=True, check=True)
                    else:
                        subprocess.run(["xdg-open", file_path], check=True)
                    status_bar.update_status(status_message=f"Opened: {artifact_name}")
                else:
                    status_bar.update_status(status_message=f"Artifact not found: {artifact_name}")

            registry.close()

        except Exception as e:
            status_bar.update_status(status_message=f"Failed to open artifact: {e}")

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

    def action_select_role(self) -> None:
        """Open the role selector modal."""
        if not self.session or not hasattr(self.session, "role_manager"):
            return

        role_manager = self.session.role_manager
        roles = role_manager.list_roles()

        if not roles:
            log = self.query_one("#output-log", OutputLog)
            log.write(Text(
                f"No roles defined. Create roles in: {role_manager.roles_file_path}",
                style="yellow"
            ))
            return

        def handle_role_selection(selected: str | None) -> None:
            """Handle the selected role from the modal."""
            if self.session and hasattr(self.session, "role_manager"):
                self.session.role_manager.set_active_role(selected)
                self._update_role_display()
                output_log = self.query_one("#output-log", OutputLog)
                if selected:
                    output_log.write(Text(f"Role set to: {selected}", style="cyan"))
                else:
                    output_log.write(Text("Role cleared.", style="dim"))

        self.push_screen(
            RoleSelectorScreen(roles, role_manager.active_role_name),
            handle_role_selection
        )

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

    @staticmethod
    def _extract_log_content(log_widget) -> str:
        """Extract plain text content from a RichLog widget."""
        import re
        from rich.text import Text as RichText

        if hasattr(log_widget, 'lines'):
            text_lines = []
            for line in log_widget.lines:
                if isinstance(line, RichText):
                    text_lines.append(line.plain)
                elif hasattr(line, 'plain'):
                    text_lines.append(line.plain)
                elif hasattr(line, 'text'):
                    text_lines.append(line.text)
                elif hasattr(line, '_segments'):
                    segment_texts = []
                    for seg in line._segments:
                        if hasattr(seg, 'text'):
                            segment_texts.append(seg.text)
                        elif isinstance(seg, tuple) and len(seg) > 0:
                            segment_texts.append(str(seg[0]))
                    text_lines.append("".join(segment_texts))
                elif isinstance(line, str):
                    plain = re.sub(r'\[/?[^\]]+\]', '', line)
                    text_lines.append(plain)
                else:
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
                    text_lines.append(str(line))
            return "\n".join(text_lines)
        return ""

    def _update_panel_sizes(self) -> None:
        """Update panel widths based on current ratio."""
        output_ratio, side_ratio = self.PANEL_RATIOS[self._panel_ratio_index]

        output_container = self.query_one("#output-container", Vertical)
        side_panel = self.query_one("#side-panel", SidePanel)

        output_container.styles.width = f"{output_ratio}fr"
        side_panel.styles.width = f"{side_ratio}fr"

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
    os.environ["CONSTAT_REPL_MODE"] = "1"

    from rich.console import Console
    console = Console()

    config = Config.from_yaml(config_path)

    with console.status("[bold]Initializing session...", spinner="dots"):
        session_config = SessionConfig(
            verbose=verbose,
            auto_approve=False,
            require_approval=True,
            ask_clarifications=True,
            skip_clarification=False,
        )
        from constat.storage.session_store import SessionStore
        session_store = SessionStore(user_id=user_id)
        session_id = session_store.get_or_create()
        session = Session(
            config,
            session_id=session_id,
            session_config=session_config,
            user_id=user_id,
        )

        from constat.storage.facts import FactStore
        fact_store = FactStore(user_id=user_id)
        fact_store.load_into_session(session)

    app = ConstatREPLApp(
        config=config,
        verbose=verbose,
        user_id=user_id,
        initial_problem=problem,
        auto_resume=auto_resume,
        debug=debug,
        session=session,
    )
    app.run()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        run_textual_repl(sys.argv[1])
    else:
        print("Usage: python -m constat.textual_repl <config.yaml>")
