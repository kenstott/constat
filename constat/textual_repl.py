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
from textual import work
from textual.worker import Worker, get_current_worker

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


class StatusBar(Static):
    """Persistent status bar widget at the bottom of the terminal."""

    mode: reactive[Mode] = reactive(Mode.EXPLORATORY)
    phase: reactive[Phase] = reactive(Phase.IDLE)
    status_message: reactive[str | None] = reactive(None)
    tables_count: reactive[int] = reactive(0)
    facts_count: reactive[int] = reactive(0)
    spinner_frame: reactive[int] = reactive(0)

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
        # Build the content
        parts = []

        # Mode badge
        if self.mode == Mode.PROOF:
            mode_text = Text(" PROOF ", style="bold black on yellow")
        else:
            mode_text = Text(" EXPLORE ", style="bold black on cyan")
        parts.append(mode_text)

        # Status/phase text with interrupt hint during processing
        is_processing = False
        if self.status_message:
            spinner = SPINNER_FRAMES[self.spinner_frame % len(SPINNER_FRAMES)]
            parts.append(Text(f" {spinner} {self.status_message}", style="cyan"))
            is_processing = True
        elif self.phase == Phase.IDLE:
            parts.append(Text(" ready", style="dim"))
        elif self.phase == Phase.PLANNING:
            spinner = SPINNER_FRAMES[self.spinner_frame % len(SPINNER_FRAMES)]
            parts.append(Text(f" {spinner} planning", style="cyan"))
            is_processing = True
        elif self.phase == Phase.EXECUTING:
            spinner = SPINNER_FRAMES[self.spinner_frame % len(SPINNER_FRAMES)]
            parts.append(Text(f" {spinner} executing", style="green"))
            is_processing = True
        elif self.phase == Phase.AWAITING_APPROVAL:
            parts.append(Text(" awaiting approval", style="yellow"))
        elif self.phase == Phase.FAILED:
            parts.append(Text(" failed", style="bold red"))

        # Add interrupt hint when processing
        if is_processing:
            parts.append(Text("  (Ctrl+C or ESC to interrupt)", style="dim"))

        # Stats
        stats_parts = []
        if self.tables_count > 0:
            stats_parts.append(f"tables: {self.tables_count}")
        if self.facts_count > 0:
            stats_parts.append(f"facts: {self.facts_count}")
        if stats_parts:
            stats_text = "  ".join(stats_parts)
            parts.append(Text(f"    {stats_text}", style="dim"))

        # Combine into single line
        content = Text()
        for part in parts:
            content.append_text(part)

        # Create the full display with rule line
        terminal_width = shutil.get_terminal_size().columns
        rule_line = "─" * terminal_width

        return Text.assemble(
            (rule_line, "dim"),
            "\n",
            content,
        )

    # Sentinel for "not provided" vs "explicitly None"
    _NOT_PROVIDED = object()

    def update_status(
        self,
        mode: Mode = None,
        phase: Phase = None,
        status_message: str | None = _NOT_PROVIDED,
        tables_count: int = None,
        facts_count: int = None,
    ) -> None:
        """Update status bar values.

        Note: Pass status_message=None to explicitly clear the message (show phase-based status).
        Omit status_message to leave it unchanged.
        """
        if mode is not None:
            self.mode = mode
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


class ConstatInput(Input):
    """Input widget styled for Constat REPL."""

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

    class Submitted(Message):
        """Message sent when input is submitted."""
        def __init__(self, value: str) -> None:
            self.value = value
            super().__init__()


class OutputLog(RichLog):
    """Scrollable output area for Rich content."""

    DEFAULT_CSS = """
    OutputLog {
        scrollbar-gutter: stable;
        padding: 0 1;
    }
    """


class SidePanelContent(Static):
    """Panel that shows DFD during approval, proof tree during execution, artifacts after.

    Uses the actual ProofTree class from constat.proof_tree for proper hierarchical display.
    """

    DEFAULT_CSS = """
    SidePanelContent {
        width: 100%;
        height: 100%;
    }
    """

    # Panel modes
    MODE_DFD = "dfd"
    MODE_PROOF_TREE = "proof_tree"
    MODE_ARTIFACTS = "artifacts"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._mode = self.MODE_DFD
        self._proof_tree: Optional[ProofTree] = None  # Use actual ProofTree class
        self._dag_lines: list[str] = []
        self._artifacts: list[dict] = []  # {type, name, description, command}

    def show_dfd(self, dag_lines: list[str]) -> None:
        """Show only the DFD diagram (during approval)."""
        self._dag_lines = dag_lines
        self._mode = self.MODE_DFD
        self._update_display()

    def start_proof_tree(self, conclusion: str = "") -> None:
        """Start showing proof tree (during execution)."""
        # Create actual ProofTree instance
        self._proof_tree = ProofTree("answer", conclusion)
        self._mode = self.MODE_PROOF_TREE
        self._update_display()

    def add_fact(self, name: str, description: str = "", parent_name: str = None) -> None:
        """Add a fact to the proof tree."""
        if self._proof_tree:
            self._proof_tree.add_fact(name, description, parent_name)
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

    def clear(self) -> None:
        """Clear the panel."""
        self._proof_tree = None
        self._dag_lines = []
        self._artifacts = []
        self.update("")

    def show_artifacts(self, artifacts: list[dict]) -> None:
        """Show artifacts after execution completes."""
        self._artifacts = artifacts
        self._mode = self.MODE_ARTIFACTS
        self._update_display()

    def _update_display(self) -> None:
        """Update panel content based on mode."""
        if self._mode == self.MODE_ARTIFACTS:
            self._render_artifacts()
        elif self._mode == self.MODE_DFD:
            self._render_dfd()
        else:
            self._render_proof_tree()

    def _render_dfd(self) -> None:
        """Render only the DFD diagram."""
        content = Text()
        content.append("DATA FLOW\n", style="bold cyan")
        content.append("\n")

        if self._dag_lines:
            for line in self._dag_lines:
                content.append(f"{line}\n", style="white")
        else:
            content.append("No data flow diagram.\n", style="dim")

        self.update(content)

    def _render_artifacts(self) -> None:
        """Render artifact links with clickable file:// URIs."""
        content = Text()
        content.append("ARTIFACTS\n", style="bold cyan")
        content.append("\n")

        if not self._artifacts:
            content.append("No artifacts created.\n", style="dim")
        else:
            for artifact in self._artifacts:
                artifact_type = artifact.get("type", "unknown")
                name = artifact.get("name", "Unknown")
                description = artifact.get("description", "")
                command = artifact.get("command", "")
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

                content.append(f"{icon} ", style="")

                # Make name clickable if we have a file URI
                if file_uri:
                    # Use file:// URI for clickable link
                    if not file_uri.startswith("file://"):
                        file_uri = f"file://{file_uri}"
                    # Rich hyperlink syntax: style="link URL"
                    content.append(f"{name}", style=f"bold green link {file_uri}")
                    content.append("\n")
                else:
                    content.append(f"{name}\n", style="bold green")

                if description:
                    content.append(f"   {description}\n", style="dim")
                if command:
                    content.append(f"   → {command}\n", style="cyan")
                content.append("\n")

        self.update(content)

    def _render_proof_tree(self) -> None:
        """Render the proof tree using the actual ProofTree class."""
        if not self._proof_tree:
            content = Text()
            content.append("PROOF TREE\n", style="bold yellow")
            content.append("\nWaiting for resolution...\n", style="dim")
            self.update(content)
            return

        # Use ProofTree's render method - it returns a proper Rich Tree
        tree = self._proof_tree.render()
        self.update(tree)


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

        # Mode switch events
        elif event_type == "mode_switch":
            new_mode = data.get("mode", "")
            reason = data.get("reason", "")
            if new_mode == "proof":
                status_bar.update_status(mode=Mode.PROOF, status_message="Switched to audit mode")
                log.write(Text(f"Mode: AUDIT (proof-based reasoning)", style="bold yellow"))
            else:
                status_bar.update_status(mode=Mode.EXPLORATORY, status_message="Switched to explore mode")

        # Clarification events
        elif event_type == "clarification_needed":
            status_bar.update_status(status_message="Clarification needed...")

        # Planning events
        elif event_type == "planning_start":
            status_bar.update_status(status_message="Planning approach...", phase=Phase.PLANNING)

        elif event_type == "planning_complete":
            status_bar.update_status(status_message="Plan complete, preparing execution...")

        elif event_type == "plan_ready":
            plan = data.get("plan", {})
            goal = plan.get("goal", "")
            steps = plan.get("steps", [])
            # Don't set status_message here - the approval UI will handle it
            # Just update phase (approval callback will clear spinner)
            status_bar.update_status(phase=Phase.AWAITING_APPROVAL)
            # Show plan in log
            log.write(Text(f"Plan: {goal}", style="bold cyan"))
            for i, step in enumerate(steps, 1):
                log.write(Text(f"  {i}. {step.get('goal', step)}", style="dim"))

        # Proof tree events
        elif event_type in ("proof_tree_start", "proof_start"):
            # proof_start is emitted when proof execution begins
            logger.debug(f"Handling proof_start event: {data}")
            conclusion_fact = data.get("conclusion_fact", "")
            conclusion_desc = data.get("conclusion_description", "")
            status_bar.update_status(status_message="Resolving proof tree...", phase=Phase.EXECUTING)
            log.write(Text("Proof Execution:", style="bold yellow"))
            if conclusion_desc:
                log.write(Text(f"  Conclusion: {conclusion_desc[:60]}...", style="dim"))

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
                panel_content.add_fact(f"{terminal}: {terminal_name}", "", parent_name="answer")
                logger.debug(f"dag_execution_start: added terminal {terminal} under root")

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
                        panel_content.add_fact(f"{dep_id}: {dep_name}", "", parent_name=parent_key)
                        logger.debug(f"dag_execution_start: added {dep_id} under {current_id}")
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
            result = data.get("result", "") or data.get("value", "")
            display_name = f"{inference_id}: {inference_name}" if inference_name else inference_id
            logger.debug(f"inference_complete: {display_name}, result={str(result)[:30]}")
            # Update proof tree - mark as resolved
            panel_content = self._get_panel_content()
            panel_content.update_resolved(display_name, result, source="derived", confidence=1.0)

        elif event_type == "proof_tree_complete":
            status_bar.update_status(status_message="Synthesizing answer...", phase=Phase.EXECUTING)
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

        elif event_type == "step_complete":
            step_num = event.step_number or 0
            log.write(Text(f"  Step {step_num} complete", style="green"))

        elif event_type == "step_failed":
            step_num = event.step_number or 0
            error = data.get("error", "Unknown error")
            log.write(Text(f"  Step {step_num} failed: {error}", style="red"))

        # Code generation/execution events
        elif event_type == "generating":
            step_num = event.step_number or 0
            status_bar.update_status(status_message=f"Step {step_num}: Generating code...")

        elif event_type == "executing":
            step_num = event.step_number or 0
            status_bar.update_status(status_message=f"Step {step_num}: Executing...")

        # Synthesis events
        elif event_type == "synthesizing":
            status_bar.update_status(status_message="Synthesizing answer...")

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
                log.write(Text(f"  Extracted {count} facts: {facts_str}", style="dim green"))

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

    #output-log {
        width: 2fr;
    }

    #side-panel {
        width: 1fr;
        background: $surface;
        display: none;
        border-left: solid $primary;
        padding: 1;
    }

    #side-panel.visible {
        display: block;
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
    ]

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

    def compose(self) -> ComposeResult:
        """Compose the UI layout."""
        with Horizontal(id="content-area"):
            yield OutputLog(id="output-log", highlight=True, markup=True, wrap=True)
            with SidePanel(id="side-panel"):
                yield ProofTreePanel(id="proof-tree-panel")
        yield Static("─" * 80, id="input-rule")
        with Horizontal(id="input-container"):
            yield Static("> ", id="input-prompt")
            yield Input(placeholder="Ask a question or type /help", id="user-input")
        yield StatusBar(id="status-bar")

    async def on_mount(self) -> None:
        """Initialize after mounting."""
        # Focus the input
        self.query_one("#user-input", Input).focus()

        # Create session
        await self._create_session()

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
        log.write(Text("Clarification needed:", style="bold yellow"))
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
        logger.debug(f"on_session_event: {message.event.event_type}")
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

        # Build artifacts list from result
        artifacts = []
        logger.debug(f"on_solve_complete: result keys = {list(result.keys())}")

        # Add tables as artifacts
        datastore_tables = result.get("datastore_tables", [])
        logger.debug(f"on_solve_complete: datastore_tables = {datastore_tables}")
        for table in datastore_tables:
            if isinstance(table, str):
                artifacts.append({
                    "type": "table",
                    "name": table,
                    "description": "",
                    "command": f"/show {table}",
                })
            elif isinstance(table, dict) and "name" in table:
                # Dict with name key (from datastore)
                table_name = table["name"]
                row_count = table.get("row_count", 0)
                artifacts.append({
                    "type": "table",
                    "name": table_name,
                    "description": f"{row_count} rows" if row_count else "",
                    "command": f"/show {table_name}",
                })
            elif hasattr(table, "name"):
                artifacts.append({
                    "type": "table",
                    "name": table.name,
                    "description": f"{table.row_count} rows" if hasattr(table, "row_count") else "",
                    "command": f"/show {table.name}",
                })

        # Add any visualizations/outputs
        if result.get("visualizations"):
            for viz in result.get("visualizations", []):
                artifacts.append({
                    "type": "chart",
                    "name": viz.get("name", "Chart"),
                    "description": viz.get("description", ""),
                    "command": "",
                })

        # Add pending outputs (charts, md files, etc. saved during execution)
        pending = get_pending_outputs()
        logger.debug(f"on_solve_complete: pending outputs = {pending}")
        for output in pending:
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
                "command": f"open {file_uri}" if file_uri else "",
                "file_uri": file_uri,  # Include URI for clickable links
            })

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
        if result.get("error"):
            log.write(Text(f"Error: {result['error']}", style="red"))
        elif result.get("meta_response"):
            output = result.get("output", "")
            if output:
                log.write(Markdown(output))
            self.suggestions = result.get("suggestions", [])
        elif result.get("output"):
            output = result.get("output", "")
            if output:
                log.write(Markdown(output))
            self.suggestions = result.get("suggestions", [])
        else:
            log.write(Text("No output returned.", style="dim"))

        # Show suggestions if any
        if self.suggestions:
            log.write("")
            log.write(Text("You might also ask:", style="dim"))
            for i, s in enumerate(self.suggestions, 1):
                log.write(Text.assemble(
                    (f"  {i}. ", "dim"),
                    (s, "cyan"),
                ))

        # Stop spinner and reset status to Ready
        await self._stop_spinner()
        logger.debug("on_solve_complete: resetting status bar to IDLE")
        status_bar.update_status(status_message=None, phase=Phase.IDLE)
        status_bar.refresh()  # Force refresh to ensure update is visible

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
        """Show plan approval prompt in the UI (called on main thread)."""
        if not self._approval_request:
            return

        log = self.query_one("#output-log", OutputLog)
        input_widget = self.query_one("#user-input", Input)
        status_bar = self.query_one("#status-bar", StatusBar)

        request = self._approval_request
        mode = request.mode

        # Update status bar mode
        status_bar.update_status(mode=mode)

        log.write("")
        log.write(Text("Plan ready for approval:", style="bold yellow"))

        # Show mode
        mode_str = "PROOF (auditable reasoning)" if mode == Mode.PROOF else "EXPLORE (iterative)"
        log.write(Text(f"  Mode: {mode_str}", style="cyan"))
        if request.mode_reasoning:
            log.write(Text(f"    {request.mode_reasoning}", style="dim"))

        # Show problem
        if request.problem:
            log.write(Text(f"  Problem: {request.problem}", style="white"))

        # Show steps (works for both proof and explore mode)
        if request.steps:
            log.write(Text(f"  Steps ({len(request.steps)}):", style="dim"))

            # Get approximate width for wrapping (leave room for indent)
            import textwrap
            wrap_width = max(40, shutil.get_terminal_size().columns - 10)

            for step in request.steps:
                step_num = step.get('number', '?')
                goal = step.get('goal', 'Unknown')
                step_type = step.get('type', '')

                # Format based on type (premise, inference, conclusion)
                if step_type == 'premise':
                    fact_id = step.get('fact_id', '')
                    prefix = f"    {fact_id}: "
                    style = "dim cyan"
                elif step_type == 'inference':
                    fact_id = step.get('fact_id', '')
                    prefix = f"    {fact_id}: "
                    style = "dim green"
                elif step_type == 'conclusion':
                    prefix = "    C: "
                    style = "dim yellow"
                else:
                    prefix = f"    {step_num}. "
                    style = "dim"

                # Wrap goal text to fit panel width
                indent = " " * len(prefix)
                wrapped = textwrap.fill(goal, width=wrap_width - len(prefix),
                                        initial_indent="", subsequent_indent=indent)
                log.write(Text(f"{prefix}{wrapped}", style=style))

            # Show DFD in side panel only
            self._show_dfd_in_side_panel(request.steps)

        log.write("")
        log.write(Text("  [y/Enter] Approve  [n] Reject  [or type feedback to modify plan]", style="green"))

        input_widget.placeholder = "Approve? [y/n or type feedback] (Enter=yes)"
        input_widget.value = ""
        input_widget.disabled = False  # Ensure input is enabled
        # Set status_message=None so phase-based display is used (no spinner for AWAITING_APPROVAL)
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
        """Show DFD diagram in the side panel."""
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

        try:
            diagram = render_dag(G, style='rounded')
            dag_lines = [line for line in diagram.split('\n') if line.strip()]

            # Show ONLY DFD in side panel (no proof tree nodes yet)
            side_panel = self.query_one("#side-panel", SidePanel)
            panel_content = self.query_one("#proof-tree-panel", SidePanelContent)

            panel_content.show_dfd(dag_lines)
            side_panel.add_class("visible")
            logger.debug(f"_show_dfd_in_side_panel: dag_lines={len(dag_lines)}")
        except Exception as e:
            logger.debug(f"_show_dfd_in_side_panel failed: {e}")

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

        # Clear input
        input_widget = self.query_one("#user-input", Input)
        input_widget.value = ""

        log = self.query_one("#output-log", OutputLog)
        status_bar = self.query_one("#status-bar", StatusBar)

        logger.debug(f"Input received: '{user_input}', awaiting_approval={self._awaiting_approval}, awaiting_clarification={self._awaiting_clarification}")

        # Handle approval input
        if self._awaiting_approval and self._approval_request:
            logger.debug("Routing to approval handler")
            await self._handle_approval_answer(user_input)
            return

        # Handle clarification input
        if self._awaiting_clarification and self._clarification_request:
            logger.debug("Routing to clarification handler")
            await self._handle_clarification_answer(user_input)
            return

        if not user_input:
            return

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

        # Handle empty input - use default (first suggestion)
        if not answer and current_q.suggestions:
            answer = current_q.suggestions[0]
            log.write(Text(f"  A{self._current_question_idx + 1}: {answer} (default)", style="dim green"))
        # Handle numbered suggestion selection
        elif answer.isdigit() and current_q.suggestions:
            idx = int(answer) - 1
            if 0 <= idx < len(current_q.suggestions):
                answer = current_q.suggestions[idx]
            log.write(Text(f"  A{self._current_question_idx + 1}: {answer}", style="green"))
        elif answer:
            log.write(Text(f"  A{self._current_question_idx + 1}: {answer}", style="green"))

        # Store answer (use question text as key)
        if answer:
            self._clarification_answers[current_q.text] = answer

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

        # Empty or 'y' or 'yes' = approve
        if not answer or lower in ('y', 'yes', 'ok', 'approve'):
            log.write(Text("Plan approved, executing...", style="green"))
            input_widget.placeholder = "Ask a question or type /help"
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
        elif cmd == "/verbose":
            self.verbose = not self.verbose
            self.session_config.verbose = self.verbose
            state = "on" if self.verbose else "off"
            log.write(Text(f"Verbose mode: {state}", style="dim"))
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
            ("/query <sql>", "Run SQL query"),
            ("/code [step]", "Show generated code"),
            ("/state", "Show session state"),
            ("/reset", "Clear session state"),
            ("/redo [instruction]", "Retry last query"),
            ("/verbose [on|off]", "Toggle verbose mode"),
            ("/raw [on|off]", "Toggle raw output"),
            ("/insights [on|off]", "Toggle insights"),
            ("/facts", "Show cached facts"),
            ("/proof", "Switch to proof/audit mode"),
            ("/explore", "Switch to exploratory mode"),
            ("/quit, /q", "Exit"),
        ]

        for cmd, desc in commands:
            table.add_row(cmd, desc)

        log.write(table)

    async def _show_tables(self) -> None:
        """Show available tables."""
        log = self.query_one("#output-log", OutputLog)

        if not self.session or not self.session.session_id:
            log.write(Text("No active session.", style="yellow"))
            return

        try:
            from constat.storage.registry import ConstatRegistry
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
        except Exception as e:
            log.write(Text(f"Error listing tables: {e}", style="red"))

    async def _solve(self, problem: str) -> None:
        """Solve a problem - starts worker thread, result comes via message."""
        log = self.query_one("#output-log", OutputLog)
        status_bar = self.query_one("#status-bar", StatusBar)

        self.last_problem = problem
        self.suggestions = []

        # Start spinner
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
        """Handle Ctrl+C."""
        log = self.query_one("#output-log", OutputLog)
        log.write(Text("Interrupted. Type /quit to exit.", style="dim"))

    def action_quit(self) -> None:
        """Handle quit."""
        self._app_running = False
        # Release any waiting threads
        self._approval_event.set()
        self._clarification_event.set()
        self.exit()


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
