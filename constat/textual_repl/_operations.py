# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""OperationsMixin — solve/prove/consolidate threads + approval/clarification flows."""

from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from rich.markdown import Markdown
from rich.rule import Rule
from rich.text import Text

from constat.execution.mode import Phase, PlanApprovalResponse
from constat.session import ClarificationResponse
from constat.textual_repl._messages import (
    ShowApprovalUI, ShowClarificationUI, SolveComplete, ProveComplete,
    ConsolidateComplete, DocumentAddComplete, SessionEvent,
)
from constat.textual_repl._widgets import (
    OutputLog, StatusBar, SidePanel, SidePanelContent, ProofTreePanel,
)
from constat.visualization.output import clear_pending_outputs, get_pending_outputs

if TYPE_CHECKING:
    from constat.textual_repl._app import ConstatREPLApp
    from textual.widgets import Input

logger = logging.getLogger(__name__)


class OperationsMixin:
    """Mixin providing solve/prove/approve/clarify operations for ConstatREPLApp."""

    async def _solve(self: "ConstatREPLApp", problem: str) -> None:
        """Solve a problem - starts worker thread, result comes via message."""
        _log = self.query_one("#output-log", OutputLog)
        status_bar = self.query_one("#status-bar", StatusBar)

        clear_pending_outputs()

        self._is_solving = True
        self.last_problem = problem
        self.suggestions = []

        status_bar.start_timer()
        status_bar.update_status(status_message="Analyzing question...")
        await self._start_spinner()

        solve_thread = threading.Thread(
            target=self._solve_in_thread,
            args=(problem,),
            daemon=True
        )
        solve_thread.start()
        logger.debug("Solve thread started")

    def _run_solve(self: "ConstatREPLApp", problem: str) -> dict:
        """Run session.solve() synchronously (called from worker thread)."""
        try:
            if self.session.session_id:
                return self.session.follow_up(problem)
            else:
                return self.session.solve(problem)
        except Exception as e:
            return {"error": str(e)}

    def _solve_in_thread(self: "ConstatREPLApp", problem: str) -> None:
        """Run solve in a thread and post result message when done."""
        logger.debug("_solve_in_thread starting")
        result = self._run_solve(problem)
        logger.debug("_solve_in_thread complete, posting SolveComplete message")
        self.post_message(SolveComplete(result))

    async def _handle_prove(self: "ConstatREPLApp") -> None:
        """Handle /prove command - verify conversation claims with auditable proof."""
        log = self.query_one("#output-log", OutputLog)
        status_bar = self.query_one("#status-bar", StatusBar)

        if not self.session or not self.session.session_id:
            log.write(Text("No active session. Ask questions first, then use /prove to verify.", style="yellow"))
            return

        status_bar.update_status(status_message="Proving claims...", phase=Phase.EXECUTING)
        status_bar.start_timer()
        await self._start_spinner()

        log.write(Text("Generating auditable proof for conversation claims...", style="cyan"))

        prove_thread = threading.Thread(
            target=self._prove_in_thread,
            daemon=True
        )
        prove_thread.start()
        logger.debug("Prove thread started")

    def _prove_in_thread(self: "ConstatREPLApp") -> None:
        """Run prove_conversation in a thread and post result message when done."""
        logger.debug("_prove_in_thread starting")
        try:
            result = self.session.prove_conversation()
        except Exception as e:
            result = {"error": str(e)}
            logger.debug(f"_prove_in_thread error: {e}", exc_info=True)
        logger.debug("_prove_in_thread complete, posting ProveComplete message")
        self.post_message(ProveComplete(result))

    async def on_prove_complete(self: "ConstatREPLApp", message: "ProveComplete") -> None:
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

            success = result.get("success", False)
            _confidence = result.get("confidence", 0.0)

            if success:
                log.write(Rule("[bold green]PROOF RESULT[/bold green]", align="left"))
            else:
                log.write(Rule("[bold red]PROOF RESULT[/bold red]", align="left"))
                log.write(Text("Proof could not be completed", style="bold red"))

            derivation = result.get("derivation_chain", "")
            if derivation:
                log.write(Text("\nDerivation:", style="bold"))
                log.write(Markdown(derivation))

            output = result.get("output", "")
            if output:
                log.write(Text("\nResult:", style="bold"))
                self._write_with_artifact_links(log, output)

            side_panel = self.query_one("#side-panel", SidePanel)
            panel_content = self.query_one("#proof-tree-panel", ProofTreePanel)

            artifacts = []
            import datetime

            table_file_paths = {}
            try:
                from constat.storage.registry import ConstatRegistry
                registry = ConstatRegistry()
                registry_tables = registry.list_tables(user_id=self.user_id, session_id=self.session.session_id)
                registry.close()
                for t in registry_tables:
                    file_path = Path(t.file_path)
                    if file_path.exists():
                        table_file_paths[t.name] = file_path.resolve().as_uri()
            except Exception as e:
                logger.debug(f"on_prove_complete: failed to get table file paths: {e}")

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

            artifacts.sort(key=lambda a: (a.get("step_number", 0), a.get("created_at", "")))
            if artifacts:
                panel_content.show_artifacts(artifacts)
                side_panel.add_class("visible")

        finally:
            await self._stop_spinner()
            status_bar.stop_timer()
            status_bar.update_status(status_message=None, phase=Phase.IDLE)

    async def on_solve_complete(self: "ConstatREPLApp", message: SolveComplete) -> None:
        """Handle SolveComplete message - runs on main thread."""
        logger.debug("on_solve_complete message handler called")
        result = message.result
        log = self.query_one("#output-log", OutputLog)
        status_bar = self.query_one("#status-bar", StatusBar)
        side_panel = self.query_one("#side-panel", SidePanel)
        panel_content = self.query_one("#proof-tree-panel", SidePanelContent)

        # Extract and store code blocks
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

        # Build artifacts list
        artifacts = []
        logger.debug(f"on_solve_complete: result keys = {list(result.keys())}")

        table_file_paths = {}
        published_table_names = set()
        try:
            from constat.storage.registry import ConstatRegistry
            registry = ConstatRegistry()

            all_tables = registry.list_tables(user_id=self.user_id, session_id=self.session.session_id)
            for t in all_tables:
                file_path = Path(t.file_path)
                if file_path.exists():
                    table_file_paths[t.name] = file_path.resolve().as_uri()

            published_tables = registry.list_published_tables(user_id=self.user_id, session_id=self.session.session_id)
            published_table_names = {t.name for t in published_tables}

            registry.close()
        except Exception as e:
            logger.debug(f"on_solve_complete: failed to get table file paths: {e}")

        datastore_tables = result.get("datastore_tables", [])
        logger.debug(f"on_solve_complete: datastore_tables = {datastore_tables}")
        for table in datastore_tables:
            if isinstance(table, str):
                artifacts.append({
                    "type": "table",
                    "name": table,
                    "row_count": None,
                    "step_number": 0,
                    "created_at": "",
                    "command": f"/show {table}",
                    "file_uri": table_file_paths.get(table, ""),
                })
            elif isinstance(table, dict) and "name" in table:
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

        pending = get_pending_outputs()
        logger.debug(f"on_solve_complete: pending outputs = {pending}")
        max_step = max((a.get("step_number", 0) for a in artifacts), default=0)
        for i, output in enumerate(pending):
            file_uri = output.get("file_uri", "")
            description = output.get("description", "")
            file_type = output.get("type", "file")

            if file_type == "chart" or file_uri.endswith(('.html', '.png', '.svg')):
                artifact_type = "chart"
            elif file_uri.endswith('.md'):
                artifact_type = "file"
            else:
                artifact_type = "file"

            filename = Path(file_uri).name if file_uri else description

            artifacts.append({
                "type": artifact_type,
                "name": filename,
                "description": description,
                "step_number": max_step + 1 + i,
                "created_at": datetime.datetime.now().isoformat(),
                "command": f"open {file_uri}" if file_uri else "",
                "file_uri": file_uri,
            })

        # Add DFD artifact
        if self.session and self.session.session_id:
            try:
                import tempfile
                artifacts_dir = Path(tempfile.gettempdir()) / "constat_artifacts"
                dfd_path = artifacts_dir / f"{self.session.session_id}_data_flow.txt"
                if dfd_path.exists():
                    dfd_uri = dfd_path.resolve().as_uri()
                    artifacts.insert(0, {
                        "type": "file",
                        "name": "Data Flow",
                        "description": "Data flow diagram",
                        "step_number": -1,
                        "created_at": "",
                        "command": f"open {dfd_uri}",
                        "file_uri": dfd_uri,
                    })
                    logger.debug(f"on_solve_complete: added DFD artifact {dfd_path}")
            except Exception as e:
                logger.debug(f"on_solve_complete: failed to add DFD artifact: {e}")

        # Sort and filter
        artifacts.sort(key=lambda a: (a.get("step_number", 0), a.get("created_at", "")))

        intermediate_count = 0
        if published_table_names:
            all_artifacts = artifacts
            artifacts = []
            for a in all_artifacts:
                if a.get("type") == "table":
                    if a.get("name") in published_table_names:
                        artifacts.append(a)
                    else:
                        intermediate_count += 1
                else:
                    artifacts.append(a)

        logger.debug(f"on_solve_complete: total artifacts = {len(artifacts)}, intermediate = {intermediate_count}")
        if artifacts:
            logger.debug(f"on_solve_complete: showing {len(artifacts)} artifacts in side panel")
            for a in artifacts[:3]:
                logger.debug(f"  artifact: {a}")
            panel_content.show_artifacts(artifacts)
            side_panel.add_class("visible")
        else:
            logger.debug("on_solve_complete: no artifacts, hiding side panel")
            side_panel.remove_class("visible")

        log.write(Rule("[bold blue]VERA[/bold blue]", align="left"))

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
            raw_output = result.get("output", "")
            final_answer = result.get("final_answer", "")
            is_synthesized = final_answer and final_answer != raw_output

            if self.session_config.show_raw_output:
                if raw_output:
                    log.write(Markdown(raw_output))
                if is_synthesized and self.session_config.enable_insights:
                    log.write("")
                    log.write(Text("Summary:", style="bold cyan"))
                    self._write_with_artifact_links(log, final_answer)
            else:
                if is_synthesized:
                    self._write_with_artifact_links(log, final_answer)
                elif final_answer:
                    self._write_with_artifact_links(log, final_answer)
                elif raw_output:
                    log.write(Markdown(raw_output))
                else:
                    log.write(Text("No output returned.", style="dim"))

            self.suggestions = result.get("suggestions", [])

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
            completed = []
            for s in self._plan_steps:
                if not s.get("completed"):
                    completed.append({
                        "number": s.get("number"),
                        "goal": s.get("goal"),
                        "completed": True,
                    })
            if completed:
                self._completed_plan_steps.extend(completed)
                logger.debug(f"on_solve_complete: saved {len(completed)} completed steps, total={len(self._completed_plan_steps)}")

        await self._stop_spinner()
        status_bar.stop_timer()
        logger.debug("on_solve_complete: resetting status bar to IDLE")
        status_bar.update_status(status_message=None, phase=Phase.IDLE)
        status_bar.refresh()

        self._is_solving = False
        if self._queued_input:
            next_input = self._queued_input.pop(0)
            remaining = len(self._queued_input)
            log.write(Text(f"\nProcessing queued input: {next_input}", style="cyan"))
            if remaining > 0:
                log.write(Text(f"  ({remaining} more queued)", style="dim"))
            await self._solve(next_input)

    async def on_consolidate_complete(self: "ConstatREPLApp", message: "ConsolidateComplete") -> None:
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
            await self._stop_spinner()
            status_bar.stop_timer()
            status_bar.update_status(status_message=None, phase=Phase.IDLE)

    async def on_document_add_complete(self: "ConstatREPLApp", message: "DocumentAddComplete") -> None:
        """Handle DocumentAddComplete message - display result."""
        log = self.query_one("#output-log", OutputLog)
        if message.success:
            log.write(Text(f"  {message.message}", style="green"))
        else:
            log.write(Text(f"  {message.message}", style="red"))

    def on_show_approval_ui(self: "ConstatREPLApp", _message: ShowApprovalUI) -> None:
        """Handle ShowApprovalUI message - runs on main thread."""
        logger.debug("on_show_approval_ui message handler called")
        self._show_approval_ui()

    def on_show_clarification_ui(self: "ConstatREPLApp", _message: ShowClarificationUI) -> None:
        """Handle ShowClarificationUI message - runs on main thread."""
        logger.debug("on_show_clarification_ui message handler called")
        self._show_clarification_ui()

    def on_session_event(self: "ConstatREPLApp", message: SessionEvent) -> None:
        """Handle SessionEvent message - runs on main thread."""
        event = message.event
        event_type = event.event_type
        if event_type in ("step_error", "step_failed") and event.data:
            error = event.data.get("error", "")
            error_type = event.data.get("error_type", "")
            logger.debug(f"on_session_event: {event_type} - {error_type}: {error[:500] if error else 'no error message'}")
        else:
            logger.debug(f"on_session_event: {event_type}")
        if hasattr(self, '_feedback_handler'):
            self._feedback_handler._handle_event_on_main(message.event)

    def _show_approval_ui(self: "ConstatREPLApp") -> None:
        """Show approval UI and set focus."""
        logger.debug(">>> _show_approval_ui ENTERED <<<")

        self._spinner_running = False
        if self._spinner_task:
            self._spinner_task.cancel()
            self._spinner_task = None

        self._show_approval_prompt()

        from textual.widgets import Input
        input_widget = self.query_one("#user-input", Input)
        logger.debug(f"Focusing input, disabled={input_widget.disabled}")
        self.set_focus(input_widget)

        logger.debug("_show_approval_ui complete")

    def _show_approval_prompt(self: "ConstatREPLApp") -> None:
        """Show plan approval prompt in the UI (called on main thread)."""
        if not self._approval_request:
            return

        from textual.widgets import Input
        log = self.query_one("#output-log", OutputLog)
        input_widget = self.query_one("#user-input", Input)
        status_bar = self.query_one("#status-bar", StatusBar)

        request = self._approval_request

        log.write("")
        log.write(Text("Plan ready. Approve? ↓", style="bold yellow"))

        if request.steps:
            premises = [s for s in request.steps if s.get("type") == "premise"]
            inferences = [s for s in request.steps if s.get("type") == "inference"]

            # noinspection DuplicatedCode
            if premises:
                log.write(Text(f"  Premises ({len(premises)}):", style="cyan"))
                for s in premises:
                    fact_id = s.get("fact_id", "")
                    goal = s.get("goal", "")
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
                    if "=" in goal:
                        var_name = goal.split("=", 1)[0].strip()
                    else:
                        var_name = goal[:50]
                    log.write(Text(f"    {fact_id}: {var_name}", style="dim"))

            self._show_dfd_in_side_panel(request.steps)

        log.write("")

        input_widget.placeholder = "[yes/Enter] Approve  [no] Reject  [or provide feedback]"
        input_widget.value = ""
        input_widget.disabled = False

        status_bar.stop_timer()
        status_bar.update_status(status_message=None, phase=Phase.AWAITING_APPROVAL)

    def _show_clarification_ui(self: "ConstatREPLApp") -> None:
        """Show clarification UI and set focus."""
        logger.debug("_show_clarification_ui called")

        self._spinner_running = False
        if self._spinner_task:
            self._spinner_task.cancel()
            self._spinner_task = None

        self._show_clarification_questions()

        from textual.widgets import Input
        input_widget = self.query_one("#user-input", Input)
        logger.debug(f"Focusing input, disabled={input_widget.disabled}")
        self.set_focus(input_widget)

        logger.debug("_show_clarification_ui complete")

    def _show_clarification_questions(self: "ConstatREPLApp") -> None:
        """Show clarification questions in the UI (called on main thread)."""
        if not self._clarification_request:
            return

        from textual.widgets import Input
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
                    if j == 1:
                        log.write(Text(f"      {j}. {s} [default]", style="green"))
                    else:
                        log.write(Text(f"      {j}. {s}", style="dim"))

        if questions:
            first_q = questions[0]
            default_hint = ""
            if first_q.suggestions:
                default_hint = f" [Enter={first_q.suggestions[0][:20]}...]"
            input_widget.placeholder = f"Q1 (1-{len(first_q.suggestions)} or type){default_hint}"
            input_widget.value = ""
            input_widget.disabled = False
            status_bar.update_status(status_message=f"Clarification Q1/{len(questions)}")

    async def _handle_clarification_answer(self: "ConstatREPLApp", answer: str) -> None:
        """Handle a clarification answer from the user."""
        if not self._clarification_request:
            return

        from textual.widgets import Input
        log = self.query_one("#output-log", OutputLog)
        input_widget = self.query_one("#user-input", Input)
        status_bar = self.query_one("#status-bar", StatusBar)
        questions = self._clarification_request.questions

        current_q = questions[self._current_question_idx]
        logger.debug(f"[CLARIFICATION] Handler received: answer={answer!r}, q_idx={self._current_question_idx}, q_text={current_q.text!r}, suggestions={current_q.suggestions}")

        if not answer and current_q.suggestions:
            answer = current_q.suggestions[0]
            logger.debug(f"[CLARIFICATION] Using default (empty input): {answer!r}")
            log.write(Text(f"  A{self._current_question_idx + 1}: {answer} (default)", style="dim green"))
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

        if answer:
            self._clarification_answers[current_q.text] = answer
            logger.debug(f"[CLARIFICATION] Stored: {current_q.text!r} -> {answer!r}")

        self._current_question_idx += 1

        if self._current_question_idx < len(questions):
            next_q = questions[self._current_question_idx]
            default_hint = ""
            if next_q.suggestions:
                default_hint = f" [Enter={next_q.suggestions[0][:15]}...]"
            input_widget.placeholder = f"Q{self._current_question_idx + 1} (1-{len(next_q.suggestions) if next_q.suggestions else 0} or type){default_hint}"
            status_bar.update_status(
                status_message=f"Clarification Q{self._current_question_idx + 1}/{len(questions)}"
            )
        else:
            log.write(Text("Clarifications received, continuing...", style="dim"))
            input_widget.placeholder = "Ask a question or type /help"
            status_bar.update_status(status_message="Processing with clarifications...")

            logger.debug(f"[CLARIFICATION] Creating response with answers: {self._clarification_answers}")
            self._clarification_response = ClarificationResponse(
                answers=self._clarification_answers,
                skip=False
            )
            self._clarification_event.set()

    async def _handle_approval_answer(self: "ConstatREPLApp", answer: str) -> None:
        """Handle a plan approval answer from the user."""
        from textual.widgets import Input
        log = self.query_one("#output-log", OutputLog)
        input_widget = self.query_one("#user-input", Input)
        status_bar = self.query_one("#status-bar", StatusBar)
        side_panel = self.query_one("#side-panel", SidePanel)

        lower = answer.lower() if answer else ""

        if answer.startswith("/"):
            self._awaiting_approval = False
            self._approval_request = None
            status_bar.update_status(status_message=None, phase=Phase.IDLE)
            side_panel.remove_class("visible")
            input_widget.placeholder = "Ask a question or type /help"
            self._approval_response = PlanApprovalResponse.pass_command(answer)
            self._approval_event.set()
            await self._handle_command(answer)
            return

        if not answer or lower in ('y', 'yes', 'ok', 'approve'):
            log.write(Text("Plan approved, executing...", style="green"))
            input_widget.placeholder = "Ask a question or type /help"
            status_bar.start_timer()
            status_bar.update_status(status_message="Executing plan...", phase=Phase.EXECUTING)

            self._approval_response = PlanApprovalResponse.approve()
            self._approval_event.set()

        elif lower in ('n', 'no', 'reject', 'cancel'):
            log.write(Text("Plan rejected.", style="yellow"))
            input_widget.placeholder = "Ask a question or type /help"
            status_bar.update_status(status_message=None, phase=Phase.IDLE)
            side_panel.remove_class("visible")

            self._approval_response = PlanApprovalResponse.reject(reason="User rejected")
            self._approval_event.set()

        else:
            log.write(Text(f"Suggestion noted: {answer}", style="dim"))
            input_widget.placeholder = "Ask a question or type /help"
            status_bar.start_timer()
            status_bar.update_status(status_message="Replanning with feedback...", phase=Phase.PLANNING)
            side_panel.remove_class("visible")

            self._approval_response = PlanApprovalResponse.suggest(suggestion=answer)
            self._approval_event.set()

    @staticmethod
    def _show_data_flow_dag(log: OutputLog, steps: list[dict]) -> None:
        """Display an ASCII data flow DAG."""
        try:
            from constat.visualization.box_dag import generate_proof_dfd
            diagram = generate_proof_dfd(steps, max_width=60, max_name_len=10)
            if diagram and diagram != "(No derivation graph available)":
                log.write("")
                log.write(Text("  DATA FLOW:", style="bold yellow"))
                for line in diagram.split('\n'):
                    if line.strip():
                        log.write(Text(f"      {line}", style="dim"))
        except Exception:
            pass

    def _show_dfd_in_side_panel(self: "ConstatREPLApp", steps: list[dict]) -> None:
        """Show DFD diagram in the side panel. Always shows side panel."""
        side_panel = self.query_one("#side-panel", SidePanel)
        panel_content = self.query_one("#proof-tree-panel", SidePanelContent)

        side_panel.add_class("visible")

        has_proof_format = any(
            s.get("fact_id", "").startswith(("P", "I")) and s.get("type") in ("premise", "inference")
            for s in steps
        )

        if has_proof_format:
            try:
                app_width = self.size.width
                output_ratio, side_ratio = self.PANEL_RATIOS[self._panel_ratio_index]
                total_ratio = output_ratio + side_ratio
                panel_width = max(20, (app_width * side_ratio // total_ratio) - 4)
            except (AttributeError, IndexError, ZeroDivisionError, TypeError):
                panel_width = 40

            max_name_len = max(6, min(12, panel_width // 4))

            try:
                from constat.visualization.box_dag import generate_proof_dfd
                diagram = generate_proof_dfd(steps, max_width=panel_width, max_name_len=max_name_len)
                if diagram and diagram != "(No derivation graph available)":
                    dag_lines = [line for line in diagram.split('\n') if line.strip()]
                    panel_content.show_plan(dag_lines)
                    logger.debug(f"_show_dfd_in_side_panel: dag_lines={len(dag_lines)}, panel_width={panel_width}")
                    return
            except Exception as e:
                logger.debug(f"_show_dfd_in_side_panel proof DFD failed: {e}")

        self._show_plan_steps_fallback(steps, panel_content)

    def _show_plan_steps_fallback(self: "ConstatREPLApp", steps: list[dict], panel_content: SidePanelContent) -> None:
        """Show simple step list for both plan and proof steps."""
        import textwrap

        try:
            app_width = self.size.width
            output_ratio, side_ratio = self.PANEL_RATIOS[self._panel_ratio_index]
            total_ratio = output_ratio + side_ratio
            panel_width = max(20, (app_width * side_ratio // total_ratio) - 4)
        except (AttributeError, IndexError, ZeroDivisionError, TypeError):
            panel_width = 40

        lines = []
        for s in steps:
            if "number" in s:
                num = s.get("number", "?")
                goal = s.get("goal", "")
                prefix = f"{num}. "
            else:
                fact_id = s.get("fact_id", "")
                goal = s.get("goal", "")
                step_type = s.get("type", "")
                type_prefix = "P" if step_type == "premise" else "I" if step_type == "inference" else "→"
                prefix = f"{type_prefix} {fact_id}: "

            indent = " " * len(prefix)
            wrapped = textwrap.wrap(
                goal,
                width=panel_width,
                initial_indent=prefix,
                subsequent_indent=indent,
            )
            lines.extend(wrapped if wrapped else [prefix])

        panel_content._dag_lines = lines
        panel_content._mode = panel_content.MODE_PLAN
        panel_content._update_display()

    def _show_steps_fallback(self: "ConstatREPLApp", steps: list[dict], panel_content: SidePanelContent) -> None:
        """Alias for _show_plan_steps_fallback for backward compatibility."""
        self._show_plan_steps_fallback(steps, panel_content)

    async def _start_spinner(self: "ConstatREPLApp") -> None:
        """Start the spinner animation."""
        if self._spinner_running:
            return
        self._spinner_running = True
        self._spinner_task = asyncio.create_task(self._animate_spinner())

    async def _stop_spinner(self: "ConstatREPLApp") -> None:
        """Stop the spinner animation."""
        self._spinner_running = False
        if self._spinner_task:
            self._spinner_task.cancel()
            try:
                await self._spinner_task
            except asyncio.CancelledError:
                pass
            self._spinner_task = None

    async def _animate_spinner(self: "ConstatREPLApp") -> None:
        """Animate the spinner in the status bar."""
        status_bar = self.query_one("#status-bar", StatusBar)
        while self._spinner_running:
            status_bar.advance_spinner()
            await asyncio.sleep(0.1)
