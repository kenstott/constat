# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""TextualFeedbackHandler — event handler that updates Textual UI during session execution."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rich.text import Text

from constat.execution.mode import Phase
from constat.textual_repl._messages import SessionEvent
from constat.textual_repl._widgets import OutputLog, StatusBar, SidePanel, SidePanelContent

if TYPE_CHECKING:
    from constat.textual_repl._app import ConstatREPLApp

logger = logging.getLogger(__name__)


class TextualFeedbackHandler:
    """
    Event handler that updates Textual UI during session execution.

    Receives events from Session and updates the status bar and log.
    """

    def __init__(self, app: "ConstatREPLApp"):
        self.app = app
        self._proof_items: dict[str, dict] = {}
        self._current_step = 0
        self._total_steps = 0
        self._steps_initialized = False

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
        self.app.post_message(SessionEvent(event))

    def _handle_event_on_main(self, event) -> None:
        """Handle event on main thread where UI updates are safe."""
        event_type = event.event_type
        data = event.data
        log = self._get_log()
        status_bar = self._get_status_bar()

        if event_type == "progress":
            message = data.get("message", "Processing...")
            status_bar.update_status(status_message=message)

        elif event_type == "clarification_needed":
            status_bar.hide_timer()
            status_bar.update_status(status_message="Clarification needed...")

        elif event_type == "planning_start":
            status_bar.start_timer()
            status_bar.update_status(status_message="Planning approach...", phase=Phase.PLANNING)

        elif event_type == "planning_complete":
            status_bar.update_status(status_message="Plan complete, validating...")

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

            app = self.app

            if is_followup and app._completed_plan_steps:
                for s in app._completed_plan_steps:
                    s["completed"] = True
                all_steps = app._completed_plan_steps + [
                    {"number": s.get("number", i + 1), "goal": s.get("goal", str(s)), "completed": False}
                    for i, s in enumerate(steps)
                ]
                app._plan_steps = all_steps
            else:
                app._completed_plan_steps = []
                app._plan_steps = [
                    {"number": s.get("number", i + 1), "goal": s.get("goal", str(s)), "completed": False}
                    for i, s in enumerate(steps)
                ]

            logger.debug(f"plan_ready: set app._plan_steps with {len(app._plan_steps)} steps, resetting _steps_initialized")
            self._steps_initialized = False

            status_bar.update_status(phase=Phase.AWAITING_APPROVAL)
            log.write(Text(goal, style="bold cyan"))

            if is_followup and app._completed_plan_steps:
                log.write(Text("  Completed:", style="dim green"))
                for s in app._completed_plan_steps:
                    log.write(Text(f"    \u2713 {s.get('number', '?')}. {s.get('goal', '')}", style="dim green"))
                log.write(Text("  New steps:", style="cyan"))

            for i, step in enumerate(steps):
                num = step.get("number", i + 1)
                log.write(Text(f"    {num}. {step.get('goal', step)}", style="dim"))

        elif event_type in ("proof_tree_start", "proof_start"):
            logger.debug(f"Handling proof_start event: {data}")
            _conclusion_fact = data.get("conclusion_fact", "")
            conclusion_desc = data.get("conclusion_description", "")
            status_bar.start_timer()
            status_bar.update_status(status_message="Resolving proof tree...", phase=Phase.EXECUTING)

            side_panel = self._get_side_panel()
            panel_content = self._get_panel_content()
            logger.debug(f"proof_start: switching to proof tree mode")
            panel_content.start_proof_tree(conclusion_desc)
            side_panel.add_class("visible")

        elif event_type == "dag_execution_start":
            # noinspection DuplicatedCode
            premises = data.get("premises", [])
            inferences = data.get("inferences", [])
            logger.debug(f"dag_execution_start: {len(premises)} premises, {len(inferences)} inferences")

            panel_content = self._get_panel_content()
            if not panel_content._proof_tree:
                logger.debug("dag_execution_start: no proof tree initialized, skipping")
                return

            import re
            name_to_id = {}
            id_to_deps = {}

            for p in premises:
                fact_id = p.get("id", "")
                name = p.get("name", fact_id)
                name_to_id[name] = fact_id
                id_to_deps[fact_id] = []

            for inf in inferences:
                fact_id = inf.get("id", "")
                name = inf.get("name", "") or fact_id
                op = inf.get("operation", "")
                name_to_id[name] = fact_id
                deps = re.findall(r'[PI]\d+', op)
                id_to_deps[fact_id] = deps
                logger.debug(f"dag_execution_start: {fact_id} deps={deps} from op={op[:50]}")

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

            if terminal:
                terminal_name = next((inf.get("name", inf.get("id")) for inf in inferences if inf.get("id") == terminal), terminal)
                terminal_deps = id_to_deps.get(terminal, [])
                panel_content.add_fact(f"{terminal}: {terminal_name}", "", parent_name="answer", dependencies=terminal_deps)
                logger.debug(f"dag_execution_start: added terminal {terminal} under root, deps={terminal_deps}")

            added = {"answer", terminal} if terminal else {"answer"}
            queue = [terminal] if terminal else []

            while queue:
                current_id = queue.pop(0)
                current_deps = id_to_deps.get(current_id, [])

                for dep_id in current_deps:
                    if dep_id not in added:
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
                        node_deps = id_to_deps.get(dep_id, [])
                        panel_content.add_fact(f"{dep_id}: {dep_name}", "", parent_name=parent_key, dependencies=node_deps)
                        logger.debug(f"dag_execution_start: added {dep_id} under {current_id}, deps={node_deps}")
                        added.add(dep_id)
                        queue.append(dep_id)

            logger.debug(f"dag_execution_start: pre-built tree with {len(added)} nodes")

            pre_resolved = data.get("pre_resolved", {})
            for fact_id, info in pre_resolved.items():
                for p in premises:
                    if p.get("id") == fact_id:
                        node_name = f"{fact_id}: {p.get('name', fact_id)}"
                        panel_content.update_resolved(
                            node_name,
                            info.get("value"),
                            source="user",
                            confidence=info.get("confidence", 0.95),
                        )
                        logger.debug(f"dag_execution_start: marked {node_name} as pre-resolved (user input)")
                        break

        elif event_type == "premise_resolving":
            fact_name = data.get("fact_name", "") or data.get("fact_id", "")
            description = data.get("description", "")
            logger.debug(f"premise_resolving: fact_name={fact_name}")
            status_bar.update_status(status_message=f"Resolving {fact_name[:40]}...")
            panel_content = self._get_panel_content()
            panel_content.update_resolving(fact_name, description)

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
            _db = data.get("database", "")
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
            fact_name = data.get("fact_name", "") or data.get("fact_id", "")
            value = data.get("value", "")
            source = data.get("source", "")
            confidence = data.get("confidence", 1.0)
            from_cache = source == "cache"
            logger.debug(f"premise_resolved: fact_name={fact_name}, value={str(value)[:30]}")
            panel_content = self._get_panel_content()
            panel_content.update_resolved(fact_name, value, source, confidence, from_cache=from_cache)

        elif event_type in ("inference_resolving", "inference_executing"):
            inference_id = data.get("inference_id", "") or data.get("fact_id", "")
            operation = data.get("operation", "") or data.get("name", "")
            display_name = f"{inference_id}: {operation}" if operation else inference_id
            logger.debug(f"inference_executing: {display_name}")
            status_bar.update_status(status_message=f"Computing {display_name[:40]}...")
            panel_content = self._get_panel_content()
            panel_content.update_resolving(display_name, operation)

        elif event_type in ("inference_resolved", "inference_complete"):
            inference_id = data.get("inference_id", "") or data.get("fact_id", "")
            inference_name = data.get("inference_name", "") or data.get("operation", "")
            result = data.get("result")
            if result is None:
                result = data.get("value", "")
            display_name = f"{inference_id}: {inference_name}" if inference_name else inference_id
            logger.debug(f"inference_complete: {display_name}, result={str(result)[:30]}")
            panel_content = self._get_panel_content()
            panel_content.update_resolved(display_name, result, source="derived", confidence=1.0)

        elif event_type == "proof_tree_complete":
            status_bar.update_status(status_message="Generating insights...", phase=Phase.EXECUTING)

        elif event_type == "step_start":
            step_num = event.step_number or 0
            goal = data.get("goal", "")
            self._current_step = step_num
            status_bar.update_status(
                status_message=f"Step {step_num}: {goal[:40]}...",
                phase=Phase.EXECUTING
            )

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

            panel_content.update_step_executing(step_num)

        elif event_type == "step_complete":
            step_num = event.step_number or 0
            result = data.get("result", data.get("stdout", ""))
            result_summary = str(result)[:100] if result else ""

            panel_content = self._get_panel_content()
            panel_content.update_step_complete(step_num, result_summary)

        elif event_type == "step_failed":
            step_num = event.step_number or 0
            error = data.get("error", "Unknown error")
            log.write(Text(f"  Step {step_num} failed: {error}", style="red"))

            panel_content = self._get_panel_content()
            panel_content.update_step_failed(step_num, error[:100] if error else "")

        elif event_type == "generating":
            step_num = event.step_number or 0
            attempt = data.get("attempt", 1)
            max_attempts = data.get("max_attempts", 10)
            is_retry = data.get("is_retry", False)
            retry_reason = data.get("retry_reason", "")

            if is_retry:
                reason_short = retry_reason[:30] + "..." if retry_reason and len(retry_reason) > 30 else retry_reason
                status_bar.update_status(
                    status_message=f"Step {step_num}: Regenerating code (attempt {attempt}/{max_attempts}) - {reason_short}" if reason_short
                    else f"Step {step_num}: Regenerating code (attempt {attempt}/{max_attempts})"
                )
                panel_content = self._get_panel_content()
                panel_content.update_step_executing(step_num, retry=True)
            else:
                goal = data.get("goal", "")
                if goal:
                    status_bar.update_status(status_message=f"Step {step_num}: {goal}. Generating code...")
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
                _next_attempt = data.get("next_attempt", attempt + 1)
                status_bar.update_status(
                    status_message=f"Step {step_num}: {error_type} on attempt {attempt}/{max_attempts}, retrying..."
                )
                log.write(Text(f"  Step {step_num} attempt {attempt} failed: {error_type}, will retry", style="yellow"))
            else:
                status_bar.update_status(status_message=f"Step {step_num}: {error_type} - max retries exceeded")
                log.write(Text(f"  Step {step_num} failed after {max_attempts} attempts: {error_type}", style="red"))

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
                fact_names = []
                for f in facts:
                    if isinstance(f, dict):
                        fact_names.append(f.get("name", f.get("key", str(f)[:20])))
                    elif hasattr(f, "name"):
                        fact_names.append(f.name)
                    else:
                        fact_names.append(str(f)[:20])
                facts_str = ", ".join(fact_names[:5])
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
            status_bar.update_status(status_message=None, phase=Phase.IDLE)

        elif event_type == "verification_error":
            error = data.get("error", "Unknown error")
            log.write(Text(f"  Verification error: {error[:80]}", style="dim red"))

        elif event_type == "complete":
            status_bar.update_status(status_message=None, phase=Phase.IDLE)
