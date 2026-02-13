# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""SessionFeedbackHandler — bridges Session events to FeedbackDisplay."""

from __future__ import annotations

import re
import time

from constat.execution.mode import Phase
from constat.repl.feedback._display import FeedbackDisplay


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

    def _update_status_for_event(self, event_type: str, data: dict) -> None:
        """Update the status line based on event type."""
        if event_type == "planning_start":
            self.display.update_status_line(phase=Phase.PLANNING)

        elif event_type == "plan_ready":
            steps = data.get("steps", [])
            problem = data.get("problem", "")
            self.display.update_status_line(
                phase=Phase.AWAITING_APPROVAL,
                plan_name=problem[:50] if problem else None,
                step_total=len(steps),
            )

        elif event_type in ("proof_start", "dag_execution_start", "step_start"):
            if event_type in ("proof_start", "dag_execution_start"):
                self.display.update_status_line(phase=Phase.EXECUTING)
            # Update step progress
            step_current = data.get("step_number", 0) or data.get("current", 0)
            step_total = data.get("total", 0)
            step_desc = data.get("goal", "") or data.get("description", "")
            if step_current > 0:
                self.display.update_status_line(
                    step_current=step_current,
                    step_total=step_total,
                    step_description=step_desc,
                )

        elif event_type == "step_complete":
            step_current = data.get("step_number", 0)
            if step_current > 0:
                self.display.update_status_line(step_current=step_current)

        elif event_type in ("verification_complete", "execution_complete", "knowledge_complete"):
            self.display.reset_status_line()

        elif event_type in ("step_failed", "verification_error"):
            error = data.get("error", "")
            self.display.update_status_line(
                phase=Phase.FAILED,
                error_message=error[:50] if error else None,
            )

    def handle_event(self, event) -> None:
        """Handle a StepEvent from Session."""
        event_type = event.event_type
        step_number = event.step_number
        data = event.data

        # Update status line based on event type
        self._update_status_for_event(event_type, data)

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
                    # Track table names from resolution_summary like "(hr.employees) 15 rows"
                    if resolution_summary and resolution_summary.startswith("("):
                        table_match = re.match(r'\(([^)]+)\)', resolution_summary)
                        if table_match:
                            table_name = table_match.group(1)
                            if table_name not in self.display._resolved_tables:
                                self.display._resolved_tables.add(table_name)
                                self.display._status_bar._tables_count = len(self.display._resolved_tables)
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
            # Stop Live display and show status bar with synthesizing message
            self.display.stop()
            self.display.stop_proof_tree()
            # Set the status message and print static status bar line
            message = data.get('message', 'Synthesizing...')
            self.display._status_bar.set_status_message(message)
            self.display.console.print()
            self.display.console.print(self.display._build_status_bar_line())

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
