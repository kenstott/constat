# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.
"""Solve mixin: solve(), resume()."""
from __future__ import annotations

import logging

from constat.core.models import PlannerResponse, StepStatus, StepResult
from constat.execution.mode import PlanApproval, PrimaryIntent
from constat.execution.scratchpad import Scratchpad
from constat.session._types import QuestionAnalysis, QuestionType, StepEvent, is_meta_question
from constat.storage.datastore import DataStore
from constat.storage.registry_datastore import RegistryAwareDataStore

logger = logging.getLogger(__name__)


# noinspection PyUnresolvedReferences
class SolveMixin:

    def solve(self, problem: str) -> dict:
        """
        Solve a problem with intent-based routing and multistep execution.

        Workflow:
        1. Classify intent (QUERY, PLAN_NEW, PLAN_CONTINUE, CONTROL)
        2. Route QUERY and CONTROL intents to handlers (no planning needed)
        3. For PLAN_NEW/PLAN_CONTINUE:
           a. Determine execution mode (exploratory vs proof)
           b. Generate plan
           c. Request user approval (if you require_approval is True)
           d. Execute steps in parallel waves
           e. Synthesize answer and generate follow-up suggestions

        Args:
            problem: Natural language problem to solve

        Returns:
            Dict with plan, results, and summary
        """
        # Fast path 1: Handle slash commands directly via command registry
        # These are explicit commands like /tables, /show, /help etc.
        # Must be checked FIRST since they're the most explicit user intent
        if problem.strip().startswith("/"):
            return self._handle_slash_command(problem.strip())

        # Fast path 2: Check for meta-questions (no intent classification needed)
        # These are questions about capabilities, available data, etc.
        if is_meta_question(problem):
            self._emit_event(StepEvent(
                event_type="progress",
                step_number=0,
                data={"message": "Reviewing available data sources..."}
            ))
            return self._answer_meta_question(problem)

        # PARALLEL OPTIMIZATION: Run intent, analysis, ambiguity, and planning ALL in parallel
        # Most queries need planning, so we speculatively start it while classifying intent.
        # If planning isn't needed (CONTROL intent, META question), we discard the speculative plan.
        self._emit_event(StepEvent(
            event_type="progress",
            step_number=0,
            data={"message": "Analyzing your question..."}
        ))

        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time

        def run_intent():
            return self._classify_turn_intent(problem)

        def run_analysis():
            return self._analyze_question(problem)

        def run_ambiguity():
            existing_tables = self.datastore.list_tables() if self.datastore else []
            return self._detect_ambiguity(problem, is_auditable_mode=True, session_tables=existing_tables)

        def run_dynamic_context():
            # Match skills and roles dynamically based on query
            try:
                return self.get_dynamic_context(problem)
            except Exception as exc:
                logger.debug(f"[PARALLEL] Dynamic context matching failed: {exc}")
                return None

        def run_planning():
            # Speculative planning - may be discarded if intent doesn't need it
            try:
                self._sync_user_facts_to_planner()
                self._sync_glossary_to_planner(problem)
                self._sync_available_roles_to_planner()
                return self.planner.plan(problem)
            except Exception as exc:
                logger.debug(f"[PARALLEL] Speculative planning failed: {exc}")
                return None

        # Determine which tasks to run
        tasks = {
            "intent": run_intent,
            "analysis": run_analysis,
            "dynamic_context": run_dynamic_context,
        }
        if self.session_config.ask_clarifications and self._clarification_callback:
            tasks["ambiguity"] = run_ambiguity
        # Always run speculative planning in parallel
        tasks["planning"] = run_planning

        # Run all tasks in parallel
        parallel_start = time.time()
        results = {}
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            futures = {executor.submit(fn): name for name, fn in tasks.items()}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result()
                except Exception as e:
                    logger.warning(f"[PARALLEL] Task {name} failed: {e}")
                    results[name] = None

        parallel_duration = time.time() - parallel_start
        logger.debug(f"[PARALLEL] All tasks completed in {parallel_duration:.2f}s")

        turn_intent = results.get("intent")
        analysis = results.get("analysis") or QuestionAnalysis(
            question_type=QuestionType.DATA_ANALYSIS, extracted_facts=[]
        )
        clarification_request = results.get("ambiguity")
        speculative_plan = results.get("planning")
        dynamic_context = results.get("dynamic_context")

        # Emit dynamic context event (role and skills matched for this query)
        logger.info(f"[DYNAMIC_CONTEXT] dynamic_context={dynamic_context}")
        if dynamic_context:
            # Activate matched skills so they flow into planner and codegen prompts
            matched_skills = dynamic_context.get("skills", [])
            if matched_skills:
                skill_names = [s["name"] for s in matched_skills]
                activated = self.skill_manager.set_active_skills(skill_names)
                logger.info(f"[DYNAMIC_CONTEXT] Activated skills: {activated}")
                if activated:
                    speculative_plan = None
                    logger.debug("[PARALLEL] Skills activated, discarding speculative plan")

            event_data = {
                "role": dynamic_context.get("role"),
                "skills": matched_skills,
                "role_source": dynamic_context.get("role_source"),
            }
            logger.info(f"[DYNAMIC_CONTEXT] Emitting event with data: role={event_data.get('role')}, skills={event_data.get('skills')}")
            self._emit_event(StepEvent(
                event_type="dynamic_context",
                step_number=0,
                data=event_data,
            ))

        # Route based on primary intent (may discard speculative plan)
        route_to_planning = False

        if turn_intent and turn_intent.primary == PrimaryIntent.QUERY:
            # QUERY intent - answer from knowledge or current context
            result = self._handle_query_intent(turn_intent, problem)
            if not result.get("_route_to_planning"):
                logger.debug("[PARALLEL] QUERY handled without planning (speculative plan discarded)")
                return result
            logger.info("[ROUTING] QUERY/LOOKUP found data sources, routing to planning")
            route_to_planning = True

        if turn_intent and turn_intent.primary == PrimaryIntent.CONTROL and not route_to_planning:
            logger.debug("[PARALLEL] CONTROL intent (speculative plan discarded)")
            return self._handle_control_intent(turn_intent, problem)

        # PLAN_NEW, PLAN_CONTINUE, or re-routed QUERY - continue with planning flow
        self._apply_phase_transition("plan_new")

        # Apply sub-intent enhancements for PLAN_NEW (COMPARE, PREDICT)
        enhanced_problem = problem
        if turn_intent and turn_intent.primary == PrimaryIntent.PLAN_NEW:
            enhancement = self._handle_plan_new_intent(turn_intent, problem)
            enhanced_problem = enhancement.get("enhanced_problem", problem)
            # If problem was enhanced, speculative plan may be stale - replan
            if enhanced_problem != problem:
                logger.debug("[PARALLEL] Problem enhanced, replanning...")
                speculative_plan = None
        problem = enhanced_problem

        # Emit facts if any were extracted
        if analysis.extracted_facts:
            self._emit_event(StepEvent(
                event_type="facts_extracted",
                step_number=0,
                data={
                    "facts": [f.to_dict() for f in analysis.extracted_facts],
                    "source": "question",
                }
            ))

        # Return cached fact answer if question was about a known fact
        if analysis.cached_fact_answer:
            return {
                "success": True,
                "meta_response": True,
                "output": analysis.cached_fact_answer,
                "plan": None,
            }

        question_type = analysis.question_type

        if question_type == QuestionType.META_QUESTION:
            logger.debug("[PARALLEL] META_QUESTION (speculative plan discarded)")
            self._emit_event(StepEvent(
                event_type="progress",
                step_number=0,
                data={"message": "Reviewing available data sources..."}
            ))
            return self._answer_meta_question(problem)
        elif question_type == QuestionType.GENERAL_KNOWLEDGE:
            logger.debug("[PARALLEL] GENERAL_KNOWLEDGE (speculative plan discarded)")
            self._emit_event(StepEvent(
                event_type="progress",
                step_number=0,
                data={"message": "Generating response..."}
            ))
            return self._answer_general_question(problem)

        # Check for clarification (clarification_request already computed in parallel above)
        if clarification_request:
            enhanced_problem = self._request_clarification(clarification_request)
            if enhanced_problem:
                problem = enhanced_problem
                # Re-analyze with clarified problem - speculative plan is stale
                logger.debug("[PARALLEL] Problem clarified, replanning...")
                speculative_plan = None
                analysis = self._analyze_question(problem)

        # Create session + datastore (idempotent — may already exist from skill execution)
        self._ensure_session_datastore(problem)

        # Initialize session state
        # noinspection PyAttributeOutsideInit
        self.scratchpad = Scratchpad(initial_context=f"Problem: {problem}")

        # Save problem statement to datastore (for UI restoration)
        self.datastore.set_session_meta("problem", problem)
        self.datastore.set_session_meta("status", "planning")

        # Log the initial query
        self.history.log_user_input(self.session_id, problem, "query")

        # Check for unclear/garbage input before processing
        if self._is_unclear_input(problem):
            return {
                "success": True,
                "meta_response": True,
                "output": "I'm not sure I understand that input. Could you rephrase your question?\n\n"
                          "You can ask me things like:\n"
                          "- Questions about your data (e.g., \"What's our total revenue?\")\n"
                          "- Verification requests (e.g., \"Prove that sales increased\")\n"
                          "- Explanations (e.g., \"How do you reason about problems?\")\n\n"
                          "Type /help for available commands.",
                "suggestions": [
                    "What data is available?",
                    "How can you help me?",
                ],
            }

        # All queries use exploratory mode by default
        # Use /prove command to generate auditable proofs when needed

        # Generate plan with approval loop
        current_problem = problem
        display_problem = problem  # What to show in UI (just feedback on replan)
        replan_attempt = 0
        planner_response = None

        while replan_attempt <= self.session_config.max_replan_attempts:
            # Use speculative plan if available (from parallel execution), otherwise generate new plan
            if speculative_plan is not None and replan_attempt == 0:
                logger.debug("[PARALLEL] Using speculative plan (saved ~1 LLM call)")
                planner_response = speculative_plan
                self.plan = planner_response.plan
                # Emit planning events for UI consistency
                self._emit_event(StepEvent(
                    event_type="planning_start",
                    step_number=0,
                    data={"message": "Plan ready..."}
                ))
            else:
                # Emit planning start event
                self._emit_event(StepEvent(
                    event_type="planning_start",
                    step_number=0,
                    data={"message": "Analyzing data sources and creating plan..."}
                ))

                # Sync user facts, glossary, and roles to planner before generating plan
                self._sync_user_facts_to_planner()
                self._sync_glossary_to_planner(current_problem)
                self._sync_available_roles_to_planner()

                # Generate plan
                planner_response = self.planner.plan(current_problem)
                self.plan = planner_response.plan

            # Emit planning complete event
            self._emit_event(StepEvent(
                event_type="planning_complete",
                step_number=0,
                data={"steps": len(self.plan.steps)}
            ))

            # Record plan to plan directory
            self.history.save_plan_data(
                self.session_id,
                raw_response=planner_response.raw_response or None,
                parsed_plan={
                    "steps": [
                        {
                            "number": s.number,
                            "goal": s.goal,
                            "inputs": s.expected_inputs,
                            "outputs": s.expected_outputs,
                            "depends_on": s.depends_on,
                            "task_type": s.task_type.value if s.task_type else None,
                            "role_id": s.role_id,
                        }
                        for s in self.plan.steps
                    ],
                },
                reasoning=planner_response.reasoning or None,
                iteration=replan_attempt,
            )

            # Request approval if required
            if self.session_config.require_approval:
                # Use display_problem for UI (just feedback on replan, full problem initially)
                approval = self._request_approval(display_problem, planner_response)

                if approval.decision == PlanApproval.REJECT:
                    # User rejected the plan
                    self.history.save_plan_data(
                        self.session_id,
                        approval_decision="rejected",
                        user_feedback=approval.reason,
                        iteration=replan_attempt,
                    )
                    self.datastore.set_session_meta("status", "rejected")
                    self.history.complete_session(self.session_id, status="rejected")
                    return {
                        "success": False,
                        "rejected": True,
                        "plan": self.plan,
                        "reason": approval.reason,
                        "message": "Plan was rejected by user.",
                    }

                elif approval.decision == PlanApproval.COMMAND:
                    # User entered a slash command - pass back to REPL
                    return {
                        "success": False,
                        "command": approval.command,
                        "message": "Slash command entered during approval.",
                    }

                elif approval.decision == PlanApproval.SUGGEST:
                    # User wants changes - check if we can skip replanning
                    suggestion_text = (approval.suggestion or "").strip()
                    has_edited_steps = bool(approval.edited_steps)
                    has_meaningful_feedback = bool(suggestion_text) and suggestion_text not in ("", "Edited plan")

                    # Record suggestion
                    self.history.save_plan_data(
                        self.session_id,
                        approval_decision="suggest",
                        user_feedback=suggestion_text or None,
                        edited_steps=approval.edited_steps,
                        iteration=replan_attempt,
                    )

                    # If user edited steps but provided no additional feedback, use edited plan directly
                    if has_edited_steps and not has_meaningful_feedback:
                        logger.info("[REPLAN] User edited steps with no feedback - using edited plan directly")

                        # Log the revision (the edited plan itself)
                        edited_summary = "; ".join(f"{s['number']}. {s['goal'][:50]}" for s in approval.edited_steps)
                        self.history.log_user_input(self.session_id, f"[Edited plan] {edited_summary}", "revision")

                        # Build Plan directly from edited steps
                        self.plan = self._build_plan_from_edited_steps(problem, approval.edited_steps)

                        # Create a synthetic planner_response for the plan_ready event
                        planner_response = PlannerResponse(
                            plan=self.plan,
                            reasoning="User-edited plan (approved without replanning)",
                        )

                        # Emit event for UI consistency
                        self._emit_event(StepEvent(
                            event_type="planning_complete",
                            step_number=0,
                            data={"steps": len(self.plan.steps), "edited": True}
                        ))

                        break  # Skip replanning, proceed to execution

                    # User has meaningful feedback - replan with feedback
                    replan_attempt += 1
                    if replan_attempt > self.session_config.max_replan_attempts:
                        self.datastore.set_session_meta("status", "max_replans_exceeded")
                        self.history.complete_session(self.session_id, status="failed")
                        return {
                            "success": False,
                            "plan": self.plan,
                            "error": f"Maximum replan attempts ({self.session_config.max_replan_attempts}) exceeded.",
                        }

                    # Log the revision
                    self.history.log_user_input(self.session_id, suggestion_text, "revision")

                    # Emit replan event
                    self._emit_event(StepEvent(
                        event_type="replanning",
                        step_number=0,
                        data={
                            "attempt": replan_attempt,
                            "feedback": suggestion_text,
                        }
                    ))

                    # Build replan prompt with original query + edited plan structure
                    # The edited plan is what the user approved/modified
                    if has_edited_steps:
                        edited_plan_text = "\n".join(
                            f"{step['number']}. {step['goal']}"
                            for step in approval.edited_steps
                        )
                        current_problem = f"""{problem}

**Requested plan structure (follow this exactly):**
{edited_plan_text}

**User notes:** {suggestion_text}"""
                    else:
                        # No edited steps provided - use original problem + feedback
                        current_problem = f"{problem}\n\n**User Revision (takes precedence):** {suggestion_text}"

                    display_problem = suggestion_text
                    continue  # Go back to planning

                # APPROVE - record and apply any edits/deletions, then proceed
                self.history.save_plan_data(
                    self.session_id,
                    approval_decision="approved",
                    edited_steps=approval.edited_steps if approval.edited_steps else None,
                    iteration=replan_attempt,
                )
                if approval.edited_steps:
                    # noinspection PyAttributeOutsideInit
                    self.plan = self._build_plan_from_edited_steps(problem, approval.edited_steps)
                elif approval.deleted_steps:
                    deleted_set = set(approval.deleted_steps)
                    self.plan.steps = [s for s in self.plan.steps if s.number not in deleted_set]
                    for i, step in enumerate(self.plan.steps, 1):
                        step.number = i
                break
            else:
                # No approval required - auto-approved
                self.history.save_plan_data(
                    self.session_id,
                    approval_decision="auto_approved",
                    iteration=replan_attempt,
                )
                break

        # Save plan to datastore (for UI restoration)
        self.datastore.set_session_meta("status", "executing")
        self.datastore.set_session_meta("mode", "exploratory")
        for step in self.plan.steps:
            self.datastore.save_plan_step(
                step_number=step.number,
                goal=step.goal,
                expected_inputs=step.expected_inputs,
                expected_outputs=step.expected_outputs,
                status="pending",
            )

        # Emit plan_ready event BEFORE execution starts
        self._emit_event(StepEvent(
            event_type="plan_ready",
            step_number=0,
            data={
                "steps": [
                    {"number": s.number, "goal": s.goal, "depends_on": s.depends_on, "role_id": s.role_id}
                    for s in self.plan.steps
                ],
                "reasoning": planner_response.reasoning,
                "is_followup": False,
            }
        ))

        # Materialize facts table before execution starts
        self._materialize_facts_table()

        # Execute steps in parallel waves based on dependencies
        # Phase 4: Reset cancellation state before starting execution
        self.reset_cancellation()
        all_results = []

        # Debug: Log full plan structure before computing waves
        logger.debug(f"[EXECUTION] Plan has {len(self.plan.steps)} steps:")
        for step in self.plan.steps:
            logger.debug(f"[EXECUTION]   Step {step.number}: status={step.status.value}, "
                         f"depends_on={step.depends_on}, goal='{step.goal[:60]}...'")

        execution_waves = self.plan.get_execution_order()
        logger.debug(f"[EXECUTION] execution_waves: {execution_waves}, total steps: {len(self.plan.steps)}")
        cancelled = False

        for wave_num, wave_step_nums in enumerate(execution_waves):
            logger.debug(f"[EXECUTION] Starting wave {wave_num + 1}, steps: {wave_step_nums}")
            # Phase 4: Check for cancellation before starting each wave
            if self.is_cancelled():
                cancelled = True
                self._emit_event(StepEvent(
                    event_type="execution_cancelled",
                    step_number=0,
                    data={
                        "message": "Execution cancelled between waves",
                        "wave": wave_num,
                        "completed_steps": len(all_results),
                    }
                ))
                break

            # Get steps for this wave
            wave_steps = [self.plan.get_step(num) for num in wave_step_nums]
            wave_steps = [s for s in wave_steps if s is not None]

            # Execute all steps in this wave in parallel
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(wave_steps)) as executor:
                # Submit all steps in wave
                future_to_step = {}
                for step in wave_steps:
                    # Phase 4: Check for cancellation before starting each step
                    if self.is_cancelled():
                        cancelled = True
                        break

                    step.status = StepStatus.RUNNING
                    self.datastore.update_plan_step(step.number, status="running")
                    self._emit_event(StepEvent(
                        event_type="wave_step_start",
                        step_number=step.number,
                        data={"wave": wave_num + 1, "goal": step.goal}
                    ))
                    future = executor.submit(self._execute_step, step)
                    future_to_step[future] = step

                if cancelled:
                    # Cancel any pending futures
                    for future in future_to_step:
                        future.cancel()
                    break

                # Collect results as they complete
                wave_failed = False
                for future in concurrent.futures.as_completed(future_to_step):
                    # Phase 4: Check for cancellation while collecting results
                    # Note: We still collect completed results even if cancelled
                    step = future_to_step[future]
                    try:
                        result = future.result()
                    except concurrent.futures.CancelledError:
                        # Step was cancelled before it started
                        result = StepResult(
                            success=False,
                            stdout="",
                            error="Step cancelled",
                            attempts=0,
                        )
                        cancelled = True
                    except Exception as e:
                        logger.error(f"[EXECUTION] Step {step.number} raised unhandled exception: {e}", exc_info=True)
                        result = StepResult(
                            success=False,
                            stdout="",
                            error=str(e),
                            attempts=1,
                        )

                    logger.debug(f"[EXECUTION] Step {step.number} result: success={result.success}, attempts={result.attempts}, error={result.error[:200] if result.error else 'none'}")
                    if result.success:
                        self.plan.mark_step_completed(step.number, result)
                        self.scratchpad.add_step_result(
                            step_number=step.number,
                            goal=step.goal,
                            result=result.stdout,
                            tables_created=result.tables_created,
                        )
                        if self.datastore:
                            self.datastore.add_scratchpad_entry(
                                step_number=step.number,
                                goal=step.goal,
                                narrative=result.stdout,
                                tables_created=result.tables_created,
                                code=result.code,
                            )
                            self.datastore.update_plan_step(
                                step.number,
                                status="completed",
                                code=step.code,
                                attempts=result.attempts,
                                duration_ms=result.duration_ms,
                            )
                        all_results.append(result)
                    else:
                        self.plan.mark_step_failed(step.number, result)
                        if self.datastore:
                            self.datastore.update_plan_step(
                                step.number,
                                status="failed" if not cancelled else "cancelled",
                                code=step.code,
                                error=result.error,
                                attempts=result.attempts,
                                duration_ms=result.duration_ms,
                            )
                        if not cancelled:  # Only mark as failed if not cancelled
                            wave_failed = True
                        all_results.append(result)

                if cancelled:
                    logger.debug(f"[EXECUTION] Wave {wave_num + 1} cancelled, breaking out of loop")
                    break

                logger.debug(f"[EXECUTION] Wave {wave_num + 1} completed: "
                             f"wave_failed={wave_failed}, all_results={len(all_results)}, "
                             f"completed_steps={self.plan.completed_steps}")

                # If any step in wave failed, stop execution
                if wave_failed:
                    self.datastore.set_session_meta("status", "failed")
                    failed_result = next(r for r in all_results if not r.success)
                    self.history.record_query(
                        session_id=self.session_id,
                        question=problem,
                        success=False,
                        attempts=failed_result.attempts,
                        duration_ms=failed_result.duration_ms,
                        error=failed_result.error,
                    )
                    self.history.complete_session(self.session_id, status="failed")
                    return {
                        "success": False,
                        "plan": self.plan,
                        "error": failed_result.error,
                        "completed_steps": self.plan.completed_steps,
                    }

        # Log exit reason
        logger.debug(f"[EXECUTION] Wave loop finished: cancelled={cancelled}, "
                     f"all_results={len(all_results)}, completed_steps={self.plan.completed_steps}")

        # Phase 4: Handle cancellation - return with completed results preserved
        if cancelled:
            self.datastore.set_session_meta("status", "cancelled")
            self.history.complete_session(self.session_id, status="cancelled")

            # Combine output from completed steps
            completed_output = ""
            if all_results:
                completed_output = "\n\n".join([
                    f"Step {i+1}: {self.plan.steps[i].goal}\n{r.stdout}"
                    for i, r in enumerate(all_results) if r.success
                ])

            # Process any queued intents
            queued_results = self.process_queued_intents()

            return {
                "success": False,
                "cancelled": True,
                "plan": self.plan,
                "completed_steps": self.plan.completed_steps,
                "partial_output": completed_output,
                "queued_intent_results": queued_results,
                "message": f"Execution cancelled. {len(self.plan.completed_steps)} step(s) completed.",
            }

        # Record successful completion
        total_duration = sum(r.duration_ms for r in all_results)
        total_attempts = sum(r.attempts for r in all_results)

        # Combine all step outputs
        combined_output = "\n\n".join([
            f"Step {i+1}: {self.plan.steps[i].goal}\n{r.stdout}"
            for i, r in enumerate(all_results)
        ])

        # Auto-publish final step artifacts (they appear in artifacts panel)
        # Find the last step that actually created tables (may not be the last step if it's just a summary)
        if all_results and self.registry:
            # Collect tables from last 2 steps that created tables (final outputs often span multiple steps)
            # noinspection DuplicatedCode
            final_tables = []
            steps_with_tables = 0
            for result in reversed(all_results):
                if result.success and result.tables_created:
                    final_tables.extend(result.tables_created)
                    steps_with_tables += 1
                    if steps_with_tables >= 2:
                        break

            # Filter to most important tables (avoid publishing every intermediate table)
            # Prioritize: tables with "final", "report", "result", "recommendation" in name
            important_keywords = ("final", "report", "result", "recommendation", "summary", "output")
            important_tables = [t for t in final_tables if any(kw in t.lower() for kw in important_keywords)]

            # If no important tables found, use tables from last step only (max 5)
            if not important_tables:
                for result in reversed(all_results):
                    if result.success and result.tables_created:
                        important_tables = result.tables_created[:5]
                        break

            # Limit to max 8 published tables to avoid clutter
            tables_to_publish = important_tables[:8] if important_tables else []

            # Also find markdown artifacts from final step (highest priority for View Result)
            final_artifacts = []
            if self.datastore:
                all_artifacts = self.datastore.list_artifacts()
                last_step_num = len(self.plan.steps) if self.plan else 0
                # Get markdown artifacts from last 2 steps
                final_artifacts = [
                    a["name"] for a in all_artifacts
                    if a.get("step_number", 0) >= last_step_num - 1
                    and a.get("type") in ("markdown", "md", "html")
                ]

            if tables_to_publish or final_artifacts:
                self.registry.mark_final_step(
                    user_id=self.user_id,
                    session_id=self.session_id,
                    table_names=tables_to_publish if tables_to_publish else None,
                    artifact_names=final_artifacts if final_artifacts else None,
                )
                logger.debug(f"Auto-published final step: tables={tables_to_publish}, artifacts={final_artifacts}")

        # Note: Facts created during role-scoped steps are tagged with role_id
        # for provenance but remain globally accessible. No promotion needed.

        # Emit raw results first (so user can see them immediately)
        self._emit_event(StepEvent(
            event_type="raw_results_ready",
            step_number=0,
            data={"output": combined_output}
        ))

        # Check for created artifacts (to mention in synthesis)
        from constat.visualization.output import peek_pending_outputs
        pending_artifacts = peek_pending_outputs()

        # Check if insights are enabled (config or per-query brief detection via LLM)
        skip_insights = not self.session_config.enable_insights or analysis.wants_brief
        # noinspection DuplicatedCode
        suggestions = []  # Initialize for brief mode (no suggestions)

        if skip_insights:
            # Use raw output as final answer
            final_answer = combined_output
            self._emit_event(StepEvent(
                event_type="answer_ready",
                step_number=0,
                data={"answer": final_answer, "brief": True}
            ))
        else:
            # Synthesize final answer from step results
            self._emit_event(StepEvent(
                event_type="synthesizing",
                step_number=0,
                data={"message": "Synthesizing final answer..."}
            ))

            final_answer = self._synthesize_answer(problem, combined_output, pending_artifacts)

            self._emit_event(StepEvent(
                event_type="answer_ready",
                step_number=0,
                data={"answer": final_answer}
            ))

            # Run facts extraction and suggestions generation in parallel
            tables = self.datastore.list_tables() if self.datastore else []
            response_facts, suggestions = self._run_post_synthesis_parallel(
                problem, final_answer, tables
            )

            if response_facts:
                self._emit_event(StepEvent(
                    event_type="facts_extracted",
                    step_number=0,
                    data={
                        "facts": [f.to_dict() for f in response_facts],
                        "source": "response",
                    }
                ))

            if suggestions:
                self._emit_event(StepEvent(
                    event_type="suggestions_ready",
                    step_number=0,
                    data={"suggestions": suggestions}
                ))

        self.history.record_query(
            session_id=self.session_id,
            question=problem,
            success=True,
            attempts=total_attempts,
            duration_ms=total_duration,
            answer=final_answer,
        )
        self.history.complete_session(self.session_id, status="completed")

        # Mark session as completed in datastore (for UI restoration)
        if self.datastore:
            self.datastore.set_session_meta("status", "completed")

        # Auto-compact if context is too large
        self._auto_compact_if_needed()

        # Ensure execution history is available as a queryable table
        if self.datastore:
            self.datastore.ensure_execution_history_table()

        return {
            "success": True,
            "plan": self.plan,
            "results": all_results,
            "output": combined_output,
            "final_answer": final_answer,
            "suggestions": suggestions,
            "scratchpad": self.scratchpad.to_markdown(),
            "datastore_tables": self.datastore.list_tables() if self.datastore else [],
            "datastore_path": str(self.datastore.db_path) if self.datastore and self.datastore.db_path else None,
        }

    def resume(self, session_id: str) -> bool:
        """
        Resume a previous session, loading its datastore and context.

        Args:
            session_id: The session ID to resume

        Returns:
            True if successfully resumed, False if session not found
        """
        # Check if session exists
        session_detail = self.history.get_session(session_id)
        if not session_detail:
            return False

        # noinspection PyAttributeOutsideInit
        self.session_id = session_id

        # Load the datastore (contains tables, state, scratchpad, artifacts)
        session_dir = self.history._session_dir(session_id)
        datastore_path = session_dir / "datastore.duckdb"
        tables_dir = session_dir / "tables"

        # Create underlying datastore
        if datastore_path.exists():
            underlying_datastore = DataStore(db_path=datastore_path)
        else:
            # No datastore file - create empty one
            underlying_datastore = DataStore(db_path=datastore_path)

        # Wrap with registry-aware datastore
        # noinspection PyAttributeOutsideInit
        self.datastore = RegistryAwareDataStore(
            datastore=underlying_datastore,
            registry=self.registry,
            user_id=self.user_id,
            session_id=session_id,
            tables_dir=tables_dir,
        )

        if datastore_path.exists():
            # Rebuild scratchpad from datastore
            scratchpad_entries = self.datastore.get_scratchpad()
            if scratchpad_entries:
                # Get the original problem from the first query
                if session_detail.queries:
                    initial_context = f"Problem: {session_detail.queries[0].question}"
                else:
                    initial_context = ""
                # noinspection PyAttributeOutsideInit
                self.scratchpad = Scratchpad(initial_context=initial_context)

                # Add each step result
                for entry in scratchpad_entries:
                    self.scratchpad.add_step_result(
                        step_number=entry["step_number"],
                        goal=entry["goal"],
                        result=entry["narrative"],
                        tables_created=entry.get("tables_created", []),
                    )

        # Update fact resolver's datastore reference (for storing large facts as tables)
        self.fact_resolver._datastore = self.datastore

        return True
