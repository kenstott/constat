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

from constat.core.models import Plan, PlannerResponse, StepStatus, StepResult, TaskType
from constat.execution.mode import PlanApproval, PrimaryIntent, SubIntent
from constat.keywords import wants_brief_output
from constat.execution.scratchpad import Scratchpad
from constat.session._types import QuestionAnalysis, QuestionType, StepEvent, is_meta_question
from constat.storage.datastore import DataStore
from constat.storage.registry_datastore import RegistryAwareDataStore

logger = logging.getLogger(__name__)

_HIGH_COMPLEXITY_SUBS = {SubIntent.COMPARE, SubIntent.PREDICT}


def _assess_planning_complexity(
    analysis: QuestionAnalysis,
    turn_intent,
    query: str = "",
) -> str:
    """Return 'low', 'medium', or 'high' based on analysis and intent signals.

    Only compare/predict sub-intents trigger 'high' — they require multi-source
    reasoning. Scope refinements are query constraints, not complexity indicators.
    """
    sub = turn_intent.sub if turn_intent else None
    mods = getattr(analysis, "fact_modifications", []) or []
    # Keyword detection as fallback for unreliable LLM wants_brief
    keyword_brief = wants_brief_output(query) if query else False
    brief = analysis.wants_brief or keyword_brief
    logger.debug(
        f"[COMPLEXITY] sub={sub}, fact_modifications={len(mods)}, "
        f"wants_brief={analysis.wants_brief}, keyword_brief={keyword_brief}"
    )

    # High: only compare/predict sub-intents
    if sub in _HIGH_COMPLEXITY_SUBS:
        return "high"

    # Low: brief request with no fact modifications
    if brief and not mods:
        return "low"

    return "medium"


# noinspection PyUnresolvedReferences
class SolveMixin:

    def solve(self, problem: str, *, force_plan: bool = False) -> dict:
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
            # Match skills and agents dynamically based on query
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
                self._sync_available_agents_to_planner()
                return self.planner.plan(problem)
            except Exception as exc:
                logger.debug(f"[PARALLEL] Speculative planning failed: {exc}")
                return None

        # Determine which tasks to run
        # Fast tasks: intent, analysis, ambiguity, dynamic_context (~2-5s each)
        # Slow task: planning (~15-30s) — launched separately, awaited only when needed
        fast_tasks = {
            "intent": run_intent,
            "analysis": run_analysis,
            "dynamic_context": run_dynamic_context,
        }
        if self.session_config.ask_clarifications and self._clarification_callback:
            fast_tasks["ambiguity"] = run_ambiguity

        # Run fast tasks in parallel; launch planning as a non-blocking background future
        parallel_start = time.time()
        task_timings: dict[str, float] = {}
        results = {}
        # Use a persistent executor so the planning future survives the fast-task loop
        executor = ThreadPoolExecutor(max_workers=len(fast_tasks) + 1)
        planning_start = time.time()
        planning_future = executor.submit(run_planning)

        fast_starts = {name: time.time() for name in fast_tasks}
        fast_futures = {executor.submit(fn): name for name, fn in fast_tasks.items()}
        for future in as_completed(fast_futures):
            name = fast_futures[future]
            elapsed = time.time() - fast_starts[name]
            task_timings[name] = elapsed
            try:
                results[name] = future.result()
                logger.info(f"[PARALLEL] {name} completed in {elapsed:.2f}s")
            except Exception as e:
                logger.warning(f"[PARALLEL] {name} failed in {elapsed:.2f}s: {e}")
                results[name] = None

        fast_duration = time.time() - parallel_start
        timing_summary = ", ".join(f"{k}={v:.1f}s" for k, v in sorted(task_timings.items(), key=lambda x: -x[1]))
        logger.info(f"[PARALLEL] Fast tasks completed in {fast_duration:.2f}s ({timing_summary}), planning still running in background")

        turn_intent = results.get("intent")
        analysis = results.get("analysis") or QuestionAnalysis(
            question_type=QuestionType.DATA_ANALYSIS, extracted_facts=[]
        )
        clarification_request = results.get("ambiguity")
        dynamic_context = results.get("dynamic_context")
        # planning_future awaited later only if needed
        speculative_plan = None  # Will be resolved from planning_future when needed

        # Track whether the speculative plan is still usable
        planning_discarded = False

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
                    planning_discarded = True
                    planning_future.cancel()
                    logger.debug("[PARALLEL] Skills activated, discarding speculative plan")

            event_data = {
                "agent": dynamic_context.get("agent"),
                "skills": matched_skills,
                "agent_source": dynamic_context.get("agent_source"),
            }
            logger.info(f"[DYNAMIC_CONTEXT] Emitting event with data: agent={event_data.get('agent')}, skills={event_data.get('skills')}")
            self._emit_event(StepEvent(
                event_type="dynamic_context",
                step_number=0,
                data=event_data,
            ))

        # Route based on primary intent (may discard speculative plan)
        route_to_planning = False

        if not force_plan and turn_intent and turn_intent.primary == PrimaryIntent.QUERY:
            # QUERY intent - answer from knowledge or current context
            result = self._handle_query_intent(turn_intent, problem)
            if not result.get("_route_to_planning"):
                logger.debug("[PARALLEL] QUERY handled without planning (speculative plan discarded)")
                planning_future.cancel()
                executor.shutdown(wait=False)
                return result
            logger.info("[ROUTING] QUERY/LOOKUP found data sources, routing to planning")
            route_to_planning = True

        if not force_plan and turn_intent and turn_intent.primary == PrimaryIntent.CONTROL and not route_to_planning:
            logger.debug("[PARALLEL] CONTROL intent (speculative plan discarded)")
            planning_future.cancel()
            executor.shutdown(wait=False)
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
                planning_discarded = True
                planning_future.cancel()
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
        # Skip when force_plan=True (REDO) — user wants re-execution, not cached answers
        if analysis.cached_fact_answer and not force_plan:
            planning_future.cancel()
            executor.shutdown(wait=False)
            return {
                "success": True,
                "meta_response": True,
                "output": analysis.cached_fact_answer,
                "plan": None,
            }

        question_type = analysis.question_type

        if not force_plan and question_type == QuestionType.META_QUESTION:
            logger.debug("[PARALLEL] META_QUESTION (speculative plan discarded)")
            planning_future.cancel()
            executor.shutdown(wait=False)
            self._emit_event(StepEvent(
                event_type="progress",
                step_number=0,
                data={"message": "Reviewing available data sources..."}
            ))
            return self._answer_meta_question(problem)
        elif not force_plan and question_type == QuestionType.GENERAL_KNOWLEDGE:
            logger.debug("[PARALLEL] GENERAL_KNOWLEDGE (speculative plan discarded)")
            planning_future.cancel()
            executor.shutdown(wait=False)
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
                planning_discarded = True
                planning_future.cancel()
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
            planning_future.cancel()
            executor.shutdown(wait=False)
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
        # Use /reason command to generate auditable reasoning chains when needed

        # Assess planning complexity for model routing
        planning_complexity = _assess_planning_complexity(analysis, turn_intent, problem)
        logger.info(f"[PLANNING] Assessed complexity: {planning_complexity} (wants_brief={analysis.wants_brief})")

        # Generate plan with approval loop
        current_problem = problem
        display_problem = problem  # What to show in UI (just feedback on replan)
        replan_attempt = 0
        planner_response = None

        while replan_attempt <= self.session_config.max_replan_attempts:
            # Use speculative plan if available (from parallel execution), otherwise generate new plan
            if not planning_discarded and replan_attempt == 0:
                # Await the background planning future
                self._emit_event(StepEvent(
                    event_type="planning_start",
                    step_number=0,
                    data={"message": "Analyzing data sources and creating plan..."}
                ))
                try:
                    speculative_plan = planning_future.result()  # blocks until planning completes
                    elapsed = time.time() - planning_start
                    logger.info(f"[PARALLEL] planning completed in {elapsed:.2f}s")
                except Exception as e:
                    logger.warning(f"[PARALLEL] planning failed: {e}")
                    speculative_plan = None
                executor.shutdown(wait=False)

                if speculative_plan is not None and planning_complexity != "low":
                    logger.debug("[PARALLEL] Using speculative plan (saved ~1 LLM call)")
                    planner_response = speculative_plan
                    self.plan = planner_response.plan
                elif speculative_plan is not None and planning_complexity == "low":
                    logger.debug("[PARALLEL] Discarding speculative plan — re-planning with low-complexity directive")
                    self._sync_user_facts_to_planner()
                    self._sync_glossary_to_planner(current_problem)
                    self._sync_available_agents_to_planner()
                    planner_response = self.planner.plan(current_problem, complexity=planning_complexity)
                    self.plan = planner_response.plan
                else:
                    # Speculative plan failed, fall through to synchronous planning
                    self._sync_user_facts_to_planner()
                    self._sync_glossary_to_planner(current_problem)
                    self._sync_available_agents_to_planner()
                    planner_response = self.planner.plan(current_problem, complexity=planning_complexity)
                    self.plan = planner_response.plan
            else:
                # Emit planning start event
                self._emit_event(StepEvent(
                    event_type="planning_start",
                    step_number=0,
                    data={"message": "Analyzing data sources and creating plan..."}
                ))

                # Sync user facts, glossary, and agents to planner before generating plan
                self._sync_user_facts_to_planner()
                self._sync_glossary_to_planner(current_problem)
                self._sync_available_agents_to_planner()

                # Generate plan
                planner_response = self.planner.plan(current_problem, complexity=planning_complexity)
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

            # Request approval if required — auto-approve simple plans
            # Auto-approve: ≤2 steps (unless high complexity), or low complexity with ≤3 steps
            n_steps = len(self.plan.steps)
            simple_plan = (
                (n_steps <= 2 and planning_complexity != "high")
                or (planning_complexity == "low" and n_steps <= 3)
            )
            if simple_plan:
                logger.info(f"[PLANNING] Auto-approving simple plan ({n_steps} steps, complexity={planning_complexity})")
            if self.session_config.require_approval and not simple_plan:
                # Use display_problem for UI (just feedback on replan, full problem initially)
                approval = self._request_approval(display_problem, planner_response)

                if approval.decision == PlanApproval.REJECT:
                    # User rejected the plan — keep session open for retry
                    self.history.save_plan_data(
                        self.session_id,
                        approval_decision="rejected",
                        user_feedback=approval.reason,
                        iteration=replan_attempt,
                    )
                    # noinspection PyAttributeOutsideInit
                    self.plan = None
                    return {
                        "success": False,
                        "rejected": True,
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

        # Emit plan_ready only for auto-approved plans (the approval callback
        # already sends plan_ready for plans that went through user approval)
        if simple_plan:
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
                    "auto_approved": True,
                },
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

        cancelled = False
        execution_waves = self.plan.get_execution_order()
        logger.debug(f"[EXECUTION] execution_waves: {execution_waves}, total steps: {len(self.plan.steps)}")

        for wave_num, wave_step_nums in enumerate(execution_waves):
            # Skip waves containing only already-completed steps
            pending_in_wave = [n for n in wave_step_nums
                               if self.plan.get_step(n) and self.plan.get_step(n).status == StepStatus.PENDING]
            if not pending_in_wave:
                continue

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

            # Get steps for this wave (only pending ones)
            wave_steps = [self.plan.get_step(num) for num in pending_in_wave]
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

    def _replan_after_user_input(self, problem: str, current_plan: Plan, user_answer: str) -> list | None:
        """Replan remaining steps after a user_input step completes.

        Gathers completed step context and calls the planner to generate
        new remaining steps informed by the user's answer.

        Returns new steps (renumbered) or None if replanning fails.
        """
        from constat.prompts import load_prompt

        try:
            scratchpad_context = self.datastore.get_scratchpad_as_markdown()
            existing_tables = self.datastore.list_tables()
            existing_tables_list = (
                chr(10).join(f'  - `{t["name"]}`: {t.get("row_count", "?")} rows' for t in existing_tables)
                if existing_tables else '(none)'
            )

            last_completed = max(
                (s.number for s in current_plan.steps if s.status == StepStatus.COMPLETED),
                default=0,
            )
            next_step_number = last_completed + 1

            replan_prompt = load_prompt("replan_after_input.md").format(
                problem=problem,
                scratchpad_context=scratchpad_context,
                existing_tables_list=existing_tables_list,
                user_answer=user_answer,
                next_step_number=next_step_number,
            )

            self._sync_user_facts_to_planner()
            self._sync_glossary_to_planner(problem)
            planner_response = self.planner.plan(replan_prompt)
            new_steps = planner_response.plan.steps

            # Renumber steps to continue from last completed
            for i, step in enumerate(new_steps):
                step.number = next_step_number + i
                step.status = StepStatus.PENDING

            # Save new steps to datastore
            for step in new_steps:
                self.datastore.save_plan_step(
                    step_number=step.number,
                    goal=step.goal,
                    expected_inputs=step.expected_inputs,
                    expected_outputs=step.expected_outputs,
                    status="pending",
                )

            # Emit plan_updated (not plan_ready) so UI updates step list without
            # disrupting execution state or showing an approval dialog
            self._emit_event(StepEvent(
                event_type="plan_updated",
                step_number=0,
                data={
                    "steps": [
                        {"number": s.number, "goal": s.goal, "depends_on": s.depends_on, "role_id": s.role_id}
                        for s in current_plan.steps if s.status == StepStatus.COMPLETED
                    ] + [
                        {"number": s.number, "goal": s.goal, "depends_on": s.depends_on, "role_id": s.role_id}
                        for s in new_steps
                    ],
                    "reasoning": planner_response.reasoning,
                }
            ))

            logger.info(f"[REPLAN] Replanned after user input: {len(new_steps)} new steps from step {next_step_number}")
            return new_steps

        except Exception as e:
            logger.warning(f"[REPLAN] Replanning after user input failed: {e}")
            return None

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

        # Restore last_proof_result from persisted state
        state = self.history.load_state(session_id)
        if state and "last_proof_result" in state:
            # noinspection PyAttributeOutsideInit
            self.last_proof_result = state["last_proof_result"]

        return True
