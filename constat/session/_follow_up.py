# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Follow-up mixin: follow_up(), replay, prove, _solve_knowledge."""

from __future__ import annotations

import json
import logging
import time

from constat.core.models import StepResult, StepStatus, TaskType
from constat.execution.mode import PlanApproval
from constat.prompts import load_prompt
from constat.session._types import StepEvent

logger = logging.getLogger(__name__)


# noinspection PyUnresolvedReferences
class FollowUpMixin:

    def classify_follow_up_intent(self, user_text: str) -> dict:
        """
        Classify the intent of a follow-up message.

        This helps determine how to handle user input that could be:
        - Providing facts (e.g., "There were 1 million people")
        - Revising the request (e.g., "Use $50k threshold instead")
        - Making a new request (e.g., "Show me sales by region")
        - A combination of the above

        Args:
            user_text: The user's follow-up message

        Returns:
            Dict with:
                - intent: PRIMARY intent (PROVIDE_FACTS, REVISE, NEW_REQUEST, MIXED)
                - facts: List of any facts detected
                - revision: Description of any revision detected
                - new_request: The new request if detected
        """
        # Check for unresolved facts
        unresolved = self.fact_resolver.get_unresolved_facts()
        unresolved_names = [f.name for f in unresolved]

        prompt = f"""Analyze this user follow-up message and classify its intent.

User message: "{user_text}"

Context:
- There are {len(unresolved)} unresolved facts: {unresolved_names if unresolved else 'none'}

Classify the PRIMARY intent as one of:
- PROVIDE_FACTS: User is providing factual information (numbers, values, definitions)
- REVISE: User wants to modify/refine the previous request
- NEW_REQUEST: User is making an unrelated new request
- MIXED: Combination of the above

Also extract any facts, revisions, or new requests detected.

Respond in this exact format:
INTENT: <one of PROVIDE_FACTS, REVISE, NEW_REQUEST, MIXED>
FACTS: <comma-separated list of fact=value pairs, or NONE>
REVISION: <description of revision, or NONE>
NEW_REQUEST: <the new request, or NONE>
"""

        try:
            response = self.llm.generate(
                system="You are an intent classifier. Analyze user messages precisely.",
                user_message=prompt,
                max_tokens=self.router.max_output_tokens,
            )

            result = {
                "intent": "NEW_REQUEST",  # Default
                "facts": [],
                "revision": None,
                "new_request": None,
            }

            for line in response.split("\n"):
                line = line.strip()
                if line.startswith("INTENT:"):
                    intent = line.split(":", 1)[1].strip().upper()
                    if intent in ("PROVIDE_FACTS", "REVISE", "NEW_REQUEST", "MIXED"):
                        result["intent"] = intent
                elif line.startswith("FACTS:"):
                    facts_str = line.split(":", 1)[1].strip()
                    if facts_str != "NONE":
                        result["facts"] = [f.strip() for f in facts_str.split(",")]
                elif line.startswith("REVISION:"):
                    rev = line.split(":", 1)[1].strip()
                    if rev != "NONE":
                        result["revision"] = rev
                elif line.startswith("NEW_REQUEST:"):
                    req = line.split(":", 1)[1].strip()
                    if req != "NONE":
                        result["new_request"] = req

            return result

        except Exception as e:
            # Default to treating as new request
            logger.debug(f"Conversational intent classification failed: {e}")
            return {
                "intent": "NEW_REQUEST",
                "facts": [],
                "revision": None,
                "new_request": user_text,
            }

    def _classify_premise_response(self, user_response: str, fact_name: str, question: str) -> dict:
        """
        Classify a user's response to a premise clarification question.

        Determines if the response is:
        - VALUE: An actual value/answer (e.g., "5% for rating 4, 3% for rating 3")
        - STEER: Guidance on where/how to find the answer (e.g., "Look in the HR policy document")

        Args:
            user_response: The user's response text
            fact_name: The name of the fact being resolved
            question: The original clarification question asked

        Returns:
            Dict with:
                - type: "VALUE" or "STEER"
                - value: The parsed value if VALUE type
                - steer: The guidance/direction if STEER type
        """
        prompt = f"""Classify this user response to a data clarification question.

Question asked: "{question}"
Fact needed: {fact_name}
User response: "{user_response}"

Is this response:
A) VALUE - A direct answer providing the actual data/value/information requested
   Examples: "5%", "use 10000 as threshold", "rating 5 gets 10%, rating 4 gets 6%"

B) STEER - Guidance on WHERE or HOW to find the answer (not the answer itself)
   Examples: "Look in the HR policy", "Check the business_rules document",
   "Use the performance review guidelines", "It should be in the config"

Respond in this exact format:
TYPE: <VALUE or STEER>
CONTENT: <the value if VALUE, or the guidance/direction if STEER>
"""

        try:
            response = self.llm.generate(
                system="You classify user responses precisely. Distinguish between direct answers and guidance about where to find answers.",
                user_message=prompt,
                max_tokens=self.router.max_output_tokens,
            )

            result = {"type": "VALUE", "value": user_response, "steer": None}

            for line in response.split("\n"):
                line = line.strip()
                if line.startswith("TYPE:"):
                    resp_type = line.split(":", 1)[1].strip().upper()
                    if resp_type == "STEER":
                        result["type"] = "STEER"
                elif line.startswith("CONTENT:"):
                    content = line.split(":", 1)[1].strip()
                    if result["type"] == "VALUE":
                        result["value"] = content
                    else:
                        result["steer"] = content

            return result

        except Exception as e:
            # Default to treating as value
            logger.debug(f"Premise response classification failed: {e}")
            return {"type": "VALUE", "value": user_response, "steer": None}

    def follow_up(self, question: str, auto_classify: bool = True) -> dict:
        """
        Ask a follow-up question that builds on the current session's context.

        The follow-up has access to all tables and state from previous steps.
        If there are unresolved facts, the system will first try to extract
        facts from the user's message.

        Automatically detects if the question suggests auditable mode (verify,
        validate, etc.) and uses the fact resolver for formal verification.

        Args:
            question: The follow-up question
            auto_classify: If True, classify intent and handle accordingly

        Returns:
            Dict with plan, results, and summary (same format as solve())
        """
        if not self.session_id:
            raise ValueError("No active session. Call solve() or resume() first.")

        if not self.datastore:
            raise ValueError("No datastore available. Session may not have been properly initialized.")

        # Fast path: Handle slash commands directly via command registry
        if question.strip().startswith("/"):
            return self._handle_slash_command(question.strip())

        # Log the follow-up query
        self.history.log_user_input(self.session_id, question, "followup")

        # Detect corrections/hints and save as learnings
        from constat.api.detection.correction import detect_nl_correction
        nl_correction = detect_nl_correction(question)
        if nl_correction.detected:
            self._save_correction_as_learning(question)

        # Fast path: check if this is a simple "show me X" request for existing data
        # This avoids expensive LLM classification for simple data lookups
        show_result = self._try_show_existing_data(question)
        if show_result:
            return show_result

        # Get previous problem for follow-up context
        previous_problem = self.datastore.get_session_meta("problem")

        # Store follow-up question for /prove command
        follow_ups_json = self.datastore.get_session_meta("follow_ups")
        try:
            follow_ups = json.loads(follow_ups_json) if follow_ups_json else []
        except json.JSONDecodeError:
            follow_ups = []
        follow_ups.append(question)
        self.datastore.set_session_meta("follow_ups", json.dumps(follow_ups))
        logger.debug(f"[follow_up] Stored follow-up #{len(follow_ups)}: {question[:50]}...")

        # Use LLM to analyze the question and detect intents (with follow-up context)
        # This single call determines intent, facts, AND execution mode
        analysis = self._analyze_question(question, previous_problem=previous_problem)

        # Check for validation constraint addition (before REDO check)
        intent_names = [i.intent for i in analysis.intents]
        _validation_keywords = ("validate", "verify", "ensure", "assert", "check that", "confirm that")
        session_mode = self.datastore.get_session_meta("mode") if self.datastore else "exploratory"
        if session_mode == "auditable" and any(kw in question.lower() for kw in _validation_keywords):
            # Extract validation from follow-up and re-run proof with it
            new_validations = self._extract_user_validations(question, [])
            if new_validations:
                for v in new_validations:
                    self.add_user_validation(v['label'], v['sql'])
                return self.prove_conversation(guidance=f"Re-run with added validation: {question}")

        # Check for REDO intent — apply fact modifications and re-execute
        if "REDO" in intent_names:
            # Apply any fact modifications first
            for mod in analysis.fact_modifications:
                self.fact_resolver.add_user_fact(
                    fact_name=mod["fact_name"],
                    value=mod["new_value"],
                    reasoning=f"User correction: {question}",
                )
                logger.debug(f"[follow_up] Applied fact modification: {mod['fact_name']}={mod['new_value']}")
            self.fact_resolver.add_user_facts_from_text(question)

            # Determine mode: redo stays in the session's current mode
            session_mode = self.datastore.get_session_meta("mode") if self.datastore else "exploratory"
            use_proof = session_mode == "auditable"
            logger.info(f"[follow_up] REDO intent detected, mode={session_mode}, recommended={analysis.recommended_mode}, use_proof={use_proof}")

            if use_proof:
                return self.prove_conversation(guidance=question)
            else:
                # Re-run exploratory: get original problem, re-solve with updated facts
                original_problem = self.datastore.get_session_meta("problem") if self.datastore else question
                return self.solve(original_problem)

        # Check for ambiguity and request clarification if needed
        if self.session_config.ask_clarifications and self._clarification_callback:
            existing_tables = self.datastore.list_tables()
            clarification_request = self._detect_ambiguity(question, session_tables=existing_tables)
            if clarification_request:
                enhanced_question = self._request_clarification(clarification_request)
                if enhanced_question:
                    question = enhanced_question
                    # Re-analyze with clarified question
                    logger.debug("[follow_up] Question clarified, re-analyzing...")
                    _analysis = self._analyze_question(question, previous_problem=previous_problem)

        # All follow-ups use exploratory mode (planning + execution)
        # Use /prove command to generate auditable proofs when needed
        # Check for unresolved facts and try to extract facts from user message
        unresolved = self.fact_resolver.get_unresolved_facts()

        if auto_classify and (unresolved or "=" in question or any(c.isdigit() for c in question)):
            # Try to extract facts from the message
            extracted_facts = self.fact_resolver.add_user_facts_from_text(question)

            if extracted_facts:
                # Clear unresolved status to allow re-resolution
                self.fact_resolver.clear_unresolved()

        # Get context from previous work
        existing_state = self.datastore.get_all_state()
        scratchpad_context = self.datastore.get_scratchpad_as_markdown()

        # Ensure execution history is available as a queryable table
        # This includes step goals, code, and outputs
        self.datastore.ensure_execution_history_table()
        existing_tables = self.datastore.list_tables()  # Refresh after adding history table

        # Calculate next step number
        existing_scratchpad = self.datastore.get_scratchpad()
        next_step_number = max((e["step_number"] for e in existing_scratchpad), default=0) + 1

        # Generate a plan for the follow-up, providing context
        existing_tables_list = (
            chr(10).join(f'  - `{t["name"]}`: {t.get("row_count", "?")} rows (step {t.get("step_number", "?")})' for t in existing_tables)
            if existing_tables else '(none)'
        )
        first_table_name = existing_tables[0]["name"] if existing_tables else "final_answer"
        context_prompt = load_prompt("followup_context.md").format(
            scratchpad_context=scratchpad_context,
            existing_tables_list=existing_tables_list,
            existing_state=existing_state if existing_state else '(none)',
            first_table_name=first_table_name,
            question=question,
        )
        # Emit planning start event
        self._emit_event(StepEvent(
            event_type="planning_start",
            step_number=0,
            data={"message": "Planning follow-up analysis..."}
        ))

        # Sync user facts, glossary, and roles to planner before generating plan
        self._sync_user_facts_to_planner()
        self._sync_glossary_to_planner(question)
        self._sync_available_roles_to_planner()

        # Generate plan for follow-up
        planner_response = self.planner.plan(context_prompt)
        follow_up_plan = planner_response.plan

        # Validate: ensure "enhance" plans include a step that updates the source table
        follow_up_plan = self._ensure_enhance_updates_source(
            question, follow_up_plan, existing_tables,
        )

        # Emit planning complete event
        self._emit_event(StepEvent(
            event_type="planning_complete",
            step_number=0,
            data={"steps": len(follow_up_plan.steps)}
        ))

        # Renumber steps to continue from where we left off
        # noinspection DuplicatedCode
        for i, step in enumerate(follow_up_plan.steps):
            step.number = next_step_number + i

        # Emit plan_ready event for display
        self._emit_event(StepEvent(
            event_type="plan_ready",
            step_number=0,
            data={
                "steps": [
                    {"number": s.number, "goal": s.goal, "depends_on": s.depends_on, "role_id": s.role_id}
                    for s in follow_up_plan.steps
                ],
                "reasoning": planner_response.reasoning,
                "is_followup": True,
            }
        ))

        # Request approval if required (same as solve())
        if self.session_config.require_approval:
            approval = self._request_approval(question, planner_response)

            if approval.decision == PlanApproval.REJECT:
                return {
                    "success": False,
                    "rejected": True,
                    "plan": follow_up_plan,
                    "reason": approval.reason,
                    "message": "Follow-up plan was rejected by user.",
                }

            elif approval.decision == PlanApproval.COMMAND:
                # User entered a slash command - pass back to REPL
                return {
                    "success": False,
                    "command": approval.command,
                    "message": "Slash command entered during approval.",
                }

            elif approval.decision == PlanApproval.SUGGEST:
                # Check if user edited steps but provided no meaningful feedback
                suggestion_text = (approval.suggestion or "").strip()
                has_edited_steps = bool(approval.edited_steps)
                has_meaningful_feedback = bool(suggestion_text) and suggestion_text not in ("", "Edited plan")

                if has_edited_steps and not has_meaningful_feedback:
                    # User edited steps directly — use edited plan without replanning
                    logger.info("[follow_up REPLAN] User edited steps with no feedback - using edited plan directly")
                    follow_up_plan = self._build_plan_from_edited_steps(
                        question, approval.edited_steps, start_number=next_step_number
                    )
                else:
                    # User has meaningful feedback — replan
                    if has_edited_steps:
                        edited_plan_text = "\n".join(
                            f"{step['number']}. {step['goal']}" for step in approval.edited_steps
                        )
                        context_prompt_with_feedback = f"""{context_prompt}

**Requested plan structure (follow this exactly):**
{edited_plan_text}

**User notes:** {suggestion_text}"""
                    else:
                        context_prompt_with_feedback = f"""{context_prompt}

User feedback: {suggestion_text}
"""

                    # Emit replanning event
                    self._emit_event(StepEvent(
                        event_type="replanning",
                        step_number=0,
                        data={"feedback": suggestion_text}
                    ))

                    self._sync_user_facts_to_planner()
                    self._sync_glossary_to_planner(question)
                    self._sync_available_roles_to_planner()
                    planner_response = self.planner.plan(context_prompt_with_feedback)
                    follow_up_plan = planner_response.plan

                    # Validate: ensure "enhance" plans include a step that updates the source table
                    follow_up_plan = self._ensure_enhance_updates_source(
                        question, follow_up_plan, existing_tables,
                    )

                    # Renumber steps to continue from where we left off
                    # noinspection DuplicatedCode
                    for i, step in enumerate(follow_up_plan.steps):
                        step.number = next_step_number + i

                    # Emit updated plan
                    self._emit_event(StepEvent(
                        event_type="plan_ready",
                        step_number=0,
                        data={
                            "steps": [
                                {"number": s.number, "goal": s.goal, "depends_on": s.depends_on, "role_id": s.role_id}
                                for s in follow_up_plan.steps
                            ],
                            "reasoning": planner_response.reasoning,
                            "is_followup": True,
                        }
                    ))

                    # Request approval again
                    approval = self._request_approval(question, planner_response)
                    if approval.decision == PlanApproval.REJECT:
                        return {
                            "success": False,
                            "rejected": True,
                            "plan": follow_up_plan,
                            "reason": approval.reason,
                            "message": "Follow-up plan was rejected by user.",
                        }
                    elif approval.decision == PlanApproval.COMMAND:
                        return {
                            "success": False,
                            "command": approval.command,
                            "message": "Slash command entered during approval.",
                        }

            # APPROVE — apply any edits/deletions to the follow-up plan
            if approval.edited_steps:
                follow_up_plan = self._build_plan_from_edited_steps(
                    question, approval.edited_steps, start_number=next_step_number
                )
            elif approval.deleted_steps:
                deleted_set = set(approval.deleted_steps)
                follow_up_plan.steps = [s for s in follow_up_plan.steps if s.number not in deleted_set]
                # Renumber remaining steps sequentially from next_step_number
                for i, step in enumerate(follow_up_plan.steps):
                    step.number = next_step_number + i

        # Materialize facts table before execution starts
        self._materialize_facts_table()

        # Execute each step
        # Phase 4: Reset cancellation state before starting execution
        self.reset_cancellation()
        all_results = []
        cancelled = False

        for step in follow_up_plan.steps:
            # Phase 4: Check for cancellation before starting each step
            if self.is_cancelled():
                cancelled = True
                self._emit_event(StepEvent(
                    event_type="execution_cancelled",
                    step_number=step.number,
                    data={
                        "message": "Execution cancelled",
                        "completed_steps": len([r for r in all_results if r.success]),
                    }
                ))
                break

            step.status = StepStatus.RUNNING

            result = self._execute_step(step)

            if result.success:
                follow_up_plan.mark_step_completed(step.number, result)
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
            else:
                follow_up_plan.mark_step_failed(step.number, result)
                self.history.record_query(
                    session_id=self.session_id,
                    question=question,
                    success=False,
                    attempts=result.attempts,
                    duration_ms=result.duration_ms,
                    error=result.error,
                )
                return {
                    "success": False,
                    "plan": follow_up_plan,
                    "error": result.error,
                    "completed_steps": follow_up_plan.completed_steps,
                }

            all_results.append(result)

        # Phase 4: Handle cancellation - return with completed results preserved
        if cancelled:
            # Combine output from completed steps
            completed_output = ""
            if all_results:
                completed_output = "\n\n".join([
                    f"Step {step.number}: {step.goal}\n{r.stdout}"
                    for step, r in zip(follow_up_plan.steps, all_results) if r.success
                ])

            # Process any queued intents
            queued_results = self.process_queued_intents()

            return {
                "success": False,
                "cancelled": True,
                "plan": follow_up_plan,
                "completed_steps": follow_up_plan.completed_steps,
                "partial_output": completed_output,
                "queued_intent_results": queued_results,
                "message": f"Execution cancelled. {len(follow_up_plan.completed_steps)} step(s) completed.",
            }

        # Record successful follow-up
        total_duration = sum(r.duration_ms for r in all_results)
        total_attempts = sum(r.attempts for r in all_results)

        combined_output = "\n\n".join([
            f"Step {step.number}: {step.goal}\n{r.stdout}"
            for step, r in zip(follow_up_plan.steps, all_results)
        ])

        # Auto-publish final step artifacts (they appear in artifacts panel)
        # Find the last step that actually created tables (may not be the last step if it's just a summary)
        if all_results and self.registry:
            # Collect tables from last 2 steps that created tables
            # noinspection DuplicatedCode
            final_tables = []
            steps_with_tables = 0
            for result in reversed(all_results):
                if result.success and result.tables_created:
                    final_tables.extend(result.tables_created)
                    steps_with_tables += 1
                    if steps_with_tables >= 2:
                        break

            # Filter to most important tables
            important_keywords = ("final", "report", "result", "recommendation", "summary", "output")
            important_tables = [t for t in final_tables if any(kw in t.lower() for kw in important_keywords)]

            if not important_tables:
                for result in reversed(all_results):
                    if result.success and result.tables_created:
                        important_tables = result.tables_created[:5]
                        break

            tables_to_publish = important_tables[:8] if important_tables else []

            if tables_to_publish:
                self.registry.mark_final_step(
                    user_id=self.user_id,
                    session_id=self.session_id,
                    table_names=tables_to_publish,
                )
                logger.debug(f"Auto-published follow-up final step tables: {tables_to_publish}")

        # Emit raw results first (so user can see them immediately)
        self._emit_event(StepEvent(
            event_type="raw_results_ready",
            step_number=0,
            data={"output": combined_output}
        ))

        # Analyze follow-up question for brief output preference (LLM-based, not keywords)
        follow_up_analysis = self._analyze_question(question)

        # Check for created artifacts (to mention in synthesis)
        from constat.visualization.output import peek_pending_outputs
        pending_artifacts = peek_pending_outputs()  # Peek without clearing

        # Check if insights are enabled (config or per-query brief detection via LLM)
        skip_insights = (
            not self.session_config.enable_insights
            or follow_up_analysis.wants_brief
        )
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
            # Synthesize final answer
            self._emit_event(StepEvent(
                event_type="synthesizing",
                step_number=0,
                data={"message": "Synthesizing final answer..."}
            ))

            final_answer = self._synthesize_answer(question, combined_output, pending_artifacts)

            self._emit_event(StepEvent(
                event_type="answer_ready",
                step_number=0,
                data={"answer": final_answer}
            ))

            # Run facts extraction and suggestions generation in parallel
            tables = self.datastore.list_tables() if self.datastore else []
            response_facts, suggestions = self._run_post_synthesis_parallel(
                question, final_answer, tables
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
            question=question,
            success=True,
            attempts=total_attempts,
            duration_ms=total_duration,
            answer=final_answer,
        )

        # Auto-compact if context is too large
        self._auto_compact_if_needed()

        return {
            "success": True,
            "plan": follow_up_plan,
            "results": all_results,
            "output": combined_output,
            "final_answer": final_answer,
            "suggestions": suggestions,
            "scratchpad": self.scratchpad.to_markdown(),
            "datastore_tables": self.datastore.list_tables() if self.datastore else [],
        }

    def _solve_knowledge(self, problem: str) -> dict:
        """
        Solve a problem in knowledge mode using document lookup + LLM synthesis.

        This mode is for explanation/knowledge requests that don't need data analysis.
        It searches configured documents and synthesizes an explanation.

        Args:
            problem: The question/request to answer

        Returns:
            Dict with synthesized explanation and sources
        """
        start_time = time.time()

        # Step 1: Search documents for relevant content
        self._emit_event(StepEvent(
            event_type="searching_documents",
            step_number=0,
            data={"message": "Searching reference documents..."}
        ))

        sources = []
        doc_context = ""

        if self.doc_tools and self.config.documents:
            # Search for relevant document excerpts
            search_results = self.doc_tools.search_documents(problem, limit=5)

            if search_results:
                doc_lines = ["Relevant document excerpts:"]
                for i, result in enumerate(search_results, 1):
                    doc_name = result.get("document", "unknown")
                    excerpt = result.get("excerpt", "")
                    relevance = result.get("relevance", 0)
                    section = result.get("section", "")

                    source_info = {
                        "document": doc_name,
                        "section": section,
                        "relevance": relevance,
                    }
                    sources.append(source_info)

                    doc_lines.append(f"\n[{i}] From '{doc_name}'" + (f" - {section}" if section else ""))
                    doc_lines.append(excerpt)

                doc_context = "\n".join(doc_lines)

        # Step 2: Build prompt for LLM synthesis
        self._emit_event(StepEvent(
            event_type="synthesizing",
            step_number=0,
            data={"message": "Synthesizing explanation..."}
        ))

        # System prompt for knowledge/explanation queries
        system_prompt = """You are a knowledgeable assistant for explanation and lookup requests.

Answer questions using configured reference documents and your general knowledge.
Be accurate and cite your sources when referencing specific documents.
If you don't have enough information, say so rather than guessing."""

        # Add context about the configuration (including active role)
        config_prompt = self._get_system_prompt()
        if config_prompt:
            system_prompt = f"{system_prompt}\n\n{config_prompt}"

        # Inject system capabilities doc if question is about the system
        _self_keywords = ("you", "vera", "constat", "your", "this system", "this tool")
        if any(kw in problem.lower() for kw in _self_keywords):
            capabilities_doc = load_prompt("system_capabilities.md")
            doc_context = f"## System Capabilities Reference\n{capabilities_doc}\n\n{doc_context}"

        # Build user message with document context
        if doc_context:
            user_message = f"""Question: {problem}

{doc_context}

Please provide a clear, accurate explanation based on the documents above and your general knowledge.
Cite specific documents when referencing them."""
        else:
            user_message = f"""Question: {problem}

No reference documents are configured. Please provide an explanation based on your general knowledge.
If you don't have enough information, say so rather than guessing."""

        # Step 3: Generate response
        try:
            result = self.router.execute(
                task_type=TaskType.SYNTHESIS,
                system=system_prompt,
                user_message=user_message,
                max_tokens=self.router.max_output_tokens,
            )

            answer = result.content
            duration_ms = int((time.time() - start_time) * 1000)

            self._emit_event(StepEvent(
                event_type="knowledge_complete",
                step_number=0,
                data={
                    "has_documents": bool(sources),
                    "source_count": len(sources),
                }
            ))

            # Build final output
            output_parts = [answer]

            if sources:
                output_parts.extend([
                    "",
                    "**Sources consulted:**",
                ])
                for src in sources:
                    src_line = f"- {src['document']}"
                    if src.get('section'):
                        src_line += f" ({src['section']})"
                    output_parts.append(src_line)

            final_output = "\n".join(output_parts)

            # Record in history (only if session datastore is initialized)
            if self.datastore:
                self.history.record_query(
                    session_id=self.session_id,
                    question=problem,
                    success=True,
                    attempts=1,
                    duration_ms=duration_ms,
                    answer=final_output,
                )

            return {
                "success": True,
                "meta_response": True,  # Display as meta-response (no tables)
                "mode": "knowledge",
                "output": final_output,
                "sources": sources,
                "plan": None,  # No plan in knowledge mode
                "suggestions": [
                    "Tell me more about a specific aspect",
                    "What data is available to analyze?",
                ],
            }

        except Exception as e:
            _duration_ms = int((time.time() - start_time) * 1000)

            self._emit_event(StepEvent(
                event_type="knowledge_error",
                step_number=0,
                data={"error": str(e)}
            ))

            return {
                "success": False,
                "mode": "knowledge",
                "error": str(e),
                "output": f"Failed to generate explanation: {e}",
            }

    def prove_conversation(self, guidance: str | None = None) -> dict:
        """
        Re-run the original request in proof/auditable mode.

        Takes the original problem from the session and runs it through
        the auditable solver, reusing existing facts and tables from
        the exploratory session.

        Args:
            guidance: Optional user guidance for the proof (e.g., "focus on X", "use table Y")

        Returns:
            Dict with proof results (same format as _solve_auditable)
        """
        import json

        logger.debug("[prove_conversation] Starting prove_conversation")

        # Check if we have a session to prove
        if not self.session_id:
            return {"error": "No active session to prove"}

        if not self.datastore:
            return {"error": "No datastore available"}

        # Get the original problem
        original_problem = self.datastore.get_session_meta("problem")
        if not original_problem:
            return {"no_claims": True, "error": "No conversation to prove"}

        # Get follow-up questions to include in proof
        follow_ups_json = self.datastore.get_session_meta("follow_ups")
        follow_ups = []
        if follow_ups_json:
            try:
                follow_ups = json.loads(follow_ups_json)
            except json.JSONDecodeError:
                pass

        # Build combined problem statement
        if follow_ups:
            combined_problem = f"""Original request: {original_problem}

Follow-up requests:
{chr(10).join(f'- {q}' for q in follow_ups)}

Prove all of the above claims and provide a complete audit trail."""
            logger.debug(f"[prove_conversation] Combined problem with {len(follow_ups)} follow-ups")
        else:
            combined_problem = original_problem

        if guidance:
            combined_problem += f"\n\nAdditional guidance for this proof: {guidance}"
            logger.debug(f"[prove_conversation] Added guidance: {guidance[:100]}")

        logger.debug(f"[prove_conversation] Running proof for: {combined_problem[:150]}...")

        # Gather step codes from exploratory session as hints for inference generation
        self._proof_step_hints = []
        if self.history and self.session_id:
            try:
                step_codes = self.history.list_step_codes(self.session_id)
                if step_codes:
                    self._proof_step_hints = step_codes
                    logger.info(f"[prove_conversation] Loaded {len(step_codes)} step code hints for proof")
            except Exception as e:
                logger.debug(f"[prove_conversation] Could not load step codes: {e}")

        # Clear old inference codes from previous proof runs
        if self.history and self.session_id:
            self.history.clear_inferences(self.session_id)

        # Emit proof_start event so UI shows "Generating proof..." instead of "Planning..."
        self._emit_event(StepEvent(
            event_type="proof_start",
            step_number=0,
            data={"problem": combined_problem[:100]}
        ))

        # For proof mode, we do NOT pass cached/derived facts as hints.
        # Proof must derive from GROUND TRUTH sources only (databases, APIs, documents).
        # This ensures the proof is independent and verifiable.
        logger.debug("[prove_conversation] Proof will derive from ground truth sources only (no cached facts)")

        # Auto-approve during /prove
        original_auto_approve = self.session_config.auto_approve
        self.session_config.auto_approve = True

        try:
            # Run the combined problem through the auditable solver
            # No cached_fact_hints - proof derives from ground truth only
            result = self._solve_auditable(combined_problem)

            self._emit_event(StepEvent(
                event_type="proof_complete",
                step_number=0,
                data={
                    "success": result.get("success", False),
                    "confidence": result.get("confidence", 0.0),
                }
            ))

            # Generate proof summary asynchronously
            proof_nodes = result.get("proof_nodes", [])
            logger.info(f"[prove_conversation] Generating summary: success={result.get('success')}, proof_nodes count={len(proof_nodes)}")
            if result.get("success") and proof_nodes:
                try:
                    from constat.api.summarization import summarize_proof
                    logger.info(f"[prove_conversation] Calling summarize_proof with {len(proof_nodes)} nodes")
                    summary_result = summarize_proof(
                        problem=result.get("problem", combined_problem),
                        proof_nodes=proof_nodes,
                        llm=self.router,
                    )
                    logger.info(f"[prove_conversation] summarize_proof returned success={summary_result.success}, has_summary={bool(summary_result.summary)}, error={summary_result.error}")
                    if summary_result.success and summary_result.summary:
                        # Save as artifact (optional - don't fail if this fails)
                        if self.datastore:
                            try:
                                self.datastore.add_artifact(
                                    step_number=0,
                                    attempt=1,
                                    artifact_type="markdown",
                                    content=f"# Proof Summary\n\n{summary_result.summary}",
                                    name="proof_summary",
                                    title="Proof Summary",
                                )
                            except Exception as ae:
                                logger.warning(f"[prove_conversation] Failed to save summary artifact: {ae}")
                        # Emit event that summary is ready (always emit if summary generated)
                        self._emit_event(StepEvent(
                            event_type="proof_summary_ready",
                            step_number=0,
                            data={"summary": summary_result.summary}
                        ))
                        logger.info(f"[prove_conversation] Proof summary generated and emitted")
                    else:
                        logger.warning(f"[prove_conversation] summarize_proof failed: {summary_result.error}")
                except Exception as e:
                    logger.warning(f"Failed to generate proof summary: {e}", exc_info=True)
            else:
                logger.warning(f"[prove_conversation] Skipping summary: success={result.get('success')}, has_nodes={bool(proof_nodes)}")

            # noinspection PyAttributeOutsideInit
            self.last_proof_result = result
            return result

        except Exception as e:
            logger.error(f"[prove_conversation] Error: {e}")
            return {"error": str(e), "success": False}

        finally:
            self.session_config.auto_approve = original_auto_approve
            # noinspection PyAttributeOutsideInit
            self._proof_step_hints = []

    def replay(self, problem: str) -> dict:
        """
        Replay a previous session by re-executing stored code without LLM codegen.

        This loads the stored code from the scratchpad and re-executes it,
        then synthesizes a new answer (which still uses the LLM).

        Useful for demos, debugging, or re-running with modified data.

        Args:
            problem: The original problem (used for answer synthesis)

        Returns:
            Dict with results (same format as solve())
        """
        if not self.datastore:
            raise ValueError("No datastore available for replay")

        # Load stored scratchpad entries
        entries = self.datastore.get_scratchpad()
        if not entries:
            raise ValueError("No stored steps to replay")

        # Emit planning complete (we're using stored plan)
        self._emit_event(StepEvent(
            event_type="plan_ready",
            step_number=0,
            data={
                "steps": [
                    {"number": e["step_number"], "goal": e["goal"], "depends_on": []}
                    for e in entries
                ],
                "reasoning": "Replaying stored execution",
                "is_followup": False,
            }
        ))

        # noinspection DuplicatedCode
        all_results = []
        for entry in entries:
            step_number = entry["step_number"]
            goal = entry["goal"]
            code = entry["code"]

            if not code:
                raise ValueError(f"Step {step_number} has no stored code to replay")

            self._emit_event(StepEvent(
                event_type="step_start",
                step_number=step_number,
                data={"goal": goal}
            ))

            self._emit_event(StepEvent(
                event_type="executing",
                step_number=step_number,
                data={"attempt": 1, "code": code}
            ))

            start_time = time.time()

            # Track tables before execution (name + version to detect updates)
            tables_before_list = self.datastore.list_tables()
            tables_before = set(t['name'] for t in tables_before_list)
            versions_before = {t['name']: t.get('version', 1) for t in tables_before_list}

            # Execute stored code
            exec_globals = self._get_execution_globals()
            result = self.executor.execute(code, exec_globals)

            # Auto-save any DataFrames
            if result.success:
                self._auto_save_results(result.namespace, step_number)

            duration_ms = int((time.time() - start_time) * 1000)
            tables_after_list = self.datastore.list_tables()
            tables_after = set(t['name'] for t in tables_after_list)
            versions_after = {t['name']: t.get('version', 1) for t in tables_after_list}
            new_tables = tables_after - tables_before
            updated_tables = {
                name for name in tables_before & tables_after
                if versions_after.get(name, 1) > versions_before.get(name, 1)
                and not name.startswith('_')
            }
            tables_created = list(new_tables | updated_tables)

            if result.success:
                self._emit_event(StepEvent(
                    event_type="step_complete",
                    step_number=step_number,
                    data={
                        "goal": goal,
                        "code": code,
                        "stdout": result.stdout,
                        "attempts": 1,
                        "duration_ms": duration_ms,
                        "tables_created": tables_created,
                    }
                ))

                all_results.append(StepResult(
                    success=True,
                    stdout=result.stdout,
                    attempts=1,
                    duration_ms=duration_ms,
                    tables_created=tables_created,
                    code=code,
                ))
            else:
                self._emit_event(StepEvent(
                    event_type="step_error",
                    step_number=step_number,
                    data={"error": result.stderr or "Execution failed", "attempt": 1}
                ))
                return {
                    "success": False,
                    "error": result.stderr or "Replay execution failed",
                    "step_number": step_number,
                }

        # Synthesize final answer (respects insights config)
        combined_output = "\n\n".join([
            f"Step {entry['step_number']}: {entry['goal']}\n{r.stdout}"
            for entry, r in zip(entries, all_results)
        ])

        # Emit raw results first
        self._emit_event(StepEvent(
            event_type="raw_results_ready",
            step_number=0,
            data={"output": combined_output}
        ))

        # Check for created artifacts (to mention in synthesis)
        from constat.visualization.output import peek_pending_outputs
        pending_artifacts = peek_pending_outputs()

        # Check if insights are enabled
        skip_insights = not self.session_config.enable_insights

        if skip_insights:
            final_answer = combined_output
            self._emit_event(StepEvent(
                event_type="answer_ready",
                step_number=0,
                data={"answer": final_answer, "brief": True}
            ))
        else:
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

        total_duration = sum(r.duration_ms for r in all_results)

        return {
            "success": True,
            "results": all_results,
            "output": combined_output,
            "final_answer": final_answer,
            "datastore_tables": self.datastore.list_tables(),
            "duration_ms": total_duration,
            "replay": True,
        }
