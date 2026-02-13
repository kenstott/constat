# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.
"""Intents mixin: intent handling, slash commands."""
from __future__ import annotations

import logging
from typing import Optional

from constat.commands import get_help_markdown
from constat.core.models import Plan, PlannerResponse, Step, StepType, TaskType
from constat.execution.mode import (
    PlanApprovalRequest, PlanApprovalResponse,
    Phase, SubIntent, TurnIntent, ConversationState,
)
from constat.execution.scratchpad import Scratchpad

logger = logging.getLogger(__name__)


# noinspection PyUnresolvedReferences
class IntentsMixin:

    def _request_approval(
        self,
        problem: str,
        planner_response: PlannerResponse,
    ) -> PlanApprovalResponse:
        """
        Request approval for a plan.

        If auto_approve is set or no callback is registered, auto-approves.
        Otherwise calls the registered callback.

        Args:
            problem: The original problem
            planner_response: The planner's response with plan and reasoning

        Returns:
            PlanApprovalResponse with user's decision
        """
        # Auto-approve if configured
        if self.session_config.auto_approve:
            return PlanApprovalResponse.approve()

        # No callback registered - auto-approve
        if not self._approval_callback:
            return PlanApprovalResponse.approve()

        # Build approval request
        steps = [
            {
                "number": step.number,
                "goal": step.goal,
                "inputs": step.expected_inputs,
                "outputs": step.expected_outputs,
                "role_id": step.role_id,
            }
            for step in planner_response.plan.steps
        ]

        request = PlanApprovalRequest(
            problem=problem,
            steps=steps,
            reasoning=planner_response.reasoning,
        )

        return self._approval_callback(request)

    def _ensure_enhance_updates_source(
        self,
        question: str,
        plan: Plan,
        existing_tables: list[dict],
    ) -> Plan:
        """Append an update step when an 'enhance' plan only creates a mapping table.

        The LLM planner consistently decomposes "enhance X with Y" into
        analyze → fetch reference → create mapping, but omits the final
        "apply mapping back to X" step. This method detects that pattern
        and appends the missing step.
        """
        import re

        # Only applies when there are steps and existing tables
        if not plan.steps or not existing_tables:
            return plan

        # Detect enhance intent
        enhance_re = re.compile(
            r'\b(?:enhance|enrich|extend|augment)\b|'
            r'\badd\s+(?:a\s+)?(?:column|field|the)\b',
            re.IGNORECASE,
        )
        if not enhance_re.search(question):
            return plan

        # Find candidate target table by matching table name fragments to question
        # Prefer higher step_number (most recent working dataset) to break ties
        question_lower = question.lower()
        candidates = [t for t in existing_tables if not t['name'].startswith('_')]

        target_table = None
        best_score = (0, -1)  # (word_overlap, step_number)
        for t in candidates:
            name = t['name']
            name_parts = [p for p in name.lower().replace('_', ' ').split() if len(p) > 3]
            overlap = sum(1 for p in name_parts if p in question_lower)
            step_num = t.get('step_number', 0) or 0
            score = (overlap, step_num)
            if score > best_score and overlap > 0:
                best_score = score
                target_table = name

        if not target_table:
            return plan

        # Check whether the last step already updates the target table
        last_step = plan.steps[-1]
        last_goal_lower = last_step.goal.lower()
        target_words = [p for p in target_table.lower().replace('_', ' ').split() if len(p) > 3]

        mentions_target = any(w in last_goal_lower for w in target_words)
        mentions_update = any(
            kw in last_goal_lower
            for kw in ['update', 'add column', 'enhance', 'enrich', 'modify', 'apply', 'save back']
        )

        if mentions_target and mentions_update:
            return plan  # Already correct

        # Check if ANY step already updates the target
        for step in plan.steps:
            goal_lower = step.goal.lower()
            if any(w in goal_lower for w in target_words) and any(
                kw in goal_lower
                for kw in ['update', 'add column', 'enhance', 'enrich', 'modify', 'apply', 'save back']
            ):
                return plan  # Some step already handles it

        # Append the missing update step
        logger.info(
            f"[PLAN_VALIDATION] Enhance plan missing update step for '{target_table}'. Appending step."
        )
        last = plan.steps[-1]
        update_step = Step(
            number=last.number + 1,
            goal=(
                f"Apply the mapping from previous steps to update `{target_table}` "
                f"by adding the new column(s) and saving it back with the same name"
            ),
            expected_inputs=[
                last.expected_outputs[0] if last.expected_outputs else "mapping",
                target_table,
            ],
            expected_outputs=[target_table],
            depends_on=[last.number],
            task_type=TaskType.PYTHON_ANALYSIS,
            complexity="low",
            role_id=last.role_id,
        )
        plan.steps.append(update_step)
        return plan

    def _build_plan_from_edited_steps(
        self,
        problem: str,
        edited_steps: list[dict],
        start_number: int = 1,
    ) -> Plan:
        """
        Build a Plan directly from user-edited steps, skipping the planner.

        Used when user edits the plan steps but provides no additional feedback,
        meaning they want exactly what they specified without re-interpretation.

        Args:
            problem: The original problem
            edited_steps: List of {"number": int, "goal": str} from user edits
            start_number: First step number (e.g. 4 for follow-up after 3 initial steps)

        Returns:
            Plan object ready for execution
        """
        from datetime import datetime

        # Get original plan steps for metadata preservation
        original_steps_by_goal = {}
        if self.plan:
            for step in self.plan.steps:
                # Key by normalized goal for fuzzy matching
                normalized = step.goal.lower().strip()
                original_steps_by_goal[normalized] = step

        # Build new steps with sequential numbering from start_number
        new_steps = []
        for i, edited in enumerate(edited_steps):
            step_num = start_number + i
            goal = edited.get("goal", "")
            original_number = edited.get("number", step_num)

            # Try to find original step for metadata (by goal similarity)
            normalized_goal = goal.lower().strip()
            original = original_steps_by_goal.get(normalized_goal)

            # Also try to find by original step number
            if not original and self.plan:
                original = self.plan.get_step(original_number)

            # Create step with preserved metadata where available
            step = Step(
                number=step_num,  # Renumber sequentially from start_number
                goal=goal,
                expected_inputs=original.expected_inputs if original else [],
                expected_outputs=original.expected_outputs if original else [],
                depends_on=[step_num - 1] if step_num > start_number else [],  # Sequential dependencies for safety
                step_type=original.step_type if original else StepType.PYTHON,
                task_type=original.task_type if original else TaskType.PYTHON_ANALYSIS,
                complexity=original.complexity if original else "medium",
                role_id=original.role_id if original else None,
                skill_ids=original.skill_ids if original else None,
            )
            new_steps.append(step)

        return Plan(
            problem=problem,
            steps=new_steps,
            created_at=datetime.now().isoformat(),
        )

    # =========================================================================
    # Phase 3: Intent Classification and Handler Methods
    # =========================================================================

    def _classify_turn_intent(self, user_input: str) -> TurnIntent:
        """
        Classify the user's input into a TurnIntent using the IntentClassifier.

        Builds context from current conversation state and delegates to the
        embedding-based classifier (with LLM fallback for low confidence).

        Args:
            user_input: The user's natural language input.

        Returns:
            TurnIntent with primary intent, optional sub-intent, and optional target.
        """
        # Build context dict for the classifier
        context = {
            "phase": self._conversation_state.phase,
            "has_plan": self._conversation_state.active_plan is not None,
        }

        # Delegate to the intent classifier
        return self._intent_classifier.classify(user_input, context)

    def _handle_query_intent(self, turn_intent: TurnIntent, user_input: str) -> dict:
        """
        Handle QUERY primary intent - answer from knowledge or current context.

        This handles sub-intents:
        - DETAIL: drill down into specific aspect
        - PROVENANCE: show proof chain
        - SUMMARY: condense results
        - LOOKUP: simple fact retrieval
        - Default: general answer using doc search + LLM fallback

        Args:
            turn_intent: The classified turn intent.
            user_input: The user's original input.

        Returns:
            Result dict with output, success, and other metadata.
        """
        sub_intent = turn_intent.sub
        target = turn_intent.target

        # Handle PROVENANCE sub-intent - show proof chain
        if sub_intent == SubIntent.PROVENANCE:
            return self._handle_provenance_query(target, user_input)

        # Handle DETAIL sub-intent - drill down into specific aspect
        if sub_intent == SubIntent.DETAIL:
            return self._handle_detail_query(target, user_input)

        # Handle SUMMARY sub-intent - condense results
        if sub_intent == SubIntent.SUMMARY:
            return self._handle_summary_query(user_input)

        # Handle LOOKUP sub-intent - simple fact retrieval
        if sub_intent == SubIntent.LOOKUP:
            result = self._handle_lookup_query(target, user_input)
            # Check if lookup found data sources that need planning
            if result.get("_route_to_planning"):
                return result  # Signal will be handled by solve()
            return result

        # Default: general answer using doc search + LLM fallback (KNOWLEDGE mode logic)
        return self._handle_general_query(user_input)

    def _handle_provenance_query(self, target: Optional[str], _user_input: str) -> dict:
        """Handle provenance/proof chain query."""
        # Check if we have resolved facts with provenance
        all_facts = self.fact_resolver.get_all_facts()

        if not all_facts:
            return {
                "success": True,
                "output": "No facts have been resolved yet. Please run an analysis first to establish a proof chain.",
                "meta_response": True,
            }

        # Build provenance output
        provenance_lines = ["**Fact Provenance:**\n"]

        for name, fact in all_facts.items():
            # Check if this fact matches the target (if specified)
            if target and target.lower() not in name.lower():
                continue

            provenance_lines.append(f"**{name}**: {fact.display_value}")
            if hasattr(fact, "source") and fact.source:
                provenance_lines.append(f"  - Source: {fact.source}")
            if hasattr(fact, "reasoning") and fact.reasoning:
                provenance_lines.append(f"  - Reasoning: {fact.reasoning}")
            if hasattr(fact, "confidence") and fact.confidence is not None:
                provenance_lines.append(f"  - Confidence: {fact.confidence:.0%}")
            provenance_lines.append("")

        if len(provenance_lines) == 1:
            return {
                "success": True,
                "output": f"No facts found matching '{target}'." if target else "No facts available.",
                "meta_response": True,
            }

        return {
            "success": True,
            "output": "\n".join(provenance_lines),
            "meta_response": True,
        }

    def _handle_detail_query(self, _target: Optional[str], user_input: str) -> dict:
        """Handle detail/drill-down query."""
        # If we have a datastore with results, try to get details from there
        if self.datastore:
            tables = self.datastore.list_tables()
            if tables:
                # Use LLM to generate a detail explanation
                table_info = "\n".join([f"- {t['name']}: {t['row_count']} rows" for t in tables])
                scratchpad_context = self.datastore.get_scratchpad_as_markdown()

                prompt = f"""The user wants more details about: {user_input}

Available data:
{table_info}

Previous analysis context:
{scratchpad_context}

Provide a detailed explanation or suggest what specific data to examine."""

                result = self.router.execute(
                    task_type=TaskType.SYNTHESIS,
                    system="You are a helpful data analyst providing detailed explanations.",
                    user_message=prompt,
                    max_tokens=self.router.max_output_tokens,
                )

                return {
                    "success": True,
                    "output": result.content,
                    "meta_response": True,
                }

        # Fallback to general query handling
        return self._handle_general_query(user_input)

    def _handle_summary_query(self, user_input: str) -> dict:
        """Handle summary/condensation query.

        Checks if the user is asking about a specific document first.
        If so, summarizes that document. Otherwise, summarizes previous analysis results.
        """
        # Check if the query mentions a document - search for relevant documents
        if self.doc_tools:
            try:
                doc_results = self.doc_tools.search_documents(user_input, limit=3)
                if doc_results and doc_results[0].get("similarity", 0) > 0.4:
                    # Found a relevant document - summarize it
                    doc_name = doc_results[0]["name"]
                    doc_content = self.doc_tools.get_document(doc_name)
                    if doc_content:
                        # Truncate if too long
                        max_chars = 15000
                        if len(doc_content) > max_chars:
                            doc_content = doc_content[:max_chars] + "\n\n[... truncated for length ...]"

                        prompt = f"""Summarize this document based on the user's request.

Document: {doc_name}
Content:
{doc_content}

User request: {user_input}

Provide a clear, structured summary focusing on what the user asked for."""

                        result = self.router.execute(
                            task_type=TaskType.SUMMARIZATION,
                            system="You are a document summarizer. Extract key concepts and structure your response clearly.",
                            user_message=prompt,
                            max_tokens=self.router.max_output_tokens,
                        )

                        return {
                            "success": True,
                            "output": result.content,
                            "meta_response": True,
                        }
            except Exception as e:
                logger.debug(f"Document search for summary failed: {e}")

        # Check if we have previous results to summarize
        if self.datastore:
            scratchpad_entries = self.datastore.get_scratchpad()
            if scratchpad_entries:
                # Combine all previous results
                context = "\n\n".join([
                    f"Step {e['step_number']}: {e['goal']}\n{e['narrative']}"
                    for e in scratchpad_entries
                ])

                prompt = f"""Summarize these analysis results concisely:

{context}

User request: {user_input}

Provide a brief, high-level summary of the key findings."""

                result = self.router.execute(
                    task_type=TaskType.SUMMARIZATION,
                    system="You are a concise summarizer. Focus on key insights.",
                    user_message=prompt,
                    max_tokens=self.router.max_output_tokens,
                )

                return {
                    "success": True,
                    "output": result.content,
                    "meta_response": True,
                }

        return {
            "success": True,
            "output": "No previous analysis results to summarize. Please run an analysis first.",
            "meta_response": True,
        }

    def _handle_lookup_query(self, _target: Optional[str], user_input: str) -> dict:
        """Handle simple fact lookup query.

        Checks all available sources: cached facts, APIs, databases, and documents.
        Returns a signal to route to planning if data sources are found.
        """
        # First, try to answer from cached facts
        cached_result = self._answer_from_cached_facts(user_input)
        if cached_result:
            return cached_result

        # Check if any data sources (APIs, tables) can answer this query
        # Use a reasonable similarity threshold
        sources = self.find_relevant_sources(
            user_input,
            table_limit=3,
            doc_limit=3,
            api_limit=3,
            min_similarity=0.3,
        )
        logger.debug(f"[LOOKUP] find_relevant_sources returned: apis={len(sources.get('apis', []))}, tables={len(sources.get('tables', []))}, docs={len(sources.get('documents', []))}")

        # If we found relevant APIs or tables, signal to route to planning
        has_data_sources = bool(sources.get("apis") or sources.get("tables"))
        if has_data_sources:
            # Log what we found for debugging
            api_names = [a["name"] for a in sources.get("apis", [])]
            table_names = [t["name"] for t in sources.get("tables", [])]
            logger.debug(
                f"[LOOKUP] Found data sources for query - APIs: {api_names}, Tables: {table_names}"
            )
            # Return signal to route to planning
            return {"_route_to_planning": True}

        # If no data sources match, fall back to document search + LLM synthesis
        return self._handle_general_query(user_input)

    def _handle_general_query(self, user_input: str) -> dict:
        """
        Handle general query using document search + LLM fallback.

        Checks for relevant data sources first — if databases or APIs match,
        routes to planning instead of answering from knowledge alone.
        """
        # Check if any data sources (APIs, tables) can answer this query
        sources = self.find_relevant_sources(
            user_input,
            table_limit=3,
            doc_limit=3,
            api_limit=3,
            min_similarity=0.3,
        )
        has_data_sources = bool(sources.get("apis") or sources.get("tables"))
        if has_data_sources:
            logger.debug(
                f"[GENERAL_QUERY] Found data sources, routing to planning: "
                f"APIs={[a['name'] for a in sources.get('apis', [])]}, "
                f"Tables={[t['name'] for t in sources.get('tables', [])]}"
            )
            return {"_route_to_planning": True}

        # No data sources match — use doc search + LLM synthesis
        return self._solve_knowledge(user_input)

    def _handle_plan_new_intent(self, turn_intent: TurnIntent, user_input: str) -> dict:
        """
        Handle PLAN_NEW primary intent - enhance problem based on sub-intent.

        NOTE: This method is called from solve() BEFORE the planning flow.
        It enhances the problem statement based on sub-intent (COMPARE, PREDICT).
        The actual planning is done by solve() after this returns.

        This handles sub-intents:
        - COMPARE: evaluate alternatives
        - PREDICT: what-if / forecast
        - Default: pass through unchanged

        Args:
            turn_intent: The classified turn intent.
            user_input: The user's original input.

        Returns:
            Dict with enhanced_problem for solve() to use.
        """
        sub_intent = turn_intent.sub
        enhanced_problem = user_input

        if sub_intent == SubIntent.COMPARE:
            # Add comparison context to the problem
            enhanced_problem = f"Compare and evaluate: {user_input}\n\nProvide a comparative analysis highlighting differences, pros/cons, and recommendations."

        elif sub_intent == SubIntent.PREDICT:
            # Add forecasting context to the problem
            enhanced_problem = f"Forecast/What-if analysis: {user_input}\n\nProvide predictive analysis with assumptions clearly stated."

        return {"enhanced_problem": enhanced_problem, "sub_intent": sub_intent}

    def _handle_plan_continue_intent(self, turn_intent: TurnIntent, user_input: str) -> dict:
        """
        Handle PLAN_CONTINUE primary intent - refine or extend the active plan.

        Uses the user's message as context for replanning.

        Args:
            turn_intent: The classified turn intent.
            user_input: The user's original input (used as modification context).

        Returns:
            Result dict from the replanning flow.
        """
        # Check for CORRECTION sub-intent - save as reusable learning
        if turn_intent.sub == SubIntent.CORRECTION:
            self._save_correction_as_learning(user_input)

        # Transition phase to PLANNING
        self._apply_phase_transition("plan_new")  # Returns to planning state

        # Check if there's a previous problem to continue from
        previous_problem = None
        if self.datastore:
            previous_problem = self.datastore.get_session_meta("problem")

        if not previous_problem:
            # No previous context - treat as a new plan
            return self._handle_plan_new_intent(turn_intent, user_input)

        # Delegate to existing follow_up() method which handles replanning
        result = self.follow_up(user_input, auto_classify=False)

        # Update conversation state based on result
        if result.get("success"):
            if result.get("plan"):
                self._conversation_state.active_plan = result["plan"]
            self._apply_phase_transition("complete")
        else:
            self._apply_phase_transition("fail")
            self._conversation_state.failure_context = result.get("error")

        return result

    def _handle_control_intent(self, turn_intent: TurnIntent, user_input: str) -> dict:
        """
        Handle CONTROL primary intent - system/session commands.

        This handles sub-intents:
        - RESET: clear session state
        - REDO_CMD: re-execute last plan
        - HELP: show available commands
        - STATUS: show current state
        - EXIT: end session (returns signal to caller)
        - CANCEL: stop execution (if executing)
        - REPLAN: stop execution, return to planning

        Args:
            turn_intent: The classified turn intent.
            user_input: The user's original input.

        Returns:
            Result dict with output, success, and control signals.
        """
        sub_intent = turn_intent.sub

        # RESET: clear session state
        if sub_intent == SubIntent.RESET:
            return self._handle_reset()

        # REDO_CMD: re-execute last plan
        if sub_intent == SubIntent.REDO_CMD:
            return self._handle_redo()

        # HELP: show available commands
        if sub_intent == SubIntent.HELP:
            return self._handle_help()

        # STATUS: show current state
        if sub_intent == SubIntent.STATUS:
            return self._handle_status()

        # EXIT: end session
        if sub_intent == SubIntent.EXIT:
            return {
                "success": True,
                "exit": True,
                "output": "Ending session.",
                "meta_response": True,
            }

        # CANCEL: stop execution
        if sub_intent == SubIntent.CANCEL:
            return self._handle_cancel()

        # REPLAN: stop execution and return to planning
        if sub_intent == SubIntent.REPLAN:
            return self._handle_replan(user_input)

        # Default: unknown control command
        return {
            "success": True,
            "output": f"Unknown control command. Use /help to see available commands.",
            "meta_response": True,
        }

    def _handle_slash_command(self, command_text: str) -> dict:
        """Handle explicit slash commands via the command registry.

        This is a fast path for slash commands that bypasses intent classification.
        Commands like /tables, /show, /help are routed directly to their handlers.

        Args:
            command_text: The full command text (e.g., "/tables" or "/show orders")

        Returns:
            Result dict with output, success, and other metadata.
        """
        from constat.commands.registry import execute_command, parse_command
        from constat.commands.base import (
            TableResult,
            ListResult,
            TextResult,
            ErrorResult,
        )

        # Parse the command
        cmd, args = parse_command(command_text)

        # Execute via registry
        # noinspection PyTypeChecker
        result = execute_command(self, cmd, args)

        # Convert CommandResult to dict format expected by solve()
        if isinstance(result, TableResult):
            # Format table as markdown
            lines = []
            if result.title:
                lines.append(f"**{result.title}**\n")
            if result.columns and result.rows:
                # Header
                lines.append("| " + " | ".join(str(c) for c in result.columns) + " |")
                lines.append("| " + " | ".join("---" for _ in result.columns) + " |")
                # Rows
                for row in result.rows:
                    lines.append("| " + " | ".join(str(v) for v in row) + " |")
            if result.footer:
                lines.append(f"\n*{result.footer}*")

            return {
                "success": result.success,
                "output": "\n".join(lines) if lines else result.footer or "No data.",
                "meta_response": True,
            }

        elif isinstance(result, ListResult):
            # Format list
            lines = []
            if result.title:
                lines.append(f"**{result.title}**\n")
            if result.items:
                for item in result.items:
                    if isinstance(item, dict):
                        # Code block
                        if "code" in item:
                            lines.append(f"### Step {item.get('step', '?')}: {item.get('goal', '')}")
                            lines.append(f"```{item.get('language', 'python')}")
                            lines.append(item["code"])
                            lines.append("```\n")
                        # Artifact-style item (has name, type, step)
                        elif "name" in item and "type" in item:
                            name = item.get("name", "")
                            atype = item.get("type", "")
                            step = item.get("step", "-")
                            title = item.get("title", "")
                            # Format: - **name** (type) - title if available
                            if title:
                                lines.append(f"- **{name}** ({atype}, step {step}) - {title}")
                            else:
                                lines.append(f"- **{name}** ({atype}, step {step})")
                        # Generic dict - format key: value pairs
                        else:
                            name = item.get("name", item.get("id", "Item"))
                            lines.append(f"- **{name}**")
                            for key, value in item.items():
                                if key not in ("name", "id") and value is not None:
                                    lines.append(f"  - {key}: {value}")
                    else:
                        lines.append(f"- {item}")
            elif result.empty_message:
                lines.append(result.empty_message)

            return {
                "success": result.success,
                "output": "\n".join(lines) if lines else result.empty_message or "No items.",
                "meta_response": True,
            }

        elif isinstance(result, TextResult):
            return {
                "success": result.success,
                "output": result.content,
                "meta_response": True,
            }

        elif isinstance(result, ErrorResult):
            return {
                "success": False,
                "output": f"Error: {result.error}" + (f"\n{result.details}" if result.details else ""),
                "meta_response": True,
            }

        else:
            # Generic fallback
            return {
                "success": getattr(result, "success", True),
                "output": str(result),
                "meta_response": True,
            }

    def _handle_reset(self) -> dict:
        """Handle reset control command - clear session state."""
        # Clear conversation state
        self._conversation_state = ConversationState(
            phase=Phase.IDLE,
        )

        # Clear fact resolver
        self.fact_resolver.clear_all_facts()

        # Clear plan
        self.plan = None

        # Clear scratchpad
        self.scratchpad = Scratchpad()

        # Clear session_id to indicate fresh start
        self.session_id = None
        self.datastore = None

        self._apply_phase_transition("abandon")

        return {
            "success": True,
            "output": "Session reset. All facts and context cleared. Ready for a new question.",
            "meta_response": True,
            "reset": True,
        }

    def _handle_redo(self) -> dict:
        """Handle redo control command - re-execute last plan."""
        if not self.datastore:
            return {
                "success": False,
                "output": "No previous session to redo. Please run an analysis first.",
                "meta_response": True,
            }

        # Get the original problem
        problem = self.datastore.get_session_meta("problem")
        if not problem:
            return {
                "success": False,
                "output": "No previous problem found to redo.",
                "meta_response": True,
            }

        # Use replay to re-execute stored code
        try:
            result = self.replay(problem)
            return result
        except ValueError as e:
            return {
                "success": False,
                "output": f"Cannot redo: {e}",
                "meta_response": True,
            }

    def _handle_help(self) -> dict:
        """Handle help control command - show available commands."""
        # Use centralized help text from HELP_COMMANDS
        return {
            "success": True,
            "output": get_help_markdown(),
            "meta_response": True,
        }

    def _handle_status(self) -> dict:
        """Handle status control command - show current state."""
        state = self._conversation_state

        status_lines = [
            "**Current Session State:**",
            "",
            f"**Mode:** {state.mode.value.upper()}",
            f"**Phase:** {state.phase.value}",
        ]

        if state.active_plan:
            status_lines.append(f"**Active Plan:** {len(state.active_plan.steps)} steps")

        if state.session_facts:
            status_lines.append(f"**Session Facts:** {len(state.session_facts)} facts")

        # Add cached facts info
        all_facts = self.fact_resolver.get_all_facts()
        if all_facts:
            status_lines.append(f"**Resolved Facts:** {len(all_facts)}")
            for name, fact in list(all_facts.items())[:5]:  # Show first 5
                status_lines.append(f"  - {name}: {fact.display_value}")
            if len(all_facts) > 5:
                status_lines.append(f"  ... and {len(all_facts) - 5} more")

        if state.failure_context:
            status_lines.append(f"**Last Error:** {state.failure_context}")

        if self.datastore:
            tables = self.datastore.list_tables()
            if tables:
                status_lines.append(f"**Data Tables:** {len(tables)}")
                for t in tables[:5]:
                    status_lines.append(f"  - {t['name']}: {t['row_count']} rows")

        return {
            "success": True,
            "output": "\n".join(status_lines),
            "meta_response": True,
        }

    def _handle_cancel(self) -> dict:
        """Handle cancel control command - stop execution.

        Uses the Phase 4 cancel_execution() method to set the cancellation
        flag which will be checked between steps. Completed facts are preserved.
        """
        # Request cancellation (signals the execution loop to stop)
        self.cancel_execution()

        # Transition to IDLE
        self._apply_phase_transition("abandon")

        # Clear any queued intents
        cleared = self.clear_intent_queue()

        output = "Execution cancelled. Returned to idle state."
        if cleared > 0:
            output += f" ({cleared} queued intent(s) cleared.)"

        return {
            "success": True,
            "output": output,
            "meta_response": True,
            "cancelled": True,
        }

    def _handle_replan(self, _user_input: str) -> dict:
        """Handle replan control command - stop and revise the plan.

        Uses the Phase 4 cancel_execution() method to stop execution
        and preserve completed facts.
        """
        # Request cancellation
        self.cancel_execution()

        # Transition to PLANNING
        self._apply_phase_transition("replan")

        return {
            "success": True,
            "output": "Ready to revise the plan. Please provide your modifications or ask a new question.",
            "meta_response": True,
            "replan": True,
        }

    def _apply_phase_transition(self, trigger: str) -> None:
        """
        Apply a phase transition based on the trigger.

        Valid triggers and their effects:
        - plan_new: IDLE -> PLANNING (or any -> PLANNING for plan_continue)
        - plan_ready: PLANNING -> AWAITING_APPROVAL
        - approve: AWAITING_APPROVAL -> EXECUTING
        - reject: AWAITING_APPROVAL -> PLANNING
        - complete: EXECUTING -> IDLE
        - fail: EXECUTING -> FAILED
        - retry: FAILED -> EXECUTING
        - replan: FAILED/EXECUTING -> PLANNING
        - abandon: any -> IDLE

        Args:
            trigger: The transition trigger name.
        """
        current_phase = self._conversation_state.phase

        if trigger == "plan_new":
            self._conversation_state.phase = Phase.PLANNING

        elif trigger == "plan_ready":
            if current_phase == Phase.PLANNING:
                self._conversation_state.phase = Phase.AWAITING_APPROVAL

        elif trigger == "approve":
            if current_phase == Phase.AWAITING_APPROVAL:
                self._conversation_state.phase = Phase.EXECUTING

        elif trigger == "reject":
            if current_phase == Phase.AWAITING_APPROVAL:
                self._conversation_state.phase = Phase.PLANNING

        elif trigger == "complete":
            self._conversation_state.phase = Phase.IDLE
            self._conversation_state.failure_context = None

        elif trigger == "fail":
            self._conversation_state.phase = Phase.FAILED

        elif trigger == "retry":
            if current_phase == Phase.FAILED:
                self._conversation_state.phase = Phase.EXECUTING

        elif trigger == "replan":
            if current_phase in (Phase.FAILED, Phase.EXECUTING):
                self._conversation_state.phase = Phase.PLANNING

        elif trigger == "abandon":
            self._conversation_state.phase = Phase.IDLE
            self._conversation_state.failure_context = None
            self._conversation_state.active_plan = None

        else:
            logger.warning(f"Unknown phase transition trigger: {trigger}")

        logger.debug(f"Phase transition: {current_phase.value} --({trigger})--> {self._conversation_state.phase.value}")

    def get_conversation_state(self) -> ConversationState:
        """
        Get the current conversation state.

        Returns:
            The current ConversationState object.
        """
        return self._conversation_state
