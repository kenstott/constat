# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Analysis mixin: question analysis, ambiguity, meta answers."""

from __future__ import annotations

import logging
from typing import Optional

from constat.core.models import TaskType
from constat.prompts import load_prompt
from constat.session._types import QuestionType, QuestionAnalysis, DetectedIntent, ClarificationQuestion, \
    ClarificationRequest, StepEvent, is_meta_question

logger = logging.getLogger(__name__)


# noinspection PyUnresolvedReferences
class AnalysisMixin:

    @staticmethod
    def _classify_question(problem: str) -> str:
        """
        Classify whether a question requires code execution or is a meta-question.

        Returns:
            QuestionType.DATA_ANALYSIS - needs code execution (queries, computation, actions)
            QuestionType.META_QUESTION - about system capabilities (what can you do?)

        Note: We route almost everything through code execution because:
        - Data questions need database queries
        - General knowledge questions can use llm_ask() + computation
        - Action requests (email, export) need code
        - Even "What is sqrt(8)?" benefits from actual computation
        """
        # Only meta-questions about the system bypass code execution
        if is_meta_question(problem):
            return QuestionType.META_QUESTION

        # Everything else goes through code execution
        # The generated code can use llm_ask() for general knowledge
        # and then compute/transform/act on the results
        return QuestionType.DATA_ANALYSIS

    def _try_show_existing_data(self, question: str) -> Optional[dict]:
        """Fast path for 'show me X' requests that display existing tables.

        Detects simple data display requests and handles them directly without
        going through LLM intent classification. This is much faster for
        simple lookups like 'show me the raise_recommendations table'.

        Args:
            question: The user's question

        Returns:
            Result dict if this was a show request, None otherwise
        """
        import re

        if not self.datastore:
            return None

        # Get available table names
        tables = self.datastore.list_tables()
        if not tables:
            return None

        table_names = {t['name'].lower(): t['name'] for t in tables}

        # Patterns for "show me X" type requests
        # Match: "show me X", "display X", "what's in X", "view X", "print X"
        show_patterns = [
            r"^(?:show\s+(?:me\s+)?(?:the\s+)?|display\s+(?:the\s+)?|view\s+(?:the\s+)?|print\s+(?:the\s+)?|what(?:'s| is)\s+in\s+(?:the\s+)?)(.+?)(?:\s+table)?(?:\s+data)?$",
            r"^(.+?)(?:\s+table|\s+data)$",  # "raise_recommendations table"
        ]

        question_lower = question.lower().strip()

        for pattern in show_patterns:
            match = re.match(pattern, question_lower, re.IGNORECASE)
            if match:
                candidate = match.group(1).strip()
                # Check if candidate matches a table name
                if candidate in table_names:
                    actual_table = table_names[candidate]
                    return self._quick_table_display(actual_table)

        # Also check if the entire question is just a table name
        if question_lower in table_names:
            return self._quick_table_display(table_names[question_lower])

        return None

    def _quick_table_display(self, table_name: str) -> dict | None:
        """Display a table quickly without going through planning.

        Args:
            table_name: The name of the table to display

        Returns:
            Result dict with table data
        """
        try:
            # Query the table
            df = self.datastore.query(f"SELECT * FROM {table_name} LIMIT 50")

            # Format output
            try:
                table_str = df.to_markdown(index=False)
            except (ImportError, ValueError):
                table_str = df.to_string(index=False)

            row_count = len(df)
            total_rows = self.datastore.query(f"SELECT COUNT(*) as cnt FROM {table_name}").iloc[0]['cnt']

            output = f"**{table_name}** ({total_rows} rows)\n\n{table_str}"
            if row_count < total_rows:
                output += f"\n\n_Showing first {row_count} of {total_rows} rows_"

            self._emit_event(StepEvent(
                event_type="quick_display",
                step_number=0,
                data={"table": table_name, "rows": row_count}
            ))

            return {
                "success": True,
                "mode": "quick_display",
                "output": output,
                "datastore_tables": self.datastore.list_tables(),
                "suggestions": [
                    f"Run SQL query on {table_name}",
                    "Ask a question about this data",
                ],
            }
        except Exception:
            # If table query fails, return None to fall through to normal processing
            return None

    def _analyze_question(self, problem: str, previous_problem: str = None) -> QuestionAnalysis:
        """
        Analyze a question in a single LLM call: extract facts, classify type, check cached facts.

        This combines what were previously separate operations into one call for efficiency:
        1. Extract embedded facts (e.g., "my role as CFO" -> user_role: CFO)
        2. Classify question type (meta-question vs data analysis)
        3. Check if question can be answered from cached facts

        Args:
            problem: The question to analyze
            previous_problem: If this is a follow-up, the original problem for context

        Returns:
            QuestionAnalysis with question_type, extracted_facts, and optional cached_fact_answer
        """
        # First, use fast regex-based classification for obvious meta-questions
        # This avoids an LLM call for simple cases like "what can you do?"
        if is_meta_question(problem):
            # Return immediately for meta-questions - no LLM classification needed
            # Note: We skip fact extraction here for efficiency. Most meta-questions
            # like "how do you reason" don't contain extractable facts anyway.
            return QuestionAnalysis(
                question_type=QuestionType.META_QUESTION,
                extracted_facts=[],
                cached_fact_answer=None,
            )

        # Get cached facts for context
        cached_facts = self.fact_resolver.get_all_facts()
        fact_context = ""
        if cached_facts:
            fact_context = "Known facts:\n" + "\n".join(
                f"- {name}: {fact.display_value}" for name, fact in cached_facts.items()
            )

        # Build data source context from SessionResources (single source of truth)
        # This includes all databases/APIs/documents from config + active projects
        data_sources = []
        for name, desc in self.resources.get_database_descriptions():
            desc_line = desc.split('\n')[0] if desc else f"database '{name}'"
            data_sources.append(f"DATABASE '{name}': {desc_line}")
        for name, desc in self.resources.get_api_descriptions():
            desc_line = desc.split('\n')[0] if desc else f"API"
            data_sources.append(f"API '{name}': {desc_line}")
        for name, desc in self.resources.get_document_descriptions():
            desc_line = desc.split('\n')[0] if desc else "reference document"
            data_sources.append(f"DOCUMENT '{name}': {desc_line}")

        source_context = ""
        if data_sources:
            source_context = "\nAvailable data sources:\n" + "\n".join(f"- {s}" for s in data_sources)

        # Build follow-up context if this is a continuation of a previous analysis
        followup_context = ""
        if previous_problem:
            followup_context = f"""
This is a FOLLOW-UP to a previous analysis.
Previous question: "{previous_problem}"

CRITICAL INTENT RULES (apply in order):
1. RE-EXECUTION language ANYWHERE in message = REDO intent, NOT NEW_QUESTION:
   "redo", "again", "retry", "rerun", "try again", "this time", "instead", "once more", etc.

2. Requests to CHANGE how something is computed/calculated = STEER_PLAN + REDO:
   "change X to use Y", "use average instead", "compute it differently", "use the 2 most recent"

3. Simple value changes (literal numbers/strings) = MODIFY_FACT:
   "change age to 50", "use $100k instead", "set threshold to 10"

4. NEW_QUESTION is ONLY for completely unrelated questions about different topics.
   If the message references the previous analysis AT ALL, it is NOT NEW_QUESTION."""

        prompt = load_prompt("analyze_question.md").format(
            problem=problem,
            source_context=source_context,
            fact_context=fact_context,
            followup_context=followup_context,
        )

        import logging
        _intent_logger = logging.getLogger(__name__)

        try:
            result = self.router.execute(
                task_type=TaskType.INTENT_CLASSIFICATION,
                system="You analyze user questions efficiently. Be precise and concise.",
                user_message=prompt,
                max_tokens=self.router.max_output_tokens,
            )

            response = result.content.strip()
            _intent_logger.debug(f"[INTENT CLASSIFICATION] Raw LLM response:\n{response}")

            # Parse response
            question_type = QuestionType.DATA_ANALYSIS
            extracted_facts = []
            cached_answer = None

            # Parse FACTS section
            if "FACTS:" in response:
                facts_section = response.split("FACTS:", 1)[1].split("---")[0].strip()
                if facts_section and facts_section != "NONE":
                    for line in facts_section.split("\n"):
                        line = line.strip().lstrip("-").strip()
                        if ":" in line and line.lower() != "none":
                            parts = line.split(":", 1)
                            fact_name = parts[0].strip()
                            value_part = parts[1].strip()

                            # Parse value and optional description (format: "value | description")
                            description = None
                            if "|" in value_part:
                                value_str, description = value_part.split("|", 1)
                                value_str = value_str.strip()
                                description = description.strip()
                            else:
                                value_str = value_part

                            # Try to parse as number
                            try:
                                value = float(value_str)
                                if value == int(value):
                                    value = int(value)
                            except ValueError:
                                value = value_str

                            # Add to fact resolver
                            fact = self.fact_resolver.add_user_fact(
                                fact_name=fact_name,
                                value=value,
                                reasoning=f"Extracted from question: {problem}",
                                description=description,
                            )
                            extracted_facts.append(fact)

            # Parse QUESTION_TYPE
            if "QUESTION_TYPE:" in response:
                type_line = response.split("QUESTION_TYPE:", 1)[1].split("\n")[0].strip()
                type_line = type_line.split("---")[0].strip().upper()
                if "META" in type_line:
                    question_type = QuestionType.META_QUESTION
                elif "GENERAL" in type_line:
                    question_type = QuestionType.GENERAL_KNOWLEDGE

            # Safety check: If classified as GENERAL_KNOWLEDGE but query mentions
            # configured data sources, override to DATA_ANALYSIS
            if question_type == QuestionType.GENERAL_KNOWLEDGE:
                problem_lower = problem.lower()
                # Check if query mentions any data source names or key terms from descriptions
                source_keywords = []
                if self.config.databases:
                    for name, db in self.config.databases.items():
                        source_keywords.append(name.lower())
                        if db.description:
                            # Extract key terms from description (nouns)
                            for word in db.description.lower().split():
                                if len(word) > 4 and word.isalpha():
                                    source_keywords.append(word)
                if self.config.documents:
                    for name, doc in self.config.documents.items():
                        source_keywords.append(name.lower().replace("_", " "))
                        source_keywords.append(name.lower())
                        if doc.description:
                            for word in doc.description.lower().split():
                                if len(word) > 4 and word.isalpha():
                                    source_keywords.append(word)

                # Check for matches
                for keyword in source_keywords:
                    if keyword in problem_lower:
                        logger.debug(f"Overriding GENERAL_KNOWLEDGE to DATA_ANALYSIS: query mentions '{keyword}'")
                        question_type = QuestionType.DATA_ANALYSIS
                        break

            # Parse CACHED_ANSWER
            if "CACHED_ANSWER:" in response:
                answer_section = response.split("CACHED_ANSWER:", 1)[1].split("---")[0].strip()
                if answer_section and answer_section.upper() != "NONE":
                    cached_answer = answer_section

            # Parse INTENTS (preserving order)
            _intent_logger.debug(f"[INTENT PARSING] Starting intent parsing, response length: {len(response)}")
            intents = []
            if "INTENTS:" in response:
                intents_section = response.split("INTENTS:", 1)[1].split("---")[0].strip()
                _intent_logger.debug(f"[INTENT PARSING] intents_section: '{intents_section}'")
                if intents_section and intents_section.upper() != "NONE":
                    for line in intents_section.split("\n"):
                        line = line.strip().lstrip("-").strip()
                        _intent_logger.debug(f"[INTENT PARSING] processing line: '{line}'")
                        if line and line.upper() != "NONE":
                            # Parse "INTENT_NAME | extracted_value" or just "INTENT_NAME"
                            if "|" in line:
                                intent_name, extracted = line.split("|", 1)
                                intent_name = intent_name.strip().upper()
                                extracted = extracted.strip()
                            else:
                                intent_name = line.strip().upper()
                                extracted = None
                            _intent_logger.debug(f"[INTENT PARSING] extracted intent: '{intent_name}'")
                            intents.append(DetectedIntent(
                                intent=intent_name,
                                confidence=0.8,
                                extracted_value=extracted,
                            ))
                else:
                    _intent_logger.debug(f"[INTENT PARSING] intents_section was empty or NONE")
            else:
                _intent_logger.debug(f"[INTENT PARSING] 'INTENTS:' not found in response")

            # Default to NEW_QUESTION if no intents detected
            if not intents:
                intents.append(DetectedIntent(intent="NEW_QUESTION", confidence=0.5))

            # Parse FACT_MODIFICATIONS
            fact_modifications = []
            if "FACT_MODIFICATIONS:" in response:
                mods_section = response.split("FACT_MODIFICATIONS:", 1)[1].split("---")[0].strip()
                if mods_section and mods_section.upper() != "NONE":
                    for line in mods_section.split("\n"):
                        line = line.strip().lstrip("-").strip()
                        if ":" in line and line.upper() != "NONE":
                            fact_name, new_value = line.split(":", 1)
                            fact_modifications.append({
                                "fact_name": fact_name.strip(),
                                "new_value": new_value.strip(),
                                "action": "modify",
                            })

            # Parse SCOPE_REFINEMENTS
            scope_refinements = []
            if "SCOPE_REFINEMENTS:" in response:
                scope_section = response.split("SCOPE_REFINEMENTS:", 1)[1].split("---")[0].strip()
                if scope_section and scope_section.upper() != "NONE":
                    for line in scope_section.split("\n"):
                        line = line.strip().lstrip("-").strip()
                        if line and line.upper() != "NONE":
                            scope_refinements.append(line)

            # Parse WANTS_BRIEF
            wants_brief = False
            if "WANTS_BRIEF:" in response:
                brief_section = response.split("WANTS_BRIEF:", 1)[1].split("---")[0].strip()
                wants_brief = brief_section.upper().startswith("YES")

            # Parse EXECUTION_MODE and MODE_REASON
            recommended_mode = "EXPLORATORY"  # Default
            mode_reasoning = None
            if "EXECUTION_MODE:" in response:
                mode_section = response.split("EXECUTION_MODE:", 1)[1].split("\n")[0].strip()
                mode_section = mode_section.split("---")[0].strip().upper()
                if "PROOF" in mode_section or "AUDITABLE" in mode_section:
                    recommended_mode = "PROOF"
                elif "EXPLORATORY" in mode_section:
                    recommended_mode = "EXPLORATORY"
            if "MODE_REASON:" in response:
                reason_section = response.split("MODE_REASON:", 1)[1].split("---")[0].strip()
                if reason_section and reason_section.upper() != "NONE":
                    mode_reasoning = reason_section.split("\n")[0].strip()

            _intent_logger.debug(f"[INTENT PARSING] Final parsed intents: {[i.intent for i in intents]}")

            # Store intent in datastore for debugging (if method exists)
            if self.datastore and hasattr(self.datastore, 'set_query_intent'):
                self.datastore.set_query_intent(
                    query_text=problem,
                    intents=[{"intent": i.intent, "confidence": i.confidence, "value": i.extracted_value} for i in intents],
                    is_followup=bool(self.session_id),
                )

            return QuestionAnalysis(
                question_type=question_type,
                extracted_facts=extracted_facts,
                cached_fact_answer=cached_answer,
                intents=intents,
                fact_modifications=fact_modifications,
                scope_refinements=scope_refinements,
                wants_brief=wants_brief,
                recommended_mode=recommended_mode,
                mode_reasoning=mode_reasoning,
            )

        except Exception as e:
            # On error, fall back to regex-based classification
            _intent_logger.exception(f"[INTENT PARSING] Exception during question analysis: {e}")
            return QuestionAnalysis(
                question_type=QuestionType.META_QUESTION if is_meta_question(problem) else QuestionType.DATA_ANALYSIS,
                extracted_facts=[],
                cached_fact_answer=None,
                intents=[DetectedIntent(intent="NEW_QUESTION", confidence=0.5)],
            )

    def _detect_ambiguity(self, problem: str, is_auditable_mode: bool = False, session_tables: Optional[list[dict]] = None) -> Optional[ClarificationRequest]:
        """
        Detect if a question is ambiguous and needs clarification before planning.

        Checks for missing parameters like:
        - Geographic scope ("how many bears" - where?)
        - Time period ("what were sales" - when?)
        - Threshold values ("top customers" - top how many?)
        - Category/segment ("product performance" - which products?)

        Args:
            problem: The user's question
            is_auditable_mode: If True, defer personal value questions to lazy resolution
            session_tables: List of existing session tables (from datastore.list_tables())

        Returns:
            ClarificationRequest if clarification needed, None otherwise
        """
        from constat.storage.learnings import LearningCategory

        ctx = self._build_source_context()

        # Inject user correction and NL correction learnings so ambiguity detection
        # respects past corrections (e.g. "use business_rules not compensation_policy")
        learnings_text = ""
        if self.learning_store:
            try:
                rule_lines = []
                for category in [LearningCategory.USER_CORRECTION, LearningCategory.NL_CORRECTION]:
                    rules = self.learning_store.list_rules(
                        category=category,
                        min_confidence=0.6,
                    )
                    for rule in rules[:3]:
                        rule_lines.append(f"- {rule['summary']}")
                if rule_lines:
                    learnings_text = "\n## Learned Rules (respect these when suggesting options)\n" + "\n".join(rule_lines)
            except (OSError, KeyError, ValueError):
                pass

        personal_values_guidance = (
            "NEVER ask for personal VALUES like age, salary, preferences - these will be requested later during fact resolution. The user explicitly referenced 'my age' means they intend to provide it - don't pre-ask."
            if is_auditable_mode else
            "For personal values mentioned (like 'my age'), you MAY ask since exploratory mode needs all values upfront."
        )
        # Format session tables so the LLM knows what datasets already exist
        session_tables_text = ""
        if session_tables:
            table_lines = ["\n## Session Tables (datasets created during this conversation)"]
            for t in session_tables:
                table_lines.append(f"- `{t['name']}`: {t.get('row_count', '?')} rows (step {t.get('step_number', '?')})")
            session_tables_text = "\n".join(table_lines)

        prompt = load_prompt("detect_ambiguity.md").format(
            problem=problem,
            schema_overview=ctx["schema_overview"],
            api_overview=ctx["api_overview"],
            doc_overview=ctx["doc_overview"],
            user_facts=ctx["user_facts"],
            learnings_text=learnings_text,
            personal_values_guidance=personal_values_guidance,
            session_tables=session_tables_text,
        )

        try:
            result = self.router.execute(
                task_type=TaskType.INTENT_CLASSIFICATION,
                system="You detect ambiguity in data analysis requests. Be practical - only flag truly ambiguous requests.",
                user_message=prompt,
                max_tokens=self.router.max_output_tokens,
            )

            response = result.content.strip()

            if response.startswith("CLEAR"):
                return None

            # Parse ambiguous response
            if "AMBIGUOUS" in response:
                lines = response.split("\n")
                reason = ""
                questions: list[ClarificationQuestion] = []
                current_question = None
                in_questions_section = False

                for line in lines:
                    line = line.strip()
                    if line.startswith("REASON:"):
                        reason = line[7:].strip()
                    elif line.upper().startswith("QUESTIONS"):
                        in_questions_section = True
                    elif line.startswith("SUGGESTIONS:") and current_question:
                        # Parse suggestions for current question
                        suggestions_text = line[12:].strip()
                        suggestions = [s.strip() for s in suggestions_text.split("|") if s.strip()]
                        current_question.suggestions = suggestions[:4]  # Max 4 suggestions
                    elif in_questions_section and line:
                        # Try to parse as a question in specific formats only:
                        # Q1: question, - question, 1. question, 1) question
                        # Do NOT capture arbitrary text as questions (could be LLM reasoning)
                        question_text = None

                        if line.startswith("Q") and ":" in line[:4]:
                            # Format: Q1: question text
                            question_text = line.split(":", 1)[1].strip()
                        elif line.startswith("- "):
                            # Format: - question text
                            question_text = line[2:].strip()
                        elif len(line) > 2 and line[0].isdigit() and line[1] in ".):":
                            # Format: 1. question or 1) question or 1: question
                            question_text = line[2:].strip()
                        elif len(line) > 3 and line[:2].isdigit() and line[2] in ".):":
                            # Format: 10. question (two-digit number)
                            question_text = line[3:].strip()
                        # NOTE: We intentionally do NOT capture arbitrary text as questions
                        # The LLM sometimes adds explanatory text that shouldn't be treated as questions

                        # Only accept if it looks like a question (ends with ? or starts with question word)
                        if question_text and len(question_text) > 5:
                            is_question = (
                                question_text.endswith("?") or
                                question_text.lower().startswith(("what ", "which ", "how ", "when ", "where ", "who ", "should ", "do ", "does ", "is ", "are "))
                            )
                            if is_question:
                                # Save previous question and start new one
                                if current_question and current_question.text:
                                    questions.append(current_question)
                                current_question = ClarificationQuestion(text=question_text)

                # Don't forget the last question
                if current_question and current_question.text:
                    questions.append(current_question)

                if questions:
                    return ClarificationRequest(
                        original_question=problem,
                        ambiguity_reason=reason,
                        questions=questions[:3],  # Max 3 questions
                    )

            return None

        except Exception as e:
            # On error, proceed without clarification
            logger.debug(f"Clarification detection failed (proceeding without): {e}")
            return None

    def _request_clarification(self, request: ClarificationRequest) -> Optional[str]:
        """
        Request clarification from the user.

        Args:
            request: The clarification request

        Returns:
            Enhanced question with clarification, or None to skip
        """
        # Skip if disabled or no callback
        if self.session_config.skip_clarification:
            return None

        if not self._clarification_callback:
            return None

        # Emit event for UI
        self._emit_event(StepEvent(
            event_type="clarification_needed",
            step_number=0,
            data={
                "reason": request.ambiguity_reason,
                "questions": request.questions,
            }
        ))

        response = self._clarification_callback(request)

        if response.skip:
            return None

        # Build enhanced question with clarifications
        clarifications = []
        logger.debug(f"[CLARIFICATION] Response answers: {response.answers}")
        for question, answer in response.answers.items():
            logger.debug(f"[CLARIFICATION] Q: {question!r} -> A: {answer!r}")
            if answer:
                clarifications.append(f"{question}: {answer}")

        if clarifications:
            enhanced = f"{request.original_question}\n\nClarifications:\n" + "\n".join(clarifications)
            logger.debug(f"[CLARIFICATION] Enhanced problem:\n{enhanced}")
            return enhanced

        return None

    def _answer_general_question(self, problem: str) -> dict:
        """
        Answer a general knowledge question directly using LLM.
        """
        result = self.router.execute(
            task_type=TaskType.GENERAL,
            system="You are a helpful assistant. Answer the question directly and concisely.",
            user_message=problem,
        )

        return {
            "success": True,
            "meta_response": True,  # Reuse this flag to skip planning display
            "output": result.content,
            "plan": None,
        }

    def _answer_from_cached_facts(self, problem: str) -> Optional[dict]:
        """
        Try to answer a question from cached facts.

        Checks if the question references a fact already in the cache
        (e.g., "what is my role" -> user_role fact).

        Returns:
            Answer dict if fact found, None otherwise
        """
        cached_facts = self.fact_resolver.get_all_facts()
        if not cached_facts:
            return None

        # Create context about available facts (use display_value for table references)
        fact_context = "\n".join(
            f"- {name}: {fact.display_value}" for name, fact in cached_facts.items()
        )

        prompt = f"""Check if this question can be answered from these known facts:

Known facts:
{fact_context}

User question: {problem}

If the question asks about one of these facts, respond with:
FACT_MATCH: <fact_name>
ANSWER: <direct answer using the fact value>

If the question cannot be answered from these facts, respond with:
NO_MATCH

Examples:
- "what is my role" + fact user_role=CFO -> FACT_MATCH: user_role, ANSWER: Your role is CFO.
- "what's the target region" + fact target_region=US -> FACT_MATCH: target_region, ANSWER: The target region is US.
- "how many customers" + no matching fact -> NO_MATCH
"""

        try:
            result = self.router.execute(
                task_type=TaskType.GENERAL,
                system="You are a helpful assistant matching questions to known facts.",
                user_message=prompt,
                max_tokens=self.router.max_output_tokens,
            )

            response = result.content.strip()
            if "NO_MATCH" in response:
                return None

            # Extract the answer
            if "ANSWER:" in response:
                answer_start = response.index("ANSWER:") + 7
                answer = response[answer_start:].strip()

                return {
                    "success": True,
                    "meta_response": True,
                    "output": answer,
                    "plan": None,
                }
        except Exception as e:
            logger.debug(f"Meta question handling failed: {e}")

        return None

    @staticmethod
    def _explain_differentiators() -> dict:
        """Explain what makes Constat different from other AI tools."""
        explanation = load_prompt("explain_differentiators.md")

        return {
            "success": True,
            "meta_response": True,
            "output": explanation,
            "suggestions": [
                "What data is available?",
                "How do you reason about problems?",
            ],
            "plan": None,
        }

    @staticmethod
    def _explain_reasoning_methodology() -> dict:
        """Explain Constat's reasoning methodology."""
        explanation = load_prompt("explain_reasoning_methodology.md")

        return {
            "success": True,
            "meta_response": True,
            "output": explanation,
            "suggestions": [
                "What data is available?",
                "Show me an example analysis",
            ],
            "plan": None,
        }

    @staticmethod
    def _answer_personal_question() -> dict:
        """Answer personal questions about Vera."""
        explanation = load_prompt("answer_personal_question.md")

        return {
            "success": True,
            "meta_response": True,
            "output": explanation,
            "suggestions": [
                "What data is available?",
                "How do you reason about problems?",
                "What makes you different, Vera?",
            ],
            "plan": None,
        }

    def _answer_meta_question(self, problem: str) -> dict:
        """
        Answer meta-questions about capabilities without planning/execution.

        Uses schema overview and domain context to answer questions like
        "what questions can you answer" directly.
        """
        # Check if asking about reasoning methodology
        problem_lower = problem.lower()
        if any(phrase in problem_lower for phrase in [
            "how do you reason", "how do you think", "how do you work",
            "reasoning process", "methodology", "how does this work"
        ]):
            return self._explain_reasoning_methodology()

        # Check if asking what makes Constat/Vera different
        if any(phrase in problem_lower for phrase in [
            "what makes", "what's different", "how is .* different",
            "unique about", "special about", "why constat", "why use constat",
            "why vera"
        ]):
            return self._explain_differentiators()

        # Check if asking personal questions about Vera
        if any(phrase in problem_lower for phrase in [
            "who are you", "what are you", "your name", "about you",
            "how old", "your age", "are you a ", "are you an ",
            "who made you", "who created you", "who built you",
            "tell me about yourself", "introduce yourself", "vera"
        ]):
            return self._answer_personal_question()

        ctx = self._build_source_context(include_user_facts=False)
        domain_context = self._get_system_prompt()

        # Get user role if known
        user_role = None
        try:
            role_fact = self.fact_resolver.get_fact("user_role")
            if role_fact:
                user_role = role_fact.value
        except Exception as e:
            logger.debug(f"Failed to get user_role fact: {e}")

        role_context = f"\nThe user's role is: {user_role}" if user_role else ""

        prompt = f"""The user is asking about your capabilities. Answer based on the available data.

User question: {problem}{role_context}

Available databases and tables:
{ctx["schema_overview"]}
{ctx["api_overview"]}
{ctx["doc_overview"]}

Domain context:
{domain_context}

Provide a helpful summary tailored to the user's role (if known):
1. What data sources are relevant to their role (databases, APIs, and reference documents)
2. What types of analyses would be most valuable

Then provide 3-6 example questions the user could ask, each on its own line in quotes like:
"What is the revenue by region?"

Keep it concise and actionable."""

        result = self.router.execute(
            task_type=TaskType.SUMMARIZATION,
            system="You are a helpful assistant explaining data analysis capabilities.",
            user_message=prompt,
        )

        # Extract example questions from output to use as suggestions
        # Don't emit event here - let REPL display after output
        suggestions = self._extract_example_questions(result.content)

        # Strip the example questions section from output to avoid duplication
        output = self._strip_example_questions_section(result.content)

        return {
            "success": True,
            "meta_response": True,
            "output": output,
            "suggestions": suggestions,
            "plan": None,
        }

    @staticmethod
    def _extract_example_questions(text: str) -> list[str]:
        """
        Extract example questions from meta-response text.

        Looks for quoted questions in the text that the user could ask.
        """
        import re
        questions = []

        # Look for questions in quotes (single or double)
        # Pattern: "question?" or 'question?'
        quoted_pattern = r'["\u201c]([^"\u201d]+\?)["\u201d]'
        matches = re.findall(quoted_pattern, text)
        for match in matches:
            q = match.strip()
            if len(q) > 10 and q not in questions:  # Skip very short matches
                questions.append(q)

        # Limit to 6 suggestions
        return questions[:6]

    @staticmethod
    def _strip_example_questions_section(text: str) -> str:
        """
        Strip the example questions section from meta-response output.

        This avoids duplicating questions that will be shown as suggestions.
        """
        import re

        # Find where example questions section starts and remove from there
        # Match various header formats
        patterns = [
            r'\n*Example Questions[^\n]*:\s*\n',  # "Example Questions You Could Ask:"
            r'\n*#+\s*Example Questions?[^\n]*\n',  # Markdown header
            r'\n*\*\*Example Questions?[^\n]*\*\*\s*\n',  # Bold header
        ]

        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                # Remove from the start of the example section to the end
                return text[:match.start()].rstrip()

        return text.rstrip()
