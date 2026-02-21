# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Synthesis mixin: answer synthesis, fact extraction, suggestions."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from constat.core.models import TaskType
from constat.execution.fact_resolver import FactSource
from constat.prompts import load_prompt

if TYPE_CHECKING:
    from constat.core.models import PlannerResponse

logger = logging.getLogger(__name__)


# noinspection PyUnresolvedReferences
class SynthesisMixin:

    def _synthesize_answer(self, problem: str, step_outputs: str, artifacts: list[dict] | None = None) -> str:
        """
        Synthesize a final user-facing answer from step execution outputs.

        This takes the raw step outputs (which may be verbose technical details)
        and creates a clear, direct answer to the user's original question.
        """
        # Build artifact context if files were created
        artifact_context = ""
        if artifacts:
            artifact_lines = ["Files created:"]
            for a in artifacts:
                desc = a.get("description", "")
                uri = a.get("file_uri", "")
                if uri:
                    filename = uri.split("/")[-1]
                    artifact_lines.append(f"- `{filename}`: {desc}")
            artifact_context = "\n" + "\n".join(artifact_lines) + "\n"

        prompt = f"""Synthesize a final insight from this data analysis.

Question: {problem}

[Step outputs for context - DO NOT repeat these, user already saw them]
{step_outputs}
{artifact_context}
IMPORTANT: The user has ALREADY SEEN all the step-by-step output above in separate messages. Your job is ONLY to provide high-level analysis - NOT to summarize what each step did.

Write a brief insight (max 150 words) with:

1. **Answer**: Direct 1-sentence answer to their question
2. **Key Insight**: The most important finding or pattern (not a list of what steps did)
3. **Next Steps**: 2-3 follow-up questions they could ask (formatted as a numbered list)

WRONG (do not do this):
- "Step 1 loaded the data, Step 2 filtered it, Step 3 calculated..."
- "First I analyzed X, then I computed Y..."

RIGHT:
- "The top 3 customers account for 60% of revenue. This concentration suggests..."
- "Revenue grew 15% YoY, driven primarily by the Enterprise segment..."

Reference tables with backticks: `table_name`"""

        result = self.router.execute(
            task_type=TaskType.SUMMARIZATION,
            system="You are a data analyst presenting findings. Be clear and direct.",
            user_message=prompt,
            max_tokens=self.router.max_output_tokens,
        )

        if not result.success:
            logger.warning(f"Answer synthesis failed: {result.content}")
            return "Analysis completed but answer synthesis failed. See step outputs above."
        return result.content

    def _extract_facts_from_response(self, problem: str, answer: str) -> list:
        """
        Extract facts from the analysis response to cache for follow-up questions.

        For example, if the answer says "Total revenue was $2.4M", we cache
        the fact `total_revenue = 2400000` so follow-up questions like
        "How does that compare to last year?" can reference it.

        Returns:
            List of extracted Fact objects
        """
        prompt = load_prompt("extract_facts_from_response.md").format(
            problem=problem,
            answer=answer,
        )

        try:
            result = self.router.execute(
                task_type=TaskType.INTENT_CLASSIFICATION,
                system="You extract key facts and metrics from analysis results.",
                user_message=prompt,
                max_tokens=self.router.max_output_tokens,
            )

            response = result.content.strip()
            if "NO_FACTS" in response:
                return []

            extracted_facts = []
            for line in response.split("\n"):
                line = line.strip()
                if line == "---" or not line:
                    continue
                if ":" in line and not line.startswith("FACT"):
                    parts = line.split(":", 1)
                    fact_name = parts[0].strip().lower().replace(" ", "_")
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
                        # Handle currency (remove $, commas)
                        clean_value = value_str.replace("$", "").replace(",", "").replace("%", "")
                        value = float(clean_value)
                        if "%" in value_str:
                            value = value / 100  # Convert percentage
                        elif value == int(value):
                            value = int(value)
                    except ValueError:
                        value = value_str

                    # Add to fact resolver - source is DERIVED since synthesized from analysis
                    fact = self.fact_resolver.add_user_fact(
                        fact_name=fact_name,
                        value=value,
                        reasoning=f"Extracted from exploratory analysis. Run in auditable mode for full provenance.",
                        source=FactSource.DERIVED,
                        description=description,
                    )
                    extracted_facts.append(fact)

            return extracted_facts

        except Exception as e:
            logger.debug(f"Failed to extract facts from response: {e}")
            return []

    def _generate_suggestions(self, problem: str, answer: str, tables: list[dict]) -> list[str]:
        """
        Generate contextual follow-up suggestions based on the answer and available data.

        Args:
            problem: The original question
            answer: The synthesized answer
            tables: Available tables in the datastore

        Returns:
            List of 1-3 suggested follow-up questions
        """
        table_info = ", ".join(t["name"] for t in tables) if tables else "none"

        prompt = f"""Based on this completed analysis, suggest 1-3 actionable follow-up requests the user could make.

Original question: {problem}

Answer provided:
{answer}

Available data tables: {table_info}

Guidelines:
- Suggest ACTIONABLE REQUESTS that extend or build on the analysis (e.g., "Show a breakdown by region", "Compare this to last quarter")
- DO NOT ask clarifying questions back to the user (e.g., "Why did you need this?" or "What will you use this for?")
- Each suggestion should be something the system can execute
- Keep suggestions concise (under 12 words each)
- Consider: breakdowns, comparisons, visualizations, exports, time periods, rankings
- If the analysis seems complete, return just 1 suggestion or nothing

Return ONLY the suggestions, one per line, no numbering or bullets."""

        try:
            result = self.router.execute(
                task_type=TaskType.SUMMARIZATION,
                system="You suggest actionable follow-up analysis requests. Never ask clarifying questions.",
                user_message=prompt,
                max_tokens=self.router.max_output_tokens,
            )

            # Parse suggestions (one per line)
            suggestions = [
                s.strip().lstrip("0123456789.-) ")
                for s in result.content.strip().split("\n")
                if s.strip() and len(s.strip()) > 5
            ]
            return suggestions[:3]  # Max 3 suggestions
        except Exception as e:
            logger.debug(f"Failed to generate suggestions (non-fatal): {e}")
            return []

    def _run_post_synthesis_parallel(
        self,
        problem: str,
        final_answer: str,
        tables: list[dict],
    ) -> tuple[list, list[str]]:
        """
        Run facts extraction and suggestions generation in parallel.

        These two tasks are independent and can run concurrently to reduce
        total synthesis time from ~3 LLM calls to ~2 (synthesis + max(facts, suggestions)).

        Args:
            problem: The original question
            final_answer: The synthesized answer
            tables: Available tables for suggestions context

        Returns:
            Tuple of (extracted_facts, suggestions)
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        response_facts = []
        suggestions = []

        def extract_facts():
            return self._extract_facts_from_response(problem, final_answer)

        def generate_suggestions():
            return self._generate_suggestions(problem, final_answer, tables)

        with ThreadPoolExecutor(max_workers=2) as executor:
            facts_future = executor.submit(extract_facts)
            suggestions_future = executor.submit(generate_suggestions)

            # Wait for both to complete
            for future in as_completed([facts_future, suggestions_future]):
                try:
                    if future == facts_future:
                        response_facts = future.result()
                    else:
                        suggestions = future.result()
                except Exception as e:
                    logger.debug(f"Post-synthesis task failed (non-fatal): {e}")

        return response_facts, suggestions

    def _replan_with_feedback(self, problem: str, feedback: str) -> PlannerResponse:
        """
        Generate a new plan incorporating user feedback.

        Feedback is appended verbatim to the problem, similar to how clarifications
        are handled. This preserves the exact user input for the planner.

        Args:
            problem: Original problem
            feedback: User's suggested changes (passed verbatim)

        Returns:
            New PlannerResponse with updated plan
        """
        # Append feedback verbatim, similar to clarifications format
        enhanced_problem = f"{problem}\n\nPlan Adjustments:\n{feedback}"

        self._sync_user_facts_to_planner()
        self._sync_available_agents_to_planner()
        return self.planner.plan(enhanced_problem)
