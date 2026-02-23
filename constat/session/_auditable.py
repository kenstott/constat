# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.
"""Auditable mixin: _solve_auditable and steer handling."""
from __future__ import annotations

import logging
import re

from constat.core.models import TaskType
from constat.execution.mode import Mode
from constat.session._types import StepEvent

logger = logging.getLogger(__name__)


# noinspection PyUnresolvedReferences
class AuditableMixin:

    def _solve_auditable_with_steer_handling(
        self, problem: str, cached_fact_hints: list[dict] = None
    ) -> dict:
        """
        Wrapper for _solve_auditable that handles user steers.

        If a user provides guidance (steer) instead of a value when asked for
        a premise, this re-plans with the steer added to the problem context.

        Args:
            problem: The original problem
            cached_fact_hints: Optional cached fact hints

        Returns:
            Result dict from _solve_auditable
        """
        max_steer_attempts = 3
        steer_attempt = 0
        augmented_problem = problem

        while steer_attempt < max_steer_attempts:
            result = self._solve_auditable(augmented_problem, cached_fact_hints)

            # Check if we need to re-plan due to user steer
            if result.get("status") == "replan_needed":
                steer_attempt += 1
                steer = result.get("steer", "")
                fact_name = result.get("fact_name", "")
                logger.debug(f"Re-planning with steer: {steer}")

                # Augment the problem with the user's guidance
                augmented_problem = f"{problem}\n\nAdditional context: For {fact_name}, {steer}"

                # Clear cached plan to force re-planning
                self._last_plan = None

                # Emit event so user sees the message
                self._emit_event(StepEvent(
                    event_type="steer_detected",
                    step_number=0,
                    data={"message": result.get("message", "Re-planning with your guidance...")}
                ))
                continue

            return result

        # Max attempts reached
        return {"error": "Max re-planning attempts reached", "status": "failed"}

    def _solve_auditable(self, problem: str, cached_fact_hints: list[dict] = None) -> dict:
        """
        Solve a problem in auditable mode using fact-based derivation.

        Instead of generating a stepwise execution plan, this method:
        1. Identifies the question to answer
        2. Decomposes into required premises (facts from sources)
        3. Shows a derivation plan for approval
        4. Resolves facts with provenance tracking
        5. Generates an auditable derivation trace

        Args:
            problem: The problem/question to solve
            cached_fact_hints: Optional list of cached facts from previous run (for redo)
                               Each dict has 'name', 'value', 'value_type' keys

        Returns:
            Dict with derivation trace and verification result
        """
        import time
        start_time = time.time()

        # Save mode to datastore for follow-up handling
        if self.datastore:
            self.datastore.set_session_meta("mode", "auditable")

        # Step 1: Generate fact-based plan (identify required facts)
        self._emit_event(StepEvent(
            event_type="planning_start",
            step_number=0,
            data={"message": "Identifying required facts for verification..."}
        ))

        # Get source context for the planner
        # Proof planner is a single LLM call (no tool use), so it needs full table schemas
        # _build_source_context may only return database names without tables
        ctx = self._build_source_context(include_user_facts=False)
        # Override schema_overview with full schema if brief summary lacks table names
        if self.schema_manager and "Table:" not in ctx.get("schema_overview", ""):
            ctx["schema_overview"] = self.schema_manager.get_overview()

        # Build hint about cached facts for redo (helps LLM use consistent names)
        cached_facts_hint = ""
        if cached_fact_hints:
            # Separate tables from scalar facts
            cached_tables = [f for f in cached_fact_hints if f.get("value_type") == "table"]
            cached_scalars = [f for f in cached_fact_hints if f.get("value_type") != "table"]

            hint_lines = [
                "CACHED DATA AVAILABLE AS INPUT - Use these as premises with [source: cache]:",
            ]
            if cached_tables:
                hint_lines.append("\nCached tables:")
                for fact in cached_tables:
                    name = fact.get("name", "")
                    desc = fact.get("description", "")
                    hint_lines.append(f"  - \"{name}\" ({fact.get('row_count', '?')} rows) {desc}")
            if cached_scalars:
                hint_lines.append("\nCached values:")
                for fact in cached_scalars:
                    name = fact.get("name", "")
                    hint_lines.append(f"  - \"{name}\" = {fact.get('value', '?')}")
            hint_lines.append("")
            hint_lines.append("Use [source: cache] with EXACT names above. Do NOT rename them.")
            hint_lines.append("CRITICAL: You must still build a COMPLETE derivation chain that PROVES the answer.")
            hint_lines.append("CRITICAL: Do NOT just verify cached data exists - you must show HOW the answer is derived.")
            hint_lines.append("")
            cached_facts_hint = "\n".join(hint_lines) + "\n"

        # Extract document sources and step summaries from exploratory session
        exploratory_docs_hint = ""
        exploratory_steps_hint = ""
        step_hints = getattr(self, '_proof_step_hints', [])
        if step_hints:
            doc_names = set()
            step_code_blocks = []
            for step in step_hints:
                code = step.get("code", "")
                for m in re.findall(r"""doc_read\(\s*['"]([^'"]+)['"]\s*\)""", code):
                    doc_names.add(m)
                goal = step.get("goal", "")
                step_num = step.get("step_number", "?")
                if code:
                    step_code_blocks.append(f"# Step {step_num}: {goal}\n{code}")
            if doc_names:
                names_str = ", ".join(f'"{n}"' for n in sorted(doc_names))
                exploratory_docs_hint = f"DOCUMENT CONSTRAINT: The exploratory analysis used these documents: {names_str}. Use the SAME document sources for consistency.\n\n"
            if step_code_blocks:
                exploratory_steps_hint = (
                    "EXPLORATORY SESSION CODE (working code that produced the accepted answer — use as reference, not prescription. "
                    "Your proof should reason independently but make informed method choices):\n```python\n"
                    + "\n\n".join(step_code_blocks) + "\n```\n\n"
                )

        fact_plan_prompt = f"""Construct a logical derivation to answer this question with full provenance.

Question: {problem}

{cached_facts_hint}{exploratory_docs_hint}{exploratory_steps_hint}Available databases:
{ctx["schema_overview"]}
{ctx["doc_overview"]}
{ctx["api_overview"]}

Build a formal derivation with EXACTLY this format:

QUESTION: <restate the question>

PREMISES:
P1: <fact_name> = ? (<what data to retrieve>) [source: database:<db_name>]
P2: <fact_name> = ? (<description>) [source: api:<api_name>]
P3: <fact_name> = ? (<description>) [source: document:<doc_name>]
P4: <fact_name> = <known_value> (<description>) [source: llm_knowledge]

PREMISE RULES:
- Premises are DATA only (tables, records, values) - NOT functions or operations
- Every premise MUST be referenced by at least one inference
- Use "cache" for data already in cache (PREFERRED - fastest)
- Use "database" for SQL queries to configured databases
- Use "api" for external API data (GraphQL or REST endpoints)
- Use "document:<doc_name>" for reference documents, policies, and guidelines (e.g., [source: document:business_rules])
- Use "llm_knowledge" for universal facts (mathematical constants, scientific facts) and well-established reference data (ISO codes, country info, currency codes)
- For known universal values, embed directly: P2: pi_value = 3.14159 (Pi constant) [source: llm_knowledge]
- NEVER ASSUME personal values (age, location, preferences) - use [source: user] and leave value as ?
- Example: P4: my_age = ? (User's age) [source: user]
- IMPORTANT: If cached data is available for what you need, ALWAYS use [source: cache] instead of fetching from database/api again.
- IMPORTANT: If the question mentions clarifications or user preferences (like "use guidelines from X"), treat these as DATA to be retrieved, NOT as embedded values. Always use = ? and resolve from the appropriate source.
- IMPORTANT: If a configured API can provide the data, use [source: api:<name>] instead of llm_knowledge.
- IMPORTANT: Extract numeric constraints from the question as premises. Example: "top 5 results" becomes P2: limit_count = 5 (Requested limit) [source: user]

INFERENCE:
I1: <result_name> = <operation>(P1, P2) -- <explanation>
I2: <result_name> = <operation>(I1) -- <explanation>

INFERENCE RULES:
- Each inference must reference at least one premise (P1/P2/etc) or prior inference (I1/I2/etc)
- CRITICAL: ALL premises MUST be used in the inference chain. Never define a premise that isn't referenced.
- CRITICAL: Each inference result_name MUST be GLOBALLY UNIQUE. NEVER reuse any name. BAD: two inferences both named "data_verified". GOOD: "validation_result" then "final_verification".
- CRITICAL: The final inference(s) should COMPUTE THE ACTUAL ANSWER, not just verify data exists. If the user asks for recommendations, calculate them. If they ask for comparisons, compute them.
- CRITICAL: Only ONE verify_exists() at the very end, referencing the computed answer. Do NOT add validate() or verify() steps before it.
- CRITICAL: Do NOT add intermediate analysis steps that are not required by the question. If the question asks to "match X to Y", plan ONE inference that does the matching — do NOT plan separate steps to "analyze characteristics", "classify categories", "score complexity" etc. unless explicitly requested.
- CRITICAL: Each inference should produce data that the NEXT inference actually needs. Do NOT generate intermediate datasets that are never consumed downstream.
- Operations like date extraction, filtering, grouping belong HERE, not in premises
- Keep operations simple: filter, join, group_sum, count, apply_rules, calculate, etc.
- For FUZZY MAPPING (e.g., free-text product names → breed names): plan the inference to first attempt mapping via data sources (APIs, databases), then use llm_map(values, allowed_list, source_desc, target_desc) as fallback for unmatched values. This reduces confidence but enables mapping when no exact data source exists.

CONCLUSION:
C: <final sentence describing what the final inference contains - use ENGLISH NAMES not I1/I2 references>
IMPORTANT: In the conclusion, ALWAYS use the English result_name (e.g., "raise_recommendations") NOT the ID (e.g., "I4")

EXAMPLE 1 - "What is revenue multiplied by Pi?":

PREMISES:
P1: orders = ? (All orders with amounts) [source: database:sales_db]
P2: pi_value = 3.14159 (Mathematical constant) [source: llm_knowledge]

INFERENCE:
I1: total_revenue = sum(P1.amount) -- Sum all order amounts
I2: adjusted_revenue = multiply(I1, P2) -- Multiply by Pi

CONCLUSION:
C: The revenue multiplied by Pi is provided in adjusted_revenue, calculated by multiplying total_revenue by pi_value.

EXAMPLE 2 - "Monthly revenue trend for last 12 months":

PREMISES:
P1: orders = ? (All orders with date and amount) [source: database:sales_db]

INFERENCE:
I1: recent_orders = filter(P1, last_12_months) -- Filter to last 12 months
I2: monthly_revenue = group_sum(I1, month, amount) -- Group by month, sum amounts
I3: trend = analyze(I2) -- Calculate trend direction

CONCLUSION:
C: Monthly revenue trend is provided in trend, showing direction based on monthly_revenue analysis.

EXAMPLE 3 - "Recommend raises based on performance reviews and guidelines":

PREMISES:
P1: employees = ? (All employees with current salary) [source: database:hr]
P2: performance_reviews = ? (Performance review records with ratings) [source: database:hr]
P3: raise_guidelines = ? (Business rules for raise percentages by rating) [source: document]

INFERENCE:
I1: recent_reviews = filter(P2, most_recent_per_employee) -- Get most recent review per employee
I2: employee_data = join(P1, I1, employee_id) -- Join employees with their reviews
I3: raises_with_rules = apply_guidelines(I2, P3) -- Apply raise guidelines based on rating (NOTE: P3 is USED here)
I4: raise_recommendations = calculate(I3, salary * raise_percentage) -- Calculate actual raise amounts

CONCLUSION:
C: Raise recommendations with calculated amounts are provided in raise_recommendations, derived by applying raise_guidelines to employee performance ratings.

Now generate the derivation. Use P1:, P2:, I1:, I2: prefixes EXACTLY as shown.
Premises are DATA. Operations (filter, extract, group, apply_guidelines) go in INFERENCE.
IMPORTANT: ALL premises must appear in at least one inference. The final inference must compute the answer.
"""

        result = self.router.execute(
            task_type=TaskType.INTENT_CLASSIFICATION,
            system="You analyze questions and decompose them into premises and inferences for auditable answers.",
            user_message=fact_plan_prompt,
            max_tokens=self.router.max_output_tokens,
        )
        # noinspection DuplicatedCode
        fact_plan_text = result.content

        # Parse the proof structure
        claim = ""
        premises = []  # P1, P2, ... - base facts from sources
        inferences = []  # I1, I2, ... - derived facts
        conclusion = ""

        lines = fact_plan_text.split("\n")
        current_section = None
        for line in lines:
            line = line.strip()
            if line.startswith("QUESTION:"):
                claim = line.split("QUESTION:", 1)[1].strip()
            elif line.startswith("PREMISES:"):
                current_section = "premises"
            elif line.startswith("INFERENCE:"):
                current_section = "inference"
            elif line.startswith("CONCLUSION:"):
                current_section = "conclusion"
            elif current_section == "premises" and re.match(r'^P\d+:', line):
                # Parse: P1: fact_name = ? (description) [source: xxx]
                # Also handle: P1: fact_name = ? (description) without source
                match = re.match(r'^(P\d+):\s*(.+?)\s*=\s*\?\s*\(([^)]+)\)\s*(?:\[source:\s*([^\]]+)\])?', line)
                if match:
                    premises.append({
                        "id": match.group(1),
                        "name": match.group(2).strip(),
                        "description": match.group(3).strip(),
                        "source": match.group(4).strip() if match.group(4) else "database",
                    })
                else:
                    # Try format with embedded value: P1: fact_name = 8 (description) [source: llm_knowledge]
                    value_match = re.match(r'^(P\d+):\s*(.+?)\s*=\s*([^\s(]+)\s*\(([^)]+)\)\s*(?:\[source:\s*([^\]]+)\])?', line)
                    if value_match:
                        # Include the value in the name so embedded value extraction works
                        fact_name = value_match.group(2).strip()
                        embedded_val = value_match.group(3).strip()
                        premises.append({
                            "id": value_match.group(1),
                            "name": f"{fact_name} = {embedded_val}",  # Include value for extraction
                            "description": value_match.group(4).strip(),
                            "source": value_match.group(5).strip() if value_match.group(5) else "knowledge",
                        })
                    else:
                        # Try simpler format: P1: fact_name (description)
                        simple_match = re.match(r'^(P\d+):\s*(.+?)\s*\(([^)]+)\)', line)
                        if simple_match:
                            premises.append({
                                "id": simple_match.group(1),
                                "name": simple_match.group(2).strip().rstrip('=?').strip(),
                                "description": simple_match.group(3).strip(),
                                "source": "database",
                            })
            elif current_section == "inference" and re.match(r'^I\d+:', line):
                # Parse: I1: derived_fact = operation(inputs) -- explanation
                match = re.match(r'^(I\d+):\s*(.+?)\s*=\s*(.+?)\s*--\s*(.+)$', line)
                if match:
                    inferences.append({
                        "id": match.group(1),
                        "name": match.group(2).strip(),
                        "operation": match.group(3).strip(),
                        "explanation": match.group(4).strip(),
                    })
                else:
                    # Simpler format without operation details
                    simple_match = re.match(r'^(I\d+):\s*(.+)$', line)
                    if simple_match:
                        inferences.append({
                            "id": simple_match.group(1),
                            "name": "",
                            "operation": simple_match.group(2).strip(),
                            "explanation": "",
                        })
            elif current_section == "conclusion" and line:
                if line.startswith("C:"):
                    conclusion = line.split("C:", 1)[1].strip()
                elif not conclusion:
                    conclusion = line

        # Validate plan structure BEFORE execution
        # This catches invalid references, unused premises, duplicates, etc.
        from constat.execution.dag import validate_proof_plan

        max_validation_retries = 3
        for validation_attempt in range(1, max_validation_retries + 1):
            # Emit validation start event
            self._emit_event(StepEvent(
                event_type="plan_validating",
                step_number=0,
                data={
                    "attempt": validation_attempt,
                    "max_attempts": max_validation_retries,
                    "premises_count": len(premises),
                    "inferences_count": len(inferences),
                }
            ))

            validation_result = validate_proof_plan(premises, inferences)

            if validation_result.valid:
                # Emit validation success
                self._emit_event(StepEvent(
                    event_type="plan_validated",
                    step_number=0,
                    data={
                        "attempt": validation_attempt,
                        "premises_count": len(premises),
                        "inferences_count": len(inferences),
                    }
                ))
                break  # Plan is valid, proceed

            # Plan has validation errors - emit detailed error event
            error_feedback = validation_result.format_for_retry()
            error_types = list(set(e.error_type for e in validation_result.errors))
            error_summary = ", ".join(error_types)

            self._emit_event(StepEvent(
                event_type="plan_validation_failed",
                step_number=0,
                data={
                    "attempt": validation_attempt,
                    "max_attempts": max_validation_retries,
                    "error_count": len(validation_result.errors),
                    "error_types": error_types,
                    "error_summary": error_summary,
                    "errors": [{"type": e.error_type, "fact_id": e.fact_id, "message": e.message}
                               for e in validation_result.errors],
                    "will_retry": validation_attempt < max_validation_retries,
                }
            ))

            if validation_attempt >= max_validation_retries:
                # Max retries reached - raise with helpful message
                raise ValueError(
                    f"Plan validation failed after {max_validation_retries} attempts.\n\n"
                    f"{error_feedback}\n\n"
                    f"Try rephrasing your question or reducing complexity."
                )

            # Emit retry event before regenerating
            self._emit_event(StepEvent(
                event_type="plan_regenerating",
                step_number=0,
                data={
                    "attempt": validation_attempt + 1,
                    "max_attempts": max_validation_retries,
                    "reason": error_summary,
                    "fixing_errors": error_types,
                }
            ))

            # Retry with explicit feedback about the errors
            retry_prompt = f"""{fact_plan_prompt}

{error_feedback}

REMEMBER:
- Each premise (P1, P2) and inference (I1, I2) must have a UNIQUE name
- ALL premises must be referenced in at least one inference operation
- Inference operations must only reference EXISTING premises (P1, P2, ...) or PRIOR inferences (I1 before I2)
- Do NOT reference facts that don't exist (e.g., P5 when only P1-P3 are defined)
"""
            retry_result = self.router.execute(
                task_type=TaskType.INTENT_CLASSIFICATION,
                system="You analyze questions and decompose them into premises and inferences for auditable answers. CRITICAL: Ensure all fact references are valid.",
                user_message=retry_prompt,
                max_tokens=self.router.max_output_tokens,
            )

            # Reparse the retried plan
            # noinspection DuplicatedCode
            fact_plan_text = retry_result.content
            claim = ""
            premises = []
            inferences = []
            conclusion = ""

            lines = fact_plan_text.split("\n")
            current_section = None
            for line in lines:
                line = line.strip()
                if line.startswith("QUESTION:"):
                    claim = line.split("QUESTION:", 1)[1].strip()
                elif line.startswith("PREMISES:"):
                    current_section = "premises"
                elif line.startswith("INFERENCE:"):
                    current_section = "inference"
                elif line.startswith("CONCLUSION:"):
                    current_section = "conclusion"
                elif current_section == "premises" and re.match(r'^P\d+:', line):
                    match = re.match(r'^(P\d+):\s*(.+?)\s*=\s*\?\s*\(([^)]+)\)\s*(?:\[source:\s*([^\]]+)\])?', line)
                    if match:
                        premises.append({
                            "id": match.group(1),
                            "name": match.group(2).strip(),
                            "description": match.group(3).strip(),
                            "source": match.group(4).strip() if match.group(4) else "database",
                        })
                    else:
                        value_match = re.match(r'^(P\d+):\s*(.+?)\s*=\s*([^\s(]+)\s*\(([^)]+)\)\s*(?:\[source:\s*([^\]]+)\])?', line)
                        if value_match:
                            fact_name = value_match.group(2).strip()
                            embedded_val = value_match.group(3).strip()
                            premises.append({
                                "id": value_match.group(1),
                                "name": f"{fact_name} = {embedded_val}",
                                "description": value_match.group(4).strip(),
                                "source": value_match.group(5).strip() if value_match.group(5) else "knowledge",
                            })
                        else:
                            simple_match = re.match(r'^(P\d+):\s*(.+?)\s*\(([^)]+)\)', line)
                            if simple_match:
                                premises.append({
                                    "id": simple_match.group(1),
                                    "name": simple_match.group(2).strip().rstrip('=?').strip(),
                                    "description": simple_match.group(3).strip(),
                                    "source": "database",
                                })
                elif current_section == "inference" and re.match(r'^I\d+:', line):
                    match = re.match(r'^(I\d+):\s*(.+?)\s*=\s*(.+?)\s*--\s*(.+)$', line)
                    if match:
                        inferences.append({
                            "id": match.group(1),
                            "name": match.group(2).strip(),
                            "operation": match.group(3).strip(),
                            "explanation": match.group(4).strip(),
                        })
                    else:
                        simple_match = re.match(r'^(I\d+):\s*(.+)$', line)
                        if simple_match:
                            inferences.append({
                                "id": simple_match.group(1),
                                "name": "",
                                "operation": simple_match.group(2).strip(),
                                "explanation": "",
                            })
                elif current_section == "conclusion" and line:
                    if line.startswith("C:"):
                        conclusion = line.split("C:", 1)[1].strip()
                    elif not conclusion:
                        conclusion = line

        # Detect collection-oriented queries (reports, lists, etc.)
        # These don't have explicit scalar comparisons - the implicit goal is "data exists"
        # (runs after validation loop so it uses the final parsed plan)
        comparison_keywords = [
            "greater than", "less than", "equal to", "equals",
            ">", "<", ">=", "<=", "==", "!=",
            "compare", "check if", "verify that", "prove that",
            "is positive", "is negative", "is zero",
        ]
        is_collection_query = True
        combined_text = (problem + " " + conclusion).lower()
        for keyword in comparison_keywords:
            if keyword.lower() in combined_text:
                is_collection_query = False
                break

        if inferences:
            last_op = inferences[-1].get("operation", "").lower()
            if any(kw in last_op for kw in ["compare", "check", "verify", ">", "<"]):
                is_collection_query = False

        # For collection queries, add implicit verification inference
        if is_collection_query and inferences:
            last_inf_id = inferences[-1]["id"]
            verify_id = f"I{len(inferences) + 1}"
            existing_names = {inf.get("name", "") for inf in inferences}
            verify_name = "final_data_verification"
            suffix = 1
            while verify_name in existing_names:
                verify_name = f"final_data_verification_{suffix}"
                suffix += 1
            inferences.append({
                "id": verify_id,
                "name": verify_name,
                "operation": f"verify_exists({last_inf_id})",
                "explanation": "Verify result has data (count > 0) to confirm derivation succeeded",
            })
            if not conclusion.lower().startswith("the data"):
                conclusion = f"The data is verified to exist with provenance: {conclusion}"

        # Emit planning complete
        total_steps = len(premises) + len(inferences) + 1  # +1 for conclusion
        self._emit_event(StepEvent(
            event_type="planning_complete",
            step_number=0,
            data={"steps": total_steps}
        ))

        # Build proof steps for display
        # Structure: Premises (resolve from sources) → Inferences (derive) → Conclusion
        proof_steps = []
        step_num = 1

        # Add premises as steps (these need to be resolved from sources)
        for p in premises:
            # Format: fact_name = ? (description) [source: xxx]
            proof_steps.append({
                "number": step_num,
                "goal": f"{p['name']} = ? ({p['description']}) [source: {p['source']}]",
                "depends_on": [],
                "type": "premise",
                "fact_id": p['id'],  # Keep P1/P2 id for execution reference
            })
            step_num += 1

        # Add inferences as steps (these depend on premises/prior inferences)
        premise_count = len(premises)
        for inf in inferences:
            # Format: derived_fact = operation -- explanation
            goal = inf['operation']
            if inf.get('name'):
                goal = f"{inf['name']} = {inf['operation']}"
            if inf.get('explanation'):
                goal += f" -- {inf['explanation']}"
            proof_steps.append({
                "number": step_num,
                "goal": goal,
                "depends_on": list(range(1, premise_count + 1)),  # Depends on all premises
                "type": "inference",
                "fact_id": inf['id'],  # Keep I1/I2 id for execution reference
            })
            step_num += 1

        # Add conclusion as final step
        all_prior_steps = list(range(1, step_num))
        proof_steps.append({
            "number": step_num,
            "goal": conclusion,
            "depends_on": all_prior_steps,
            "type": "conclusion",
        })

        # Store parsed derivation for later use
        self._current_proof = {
            "question": claim,  # The question being answered
            "premises": premises,
            "inferences": inferences,
            "conclusion": conclusion,
        }

        # Request approval if required
        # For auditable mode, we call the approval callback directly with proof_steps
        # that preserve the type and fact_id fields for proper P1:/I1:/C: display
        if self.session_config.require_approval:
            from constat.execution.mode import PlanApprovalRequest, PlanApprovalResponse, PlanApproval

            logger.debug(f"[_solve_auditable] require_approval=True, auto_approve={self.session_config.auto_approve}, has_callback={self._approval_callback is not None}")

            # Auto-approve if configured
            if self.session_config.auto_approve:
                logger.debug("[_solve_auditable] Auto-approving (auto_approve=True)")
                approval = PlanApprovalResponse.approve()
            elif not self._approval_callback:
                logger.debug("[_solve_auditable] Auto-approving (no callback)")
                approval = PlanApprovalResponse.approve()
            else:
                logger.debug("[_solve_auditable] Calling approval callback...")
                # Build approval request with full proof structure (preserves type, fact_id)
                request = PlanApprovalRequest(
                    problem=problem,
                    steps=proof_steps,  # Includes type, fact_id for proper display
                    reasoning=f"Question: {claim}",
                )
                approval = self._approval_callback(request)

            if approval.decision == PlanApproval.REJECT:
                self.datastore.set_session_meta("status", "rejected")
                return {
                    "success": False,
                    "rejected": True,
                    "reason": approval.reason,
                    "message": "Verification plan was rejected by user.",
                }

            elif approval.decision == PlanApproval.COMMAND:
                # User entered a slash command - pass back to REPL
                return {
                    "success": False,
                    "command": approval.command,
                    "message": "Slash command entered during approval.",
                }

            elif approval.decision == PlanApproval.SUGGEST:
                # Replan with feedback - for now, just include feedback in context
                problem = f"{problem}\n\nUser guidance: {approval.suggestion}"

            # Filter out deleted steps if any
            if approval.deleted_steps:
                deleted_set = set(approval.deleted_steps)
                # Filter premises by step number (P1, P2, etc. -> 1, 2, etc.)
                premises = [p for p in premises if p.get('number', 0) not in deleted_set]
                # Filter inferences by step number
                inferences = [i for i in inferences if i.get('number', 0) not in deleted_set]
                logger.info(f"Filtered out {len(deleted_set)} deleted steps: {approval.deleted_steps}")

        # Step 2: Execute plan using DAG-based parallel resolution
        # Start proof tree display for auditable mode (will print at end)
        self._emit_event(StepEvent(
            event_type="proof_start",
            step_number=0,
            data={
                "conclusion_fact": "answer",
                "conclusion_description": conclusion,
            }
        ))

        self._emit_event(StepEvent(
            event_type="verifying",
            step_number=0,
            data={"message": f"Resolving facts for: {claim or problem}"}
        ))

        try:
            from constat.execution.dag import parse_plan_to_dag, DAGExecutor, NodeStatus
            from constat.execution.fact_resolver import Fact, FactSource

            # Parse plan into DAG
            dag = parse_plan_to_dag(premises, inferences)

            # Emit fact_start for ALL nodes upfront so UI can show complete DAG
            for node in dag.nodes.values():
                # Determine if premise or inference (use fact_id, not dict key which is node.name)
                is_premise = node.fact_id.startswith("P")
                self._emit_event(StepEvent(
                    event_type="fact_start",
                    step_number=0,
                    data={
                        "fact_name": f"{node.fact_id}: {node.name}",
                        "fact_id": node.fact_id,
                        "fact_description": node.description if hasattr(node, 'description') else None,
                        "dependencies": [f"{dag.nodes[dep].fact_id}: {dep}" for dep in node.dependencies if dep in dag.nodes] if node.dependencies else [],
                        "is_premise": is_premise,
                    }
                ))

            # Get schema for SQL generation
            detailed_schema = self.schema_manager.get_overview()

            # Shared state for node execution
            resolved_premises = {}
            resolved_inferences = {}
            inference_names = {}
            derivation_lines = ["**Premise Resolution:**", ""]

            # Define node executor that calls back to Session
            def execute_node(local_node):
                return self._execute_dag_node(
                    node=local_node,
                    dag=dag,
                    problem=problem,
                    detailed_schema=detailed_schema,
                    premises=premises,
                    inferences=inferences,
                    resolved_premises=resolved_premises,
                    resolved_inferences=resolved_inferences,
                    inference_names=inference_names,
                )

            # Define event callback for progress reporting
            def dag_event_callback(event_type, data):
                node_name = data.get("name", "")
                fact_id = data.get("fact_id", "")
                level = data.get("level", 0)

                if event_type == "node_running":
                    # Determine if premise or inference
                    node_is_premise = fact_id.startswith("P") if fact_id else level == 0

                    # INVERTED TREE: Find what this node FEEDS INTO (not what it depends on)
                    # This shows derivation flow: premises -> inferences -> answer
                    def find_consumer(source_name: str) -> str | None:
                        """Find the first node that uses this node as input."""
                        for other_node in dag.nodes.values():
                            if source_name in other_node.dependencies:
                                return other_node.name
                        return None

                    # Emit fact_executing for DAG visualization
                    self._emit_event(StepEvent(
                        event_type="fact_executing",
                        step_number=level + 1,
                        data={
                            "fact_name": f"{fact_id}: {node_name}",
                            "fact_id": fact_id,
                        }
                    ))

                    if node_is_premise:
                        # Parent = what inference uses this premise
                        consumer = find_consumer(node_name)
                        logger.debug(f"[DAG] {fact_id} (premise) feeds into: {consumer}")
                        self._emit_event(StepEvent(
                            event_type="premise_resolving",
                            step_number=level + 1,
                            data={
                                "fact_name": f"{fact_id}: {node_name}",
                                "step": level + 1,
                                "parent": consumer,  # What this premise feeds into
                            }
                        ))
                    else:
                        # Parent = what inference uses this inference (or root if terminal)
                        consumer = find_consumer(node_name)
                        logger.debug(f"[DAG] {fact_id} (inference) feeds into: {consumer}")
                        self._emit_event(StepEvent(
                            event_type="inference_executing",
                            step_number=level + 1,
                            data={
                                "inference_id": fact_id,
                                "operation": node_name,
                                "parent": consumer,  # What this inference feeds into
                            }
                        ))

                elif event_type == "node_resolved":
                    resolved_value = data.get("value")
                    resolved_confidence = data.get("confidence", 0.9)
                    source = data.get("source", "")
                    node_is_premise = fact_id.startswith("P") if fact_id else level == 0

                    # Get dependencies for this node (dag.nodes keyed by name, not fact_id)
                    node_obj = dag.nodes.get(node_name)
                    node_deps = [f"{dag.nodes[dep].fact_id}: {dep}" for dep in node_obj.dependencies if dep in dag.nodes] if node_obj and node_obj.dependencies else []

                    # Emit fact_resolved for DAG visualization
                    fact_resolved_data = {
                        "fact_name": f"{fact_id}: {node_name}",
                        "fact_id": fact_id,
                        "value": resolved_value,
                        "confidence": resolved_confidence,
                        "source": source,
                        "dependencies": node_deps,
                    }
                    validations = data.get("validations")
                    if validations:
                        fact_resolved_data["validations"] = validations
                    elapsed_ms = data.get("elapsed_ms")
                    if elapsed_ms is not None:
                        fact_resolved_data["elapsed_ms"] = elapsed_ms
                    attempt = data.get("attempt")
                    if attempt is not None:
                        fact_resolved_data["attempt"] = attempt
                    self._emit_event(StepEvent(
                        event_type="fact_resolved",
                        step_number=level + 1,
                        data=fact_resolved_data,
                    ))

                    if node_is_premise:
                        derivation_lines.append(f"- {fact_id}: {node_name} = {str(resolved_value)[:100]} (confidence: {resolved_confidence:.0%})")
                        self._emit_event(StepEvent(
                            event_type="premise_resolved",
                            step_number=level + 1,
                            data={"fact_name": f"{fact_id}: {node_name}", "value": resolved_value, "confidence": resolved_confidence, "source": source}
                        ))
                    else:
                        self._emit_event(StepEvent(
                            event_type="inference_complete",
                            step_number=level + 1,
                            data={"inference_id": fact_id, "inference_name": node_name, "result": resolved_value}
                        ))

                elif event_type == "node_failed":
                    error = data.get("error", "Unknown error")
                    logger.error(f"{fact_id} ({node_name}) failed: {error}")

                    # Emit fact_failed for DAG visualization
                    self._emit_event(StepEvent(
                        event_type="fact_failed",
                        step_number=level + 1,
                        data={
                            "fact_name": f"{fact_id}: {node_name}",
                            "fact_id": fact_id,
                            "reason": error,
                        }
                    ))

                    self._emit_event(StepEvent(
                        event_type="premise_resolved" if fact_id.startswith("P") else "inference_failed",
                        step_number=level + 1,
                        data={"fact_name": f"{fact_id}: {node_name}", "error": error}
                    ))

                elif event_type == "node_blocked":
                    blocked_by = data.get("blocked_by", "dependency failed")
                    logger.info(f"{fact_id} ({node_name}) blocked by {blocked_by}")

                    self._emit_event(StepEvent(
                        event_type="fact_blocked",
                        step_number=level + 1,
                        data={
                            "fact_name": f"{fact_id}: {node_name}",
                            "fact_id": fact_id,
                            "reason": f"blocked by {blocked_by}",
                        }
                    ))

                elif event_type == "node_started":
                    # Log actual thread start for parallelism diagnosis
                    node_start_time = data.get("start_time_ms", 0)
                    logger.debug(f"DAG node {fact_id} STARTED at {node_start_time}ms")

                elif event_type == "node_timing":
                    # Log timing for parallelism analysis
                    start_ms = data.get("start_ms", 0)
                    end_ms = data.get("end_ms", 0)
                    node_duration_ms = data.get("duration_ms", 0)
                    node_failed = data.get("failed", False)
                    status = "FAILED" if node_failed else "COMPLETED"
                    logger.info(f"DAG timing: {fact_id} {status} - start:{start_ms} end:{end_ms} duration:{node_duration_ms}ms")

            # Phase 4: Extract user-specified validation constraints from the problem
            self._proof_user_validations = self._extract_user_validations(problem, inferences)

            # Reset cancellation state before starting execution
            self.reset_cancellation()

            # Execute DAG with parallel resolution
            executor = DAGExecutor(
                dag=dag,
                node_executor=execute_node,
                max_workers=min(10, len(premises) + len(inferences)),
                event_callback=dag_event_callback,
                fail_fast=True,
                execution_context=self._execution_context,
            )

            # Build pre-resolved info for nodes with embedded values (e.g., "breed_limit = 10")
            pre_resolved = {}
            for node in dag.nodes.values():
                if node.status == NodeStatus.RESOLVED and node.is_leaf:
                    pre_resolved[node.fact_id] = {
                        "value": node.value,
                        "confidence": node.confidence,
                    }

            # Emit event to start live plan display (includes pre-resolved info)
            self._emit_event(StepEvent(
                event_type="dag_execution_start",
                step_number=0,
                data={"premises": premises, "inferences": inferences, "pre_resolved": pre_resolved}
            ))

            # Phase 4: Check for cancellation before starting DAG execution
            if self.is_cancelled():
                self._emit_event(StepEvent(
                    event_type="execution_cancelled",
                    step_number=0,
                    data={"message": "Execution cancelled before DAG execution started"}
                ))
                return {
                    "success": False,
                    "cancelled": True,
                    "message": "Execution cancelled before fact resolution started.",
                    "queued_intent_results": self.process_queued_intents(),
                }

            logger.info(f"Executing DAG with {len(premises)} premises and {len(inferences)} inferences")

            # Temporarily disable fact_resolver events during DAG execution
            # Session.py already emits all necessary events with consistent naming (e.g., "P1: employees")
            # fact_resolver.resolve_tiered() emits duplicate events with different naming (e.g., "employees")
            saved_callback = self.fact_resolver._event_callback
            self.fact_resolver._event_callback = None
            try:
                result = executor.execute()
            finally:
                self.fact_resolver._event_callback = saved_callback

            # Emit event to stop live plan display
            self._emit_event(StepEvent(
                event_type="dag_execution_complete",
                step_number=0,
                data={"success": result.success, "failed_nodes": result.failed_nodes, "cancelled": result.cancelled}
            ))

            # Check for cancellation
            if result.cancelled:
                self._emit_event(StepEvent(
                    event_type="execution_cancelled",
                    step_number=0,
                    data={"message": "Execution cancelled during DAG execution"}
                ))
                return {
                    "success": False,
                    "cancelled": True,
                    "message": "Execution cancelled during fact resolution.",
                    "queued_intent_results": self.process_queued_intents(),
                }

            if not result.success:
                failed = ", ".join(result.failed_nodes)
                raise Exception(f"Plan execution failed. Could not resolve: {failed}")

            # Build inference lines from results
            inference_lines = ["", "**Inference Execution:**", ""]
            for inf in inferences:
                inf_id = inf['id']
                inf_name = inf.get('name', inf_id)
                if inf_id in resolved_inferences:
                    val = resolved_inferences[inf_id]
                    inference_lines.append(f"- {inf_id}: {inf_name} = {val} ✓")
                else:
                    # Check DAG node for value
                    node = dag.get_node(inf_name)
                    if node and node.value is not None:
                        inference_lines.append(f"- {inf_id}: {inf_name} = {node.value} ✓")
                        resolved_inferences[inf_id] = node.value

            derivation_lines.extend(inference_lines)
            # Step 4: Synthesize answer from resolved premises and inferences
            self._emit_event(StepEvent(
                event_type="synthesizing",
                step_number=0,
                data={"message": "Synthesizing answer from resolved facts..."}
            ))

            # Build synthesis context - use English names for clarity
            # Truncate values to prevent token overflow
            def truncate_value(v, max_chars=500):
                s = str(v)
                return s[:max_chars] + "..." if len(s) > max_chars else s

            resolved_context = "\n".join([
                f"- {pid} ({premises[int(pid[1:])-1]['name']}): {truncate_value(p.value)}"
                for pid, p in resolved_premises.items() if p and p.value
            ])
            # Use English variable names (e.g., "budget_validated_raises") not IDs (e.g., "I6")
            inference_context = "\n".join([
                f"- {inf_id} ({inference_names.get(inf_id, inf_id)}): {truncate_value(result, 200)}"
                for inf_id, result in resolved_inferences.items()
            ])

            # Collect artifact table names for synthesis (don't include table data)
            artifact_tables = []
            if inferences and self.datastore:
                available_tables = {t['name'] for t in self.datastore.list_tables()}
                for inf in inferences:
                    inf_id = inf['id']
                    table_name = inference_names.get(inf_id, inf_id.lower())
                    result = resolved_inferences.get(inf_id, "")
                    # Track tables that were created (skip verification steps)
                    if "rows" in str(result) and "verified" not in str(result).lower() and "FAILED" not in str(result):
                        if table_name in available_tables:
                            artifact_tables.append(table_name)

            # Build artifact reference for synthesis
            artifact_reference = ""
            if artifact_tables:
                artifact_reference = f"\n\nArtifact tables created: {', '.join(artifact_tables)}"
                artifact_reference += "\n(User can view these via /tables command)"

            synthesis_prompt = f"""Based on the resolved premises and inference plan, provide the answer.

Question: {claim}

Resolved Premises:
{resolved_context if resolved_context else "(no premises resolved)"}

Inference Steps:
{inference_context}

Conclusion to derive: {conclusion}
{artifact_reference}

IMPORTANT INSTRUCTIONS:
1. Always refer to data by its English variable name (e.g., "budget_validated_raises"), NEVER by ID (e.g., "I6")
2. Do NOT display tables inline - data is available as clickable artifacts
3. ALWAYS use backticks when referencing artifacts: `table_name` (these become clickable links)
4. For tables, include row count: `budget_validated_raises` (15 rows)
5. ONLY reference artifacts that were actually created - do not invent table names
6. Focus on the key findings and conclusions, not on showing raw data
7. If the user asked for recommendations/suggestions, summarize them with key values
8. EVALUATE GOAL COMPLETENESS: Re-read the original question carefully. Explicitly assess whether EVERY aspect of the question has been addressed. If any goal was partially or not addressed, state what is missing and why.

Provide a concise, clear answer with inline artifact references."""

            synthesis_result = self.router.execute(
                task_type=TaskType.SYNTHESIS,
                system="You synthesize answers from resolved facts with full provenance.",
                user_message=synthesis_prompt,
                max_tokens=self.router.max_output_tokens,
            )

            if not synthesis_result.success:
                logger.warning(f"Synthesis failed: {synthesis_result.content}")
                answer = f"Verification completed but answer synthesis failed. See derivation below."
            else:
                answer = synthesis_result.content
            confidence = sum(p.confidence for p in resolved_premises.values() if p) / max(len(resolved_premises), 1)
            derivation_trace = "\n".join(derivation_lines)

            verify_result = {
                "answer": answer,
                "confidence": confidence,
                "derivation": derivation_trace,
                "sources": [{"type": p["source"], "description": p["description"]} for p in premises],
            }

            # Generate insights if enabled
            insights = ""
            skip_insights = not self.session_config.enable_insights
            if not skip_insights:
                self._emit_event(StepEvent(
                    event_type="generating_insights",
                    step_number=0,
                    data={"message": "Generating insights..."}
                ))

                # Build source summary
                source_types = set()
                for p in resolved_premises.values():
                    if p and p.source:
                        source_types.add(p.source.value if hasattr(p.source, 'value') else str(p.source))

                insights_prompt = f"""Provide a brief summary of this analysis.

Original question: {claim}

Resolved premises:
{resolved_context}

Inference results:
{inference_context}

Conclusion: {conclusion}

Sources used: {', '.join(source_types) if source_types else 'various'}

Write a SHORT summary (2-3 sentences max) in plain prose explaining what this analysis shows.
Do NOT use bullet points or numbered lists. Just a brief paragraph.
Focus on the key finding and its significance.
If the original question had multiple goals or sub-questions, note whether all were addressed."""

                try:
                    insights_result = self.router.execute(
                        task_type=TaskType.SYNTHESIS,
                        system="You analyze proofs and provide actionable insights.",
                        user_message=insights_prompt,
                        max_tokens=self.router.max_output_tokens,
                    )
                    insights = insights_result.content
                except Exception as e:
                    logger.debug(f"Failed to generate insights (non-fatal): {e}")
                    insights = ""

            duration_ms = int((time.time() - start_time) * 1000)

            # Format output
            answer = verify_result.get("answer", "")
            confidence = verify_result.get("confidence", 0.0)
            derivation_trace = verify_result.get("derivation", "")
            sources = verify_result.get("sources", [])

            # Build final output (derivation details shown during execution, not in final answer)
            output_parts = [
                f"**Verification Result** (confidence: {confidence:.0%})",
                "",
                answer,
            ]

            if insights:
                output_parts.extend([
                    "",
                    "**Summary:**",
                    insights,
                ])

            final_output = "\n".join(output_parts)

            self._emit_event(StepEvent(
                event_type="verification_complete",
                step_number=0,
                data={
                    "answer": answer,
                    "confidence": confidence,
                    "has_derivation": bool(derivation_trace),
                }
            ))

            # Record in history
            self.history.record_query(
                session_id=self.session_id,
                question=problem,
                success=True,
                attempts=1,
                duration_ms=duration_ms,
                answer=final_output,
            )

            # Save resolved facts for redo operations
            if self.datastore:
                import json
                cached_facts = self.fact_resolver.export_cache()
                self.datastore.set_session_meta("resolved_facts", json.dumps(cached_facts))

            # Generate and save Data Flow Diagram (DFD) as published artifact
            try:
                from constat.visualization.box_dag import generate_proof_dfd
                dfd_text = generate_proof_dfd(proof_steps, max_width=80, max_name_len=10)
                if dfd_text and self.datastore:
                    from pathlib import Path
                    artifacts_dir = Path(".constat") / self.user_id / "sessions" / self.session_id / "artifacts"
                    artifacts_dir.mkdir(parents=True, exist_ok=True)
                    dfd_path = artifacts_dir / "data_flow.txt"
                    dfd_path.write_text(dfd_text)

                    # Register as published artifact
                    self.registry.register_artifact(
                        user_id=self.user_id,
                        session_id=self.session_id,
                        name="data_flow",
                        file_path=str(dfd_path.resolve()),
                        artifact_type="diagram",
                        size_bytes=len(dfd_text.encode('utf-8')),
                        description="Data flow diagram showing proof dependencies",
                        is_published=True,
                        title="Data Flow Diagram",
                    )
                    logger.debug(f"Saved DFD artifact: {dfd_path}")
            except Exception as e:
                logger.warning(f"Failed to generate DFD: {e}")

            # Build proof nodes for summary generation
            proof_nodes = []
            for p in premises:
                pid = p['id']
                resolved = resolved_premises.get(pid)
                proof_nodes.append({
                    "id": pid,
                    "name": p['name'],
                    "value": resolved.value if resolved else None,
                    "source": resolved.source.value if resolved and hasattr(resolved.source, 'value') else str(resolved.source) if resolved else p.get('source'),
                    "confidence": resolved.confidence if resolved else None,
                    "dependencies": [],
                })
            for inf in inferences:
                iid = inf['id']
                value = resolved_inferences.get(iid)
                # Get actual confidence and reasoning from DAG node
                inf_name = inf.get('name', iid)
                dag_node = dag.get_node(inf_name)
                node_confidence = dag_node.confidence if dag_node else (1.0 if value is not None else 0.0)
                # Get reasoning from the fact resolver if available
                node_reasoning = None
                if self.fact_resolver:
                    fact = self.fact_resolver.get_fact(inf_name)
                    if fact and hasattr(fact, 'reasoning'):
                        node_reasoning = fact.reasoning
                # Get actual dependencies from DAG (names → fact_ids)
                deps = []
                if dag_node and dag_node.dependencies:
                    for dep_name in dag_node.dependencies:
                        dep_node = dag.get_node(dep_name)
                        if dep_node:
                            deps.append(dep_node.fact_id)
                if not deps:
                    deps = [p['id'] for p in premises]  # Fallback
                proof_nodes.append({
                    "id": iid,
                    "name": inf_name,
                    "value": value,
                    "source": "derived",
                    "confidence": node_confidence,
                    "dependencies": deps,
                    "reasoning": node_reasoning,
                })

            return {
                "success": True,
                "mode": Mode.PROOF.value,
                "output": final_output,
                "final_answer": answer,
                "confidence": confidence,
                "derivation": derivation_trace,
                "derivation_chain": derivation_trace,  # Alias for UI
                "sources": sources,
                "proof_nodes": proof_nodes,  # For summary generation
                "problem": problem,  # Original problem text
                "suggestions": [
                    "Show me the supporting data for this verification",
                    "What assumptions were made in this analysis?",
                ],
                "datastore_tables": self.datastore.list_tables() if self.datastore else [],
            }

        except Exception as e:
            _duration_ms = int((time.time() - start_time) * 1000)

            self._emit_event(StepEvent(
                event_type="verification_error",
                step_number=0,
                data={"error": str(e)}
            ))

            return {
                "success": False,
                "mode": Mode.PROOF.value,
                "error": str(e),
                "output": f"Verification failed: {e}",
            }
