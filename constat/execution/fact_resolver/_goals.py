# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Goals mixin: Prolog-style goal resolution."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ._types import (
    Fact,
    FactSource,
    format_source_attribution,
)

if TYPE_CHECKING:
    from . import FactResolver

logger = logging.getLogger(__name__)


class GoalsMixin:
    def resolve_goal(self: "FactResolver", question: str, schema_context: str = "") -> dict:
        """
        Resolve a question using Prolog-style goal decomposition.

        The question is decomposed into a goal and rules using Prolog syntax:
        - Goal: `answer(Q3, premium, Revenue)` - what we want to prove
        - Rules: `answer(Q, T, R) :- dates(Q, S, E), tier(T, C), revenue(C, S, E, R).`
        - Facts: Base predicates resolved through the hierarchy

        This approach naturally captures:
        - Variable binding/unification
        - Dependency graphs (implicit in rule bodies)
        - Recursive decomposition

        Args:
            question: The question to answer
            schema_context: Optional schema/database context

        Returns:
            Dict with:
            - answer: The resolved goal with bound variables
            - goal: The original goal predicate
            - rules: The decomposition rules
            - bindings: Variable -> value bindings
            - derivation: Prolog-style derivation trace
            - confidence: Overall confidence
            - unresolved: Goals that couldn't be resolved
        """
        if not self.llm:
            return {"answer": None, "error": "No LLM configured"}

        # Step 1: Decompose question into Prolog-style goal and rules
        decompose_prompt = f"""Express this question as a Prolog-style goal with decomposition rules.

Question: {question}

{f"Available data context:\\n{schema_context}" if schema_context else ""}

Format your response as:

GOAL: predicate(Arg1, Arg2, Result)

RULES:
predicate(A, B, R) :-
    subgoal1(A, X),
    subgoal2(B, Y),
    compute(X, Y, R).

SOURCES:
subgoal1: DATABASE | DOCUMENT | LLM_KNOWLEDGE | USER_PROVIDED
subgoal2: DATABASE | DOCUMENT | LLM_KNOWLEDGE | USER_PROVIDED

Example for "What was revenue by customer tier in Q3?":

GOAL: revenue_by_tier(q3, Tier, Revenue)

RULES:
revenue_by_tier(Quarter, Tier, Revenue) :-
    date_range(Quarter, StartDate, EndDate),
    tier_criteria(Tier, Criteria),
    customers_matching(Criteria, Customers),
    sum_revenue(Customers, StartDate, EndDate, Revenue).

SOURCES:
date_range: LLM_KNOWLEDGE (calendar knowledge)
tier_criteria: DOCUMENT or USER_PROVIDED (business definition)
customers_matching: DATABASE (query customers table)
sum_revenue: DATABASE (aggregate orders table)

Use uppercase for Variables that need binding, lowercase for constants.
"""

        self._emit_event("decomposing_goal", {"question": question})

        response = self.llm.generate(
            system="You decompose questions into Prolog-style goals and rules.",
            user_message=decompose_prompt,
            max_tokens=self.llm.max_output_tokens,
        )

        # Parse the response
        goal = ""
        rules: list[str] = []
        sources: dict[str, str] = {}

        current_section = None
        rule_lines = []

        for line in response.split("\n"):
            line_stripped = line.strip()

            if line_stripped.startswith("GOAL:"):
                goal = line_stripped.split("GOAL:", 1)[1].strip()
                current_section = "goal"
            elif line_stripped == "RULES:":
                current_section = "rules"
            elif line_stripped == "SOURCES:":
                # Save accumulated rule
                if rule_lines:
                    rules.append(" ".join(rule_lines))
                    rule_lines = []
                current_section = "sources"
            elif current_section == "rules" and line_stripped:
                rule_lines.append(line_stripped)
                # Check if rule is complete (ends with period)
                if line_stripped.endswith("."):
                    rules.append(" ".join(rule_lines))
                    rule_lines = []
            elif current_section == "sources" and ":" in line_stripped:
                parts = line_stripped.split(":", 1)
                pred_name = parts[0].strip()
                source_hint = parts[1].strip()
                sources[pred_name] = source_hint

        self._emit_event("goal_decomposed", {
            "goal": goal,
            "rules": rules,
            "sources": list(sources.keys()),
        })

        # Step 2: Parse the goal to extract predicate name and arguments
        goal_pred, goal_args = self._parse_predicate(goal)
        if not goal_pred:
            return {"answer": None, "error": f"Could not parse goal: {goal}"}

        # Step 3: Resolve the goal using rules and hierarchy
        bindings: dict[str, Any] = {}
        resolved_facts: dict[str, Fact] = {}
        unresolved: list[str] = []

        # First, bind any constants in the goal
        for arg in goal_args:
            if arg and arg[0].islower():  # lowercase = constant
                bindings[arg] = arg

        # Parse rules to understand dependencies
        rule_deps = self._parse_rules(rules)

        # Resolve sub-goals in dependency order
        self._resolve_subgoals(
            goal_pred, goal_args, rule_deps, sources, bindings, resolved_facts, unresolved
        )

        # Step 4: Build the derivation trace in Prolog style
        derivation = self._build_prolog_derivation(
            goal, rules, bindings, resolved_facts, unresolved
        )

        # Calculate confidence
        if resolved_facts:
            confidence = min(f.confidence for f in resolved_facts.values())
        else:
            confidence = 0.0

        # Build final answer by substituting bindings into goal
        answer = self._substitute_bindings(goal, bindings)

        return {
            "answer": answer,
            "goal": goal,
            "rules": rules,
            "sources": sources,
            "bindings": bindings,
            "derivation": derivation,
            "confidence": confidence,
            "unresolved": unresolved,
            "facts": {k: v.to_dict() for k, v in resolved_facts.items()},
        }

    @staticmethod
    def _parse_predicate(pred_str: str) -> tuple[str, list[str]]:
        """Parse a predicate string like 'foo(X, Y, Z)' into name and args."""
        pred_str = pred_str.strip().rstrip(".")
        if "(" not in pred_str:
            return pred_str, []

        name = pred_str.split("(")[0].strip()
        args_str = pred_str.split("(", 1)[1].rsplit(")", 1)[0]
        args = [a.strip() for a in args_str.split(",")]
        return name, args

    def _parse_rules(self: "FactResolver", rules: list[str]) -> dict[str, list[str]]:
        """Parse rules to extract head -> body dependencies."""
        deps = {}
        for rule in rules:
            if ":-" not in rule:
                continue
            head, body = rule.split(":-", 1)
            head_pred, _ = self._parse_predicate(head)

            # Extract predicates from body
            body_predicates = []
            # Simple parsing - split by comma but handle nested parens
            depth = 0
            current = ""
            for char in body:
                if char == "(":
                    depth += 1
                    current += char
                elif char == ")":
                    depth -= 1
                    current += char
                elif char == "," and depth == 0:
                    if current.strip():
                        pred_name, _ = self._parse_predicate(current.strip())
                        if pred_name:
                            body_predicates.append(pred_name)
                    current = ""
                else:
                    current += char
            # Don't forget the last one
            if current.strip():
                pred_name, _ = self._parse_predicate(current.strip().rstrip("."))
                if pred_name:
                    body_predicates.append(pred_name)

            deps[head_pred] = body_predicates

        return deps

    def _resolve_subgoals(
        self: "FactResolver",
        goal_pred: str,
        goal_args: list[str],
        rule_deps: dict[str, list[str]],
        sources: dict[str, str],
        bindings: dict[str, Any],
        resolved_facts: dict[str, Fact],
        unresolved: list[str],
    ) -> bool:
        """Recursively resolve subgoals using the hierarchy."""
        # Get subgoals for this predicate
        subgoals = rule_deps.get(goal_pred, [])

        if not subgoals:
            # This is a base predicate - resolve it directly
            return self._resolve_base_predicate(
                goal_pred, goal_args, sources, bindings, resolved_facts, unresolved
            )

        # Resolve each subgoal
        # First, identify which can be resolved in parallel (no shared unbound vars)
        dependent = []

        for subgoal in subgoals:
            # Check if this subgoal depends on variables from previous subgoals
            # For simplicity, assume all are potentially dependent for now
            dependent.append(subgoal)

        # Resolve subgoals
        for subgoal in dependent:
            sub_args = []  # Would need to track args from rule body
            success = self._resolve_subgoals(
                subgoal, sub_args, rule_deps, sources, bindings, resolved_facts, unresolved
            )
            if not success:
                # Continue trying others, but mark as unresolved
                pass

        return len(unresolved) == 0

    def _resolve_base_predicate(
        self: "FactResolver",
        pred_name: str,
        pred_args: list[str],
        sources: dict[str, str],
        bindings: dict[str, Any],
        resolved_facts: dict[str, Fact],
        unresolved: list[str],
    ) -> bool:
        """Resolve a base predicate (leaf in the dependency tree)."""
        self._emit_event("resolving_predicate", {
            "predicate": pred_name,
            "args": pred_args,
        })

        # Try to resolve using our hierarchy
        fact = self.resolve(pred_name)

        if fact.is_resolved:
            resolved_facts[pred_name] = fact
            # Bind the result to any output variable
            if pred_args:
                # Last arg is typically the result/output
                result_var = pred_args[-1]
                if result_var and result_var[0].isupper():
                    bindings[result_var] = fact.value

            self._emit_event("predicate_resolved", {
                "predicate": pred_name,
                "value": fact.value,
                "source": fact.source.value,
            })
            return True
        else:
            unresolved.append(pred_name)
            self._emit_event("predicate_unresolved", {
                "predicate": pred_name,
                "source_hint": sources.get(pred_name, "unknown"),
            })
            return False

    @staticmethod
    def _substitute_bindings(term: str, bindings: dict[str, Any]) -> str:
        """Substitute variable bindings into a term."""
        result = term
        for var, value in bindings.items():
            # Replace variable with its bound value
            result = result.replace(var, str(value))
        return result

    def _build_prolog_derivation(
        self: "FactResolver",
        goal: str,
        rules: list[str],
        bindings: dict[str, Any],
        resolved: dict[str, Fact],
        unresolved: list[str],
    ) -> str:
        """Build Prolog-style derivation trace."""
        lines = [
            "/* Query */",
            f"?- {goal}",
            "",
            "/* Rules */",
        ]
        for rule in rules:
            lines.append(rule)

        lines.append("")
        lines.append("/* Resolution */")

        for pred_name, fact in resolved.items():
            source = fact.source.value
            if fact.source_name:
                source = f"{source}:{fact.source_name}"
            lines.append(f"{pred_name}({fact.value}).  % from {source}, confidence={fact.confidence:.0%}")
            if fact.query:
                lines.append(f"  % SQL: {fact.query[:60]}...")

        if unresolved:
            lines.append("")
            lines.append("/* Unresolved (need user input) */")
            for pred in unresolved:
                lines.append(f"% {pred}(?).  % Could not resolve")

        lines.append("")
        lines.append("/* Answer */")
        answer = self._substitute_bindings(goal, bindings)
        lines.append(f"{answer}.")

        if bindings:
            lines.append("")
            lines.append("/* Bindings */")
            for var, val in bindings.items():
                if var[0].isupper():  # Only show variable bindings
                    lines.append(f"% {var} = {val}")

        return "\n".join(lines)

    def resolve_conclusion(self: "FactResolver", question: str, schema_context: str = "") -> dict:
        """
        Resolve a question using template-based symbolic evaluation.

        1. Create a statement template with {variables} that would answer the question
        2. Extract variables from the template
        3. Resolve each variable through the hierarchy (recursively)
        4. Substitute resolved values back into the template
        5. Evaluate to get the final answer

        Args:
            question: The question to answer
            schema_context: Optional schema/database context

        Returns:
            Dict with:
            - answer: The final resolved answer
            - template: The statement template with variables
            - substitutions: Dict of variable -> resolved value
            - derivation: Human-readable trace
            - confidence: Overall confidence (min of all resolved facts)
            - unresolved: List of variables that couldn't be resolved
        """
        if not self.llm:
            return {"answer": None, "error": "No LLM configured"}

        # Step 1: Generate statement template with variables
        template_prompt = f"""Given this question, create a statement template that would answer it.
Use {{variable_name}} for values that need to be resolved.

Question: {question}

{f"Available data context: {schema_context}" if schema_context else ""}

Create a template where resolving all variables would answer the question.
For each variable, briefly describe what it represents.

Format:
TEMPLATE: <statement with {{variables}}>
VARIABLES:
- {{var1}}: description of what this variable represents
- {{var2}}: description of what this variable represents
...

Example for "What was total revenue in Q3 for premium customers?":
TEMPLATE: The total revenue in Q3 for premium customers was ${{total_revenue}}, calculated from {{order_count}} orders with an average of ${{avg_order_value}} per order.
VARIABLES:
- {{total_revenue}}: Sum of all order amounts for premium customers in Q3
- {{order_count}}: Number of orders from premium customers in Q3
- {{avg_order_value}}: Average order value (total_revenue / order_count)
- {{q3_start_date}}: Start date of Q3 (derived from "Q3")
- {{q3_end_date}}: End date of Q3 (derived from "Q3")
- {{premium_customer_ids}}: List of customer IDs classified as premium
"""

        self._emit_event("template_generating", {"question": question})

        template_response = self.llm.generate(
            system="You create answer templates with variables. Be thorough - include ALL variables needed.",
            user_message=template_prompt,
            max_tokens=self.llm.max_output_tokens,
        )

        # Parse template and variables
        template = ""
        variables: dict[str, str] = {}  # var_name -> description

        for line in template_response.split("\n"):
            line = line.strip()
            if line.startswith("TEMPLATE:"):
                template = line.split("TEMPLATE:", 1)[1].strip()
            elif line.startswith("- {") and "}:" in line:
                # Parse variable definition
                var_part = line[2:]  # Remove "- "
                if "}:" in var_part:
                    var_name = var_part.split("}")[0].strip("{}")
                    var_desc = var_part.split("}:", 1)[1].strip()
                    variables[var_name] = var_desc

        self._emit_event("template_created", {
            "template": template,
            "variables": list(variables.keys()),
        })

        # Step 2: Identify dependencies between variables
        # Some variables depend on others (e.g., avg = total / count)
        dependency_prompt = f"""Given these variables, identify which ones depend on others.

Variables:
{chr(10).join(f"- {k}: {v}" for k, v in variables.items())}

For each variable, list its dependencies (other variables it needs).
Variables with no dependencies can be resolved in PARALLEL.

Format each line as:
{{variable}}: [{{dep1}}, {{dep2}}] or [] if no dependencies

Example:
{{total_revenue}}: []
{{order_count}}: []
{{avg_order_value}}: [{{total_revenue}}, {{order_count}}]
"""

        dep_response = self.llm.generate(
            system="You analyze variable dependencies.",
            user_message=dependency_prompt,
            max_tokens=self.llm.max_output_tokens,
        )

        # Parse dependencies
        dependencies: dict[str, list[str]] = {var: [] for var in variables}
        for line in dep_response.split("\n"):
            if ":" in line and "{" in line:
                parts = line.split(":", 1)
                var_name = parts[0].strip().strip("{}")
                if var_name in dependencies:
                    deps_str = parts[1].strip()
                    if deps_str.startswith("[") and "]" in deps_str:
                        deps_list = deps_str[1:deps_str.index("]")]
                        deps = [d.strip().strip("{}") for d in deps_list.split(",") if d.strip()]
                        dependencies[var_name] = [d for d in deps if d in variables]

        # Step 3: Resolve variables in dependency order
        # Independent variables first (parallel), then dependent ones
        resolved: dict[str, Fact] = {}
        unresolved: list[str] = []

        # Find independent variables (no dependencies)
        independent = [v for v, deps in dependencies.items() if not deps]
        dependent = [v for v, deps in dependencies.items() if deps]

        self._emit_event("resolving_independent", {
            "independent": independent,
            "dependent": dependent,
        })

        # Resolve independent variables in parallel
        # Note: We don't pass description as a param since it would change the cache key
        if independent:
            fact_requests = [(var, {}) for var in independent]
            facts = self.resolve_many_sync(fact_requests)

            for var, fact in zip(independent, facts):
                if fact.is_resolved:
                    resolved[var] = fact
                    self._emit_event("variable_resolved", {
                        "variable": var,
                        "value": fact.value,
                        "source": fact.source.value,
                    })
                else:
                    unresolved.append(var)

        # Resolve dependent variables in dependency order
        max_iterations = len(dependent) + 1  # Prevent infinite loop
        iterations = 0

        while dependent and iterations < max_iterations:
            iterations += 1
            resolved_this_round = []

            for var in dependent:
                deps = dependencies[var]
                # Check if all dependencies are resolved
                if all(d in resolved for d in deps):
                    # First check if already in cache with just the variable name
                    cached = self._cache.get(var)
                    if cached and cached.is_resolved:
                        fact = cached
                    else:
                        # Try to resolve - pass dependencies for context, but they won't
                        # affect the cache key since we use just the var name
                        fact = self.resolve(var)

                    if fact.is_resolved:
                        # Link to dependency facts
                        fact.because = [resolved[d] for d in deps]
                        resolved[var] = fact
                        resolved_this_round.append(var)
                        self._emit_event("variable_resolved", {
                            "variable": var,
                            "value": fact.value,
                            "source": fact.source.value,
                            "derived_from": deps,
                        })
                    else:
                        unresolved.append(var)
                        resolved_this_round.append(var)  # Remove from dependent list

            # Remove resolved variables from dependent list
            for var in resolved_this_round:
                if var in dependent:
                    dependent.remove(var)

        # Any remaining dependent variables couldn't be resolved
        unresolved.extend(dependent)

        # Step 4: Substitute resolved values into template
        final_template = template
        substitutions = {}
        for var, fact in resolved.items():
            placeholder = "{" + var + "}"
            if placeholder in final_template:
                value_str = str(fact.value)
                final_template = final_template.replace(placeholder, value_str)
                substitutions[var] = fact.value

        # Step 5: Calculate overall confidence
        if resolved:
            confidence = min(f.confidence for f in resolved.values())
        else:
            confidence = 0.0

        # Build derivation trace
        derivation = self._build_derivation_trace(
            template=template,
            resolved=resolved,
            unresolved=unresolved,
            variables=variables,
            answer=final_template,
        )

        return {
            "answer": final_template,
            "template": template,
            "variables": variables,
            "substitutions": substitutions,
            "derivation": derivation,
            "confidence": confidence,
            "unresolved": unresolved,
            "facts": {k: v.to_dict() for k, v in resolved.items()},
        }

    @staticmethod
    def _build_derivation_trace(
        template: str,
        resolved: dict[str, Fact],
        unresolved: list[str],
        variables: dict[str, str],
        answer: str,
    ) -> str:
        """Build a human-readable derivation trace with provenance."""
        lines = [
            "**Statement:**",
            f"  {template}",
            "",
            "**Variable Resolution:**",
        ]

        for var, fact in resolved.items():
            source_str = fact.source.value
            if fact.source_name:
                source_str = f"{fact.source.value}:{fact.source_name}"
            lines.append(f"  {{{var}}} = {fact.display_value}")
            lines.append(f"    source: {source_str}, confidence: {fact.confidence:.0%}")
            if fact.query:
                lines.append(f"    query: {fact.query[:80]}...")
            if fact.reasoning:
                lines.append(f"    reasoning: {fact.reasoning}")
            if fact.because:
                deps = ", ".join(f.name for f in fact.because)
                lines.append(f"    derived from: {deps}")

        if unresolved:
            lines.append("")
            lines.append("**Unresolved (need user input):**")
            for var in unresolved:
                desc = variables.get(var, "")
                lines.append(f"  {{{var}}}: {desc}")

        lines.append("")
        lines.append("**Conclusion:**")
        lines.append(f"  {answer}")

        return "\n".join(lines)

    def resolve_question(self: "FactResolver", context: str) -> dict:
        """
        Resolve a verification question using fact-based derivation.

        Decomposes the question into required facts, resolves each fact,
        and generates a derivation trace showing how the conclusion was reached.

        Args:
            context: Context string containing the verification request and
                    any prior analysis results

        Returns:
            Dict with:
            - answer: The verification result
            - confidence: Overall confidence (0.0-1.0)
            - derivation: Human-readable derivation trace
            - sources: List of source citations
            - facts_resolved: List of fact names that were resolved
        """
        # Step 1: Decompose the question into required facts
        decompose_prompt = f"""Analyze this verification request and identify the specific facts needed to answer it.

{context}

List each required fact on its own line in the format:
FACT: <fact_name> - <description>

Example:
FACT: total_orders - Number of orders in the time period
FACT: discount_policy_max - Maximum allowed discount percentage
FACT: violations_count - Number of policy violations found

Be specific and exhaustive - list ALL facts needed to verify the claim."""

        decompose_text = self.llm.generate(
            system="You are analyzing verification requests to identify required facts.",
            user_message=decompose_prompt,
            max_tokens=self.llm.max_output_tokens,
        )

        # Parse required facts
        required_facts = []
        for line in decompose_text.split("\n"):
            if line.strip().startswith("FACT:"):
                fact_part = line.split("FACT:", 1)[1].strip()
                if " - " in fact_part:
                    fact_name, fact_desc = fact_part.split(" - ", 1)
                    required_facts.append((fact_name.strip(), fact_desc.strip()))
                else:
                    required_facts.append((fact_part, ""))

        # Step 2: Resolve each required fact
        resolved_facts = []
        derivation_lines = ["**Fact Resolution:**", ""]
        total_facts = len(required_facts)

        for idx, (fact_name, fact_desc) in enumerate(required_facts):
            # Emit event: starting to resolve this fact
            self._emit_event("premise_resolving", {
                "fact_name": fact_name,
                "description": fact_desc,
                "step": idx + 1,
                "total": total_facts,
            })
            # Check cache first
            if fact_name in self._cache:
                fact = self._cache[fact_name]
                derivation_lines.append(f"- {fact_name} = {fact.value} (cached)")
                resolved_facts.append(fact)
                # Emit resolved event for cached fact
                self._emit_event("premise_resolved", {
                    "fact_name": fact_name,
                    "value": fact.value,
                    "source": "cache",
                    "step": idx + 1,
                    "total": total_facts,
                })
            else:
                # Try to resolve the fact
                try:
                    fact = self.resolve(fact_name)
                    resolved_facts.append(fact)

                    # If fact has nested derivations (sub-plan resolution), show full trace
                    if fact.because:
                        # Show that this fact was derived from sub-facts
                        derivation_lines.append(f"- {fact_name} = {fact.value} (derived, confidence: {fact.confidence:.0%})")
                        derivation_lines.append(f"  ↳ Derived from:")
                        for sub_fact in fact.because:
                            sub_source = format_source_attribution(
                                sub_fact.source, sub_fact.source_name, sub_fact.api_endpoint
                            )
                            derivation_lines.append(f"    - {sub_fact.name} = {sub_fact.value} ({sub_source})")
                        source_detail = "derived"
                    else:
                        # Simple fact, use common source format
                        source_detail = format_source_attribution(
                            fact.source, fact.source_name, fact.api_endpoint
                        )
                        derivation_lines.append(f"- {fact_name} = {fact.value} ({source_detail}, confidence: {fact.confidence:.0%})")

                    # Emit resolved event
                    self._emit_event("premise_resolved", {
                        "fact_name": fact_name,
                        "value": fact.value,
                        "source": source_detail,
                        "confidence": fact.confidence,
                        "step": idx + 1,
                        "total": total_facts,
                    })
                except Exception as e:
                    # Mark as unresolved
                    unresolved = Fact(
                        name=fact_name,
                        value=None,
                        confidence=0.0,
                        source=FactSource.UNRESOLVED,
                        description=fact_desc,
                        reasoning=str(e),
                    )
                    self.resolution_log.append(unresolved)
                    derivation_lines.append(f"- {fact_name} = UNRESOLVED ({e})")
                    # Emit resolved event for unresolved fact
                    self._emit_event("premise_resolved", {
                        "fact_name": fact_name,
                        "value": None,
                        "source": "unresolved",
                        "error": str(e),
                        "step": idx + 1,
                        "total": total_facts,
                    })

        # Step 3: Synthesize the answer
        self._emit_event("synthesizing", {
            "message": "Synthesizing answer from resolved facts",
            "resolved_count": len([f for f in resolved_facts if f.is_resolved]),
            "total_count": total_facts,
        })

        facts_context = "\n".join([
            f"- {f.name}: {f.value} (confidence: {f.confidence:.0%})"
            for f in resolved_facts if f.is_resolved
        ])

        synthesis_prompt = f"""Based on the resolved facts, provide a verification answer.

{context}

Resolved Facts:
{facts_context}

Provide:
1. A direct answer to the verification question
2. The confidence level (HIGH/MEDIUM/LOW) with justification
3. Any caveats or limitations

Format your response as:
ANSWER: <your answer>
CONFIDENCE: <HIGH/MEDIUM/LOW> - <justification>
CAVEATS: <any limitations or caveats>"""

        synthesis_text = self.llm.generate(
            system="You are synthesizing verification results from resolved facts.",
            user_message=synthesis_prompt,
            max_tokens=self.llm.max_output_tokens,
        )

        # Parse the synthesis
        answer = ""
        confidence = 0.8  # Default
        caveats = ""

        for line in synthesis_text.split("\n"):
            if line.strip().startswith("ANSWER:"):
                answer = line.split("ANSWER:", 1)[1].strip()
            elif line.strip().startswith("CONFIDENCE:"):
                conf_part = line.split("CONFIDENCE:", 1)[1].strip()
                if conf_part.startswith("HIGH"):
                    confidence = 0.9
                elif conf_part.startswith("MEDIUM"):
                    confidence = 0.7
                elif conf_part.startswith("LOW"):
                    confidence = 0.5
            elif line.strip().startswith("CAVEATS:"):
                caveats = line.split("CAVEATS:", 1)[1].strip()

        # If we didn't parse an answer, use the full synthesis
        if not answer:
            answer = synthesis_text

        # Build derivation trace
        derivation_lines.append("")
        derivation_lines.append("**Conclusion:**")
        derivation_lines.append(answer)
        if caveats and caveats.lower() not in ("none", "n/a", "-"):
            derivation_lines.append("")
            derivation_lines.append(f"**Caveats:** {caveats}")

        derivation = "\n".join(derivation_lines)

        # Build sources list
        sources = []
        for fact in resolved_facts:
            if fact.is_resolved:
                source = {"type": fact.source.value, "description": f"{fact.name}: {fact.value}"}
                if fact.query:
                    source["query"] = fact.query
                sources.append(source)

        return {
            "answer": answer,
            "confidence": confidence,
            "derivation": derivation,
            "sources": sources,
            "facts_resolved": [f.name for f in resolved_facts if f.is_resolved],
        }
