"""Multi-step planner for problem decomposition."""

import json
import re
from datetime import datetime, timezone
from typing import Optional, Union, TYPE_CHECKING

from constat.core.config import Config
from constat.core.models import Plan, PlannerResponse, Step, StepType, TaskType
from constat.providers import TaskRouter
from constat.providers.base import BaseLLMProvider
from constat.catalog.schema_manager import SchemaManager
from constat.storage.learnings import LearningCategory
from constat.discovery.concept_detector import ConceptDetector

if TYPE_CHECKING:
    from constat.discovery.doc_tools import DocumentDiscoveryTools


# System prompt for planning - base version
# Email policy is injected conditionally by ConceptDetector
PLANNER_SYSTEM_PROMPT = """You are a data analysis planner. Given a user problem, break it down into a clear plan of steps.

## Your Task
Analyze the user's question and create a step-by-step plan to answer it. Each step should be:
1. A single, focused action
2. Clear about what data it needs (inputs)
3. Clear about what it produces (outputs)
4. Clear about dependencies on other steps
5. Classified by task type for optimal model routing

## Available Resources
**Database tools:** get_table_schema(table), find_relevant_tables(query)
**API tools:** get_api_schema_overview(api_name), get_api_query_schema(api_name, query_name)

## Code Environment Capabilities
- Database connections (`db_<name>`), API clients (`api_<name>`)
- `pd` (pandas), `np` (numpy), `store` for persisting data between steps
- `llm_ask(question)` for general knowledge, `send_email(to, subject, body, df=None)` for emails

## Data Source Selection
1. Check configured sources first (databases, APIs, documents)
2. Fall back to `llm_ask()` for world knowledge not in databases
3. Use documents for policies/rules and business thresholds

## Planning Guidelines
1. **ALWAYS FILTER AT THE SOURCE** - use SQL WHERE, API filters, or GraphQL arguments
2. **PREFER SQL JOINs over separate queries** for related tables
3. Keep steps atomic - one main action per step
4. Identify parallelizable steps (empty depends_on)
5. End with a step that synthesizes the final answer

## Data Sensitivity
Set `contains_sensitive_data: true` for data under privacy regulations (GDPR, HIPAA).

## Output Format
Return a JSON object:
```json
{
  "reasoning": "Brief explanation of your approach",
  "contains_sensitive_data": false,
  "steps": [
    {"number": 1, "goal": "...", "inputs": [], "outputs": ["df"], "depends_on": [], "task_type": "sql_generation", "complexity": "medium"}
  ]
}
```

## Task Types
- **sql_generation**: Database queries (SELECT, joins, aggregations)
- **python_analysis**: DataFrame transformations and analysis
- **summarization**: Synthesize or explain results

## Complexity Levels
- **low**: Simple single-table queries
- **medium**: Multi-table joins, moderate aggregations
- **high**: Complex joins, window functions

Return ONLY the JSON object, no additional text.
"""


PLANNER_PROMPT_TEMPLATE = """{system_prompt}
{injected_sections}
## Available Databases
{schema_overview}
{api_overview}
{doc_overview}
## Domain Context
{domain_context}
{user_facts}
{learnings}"""


class Planner:
    """
    Generates multi-step plans from natural language problems.

    The planner uses the LLM to break down complex questions into
    sequential steps that can be executed by the session.
    """

    def __init__(
        self,
        config: Config,
        schema_manager: SchemaManager,
        router_or_provider: Optional[Union[BaseLLMProvider, TaskRouter]] = None,
        learning_store=None,
        doc_tools: Optional["DocumentDiscoveryTools"] = None,
    ):
        self.config = config
        self.schema_manager = schema_manager
        self.doc_tools = doc_tools  # For enriching schema search with documents
        self._user_facts: dict = {}  # name -> value mapping
        self._learning_store = learning_store  # For injecting learned rules

        # Support both direct provider (backward compat) and router (new)
        if isinstance(router_or_provider, TaskRouter):
            self.router = router_or_provider
            self.llm = None
        elif isinstance(router_or_provider, BaseLLMProvider):
            self.router = None
            self.llm = router_or_provider
        else:
            # Create a router from config
            self.router = TaskRouter(config.llm)
            self.llm = None

        # Concept detector for conditional prompt injection
        self._concept_detector = ConceptDetector()
        self._concept_detector.initialize()

    def set_user_facts(self, facts: dict) -> None:
        """Set user facts for inclusion in planning prompts.

        Args:
            facts: Dictionary of fact_name -> value (e.g., {"user_email": "ken@example.com"})
        """
        self._user_facts = facts or {}

    def set_learning_store(self, learning_store) -> None:
        """Set learning store for injecting learned rules.

        Args:
            learning_store: LearningStore instance
        """
        self._learning_store = learning_store

    def _build_system_prompt(self, query: str) -> str:
        """Build the full system prompt for planning.

        Args:
            query: The user's problem/question for concept detection
        """
        # Detect relevant concepts and inject specialized sections
        injected_sections = self._concept_detector.get_sections_for_prompt(
            query=query,
            target="planner",
        )

        # Build API overview if configured
        api_overview = ""
        if self.config.apis:
            api_lines = ["\n## Available APIs"]
            for name, api_config in self.config.apis.items():
                api_type = api_config.type.upper()
                desc = api_config.description or f"{api_type} endpoint"
                url = api_config.url or ""
                api_lines.append(f"- **{name}** ({api_type}): {desc}")
                if url:
                    api_lines.append(f"  URL: {url}")
            api_overview = "\n".join(api_lines)

        # Build document overview if configured
        doc_overview = ""
        if self.config.documents:
            doc_lines = ["\n## Reference Documents"]
            for name, doc_config in self.config.documents.items():
                desc = doc_config.description or doc_config.type
                doc_lines.append(f"- **{name}**: {desc}")
            doc_overview = "\n".join(doc_lines)

        # Build user facts section - essential for using correct values like email addresses
        user_facts_text = ""
        if self._user_facts:
            fact_lines = ["\n## Known User Facts (use these exact values)"]
            for name, value in self._user_facts.items():
                fact_lines.append(f"- **{name}**: {value}")
            user_facts_text = "\n".join(fact_lines)

        # Build learnings section - only user/NL corrections (not codegen errors)
        # Codegen errors are only relevant to code generation steps, not planning
        learnings_text = ""
        if self._learning_store:
            try:
                # Get user corrections and NL corrections (domain terminology, etc.)
                rule_lines = ["\n## Learned Rules (apply these patterns)"]
                rules_count = 0

                for category in [LearningCategory.USER_CORRECTION, LearningCategory.NL_CORRECTION]:
                    rules = self._learning_store.list_rules(
                        category=category,
                        min_confidence=0.6,
                    )
                    for rule in rules[:3]:  # Max 3 per category
                        rule_lines.append(f"- {rule['summary']}")
                        rules_count += 1

                if rules_count > 0:
                    learnings_text = "\n".join(rule_lines)
            except Exception:
                pass  # Don't fail planning if learnings can't be loaded

        return PLANNER_PROMPT_TEMPLATE.format(
            system_prompt=PLANNER_SYSTEM_PROMPT,
            injected_sections=injected_sections,
            schema_overview=self.schema_manager.get_brief_summary(),
            api_overview=api_overview,
            doc_overview=doc_overview,
            domain_context=self.config.system_prompt or "No additional domain context provided.",
            user_facts=user_facts_text,
            learnings=learnings_text,
        )

    def _get_tool_handlers(self) -> dict:
        """Get tool handler functions for schema exploration."""
        handlers = {
            "get_table_schema": lambda table: self.schema_manager.get_table_schema(table),
            "find_relevant_tables": lambda query, top_k=5: self.schema_manager.find_relevant_tables(
                query, top_k, doc_tools=self.doc_tools
            ),
        }

        # Add API schema handlers if APIs are configured
        if self.config.apis:
            from constat.catalog.api_executor import APIExecutor
            api_executor = APIExecutor(self.config)
            handlers["get_api_schema_overview"] = lambda api_name: api_executor.get_schema_overview(api_name)
            handlers["get_api_query_schema"] = lambda api_name, query_name: api_executor.get_query_schema(api_name, query_name)

        return handlers

    def _parse_plan_response(self, response: str) -> dict:
        """Parse the LLM's plan response as JSON."""
        # Try to extract JSON from markdown code block
        json_pattern = r"```(?:json)?\s*(.*?)\s*```"
        match = re.search(json_pattern, response, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            json_str = response.strip()

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse plan response as JSON: {e}\nResponse: {response[:500]}")

    def plan(self, problem: str) -> PlannerResponse:
        """
        Generate a multi-step plan for a problem.

        Args:
            problem: Natural language problem to solve

        Returns:
            PlannerResponse with the generated plan
        """
        # Schema tools for exploring the database
        schema_tools = [
            {
                "name": "get_table_schema",
                "description": "Get detailed schema for a specific table including columns, types, primary keys, foreign keys, and row count.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "table": {
                            "type": "string",
                            "description": "Table name as 'database.table' or just 'table' if unambiguous"
                        }
                    },
                    "required": ["table"]
                }
            },
            {
                "name": "find_relevant_tables",
                "description": "Search for tables relevant to a query using semantic similarity.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language description of what data is needed"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Maximum number of results (default 5)",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            }
        ]

        # Add API schema tools if APIs are configured
        if self.config.apis:
            api_names = list(self.config.apis.keys())
            schema_tools.extend([
                {
                    "name": "get_api_schema_overview",
                    "description": f"Get overview of an API's available queries/endpoints including names and descriptions. Available APIs: {api_names}",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "api_name": {
                                "type": "string",
                                "description": "Name of the API to introspect",
                                "enum": api_names
                            }
                        },
                        "required": ["api_name"]
                    }
                },
                {
                    "name": "get_api_query_schema",
                    "description": "Get detailed schema for a specific API query/endpoint, including arguments, filters, and return types.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "api_name": {
                                "type": "string",
                                "description": "Name of the API",
                                "enum": api_names
                            },
                            "query_name": {
                                "type": "string",
                                "description": "Name of the query or endpoint (e.g., 'countries', 'GET /users')"
                            }
                        },
                        "required": ["api_name", "query_name"]
                    }
                }
            ])

        system_prompt = self._build_system_prompt(problem)

        # Use router for planning task
        if self.router:
            result = self.router.execute(
                task_type=TaskType.PLANNING,
                system=system_prompt,
                user_message=f"Create a plan to answer this question:\n\n{problem}",
                tools=schema_tools,
                tool_handlers=self._get_tool_handlers(),
            )
            if not result.success:
                raise ValueError(f"Planning failed: {result.content}")
            response = result.content
        else:
            # Fallback to direct provider
            response = self.llm.generate(
                system=system_prompt,
                user_message=f"Create a plan to answer this question:\n\n{problem}",
                tools=schema_tools,
                tool_handlers=self._get_tool_handlers(),
            )

        # Parse the response
        plan_data = self._parse_plan_response(response)

        # Build Step objects
        steps = []
        for step_data in plan_data.get("steps", []):
            # Parse task_type from response
            task_type_str = step_data.get("task_type", "python_analysis")
            try:
                task_type = TaskType(task_type_str)
            except ValueError:
                task_type = TaskType.PYTHON_ANALYSIS

            steps.append(Step(
                number=step_data.get("number", len(steps) + 1),
                goal=step_data.get("goal", ""),
                expected_inputs=step_data.get("inputs", []),
                expected_outputs=step_data.get("outputs", []),
                depends_on=step_data.get("depends_on", []),
                step_type=StepType.PYTHON,  # Phase 1: Python only
                task_type=task_type,
                complexity=step_data.get("complexity", "medium"),
            ))

        plan = Plan(
            problem=problem,
            steps=steps,
            created_at=datetime.now(timezone.utc).isoformat(),
            contains_sensitive_data=plan_data.get("contains_sensitive_data", False),
        )

        # If LLM didn't provide depends_on, infer from inputs/outputs
        if all(not step.depends_on for step in steps):
            plan.infer_dependencies()

        return PlannerResponse(
            plan=plan,
            reasoning=plan_data.get("reasoning", ""),
        )

    def replan(
        self,
        original_plan: Plan,
        feedback: str,
        completed_steps: list[int],
    ) -> PlannerResponse:
        """
        Revise a plan based on user feedback.

        Args:
            original_plan: The original plan being revised
            feedback: User's correction or additional context
            completed_steps: List of already completed step numbers

        Returns:
            PlannerResponse with the revised plan
        """
        # Build context showing completed steps
        completed_context = "\n".join([
            f"Step {num}: {original_plan.get_step(num).goal} (COMPLETED)"
            for num in completed_steps
        ])

        prompt = f"""The user wants to revise the current plan.

Original problem: {original_plan.problem}

Completed steps:
{completed_context if completed_context else "(none)"}

User feedback: {feedback}

Create a revised plan that:
1. Keeps completed steps as-is (reference their outputs)
2. Addresses the user's feedback
3. Produces a complete solution

Return the plan in JSON format."""

        # Use original problem + feedback for concept detection
        system_prompt = self._build_system_prompt(f"{original_plan.problem} {feedback}")

        # Schema tools for replanning
        schema_tools = [
            {
                "name": "get_table_schema",
                "description": "Get detailed schema for a specific table.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "table": {"type": "string", "description": "Table name"}
                    },
                    "required": ["table"]
                }
            },
            {
                "name": "find_relevant_tables",
                "description": "Search for tables relevant to a query.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "top_k": {"type": "integer", "default": 5}
                    },
                    "required": ["query"]
                }
            }
        ]

        # Use router for replanning task
        if self.router:
            result = self.router.execute(
                task_type=TaskType.REPLANNING,
                system=system_prompt,
                user_message=prompt,
                tools=schema_tools,
                tool_handlers=self._get_tool_handlers(),
            )
            if not result.success:
                raise ValueError(f"Replanning failed: {result.content}")
            response = result.content
        else:
            # Fallback to direct provider
            response = self.llm.generate(
                system=system_prompt,
                user_message=prompt,
                tools=schema_tools,
                tool_handlers=self._get_tool_handlers(),
            )

        plan_data = self._parse_plan_response(response)

        steps = []
        for step_data in plan_data.get("steps", []):
            # Parse task_type from response
            task_type_str = step_data.get("task_type", "python_analysis")
            try:
                task_type = TaskType(task_type_str)
            except ValueError:
                task_type = TaskType.PYTHON_ANALYSIS

            steps.append(Step(
                number=step_data.get("number", len(steps) + 1),
                goal=step_data.get("goal", ""),
                expected_inputs=step_data.get("inputs", []),
                expected_outputs=step_data.get("outputs", []),
                depends_on=step_data.get("depends_on", []),
                step_type=StepType.PYTHON,
                task_type=task_type,
                complexity=step_data.get("complexity", "medium"),
            ))

        # Mark completed steps
        revised_plan = Plan(
            problem=original_plan.problem,
            steps=steps,
            created_at=datetime.now(timezone.utc).isoformat(),
            completed_steps=list(completed_steps),
            contains_sensitive_data=plan_data.get("contains_sensitive_data", original_plan.contains_sensitive_data),
        )

        # Copy results from original completed steps
        for num in completed_steps:
            original_step = original_plan.get_step(num)
            revised_step = revised_plan.get_step(num)
            if original_step and revised_step and original_step.result:
                revised_plan.mark_step_completed(num, original_step.result)

        return PlannerResponse(
            plan=revised_plan,
            reasoning=plan_data.get("reasoning", ""),
        )
