"""Multi-step planner for problem decomposition."""

import json
import re
from datetime import datetime, timezone
from typing import Optional, Union

from constat.core.config import Config
from constat.core.models import Plan, PlannerResponse, Step, StepType, TaskType
from constat.providers import TaskRouter
from constat.providers.base import BaseLLMProvider
from constat.catalog.schema_manager import SchemaManager


# System prompt for planning
PLANNER_SYSTEM_PROMPT = """You are a data analysis planner. Given a user problem, break it down into a clear plan of steps.

## Your Task
Analyze the user's question and create a step-by-step plan to answer it. Each step should be:
1. A single, focused action
2. Clear about what data it needs (inputs)
3. Clear about what it produces (outputs)
4. Clear about dependencies on other steps
5. Classified by task type for optimal model routing

## Available Resources
You have access to these tools to explore schemas:

**Database tools:**
- get_table_schema(table): Get detailed column info for a specific table
- find_relevant_tables(query): Semantic search for tables relevant to your query

**API tools (if APIs configured):**
- get_api_schema_overview(api_name): Get list of available queries/endpoints for an API
- get_api_query_schema(api_name, query_name): Get detailed schema for a specific query including arguments, filters, and return types

## Code Environment Capabilities
Generated code has access to:
- Database connections (`db_<name>`) for SQL queries
- API clients (`api_<name>`) for REST/GraphQL endpoints
- `pd` (pandas) and `np` (numpy) for data manipulation
- `store` for persisting data between steps
- `llm_ask(question)` to get general knowledge from LLM (e.g., "How many planets are in our solar system?")
- `send_email(to, subject, body, df=None)` to send emails with optional DataFrame attachment

Use `llm_ask()` when the question requires general knowledge not in the databases.
Use `send_email()` when the user wants to email results to someone.
Use `api_<name>` to fetch data from configured APIs (GraphQL or REST endpoints).

## Data Source Selection
1. **Check configured sources first**: Look at Available Databases, APIs, and Documents below
2. **Match request to available sources**: Use whichever configured source has the relevant data
3. **Fall back to LLM knowledge**: If no configured source has the data, use `llm_ask()` for world knowledge
4. **Use documents for policies/rules**: Search documents when questions involve thresholds, limits, or business rules

## Data Enrichment Pattern
When a data source provides partial information, use `llm_ask()` to fill gaps:
1. Fetch primary data from the best available source (database, API, or document)
2. **Filter/sample FIRST** - if user wants random/top-N/filtered results, do that before enrichment
3. Use `llm_ask()` to enrich only the filtered rows with missing data
Example: For "10 random items with extra field X":
- Fetch data from source → sample 10 → THEN add field X via `llm_ask()` for just those 10
- This is much cheaper than enriching all rows then sampling

## Planning Guidelines
1. Start by understanding what data is needed
2. **Review available sources below** and pick the best match for the data needed
3. **PREFER SQL JOINs over separate queries** - when data from multiple related tables is needed, use a single SQL query with JOINs rather than multiple separate queries followed by Python merges
4. Each step should produce data that later steps can use
5. Keep steps atomic - one main action per step
6. **Identify parallelizable steps** - steps that don't depend on each other can run in parallel
7. End with a step that synthesizes the final answer

## JOIN Optimization Guidelines
- **Use JOINs when:** Tables share foreign key relationships (e.g., orders.customer_id -> customers.id), you need data from 2-3 related tables, the relationships are straightforward
- **Use separate queries when:** Tables are in different databases, no clear relationship exists, more than 3 tables would need complex multi-way joins, you need to apply complex transformations before joining
- **JOIN syntax tips:** Always qualify column names with table aliases (e.g., o.customer_id, c.name), use explicit JOIN syntax (INNER JOIN, LEFT JOIN) rather than comma-joins, keep JOINs simple - if it requires more than 3 tables, consider breaking it up

## Output Format
Return your plan as a JSON object with this structure:
```json
{
  "reasoning": "Brief explanation of your approach",
  "steps": [
    {
      "number": 1,
      "goal": "Query orders with customer details using JOIN on customer_id",
      "inputs": [],
      "outputs": ["orders_with_customers_df"],
      "depends_on": [],
      "task_type": "sql_generation",
      "complexity": "medium"
    },
    {
      "number": 2,
      "goal": "Analyze and summarize order patterns by customer segment",
      "inputs": ["orders_with_customers_df"],
      "outputs": ["analysis_df"],
      "depends_on": [1],
      "task_type": "python_analysis",
      "complexity": "medium"
    }
  ]
}
```

## Task Types
- **sql_generation**: Steps that primarily query databases (SELECT, joins, aggregations)
- **python_analysis**: Steps that transform, analyze, or compute on DataFrames
- **summarization**: Steps that synthesize or explain results in natural language

## Complexity Levels
- **low**: Simple single-table queries, basic transformations
- **medium**: Multi-table joins, moderate aggregations, standard analysis
- **high**: Complex multi-way joins, window functions, sophisticated analysis

Important:
- Return ONLY the JSON object, no additional text
- Goals should be natural language descriptions
- Inputs/outputs are variable or table names for data flow
- depends_on lists step numbers that must complete before this step can start
- Steps with empty depends_on (or depends_on: []) can run in parallel with other independent steps
"""


PLANNER_PROMPT_TEMPLATE = """{system_prompt}

## Available Databases
{schema_overview}
{api_overview}
{doc_overview}
## Domain Context
{domain_context}"""


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
    ):
        self.config = config
        self.schema_manager = schema_manager

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

    def _build_system_prompt(self) -> str:
        """Build the full system prompt for planning."""
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

        return PLANNER_PROMPT_TEMPLATE.format(
            system_prompt=PLANNER_SYSTEM_PROMPT,
            schema_overview=self.schema_manager.get_overview(),
            api_overview=api_overview,
            doc_overview=doc_overview,
            domain_context=self.config.system_prompt or "No additional domain context provided.",
        )

    def _get_tool_handlers(self) -> dict:
        """Get tool handler functions for schema exploration."""
        handlers = {
            "get_table_schema": lambda table: self.schema_manager.get_table_schema(table),
            "find_relevant_tables": lambda query, top_k=5: self.schema_manager.find_relevant_tables(query, top_k),
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

        system_prompt = self._build_system_prompt()

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

        system_prompt = self._build_system_prompt()

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
