# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Multi-step planner for problem decomposition."""

import json
import logging
import re
from datetime import datetime, timezone
from typing import Optional, Union, TYPE_CHECKING

from constat.core.config import Config

logger = logging.getLogger(__name__)
from constat.core.models import Plan, PlannerResponse, Step, StepType, TaskType
from constat.providers import TaskRouter
from constat.providers.base import BaseLLMProvider
from constat.catalog.schema_manager import SchemaManager
from constat.storage.learnings import LearningCategory
from constat.discovery.concept_detector import ConceptDetector

if TYPE_CHECKING:
    from constat.discovery.doc_tools import DocumentDiscoveryTools
    from constat.catalog.api_schema_manager import APISchemaManager
    from constat.core.resources import SessionResources


# Prompts loaded from external files
from constat.prompts import load_prompt

PLANNER_SYSTEM_PROMPT = load_prompt("planner_system.md")
PLANNER_PROMPT_TEMPLATE = load_prompt("planner_template.md")


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
        api_schema_manager: Optional["APISchemaManager"] = None,
        resources: Optional["SessionResources"] = None,
        allowed_databases: Optional[set[str]] = None,
        allowed_apis: Optional[set[str]] = None,
        allowed_documents: Optional[set[str]] = None,
    ):
        """Initialize the planner.

        Args:
            config: Configuration
            schema_manager: Schema manager for database metadata
            router_or_provider: LLM router or provider
            learning_store: Learning store for injecting learned rules
            doc_tools: Document discovery tools for enriching schema search
            api_schema_manager: API schema manager for semantic search
            resources: SessionResources - consolidated view of available resources
            allowed_databases: Set of allowed database names (None = no filtering)
            allowed_apis: Set of allowed API names (None = no filtering)
            allowed_documents: Set of allowed document names (None = no filtering)
        """
        self.config = config
        self.schema_manager = schema_manager
        self.doc_tools = doc_tools  # For enriching schema search with documents
        self.api_schema_manager = api_schema_manager  # For API semantic search
        self.resources = resources  # Consolidated resources (single source of truth)
        self._user_facts: dict = {}  # name -> value mapping
        self._learning_store = learning_store  # For injecting learned rules
        self.allowed_databases = allowed_databases
        self.allowed_apis = allowed_apis
        self.allowed_documents = allowed_documents
        self._available_roles: list[dict] = []  # For role-based step assignment

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

    def _is_database_allowed(self, db_name: str) -> bool:
        """Check if a database is allowed based on permissions."""
        if self.allowed_databases is None:
            return True
        return db_name in self.allowed_databases

    def _is_api_allowed(self, api_name: str) -> bool:
        """Check if an API is allowed based on permissions."""
        if self.allowed_apis is None:
            return True
        return api_name in self.allowed_apis

    def _is_document_allowed(self, doc_name: str) -> bool:
        """Check if a document is allowed based on permissions."""
        if self.allowed_documents is None:
            return True
        return doc_name in self.allowed_documents

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

    def set_available_roles(self, roles: list[dict]) -> None:
        """Set available roles for role-based step assignment.

        Args:
            roles: List of role dicts with 'name' and 'description' keys
        """
        self._available_roles = roles or []

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

        # Build API overview if configured (filtered by permissions)
        api_overview = ""
        if self.config.apis:
            api_lines = ["\n## Available APIs"]
            for name, api_config in self.config.apis.items():
                # Skip APIs not allowed by permissions
                if not self._is_api_allowed(name):
                    continue
                api_type = api_config.type.upper()
                desc = api_config.description or f"{api_type} endpoint"
                url = api_config.url or ""
                api_lines.append(f"- **{name}** ({api_type}): {desc}")
                if url:
                    api_lines.append(f"  URL: {url}")
            # Only include header if we have allowed APIs
            if len(api_lines) > 1:
                api_overview = "\n".join(api_lines)

        # Build document overview from consolidated resources (single source of truth)
        doc_overview = ""
        if self.resources and self.resources.has_documents():
            doc_lines = ["\n## Reference Documents"]
            for name, info in self.resources.documents.items():
                # Skip documents not allowed by permissions
                if not self._is_document_allowed(name):
                    continue
                doc_lines.append(f"- **{name}**: {info.description or info.doc_type}")
            # Only include header if we have allowed documents
            if len(doc_lines) > 1:
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

        # Build available roles section for role-based step assignment
        roles_text = ""
        if self._available_roles:
            role_lines = ["\n## Available Roles - ASSIGN TO EACH STEP"]
            role_lines.append("**You MUST assign one of these role_id values to each step based on what the step does:**")
            for role in self._available_roles:
                name = role.get("name", "")
                desc = role.get("description", "")
                role_lines.append(f"- **{name}**: {desc}")
            role_lines.append("\nChoose the most appropriate role for each step. Use `null` only if no role applies.")
            roles_text = "\n".join(role_lines)
            logger.info(f"[PLANNER] Including {len(self._available_roles)} roles in prompt")
        else:
            logger.debug("[PLANNER] No roles available for prompt")

        return PLANNER_PROMPT_TEMPLATE.format(
            system_prompt=PLANNER_SYSTEM_PROMPT,
            injected_sections=injected_sections,
            schema_overview=self.schema_manager.get_brief_summary(self.allowed_databases),
            api_overview=api_overview,
            doc_overview=doc_overview,
            domain_context=self.config.system_prompt or "No additional domain context provided.",
            user_facts=user_facts_text,
            learnings=learnings_text,
            available_roles=roles_text,
        )

    def _get_tool_handlers(self) -> dict:
        """Get tool handler functions for schema exploration."""
        handlers = {
            "get_table_schema": lambda table: self.schema_manager.get_table_schema(table),
            "find_relevant_tables": lambda query, top_k=5: self.schema_manager.find_relevant_tables(
                query, top_k, doc_tools=self.doc_tools
            ),
        }

        # Add document discovery tools if available
        if self.doc_tools:
            handlers["list_documents"] = self.doc_tools.list_documents
            handlers["search_documents"] = lambda query, limit=5: self.doc_tools.search_documents(query, limit)
            handlers["get_document"] = lambda name: self.doc_tools.get_document(name)

        # Add API schema handlers if APIs are configured
        if self.config.apis:
            from constat.catalog.api_executor import APIExecutor
            api_executor = APIExecutor(self.config)
            handlers["get_api_schema_overview"] = lambda api_name: api_executor.get_schema_overview(api_name)
            handlers["get_api_query_schema"] = lambda api_name, query_name: api_executor.get_query_schema(api_name, query_name)

            # Add semantic search for APIs if api_schema_manager is available
            if self.api_schema_manager:
                handlers["find_relevant_apis"] = lambda query, limit=5: self.api_schema_manager.find_relevant_apis(query, limit=limit)

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

        # Add API schema tools if APIs are configured (filtered by permissions)
        if self.config.apis:
            # Filter API names by permissions
            api_names = [
                name for name in self.config.apis.keys()
                if self._is_api_allowed(name)
            ]
            if api_names:
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
                    },
                ])

                # Add semantic search for APIs if api_schema_manager is available
                if self.api_schema_manager:
                    schema_tools.append({
                        "name": "find_relevant_apis",
                        "description": "Search for API endpoints relevant to a query using semantic similarity. Returns APIs that might have the data you need.",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Natural language description of what API data is needed"
                                },
                                "limit": {
                                    "type": "integer",
                                    "description": "Maximum number of results (default 5)",
                                    "default": 5
                                }
                            },
                            "required": ["query"]
                        }
                    })

        # Add document discovery tools if documents are available (from resources)
        if self.doc_tools and self.resources and self.resources.has_documents():
            # Get document names from consolidated resources (single source of truth)
            doc_names = [
                name for name in self.resources.document_names
                if self._is_document_allowed(name)
            ]

            if doc_names:
                schema_tools.extend([
                    {
                        "name": "list_documents",
                        "description": "List all available reference documents with descriptions.",
                        "input_schema": {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    },
                    {
                        "name": "search_documents",
                        "description": "Search across all documents for relevant content using semantic search. Use this to find policies, rules, guidelines, or business definitions.",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Natural language description of what information you're looking for"
                                },
                                "limit": {
                                    "type": "integer",
                                    "description": "Maximum number of results (default 5)",
                                    "default": 5
                                }
                            },
                            "required": ["query"]
                        }
                    },
                    {
                        "name": "get_document",
                        "description": f"Get the full content of a reference document. Available documents: {doc_names}",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "Name of the document to retrieve"
                                }
                            },
                            "required": ["name"]
                        }
                    },
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
                role_id=step_data.get("role_id"),  # Role context for this step
            ))

        plan = Plan(
            problem=problem,
            steps=steps,
            created_at=datetime.now(timezone.utc).isoformat(),
            contains_sensitive_data=plan_data.get("contains_sensitive_data", False),
        )

        # Validate: steps must have expected_outputs (except final step)
        for step in steps[:-1]:
            if not step.expected_outputs:
                raise ValueError(
                    f"Step {step.number} has no expected_outputs. "
                    f"Plan is malformed - each step must declare its outputs."
                )

        # Clear LLM's depends_on (unreliable format - often strings instead of ints)
        # and rebuild DAG from expected_inputs/expected_outputs data flow
        for step in steps:
            step.depends_on = []
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

        Creates a fresh plan that incorporates the feedback, rather than
        appending to the original plan which can confuse the LLM.

        Args:
            original_plan: The original plan being revised
            feedback: User's correction or additional context
            completed_steps: List of already completed step numbers

        Returns:
            PlannerResponse with the revised plan
        """
        # Build context about completed work (what data is available, not old plan structure)
        completed_outputs = []
        for num in completed_steps:
            step = original_plan.get_step(num)
            if step and step.result and step.expected_outputs:
                for output in step.expected_outputs:
                    completed_outputs.append(output)

        available_data = ""
        if completed_outputs:
            available_data = f"\n\nAvailable data from prior steps: {', '.join(completed_outputs)}"

        # Present as a fresh planning request with the feedback incorporated
        prompt = f"""Create a plan to answer this question:

{original_plan.problem}

User clarification: {feedback}{available_data}

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
                role_id=step_data.get("role_id"),  # Role context for this step
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
