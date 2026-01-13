"""Session orchestration for multi-step plan execution."""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from constat.core.config import Config
from constat.core.models import Plan, PlannerResponse, Step, StepResult, StepStatus, StepType, TaskType
from constat.storage.datastore import DataStore
from constat.storage.history import SessionHistory
from constat.execution.executor import ExecutionResult, PythonExecutor, format_error_for_retry
from constat.execution.planner import Planner
from constat.execution.scratchpad import Scratchpad
from constat.execution.fact_resolver import FactResolver, FactSource
from constat.execution.mode import (
    ExecutionMode,
    suggest_mode,
    PlanApproval,
    PlanApprovalRequest,
    PlanApprovalResponse,
)
from constat.execution.parallel_scheduler import ParallelStepScheduler, SchedulerConfig
from constat.providers import TaskRouter
from constat.catalog.schema_manager import SchemaManager


# Meta-questions that don't require data queries
META_QUESTION_PATTERNS = [
    "what questions",
    "what can you",
    "help me",
    "capabilities",
    "what do you know",
    "describe yourself",
    "what data",
    "what databases",
    "what tables",
]


def is_meta_question(query: str) -> bool:
    """Check if query is a meta-question about capabilities."""
    query_lower = query.lower()
    return any(pattern in query_lower for pattern in META_QUESTION_PATTERNS)


# Question classification types
class QuestionType:
    DATA_ANALYSIS = "data_analysis"  # Requires database queries
    GENERAL_KNOWLEDGE = "general_knowledge"  # LLM can answer directly
    META_QUESTION = "meta_question"  # About system capabilities


# System prompt for step code generation
STEP_SYSTEM_PROMPT = """You are a data analyst executing a step in a multi-step plan.

## Your Task
Generate Python code to accomplish the current step's goal.

## Code Environment
Your code has access to:
- Database connections: `db_<name>` for each database (e.g., `db_chinook`, `db_northwind`)
- `db`: alias for the first database
- `pd`: pandas (imported as pd)
- `np`: numpy (imported as np)
- `store`: a persistent DuckDB datastore for sharing data between steps
- `llm_ask`: a function to query the LLM for general knowledge

## LLM Knowledge (via llm_ask)
Use `llm_ask(question)` to get general knowledge not available in databases:
```python
# Get industry benchmarks
industry_avg = llm_ask("What is the average profit margin for retail companies?")

# Get conversion factors
exchange_rate = llm_ask("What is the current USD to EUR exchange rate?")

# Get domain knowledge
definition = llm_ask("What qualifies as a 'high-value customer' in e-commerce?")
```
Note: llm_ask returns a string. Parse numeric values if needed.

## State Management (via store)
Each step runs in complete isolation. The ONLY way to share data between steps is through `store`.

For DataFrames:
```python
# Save a DataFrame for later steps
store.save_dataframe('customers', df, step_number=1, description='Customer data')

# Load a DataFrame from a previous step
customers = store.load_dataframe('customers')

# Query saved data with SQL
result = store.query('SELECT * FROM customers WHERE revenue > 1000')

# List available tables
tables = store.list_tables()
```

For simple values (numbers, strings, lists, dicts):
```python
# Save a state variable for later steps
store.set_state('total_revenue', total, step_number=1)
store.set_state('top_genres', ['Rock', 'Latin', 'Metal'], step_number=1)

# Load a state variable from a previous step
total = store.get_state('total_revenue')
genres = store.get_state('top_genres')

# Get all state variables
all_state = store.get_all_state()
```

## Code Rules
1. Use pandas `pd.read_sql(query, db_<name>)` to query source databases
2. For cross-database queries, load from each DB and join in pandas
3. **ALWAYS save results to store** - this is the ONLY way to share data between steps:
   - Any DataFrame result MUST be saved with `store.save_dataframe()`
   - Any list, dict, or computed value MUST be saved with `store.set_state()`
   - Nothing in local variables persists between steps!
4. Print informative output about what was done
5. Keep code focused on the current step's goal

## Output Format
Return ONLY the Python code wrapped in ```python ... ``` markers.
"""


STEP_PROMPT_TEMPLATE = """{system_prompt}

## Available Databases
{schema_overview}

## Domain Context
{domain_context}

## Intermediate Tables (from previous steps)
{datastore_tables}

## Previous Context
{scratchpad}

## Current Step
Step {step_number} of {total_steps}: {goal}

Expected inputs: {inputs}
Expected outputs: {outputs}

Generate the Python code to accomplish this step."""


RETRY_PROMPT_TEMPLATE = """Your previous code failed to execute.

{error_details}

Previous code:
```python
{previous_code}
```

Please fix the code and try again. Return ONLY the corrected Python code wrapped in ```python ... ``` markers."""


# Type for approval callback: (request) -> response
ApprovalCallback = Callable[[PlanApprovalRequest], PlanApprovalResponse]


@dataclass
class SessionConfig:
    """Configuration for a session."""
    max_retries_per_step: int = 10
    verbose: bool = False

    # Plan approval settings
    require_approval: bool = True  # If True, require approval before execution
    max_replan_attempts: int = 3  # Max attempts to replan with user feedback
    auto_approve: bool = False  # If True, auto-approve plans (for testing/scripts)


@dataclass
class StepEvent:
    """Event emitted during step execution."""
    event_type: str  # step_start, generating, executing, step_complete, step_error
    step_number: int
    data: dict = field(default_factory=dict)


class Session:
    """
    Orchestrates multi-step plan execution with step isolation.

    The session:
    1. Takes a problem and generates a plan
    2. Executes each step sequentially in isolation
    3. State is shared ONLY via DuckDB datastore (no in-memory sharing)
    4. Handles errors and retries
    5. Records history for review/resumption

    Step Isolation:
    - Each step runs independently with only the datastore for state sharing
    - Steps use store.save_dataframe() / store.load_dataframe() for DataFrames
    - Steps use store.set_state() / store.get_state() for simple values
    - No in-memory state dict is shared between steps
    """

    def __init__(
        self,
        config: Config,
        session_config: Optional[SessionConfig] = None,
        history: Optional[SessionHistory] = None,
        progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
    ):
        self.config = config
        self.session_config = session_config or SessionConfig()

        # Initialize components
        self.schema_manager = SchemaManager(config)
        self.schema_manager.initialize(progress_callback=progress_callback)

        # Task router for model routing with escalation
        self.router = TaskRouter(config.llm)

        # Default provider (for backward compatibility - e.g., fact resolver)
        self.llm = self.router._get_provider(
            self.router.routing_config.get_models_for_task("general")[0]
        )

        self.planner = Planner(config, self.schema_manager, self.router)

        self.executor = PythonExecutor(
            timeout_seconds=config.execution.timeout_seconds,
            allowed_imports=config.execution.allowed_imports or None,
        )

        self.history = history or SessionHistory()

        # Session state
        self.session_id: Optional[str] = None
        self.plan: Optional[Plan] = None
        self.scratchpad = Scratchpad()
        self.datastore: Optional[DataStore] = None  # Persistent storage (only shared state between steps)

        # Fact resolver for auditable mode
        self.fact_resolver = FactResolver(
            llm=self.llm,
            schema_manager=self.schema_manager,
            config=self.config,
        )

        # Event callbacks for monitoring
        self._event_handlers: list[Callable[[StepEvent], None]] = []

        # Approval callback (set via set_approval_callback)
        self._approval_callback: Optional[ApprovalCallback] = None

    def set_approval_callback(self, callback: ApprovalCallback) -> None:
        """
        Set the callback for plan approval.

        The callback receives a PlanApprovalRequest and must return a PlanApprovalResponse.

        Args:
            callback: Function that handles approval requests
        """
        self._approval_callback = callback

    def on_event(self, handler: Callable[[StepEvent], None]) -> None:
        """Register an event handler for step events."""
        self._event_handlers.append(handler)

    def _emit_event(self, event: StepEvent) -> None:
        """Emit an event to all handlers."""
        for handler in self._event_handlers:
            handler(event)

    def _build_step_prompt(self, step: Step) -> str:
        """Build the prompt for generating step code."""
        # Format datastore tables info
        if self.datastore:
            tables = self.datastore.list_tables()
            if tables:
                table_lines = ["Available in `store` (load with `store.load_dataframe('name')` or query with SQL):"]
                for t in tables:
                    table_lines.append(f"  - {t['name']}: {t['row_count']} rows (from step {t['step_number']})")
                datastore_info = "\n".join(table_lines)
            else:
                datastore_info = "(no tables saved yet)"
        else:
            datastore_info = "(no datastore)"

        # Get scratchpad from datastore (persistent) - source of truth for isolation
        if self.datastore:
            scratchpad_context = self.datastore.get_scratchpad_as_markdown()
        else:
            scratchpad_context = self.scratchpad.get_recent_context(max_steps=5)

        return STEP_PROMPT_TEMPLATE.format(
            system_prompt=STEP_SYSTEM_PROMPT,
            schema_overview=self.schema_manager.get_overview(),
            domain_context=self.config.system_prompt or "No additional context.",
            datastore_tables=datastore_info,
            scratchpad=scratchpad_context,
            step_number=step.number,
            total_steps=len(self.plan.steps) if self.plan else 1,
            goal=step.goal,
            inputs=", ".join(step.expected_inputs) if step.expected_inputs else "(none)",
            outputs=", ".join(step.expected_outputs) if step.expected_outputs else "(none)",
        )

    def _get_tool_handlers(self) -> dict:
        """Get schema tool handlers."""
        return {
            "get_table_schema": lambda table: self.schema_manager.get_table_schema(table),
            "find_relevant_tables": lambda query, top_k=5: self.schema_manager.find_relevant_tables(query, top_k),
        }

    def _get_schema_tools(self) -> list[dict]:
        """Get schema tool definitions."""
        return [
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

    def _create_llm_ask_helper(self) -> callable:
        """Create a helper function for step code to query LLM for general knowledge."""
        def llm_ask(question: str) -> str:
            """
            Ask the LLM a general knowledge question.

            Use this for facts not available in the databases, such as:
            - Industry benchmarks and averages
            - General domain knowledge
            - Conversion factors or standard values
            - Definitions and explanations

            Args:
                question: The question to ask

            Returns:
                The LLM's response as a string
            """
            result = self.router.execute(
                task_type=TaskType.GENERAL,
                system="You are a helpful assistant. Provide factual, concise answers. If you're uncertain, say so.",
                user_message=question,
                max_tokens=500,
            )
            return result.content
        return llm_ask

    def _get_execution_globals(self) -> dict:
        """Get globals dict for code execution.

        Each step runs in isolation - only `store` (DuckDB) is shared.
        """
        globals_dict = {
            "store": self.datastore,  # Persistent datastore - only shared state between steps
            "llm_ask": self._create_llm_ask_helper(),  # LLM query helper for general knowledge
        }

        # Provide database connections
        for i, (db_name, db_config) in enumerate(self.config.databases.items()):
            conn = self.schema_manager.get_connection(db_name)
            globals_dict[f"db_{db_name}"] = conn
            if i == 0:
                globals_dict["db"] = conn

        return globals_dict

    def _auto_save_results(self, namespace: dict, step_number: int) -> None:
        """
        Auto-save any DataFrames or lists found in the execution namespace.

        This ensures intermediate results are persisted even if the LLM
        forgot to explicitly save them.
        """
        import pandas as pd

        # Skip internal/injected variables
        skip_vars = {"store", "db", "pd", "np", "llm_ask", "__builtins__"}
        skip_prefixes = ("db_", "_")

        # Already-saved tables (don't duplicate)
        existing_tables = {t["name"] for t in self.datastore.list_tables()}

        for var_name, value in namespace.items():
            # Skip internal variables
            if var_name in skip_vars or var_name.startswith(skip_prefixes):
                continue

            # Auto-save DataFrames
            if isinstance(value, pd.DataFrame) and var_name not in existing_tables:
                self.datastore.save_dataframe(
                    name=var_name,
                    df=value,
                    step_number=step_number,
                    description=f"Auto-saved from step {step_number}",
                )

            # Auto-save lists (as state, since they might be useful)
            elif isinstance(value, (list, dict)) and len(value) > 0:
                # Check if already saved in state
                existing = self.datastore.get_state(var_name)
                if existing is None:
                    try:
                        self.datastore.set_state(var_name, value, step_number)
                    except Exception:
                        pass  # Skip if not JSON-serializable

    def _execute_step(self, step: Step) -> StepResult:
        """
        Execute a single step with retry on errors.

        Returns:
            StepResult with success/failure info
        """
        start_time = time.time()
        last_code = ""
        last_error = None

        self._emit_event(StepEvent(
            event_type="step_start",
            step_number=step.number,
            data={"goal": step.goal}
        ))

        for attempt in range(1, self.session_config.max_retries_per_step + 1):
            self._emit_event(StepEvent(
                event_type="generating",
                step_number=step.number,
                data={"attempt": attempt}
            ))

            # Use router with step's task_type for automatic model selection/escalation
            if attempt == 1:
                prompt = self._build_step_prompt(step)
                result = self.router.execute_code(
                    task_type=step.task_type,
                    system=STEP_SYSTEM_PROMPT,
                    user_message=prompt,
                    tools=self._get_schema_tools(),
                    tool_handlers=self._get_tool_handlers(),
                    complexity=step.complexity,
                )
            else:
                retry_prompt = RETRY_PROMPT_TEMPLATE.format(
                    error_details=last_error,
                    previous_code=last_code,
                )
                result = self.router.execute_code(
                    task_type=step.task_type,
                    system=STEP_SYSTEM_PROMPT,
                    user_message=retry_prompt,
                    tools=self._get_schema_tools(),
                    tool_handlers=self._get_tool_handlers(),
                    complexity=step.complexity,
                )

            if not result.success:
                # Router exhausted all models
                raise RuntimeError(f"Code generation failed: {result.content}")

            code = result.content

            step.code = code

            self._emit_event(StepEvent(
                event_type="executing",
                step_number=step.number,
                data={"attempt": attempt, "code": code}
            ))

            # Track tables before execution
            tables_before = set(t['name'] for t in self.datastore.list_tables()) if self.datastore else set()

            # Execute
            exec_globals = self._get_execution_globals()
            result = self.executor.execute(code, exec_globals)

            # Auto-save any DataFrames or lists created during execution
            if result.success and self.datastore:
                self._auto_save_results(result.namespace, step.number)

            # Record artifacts in datastore
            if self.datastore:
                self.datastore.add_artifact(step.number, attempt, "code", code)
                if result.stdout:
                    self.datastore.add_artifact(step.number, attempt, "output", result.stdout)

            if result.success:
                duration_ms = int((time.time() - start_time) * 1000)

                # Detect new tables created
                tables_after = set(t['name'] for t in self.datastore.list_tables()) if self.datastore else set()
                tables_created = list(tables_after - tables_before)

                self._emit_event(StepEvent(
                    event_type="step_complete",
                    step_number=step.number,
                    data={"stdout": result.stdout, "attempts": attempt, "duration_ms": duration_ms, "tables_created": tables_created}
                ))

                return StepResult(
                    success=True,
                    stdout=result.stdout,
                    attempts=attempt,
                    duration_ms=duration_ms,
                    tables_created=tables_created,
                )

            # Prepare for retry
            last_code = code
            last_error = format_error_for_retry(result, code)

            # Record error artifact
            if self.datastore:
                self.datastore.add_artifact(step.number, attempt, "error", last_error)

            self._emit_event(StepEvent(
                event_type="step_error",
                step_number=step.number,
                data={"error": last_error, "attempt": attempt}
            ))

        # Max retries exceeded
        duration_ms = int((time.time() - start_time) * 1000)
        return StepResult(
            success=False,
            stdout="",
            error=f"Failed after {self.session_config.max_retries_per_step} attempts. Last error: {last_error}",
            attempts=self.session_config.max_retries_per_step,
            duration_ms=duration_ms,
        )

    def _request_approval(
        self,
        problem: str,
        planner_response: PlannerResponse,
        mode_selection,
    ) -> PlanApprovalResponse:
        """
        Request approval for a plan.

        If auto_approve is set or no callback is registered, auto-approves.
        Otherwise calls the registered callback.

        Args:
            problem: The original problem
            planner_response: The planner's response with plan and reasoning
            mode_selection: The selected execution mode

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
            }
            for step in planner_response.plan.steps
        ]

        request = PlanApprovalRequest(
            problem=problem,
            mode=mode_selection.mode,
            mode_reasoning=mode_selection.reasoning,
            steps=steps,
            reasoning=planner_response.reasoning,
        )

        return self._approval_callback(request)

    def _classify_question(self, problem: str) -> str:
        """
        Classify whether a question requires data analysis or can be answered directly.

        Returns:
            QuestionType.DATA_ANALYSIS - needs database queries
            QuestionType.GENERAL_KNOWLEDGE - LLM can answer directly
            QuestionType.META_QUESTION - about system capabilities
        """
        # Fast path for meta-questions
        if is_meta_question(problem):
            return QuestionType.META_QUESTION

        schema_overview = self.schema_manager.get_overview()

        prompt = f"""Classify this question into one of these categories:

Question: "{problem}"

Available data sources:
{schema_overview}

Categories:
1. DATA_ANALYSIS - The question requires querying the available databases/data sources to answer
2. GENERAL_KNOWLEDGE - The question is about general facts, concepts, or information that doesn't require the data sources

Respond with just the category name: DATA_ANALYSIS or GENERAL_KNOWLEDGE"""

        result = self.router.execute(
            task_type=TaskType.INTENT_CLASSIFICATION,
            system="You are a question classifier. Be brief.",
            user_message=prompt,
            max_tokens=50,
        )

        response = result.content.strip().upper()
        if "GENERAL" in response:
            return QuestionType.GENERAL_KNOWLEDGE
        return QuestionType.DATA_ANALYSIS

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

    def _answer_meta_question(self, problem: str) -> dict:
        """
        Answer meta-questions about capabilities without planning/execution.

        Uses schema overview and domain context to answer questions like
        "what questions can you answer" directly.
        """
        schema_overview = self.schema_manager.get_overview()
        domain_context = self.config.system_prompt or ""

        prompt = f"""The user is asking about your capabilities. Answer based on the available data.

User question: {problem}

Available databases and tables:
{schema_overview}

Domain context:
{domain_context}

Provide a helpful summary of:
1. What data sources are available
2. What types of questions can be answered
3. Example questions the user could ask

Keep it concise and actionable."""

        result = self.router.execute(
            task_type=TaskType.SUMMARIZATION,
            system="You are a helpful assistant explaining data analysis capabilities.",
            user_message=prompt,
        )

        return {
            "success": True,
            "meta_response": True,
            "output": result.content,
            "plan": None,
        }

    def _synthesize_answer(self, problem: str, step_outputs: str) -> str:
        """
        Synthesize a final user-facing answer from step execution outputs.

        This takes the raw step outputs (which may be verbose technical details)
        and creates a clear, direct answer to the user's original question.
        """
        prompt = f"""You are synthesizing results from a multi-step data analysis.

Original question: {problem}

Analysis results from each step:
{step_outputs}

Create a clear, direct answer to the user's question. Include:
1. A direct answer to their question (the main finding)
2. Key supporting data points or insights
3. Any notable observations or caveats

Format the answer for readability with markdown. Be concise but complete."""

        result = self.router.execute(
            task_type=TaskType.SUMMARIZATION,
            system="You are a data analyst presenting findings. Be clear and direct.",
            user_message=prompt,
            max_tokens=1000,
        )

        return result.content

    def _replan_with_feedback(self, problem: str, feedback: str) -> PlannerResponse:
        """
        Generate a new plan incorporating user feedback.

        Args:
            problem: Original problem
            feedback: User's suggested changes

        Returns:
            New PlannerResponse with updated plan
        """
        enhanced_problem = f"""{problem}

User feedback on previous plan:
{feedback}

Please create a revised plan that addresses this feedback."""

        return self.planner.plan(enhanced_problem)

    def solve(self, problem: str) -> dict:
        """
        Solve a problem with multi-step planning and execution.

        Workflow:
        1. Check for meta-questions (capability queries)
        2. Determine execution mode (exploratory vs auditable)
        3. Generate plan
        4. Request user approval (if require_approval is True)
           - If approved: execute
           - If rejected: return without executing
           - If suggestions: replan and ask again
        5. Execute steps in parallel waves

        Args:
            problem: Natural language problem to solve

        Returns:
            Dict with plan, results, and summary
        """
        # Classify question and handle non-data-analysis questions directly
        question_type = self._classify_question(problem)
        if question_type == QuestionType.META_QUESTION:
            return self._answer_meta_question(problem)
        elif question_type == QuestionType.GENERAL_KNOWLEDGE:
            return self._answer_general_question(problem)

        # Create session
        db_names = list(self.config.databases.keys())
        self.session_id = self.history.create_session(
            config_dict=self.config.model_dump(),
            databases=db_names,
        )

        # Initialize session state
        self.scratchpad = Scratchpad(initial_context=f"Problem: {problem}")

        # Create persistent datastore for this session
        session_dir = self.history._session_dir(self.session_id)
        datastore_path = session_dir / "datastore.db"
        self.datastore = DataStore(db_path=datastore_path)

        # Save problem statement to datastore (for UI restoration)
        self.datastore.set_session_meta("problem", problem)
        self.datastore.set_session_meta("status", "planning")

        # Determine execution mode
        mode_selection = suggest_mode(problem)

        # Generate plan with approval loop
        current_problem = problem
        replan_attempt = 0

        while replan_attempt <= self.session_config.max_replan_attempts:
            # Emit planning start event
            self._emit_event(StepEvent(
                event_type="planning_start",
                step_number=0,
                data={"message": "Analyzing data sources and creating plan..."}
            ))

            # Generate plan
            planner_response = self.planner.plan(current_problem)
            self.plan = planner_response.plan

            # Emit planning complete event
            self._emit_event(StepEvent(
                event_type="planning_complete",
                step_number=0,
                data={"steps": len(self.plan.steps)}
            ))

            # Request approval if required
            if self.session_config.require_approval:
                approval = self._request_approval(problem, planner_response, mode_selection)

                if approval.decision == PlanApproval.REJECT:
                    # User rejected the plan
                    self.datastore.set_session_meta("status", "rejected")
                    self.history.complete_session(self.session_id, status="rejected")
                    return {
                        "success": False,
                        "rejected": True,
                        "plan": self.plan,
                        "reason": approval.reason,
                        "message": "Plan was rejected by user.",
                    }

                elif approval.decision == PlanApproval.SUGGEST:
                    # User wants changes - replan with feedback
                    replan_attempt += 1
                    if replan_attempt > self.session_config.max_replan_attempts:
                        self.datastore.set_session_meta("status", "max_replans_exceeded")
                        self.history.complete_session(self.session_id, status="failed")
                        return {
                            "success": False,
                            "plan": self.plan,
                            "error": f"Maximum replan attempts ({self.session_config.max_replan_attempts}) exceeded.",
                        }

                    # Emit replan event
                    self._emit_event(StepEvent(
                        event_type="replanning",
                        step_number=0,
                        data={
                            "attempt": replan_attempt,
                            "feedback": approval.suggestion,
                        }
                    ))

                    # Replan with feedback
                    current_problem = f"{problem}\n\nUser feedback: {approval.suggestion}"
                    continue  # Go back to planning

                # APPROVE - proceed with execution
                break
            else:
                # No approval required - proceed
                break

        # Save plan to datastore (for UI restoration)
        self.datastore.set_session_meta("status", "executing")
        self.datastore.set_session_meta("mode", mode_selection.mode.value)
        for step in self.plan.steps:
            self.datastore.save_plan_step(
                step_number=step.number,
                goal=step.goal,
                expected_inputs=step.expected_inputs,
                expected_outputs=step.expected_outputs,
                status="pending",
            )

        # Emit plan_ready event BEFORE execution starts
        self._emit_event(StepEvent(
            event_type="plan_ready",
            step_number=0,
            data={
                "steps": [
                    {"number": s.number, "goal": s.goal, "depends_on": s.depends_on}
                    for s in self.plan.steps
                ],
                "reasoning": planner_response.reasoning,
            }
        ))

        # Execute steps in parallel waves based on dependencies
        all_results = []
        execution_waves = self.plan.get_execution_order()

        for wave_num, wave_step_nums in enumerate(execution_waves):
            # Get steps for this wave
            wave_steps = [self.plan.get_step(num) for num in wave_step_nums]
            wave_steps = [s for s in wave_steps if s is not None]

            # Execute all steps in this wave in parallel
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(wave_steps)) as executor:
                # Submit all steps in wave
                future_to_step = {}
                for step in wave_steps:
                    step.status = StepStatus.RUNNING
                    self.datastore.update_plan_step(step.number, status="running")
                    self._emit_event(StepEvent(
                        event_type="wave_step_start",
                        step_number=step.number,
                        data={"wave": wave_num + 1, "goal": step.goal}
                    ))
                    future = executor.submit(self._execute_step, step)
                    future_to_step[future] = step

                # Collect results as they complete
                wave_failed = False
                for future in concurrent.futures.as_completed(future_to_step):
                    step = future_to_step[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        result = StepResult(
                            success=False,
                            stdout="",
                            error=str(e),
                            attempts=1,
                        )

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
                                status="failed",
                                code=step.code,
                                error=result.error,
                                attempts=result.attempts,
                                duration_ms=result.duration_ms,
                            )
                        wave_failed = True
                        all_results.append(result)

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

        # Record successful completion
        total_duration = sum(r.duration_ms for r in all_results)
        total_attempts = sum(r.attempts for r in all_results)

        # Combine all step outputs
        combined_output = "\n\n".join([
            f"Step {i+1}: {self.plan.steps[i].goal}\n{r.stdout}"
            for i, r in enumerate(all_results)
        ])

        # Synthesize final answer from step results
        self._emit_event(StepEvent(
            event_type="synthesizing",
            step_number=0,
            data={"message": "Synthesizing final answer..."}
        ))

        final_answer = self._synthesize_answer(problem, combined_output)

        self._emit_event(StepEvent(
            event_type="answer_ready",
            step_number=0,
            data={"answer": final_answer}
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

        return {
            "success": True,
            "plan": self.plan,
            "results": all_results,
            "output": combined_output,
            "final_answer": final_answer,
            "scratchpad": self.scratchpad.to_markdown(),
            "datastore_tables": self.datastore.list_tables() if self.datastore else [],
            "datastore_path": str(self.datastore.db_path) if self.datastore and self.datastore.db_path else None,
        }

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

        self.session_id = session_id

        # Load the datastore (contains tables, state, scratchpad, artifacts)
        session_dir = self.history._session_dir(session_id)
        datastore_path = session_dir / "datastore.db"

        if datastore_path.exists():
            self.datastore = DataStore(db_path=datastore_path)

            # Rebuild scratchpad from datastore
            scratchpad_entries = self.datastore.get_scratchpad()
            if scratchpad_entries:
                # Get the original problem from the first query
                if session_detail.queries:
                    initial_context = f"Problem: {session_detail.queries[0].question}"
                else:
                    initial_context = ""
                self.scratchpad = Scratchpad(initial_context=initial_context)

                # Add each step result
                for entry in scratchpad_entries:
                    self.scratchpad.add_step_result(
                        step_number=entry["step_number"],
                        goal=entry["goal"],
                        result=entry["narrative"],
                        tables_created=entry.get("tables_created", []),
                    )
        else:
            # No datastore file - create empty one
            self.datastore = DataStore(db_path=datastore_path)

        return True

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
                max_tokens=300,
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

        except Exception:
            # Default to treating as new request
            return {
                "intent": "NEW_REQUEST",
                "facts": [],
                "revision": None,
                "new_request": user_text,
            }

    def follow_up(self, question: str, auto_classify: bool = True) -> dict:
        """
        Ask a follow-up question that builds on the current session's context.

        The follow-up has access to all tables and state from previous steps.
        If there are unresolved facts, the system will first try to extract
        facts from the user's message.

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

        # Check for unresolved facts and try to extract facts from user message
        unresolved = self.fact_resolver.get_unresolved_facts()
        extracted_facts = []

        if auto_classify and (unresolved or "=" in question or any(c.isdigit() for c in question)):
            # Try to extract facts from the message
            extracted_facts = self.fact_resolver.add_user_facts_from_text(question)

            if extracted_facts:
                # Clear unresolved status to allow re-resolution
                self.fact_resolver.clear_unresolved()

        # Get context from previous work
        existing_tables = self.datastore.list_tables()
        existing_state = self.datastore.get_all_state()
        scratchpad_context = self.datastore.get_scratchpad_as_markdown()

        # Calculate next step number
        existing_scratchpad = self.datastore.get_scratchpad()
        next_step_number = max((e["step_number"] for e in existing_scratchpad), default=0) + 1

        # Generate a plan for the follow-up, providing context
        context_prompt = f"""Previous work in this session:

{scratchpad_context}

Available tables from previous steps:
{', '.join(t['name'] for t in existing_tables) if existing_tables else '(none)'}

Available state variables:
{existing_state if existing_state else '(none)'}

Follow-up question: {question}
"""
        # Emit planning start event
        self._emit_event(StepEvent(
            event_type="planning_start",
            step_number=0,
            data={"message": "Planning follow-up analysis..."}
        ))

        # Generate plan for follow-up
        planner_response = self.planner.plan(context_prompt)
        follow_up_plan = planner_response.plan

        # Emit planning complete event
        self._emit_event(StepEvent(
            event_type="planning_complete",
            step_number=0,
            data={"steps": len(follow_up_plan.steps)}
        ))

        # Renumber steps to continue from where we left off
        for i, step in enumerate(follow_up_plan.steps):
            step.number = next_step_number + i

        # Emit plan_ready event for display
        self._emit_event(StepEvent(
            event_type="plan_ready",
            step_number=0,
            data={
                "steps": [
                    {"number": s.number, "goal": s.goal, "depends_on": s.depends_on}
                    for s in follow_up_plan.steps
                ],
                "reasoning": planner_response.reasoning,
            }
        ))

        # Execute each step
        all_results = []
        for step in follow_up_plan.steps:
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

        # Record successful follow-up
        total_duration = sum(r.duration_ms for r in all_results)
        total_attempts = sum(r.attempts for r in all_results)

        combined_output = "\n\n".join([
            f"Step {step.number}: {step.goal}\n{r.stdout}"
            for step, r in zip(follow_up_plan.steps, all_results)
        ])

        self.history.record_query(
            session_id=self.session_id,
            question=question,
            success=True,
            attempts=total_attempts,
            duration_ms=total_duration,
            answer=combined_output,
        )

        return {
            "success": True,
            "plan": follow_up_plan,
            "results": all_results,
            "output": combined_output,
            "scratchpad": self.scratchpad.to_markdown(),
            "datastore_tables": self.datastore.list_tables() if self.datastore else [],
        }

    def get_state(self) -> dict:
        """Get current session state for inspection or resumption."""
        return {
            "session_id": self.session_id,
            "plan": self.plan,
            "scratchpad": self.scratchpad.to_markdown(),
            "state": self.datastore.get_all_state() if self.datastore else {},
            "completed_steps": self.plan.completed_steps if self.plan else [],
            "datastore_tables": self.datastore.list_tables() if self.datastore else [],
        }

    def get_unresolved_facts(self) -> list[dict]:
        """Get list of facts that could not be resolved."""
        return [f.to_dict() for f in self.fact_resolver.get_unresolved_facts()]

    def get_unresolved_summary(self) -> str:
        """Get human-readable summary of unresolved facts."""
        return self.fact_resolver.get_unresolved_summary()

    def provide_facts(self, user_text: str) -> dict:
        """
        Extract facts from user text and add to resolver cache.

        This is used in auditable mode when facts could not be resolved.
        The user provides facts in natural language, and the LLM extracts
        them into structured facts that can be used for re-resolution.

        Example:
            session.provide_facts("There were 1 million people at the march")
            # Extracts: march_attendance = 1000000

        Args:
            user_text: Natural language text containing facts

        Returns:
            Dict with:
                - extracted_facts: List of facts extracted and added
                - unresolved_remaining: List of still-unresolved facts
        """
        # Extract facts from user text
        extracted = self.fact_resolver.add_user_facts_from_text(user_text)

        # Clear unresolved facts to allow re-resolution
        self.fact_resolver.clear_unresolved()

        return {
            "extracted_facts": [f.to_dict() for f in extracted],
            "unresolved_remaining": [f.to_dict() for f in self.fact_resolver.get_unresolved_facts()],
        }

    def add_fact(self, fact_name: str, value, reasoning: str = None, **params) -> dict:
        """
        Explicitly add a fact to the resolver cache.

        This is a more direct way to provide facts than provide_facts(),
        useful when you know the exact fact name and value.

        Args:
            fact_name: Name of the fact (e.g., "march_attendance")
            value: The value to set
            reasoning: Optional explanation
            **params: Additional parameters for the fact

        Returns:
            Dict with the created fact
        """
        fact = self.fact_resolver.add_user_fact(
            fact_name=fact_name,
            value=value,
            reasoning=reasoning,
            **params,
        )
        return fact.to_dict()


def create_session(config_path: str) -> Session:
    """Create a session from a config file path."""
    config = Config.from_yaml(config_path)
    return Session(config)
