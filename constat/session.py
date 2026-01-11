"""Session orchestration for multi-step plan execution."""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from constat.core.config import Config
from constat.core.models import Plan, PlannerResponse, Step, StepResult, StepStatus, StepType
from constat.storage.datastore import DataStore
from constat.storage.history import SessionHistory
from constat.execution.executor import ExecutionResult, PythonExecutor, format_error_for_retry
from constat.execution.planner import Planner
from constat.execution.scratchpad import Scratchpad
from constat.providers.anthropic import AnthropicProvider
from constat.catalog.schema_manager import SchemaManager


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


@dataclass
class SessionConfig:
    """Configuration for a session."""
    max_retries_per_step: int = 10
    verbose: bool = False


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
    ):
        self.config = config
        self.session_config = session_config or SessionConfig()

        # Initialize components
        self.schema_manager = SchemaManager(config)
        self.schema_manager.initialize()

        self.llm = AnthropicProvider(
            api_key=config.llm.api_key,
            model=config.llm.model,
        )

        self.planner = Planner(config, self.schema_manager, self.llm)

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

        # Event callbacks for monitoring
        self._event_handlers: list[Callable[[StepEvent], None]] = []

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

    def _get_execution_globals(self) -> dict:
        """Get globals dict for code execution.

        Each step runs in isolation - only `store` (DuckDB) is shared.
        """
        globals_dict = {
            "store": self.datastore,  # Persistent datastore - only shared state between steps
        }

        # Provide database connections
        for i, db_config in enumerate(self.config.databases):
            conn = self.schema_manager.get_connection(db_config.name)
            globals_dict[f"db_{db_config.name}"] = conn
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
        skip_vars = {"store", "db", "pd", "np", "__builtins__"}
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

            # Generate code (use codegen tier model if configured)
            codegen_model = self.config.llm.get_model("codegen")

            if attempt == 1:
                prompt = self._build_step_prompt(step)
                code = self.llm.generate_code(
                    system=STEP_SYSTEM_PROMPT,
                    user_message=prompt,
                    tools=self._get_schema_tools(),
                    tool_handlers=self._get_tool_handlers(),
                    model=codegen_model,
                )
            else:
                retry_prompt = RETRY_PROMPT_TEMPLATE.format(
                    error_details=last_error,
                    previous_code=last_code,
                )
                code = self.llm.generate_code(
                    system=STEP_SYSTEM_PROMPT,
                    user_message=retry_prompt,
                    tools=self._get_schema_tools(),
                    tool_handlers=self._get_tool_handlers(),
                    model=codegen_model,
                )

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

    def solve(self, problem: str) -> dict:
        """
        Solve a problem with multi-step planning and execution.

        Args:
            problem: Natural language problem to solve

        Returns:
            Dict with plan, results, and summary
        """
        # Create session
        db_names = [db.name for db in self.config.databases]
        self.session_id = self.history.create_session(
            config_dict=self.config.model_dump(),
            databases=db_names,
        )

        # Initialize session state
        self.scratchpad = Scratchpad(initial_context=f"Problem: {problem}")

        # Create persistent datastore for this session
        session_dir = self.history._session_dir(self.session_id)
        datastore_path = session_dir / "datastore.duckdb"
        self.datastore = DataStore(db_path=datastore_path)

        # Save problem statement to datastore (for UI restoration)
        self.datastore.set_session_meta("problem", problem)
        self.datastore.set_session_meta("status", "planning")

        # Generate plan
        planner_response = self.planner.plan(problem)
        self.plan = planner_response.plan

        # Save plan to datastore (for UI restoration)
        self.datastore.set_session_meta("status", "executing")
        for step in self.plan.steps:
            self.datastore.save_plan_step(
                step_number=step.number,
                goal=step.goal,
                expected_inputs=step.expected_inputs,
                expected_outputs=step.expected_outputs,
                status="pending",
            )

        # Execute each step
        all_results = []
        for step in self.plan.steps:
            step.status = StepStatus.RUNNING
            self.datastore.update_plan_step(step.number, status="running")

            result = self._execute_step(step)

            if result.success:
                self.plan.mark_step_completed(step.number, result)
                # Update both scratchpad (in-memory) and datastore (persistent)
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
                    # Update plan step in datastore (for UI restoration)
                    self.datastore.update_plan_step(
                        step.number,
                        status="completed",
                        code=step.code,
                        attempts=result.attempts,
                        duration_ms=result.duration_ms,
                    )
            else:
                self.plan.mark_step_failed(step.number, result)
                # Update plan step in datastore (for UI restoration)
                if self.datastore:
                    self.datastore.update_plan_step(
                        step.number,
                        status="failed",
                        code=step.code,
                        error=result.error,
                        attempts=result.attempts,
                        duration_ms=result.duration_ms,
                    )
                    self.datastore.set_session_meta("status", "failed")
                # Record and stop on failure
                self.history.record_query(
                    session_id=self.session_id,
                    question=problem,
                    success=False,
                    attempts=result.attempts,
                    duration_ms=result.duration_ms,
                    error=result.error,
                )
                self.history.complete_session(self.session_id, status="failed")
                return {
                    "success": False,
                    "plan": self.plan,
                    "error": result.error,
                    "completed_steps": self.plan.completed_steps,
                }

            all_results.append(result)

        # Record successful completion
        total_duration = sum(r.duration_ms for r in all_results)
        total_attempts = sum(r.attempts for r in all_results)

        # Combine all step outputs
        combined_output = "\n\n".join([
            f"Step {i+1}: {self.plan.steps[i].goal}\n{r.stdout}"
            for i, r in enumerate(all_results)
        ])

        self.history.record_query(
            session_id=self.session_id,
            question=problem,
            success=True,
            attempts=total_attempts,
            duration_ms=total_duration,
            answer=combined_output,
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
        datastore_path = session_dir / "datastore.duckdb"

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

    def follow_up(self, question: str) -> dict:
        """
        Ask a follow-up question that builds on the current session's context.

        The follow-up has access to all tables and state from previous steps.

        Args:
            question: The follow-up question

        Returns:
            Dict with plan, results, and summary (same format as solve())
        """
        if not self.session_id:
            raise ValueError("No active session. Call solve() or resume() first.")

        if not self.datastore:
            raise ValueError("No datastore available. Session may not have been properly initialized.")

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
        # Generate plan for follow-up
        planner_response = self.planner.plan(context_prompt)
        follow_up_plan = planner_response.plan

        # Renumber steps to continue from where we left off
        for i, step in enumerate(follow_up_plan.steps):
            step.number = next_step_number + i

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


def create_session(config_path: str) -> Session:
    """Create a session from a config file path."""
    config = Config.from_yaml(config_path)
    return Session(config)
