"""Single-shot query engine with automatic retry on errors."""

from dataclasses import dataclass, field
from typing import Optional

from .config import Config
from .executor import ExecutionResult, PythonExecutor, format_error_for_retry
from .providers.anthropic import AnthropicProvider
from .schema_manager import SchemaManager


@dataclass
class QueryResult:
    """Result of a single-shot query."""
    success: bool
    answer: str
    code: str
    attempts: int
    error: Optional[str] = None
    # History of attempts for debugging
    attempt_history: list[dict] = field(default_factory=list)


# Tool definitions for Anthropic API
SCHEMA_TOOLS = [
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
        "description": "Search for tables relevant to a query using semantic similarity. Returns tables with relevance scores.",
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


SYSTEM_PROMPT_TEMPLATE = """You are a data analyst assistant. Answer questions by writing Python code that queries databases and prints the answer.

## Available Databases
{schema_overview}

## Domain Context
{domain_context}

## Tools
You have access to these tools to explore the schema:
- get_table_schema(table): Get detailed column info for a table
- find_relevant_tables(query): Search for tables relevant to your query

## Code Environment
Your code will have access to:
- `db`: SQLAlchemy engine connected to the database
- `pd`: pandas (imported as pd)
- `np`: numpy (imported as np)

## Instructions
1. First, use tools to understand which tables and columns you need
2. Write Python code that queries the database and prints the answer
3. Use pandas read_sql() with the db connection
4. Print a clear, formatted answer at the end
5. Keep code simple and focused on answering the question

## Output Format
Return ONLY the Python code wrapped in ```python ... ``` markers.
Do not include explanations outside the code block."""


RETRY_PROMPT_TEMPLATE = """Your previous code failed to execute.

{error_details}

Previous code:
```python
{previous_code}
```

Please fix the code and try again. Return ONLY the corrected Python code wrapped in ```python ... ``` markers."""


class QueryEngine:
    """
    Single-shot query engine that converts natural language to executable Python.

    Features:
    - Schema context via overview + on-demand tools
    - Automatic retry on compile and runtime errors
    - Configurable retry limit
    """

    def __init__(
        self,
        config: Config,
        schema_manager: SchemaManager,
        llm: Optional[AnthropicProvider] = None,
        max_retries: int = 10,
    ):
        self.config = config
        self.schema_manager = schema_manager
        self.llm = llm or AnthropicProvider(
            api_key=config.llm.api_key,
            model=config.llm.model,
        )
        self.max_retries = max_retries
        self.executor = PythonExecutor(
            timeout_seconds=config.execution.timeout_seconds,
            allowed_imports=config.execution.allowed_imports or None,
        )

    def _build_system_prompt(self) -> str:
        """Build the system prompt with schema context."""
        return SYSTEM_PROMPT_TEMPLATE.format(
            schema_overview=self.schema_manager.get_overview(),
            domain_context=self.config.system_prompt or "No additional domain context provided.",
        )

    def _get_tool_handlers(self) -> dict:
        """Get tool handler functions."""
        return {
            "get_table_schema": lambda table: self.schema_manager.get_table_schema(table),
            "find_relevant_tables": lambda query, top_k=5: self.schema_manager.find_relevant_tables(query, top_k),
        }

    def _get_execution_globals(self) -> dict:
        """Get globals dict for code execution."""
        # For now, assume single database - get first connection
        db_name = self.config.databases[0].name if self.config.databases else None
        db = self.schema_manager.get_connection(db_name) if db_name else None

        return {
            "db": db,
        }

    def query(self, question: str) -> QueryResult:
        """
        Answer a natural language question by generating and executing Python code.

        Args:
            question: Natural language question about the data

        Returns:
            QueryResult with answer, code, and attempt history
        """
        system_prompt = self._build_system_prompt()
        tool_handlers = self._get_tool_handlers()
        exec_globals = self._get_execution_globals()

        attempt_history = []
        current_prompt = question
        last_code = ""
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            # Generate code
            if attempt == 1:
                code = self.llm.generate_code(
                    system=system_prompt,
                    user_message=current_prompt,
                    tools=SCHEMA_TOOLS,
                    tool_handlers=tool_handlers,
                )
            else:
                # Retry prompt with error context
                retry_message = RETRY_PROMPT_TEMPLATE.format(
                    error_details=last_error,
                    previous_code=last_code,
                )
                code = self.llm.generate_code(
                    system=system_prompt,
                    user_message=retry_message,
                    tools=SCHEMA_TOOLS,
                    tool_handlers=tool_handlers,
                )

            # Execute code
            result = self.executor.execute(code, exec_globals)

            # Record attempt
            attempt_history.append({
                "attempt": attempt,
                "code": code,
                "success": result.success,
                "stdout": result.stdout,
                "error": result.error_message(),
            })

            if result.success:
                return QueryResult(
                    success=True,
                    answer=result.stdout.strip(),
                    code=code,
                    attempts=attempt,
                    attempt_history=attempt_history,
                )

            # Prepare for retry
            last_code = code
            last_error = format_error_for_retry(result, code)

        # Max retries exceeded
        return QueryResult(
            success=False,
            answer="",
            code=last_code,
            attempts=self.max_retries,
            error=f"Failed after {self.max_retries} attempts. Last error: {last_error}",
            attempt_history=attempt_history,
        )


def create_engine(config_path: str) -> QueryEngine:
    """Create a QueryEngine from a config file path."""
    config = Config.from_yaml(config_path)
    schema_manager = SchemaManager(config)
    schema_manager.initialize()
    return QueryEngine(config, schema_manager)
