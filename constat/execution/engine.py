"""Single-shot query engine with automatic retry on errors."""

from dataclasses import dataclass, field
from typing import Optional

from constat.core.config import Config
from .executor import ExecutionResult, PythonExecutor, format_error_for_retry
from constat.providers.anthropic import AnthropicProvider
from constat.catalog.schema_manager import SchemaManager
from constat.discovery.doc_tools import DocumentDiscoveryTools, DOC_TOOL_SCHEMAS
from constat.email import create_send_email


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


# Tool definitions for Anthropic API - schema discovery
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

# Combined tools: schema + documents
ENGINE_TOOLS = SCHEMA_TOOLS + DOC_TOOL_SCHEMAS


# Engine system prompt (constat-owned) - controls code generation behavior
ENGINE_SYSTEM_PROMPT = """You are a data analyst assistant. Answer questions by writing Python code that queries data sources and prints the answer.

## Discovery Tools
Use these tools to explore available data sources BEFORE writing code:

- get_table_schema(table): Get column info for a table (e.g., "sales.customers", "metrics.web_metrics")
- find_relevant_tables(query): Semantic search for relevant tables

IMPORTANT: Use discovery tools FIRST to understand what data is available.

## Code Environment
Your code has access to:
- `pd`: pandas (pre-imported)
- `np`: numpy (pre-imported)
- SQL database connections: `db_<name>` (e.g., `db_sales`)
- File data source paths: `file_<name>` (e.g., `file_web_metrics`)
- `send_email(to, subject, body, df=None)`: Send email with optional DataFrame attachment

## Data Source Types

Data sources appear in schema discovery. Check the `database_type` field:
- `postgresql`, `mysql`, `sqlite`: SQL databases → use `pd.read_sql()`
- `mongodb`, `elasticsearch`, etc.: NoSQL → use connector methods
- `csv`: CSV files → use `pd.read_csv(file_<name>)`
- `json`: JSON files → use `pd.read_json(file_<name>)`
- `jsonl`: JSON Lines → use `pd.read_json(file_<name>, lines=True)`
- `parquet`: Parquet files → use `pd.read_parquet(file_<name>)`
- `arrow`, `feather`: Arrow files → use `pd.read_feather(file_<name>)`

## Data Loading Examples

**SQL Databases:**
```python
df = pd.read_sql("SELECT * FROM customers", db_sales)
```

**CSV Files:**
```python
df = pd.read_csv(file_web_metrics)  # file_<name> contains the path
```

**JSON Files:**
```python
df = pd.read_json(file_events)
```

**Parquet Files:**
```python
df = pd.read_parquet(file_transactions)
```

## Code Generation Rules
1. Use discovery tools to find relevant data sources and their types
2. Check the database_type to determine how to load the data
3. For SQL: use pd.read_sql() with db_<name> connection
4. For files: use pd.read_csv/json/parquet() with file_<name> path
5. Print a clear, formatted answer at the end

## Output Format
Return ONLY the Python code wrapped in ```python ... ``` markers.
Do not include explanations outside the code block."""


# Template combining engine prompt with schema and domain context
SYSTEM_PROMPT_TEMPLATE = """{engine_prompt}

## Available Databases
{schema_overview}

## Domain Context
{domain_context}"""


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
        # Initialize document discovery tools
        self.doc_tools = DocumentDiscoveryTools(config) if config.documents else None

    def _build_system_prompt(self) -> str:
        """Build the full system prompt.

        Combines:
        - Engine prompt (constat-owned): code generation rules, tool usage
        - Schema overview (auto-generated): databases, tables, relationships
        - Domain context (user-owned): business context, terminology
        """
        return SYSTEM_PROMPT_TEMPLATE.format(
            engine_prompt=ENGINE_SYSTEM_PROMPT,
            schema_overview=self.schema_manager.get_overview(),
            domain_context=self.config.system_prompt or "No additional domain context provided.",
        )

    def _get_tool_handlers(self) -> dict:
        """Get tool handler functions."""
        handlers = {
            "get_table_schema": lambda table: self.schema_manager.get_table_schema(table),
            "find_relevant_tables": lambda query, top_k=5: self.schema_manager.find_relevant_tables(query, top_k),
        }

        # Add document tools if available
        if self.doc_tools:
            handlers.update({
                "list_documents": self.doc_tools.list_documents,
                "get_document": lambda name: self.doc_tools.get_document(name),
                "search_documents": lambda query, limit=5: self.doc_tools.search_documents(query, limit),
                "get_document_section": lambda name, section: self.doc_tools.get_document_section(name, section),
            })

        return handlers

    def _get_tools(self) -> list:
        """Get tool schemas based on available tools."""
        if self.doc_tools:
            return ENGINE_TOOLS
        return SCHEMA_TOOLS

    def _get_execution_globals(self) -> dict:
        """Get globals dict for code execution."""
        globals_dict = {}

        # Provide all database connections and file paths
        first_sql_db = None
        for db_name, db_config in self.config.databases.items():
            if db_config.is_file_source():
                # For file sources, provide the path as file_<name>
                globals_dict[f"file_{db_name}"] = db_config.path
            else:
                # For SQL/NoSQL, provide connection as db_<name>
                conn = self.schema_manager.get_connection(db_name)
                globals_dict[f"db_{db_name}"] = conn
                if first_sql_db is None:
                    first_sql_db = conn

        # Backwards compat: 'db' alias for first SQL database
        if first_sql_db is not None:
            globals_dict["db"] = first_sql_db

        # Inject send_email function
        globals_dict["send_email"] = create_send_email(self.config.email)

        return globals_dict

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
                    tools=self._get_tools(),
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
                    tools=self._get_tools(),
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
