# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Single-shot query engine with automatic retry on errors."""

from dataclasses import dataclass, field
from typing import Optional

from constat.core.config import Config
from .executor import ExecutionResult, PythonExecutor, format_error_for_retry
from . import RETRY_PROMPT_TEMPLATE
from constat.providers.anthropic import AnthropicProvider
from constat.catalog.schema_manager import SchemaManager
from constat.discovery.doc_tools import DocumentDiscoveryTools, DOC_TOOL_SCHEMAS
from constat.discovery.concept_detector import ConceptDetector
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


# Engine system prompt (constat-owned) - base version
# API filtering details are injected conditionally by ConceptDetector
ENGINE_SYSTEM_PROMPT = """You are a data analyst assistant. Answer questions by writing Python code that queries data sources and prints the answer.

## Discovery Tools
Use these tools FIRST to explore available data:
- get_table_schema(table): Get column info (e.g., "sales.customers")
- find_relevant_tables(query): Semantic search for relevant tables
- list_documents(), search_documents(query): Find reference documents
- get_document(name), get_document_section(name, section): Read documents

## Code Environment
- `pd`: pandas, `np`: numpy (pre-imported)
- `db_<name>`: SQL database connections
- `api_<name>`: GraphQL/REST API clients (response is data payload directly, no 'data' wrapper)
- `file_<name>`: File paths for CSV/JSON/Parquet
- `send_email(to, subject, body, df=None)`: Email with optional attachment

## Data Loading
- SQL: `pd.read_sql("SELECT ...", db_<name>)`
- CSV: `pd.read_csv(file_<name>)`
- JSON: `pd.read_json(file_<name>)`
- Parquet: `pd.read_parquet(file_<name>)`

## Variable vs Hardcoded Values
- Relative terms ("today", "last month") → use `datetime.now()`, relative calculations
- Explicit values ("January 2006", "above 100") → hardcode

## Code Rules
1. Use discovery tools first to understand available data
2. For SQL use pd.read_sql(), for files use appropriate pd.read_* function
3. Print a clear, formatted answer at the end

## Output Format
Return ONLY Python code wrapped in ```python ... ``` markers."""


# Template combining engine prompt with schema and domain context
SYSTEM_PROMPT_TEMPLATE = """{engine_prompt}
{injected_sections}
## Available Databases
{schema_overview}
{api_overview}
{doc_overview}
## Domain Context
{domain_context}"""


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

        # Concept detector for conditional prompt injection
        self._concept_detector = ConceptDetector()
        self._concept_detector.initialize()

    def _build_system_prompt(self, query: str) -> str:
        """Build the full system prompt.

        Args:
            query: The user's question for concept detection

        Combines:
        - Engine prompt (constat-owned): code generation rules, tool usage
        - Injected sections: conditionally added based on query semantics
        - Schema overview (auto-generated): databases, tables, relationships
        - API overview (from config): available GraphQL/REST APIs
        - Document overview (from config): reference documents
        - Domain context (user-owned): business context, terminology
        """
        # Detect relevant concepts and inject specialized sections
        injected_sections = self._concept_detector.get_sections_for_prompt(
            query=query,
            target="engine",
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

                # Add GraphQL flavor-specific hints
                if api_config.type == "graphql" and api_config.graphql_flavor:
                    flavor = api_config.graphql_flavor.lower()
                    if flavor == "hasura":
                        api_lines.append(f"  **Hasura GraphQL** - Filter syntax:")
                        api_lines.append(f"    - where: `{{field: {{_eq: value}}}}` operators: _eq, _neq, _gt, _lt, _gte, _lte, _in, _like, _ilike")
                        api_lines.append(f"    - order_by: `{{field: asc}}` or `{{field: desc}}`")
                        api_lines.append(f"    - limit/offset: `limit: 10, offset: 0`")
                        api_lines.append(f"    - Aggregates: use `<table>_aggregate` for count, sum, avg, max, min")
                        api_lines.append(f"    - Example: `orders_aggregate(where: {{status: {{_eq: \"pending\"}}}}) {{ aggregate {{ count sum {{ total }} }} }}`")
                    elif flavor == "prisma":
                        api_lines.append(f"  **Prisma GraphQL** - Filter syntax:")
                        api_lines.append(f"    - where: `{{field: {{equals: value}}}}` operators: equals, not, in, notIn, lt, lte, gt, gte, contains, startsWith, endsWith")
                        api_lines.append(f"    - orderBy: `{{field: asc}}` or `{{field: desc}}`")
                        api_lines.append(f"    - take/skip: `take: 10, skip: 0`")
                    elif flavor == "apollo":
                        api_lines.append(f"  **Apollo GraphQL** - Check schema for filter arguments (no standard syntax)")
                    elif flavor == "relay":
                        api_lines.append(f"  **Relay GraphQL** - Uses connection pattern with edges/nodes, first/after for pagination")

            api_overview = "\n".join(api_lines)

        # Build document overview if configured
        doc_overview = ""
        if self.config.documents:
            doc_lines = ["\n## Reference Documents"]
            for name, doc_config in self.config.documents.items():
                desc = doc_config.description or doc_config.type
                doc_lines.append(f"- **{name}**: {desc}")
            doc_overview = "\n".join(doc_lines)

        # Build SQL dialect hints based on database types
        sql_hints = ""
        dialect_hints_map = {
            "sqlite": "SQLite: Use strftime('%Y-%m', date_col), date('now', '-12 months'). Do NOT use schema prefixes (use 'customers' not 'sales.customers').",
            "duckdb": "DuckDB: Use strftime(date_col, '%Y-%m'), current_date - interval '12 months'.",
            "postgresql": "PostgreSQL: Use to_char(date_col, 'YYYY-MM'), current_date - interval '12 months'.",
            "mysql": "MySQL: Use DATE_FORMAT(date_col, '%Y-%m'), DATE_SUB(CURDATE(), INTERVAL 12 MONTH).",
        }
        detected_dialects = set()
        for db_name, db_config in self.config.databases.items():
            if db_config.is_file_source():
                continue  # Skip file sources
            config_type = db_config.type or "sqlite"
            # Detect actual dialect from URI for 'sql' type
            if config_type == "sql" and db_config.uri:
                uri_lower = db_config.uri.lower()
                if uri_lower.startswith("sqlite"):
                    detected_dialects.add("sqlite")
                elif uri_lower.startswith("postgresql") or uri_lower.startswith("postgres"):
                    detected_dialects.add("postgresql")
                elif uri_lower.startswith("mysql"):
                    detected_dialects.add("mysql")
                elif uri_lower.startswith("duckdb"):
                    detected_dialects.add("duckdb")
            elif config_type in dialect_hints_map:
                detected_dialects.add(config_type)

        if detected_dialects:
            hint_lines = ["\n## SQL Dialect Notes"]
            for dialect in detected_dialects:
                if dialect in dialect_hints_map:
                    hint_lines.append(f"- **{dialect.upper()}**: {dialect_hints_map[dialect]}")
            sql_hints = "\n".join(hint_lines)

        return SYSTEM_PROMPT_TEMPLATE.format(
            engine_prompt=ENGINE_SYSTEM_PROMPT,
            injected_sections=injected_sections,
            schema_overview=self.schema_manager.get_brief_summary() + sql_hints,
            api_overview=api_overview,
            doc_overview=doc_overview,
            domain_context=self.config.system_prompt or "No additional domain context provided.",
        )

    def _get_tool_handlers(self) -> dict:
        """Get tool handler functions."""
        handlers = {
            "get_table_schema": lambda table: self.schema_manager.get_table_schema(table),
            "find_relevant_tables": lambda query, top_k=5: self.schema_manager.find_relevant_tables(
                query, top_k, doc_tools=self.doc_tools
            ),
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

        # Provide API clients for GraphQL/REST APIs
        if self.config.apis:
            from constat.catalog.api_executor import APIExecutor
            api_executor = APIExecutor(self.config)
            for api_name, api_config in self.config.apis.items():
                if api_config.type == "graphql":
                    # Create a GraphQL query function
                    globals_dict[f"api_{api_name}"] = lambda query, variables=None, _name=api_name: \
                        api_executor.execute_graphql(_name, query, variables)
                else:
                    # Create a REST call function
                    globals_dict[f"api_{api_name}"] = lambda operation, params=None, _name=api_name: \
                        api_executor.execute_rest(_name, operation, params or {})

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
        system_prompt = self._build_system_prompt(question)
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
