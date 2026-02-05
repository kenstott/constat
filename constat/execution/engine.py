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

## Discovery Tools (for planning only - NOT available in generated code)
These help understand available data BEFORE writing code:
- get_table_schema(table): Get column info (e.g., "sales.customers")
- find_relevant_tables(query): Semantic search for relevant tables
- list_documents(), search_documents(query): Find reference documents
- get_document(name), get_document_section(name, section): Read documents

NOTE: Do NOT call these functions in your generated code. Use schema info provided below to write queries directly.

## Code Environment
- `pd`: pandas, `np`: numpy (pre-imported)
- `db_<name>`: Database connections (SQL or NoSQL depending on config)
- `api_<name>`: GraphQL/REST API clients (response is data payload directly, no 'data' wrapper)
- `file_<name>`: File paths for CSV/JSON/Parquet
- `send_email(to, subject, body, format="markdown", df=None)`: Email with optional attachment. ALWAYS use format="markdown" when body contains markdown formatting (headers, lists, bold, etc.) - this converts to styled HTML

## Data Loading
**SQL databases** (SQLite, PostgreSQL, DuckDB):
- `pd.read_sql("SELECT ...", db_<name>)` - ALWAYS use pd.read_sql(), NEVER use db.execute()

**NoSQL databases** (MongoDB, Cassandra, Elasticsearch):
- MongoDB: `pd.DataFrame(list(db_<name>['collection'].find(query)))`
- Elasticsearch: `pd.DataFrame(db_<name>.query('index', query_dict))`
- Cassandra: `pd.DataFrame(db_<name>.query('table', filter_dict))`

**Files:**
- CSV: `pd.read_csv(file_<name>)`
- JSON: `pd.read_json(file_<name>)`
- Parquet: `pd.read_parquet(file_<name>)`

**CRITICAL for SQL**: Do NOT use `db.execute()` or `db_<name>.execute()` - this does not work. Use `pd.read_sql(query, db_<name>)` instead.

## Variable vs Hardcoded Values
- Relative terms ("today", "last month") → use `datetime.now()`, relative calculations
- Explicit values ("January 2006", "above 100") → hardcode

## Code Rules
1. Use discovery tools first to understand available data
2. Use appropriate access pattern for database type (see Data Loading above)
3. Print a clear, formatted answer at the end

## Error Prevention
- DataFrame: verify columns before access → `if 'col' in df.columns`
- SQL: confirm table/column names from schema before query
- Strings: complete all quotes, escape special chars
- Syntax: match all brackets/parens before execution

## Output Guidelines
- Print brief summaries and key metrics (e.g., "Found 150 employees, Average: $85,000")
- **NEVER print raw DataFrames** - produces unreadable output
- For final reports/exports: Use `viz` methods to save files (creates clickable file:// URIs)

## Table Naming Rules
- **RESERVED WORDS**: Do NOT use "final", "recommendation", "summary", "report", "result", "output" in names for INTERMEDIATE tables
- These words are reserved for the LAST step's output only
- Intermediate tables should use descriptive names like: `employee_reviews`, `salary_data`, `performance_scores`
- Example BAD: `final_calculations` (step 1), `summary_data` (step 2)
- Example GOOD: `performance_data` (step 1), `raise_calculations` (step 2), `final_recommendations` (step 3 - LAST step only)

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
        allowed_databases: Optional[set[str]] = None,
        allowed_apis: Optional[set[str]] = None,
        allowed_documents: Optional[set[str]] = None,
    ):
        """Initialize the code generation engine.

        Args:
            config: Configuration
            schema_manager: Schema manager for database metadata
            llm: LLM provider (defaults to Anthropic)
            max_retries: Max retry attempts per step
            allowed_databases: Set of allowed database names (None = no filtering)
            allowed_apis: Set of allowed API names (None = no filtering)
            allowed_documents: Set of allowed document names (None = no filtering)
        """
        self.config = config
        self.schema_manager = schema_manager
        self.allowed_databases = allowed_databases
        self.allowed_apis = allowed_apis
        self.allowed_documents = allowed_documents
        self.llm = llm or AnthropicProvider(
            api_key=config.llm.api_key,
            model=config.llm.model,
        )
        self.max_retries = max_retries
        self.executor = PythonExecutor(
            timeout_seconds=config.execution.timeout_seconds,
            allowed_imports=config.execution.allowed_imports or None,
        )
        # Initialize document discovery tools with permission filtering
        self.doc_tools = DocumentDiscoveryTools(
            config,
            allowed_documents=allowed_documents,
        ) if config.documents else None

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

        All sections are filtered based on allowed_databases/apis/documents.
        """
        # Detect relevant concepts and inject specialized sections
        injected_sections = self._concept_detector.get_sections_for_prompt(
            query=query,
            target="engine",
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

        # Build document overview if configured (filtered by permissions)
        doc_overview = ""
        if self.config.documents:
            doc_lines = ["\n## Reference Documents"]
            for name, doc_config in self.config.documents.items():
                # Skip documents not allowed by permissions
                if not self._is_document_allowed(name):
                    continue
                desc = doc_config.description or doc_config.type
                doc_lines.append(f"- **{name}**: {desc}")
            # Only include header if we have allowed documents
            if len(doc_lines) > 1:
                doc_overview = "\n".join(doc_lines)

        # Build SQL dialect hints based on database types (filtered by permissions)
        sql_hints = ""
        dialect_hints_map = {
            "sqlite": "SQLite: Use strftime('%Y-%m', date_col), date('now', '-12 months'). Do NOT use schema prefixes (use 'customers' not 'sales.customers').",
            "duckdb": "DuckDB: Use strftime(date_col, '%Y-%m'), current_date - interval '12 months'.",
            "postgresql": "PostgreSQL: Use to_char(date_col, 'YYYY-MM'), current_date - interval '12 months'.",
            "mysql": "MySQL: Use DATE_FORMAT(date_col, '%Y-%m'), DATE_SUB(CURDATE(), INTERVAL 12 MONTH).",
        }
        detected_dialects = set()
        for db_name, db_config in self.config.databases.items():
            # Skip databases not allowed by permissions
            if not self._is_database_allowed(db_name):
                continue
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
            schema_overview=self.schema_manager.get_brief_summary(self.allowed_databases) + sql_hints,
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

        # Provide all database connections and file paths from config
        first_sql_db = None
        config_db_names = set()
        for db_name, db_config in self.config.databases.items():
            config_db_names.add(db_name)
            if db_config.is_file_source():
                # For file sources, provide the path as file_<name>
                globals_dict[f"file_{db_name}"] = db_config.path
            else:
                # For SQL/NoSQL, provide connection as db_<name>
                conn = self.schema_manager.get_connection(db_name)
                globals_dict[f"db_{db_name}"] = conn
                if first_sql_db is None:
                    first_sql_db = conn

        # Also include dynamically added databases (from projects) not in config
        # SQL connections
        for db_name in self.schema_manager.connections.keys():
            if db_name not in config_db_names:
                conn = self.schema_manager.connections[db_name]
                globals_dict[f"db_{db_name}"] = conn
                if first_sql_db is None:
                    first_sql_db = conn
        # NoSQL connections
        for db_name in self.schema_manager.nosql_connections.keys():
            if db_name not in config_db_names:
                globals_dict[f"db_{db_name}"] = self.schema_manager.nosql_connections[db_name]
        # File connections
        for db_name in self.schema_manager.file_connections.keys():
            if db_name not in config_db_names:
                conn = self.schema_manager.file_connections[db_name]
                # For file connectors, provide the path
                if hasattr(conn, 'path'):
                    globals_dict[f"file_{db_name}"] = conn.path

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
