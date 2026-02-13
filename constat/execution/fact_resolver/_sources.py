# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Sources mixin: source resolution methods."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Optional

from ._types import (
    Fact,
    FactSource,
    ARRAY_ROW_THRESHOLD,
    ARRAY_SIZE_THRESHOLD,
    ProofNode,
    ResolutionSpec,
)

if TYPE_CHECKING:
    from . import FactResolver

logger = logging.getLogger(__name__)


class SourcesMixin:

    def _try_resolve(
        self: "FactResolver",
        source: FactSource,
        fact_name: str,
        params: dict,
        cache_key: str
    ) -> Optional[Fact]:
        """Try to resolve from a specific source."""
        import logging
        logger = logging.getLogger(__name__)

        if source == FactSource.CACHE:
            cached = self._cache.get(cache_key)
            if cached:
                # For table facts, verify the table still exists in datastore
                if cached.table_name and self._datastore:
                    try:
                        existing_tables = [t["name"] for t in self._datastore.list_tables()]
                        if cached.table_name not in existing_tables:
                            logger.debug(f"[_try_resolve] CACHE table {cached.table_name} no longer exists")
                            return None  # Table doesn't exist, need to re-resolve
                    except Exception as e:
                        logger.debug(f"[_try_resolve] Cache validation failed for {cache_key}, assuming valid: {e}")
                logger.debug(f"[_try_resolve] CACHE hit for {cache_key}")
            return cached

        elif source == FactSource.CONFIG:
            result = self._resolve_from_config(fact_name, params)
            logger.debug(f"[_try_resolve] CONFIG for {fact_name}: {result is not None}")
            return result

        elif source == FactSource.RULE:
            result = self._resolve_from_rule(fact_name, params)
            logger.debug(f"[_try_resolve] RULE for {fact_name}: {result is not None}")
            return result

        elif source == FactSource.DATABASE:
            logger.debug(f"_try_resolve DATABASE attempting for {fact_name}")
            logger.debug(f"[_try_resolve] DATABASE attempting for {fact_name}")
            result = self._resolve_from_database(fact_name, params)
            logger.debug(f"_try_resolve DATABASE for {fact_name}: result={result is not None}")
            logger.debug(f"[_try_resolve] DATABASE for {fact_name}: {result is not None}")
            return result

        elif source == FactSource.DOCUMENT:
            logger.debug(f"_try_resolve DOCUMENT attempting for {fact_name}")
            logger.debug(f"[_try_resolve] DOCUMENT attempting for {fact_name}")
            result = self._resolve_from_document(fact_name, params)
            logger.debug(f"_try_resolve DOCUMENT for {fact_name}: result={result is not None}")
            logger.debug(f"[_try_resolve] DOCUMENT for {fact_name}: {result is not None}")
            return result

        elif source == FactSource.API:
            logger.debug(f"[_try_resolve] API attempting for {fact_name}")
            result = self._resolve_from_api(fact_name, params)
            logger.debug(f"[_try_resolve] API for {fact_name}: {result is not None}")
            return result

        elif source == FactSource.LLM_KNOWLEDGE:
            logger.debug(f"[_try_resolve] LLM_KNOWLEDGE attempting for {fact_name}")
            result = self._resolve_from_llm(fact_name, params)
            logger.debug(f"[_try_resolve] LLM_KNOWLEDGE for {fact_name}: {result is not None}")
            return result

        elif source == FactSource.USER_PROVIDED:
            # User-provided facts are only in cache (added via add_user_fact)
            # This is a fallback - if we reach here, fact is unresolved
            logger.debug(f"[_try_resolve] USER_PROVIDED - no fallback for {fact_name}")
            return None

        elif source == FactSource.SUB_PLAN:
            logger.debug(f"[_try_resolve] SUB_PLAN attempting for {fact_name}")
            result = self._resolve_from_sub_plan(fact_name, params)
            logger.debug(f"[_try_resolve] SUB_PLAN for {fact_name}: {result is not None}")
            return result

        return None

    def _resolve_from_config(self: "FactResolver", fact_name: str, params: dict) -> Optional[Fact]:
        """Check if fact is defined in config/system prompt."""
        if not self.config:
            return None

        # TODO: Parse config.system_prompt for defined facts/thresholds
        # For now, return None
        return None

    def _resolve_from_rule(self: "FactResolver", fact_name: str, params: dict) -> Optional[Fact]:
        """Try to derive fact using a registered rule."""
        rule = self._rules.get(fact_name)
        if not rule:
            return None

        try:
            return rule(self, params)
        except Exception as e:
            # Rule failed - log but don't crash
            return None

    def _resolve_from_api(self: "FactResolver", fact_name: str, params: dict, lazy: bool = True) -> Optional[Fact]:
        """Resolve a fact by querying external APIs.

        Uses LLM to determine which API to query and how.

        Args:
            fact_name: Name of the fact to resolve
            params: Parameters for the fact
            lazy: If True (default), return a binding without executing the API.
                  The binding shows which API/endpoint will be used.
                  If False, execute the API and return actual data.

        Returns:
            Fact with API binding (lazy) or actual data (eager)
        """
        import logging
        import json
        logger = logging.getLogger(__name__)

        if not self.config or not self.config.apis:
            logger.debug(f"[_resolve_from_api] No APIs configured")
            return None

        if not self.llm:
            logger.debug(f"[_resolve_from_api] No LLM available for API query generation")
            return None

        # Build API overview for LLM
        api_overview_lines = []
        api_descriptions = {}
        for api_name, api_config in self.config.apis.items():
            api_type = api_config.type.upper()
            desc = api_config.description or f"{api_type} endpoint"
            url = api_config.url or ""
            api_overview_lines.append(f"- {api_name} ({api_type}): {desc}")
            api_descriptions[api_name] = {"type": api_type, "desc": desc, "url": url}
            if url:
                api_overview_lines.append(f"  URL: {url}")
        api_overview = "\n".join(api_overview_lines)

        # Ask LLM which API can provide this fact and how to query it
        prompt = f"""I need to resolve this fact from an external API:

Fact: {fact_name}
Parameters: {json.dumps(params) if params else "none"}

Available APIs:
{api_overview}

If this fact can be resolved from one of these APIs, provide the query details.
Otherwise respond NOT_POSSIBLE.

For GraphQL APIs, respond in this format:
API: <api_name>
GRAPHQL: <graphql_query>

For REST/OpenAPI APIs, respond in this format:
API: <api_name>
REST: <endpoint_path>
METHOD: GET|POST|etc
PARAMS: {{"key": "value"}}

If this fact cannot be resolved from any available API, respond:
NOT_POSSIBLE: <reason>
"""

        try:
            response = self.llm.generate(
                system="You are an API expert. Determine which API can provide the requested data and how to query it.",
                user_message=prompt,
                max_tokens=self.llm.max_output_tokens,
            )

            if "NOT_POSSIBLE" in response:
                logger.debug(f"[_resolve_from_api] LLM says not possible: {response}")
                return None

            # Parse the response
            lines = response.strip().split("\n")
            api_name = None
            graphql_query = None
            rest_endpoint = None
            rest_method = "GET"
            rest_params = {}

            for idx, line in enumerate(lines):
                line = line.strip()
                if line.startswith("API:"):
                    api_name = line.split(":", 1)[1].strip()
                elif line.startswith("GRAPHQL:"):
                    graphql_query = line.split(":", 1)[1].strip()
                    # Collect multi-line GraphQL query
                    for subsequent in lines[idx + 1:]:
                        if subsequent.strip().startswith(("API:", "REST:", "METHOD:", "PARAMS:", "NOT_POSSIBLE")):
                            break
                        graphql_query += "\n" + subsequent
                    graphql_query = graphql_query.strip()
                    # Clean up code blocks
                    graphql_query = graphql_query.replace("```graphql", "").replace("```", "").strip()
                elif line.startswith("REST:"):
                    rest_endpoint = line.split(":", 1)[1].strip()
                elif line.startswith("METHOD:"):
                    rest_method = line.split(":", 1)[1].strip().upper()
                elif line.startswith("PARAMS:"):
                    params_str = line.split(":", 1)[1].strip()
                    try:
                        rest_params = json.loads(params_str)
                    except json.JSONDecodeError:
                        pass

            if not api_name:
                logger.debug(f"[_resolve_from_api] Could not parse API name from response")
                return None

            # Build endpoint description
            if graphql_query:
                api_endpoint = graphql_query[:100] + "..." if len(graphql_query) > 100 else graphql_query
                endpoint_display = f"GraphQL query"
            elif rest_endpoint:
                api_endpoint = f"{rest_method} {rest_endpoint}"
                endpoint_display = api_endpoint
            else:
                logger.debug(f"[_resolve_from_api] No query found in LLM response")
                return None

            # Lazy binding: return API source info without executing
            if lazy:
                cache_key = self._cache_key(fact_name, params)
                api_info = api_descriptions.get(api_name, {})
                api_desc = api_info.get("desc", "API")
                api_type = api_info.get("type", "REST")

                # Build descriptive value like database does: "(api_name) endpoint_info"
                value_str = f"({api_name}) {endpoint_display}"
                reasoning = f"API '{api_name}' ({api_type}): {api_desc}. Endpoint: {api_endpoint}"

                logger.info(f"[_resolve_from_api] Lazy binding for '{fact_name}' -> {api_name}: {endpoint_display}")

                fact = Fact(
                    name=cache_key,
                    value=value_str,
                    confidence=0.95,
                    source=FactSource.API,
                    source_name=api_name,
                    api_endpoint=api_endpoint,
                    reasoning=reasoning,
                    context=f"API: {api_name} - {endpoint_display}",
                )

                self._cache[cache_key] = fact
                self.resolution_log.append(fact)
                return fact

            # Eager execution: actually call the API
            from constat.catalog.api_executor import APIExecutor, APIExecutionError
            executor = APIExecutor(self.config)

            try:
                if graphql_query:
                    logger.info(f"[_resolve_from_api] Executing GraphQL query on {api_name}")
                    result = executor.execute_graphql(api_name, graphql_query)
                elif rest_endpoint:
                    logger.info(f"[_resolve_from_api] Executing REST call on {api_name}: {rest_method} {rest_endpoint}")
                    result = executor.execute_rest(
                        api_name,
                        operation=rest_endpoint,
                        query_params=rest_params if rest_method == "GET" else None,
                        body=rest_params if rest_method in ("POST", "PUT", "PATCH") else None,
                        method=rest_method,
                    )
                else:
                    logger.debug(f"[_resolve_from_api] No query found in LLM response")
                    return None

                # Create the fact from the result
                cache_key = self._cache_key(fact_name, params)

                # Determine value and whether to store as table
                if isinstance(result, dict):
                    # Check if result has a 'data' key with list (common pattern)
                    if "data" in result and isinstance(result["data"], list):
                        value = result["data"]
                    else:
                        value = result
                elif isinstance(result, list):
                    value = result
                else:
                    value = result

                # Check if should store as table
                row_count = None
                table_name = None
                if self._datastore and isinstance(value, list) and len(value) > 10:
                    # Store large results as table
                    import pandas as pd
                    df = pd.DataFrame(value)
                    table_name = f"api_{api_name}_{fact_name}".replace(" ", "_").replace("-", "_")[:50]
                    self._datastore.save_table(table_name, df)
                    row_count = len(df)
                    logger.info(f"[_resolve_from_api] Stored {row_count} rows as table {table_name}")

                fact = Fact(
                    name=cache_key,
                    value=value,
                    confidence=0.95,
                    source=FactSource.API,
                    source_name=api_name,
                    api_endpoint=api_endpoint,
                    table_name=table_name,
                    row_count=row_count,
                    context=f"API Query: {api_endpoint}",
                )

                self._cache[cache_key] = fact
                self.resolution_log.append(fact)
                return fact

            except APIExecutionError as e:
                logger.warning(f"[_resolve_from_api] API execution failed: {e}")
                return None

        except Exception as e:
            logger.warning(f"[_resolve_from_api] Failed to resolve {fact_name} from API: {e}")
            return None

    def _transform_sql_for_sqlite(self: "FactResolver", sql: str) -> str:
        """Transform MySQL/PostgreSQL SQL syntax to SQLite equivalents.

        Handles common incompatibilities:
        - DATE_FORMAT(col, fmt) -> strftime(fmt, col)
        - DATE_SUB(date, INTERVAL n MONTH) -> date(date, '-n months')
        - CURDATE() -> date('now')
        - YEAR(col) -> CAST(strftime('%Y', col) AS INTEGER)
        - MONTH(col) -> CAST(strftime('%m', col) AS INTEGER)
        - EXTRACT(YEAR FROM col) -> CAST(strftime('%Y', col) AS INTEGER)
        - EXTRACT(MONTH FROM col) -> CAST(strftime('%m', col) AS INTEGER)
        """
        import re

        # DATE_FORMAT(col, '%Y-%m') -> strftime('%Y-%m', col)
        def replace_date_format(match):
            col = match.group(1)
            fmt = match.group(2)
            return f"strftime({fmt}, {col})"
        sql = re.sub(r"DATE_FORMAT\s*\(\s*([^,]+),\s*('[^']+')\s*\)", replace_date_format, sql, flags=re.IGNORECASE)

        # DATE_SUB(CURDATE(), INTERVAL n MONTH) -> date('now', '-n months')
        # Also handles DATE_SUB(date_col, INTERVAL n MONTH)
        def replace_date_sub(match):
            date_expr = match.group(1)
            num = match.group(2)
            unit = match.group(3).lower()
            # Convert CURDATE() to 'now'
            if date_expr.strip().upper() == "CURDATE()":
                date_expr = "'now'"
            return f"date({date_expr}, '-{num} {unit}s')"
        sql = re.sub(r"DATE_SUB\s*\(\s*([^,]+),\s*INTERVAL\s+(\d+)\s+(MONTH|DAY|YEAR)\s*\)", replace_date_sub, sql, flags=re.IGNORECASE)

        # CURDATE() -> date('now')
        sql = re.sub(r"\bCURDATE\s*\(\s*\)", "date('now')", sql, flags=re.IGNORECASE)

        # NOW() -> datetime('now')
        sql = re.sub(r"\bNOW\s*\(\s*\)", "datetime('now')", sql, flags=re.IGNORECASE)

        # YEAR(col) -> CAST(strftime('%Y', col) AS INTEGER)
        def replace_year(match):
            col = match.group(1)
            return f"CAST(strftime('%Y', {col}) AS INTEGER)"
        sql = re.sub(r"\bYEAR\s*\(\s*([^)]+)\s*\)", replace_year, sql, flags=re.IGNORECASE)

        # MONTH(col) -> CAST(strftime('%m', col) AS INTEGER)
        def replace_month(match):
            col = match.group(1)
            return f"CAST(strftime('%m', {col}) AS INTEGER)"
        sql = re.sub(r"\bMONTH\s*\(\s*([^)]+)\s*\)", replace_month, sql, flags=re.IGNORECASE)

        # EXTRACT(YEAR FROM col) -> CAST(strftime('%Y', col) AS INTEGER)
        def replace_extract_year(match):
            col = match.group(1)
            return f"CAST(strftime('%Y', {col}) AS INTEGER)"
        sql = re.sub(r"\bEXTRACT\s*\(\s*YEAR\s+FROM\s+([^)]+)\s*\)", replace_extract_year, sql, flags=re.IGNORECASE)

        # EXTRACT(MONTH FROM col) -> CAST(strftime('%m', col) AS INTEGER)
        def replace_extract_month(match):
            col = match.group(1)
            return f"CAST(strftime('%m', {col}) AS INTEGER)"
        sql = re.sub(r"\bEXTRACT\s*\(\s*MONTH\s+FROM\s+([^)]+)\s*\)", replace_extract_month, sql, flags=re.IGNORECASE)

        return sql

    def _resolve_from_database(self: "FactResolver", fact_name: str, params: dict) -> Optional[Fact]:
        """Generate and execute Python code to resolve a fact from data sources.

        Uses code generation (like exploratory mode) to support all data source types:
        - SQL databases: pd.read_sql()
        - NoSQL databases: connector methods
        - File sources: pd.read_csv(), pd.read_json(), etc.

        Uses PythonExecutor for consistent execution and error handling.
        Participates in learning loop when syntax errors are fixed.
        """
        import logging
        import pandas as pd
        from constat.execution.executor import PythonExecutor, format_error_for_retry
        logger = logging.getLogger(__name__)

        logger.debug(f"DB: _resolve_from_database called for: {fact_name}")
        logger.debug(f"DB: llm={self.llm is not None}, schema_manager={self.schema_manager is not None}, config={self.config is not None}")

        if not self.llm or not self.schema_manager:
            logger.debug(f"DB: MISSING: LLM={self.llm is not None}, schema_manager={self.schema_manager is not None}")
            logger.debug(f"[_resolve_from_database] Missing LLM ({self.llm is not None}) "
                        f"or schema_manager ({self.schema_manager is not None})")
            return None

        # Check if fact_name matches a table name - return "referenced" instead of loading data
        # This allows inferences to query the table directly from the original database
        fact_name_lower = fact_name.lower().strip()
        cache_tables = list(self.schema_manager.metadata_cache.keys())
        logger.debug(f"[_resolve_from_database] Checking table match for '{fact_name_lower}', metadata_cache has {len(cache_tables)} tables: {cache_tables[:5]}")
        for full_name, table_meta in self.schema_manager.metadata_cache.items():
            # Match by table name (case-insensitive)
            if table_meta.name.lower() == fact_name_lower:
                logger.info(f"[_resolve_from_database] Table match for '{fact_name}' -> {full_name}")
                # Store column metadata in reasoning for use by inferences
                columns = [c.name for c in table_meta.columns]
                reasoning = f"Table '{table_meta.name}' from database '{table_meta.database}'. Columns: {columns}"
                # Build descriptive value for UI display
                row_info = f"{table_meta.row_count:,} rows" if table_meta.row_count else "table"
                value_str = f"({table_meta.database}.{table_meta.name}) {row_info}"
                return Fact(
                    name=fact_name,
                    value=value_str,
                    source=FactSource.DATABASE,
                    source_name=table_meta.database,
                    reasoning=reasoning,
                    confidence=0.95,
                    table_name=table_meta.name,
                    row_count=table_meta.row_count,
                )

        # Build execution globals with database connections and file paths
        exec_globals = {"pd": pd, "Fact": Fact, "FactSource": FactSource}
        config_db_names = set(self.config.databases.keys()) if self.config else set()

        for db_name in config_db_names:
            db_config = self.config.databases.get(db_name)
            if db_config:
                if db_config.is_file_source():
                    # Provide file path for CSV, JSON, Parquet, etc.
                    exec_globals[f"file_{db_name}"] = db_config.path
                else:
                    # Provide database connection for SQL/NoSQL
                    conn = self.schema_manager.get_connection(db_name)
                    exec_globals[f"db_{db_name}"] = conn

        # Also include dynamically added databases (from projects) not in config
        # SQL connections
        for db_name in self.schema_manager.connections.keys():
            if db_name not in config_db_names:
                exec_globals[f"db_{db_name}"] = self.schema_manager.connections[db_name]
        # NoSQL connections
        for db_name in self.schema_manager.nosql_connections.keys():
            if db_name not in config_db_names:
                exec_globals[f"db_{db_name}"] = self.schema_manager.nosql_connections[db_name]
        # File connections
        for db_name in self.schema_manager.file_connections.keys():
            if db_name not in config_db_names:
                conn = self.schema_manager.file_connections[db_name]
                if hasattr(conn, 'path'):
                    exec_globals[f"file_{db_name}"] = conn.path

        # Build data source hints for the prompt (from config databases)
        source_hints = []
        for db_name, db_config in (self.config.databases.items() if self.config else []):
            if db_config.is_file_source():
                file_type = db_config.type
                source_hints.append(f"- {db_name} ({file_type}): use pd.read_{file_type}(file_{db_name})")
            elif db_config.is_nosql():
                source_hints.append(f"- {db_name} (NoSQL {db_config.type}): use db_{db_name} connector methods")
            else:
                # SQL database - detect dialect
                dialect = "sql"
                if db_config.type == "sql" and db_config.uri:
                    uri_lower = db_config.uri.lower()
                    if uri_lower.startswith("sqlite"):
                        dialect = "sqlite"
                    elif uri_lower.startswith("postgresql") or uri_lower.startswith("postgres"):
                        dialect = "postgresql"
                    elif uri_lower.startswith("mysql"):
                        dialect = "mysql"
                    elif uri_lower.startswith("duckdb"):
                        dialect = "duckdb"
                source_hints.append(f"- {db_name} ({dialect}): use pd.read_sql(query, db_{db_name})")

        # Add hints for dynamically added databases
        for db_name in self.schema_manager.connections.keys():
            if db_name not in config_db_names:
                source_hints.append(f"- {db_name} (sql): use pd.read_sql(query, db_{db_name})")
        for db_name in self.schema_manager.nosql_connections.keys():
            if db_name not in config_db_names:
                source_hints.append(f"- {db_name} (nosql): use db_{db_name} connector methods")
        for db_name in self.schema_manager.file_connections.keys():
            if db_name not in config_db_names:
                source_hints.append(f"- {db_name} (file): use pd.read_csv/json/parquet(file_{db_name})")

        source_hints_text = "\n".join(source_hints) if source_hints else "No data sources configured."
        logger.debug(f"DB: source_hints_text: {source_hints_text[:200]}...")
        logger.debug(f"DB: config_db_names: {config_db_names}")

        # Get schema overview
        schema_overview = self.schema_manager.get_overview()
        logger.debug(f"DB: schema_overview length: {len(schema_overview)}")

        # Build prompt for code generation
        prompt = f"""Generate Python code to resolve this fact from the available data sources.

Fact to resolve: {fact_name}
Parameters: {params}

Available data sources:
{source_hints_text}

Schema:
{schema_overview}

Generate a `get_result()` function that:
1. Queries the appropriate data source(s)
2. Returns the result value (scalar, list, or DataFrame)

IMPORTANT - PREFER SQL OVER PANDAS:
- SQL is more robust, scalable, and has clearer error messages
- Use DuckDB to query DataFrames/JSON: `duckdb.query("SELECT ... FROM df").df()`
- For SQLite: Do NOT use schema prefixes (use 'customers' not 'sales.customers')
- For SQLite: Use strftime() for date formatting, date() for date math
- Return the raw result - the caller will wrap it in a Fact

Example for SQL database:
```python
def get_result():
    df = pd.read_sql("SELECT SUM(amount) as total FROM orders", db_sales)
    return df.iloc[0, 0]  # Return scalar
```

Example for API response (use DuckDB for transformations):
```python
def get_result():
    import duckdb
    response = api_catfacts.get("/breeds", params={{"limit": 10}})
    data = response["data"]  # API responses often have data in nested field
    return duckdb.query("SELECT breed, country FROM data").df()
```

Example for CSV:
```python
def get_result():
    import duckdb
    df = pd.read_csv(file_web_metrics)
    return duckdb.query("SELECT page, SUM(visitors) as total FROM df GROUP BY page").df()
```

CRITICAL - When to respond NOT_POSSIBLE:
- If the fact asks for POLICY, RULES, GUIDELINES, or THRESHOLDS but no such table/config exists in the schema
- If you would need to ANALYZE PATTERNS or DERIVE rules from transactional data - that is NOT the same as having actual rules
- If the schema only has operational/transactional data (reviews, orders, etc.) but the fact asks for policy/rules ABOUT that data
- Statistical summaries of data (avg rating, count, distribution) are NOT policies - policies are prescriptive rules like "rating 5 = 10% raise"
- Do NOT return approximations, pattern analysis, or inferred rules as substitutes for explicitly stored policies

If this fact cannot be DIRECTLY resolved from the available sources, respond with "NOT_POSSIBLE: <reason>".
"""

        # Use PythonExecutor for consistent execution (DRY with exploratory mode)
        executor = PythonExecutor()
        max_retries = 3
        last_code = None
        last_error = None
        original_error_code = None  # Track original failing code for learning

        for attempt in range(1, max_retries + 1):
            # Generate code
            if attempt == 1:
                response = self.llm.generate(
                    system="You are a Python data expert. Generate code to extract facts from data sources.",
                    user_message=prompt,
                    max_tokens=self.llm.max_output_tokens,
                )
            else:
                # Retry with error context
                retry_prompt = f"""Your previous code failed:

{last_error}

Previous code:
```python
{last_code}
```

Please fix the error and regenerate the code.

Original request:
{prompt}"""
                response = self.llm.generate(
                    system="You are a Python data expert. Generate code to extract facts from data sources.",
                    user_message=retry_prompt,
                    max_tokens=self.llm.max_output_tokens,
                )

            logger.debug(f"DB: LLM response (first 300 chars): {response[:300]}...")
            if "NOT_POSSIBLE" in response:
                logger.debug(f"DB: LLM said NOT_POSSIBLE: {response}")
                logger.debug(f"[_resolve_from_database] LLM said not possible: {response}")
                return None

            # Extract code from response
            code = response
            if "```python" in code:
                code = code.split("```python", 1)[1].split("```", 1)[0]
            elif "```" in code:
                code = code.split("```", 1)[1].split("```", 1)[0]

            last_code = code
            logger.debug(f"[_resolve_from_database] Attempt {attempt} generated code:\n{code}")

            # Execute using PythonExecutor
            result = executor.execute(code, exec_globals)

            if result.compile_error:
                # Syntax error - retry with feedback
                if original_error_code is None:
                    original_error_code = code
                last_error = format_error_for_retry(result, code)
                logger.warning(f"[_resolve_from_database] Attempt {attempt} syntax error")
                continue

            if result.runtime_error:
                # Runtime error - don't retry, move to next source type
                logger.warning(f"[_resolve_from_database] Runtime error for {fact_name}: {result.runtime_error.error}")
                return None

            # Execution succeeded - check for get_result function
            get_result = result.namespace.get("get_result")
            if not get_result:
                if original_error_code is None:
                    original_error_code = code
                last_error = "No get_result() function found in generated code. Please define a get_result() function."
                logger.warning(f"[_resolve_from_database] Attempt {attempt}: no get_result() function")
                continue

            # Call get_result and process the result
            try:
                value = get_result()

                # If we fixed a syntax error, record the learning
                if original_error_code is not None and self._learning_callback:
                    self._learning_callback(
                        category="code_error",
                        context={
                            "fact_name": fact_name,
                            "error_message": last_error or "Syntax error",
                            "original_code": original_error_code,
                        },
                        fixed_code=code,
                    )

                cache_key = self._cache_key(fact_name, params)
                source_name = next(iter(config_db_names), None)

                # Handle DataFrame results
                if isinstance(value, pd.DataFrame):
                    if len(value) == 1 and len(value.columns) == 1:
                        value = value.iloc[0, 0]
                    else:
                        value = value.to_dict(orient='records')

                # Check if should store as table
                if isinstance(value, list) and self._datastore and self._should_store_as_table(value):
                    table_name, row_count = self._store_value_as_table(
                        fact_name, value, source_name=source_name
                    )
                    return Fact(
                        name=cache_key,
                        value=f"table:{table_name}",
                        confidence=1.0,
                        source=FactSource.DATABASE,
                        source_name=source_name,
                        query=code,
                        table_name=table_name,
                        row_count=row_count,
                        context=f"Generated code:\n{code}",
                    )

                return Fact(
                    name=cache_key,
                    value=value,
                    confidence=1.0,
                    source=FactSource.DATABASE,
                    source_name=source_name,
                    query=code,
                    context=f"Generated code:\n{code}",
                )

            except Exception as e:
                # Runtime error in get_result() - retry with feedback
                if original_error_code is None:
                    original_error_code = code
                import traceback
                tb = traceback.format_exc()
                last_error = f"get_result() raised an exception:\n{type(e).__name__}: {e}\n\nTraceback:\n{tb}"
                logger.warning(f"[_resolve_from_database] get_result() failed for {fact_name}: {e}")
                continue  # Retry with error feedback

        # All retry attempts exhausted
        logger.error(f"[_resolve_from_database] All {max_retries} attempts failed for {fact_name}")
        return None

    def _resolve_from_document(self: "FactResolver", fact_name: str, params: dict) -> Optional[Fact]:
        """Search reference documents for fact information.

        Uses a two-stage approach:
        1. Semantic search to find potentially relevant document chunks
        2. If chunks have low relevance, load full document sections for better context
        """
        import logging
        logger = logging.getLogger(__name__)

        logger.debug(f"DOC: _resolve_from_document called for: {fact_name}")

        if not self.llm:
            logger.debug(f"DOC: No LLM configured")
            logger.debug(f"[_resolve_from_document] No LLM configured")
            return None

        # Check if we have document tools configured
        doc_tools = getattr(self, '_doc_tools', None)
        logger.debug(f"DOC: doc_tools={doc_tools is not None}")
        if not doc_tools:
            logger.debug(f"DOC: No doc_tools configured - returning None")
            logger.debug(f"[_resolve_from_document] No doc_tools configured")
            return None

        cache_key = self._cache_key(fact_name, params)

        try:
            # Build search query from fact name and params
            param_str = ' '.join(str(v) for v in params.values() if v)
            fact_readable = fact_name.replace('_', ' ')
            search_query = f"{fact_readable} {param_str}".strip()

            logger.debug(f"DOC: Searching for: {search_query}")
            logger.debug(f"[_resolve_from_document] Searching for: {search_query}")

            # Get more results (top 10) and let the LLM evaluate relevance
            # Don't filter by score - semantic search scores can be misleading
            search_results = doc_tools.search_documents(search_query, limit=10)
            logger.debug(f"DOC: Search returned {len(search_results) if search_results else 0} results")

            if not search_results:
                logger.debug(f"DOC: No search results - returning None")
                logger.debug(f"[_resolve_from_document] No search results")
                return None

            best_relevance = max(r.get('relevance', 0) for r in search_results)
            logger.debug(f"DOC: Best relevance: {best_relevance}")
            logger.debug(f"[_resolve_from_document] Best relevance: {best_relevance}")

            # Include ALL results - let the LLM decide what's relevant
            # Semantic search scores are not reliable for filtering
            context = "\n\n".join([
                f"From {r.get('document', 'document')} (section: {r.get('section', 'unknown')}, relevance: {r.get('relevance', 0):.2f}):\n{r.get('excerpt', '')}"
                for r in search_results
            ])

            if not context.strip():
                logger.debug(f"DOC: No context built - returning None")
                logger.debug(f"[_resolve_from_document] No context built")
                return None

            logger.debug(f"DOC: Context built, length: {len(context)}")
            logger.debug(f"DOC: Context preview: {context[:300]}...")

            # Ask LLM to extract the fact from document context
            prompt = f"""Extract information for this fact from the document content:

Fact needed: {fact_name}
Parameters: {params}

Document content:
{context}

If relevant information is found, respond with:
VALUE: <extract the relevant content - table, paragraph, rules, or data as-is>
CONFIDENCE: <0.0-1.0>
SOURCE: <which document/section>
REASONING: <brief explanation>

The VALUE can be:
- A number or string for simple facts
- A table or list for structured policies
- A paragraph or multiple paragraphs for descriptive policies
- JSON if that best represents the data

If no relevant information is found, respond with:
NOT_FOUND
"""

            logger.debug(f"DOC: Calling LLM to extract fact...")
            response = self.llm.generate(
                system="You extract facts and policies from documents. Return content in its natural format.",
                user_message=prompt,
                max_tokens=self.llm.max_output_tokens,
            )

            logger.debug(f"DOC: LLM response: {response[:300]}...")
            logger.debug(f"[_resolve_from_document] LLM response: {response[:200]}...")

            if "NOT_FOUND" in response:
                logger.debug(f"DOC: LLM returned NOT_FOUND - returning None")
                logger.debug(f"[_resolve_from_document] LLM returned NOT_FOUND")
                return None

            # Parse response
            value = None
            confidence = 0.8
            source_name = None
            reasoning = None

            # Extract VALUE - may be JSON spanning multiple lines
            if "VALUE:" in response:
                value_start = response.index("VALUE:") + len("VALUE:")
                # Find where VALUE ends (next field or end)
                value_end = len(response)
                for marker in ["CONFIDENCE:", "SOURCE:", "REASONING:"]:
                    if marker in response[value_start:]:
                        marker_pos = response.index(marker, value_start)
                        if marker_pos < value_end:
                            value_end = marker_pos
                value_str = response[value_start:value_end].strip()

                # Try to parse as JSON first, then number, then string
                import json
                try:
                    value = json.loads(value_str)
                except (json.JSONDecodeError, ValueError):
                    try:
                        value = float(value_str)
                    except ValueError:
                        value = value_str

            # Parse other fields
            for line in response.split("\n"):
                if line.startswith("CONFIDENCE:"):
                    try:
                        confidence = float(line.split(":", 1)[1].strip())
                    except ValueError:
                        pass
                elif line.startswith("SOURCE:"):
                    source_name = line.split(":", 1)[1].strip()
                elif line.startswith("REASONING:"):
                    reasoning = line.split(":", 1)[1].strip()

            logger.debug(f"DOC: Parsed value: {value}, confidence: {confidence}, source: {source_name}")
            if value is not None:
                logger.debug(f"DOC: SUCCESS! Resolved {fact_name} = {value}")
                logger.debug(f"[_resolve_from_document] Resolved {fact_name} = {value} from {source_name}")
                return Fact(
                    name=cache_key,
                    value=value,
                    confidence=confidence,
                    source=FactSource.DOCUMENT,
                    source_name=source_name,
                    reasoning=reasoning,
                )
            else:
                logger.debug(f"DOC: No VALUE found in LLM response")
        except Exception as e:
            logger.debug(f"DOC: Exception resolving {fact_name}: {e}")
            import traceback
            logger.debug(f"DOC: Traceback: {traceback.format_exc()}")
            logger.warning(f"[_resolve_from_document] Error resolving {fact_name}: {e}")

        logger.debug(f"DOC: Returning None for {fact_name}")
        return None

    def _resolve_from_llm(self: "FactResolver", fact_name: str, params: dict) -> Optional[Fact]:
        """Ask LLM for world knowledge or heuristics."""
        import logging
        logger = logging.getLogger(__name__)

        if not self.llm:
            logger.debug(f"[_resolve_from_llm] No LLM configured")
            return None

        logger.debug(f"[_resolve_from_llm] Asking LLM about {fact_name}")

        prompt = f"""I need to know this fact:
Fact: {fact_name}
Parameters: {params}

Do you know this from your training? This could be:
- World knowledge (e.g., "capital of France")
- Industry standards (e.g., "typical VIP threshold is $10,000")
- Common heuristics (e.g., "underperforming means <80% of target")

If you know this, respond with:
VALUE: <the value>
CONFIDENCE: <0.0-1.0, how confident are you>
TYPE: knowledge | heuristic
REASONING: <brief explanation>

If you don't know, respond with:
UNKNOWN
"""

        try:
            response = self.llm.generate(
                system="You are a knowledgeable assistant. Provide facts you're confident about.",
                user_message=prompt,
                max_tokens=self.llm.max_output_tokens,
            )

            if "UNKNOWN" in response:
                return None

            # Parse response
            value = None
            confidence = 0.6  # Default for LLM knowledge
            reasoning = None
            source = FactSource.LLM_KNOWLEDGE

            for line in response.split("\n"):
                if line.startswith("VALUE:"):
                    value_str = line.split(":", 1)[1].strip()
                    # Try to parse as number
                    try:
                        value = float(value_str)
                    except (ValueError, TypeError):
                        value = value_str
                elif line.startswith("CONFIDENCE:"):
                    try:
                        confidence = float(line.split(":", 1)[1].strip())
                    except (ValueError, TypeError):
                        pass
                elif line.startswith("TYPE:"):
                    if "heuristic" in line.lower():
                        source = FactSource.LLM_HEURISTIC
                elif line.startswith("REASONING:"):
                    reasoning = line.split(":", 1)[1].strip()

            if value is not None:
                logger.debug(f"[_resolve_from_llm] Got value for {fact_name}: {value} (conf={confidence})")
                return Fact(
                    name=self._cache_key(fact_name, params),
                    value=value,
                    confidence=confidence,
                    source=source,
                    reasoning=reasoning,
                )
            else:
                logger.debug(f"[_resolve_from_llm] No value parsed for {fact_name}")
        except Exception as e:
            logger.error(f"[_resolve_from_llm] Error for {fact_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())

        return None

    def _resolve_from_sub_plan(self: "FactResolver", fact_name: str, params: dict) -> Optional[Fact]:
        """Generate a mini-plan to derive a complex fact with parallel resolution.

        IMPORTANT: This method enforces the 2+ inputs requirement.
        Derivation must compose multiple DISTINCT facts - not just try synonyms.
        """
        import logging
        logger = logging.getLogger(__name__)

        if not self.strategy.allow_sub_plans:
            logger.debug(f"[_resolve_from_sub_plan] Sub-plans disabled")
            return None

        if self._resolution_depth >= self.strategy.max_sub_plan_depth:
            logger.debug(f"[_resolve_from_sub_plan] Max depth {self.strategy.max_sub_plan_depth} reached")
            return None  # Prevent infinite recursion

        if not self.llm:
            logger.debug(f"[_resolve_from_sub_plan] No LLM configured")
            return None

        # NEW: With tiered resolution enabled, sub-plan is handled by Tier 2 assessment
        # This legacy method should only run if tiered resolution is disabled
        if self.strategy.use_tiered_resolution:
            logger.debug(f"[_resolve_from_sub_plan] Skipping - tiered resolution handles sub-plans via Tier 2")
            return None

        logger.debug(f"[_resolve_from_sub_plan] Attempting sub-plan for {fact_name} at depth {self._resolution_depth}")

        # Emit event: starting sub-plan expansion
        self._emit_event("premise_expanding", {
            "fact_name": fact_name,
            "params": params,
            "depth": self._resolution_depth,
        })

        # Ask LLM to create a plan to derive this fact
        # CRITICAL: Enforce 2+ distinct inputs requirement to prevent synonym hunting
        prompt = f"""I need to derive this fact, but it's not directly available:
Fact: {fact_name}
Parameters: {params}

This fact needs to be COMPUTED from 2 or more OTHER facts using a formula.

CRITICAL REQUIREMENTS:
1. You MUST resolve 2+ DISTINCT facts and COMPOSE them with a formula
2. Valid: "result = fact_A / fact_B" (two inputs, mathematical composition)
3. Valid: "result = filter(fact_A, condition from fact_B)" (two inputs)
4. INVALID: Just resolving the same fact with a different name (synonym hunting)
5. INVALID: Trying to look up "alternative_name" or "similar_concept" - this is NOT derivation

If this fact CANNOT be computed from 2+ other facts, respond with:
```python
def derive(resolver, params):
    # NOT_DERIVABLE: This fact cannot be computed from other facts
    return None
```

Example with VALID derivation (2+ inputs):
```python
def derive(resolver, params):
    # Resolve 2+ DISTINCT facts
    facts = resolver.resolve_many_sync([
        ("employee_salaries", {{}}),  # Input 1: from database
        ("industry_benchmark", {{}}),  # Input 2: general knowledge
    ])
    salaries, benchmark = facts

    # COMPOSE with formula
    avg_salary = sum(s["salary"] for s in salaries.value) / len(salaries.value)
    competitive_ratio = avg_salary / benchmark.value

    return Fact(
        name="{fact_name}",
        value=competitive_ratio,
        confidence=min(f.confidence for f in facts),
        source=FactSource.DERIVED,
        because=facts
    )
```

Generate the derivation function for {fact_name}.
Remember: 2+ DISTINCT inputs with a FORMULA, or return None if not derivable.
"""

        max_retries = 3
        last_code = None
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                # Generate or regenerate code
                if attempt == 1:
                    response = self.llm.generate(
                        system="You are a Python expert. Generate fact derivation functions. Keep code simple and complete.",
                        user_message=prompt,
                        max_tokens=self.llm.max_output_tokens,
                    )
                else:
                    # Retry with error context
                    retry_prompt = f"""Your previous derive() function failed with an error:

{last_error}

Previous code:
```python
{last_code}
```

Please fix the error and regenerate the derive() function.

Original request:
{prompt}"""
                    response = self.llm.generate(
                        system="You are a Python expert. Generate fact derivation functions. Keep code simple and complete.",
                        user_message=retry_prompt,
                        max_tokens=self.llm.max_output_tokens,
                    )

                # Extract code
                code = response
                if "```python" in code:
                    code = code.split("```python", 1)[1].split("```", 1)[0]
                elif "```" in code:
                    code = code.split("```", 1)[1].split("```", 1)[0]

                last_code = code
                logger.debug(f"[_resolve_from_sub_plan] Attempt {attempt} generated code:\n{code}")

                # Validate syntax before executing
                try:
                    compile(code, "<sub_plan>", "exec")
                except SyntaxError as syn_err:
                    last_error = f"Syntax error: {syn_err}"
                    logger.warning(f"[_resolve_from_sub_plan] Attempt {attempt} syntax error: {syn_err}")
                    continue  # Retry

                # Execute the generated function
                local_ns = {"Fact": Fact, "FactSource": FactSource}
                exec(code, local_ns)

                derive_func = local_ns.get("derive")
                if not derive_func:
                    last_error = "No derive() function found in generated code. Please define a derive(resolver, params) function."
                    logger.warning(f"[_resolve_from_sub_plan] Attempt {attempt}: no derive() function")
                    continue  # Retry

                # Execute the derive function
                self._resolution_depth += 1
                try:
                    result = derive_func(self, params)
                    # Emit event: sub-plan expansion completed
                    if result and result.is_resolved:
                        self._emit_event("premise_expanded", {
                            "fact_name": fact_name,
                            "value": result.value,
                            "confidence": result.confidence,
                            "sub_facts": [f.name for f in result.because] if result.because else [],
                            "derivation_trace": result.derivation_trace,
                            "depth": self._resolution_depth,
                        })
                    return result
                except Exception as derive_err:
                    # Runtime error in derive() - retry with feedback
                    import traceback
                    tb = traceback.format_exc()
                    last_error = f"derive() raised an exception:\n{type(derive_err).__name__}: {derive_err}\n\nTraceback:\n{tb}"
                    logger.warning(f"[_resolve_from_sub_plan] Attempt {attempt} derive() failed: {derive_err}")
                    continue  # Retry
                finally:
                    self._resolution_depth -= 1

            except Exception as e:
                import traceback
                last_error = f"Unexpected error: {e}\n{traceback.format_exc()}"
                logger.error(f"[_resolve_from_sub_plan] Attempt {attempt} unexpected error: {e}")
                continue  # Retry

        # All retry attempts exhausted
        logger.error(f"[_resolve_from_sub_plan] All {max_retries} attempts failed for {fact_name}")
        return None

    def _execute_leaf_sql(
        self: "FactResolver",
        spec: ResolutionSpec,
        params: dict,
        build_proof: bool = True,
    ) -> tuple[Fact, Optional[ProofNode]]:
        """Execute SQL for a leaf database fact."""
        import logging
        import pandas as pd
        logger = logging.getLogger(__name__)

        cache_key = self._cache_key(spec.fact_name, params)

        if not self.schema_manager:
            logger.warning("[_execute_leaf_sql] No schema_manager")
            return Fact(
                name=cache_key,
                value=None,
                confidence=0.0,
                source=FactSource.UNRESOLVED,
            ), None

        # Get database connection
        db_names = list(self.config.databases.keys()) if self.config else []
        db_name = db_names[0] if db_names else None
        if not db_name:
            return Fact(
                name=cache_key,
                value=None,
                confidence=0.0,
                source=FactSource.UNRESOLVED,
            ), None

        # Get database type for SQL transformation
        db_type = "sqlite"
        if self.config:
            db_config = self.config.databases.get(db_name)
            if db_config:
                db_type = db_config.type or "sqlite"

        sql = spec.sql

        # Transform SQL for SQLite if needed
        if db_type.lower() == "sqlite":
            import re
            # Strip schema prefixes
            sql = re.sub(r'\b(\w+)\.(\w+)\b', r'\2', sql)
            # Transform date functions
            sql = self._transform_sql_for_sqlite(sql)

        logger.debug(f"[_execute_leaf_sql] Executing: {sql[:200]}...")

        try:
            conn = self.schema_manager.get_connection(db_name)
            result_df = pd.read_sql(sql, conn)

            # Convert result
            if len(result_df) == 1 and len(result_df.columns) == 1:
                value = result_df.iloc[0, 0]
            else:
                value = result_df.to_dict(orient='records')

            fact = Fact(
                name=cache_key,
                value=value,
                confidence=1.0,
                source=FactSource.DATABASE,
                source_name=db_name,
                query=sql,
            )

            # Handle large results
            if self._datastore and isinstance(value, list) and self._should_store_as_table(value):
                table_name, row_count = self._store_value_as_table(spec.fact_name, value, db_name)
                fact.value = f"table:{table_name}"
                fact.table_name = table_name
                fact.row_count = row_count

            self._cache[cache_key] = fact
            self.resolution_log.append(fact)

            # Build proof
            proof = None
            if build_proof:
                proof = ProofNode(
                    conclusion=f"{spec.fact_name} = {fact.display_value}",
                    source=FactSource.DATABASE,
                    source_name=db_name,
                    evidence=sql,
                    confidence=1.0,
                )

            return fact, proof

        except Exception as e:
            logger.error(f"[_execute_leaf_sql] Error: {e}")
            return Fact(
                name=cache_key,
                value=None,
                confidence=0.0,
                source=FactSource.UNRESOLVED,
                reasoning=f"SQL error: {e}",
            ), None

    def _execute_leaf_doc_query(
        self: "FactResolver",
        spec: ResolutionSpec,
        params: dict,
        build_proof: bool = True,
    ) -> tuple[Fact, Optional[ProofNode]]:
        """Execute document search for a leaf document fact."""
        import logging
        logger = logging.getLogger(__name__)

        cache_key = self._cache_key(spec.fact_name, params)

        if not self._doc_tools:
            return Fact(
                name=cache_key,
                value=None,
                confidence=0.0,
                source=FactSource.UNRESOLVED,
                reasoning="No document tools configured",
            ), None

        try:
            # Use doc_tools to search
            results = self._doc_tools.search(spec.doc_query)

            fact = Fact(
                name=cache_key,
                value=results,
                confidence=0.8,  # Lower confidence for doc search
                source=FactSource.DOCUMENT,
                query=spec.doc_query,
            )

            self._cache[cache_key] = fact
            self.resolution_log.append(fact)

            proof = None
            if build_proof:
                proof = ProofNode(
                    conclusion=f"{spec.fact_name} = {fact.display_value}",
                    source=FactSource.DOCUMENT,
                    evidence=spec.doc_query,
                    confidence=0.8,
                )

            return fact, proof

        except Exception as e:
            logger.error(f"[_execute_leaf_doc_query] Error: {e}")
            return Fact(
                name=cache_key,
                value=None,
                confidence=0.0,
                source=FactSource.UNRESOLVED,
                reasoning=f"Document search error: {e}",
            ), None

    def _execute_sandboxed_logic(
        self: "FactResolver",
        logic: str,
        resolved_facts: dict[str, Any],
    ) -> tuple[Any, float]:
        """
        Execute derivation logic with only resolved facts as input.

        The logic code is sandboxed - it only gets:
        - facts: dict of resolved fact values
        - pd: pandas for DataFrame operations

        Args:
            logic: Python code with a derive(facts) function
            resolved_facts: Dict mapping fact names to their values

        Returns:
            Tuple of (result, confidence)
        """
        import logging
        import pandas as pd
        import numpy as np
        logger = logging.getLogger(__name__)

        if not logic:
            return None, 0.0

        try:
            # Validate syntax
            compile(logic, "<sandboxed_logic>", "exec")

            # Sandboxed namespace - only facts and pandas
            local_ns = {
                "pd": pd,
                "np": np,
            }

            # Execute to define the derive function
            exec(logic, local_ns)

            derive_func: Callable = local_ns.get("derive")
            if not derive_func:
                logger.error("[_execute_sandboxed_logic] No 'derive' function found")
                return None, 0.0

            # Call with resolved facts only
            result = derive_func(resolved_facts)

            # Calculate confidence as min of all input facts
            # (we don't have confidence here, assume 1.0 for now)
            confidence = 1.0

            return result, confidence

        except SyntaxError as e:
            logger.error(f"[_execute_sandboxed_logic] Syntax error: {e}")
            return None, 0.0
        except Exception as e:
            logger.error(f"[_execute_sandboxed_logic] Execution error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None, 0.0

    def _should_store_as_table(self: "FactResolver", value: Any) -> bool:
        """
        Check if a value should be stored as a table instead of inline.

        Uses threshold-based logic to avoid context bloat for large arrays.

        Args:
            value: The resolved fact value

        Returns:
            True if value should be stored as a table
        """
        if not isinstance(value, list):
            return False

        # Check row count threshold
        if len(value) > ARRAY_ROW_THRESHOLD:
            return True

        # Check JSON size threshold
        try:
            import json
            json_size = len(json.dumps(value))
            if json_size > ARRAY_SIZE_THRESHOLD:
                return True
        except (TypeError, ValueError):
            # Can't serialize - keep as-is
            pass

        return False

    def _store_value_as_table(self: "FactResolver", fact_name: str, value: list, source_name: str = None) -> tuple[str, int]:
        """
        Store a list value as a table in the datastore.

        Args:
            fact_name: Name of the fact (used to generate table name)
            value: List of dicts to store as table
            source_name: Optional source name for table naming

        Returns:
            Tuple of (table_name, row_count)
        """
        if not self._datastore:
            raise ValueError("No datastore configured for table storage")

        import pandas as pd

        # Generate a clean table name from fact name
        # Replace special chars, ensure valid SQL identifier
        table_name = f"fact_{fact_name}".replace("(", "_").replace(")", "").replace(",", "_").replace("=", "_")
        table_name = table_name.replace("-", "_").replace(" ", "_").lower()

        # Convert to DataFrame
        if value and isinstance(value[0], dict):
            df = pd.DataFrame(value)
        else:
            # Handle list of primitives
            df = pd.DataFrame({"value": value})

        # Store in datastore
        self._datastore.save_dataframe(table_name, df)

        return table_name, len(df)
