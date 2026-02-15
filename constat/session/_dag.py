# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.
"""DAG mixin: _execute_dag_node and related helpers."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Callable, Optional

import constat.llm
from constat.core.models import TaskType
from constat.execution.fact_resolver import format_source_attribution
from constat.prompts import load_prompt
from constat.session._types import StepEvent
from constat.storage.datastore import DataStore
from constat.storage.registry_datastore import RegistryAwareDataStore

logger = logging.getLogger(__name__)


# noinspection PyUnresolvedReferences
class DagMixin:

    def _resolve_llm_knowledge(self, question: str) -> int | float | str:
        """Resolve a fact from LLM general knowledge.

        Args:
            question: The question or fact description to resolve

        Returns:
            The value (number, string, or ISO date string)

        Raises:
            Exception: If the LLM response cannot be parsed
        """
        import json
        from datetime import datetime

        json_key = question.replace(" ", "_").lower()[:30]

        knowledge_prompt = load_prompt("knowledge_prompt.md").format(
            question=question, json_key=json_key,
        )

        result = self.router.execute(
            task_type=TaskType.SYNTHESIS,
            system="You output ONLY valid JSON with a single value (number, string, or ISO date). No explanations.",
            user_message=knowledge_prompt,
            max_tokens=self.router.max_output_tokens,
        )

        response = result.content.strip()
        logger.debug(f"[LLM_KNOWLEDGE] Response for {question}: {response[:200]}")

        # Parse JSON response
        json_str = response
        if "```" in json_str:
            json_str = re.sub(r'```\w*\n?', '', json_str).strip()

        if not json_str.startswith("{"):
            raise Exception(f"Could not parse LLM response: {response[:100]}")

        data = json.loads(json_str)
        if not data:
            raise Exception(f"Empty JSON response: {response[:100]}")

        raw_value = list(data.values())[0]

        # Return typed value
        if isinstance(raw_value, (int, float)):
            return raw_value
        elif isinstance(raw_value, str):
            # Check if it's an ISO date (validate but return as string)
            if re.match(r'^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2})?', raw_value):
                try:
                    datetime.fromisoformat(raw_value.replace('Z', '+00:00'))
                except ValueError:
                    pass  # Not a valid date, return as regular string
            return raw_value
        else:
            return raw_value  # Other types (bool, etc.)

    def _execute_dag_node(
        self,
        node: "FactNode",
        dag: "ExecutionDAG",
        _problem: str,
        detailed_schema: str,
        premises: list[dict],
        _inferences: list[dict],
        resolved_premises: dict,
        resolved_inferences: dict,
        inference_names: dict,
    ) -> tuple[Any, float, str] | tuple[Any, float, str, list, list]:
        """Execute a single DAG node (premise or inference).

        Called by DAGExecutor for each node. Handles both:
        - Leaf nodes (premises): resolve from database/document/cache
        - Internal nodes (inferences): execute Python code using dependency values

        Args:
            node: The FactNode to execute
            dag: The full DAG (for accessing dependency values)
            problem: The original problem being solved
            detailed_schema: Schema info for SQL generation
            premises: Original premise list for reference
            inferences: Original inference list for reference
            resolved_premises: Dict to store resolved premise facts
            resolved_inferences: Dict to store resolved inference results
            inference_names: Dict mapping inference ID to table name

        Returns:
            Tuple of (value, confidence, source)
        """
        import re
        import pandas as pd
        from constat.execution.fact_resolver import Fact, FactSource

        if node.is_leaf:
            # === PREMISE RESOLUTION ===
            fact_id = node.fact_id
            fact_name = node.name
            fact_desc = node.description
            source = f"{node.source}:{node.source_db}" if node.source_db else node.source or "database"

            # Check for pre-resolved node (embedded values handled by DAG parser)
            if node.value is not None:
                fact = Fact(
                    name=fact_name,
                    value=node.value,
                    confidence=node.confidence,
                    source=FactSource.LLM_KNOWLEDGE,
                    reasoning="Embedded value from plan",
                )
                resolved_premises[fact_id] = fact
                self.fact_resolver.add_user_fact(
                    fact_name=fact_name,
                    value=node.value,
                    reasoning="Embedded value",
                    source=FactSource.LLM_KNOWLEDGE,
                )
                if self.history:
                    self.history.save_inference_premise(
                        self.session_id, fact_id, fact_name,
                        node.value, "embedded", fact_desc or ""
                    )
                return node.value, node.confidence, "user"

            # Check cache
            cached_fact = self.fact_resolver.get_fact(fact_name)
            if cached_fact and cached_fact.value is not None:
                resolved_premises[fact_id] = cached_fact
                source_str = format_source_attribution(
                    cached_fact.source, cached_fact.source_name, cached_fact.api_endpoint
                )
                return cached_fact.value, cached_fact.confidence, source_str

            fact = None
            sql = None

            # Route based on source type (source is required, validated by DAG parser)
            logger.debug(f"[DAG] Premise {fact_id} '{fact_name}' routing with source='{source}'")

            if source == FactSource.CACHE.value:
                # Cached data resolution - look up in fact cache or datastore
                cached_fact = self.fact_resolver.get_fact(fact_name)
                if cached_fact:
                    logger.debug(f"[DAG] Found cached fact: {fact_name} = {str(cached_fact.value)[:50]}...")
                    resolved_premises[fact_id] = cached_fact
                    source_str = format_source_attribution(
                        cached_fact.source, cached_fact.source_name, cached_fact.api_endpoint
                    )
                    return cached_fact.value, cached_fact.confidence, source_str

                # Try datastore table
                if self.datastore:
                    tables = self.datastore.list_tables()
                    for table in tables:
                        if table["name"].lower() == fact_name.lower():
                            # Found table in datastore
                            row_count = table.get("row_count", 0)
                            value_str = f"({table['name']}) {row_count} rows"
                            fact = Fact(
                                name=fact_name,
                                value=value_str,
                                source=FactSource.CACHE,
                                reasoning=f"Cached table from previous analysis",
                                confidence=0.95,
                                table_name=table["name"],
                                row_count=row_count,
                            )
                            resolved_premises[fact_id] = fact
                            self.fact_resolver.add_user_fact(
                                fact_name=fact_name,
                                value=value_str,
                                reasoning="Retrieved from session cache",
                                source=FactSource.CACHE,
                                table_name=table["name"],
                                row_count=row_count,
                            )
                            return value_str, 0.95, "cache"

                raise ValueError(f"Cached data not found: {fact_name}")

            elif source.startswith(FactSource.DATABASE.value):
                # Database resolution
                db_name = source.split(":", 1)[1].strip() if ":" in source else None
                if not db_name:
                    available_dbs = list(self.schema_manager.connections.keys())
                    if available_dbs:
                        db_name = available_dbs[0]
                    else:
                        raise Exception("No SQL databases configured")

                # Check if fact_name matches a table name - return table reference instead of loading
                fact_name_lower = fact_name.lower().strip()
                cache_keys = list(self.schema_manager.metadata_cache.keys())
                logger.debug(f"[DAG] Checking table match for '{fact_name_lower}' in {len(cache_keys)} tables: {cache_keys[:5]}")
                for full_name, table_meta in self.schema_manager.metadata_cache.items():
                    if table_meta.name.lower() == fact_name_lower:
                        # Table match - return reference without loading data
                        columns = [c.name for c in table_meta.columns]
                        row_info = f"{table_meta.row_count:,} rows" if table_meta.row_count else "table"
                        # Use parentheses instead of brackets (Rich interprets [] as markup)
                        value_str = f"({table_meta.database}.{table_meta.name}) {row_info}"
                        fact = Fact(
                            name=fact_name,
                            value=value_str,
                            source=FactSource.DATABASE,
                            source_name=table_meta.database,
                            reasoning=f"Table '{table_meta.name}' from database '{table_meta.database}'. Columns: {columns}",
                            confidence=0.95,
                            table_name=table_meta.name,
                            row_count=table_meta.row_count,
                        )
                        resolved_premises[fact_id] = fact
                        self.fact_resolver.add_user_fact(
                            fact_name=fact_name,
                            value=value_str,
                            reasoning=f"Table reference: {table_meta.database}.{table_meta.name}",
                            source=FactSource.DATABASE,
                        )
                        return value_str, 0.95, f"database:{db_name}"

                engine = self.schema_manager.get_sql_connection(db_name)
                max_retries = 7
                last_error = None

                for attempt in range(max_retries):
                    # Emit SQL generation event
                    self._emit_event(StepEvent(
                        event_type="sql_generating",
                        step_number=0,
                        data={
                            "fact_name": fact_name,
                            "database": db_name,
                            "attempt": attempt + 1,
                            "max_attempts": max_retries,
                            "is_retry": attempt > 0,
                            "retry_reason": last_error[:50] if last_error else None,
                        }
                    ))

                    error_context = f"\nPREVIOUS ERROR: {last_error}\nFix the query." if last_error else ""
                    sql_learnings = self._get_codegen_learnings(fact_desc, TaskType.SQL_GENERATION)

                    sql_prompt = f"""Generate a SQL query to retrieve: {fact_desc}

Schema:
{detailed_schema}
{sql_learnings}
{error_context}
RULES:
- Always SELECT primary key columns for joins
- Always quote identifiers with double quotes (e.g., "group", "order") to avoid reserved word conflicts"""

                    sql_result = self.router.execute(
                        task_type=TaskType.SQL_GENERATION,
                        system="Output raw SQL only. No markdown.",
                        user_message=sql_prompt,
                        max_tokens=self.router.max_output_tokens,
                    )

                    sql = sql_result.content.strip()
                    code_block = re.search(r'```(?:sql)?\s*\n?(.*?)\n?```', sql, re.DOTALL | re.IGNORECASE)
                    if code_block:
                        sql = code_block.group(1).strip()

                    if "sqlite" in str(engine.url):
                        sql = re.sub(rf'\b{db_name}\.(\w+)', r'\1', sql)

                    # Log generated SQL for debugging
                    logger.debug(f"[SQL] Generated SQL for '{fact_name}' (attempt {attempt + 1}/{max_retries}):\n{sql}")

                    # Emit SQL executing event
                    self._emit_event(StepEvent(
                        event_type="sql_executing",
                        step_number=0,
                        data={
                            "fact_name": fact_name,
                            "database": db_name,
                            "attempt": attempt + 1,
                            "max_attempts": max_retries,
                        }
                    ))

                    try:
                        result_df = pd.read_sql(sql, engine)
                        row_count = len(result_df)
                        node.sql_query = sql

                        if row_count == 1 and len(result_df.columns) == 1:
                            scalar_value = result_df.iloc[0, 0]
                            if hasattr(scalar_value, 'item'):
                                scalar_value = scalar_value.item()
                            fact_value = scalar_value
                        else:
                            fact_value = f"{row_count} rows"

                        table_name = fact_name.lower().replace(' ', '_').replace('-', '_')
                        if row_count > 0 and self.datastore:
                            self.datastore.save_dataframe(table_name, result_df)
                            node.row_count = row_count

                        fact = Fact(
                            name=fact_name,
                            value=fact_value,
                            confidence=0.9,
                            source=FactSource.DATABASE,
                            query=sql,
                            table_name=table_name if row_count > 1 else None,
                            row_count=row_count if row_count > 1 else None,
                        )
                        break
                    except Exception as sql_err:
                        last_error = str(sql_err)
                        will_retry = attempt < max_retries - 1
                        # Log full error for debugging
                        logger.warning(f"[SQL] Error for '{fact_name}' (attempt {attempt + 1}/{max_retries}): {sql_err}")
                        logger.debug(f"[SQL] Failed query:\n{sql}")
                        if will_retry:
                            logger.debug(f"[SQL] Will retry ({max_retries - attempt - 1} attempts remaining)")
                        # Emit SQL error event
                        self._emit_event(StepEvent(
                            event_type="sql_error",
                            step_number=0,
                            data={
                                "fact_name": fact_name,
                                "database": db_name,
                                "error": str(sql_err),  # Full error, not truncated
                                "sql": sql[:500],  # Include SQL (truncated to 500 for events)
                                "attempt": attempt + 1,
                                "max_attempts": max_retries,
                                "will_retry": will_retry,
                            }
                        ))
                        if attempt == max_retries - 1:
                            raise Exception(f"SQL error after {max_retries} attempts: {sql_err}")

            elif source.startswith(FactSource.LLM_KNOWLEDGE.value):
                value = self._resolve_llm_knowledge(fact_desc)
                fact = Fact(
                    name=fact_name,
                    value=value,
                    confidence=0.7,
                    source=FactSource.LLM_KNOWLEDGE,
                    reasoning=f"LLM estimate: {fact_desc}",
                )

            elif source.startswith(FactSource.API.value):
                # API resolution via fact resolver
                api_name = source.split(":", 1)[1].strip() if ":" in source else None
                logger.debug(f"[DAG] Resolving API premise {fact_id} '{fact_name}' from API: {api_name}")
                fact, _ = self.fact_resolver.resolve_tiered(fact_name, fact_description=fact_desc)
                if fact and fact.source != FactSource.API:
                    # Fact resolver may have used a different source - update if API was requested
                    logger.debug(f"[DAG] API resolution fell back to {fact.source.value}")

            else:
                # Generic resolution (document, user, or other sources)
                logger.debug(f"[DAG] Using tiered resolution for {fact_id} '{fact_name}' (source={source})")
                fact, _ = self.fact_resolver.resolve_tiered(fact_name, fact_description=fact_desc)

            if fact and fact.value is not None:
                resolved_premises[fact_id] = fact
                if self.history:
                    self.history.save_inference_premise(
                        self.session_id, fact_id, fact_name,
                        fact.value,
                        fact.source.value if hasattr(fact.source, 'value') else str(fact.source),
                        fact_desc or ""
                    )
                self.fact_resolver.add_user_fact(
                    fact_name=fact_name,
                    value=fact.value,
                    query=sql,
                    reasoning=fact.reasoning,
                    source=fact.source,
                    table_name=getattr(fact, 'table_name', None),
                    row_count=getattr(fact, 'row_count', None),
                    context=f"SQL: {sql}" if sql else None,
                )
                source_str = format_source_attribution(
                    fact.source, fact.source_name, fact.api_endpoint
                )
                return fact.value, fact.confidence, source_str
            else:
                raise ValueError(f"Failed to resolve premise: {fact_name}")

        else:
            # === INFERENCE EXECUTION ===
            inf_id = node.fact_id
            operation = node.operation
            explanation = node.description
            inf_name = node.name
            table_name = inf_name.lower().replace(' ', '_').replace('-', '_')
            inference_names[inf_id] = table_name

            # Handle verify_exists operation
            if operation and operation.startswith("verify_exists("):
                ref_match = re.match(r'verify_exists\((\w+)\)', operation)
                if ref_match:
                    ref_id = ref_match.group(1)
                    ref_table = inference_names.get(ref_id, ref_id.lower())
                    if ref_id.startswith("P"):
                        p_idx = int(ref_id[1:]) - 1
                        if p_idx < len(premises):
                            ref_table = premises[p_idx]['name'].lower().replace(' ', '_')

                    if self.datastore:
                        try:
                            profile = self._profile_table(ref_table)
                            row_count = profile["row_count"]

                            # Build profile summary
                            profile_lines = [f"**{ref_table}**: {row_count} rows, {profile['column_count']} columns"]
                            issues = []
                            for col in profile["columns"]:
                                col_line = f"  - `{col['name']}` ({col['type']}): {col['distinct']} distinct"
                                if col["null_pct"] > 0:
                                    col_line += f", {col['null_pct']:.0f}% null"
                                if col["all_null"]:
                                    col_line += " **ALL NULL**"
                                    issues.append(col["name"])
                                profile_lines.append(col_line)

                            if profile["duplicate_rows"] > 0:
                                profile_lines.append(f"\n  Duplicate rows: {profile['duplicate_rows']}")
                                issues.append(f"{profile['duplicate_rows']} duplicate rows")

                            profile_text = "\n".join(profile_lines)
                            logger.info(f"[VERIFY] {inf_id} profile for '{ref_table}':\n{profile_text}")

                            if issues:
                                logger.warning(f"[VERIFY] {inf_id} data quality issues in '{ref_table}': {issues}")

                            # Save profile as artifact
                            if self.datastore:
                                inf_step = 1000 + int(inf_id[1:])
                                self.datastore.add_artifact(
                                    inf_step, 0, "markdown", profile_text,
                                    name=f"profile_{ref_table}",
                                    title=f"Data Profile: {ref_table}",
                                    content_type="text/markdown",
                                )

                            resolved_inferences[inf_id] = f"Verified: {row_count} records"
                            self.fact_resolver.add_user_fact(
                                fact_name=inf_name,
                                value=f"{row_count} records verified" + (f" (issues: {', '.join(issues)})" if issues else ""),
                                reasoning=profile_text,
                                source=FactSource.DERIVED,
                            )
                            confidence = 0.7 if issues else 0.95
                            return f"Verified: {row_count} records", confidence, "derived"
                        except Exception as ve:
                            raise ValueError(f"Verification failed: {ve}")

            # Build context from dependencies including column names
            scalars = []
            tables = []
            referenced_tables = []  # Tables that need to be queried from original database
            api_sources = []  # Data that needs to be fetched from APIs
            for dep_name in node.dependencies:
                dep_node = dag.get_node(dep_name)
                if not dep_node:
                    raise ValueError(f"Dependency '{dep_name}' not found in DAG")
                if dep_node.value is None:
                    raise ValueError(
                        f"Dependency '{dep_name}' ({dep_node.fact_id}) has no value. "
                        f"Status: {dep_node.status}, Error: {dep_node.error}"
                    )
                val_str = str(dep_node.value)
                val_lower = val_str.lower()
                # Check if this is a loaded table (value contains row count)
                is_loaded_table = "rows" in val_lower or (dep_node.row_count and dep_node.row_count > 1)
                # Check if this is a referenced database table (format: (db.table) X rows)
                is_db_referenced = val_str.startswith("(") and ")" in val_str and "." in val_str.split(")")[0]
                # Check if this is an API-sourced reference (format: (api_name) endpoint_display)
                is_api_referenced = (
                    val_str.startswith("(") and ")" in val_str
                    and "." not in val_str.split(")")[0]
                ) or (dep_node.source and dep_node.source.startswith("api"))

                if is_loaded_table or is_db_referenced or is_api_referenced:
                    dep_table = dep_name.lower().replace(' ', '_').replace('-', '_')
                    columns_info = ""

                    if is_api_referenced and not is_loaded_table:
                        # API-sourced reference: extract api name
                        api_match = re.match(r'\((\w+)\)\s*(.*)', val_str)
                        api_name = None
                        if api_match:
                            api_name = api_match.group(1)
                        elif dep_node.source and ":" in dep_node.source:
                            api_name = dep_node.source.split(":", 1)[1].strip()
                        api_sources.append(
                            f"- {dep_node.fact_id}: fetch from API 'api_{api_name or 'unknown'}' into '{dep_table}'"
                        )
                    elif is_db_referenced:
                        # Referenced table: get metadata from original database
                        # Extract table name from the value format (db.table)
                        match = re.match(r'\(([^.]+)\.([^)]+)\)', val_str)
                        if match:
                            ref_db, ref_table = match.groups()
                            dep_table = ref_table  # Use actual table name

                        # Try to get column info from the original database table
                        if self.schema_manager:
                            try:
                                # Find matching table in schema
                                for db_name in self.schema_manager.connections.keys():
                                    table_meta = self.schema_manager.get_table_metadata(db_name, dep_table)
                                    if table_meta:
                                        cols = [c.name for c in table_meta.columns]
                                        columns_info = f" columns: {cols}"
                                        referenced_tables.append(
                                            f"- {dep_node.fact_id}: query from database table '{dep_table}'{columns_info}"
                                        )
                                        break
                            except Exception as e:
                                logger.debug(f"Failed to get table metadata for {dep_table}: {e}")

                        if not columns_info:
                            # Fallback: still mark as referenced
                            referenced_tables.append(
                                f"- {dep_node.fact_id}: referenced table '{dep_table}' (query from database)"
                            )
                    else:
                        # Regular table in datastore
                        sample_info = ""
                        if self.datastore:
                            try:
                                schema_df = self.datastore.query(f"DESCRIBE {dep_table}")
                                if len(schema_df) > 0:
                                    cols = list(schema_df['column_name']) if 'column_name' in schema_df.columns else list(schema_df.iloc[:, 0])
                                    columns_info = f" columns: {cols}"
                            except Exception as e:
                                logger.debug(f"DESCRIBE failed for {dep_table}, trying SELECT: {e}")
                                try:
                                    sample = self.datastore.query(f"SELECT * FROM {dep_table} LIMIT 1")
                                    columns_info = f" columns: {list(sample.columns)}"
                                except Exception as e2:
                                    logger.debug(f"Failed to get columns for {dep_table}: {e2}")
                            # Fetch sample row to show actual data shape
                            try:
                                sample_df = self.datastore.query(f"SELECT * FROM {dep_table} LIMIT 1")
                                if len(sample_df) > 0:
                                    row_dict = {}
                                    for col in sample_df.columns:
                                        val = sample_df.iloc[0][col]
                                        val_str = str(val)
                                        if len(val_str) > 120:
                                            val_str = val_str[:120] + "..."
                                        row_dict[col] = val_str
                                    sample_info = f"\n    sample row: {row_dict}"
                                    row_count = self.datastore.query(f"SELECT COUNT(*) AS cnt FROM {dep_table}").iloc[0, 0]
                                    sample_info += f"\n    total rows: {row_count}"
                            except Exception as e:
                                logger.debug(f"Failed to get sample for {dep_table}: {e}")
                        tables.append(f"- {dep_node.fact_id}: stored as '{dep_table}'{columns_info}{sample_info}")
                else:
                    if dep_node.source == "document":
                        # Document premises: tell LLM to use doc_read() instead of variable reference
                        # Use source_db (from "document:<name>") if available, else fall back to premise name
                        doc_name = dep_node.source_db or dep_name.lower().replace(' ', '_').replace('-', '_')
                        # Validate the doc name against configured documents
                        if self.doc_tools:
                            configured_docs = list(self.doc_tools._loaded_documents.keys()) + list(getattr(self.doc_tools, '_document_configs', {}).keys())
                            if doc_name not in configured_docs and configured_docs:
                                # Try to find a matching configured document
                                for cfg_name in configured_docs:
                                    if cfg_name in dep_name.lower() or dep_name.lower() in cfg_name:
                                        doc_name = cfg_name
                                        break
                        scalars.append(
                            f"- {dep_node.fact_id} ({dep_name}): [DOCUMENT] "
                            f"Load at runtime with `doc_read('{doc_name}')` — do NOT reference {dep_node.fact_id} as a variable"
                        )
                    else:
                        scalars.append(f"- {dep_node.fact_id} ({dep_name}): {dep_node.value}")

            # Build referenced tables section for prompt
            referenced_section = ""
            if referenced_tables:
                referenced_section = f"""
REFERENCED TABLES (query with pd.read_sql(sql, db_<name>)):
{chr(10).join(referenced_tables)}
"""

            # Build API sources section for prompt (with schema info)
            api_sources_section = ""
            api_schema_cache: dict[str, dict] = {}
            if api_sources:
                # Fetch schema for referenced APIs to include in prompt
                schema_lines = list(api_sources)
                all_apis = self.get_all_apis()
                if all_apis:
                    from constat.catalog.api_executor import APIExecutor
                    api_executor = APIExecutor(self.config, project_apis=self._domain_apis)
                    for api_name, api_config in all_apis.items():
                        # Only fetch schema for APIs referenced in this inference
                        if not any(f"api_{api_name}" in s for s in api_sources):
                            continue
                        try:
                            if api_config.type == "graphql":
                                overview = api_executor.get_schema_overview(api_name)
                                api_schema_cache[api_name] = overview
                                schema_lines.append(f"\nSchema for api_{api_name} (GraphQL):")
                                for q in overview.get("queries", []):
                                    args_str = ", ".join(q.get("args", []))
                                    schema_lines.append(f"  query {q['name']}({args_str}) -> {q.get('returns', '?')}")
                                    # Get detailed return fields for this query
                                    try:
                                        detail = api_executor.get_query_schema(api_name, q["name"])
                                        if detail.get("return_fields"):
                                            GRAPHQL_SCALARS = {"String", "Int", "Float", "Boolean", "ID"}
                                            field_strs = []
                                            for f in detail["return_fields"]:
                                                ftype = f['type']
                                                # Extract base type name (strip [], !)
                                                base = ftype.replace("[", "").replace("]", "").replace("!", "").strip()
                                                if base in GRAPHQL_SCALARS:
                                                    field_strs.append(f"{f['name']}: {ftype} (scalar, NO subfields)")
                                                else:
                                                    field_strs.append(f"{f['name']}: {ftype}")
                                            schema_lines.append(f"    fields: {', '.join(field_strs)}")
                                    except (KeyError, ValueError, TypeError):
                                        pass
                            else:
                                # REST API: include endpoint info if available
                                if self.schema_manager and hasattr(self.schema_manager, 'api_schema_manager'):
                                    endpoints = self.schema_manager.api_schema_manager.get_api_schema(api_name)
                                    if endpoints:
                                        schema_lines.append(f"\nEndpoints for api_{api_name} (REST):")
                                        for ep in endpoints[:10]:
                                            schema_lines.append(f"  {ep.method} {ep.path} - {ep.description or ep.endpoint_name}")
                                schema_lines.append(f"\nIMPORTANT: api_{api_name} is REST. Response is often paginated:")
                                schema_lines.append(f"  response = api_{api_name}('GET /endpoint', {{params}})")
                                schema_lines.append(f"  # Extract array from wrapper: check 'data', 'results', 'items' keys")
                        except Exception as e:
                            logger.debug(f"Failed to get API schema for {api_name}: {e}")

                api_sources_section = f"""
API SOURCES (fetch with api_<name>() functions):
{chr(10).join(schema_lines)}

IMPORTANT: api_<name>(query) returns the 'data' dict from the GraphQL response.
Example: result = api_countries('{{ country(code: "GB") {{ name languages {{ name }} currency }} }}')
         result == {{"country": {{"name": "United Kingdom", "languages": [{{"name": "English"}}], "currency": "GBP"}}}}
"""

            # Build dynamic data source descriptions
            data_source_apis = [
                "- store.query(sql) -> pd.DataFrame (for datastore tables)",
                "- store.save_dataframe(name, df)",
            ]

            # SQL databases
            from constat.catalog.sql_transpiler import TranspilingConnection
            if self.schema_manager:
                for db_name in self.schema_manager.connections.keys():
                    conn = self.schema_manager.get_connection(db_name)
                    if isinstance(conn, TranspilingConnection):
                        data_source_apis.append(f"- pd.read_sql(query, db_{db_name}) -> DataFrame")
                        data_source_apis.append(f"- sql_{db_name}(query) -> DataFrame (auto-transpiles SQL)")
                    else:
                        data_source_apis.append(f"- pd.read_sql(query, db_{db_name}) -> DataFrame")
                # NoSQL
                for db_name in self.schema_manager.nosql_connections.keys():
                    data_source_apis.append(f"- db_{db_name}: NoSQL connector")
                # File sources
                for db_name in self.schema_manager.file_connections.keys():
                    conn = self.schema_manager.file_connections[db_name]
                    if hasattr(conn, 'path'):
                        ext = str(conn.path).rsplit('.', 1)[-1].lower() if '.' in str(conn.path) else 'csv'
                        reader = {'csv': 'read_csv', 'json': 'read_json', 'parquet': 'read_parquet'}.get(ext, 'read_csv')
                        data_source_apis.append(f"- pd.{reader}(file_{db_name}) -> DataFrame")

            # APIs
            all_apis_for_desc = self.get_all_apis()
            if all_apis_for_desc:
                for api_name, api_config in all_apis_for_desc.items():
                    if api_config.type == "graphql":
                        data_source_apis.append(f"- api_{api_name}(query_string) -> dict (GraphQL 'data' portion)")
                    else:
                        data_source_apis.append(f"- api_{api_name}('GET /endpoint', {{params}}) -> data (REST)")

            # Build step code hints section (from exploratory session)
            step_hints_section = ""
            step_hints = getattr(self, '_proof_step_hints', [])
            if step_hints:
                relevant = []
                op_lower = (operation or "").lower()
                name_lower = (inf_name or "").lower()
                for step in step_hints:
                    goal_lower = (step.get("goal", "") or "").lower()
                    # Include step if goal overlaps with inference operation or name
                    if (any(word in goal_lower for word in name_lower.split() if len(word) > 3)
                            or any(word in goal_lower for word in op_lower.split() if len(word) > 3)):
                        relevant.append(step)
                if not relevant and step_hints:
                    # No keyword match — include all steps as general reference
                    relevant = step_hints
                if relevant:
                    hints = []
                    for step in relevant[:3]:  # Limit to 3 most relevant
                        goal = step.get("goal", f"Step {step.get('step_number', '?')}")
                        code_text = step.get("code", "")
                        if len(code_text) > 800:
                            code_text = code_text[:800] + "\n# ... (truncated)"
                        hints.append(f"# Step: {goal}\n{code_text}")
                    step_hints_section = (
                        "\n\nREFERENCE CODE from exploratory session (use as hints, adapt as needed):\n"
                        + "\n---\n".join(hints) + "\n"
                    )

            # Build codegen learnings for inference
            inference_learnings = ""
            try:
                inference_learnings = self._get_codegen_learnings(
                    f"inference {inf_id}: {operation}", TaskType.SQL_GENERATION
                )
            except Exception as e:
                logger.debug(f"Failed to get codegen learnings for inference: {e}")

            # Generate inference code
            inference_prompt = load_prompt("inference_prompt.md").format(
                inf_id=inf_id,
                inf_name=inf_name,
                operation=operation,
                explanation=explanation,
                scalars="\n".join(scalars) if scalars else "(none)",
                tables="\n".join(tables) if tables else "(none)",
                referenced_section=referenced_section,
                api_sources_section=api_sources_section,
                data_source_apis="\n".join(data_source_apis),
                table_name=table_name,
            ) + step_hints_section
            if inference_learnings:
                inference_prompt += f"\n\nLEARNINGS FROM PREVIOUS ERRORS:\n{inference_learnings}"

            import io

            max_retries = 7
            last_error = None
            code = None
            first_error = None  # Track for learning capture
            first_code = None
            _val_passed = []  # True assertions (structural + user-specified)
            _val_profile = []  # Data profile stats for human review
            attempt = 0
            exec_globals: dict = {}
            captured = io.StringIO()

            for attempt in range(max_retries):
                prompt = inference_prompt
                if last_error:
                    prompt = f"PREVIOUS ERROR: {last_error}\n\n{inference_prompt}"

                code_result = self.router.execute(
                    task_type=TaskType.SQL_GENERATION,
                    system="Generate Python code. Return only executable code.",
                    user_message=prompt,
                    max_tokens=self.router.max_output_tokens,
                )

                code = code_result.content.strip()
                if code.startswith("```"):
                    code = re.sub(r'^```\w*\n?', '', code)
                    code = re.sub(r'\n?```$', '', code)

                logger.info(f"[INFERENCE_CODE] {inf_id} attempt {attempt+1}: code length={len(code)} chars")
                logger.debug(f"[INFERENCE_CODE] {inf_id} attempt {attempt+1} code:\n{code}")

                # Auto-fix DataFrame boolean errors
                code = re.sub(r'\bif\s+(df|result|data)\s*:', r'if not \1.empty:', code)
                code = re.sub(r'\bif\s+not\s+(df|result|data)\s*:', r'if \1.empty:', code)

                node.code = code

                import sys
                import numpy as np

                # Build full execution globals (databases, APIs, file sources)
                exec_globals = self._get_execution_globals()
                exec_globals["store"] = self.datastore
                exec_globals["pd"] = pd
                exec_globals["np"] = np
                exec_globals["llm_map"] = constat.llm.llm_map
                exec_globals["llm_classify"] = constat.llm.llm_classify
                exec_globals["llm_extract"] = constat.llm.llm_extract
                exec_globals["llm_summarize"] = constat.llm.llm_summarize
                exec_globals["llm_score"] = constat.llm.llm_score
                exec_globals["doc_read"] = self._create_doc_read_helper()
                self._inference_used_llm_map = False  # Reset per inference

                # Add resolved values to context
                for pid, fact in resolved_premises.items():
                    if fact and fact.value is not None:
                        val = fact.value
                        if isinstance(val, str) and "rows" in val:
                            try:
                                val = int(val.split()[0])
                            except (ValueError, IndexError):
                                pass
                        exec_globals[pid] = val

                for iid, ival in resolved_inferences.items():
                    if ival is not None:
                        val = ival
                        if isinstance(val, str) and "rows" in val:
                            try:
                                val = int(val.split()[0])
                            except (ValueError, IndexError):
                                pass
                        exec_globals[iid] = val

                captured = io.StringIO()
                old_stdout = sys.stdout
                try:
                    sys.stdout = captured
                    exec(code, exec_globals)

                    # --- Post-execution validation ---
                    _val_computed = exec_globals.get('_result')
                    _val_tables = [t['name'] for t in self.datastore.list_tables()] if self.datastore else []
                    _val_output = captured.getvalue().strip()
                    _val_error = None
                    _val_passed = []  # True assertions (structural + user-specified)
                    _val_profile = []  # Data profile stats for human review

                    # 1. Result exists (unless table saved or stdout produced)
                    if _val_computed is None and table_name not in _val_tables and not _val_output:
                        _val_error = f"No result produced. Set _result, save table '{table_name}', or print output."
                    else:
                        _val_passed.append("Result produced")

                    # 2. DataFrame not empty
                    if not _val_error and _val_computed is not None and hasattr(_val_computed, 'empty') and _val_computed.empty:
                        _val_error = f"Result DataFrame is empty (0 rows). Expected data in '{table_name}'."
                    elif not _val_error and _val_computed is not None and hasattr(_val_computed, 'empty'):
                        _val_passed.append(f"DataFrame has {len(_val_computed)} rows")

                    # 3. Saved table has rows
                    if not _val_error and table_name in _val_tables:
                        _row_ct = int(self.datastore.query(f'SELECT COUNT(*) FROM "{table_name}"').iloc[0, 0])
                        if _row_ct == 0:
                            _val_error = f"Table '{table_name}' saved but has 0 rows."
                        else:
                            _val_passed.append(f"Table '{table_name}' has {_row_ct} rows")

                    # 4. No all-null columns
                    if not _val_error and table_name in _val_tables:
                        try:
                            _null_df = self.datastore.query(
                                f'SELECT column_name FROM (SELECT * FROM "{table_name}" LIMIT 1000) '
                                f'UNPIVOT (value FOR column_name IN (*)) '
                                f'GROUP BY column_name HAVING COUNT(CASE WHEN value IS NOT NULL AND value != \'\' THEN 1 END) = 0'
                            )
                            if len(_null_df) > 0:
                                _val_error = f"Table '{table_name}' has all-NULL columns: {list(_null_df['column_name'])}. Data enrichment likely failed."
                            else:
                                _val_passed.append("No all-NULL columns")
                        except Exception:
                            pass

                    # 5. Column-level profiling → _val_profile (stats for human review)
                    if not _val_error and table_name in _val_tables:
                        try:
                            _cols_df = self.datastore.query(f'SELECT * FROM "{table_name}" LIMIT 0')
                            for _c in _cols_df.columns:
                                try:
                                    _stats = self.datastore.query(
                                        f'SELECT MIN("{_c}") as mn, MAX("{_c}") as mx, '
                                        f'COUNT("{_c}") as cnt, COUNT(*) as total '
                                        f'FROM "{table_name}"'
                                    )
                                    # noinspection PyTypeChecker
                                    _mn, _mx = _stats.iloc[0]['mn'], _stats.iloc[0]['mx']
                                    # noinspection PyTypeChecker
                                    _cnt = int(_stats.iloc[0]['cnt'])
                                    # noinspection PyTypeChecker
                                    _tot = int(_stats.iloc[0]['total'])
                                    if isinstance(_mn, (int, float)) and isinstance(_mx, (int, float)):
                                        if _mn == _mx:
                                            _val_profile.append(f"{_c}: all values = {_mn}")
                                        else:
                                            _val_profile.append(f"{_c}: {_mn} to {_mx}")
                                    if _cnt < _tot:
                                        _val_profile.append(f"{_c}: {_cnt}/{_tot} non-null")
                                except Exception:
                                    pass
                        except Exception:
                            pass

                    # 6. Input-output row ratio vs dependencies
                    if not _val_error and table_name in _val_tables and node.dependencies:
                        try:
                            _out_ct = int(self.datastore.query(f'SELECT COUNT(*) FROM "{table_name}"').iloc[0, 0])
                            for _dep_name in node.dependencies:
                                _dep_node = dag.get_node(_dep_name)
                                if _dep_node and _dep_node.row_count and _dep_node.row_count > 1:
                                    if abs(_out_ct - _dep_node.row_count) <= max(1, _dep_node.row_count * 0.2):
                                        _val_profile.append(f"Row count matches {_dep_node.fact_id} ({_out_ct} vs {_dep_node.row_count})")
                                    elif _out_ct > _dep_node.row_count:
                                        _val_profile.append(f"Expanded from {_dep_node.fact_id}: {_dep_node.row_count} → {_out_ct} rows")
                                    else:
                                        _val_profile.append(f"Filtered from {_dep_node.fact_id}: {_dep_node.row_count} → {_out_ct} rows")
                        except Exception:
                            pass

                    # 7. User-specified validations (from query constraints)
                    _user_validations = getattr(self, '_proof_user_validations', [])
                    if not _val_error and _user_validations and table_name in _val_tables:
                        for _uv in _user_validations:
                            try:
                                _uv_result = self.datastore.query(_uv['sql'].format(table=table_name))
                                _uv_ok = bool(_uv_result.iloc[0, 0]) if len(_uv_result) > 0 else False
                                if _uv_ok:
                                    _val_passed.append(f"{_uv['label']}")
                                else:
                                    _val_error = f"User validation failed: {_uv['label']}"
                            except Exception as _uv_e:
                                logger.debug(f"User validation '{_uv.get('label', '?')}' skipped for {table_name}: {_uv_e}")

                    if _val_error:
                        last_error = _val_error
                        logger.warning(f"[INFERENCE_CODE] {inf_id} attempt {attempt+1} validation: {_val_error}")
                        if first_error is None:
                            first_error = last_error
                            first_code = code
                        continue
                    # --- End validation ---

                    last_error = None
                    # Capture learning if this was a successful retry
                    if attempt > 0 and first_error and self.learning_store:
                        try:
                            self._capture_error_learning(
                                context={
                                    "error_message": first_error,
                                    "original_code": first_code[:500] if first_code else "",
                                    "step_goal": f"inference {inf_id}: {operation}",
                                },
                                fixed_code=code,
                            )
                        except Exception as le:
                            logger.debug(f"Learning capture failed: {le}")
                    break
                except Exception as e:
                    last_error = str(e)
                    logger.warning(f"[INFERENCE_CODE] {inf_id} attempt {attempt+1}/{max_retries} failed: {last_error}")
                    logger.debug(f"[INFERENCE_CODE] {inf_id} attempt {attempt+1} code:\n{code}")
                    if first_error is None:
                        first_error = last_error
                        first_code = code
                finally:
                    sys.stdout = old_stdout

            if last_error:
                logger.error(f"[INFERENCE_CODE] {inf_id} all attempts failed: {last_error}")
                logger.debug(f"[INFERENCE_CODE] Failed code for {inf_id}:\n{code}")
                raise Exception(last_error)

            computed = exec_globals.get('_result')
            # Log execution results for debugging
            tables_after = [t['name'] for t in self.datastore.list_tables()] if self.datastore else []
            logger.debug(f"[INFERENCE_CODE] {inf_id} exec complete: _result={computed is not None}, tables={tables_after}, expected={table_name}")

            # Check if result was saved as table
            if self.datastore and table_name in [t['name'] for t in self.datastore.list_tables()]:
                count_df = self.datastore.query(f"SELECT COUNT(*) FROM {table_name}")
                row_count = int(count_df.iloc[0, 0])
                node.row_count = row_count
                resolved_inferences[inf_id] = f"{row_count} rows"
                result_value = f"{row_count} rows"

                # Warn if row count is suspiciously low compared to inputs
                if row_count <= 1 and node.dependencies:
                    # Check if any dependency had more rows
                    for dep_name in node.dependencies:
                        dep_node = dag.get_node(dep_name)
                        if dep_node and dep_node.row_count and dep_node.row_count > 5:
                            logger.warning(
                                f"[INFERENCE_CODE] {inf_id} produced only {row_count} row(s) but "
                                f"dependency '{dep_name}' had {dep_node.row_count} rows. "
                                f"This may indicate incorrect aggregation."
                            )
            elif computed is not None:
                resolved_inferences[inf_id] = computed
                result_value = computed
            else:
                output = captured.getvalue().strip()
                if output:
                    resolved_inferences[inf_id] = output
                    result_value = output
                else:
                    # No table created, no _result, no output - this is likely a failure
                    # Check if operation suggests a table should have been created
                    if any(kw in operation.lower() for kw in ['join', 'filter', 'merge', 'apply', 'calculate', 'select']):
                        logger.error(f"[INFERENCE_CODE] {inf_id} ({inf_name}) produced no output. Code:\n{code[:500]}...")
                        raise ValueError(f"Inference {inf_id} ({inf_name}) did not produce expected table '{table_name}'")
                    resolved_inferences[inf_id] = "completed"
                    result_value = "completed"

            # Persist inference code to disk (separate from step plan code)
            if self.history and code:
                output = captured.getvalue().strip()
                self.history.save_inference_code(
                    session_id=self.session_id,
                    inference_id=inf_id,
                    name=inf_name,
                    operation=operation,
                    code=code,
                    attempt=attempt + 1,
                    output=output or None,
                )

            # Reduce confidence if inference used LLM fuzzy mapping
            used_llm = getattr(self, '_inference_used_llm_map', False)

            # Safety net: detect hardcoded mapping dicts in generated code
            # LLM sometimes embeds its knowledge as a literal dict instead of calling llm_map()
            if not used_llm and code and '.map(' in code:
                # Check for dict literals with 3+ string key-value pairs followed by .map()
                import ast
                try:
                    tree = ast.parse(code)
                    for node_ast in ast.walk(tree):
                        if isinstance(node_ast, ast.Dict) and len(node_ast.keys) >= 3:
                            # Check if most keys are string constants
                            str_keys = sum(1 for k in node_ast.keys if isinstance(k, ast.Constant) and isinstance(k.value, str))
                            if str_keys >= 3:
                                used_llm = True
                                logger.info(f"[INFERENCE_CODE] {inf_id}: detected hardcoded mapping dict ({str_keys} string keys) — flagging as LLM knowledge")
                                break
                except SyntaxError:
                    pass

            confidence = 0.65 if used_llm else 0.9
            source = FactSource.LLM_KNOWLEDGE if used_llm else FactSource.DERIVED

            # noinspection PyTypeChecker
            self.fact_resolver.add_user_fact(
                fact_name=inf_name,
                value=result_value,
                reasoning=f"Computed: {operation}" + (" (includes LLM fuzzy mapping)" if used_llm else ""),
                source=source,
                context=f"Code:\n{code}" if code else None,
            )

            return result_value, confidence, "llm_knowledge" if used_llm else "derived", _val_passed, _val_profile

    @staticmethod
    def _find_skill_script(scripts_dir: Path) -> Path | None:
        """Find the first executable Python script in a skill's scripts directory."""
        if not scripts_dir.exists():
            return None
        for ext in ("*.py",):
            scripts = sorted(scripts_dir.glob(ext))
            if scripts:
                return scripts[0]
        return None

    def _ensure_session_datastore(self, _problem: str) -> None:
        """Create history session_id and datastore if not yet initialized."""
        # session_id may be a server UUID (not a valid history directory) — check filesystem
        needs_history = not self.session_id
        if self.session_id:
            history_dir = self.history._session_dir(self.session_id)
            if not (history_dir / "session.json").exists():
                needs_history = True
        if needs_history:
            self.session_id = self.history.create_session(
                config_dict=self.config.model_dump(),
                databases=self.resources.database_names,
                apis=self.resources.api_names,
                documents=self.resources.document_names,
                server_session_id=self.server_session_id,
            )
        if not self.datastore:
            session_dir = self.history._session_dir(self.session_id)
            datastore_path = session_dir / "datastore.duckdb"
            tables_dir = session_dir / "tables"
            underlying_datastore = DataStore(db_path=datastore_path)
            self.datastore = RegistryAwareDataStore(
                datastore=underlying_datastore,
                registry=self.registry,
                user_id=self.user_id,
                session_id=self.session_id,
                tables_dir=tables_dir,
            )
            self.fact_resolver._datastore = self.datastore

    def _execute_skill_script(self, script_path: Path, _problem: str) -> dict | None:
        """Execute a skill script directly, bypassing planning.

        Loads the skill script via importlib.util (no sys.path/sys.modules)
        and calls its entry point (run_proof, run, or main). Returns
        solve()-compatible result dict on success, None on failure.
        """
        import importlib.util
        import time
        start_time = time.time()
        skill_name = script_path.parent.parent.name

        self._emit_event(StepEvent(
            event_type="step_start",
            step_number=1,
            data={"goal": f"Executing skill: {skill_name}"}
        ))

        try:
            # Load the script directly — no sys.path or sys.modules needed
            pkg_name = skill_name.replace("-", "_").replace(" ", "_")
            module_name = f"_constat_skill_{pkg_name}_{script_path.stem}"
            spec = importlib.util.spec_from_file_location(module_name, script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Resolve dependencies: inject dependency functions into module namespace
            skill = self.skill_manager.get_skill(skill_name)
            if skill and skill.dependencies:
                for dep_name in skill.dependencies:
                    dep_skill = self.skill_manager.get_skill(dep_name)
                    if not dep_skill or not dep_skill.exports:
                        continue
                    dep_dir = self.skill_manager.get_skill_dir(dep_name)
                    if not dep_dir:
                        continue
                    dep_scripts_dir = dep_dir / "scripts"
                    dep_pkg = dep_name.replace("-", "_").replace(" ", "_")
                    for export_entry in dep_skill.exports:
                        dep_script = dep_scripts_dir / export_entry.get("script", "")
                        if not dep_script.exists():
                            continue
                        dep_mod_name = f"_constat_skill_{dep_pkg}_{dep_script.stem}"
                        dep_spec = importlib.util.spec_from_file_location(dep_mod_name, dep_script)
                        dep_module = importlib.util.module_from_spec(dep_spec)
                        dep_spec.loader.exec_module(dep_module)
                        for fn_name in export_entry.get("functions", []):
                            fn = getattr(dep_module, fn_name, None)
                            if fn and callable(fn):
                                setattr(module, f"{dep_pkg}_{fn_name}", fn)

            # Find entry point: try common names
            entry_fn = None
            for fn_name in ('run_proof', 'run', 'main'):
                fn = getattr(module, fn_name, None)
                if callable(fn):
                    entry_fn = fn
                    break

            if entry_fn is None:
                logger.warning(f"[SKILL_EXEC] No entry point (run_proof/run/main) found in {pkg_name}")
                return None

            # Run the entry point
            results = entry_fn()

            if not isinstance(results, dict):
                logger.warning(f"[SKILL_EXEC] Entry point returned {type(results)}, expected dict")
                return None

            # Save result tables to datastore
            # run_proof() returns {name: parquet_path} or {name: DataFrame}
            import pandas as pd
            saved_tables = []
            for name, val in results.items():
                if name == '_result':
                    continue
                if isinstance(val, pd.DataFrame):
                    self.datastore.save_dataframe(name, val, step_number=1)
                    saved_tables.append(name)
                elif isinstance(val, str) and val.endswith('.parquet'):
                    df = pd.read_parquet(val)
                    self.datastore.save_dataframe(name, df, step_number=1)
                    saved_tables.append(name)

            # Save _result as the primary output
            primary_result = results.get('_result')
            if isinstance(primary_result, pd.DataFrame):
                self.datastore.save_dataframe('_result', primary_result, step_number=1)
            elif isinstance(primary_result, str) and primary_result.endswith('.parquet'):
                df = pd.read_parquet(primary_result)
                self.datastore.save_dataframe('_result', df, step_number=1)

            duration_ms = int((time.time() - start_time) * 1000)

            self._emit_event(StepEvent(
                event_type="step_complete",
                step_number=1,
                data={
                    "goal": f"Skill execution complete ({len(saved_tables)} tables)",
                    "duration_ms": duration_ms,
                }
            ))

            logger.info(f"[SKILL_EXEC] Success: {len(saved_tables)} tables saved in {duration_ms}ms")

            return {
                "steps": [{"step_number": 1, "goal": f"Skill: {skill_name}", "status": "success"}],
                "tables": saved_tables,
                "mode": "skill",
                "duration_ms": duration_ms,
            }

        except Exception as e:
            logger.warning(f"[SKILL_EXEC] Execution failed: {e}")
            self._emit_event(StepEvent(
                event_type="step_error",
                step_number=1,
                data={"error": str(e)}
            ))
            return None

    def _extract_user_validations(self, problem: str, inferences: list[dict]) -> list[dict]:
        """Extract user-specified validation constraints from the problem text.

        Looks for explicit constraints like "ensure no raise exceeds 15%",
        "verify total budget under $100k", etc. and converts them to SQL checks.

        Returns list of dicts with 'label' and 'sql' keys.
        The 'sql' value uses {table} placeholder for the target table name.
        """
        # Build inference context for the LLM
        inf_desc = "\n".join(
            f"- {inf.get('inference_id', '?')}: {inf.get('name', '')} = {inf.get('operation', '')}"
            for inf in inferences
        )

        prompt = f"""Analyze this user request for explicit validation constraints (ensure, verify, validate, must, should not exceed, at least, between, limit, cap, maximum, minimum, etc.):

USER REQUEST: {problem}

INFERENCES (output tables):
{inf_desc}

Extract ONLY explicitly stated constraints. Do NOT invent constraints that aren't in the request.

For each constraint, provide:
- label: Human-readable description of the check
- sql: A DuckDB SQL expression that returns TRUE if the constraint passes, using {{table}} as placeholder for the table name

Respond with ONLY valid JSON array. Empty array [] if no explicit constraints found.

Example:
[
  {{"label": "No raise exceeds 15%", "sql": "SELECT COUNT(*) = 0 FROM \\"{{table}}\\" WHERE raise_pct > 0.15"}},
  {{"label": "Total budget under $100k", "sql": "SELECT SUM(raise_amount) < 100000 FROM \\"{{table}}\\""}}
]

YOUR JSON RESPONSE:"""

        try:
            result = self.router.execute(
                task_type=TaskType.SQL_GENERATION,
                system="Extract validation constraints from user requests. Output ONLY valid JSON.",
                user_message=prompt,
            )
            content = result.content if hasattr(result, 'content') else str(result)
            # Strip markdown fences
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            import json
            validations = json.loads(content)
            if isinstance(validations, list):
                valid = [v for v in validations if isinstance(v, dict) and 'label' in v and 'sql' in v]
                if valid:
                    logger.info(f"[USER_VALIDATIONS] Extracted {len(valid)} constraints: {[v['label'] for v in valid]}")
                return valid
        except Exception as e:
            logger.debug(f"[USER_VALIDATIONS] Extraction failed: {e}")

        return []

    def add_user_validation(self, label: str, sql: str) -> None:
        """Add a user-specified validation constraint for proof inference checks.

        Args:
            label: Human-readable description (e.g., "No raise exceeds 15%")
            sql: DuckDB SQL that returns TRUE if valid. Use {table} placeholder.
        """
        self._proof_user_validations.append({"label": label, "sql": sql})
        logger.info(f"[USER_VALIDATIONS] Added: {label}")

    def _profile_table(self, table_name: str) -> dict:
        """Profile a datastore table for data quality assessment.

        Returns dict with row_count, column_count, duplicate_rows, and per-column stats.
        """
        row_count_df = self.datastore.query(f"SELECT COUNT(*) as cnt FROM {table_name}")
        row_count = int(row_count_df.iloc[0, 0])

        schema_df = self.datastore.query(f"DESCRIBE {table_name}")
        columns = []
        for _, row in schema_df.iterrows():
            col_name = row.iloc[0]
            col_type = str(row.iloc[1]) if len(row) > 1 else "unknown"
            try:
                stats = self.datastore.query(
                    f"SELECT "
                    f"COUNT(DISTINCT \"{col_name}\") as distinct_count, "
                    f"SUM(CASE WHEN \"{col_name}\" IS NULL OR CAST(\"{col_name}\" AS VARCHAR) = '' THEN 1 ELSE 0 END) as null_count "
                    f"FROM {table_name}"
                )
                # noinspection PyTypeChecker
                distinct = int(stats.iloc[0]['distinct_count'])
                # noinspection PyTypeChecker
                null_count = int(stats.iloc[0]['null_count'])
            except Exception:
                distinct = 0
                null_count = row_count

            null_pct = (null_count / row_count * 100) if row_count > 0 else 0
            columns.append({
                "name": col_name,
                "type": col_type,
                "distinct": distinct,
                "null_count": null_count,
                "null_pct": null_pct,
                "all_null": null_count == row_count,
            })

        # Check for duplicate rows
        try:
            dup_df = self.datastore.query(
                f"SELECT COUNT(*) - COUNT(DISTINCT *) as dups FROM (SELECT * FROM {table_name})"
            )
            duplicate_rows = int(dup_df.iloc[0, 0])
        except Exception:
            duplicate_rows = 0

        return {
            "row_count": row_count,
            "column_count": len(columns),
            "columns": columns,
            "duplicate_rows": duplicate_rows,
        }

    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert an object to be JSON-serializable.

        Handles pandas NA values (NAType), numpy types, and nested structures.
        """
        import numpy as np
        import pandas as pd

        if obj is None:
            return None
        # Handle pandas NA
        if pd.isna(obj):
            return None
        # Handle numpy types
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # Handle nested structures
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(v) for v in obj]
        return obj
