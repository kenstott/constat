# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Shared grounding utilities for proof node source pattern extraction."""

import re

_GROUNDABLE_SOURCES = {"database", "document", "api"}
_NON_GROUNDABLE_SOURCES = {"embedded", "derived", "llm_knowledge", "llm_heuristic",
                           "cache", "replay", "user_provided", "config", "unresolved"}

# Regex to extract table names from SQL FROM/JOIN clauses.
_SQL_TABLE_RE = re.compile(
    r'(?:FROM|JOIN)\s+'
    r'"?(\w+)"?'
    r'(?:\s*\.\s*"?(\w+)"?)?',
    re.IGNORECASE,
)


def _extract_tables_from_sql(sql: str) -> list[str]:
    """Extract real table names from a SQL query's FROM/JOIN clauses."""
    tables: list[str] = []
    seen: set[str] = set()
    for m in _SQL_TABLE_RE.finditer(sql):
        part1, part2 = m.group(1), m.group(2)
        name = part2 if part2 else part1
        key = name.lower()
        if key not in seen:
            seen.add(key)
            tables.append(name)
    return tables


def build_source_patterns(node: dict) -> list[str]:
    """Build grounding source patterns from a proof node dict.

    Returns a list of pattern strings like "schema:db.table", "api:endpoint",
    "document:name", or the raw source string.
    """
    raw_source = node.get("source", "")
    source_type = raw_source.split(":")[0] if raw_source else ""
    if source_type in _NON_GROUNDABLE_SOURCES or not source_type:
        return []

    source_name = node.get("source_name")
    table_name = node.get("table_name")
    api_endpoint = node.get("api_endpoint")
    query = node.get("query")

    source_suffix = raw_source.split(":", 1)[1] if ":" in raw_source else None
    if not source_name and source_type == "database" and source_suffix:
        source_name = source_suffix
    if not api_endpoint and source_type == "api" and source_suffix:
        api_endpoint = source_suffix

    resolves_to: list[str] = []
    if source_type == "database" and source_name:
            sql_tables = _extract_tables_from_sql(query) if query else []
            if sql_tables:
                for t in sql_tables:
                    resolves_to.append(f"schema:{source_name}.{t}")
            elif table_name:
                # Direct table reference (no SQL query executed)
                resolves_to.append(f"schema:{source_name}.{table_name}")
            else:
                resolves_to.append(f"schema:{source_name}")
    elif source_type == "api" and api_endpoint:
        resolves_to.append(f"api:{api_endpoint}")
    elif source_type == "api" and source_name:
        resolves_to.append(f"api:{source_name}")
    elif source_type == "document":
        # Prefer source_suffix (stable identifier from raw source like "document:business_rules")
        # over source_name (may contain volatile section metadata)
        doc_name = source_suffix or source_name
        if doc_name:
            resolves_to.append(f"document:{doc_name}")
    elif source_type not in _GROUNDABLE_SOURCES:
        # Domain-specific source (e.g., "hr", "business_rules") — LLM used the
        # domain config name as source rather than a canonical type.
        # Treat as a domain data source for grounding.
        resolves_to.append(f"domain:{source_type}")

    return resolves_to
