# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.
"""Metadata mixin: schema, entities, source discovery."""
from __future__ import annotations

import logging
import time
from typing import Any

from constat.prompts import load_yaml

logger = logging.getLogger(__name__)


# noinspection PyUnresolvedReferences
class MetadataMixin:

    def _cached_get_table_schema(self, table: str) -> dict:
        """Get table schema with caching."""
        cache_key = f"schema:{table}"
        if cache_key not in self._tool_cache:
            self._tool_cache[cache_key] = self.schema_manager.get_table_schema(table)
        return self._tool_cache[cache_key]

    def _cached_find_relevant_tables(self, query: str, top_k: int = 5) -> list[dict]:
        """Find relevant tables with caching and document enrichment."""
        cache_key = f"relevant:{query}:{top_k}"
        if cache_key not in self._tool_cache:
            self._tool_cache[cache_key] = self.schema_manager.find_relevant_tables(
                query, top_k, doc_tools=self.doc_tools
            )
        return self._tool_cache[cache_key]

    def _cached_find_relevant_apis(self, query: str, limit: int = 5) -> list[dict]:
        """Find relevant APIs with caching."""
        cache_key = f"apis:{query}:{limit}"
        if cache_key not in self._tool_cache:
            self._tool_cache[cache_key] = self.api_schema_manager.find_relevant_apis(
                query, limit=limit
            )
        return self._tool_cache[cache_key]

    def find_relevant_sources(
        self,
        query: str,
        table_limit: int = 5,
        doc_limit: int = 3,
        api_limit: int = 3,
        min_similarity: float = 0.3,
    ) -> dict:
        """Find all relevant sources (tables, documents, APIs) for a query.

        Performs unified semantic search across all configured data sources
        and returns ranked results from each category.

        Args:
            query: Natural language query
            table_limit: Max tables to return
            doc_limit: Max documents to return
            api_limit: Max API endpoints to return
            min_similarity: Minimum similarity threshold

        Returns:
            Dict with 'tables', 'documents', 'apis' keys, each containing
            a list of relevant sources with similarity scores.
        """
        results = {
            "tables": [],
            "documents": [],
            "apis": [],
        }

        # Find relevant tables (with doc enrichment)
        tables = self._cached_find_relevant_tables(query, top_k=table_limit)
        results["tables"] = [
            {
                "source_type": "table",
                "name": t["full_name"],
                "database": t["database"],
                "summary": t["summary"],
                "relevance": t["relevance"],
                "documentation": t.get("documentation", []),
            }
            for t in tables
            if t.get("relevance", 0) >= min_similarity
        ]

        # Find relevant documents
        if self.doc_tools:
            docs = self.doc_tools.search_documents(query, limit=doc_limit)
            results["documents"] = [
                {
                    "source_type": "document",
                    "name": d["document"],
                    "section": d.get("section"),
                    "excerpt": d.get("excerpt", ""),
                    "relevance": d.get("relevance", 0),
                }
                for d in docs
                if d.get("relevance", 0) >= min_similarity
            ]

        # Find relevant APIs
        if not self.config.apis:
            logger.debug(f"[find_relevant_sources] No APIs configured")
        if self.config.apis:
            apis = self._cached_find_relevant_apis(query, limit=api_limit)
            logger.debug(f"[find_relevant_sources] API search for '{query[:50]}': {len(apis)} results, min_sim={min_similarity}")
            for a in apis[:3]:
                logger.debug(f"[find_relevant_sources]   - {a.get('api_name')}.{a.get('endpoint')}: sim={a.get('similarity', 0):.3f}")
            results["apis"] = [
                {
                    "source_type": "api",
                    "name": f"{a['api_name']}.{a['endpoint']}",
                    "api_name": a["api_name"],
                    "endpoint": a["endpoint"],
                    "type": a["type"],
                    "description": a.get("description"),
                    "fields": a.get("fields", []),
                    "relevance": a.get("similarity", 0),
                }
                for a in apis
                if a.get("similarity", 0) >= min_similarity
            ]

        return results

    def _find_entity(self, name: str, limit: int = 3) -> dict:
        """Find all occurrences of an entity across schema and documents."""
        from constat.discovery.schema_tools import SchemaDiscoveryTools
        tools = SchemaDiscoveryTools(self.schema_manager, self.doc_tools)
        return tools.find_entity(name, limit)

    def _get_tool_handlers(self) -> dict:
        """Get schema tool handlers with caching."""
        handlers = {
            "get_table_schema": self._cached_get_table_schema,
            "find_relevant_tables": self._cached_find_relevant_tables,
            "find_entity": self._find_entity,
        }

        # Add API schema tools if APIs are configured
        if self.config.apis:
            # noinspection PyTypeChecker
            handlers["get_api_schema_overview"] = self._get_api_schema_overview
            # noinspection PyTypeChecker
            handlers["get_api_query_schema"] = self._get_api_query_schema
            handlers["find_relevant_apis"] = self._cached_find_relevant_apis

        return handlers

    def _get_api_schema_overview(self, api_name: str) -> dict:
        """Get overview of an API's schema (queries/endpoints)."""
        cache_key = f"api_overview:{api_name}"
        if cache_key not in self._tool_cache:
            from constat.catalog.api_executor import APIExecutor
            executor = APIExecutor(self.config)
            self._tool_cache[cache_key] = executor.get_schema_overview(api_name)
        return self._tool_cache[cache_key]

    def _get_api_query_schema(self, api_name: str, query_name: str) -> dict:
        """Get detailed schema for a specific API query/endpoint."""
        cache_key = f"api_query:{api_name}:{query_name}"
        if cache_key not in self._tool_cache:
            from constat.catalog.api_executor import APIExecutor
            executor = APIExecutor(self.config)
            self._tool_cache[cache_key] = executor.get_query_schema(api_name, query_name)
        return self._tool_cache[cache_key]

    def refresh_metadata(self, force_full: bool = False) -> dict:
        """Refresh all metadata: schema, documents, APIs, and preload cache.

        Args:
            force_full: If True, force full rebuild of all caches

        Returns:
            Dict with refresh statistics
        """
        self._tool_cache.clear()
        self.schema_manager.refresh()

        # Refresh API schema manager
        api_count = 0
        if self.config.apis:
            self.api_schema_manager.initialize()  # Re-introspect APIs
            api_count = len(self.api_schema_manager.metadata_cache)

        # Pass schema entities to doc_tools for entity extraction
        # IMPORTANT: Keep schema and API entities separate to avoid type confusion
        if self.doc_tools:
            schema_entities = set(self.schema_manager.get_entity_names())
            api_entities = list(self._get_api_entity_names())
            self.doc_tools.set_schema_entities(schema_entities)
            # Set API entities separately for proper type assignment
            if api_entities:
                self.doc_tools.set_openapi_entities(api_entities, api_entities)

            # Process schema metadata (names + descriptions) through NER
            schema_metadata = self.schema_manager.get_description_text()
            if schema_metadata:
                self.doc_tools.process_metadata_through_ner(schema_metadata, source_type="schema")

            # Process API metadata through NER
            api_metadata = self.api_schema_manager.get_description_text()
            if api_metadata:
                self.doc_tools.process_metadata_through_ner(api_metadata, source_type="api")

        # Refresh document vector index (incremental by default)
        doc_stats = {}
        if self.doc_tools:
            doc_stats = self.doc_tools.refresh(force_full=force_full)

        # Rebuild preload cache with fresh metadata
        self._rebuild_preload_cache()

        return {
            "preloaded_tables": self.get_preloaded_tables_count(),
            "documents": doc_stats,
            "api_endpoints": api_count,
        }

    def _get_api_entity_names(self) -> set[str]:
        """Get API endpoint names for entity extraction."""
        if not self.config.apis:
            return set()

        entities = set()
        for meta in self.api_schema_manager.metadata_cache.values():
            # Add endpoint name
            entities.add(meta.endpoint_name)
            # Add API.endpoint as full name
            entities.add(meta.full_name)
            # Add field names
            for field in meta.fields:
                entities.add(field.name)
        return entities

    def extract_entities(self, domain_ids: list[str] | None = None) -> int:
        """Extract NER entities from all chunks for this session.

        Should be called after session_id is set. Extracts entities from
        base + specified domain chunks.

        Args:
            domain_ids: Domain IDs to include (in addition to base)

        Returns:
            Number of entities extracted
        """
        if not self.session_id:
            logger.warning("extract_entities called without session_id")
            return 0

        if self._entities_extracted:
            logger.debug("Entities already extracted for this session")
            return 0

        t0 = time.time()
        schema_terms = list(self.schema_manager.get_entity_names())
        api_terms = list(self._get_api_entity_names())

        entity_count = self.doc_tools._vector_store.extract_entities_for_session(
            session_id=self.session_id,
            domain_ids=domain_ids,
            schema_terms=schema_terms,
            api_terms=api_terms,
        )

        # noinspection PyAttributeOutsideInit
        self._entities_extracted = True
        logger.debug(f"Entity extraction took {time.time() - t0:.2f}s ({entity_count} entities)")
        return entity_count

    def rebuild_entities(self, domain_ids: list[str] | None = None) -> int:
        """Rebuild entity catalog (e.g., when domains change).

        Clears existing entities for this session and re-extracts.

        Args:
            domain_ids: Domain IDs to include (in addition to base)

        Returns:
            Number of entities extracted
        """
        if not self.session_id:
            logger.warning("rebuild_entities called without session_id")
            return 0

        # noinspection PyAttributeOutsideInit
        self._entities_extracted = False
        return self.extract_entities(domain_ids)

    def add_domain_api(self, name: str, api_config: Any) -> None:
        """Register an API from an active domain.

        Args:
            name: API name
            api_config: API configuration (ApiConfig object)
        """
        self._domain_apis[name] = api_config
        logger.info(f"Registered domain API: {name}")

    def clear_domain_apis(self) -> None:
        """Clear all registered domain APIs."""
        self._domain_apis.clear()

    def get_all_apis(self) -> dict[str, Any]:
        """Get all APIs (config + domain).

        Returns:
            Dict of API name to config, combining config.apis and domain APIs
        """
        all_apis = dict(self.config.apis) if self.config.apis else {}
        all_apis.update(self._domain_apis)
        return all_apis

    def _load_preloaded_context(self) -> None:
        """Load preloaded metadata context from cache if available."""
        if self.config.context_preload.seed_patterns:
            self._preloaded_context = self.preload_cache.get_context_string()

    def _rebuild_preload_cache(self) -> None:
        """Rebuild the preload cache with current metadata."""
        if self.config.context_preload.seed_patterns:
            self.preload_cache.build(self.schema_manager)
            self._preloaded_context = self.preload_cache.get_context_string()

    def get_preloaded_tables_count(self) -> int:
        """Get the number of tables in the preload cache."""
        return len(self.preload_cache.get_cached_tables())

    def _get_brief_schema_summary(self) -> str:
        """Get a brief summary of available databases without listing all tables.

        This is used when no preload config is set. It provides enough context
        for the LLM to know what databases exist without bloating the prompt
        with potentially hundreds of tables.
        """
        lines = ["Available databases:"]

        # Group tables by database to get counts
        by_db: dict[str, int] = {}
        total_rows_by_db: dict[str, int] = {}
        for table_meta in self.schema_manager.metadata_cache.values():
            by_db[table_meta.database] = by_db.get(table_meta.database, 0) + 1
            total_rows_by_db[table_meta.database] = (
                total_rows_by_db.get(table_meta.database, 0) + table_meta.row_count
            )

        # Build database descriptions with descriptions if available
        db_descriptions = {
            db_name: db_config.description
            for db_name, db_config in self.config.databases.items()
            if db_config.description
        }

        for db_name in sorted(by_db.keys()):
            table_count = by_db[db_name]
            row_count = total_rows_by_db[db_name]
            if db_name in db_descriptions:
                lines.append(f"  {db_name} (connection: db_{db_name}): {db_descriptions[db_name]} ({table_count} tables, ~{row_count:,} rows)")
            else:
                lines.append(f"  {db_name} (connection: db_{db_name}): {table_count} tables, ~{row_count:,} rows")

        return "\n".join(lines)

    def _format_relevant_tables(self, tables: list[dict]) -> str:
        """Format semantically matched tables with database connection names."""
        if not tables:
            return self._get_brief_schema_summary()

        lines = ["Relevant tables (use pd.read_sql(query, db_<name>) with the correct connection):"]

        # Group by database for clarity
        by_db: dict[str, list] = {}
        for t in tables:
            by_db.setdefault(t["database"], []).append(t)

        for db_name, db_tables in sorted(by_db.items()):
            lines.append(f"\n  Database '{db_name}' — connection: db_{db_name}")
            for t in db_tables:
                full_name = t["name"]
                table_meta = self.schema_manager.metadata_cache.get(full_name) if self.schema_manager else None
                if table_meta:
                    col_names = ", ".join(c.name for c in table_meta.columns[:15])
                    if len(table_meta.columns) > 15:
                        col_names += f", ... (+{len(table_meta.columns) - 15} more)"
                    lines.append(f"    {table_meta.name}({col_names}) ~{table_meta.row_count} rows")
                else:
                    lines.append(f"    {t.get('name', '')}: {t.get('summary', '')}")

        lines.append("\nUse `find_relevant_tables(query)` or `get_table_schema(table)` for other tables.")
        return "\n".join(lines)

    def _format_relevant_apis(self, apis: list[dict]) -> str:
        """Format semantically matched APIs."""
        if not apis:
            # Fall back to listing all configured APIs
            if self.resources.has_apis():
                api_lines = ["\n## Available APIs"]
                for name, api_info in self.resources.apis.items():
                    api_type = api_info.api_type.upper()
                    desc = api_info.description or f"{api_type} endpoint"
                    api_lines.append(f"- **{name}** ({api_type}): {desc}")
                return "\n".join(api_lines)
            return ""

        lines = ["\n## Relevant APIs"]
        for a in apis:
            name = a.get("name", "")
            desc = a.get("description", "")
            api_type = a.get("type", "").upper()
            lines.append(f"- **{name}** ({api_type}): {desc}")
            if a.get("fields"):
                lines.append(f"  Fields: {', '.join(a['fields'][:10])}")
        return "\n".join(lines)

    def _format_relevant_docs(self, docs: list[dict]) -> str:
        """Format semantically matched documents."""
        if not docs:
            # Fall back to listing all configured documents
            if self.resources.has_documents():
                doc_lines = ["\n## Reference Documents"]
                for name, doc_info in self.resources.documents.items():
                    desc = doc_info.description or doc_info.doc_type
                    doc_lines.append(f"- **{name}**: {desc}")
                return "\n".join(doc_lines)
            return ""

        lines = ["\n## Relevant Documents"]
        for d in docs:
            name = d.get("name", "")
            section = d.get("section", "")
            excerpt = d.get("excerpt", "")
            section_info = f" (section: {section})" if section else ""
            lines.append(f"- **{name}**{section_info}")
            if excerpt:
                lines.append(f"  > {excerpt[:200]}")
        return "\n".join(lines)

    def _get_schema_tools(self) -> list[dict]:
        """Get schema tool definitions loaded from schema_tools.yaml."""
        import copy

        data = load_yaml("schema_tools.yaml")
        tools = copy.deepcopy(data["base_tools"])

        if self.config.apis:
            api_names = list(self.config.apis.keys())
            api_tools = copy.deepcopy(data["api_tools"])
            for tool in api_tools:
                tool["description"] = tool["description"].format(api_names=api_names)
                for prop in tool["input_schema"]["properties"].values():
                    if prop.get("type") == "string" and "api_name" in tool["input_schema"].get("required", []):
                        if "enum" not in prop and prop.get("description", "").startswith("Name of the API"):
                            prop["enum"] = api_names
            tools.extend(api_tools)

        return tools
