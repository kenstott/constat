# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Schema discovery tools for databases and tables.

These tools allow the LLM to discover database schemas on-demand
rather than loading everything into the system prompt upfront.
"""

from typing import Optional, TYPE_CHECKING

from constat.catalog.schema_manager import SchemaManager

if TYPE_CHECKING:
    from constat.discovery.doc_tools import DocumentDiscoveryTools
    from constat.discovery.api_tools import APIDiscoveryTools


class SchemaDiscoveryTools:
    """Tools for discovering database schemas on-demand."""

    def __init__(
        self,
        schema_manager: SchemaManager,
        doc_tools: Optional["DocumentDiscoveryTools"] = None,
        api_tools: Optional["APIDiscoveryTools"] = None,
        allowed_databases: Optional[set[str]] = None,
        session_id: Optional[str] = None,
    ):
        """Initialize schema discovery tools.

        Args:
            schema_manager: Schema manager for database metadata
            doc_tools: Optional document discovery tools
            api_tools: Optional API discovery tools
            allowed_databases: Set of allowed database names. If None, all databases
                are visible. If empty set, no databases are visible.
            session_id: Session ID for glossary enrichment
        """
        self.schema_manager = schema_manager
        self.doc_tools = doc_tools
        self.api_tools = api_tools
        self.allowed_databases = allowed_databases
        self.session_id = session_id

    def _is_database_allowed(self, db_name: str) -> bool:
        """Check if a database is allowed based on permissions."""
        if self.allowed_databases is None:
            return True  # No filtering
        return db_name in self.allowed_databases

    def list_databases(self) -> list[dict]:
        """
        List all available databases with their types and descriptions.

        Returns:
            List of database info dicts with name, type, description, table_count
        """
        results = []

        # Get database configs
        for db_name, db_config in self.schema_manager.config.databases.items():
            # Skip databases not allowed by permissions
            if not self._is_database_allowed(db_name):
                continue

            # Count tables in this database
            table_count = sum(
                1 for meta in self.schema_manager.metadata_cache.values()
                if meta.database == db_name
            )

            results.append({
                "name": db_name,
                "type": db_config.type,
                "description": db_config.description or f"Database: {db_name}",
                "table_count": table_count,
            })

        return results

    def list_tables(self, database: str) -> list[dict]:
        """
        List all tables in a specific database with row counts and descriptions.

        Args:
            database: Database name

        Returns:
            List of table info dicts with name, row_count, description, column_count
        """
        # Check permissions
        if not self._is_database_allowed(database):
            return []

        results = []

        for meta in self.schema_manager.metadata_cache.values():
            if meta.database == database:
                results.append({
                    "name": meta.name,
                    "row_count": meta.row_count,
                    "description": meta.comment or f"Table with {len(meta.columns)} columns",
                    "column_count": len(meta.columns),
                    "has_foreign_keys": len(meta.foreign_keys) > 0,
                })

        # Sort by name
        results.sort(key=lambda x: x["name"])
        return results

    def get_table_schema(self, database: str, table: str) -> dict:
        """
        Get detailed schema for a specific table.

        Args:
            database: Database name
            table: Table name

        Returns:
            Dict with full column details, types, keys, relationships, sample values
        """
        # Check permissions
        if not self._is_database_allowed(database):
            return {"error": f"Access denied to database: {database}"}

        full_name = f"{database}.{table}"

        if full_name not in self.schema_manager.metadata_cache:
            # Try just table name
            try:
                return self.schema_manager.get_table_schema(table)
            except (KeyError, ValueError) as e:
                return {"error": str(e)}

        meta = self.schema_manager.metadata_cache[full_name]
        return meta.to_dict()

    def search_tables(self, query: str, limit: int = 5) -> list[dict]:
        """
        Find tables relevant to a natural language query using vector search.

        Args:
            query: Natural language description (e.g., "customer purchases", "product inventory")
            limit: Maximum number of results

        Returns:
            List of relevant tables with database, name, relevance score, and summary
        """
        # Get more results to account for filtering
        fetch_limit = limit * 3 if self.allowed_databases is not None else limit
        results = self.schema_manager.find_relevant_tables(query, top_k=fetch_limit)

        # Filter by allowed databases and enhance with descriptions
        filtered = []
        for result in results:
            db_name = result.get("database")
            if not self._is_database_allowed(db_name):
                continue

            full_name = result.get("full_name", f"{db_name}.{result['table']}")
            if full_name in self.schema_manager.metadata_cache:
                meta = self.schema_manager.metadata_cache[full_name]
                if meta.comment:
                    result["description"] = meta.comment

            filtered.append(result)
            if len(filtered) >= limit:
                break

        return filtered

    def get_table_relationships(self, database: str, table: str) -> dict:
        """
        Get foreign key relationships for a table.

        Args:
            database: Database name
            table: Table name

        Returns:
            Dict with outgoing relationships (this table references) and
            incoming relationships (tables that reference this table)
        """
        # Check permissions
        if not self._is_database_allowed(database):
            return {"error": f"Access denied to database: {database}"}

        full_name = f"{database}.{table}"

        if full_name not in self.schema_manager.metadata_cache:
            return {"error": f"Table not found: {full_name}"}

        meta = self.schema_manager.metadata_cache[full_name]

        return {
            "table": full_name,
            "references": [
                {
                    "column": fk.from_column,
                    "references_table": fk.to_table,
                    "references_column": fk.to_column,
                }
                for fk in meta.foreign_keys
            ],
            "referenced_by": meta.referenced_by,
        }

    def get_sample_values(
        self, database: str, table: str, column: str, limit: int = 10
    ) -> dict:
        """
        Get sample distinct values for a column.

        Useful for understanding enum-like columns or data distribution.

        Args:
            database: Database name
            table: Table name
            column: Column name
            limit: Maximum distinct values to return

        Returns:
            Dict with column info and sample values
        """
        from sqlalchemy import text

        # Check if it's a SQL database
        if database not in self.schema_manager.connections:
            return {"error": f"Sample values only supported for SQL databases. {database} is NoSQL."}

        engine = self.schema_manager.connections[database]

        try:
            with engine.connect() as conn:
                # Get distinct values
                query = text(f'''
                    SELECT DISTINCT "{column}"
                    FROM "{table}"
                    WHERE "{column}" IS NOT NULL
                    LIMIT :limit
                ''')
                result = conn.execute(query, {"limit": limit})
                values = [row[0] for row in result]

                # Get count of distinct values
                count_query = text(f'SELECT COUNT(DISTINCT "{column}") FROM "{table}"')
                total_distinct = conn.execute(count_query).scalar()

                return {
                    "database": database,
                    "table": table,
                    "column": column,
                    "sample_values": values,
                    "total_distinct": total_distinct,
                    "is_complete": total_distinct <= limit,
                }
        except Exception as e:
            return {"error": str(e)}

    def find_entity(self, name: str, limit: int = 3) -> dict:
        """
        Find all occurrences of an entity across schema and documents.

        This unified search returns:
        - Schema matches: tables and columns matching the entity name
        - Document mentions: excerpts from docs that reference the entity

        Args:
            name: Entity name to search for (e.g., "Customer", "order_id")
            limit: Maximum document excerpts per entity (default 3)

        Returns:
            Dict with entity name, schema matches, and document excerpts
        """
        name_lower = name.lower()
        results = {
            "entity": name,
            "schema": [],
            "documents": [],
        }

        # 1. Find matching tables
        for full_name, table_meta in self.schema_manager.metadata_cache.items():
            # Check if table name matches (case-insensitive, partial match)
            if name_lower in table_meta.name.lower():
                col_names = [c.name for c in table_meta.columns[:5]]
                if len(table_meta.columns) > 5:
                    col_names.append(f"... +{len(table_meta.columns) - 5} more")
                results["schema"].append({
                    "type": "table",
                    "name": table_meta.name,
                    "database": table_meta.database,
                    "columns": col_names,
                    "row_count": table_meta.row_count,
                })

        # 2. Find matching columns across all tables
        seen_columns = set()  # Avoid duplicates
        for full_name, table_meta in self.schema_manager.metadata_cache.items():
            for col in table_meta.columns:
                col_key = f"{table_meta.name}.{col.name}"
                if name_lower in col.name.lower() and col_key not in seen_columns:
                    seen_columns.add(col_key)
                    # noinspection PyUnresolvedReferences
                    results["schema"].append({
                        "type": "column",
                        "name": col.name,
                        "table": table_meta.name,
                        "database": table_meta.database,
                        "data_type": col.type,
                        "nullable": col.nullable,
                        "is_primary_key": col.primary_key,
                    })

        # 3. Find document mentions via explore_entity
        if self.doc_tools:
            try:
                doc_context = self.doc_tools.explore_entity(name, limit=limit)
                if doc_context:
                    results["documents"] = [
                        {
                            "document": d.get("document"),
                            "excerpt": d.get("excerpt"),
                            "section": d.get("section"),
                            "relevance": d.get("relevance"),
                        }
                        for d in doc_context
                    ]
            except Exception:
                pass  # Don't fail if doc search fails

        # Add summary counts
        # noinspection PyTypeChecker
        results["summary"] = {
            "tables_found": sum(1 for s in results["schema"] if s["type"] == "table"),
            "columns_found": sum(1 for s in results["schema"] if s["type"] == "column"),
            "documents_found": len(results["documents"]),
        }

        return results

    def search_all(self, query: str, limit: int = 10) -> dict:
        """
        Universal semantic search across ALL data sources: tables, APIs, and documents.

        Uses vector embeddings to find relevant resources based on natural language query.
        This is the primary discovery tool - use it first to find what's relevant to
        your question before exploring specific resources.

        Args:
            query: Natural language query (e.g., "employee compensation", "order history")
            limit: Maximum total results to return (distributed across sources)

        Returns:
            Dict with categorized results:
            - tables: Relevant database tables with similarity scores
            - apis: Relevant API endpoints with similarity scores
            - documents: Relevant document excerpts with similarity scores
            - summary: Counts per category
        """
        results = {
            "query": query,
            "tables": [],
            "apis": [],
            "documents": [],
            "glossary": [],
            "relationships": [],
        }

        # Calculate per-source limits (distribute evenly, favor tables)
        per_source_limit = max(3, limit // 3)

        # 1. Search tables via vector store
        try:
            table_results = self.schema_manager.find_relevant_tables(query, top_k=per_source_limit * 2)
            for r in table_results:
                db_name = r.get("database")
                if not self._is_database_allowed(db_name):
                    continue
                results["tables"].append({
                    "type": "table",
                    "name": r.get("table"),
                    "database": db_name,
                    "relevance": round(r.get("relevance", 0), 3),
                    "summary": r.get("summary", ""),
                })
                if len(results["tables"]) >= per_source_limit:
                    break
        except Exception:
            pass  # Continue even if table search fails

        # 2. Search APIs via API tools if available
        if self.api_tools:
            try:
                api_results = self.api_tools.search_operations(query, limit=per_source_limit)
                for r in api_results:
                    results["apis"].append({
                        "type": "api_operation",
                        "name": r.get("name"),
                        "api": r.get("api_name", ""),
                        "operation_type": r.get("type", ""),
                        "protocol": r.get("protocol", ""),
                        "relevance": round(r.get("relevance", 0), 3),
                        "summary": r.get("summary", ""),
                    })
            except Exception:
                pass  # Continue even if API search fails

        # 3. Search documents via doc tools if available
        if self.doc_tools:
            try:
                doc_results = self.doc_tools.search_documents(query, limit=per_source_limit)
                for r in doc_results:
                    results["documents"].append({
                        "type": "document",
                        "name": r.get("document"),
                        "section": r.get("section"),
                        "excerpt": r.get("excerpt", "")[:300] + "..." if len(r.get("excerpt", "")) > 300 else r.get("excerpt", ""),
                        "relevance": round(r.get("relevance", 0), 3),
                    })
            except Exception:
                pass  # Continue even if doc search fails

        # 4. Search glossary + relationship chunks via vector store
        if self.doc_tools and hasattr(self.doc_tools, '_vector_store') and self.doc_tools._vector_store:
            try:
                from constat.discovery.models import ChunkType
                import threading

                vs = self.doc_tools._vector_store
                model = self.doc_tools._model
                model_lock = self.doc_tools._model_lock if hasattr(self.doc_tools, '_model_lock') else threading.Lock()

                # Encode query to embedding
                with model_lock:
                    query_embedding = model.encode([query], normalize_embeddings=True)

                # Search glossary term chunks
                glossary_hits = vs.search(
                    query_embedding, limit=per_source_limit,
                    chunk_types=[ChunkType.GLOSSARY_TERM],
                )

                # Batch-fetch full glossary terms for enrichment
                hit_names = [
                    chunk.document_name.replace("glossary:", "")
                    for _, _, chunk in glossary_hits
                ]
                terms_by_name = {}
                if hit_names and self.session_id:
                    fetched = vs.get_glossary_terms_by_names(hit_names, self.session_id)
                    terms_by_name = {t.name.lower(): t for t in fetched}

                for chunk_id, similarity, chunk in glossary_hits:
                    name = chunk.document_name.replace("glossary:", "")
                    term = terms_by_name.get(name.lower())

                    entry = {
                        "type": "glossary_term",
                        "name": name,
                        "relevance": round(similarity, 3),
                    }

                    if term:
                        entry["definition"] = term.definition
                        entry["aliases"] = term.aliases or []
                        entry["status"] = term.status

                        # Parent name
                        if term.parent_id:
                            parent = vs.get_glossary_term_by_id(term.parent_id)
                            entry["parent"] = parent.display_name if parent else None

                        # 1st-order SVO relationships
                        entity = vs.find_entity_by_name(name, session_id=self.session_id)
                        if entity:
                            rels = vs.get_relationships_for_entity(entity.id, self.session_id)
                            entry["relationships"] = [
                                f"{r['subject_name']} {r['verb']} {r['object_name']}"
                                for r in rels[:5]
                            ]
                            # Physical source count (exclude glossary/relationship chunks)
                            chunks_for = vs.get_chunks_for_entity(entity.id, limit=50)
                            sources = set(
                                c.document_name for _, c, _ in chunks_for
                                if not c.document_name.startswith("glossary:") and not c.document_name.startswith("relationship:")
                            )
                            entry["source_count"] = len(sources)
                    else:
                        entry["definition"] = chunk.content[:300]

                    results["glossary"].append(entry)

                # Search relationship chunks
                rel_hits = vs.search(
                    query_embedding, limit=per_source_limit,
                    chunk_types=[ChunkType.RELATIONSHIP],
                )
                for chunk_id, similarity, chunk in rel_hits:
                    results["relationships"].append({
                        "type": "relationship",
                        "name": chunk.document_name,
                        "definition": chunk.content[:300],
                        "relevance": round(similarity, 3),
                    })
            except Exception:
                pass

        # Add summary
        # noinspection PyTypeChecker
        results["summary"] = {
            "tables_found": len(results["tables"]),
            "apis_found": len(results["apis"]),
            "documents_found": len(results["documents"]),
            "glossary_found": len(results["glossary"]),
            "relationships_found": len(results["relationships"]),
            "total": (len(results["tables"]) + len(results["apis"]) +
                      len(results["documents"]) + len(results["glossary"]) +
                      len(results["relationships"])),
        }

        return results

    def lookup_glossary_term(self, name: str) -> dict:
        """Look up a glossary term by name or alias with full details.

        Args:
            name: Term name or alias

        Returns:
            Full term details including hierarchy, relationships, physical resources
        """
        if not self.doc_tools or not hasattr(self.doc_tools, '_vector_store') or not self.doc_tools._vector_store:
            return {"error": "Glossary not available"}
        if not self.session_id:
            return {"error": "No session context"}

        vs = self.doc_tools._vector_store
        term = vs.get_glossary_term_by_name_or_alias(name, self.session_id)
        if not term:
            return {"error": f"Term '{name}' not found"}

        result: dict = {
            "name": term.name,
            "display_name": term.display_name,
            "definition": term.definition,
            "aliases": term.aliases or [],
            "semantic_type": term.semantic_type,
            "status": term.status,
            "cardinality": term.cardinality,
        }

        # Parent
        if term.parent_id:
            parent = vs.get_glossary_term_by_id(term.parent_id)
            result["parent"] = parent.display_name if parent else None

        # Children
        children = vs.get_child_terms(term.id)
        if children:
            result["children"] = [
                {"name": c.name, "display_name": c.display_name}
                for c in children
            ]

        # SVO relationships
        entity = vs.find_entity_by_name(term.name, session_id=self.session_id)
        if entity:
            rels = vs.get_relationships_for_entity(entity.id, self.session_id)
            result["relationships"] = [
                {
                    "subject": r["subject_name"],
                    "verb": r["verb"],
                    "object": r["object_name"],
                    "confidence": r["confidence"],
                }
                for r in rels
            ]

        # Physical resources
        from constat.discovery.glossary_generator import resolve_physical_resources
        resources = resolve_physical_resources(term.name, self.session_id, vs)
        if resources:
            result["physical_resources"] = resources

        return result


# Tool schemas for LLM
SCHEMA_TOOL_SCHEMAS = [
    {
        "name": "list_databases",
        "description": "List all available databases with their types and descriptions. Use this first to see what data sources are available.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "list_tables",
        "description": "List all tables in a specific database with row counts and descriptions. Use after list_databases to explore a database.",
        "input_schema": {
            "type": "object",
            "properties": {
                "database": {
                    "type": "string",
                    "description": "Name of the database to list tables from",
                },
            },
            "required": ["database"],
        },
    },
    {
        "name": "get_table_schema",
        "description": "Get detailed schema for a specific table including columns, types, primary keys, and foreign keys. Use this to understand table structure before writing queries.",
        "input_schema": {
            "type": "object",
            "properties": {
                "database": {
                    "type": "string",
                    "description": "Name of the database",
                },
                "table": {
                    "type": "string",
                    "description": "Name of the table",
                },
            },
            "required": ["database", "table"],
        },
    },
    {
        "name": "search_tables",
        "description": "Find tables relevant to a natural language query using semantic search. Use this when you're not sure which tables contain the data you need.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language description of what data you're looking for (e.g., 'customer purchases', 'product inventory')",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_table_relationships",
        "description": "Get foreign key relationships for a table - both tables it references and tables that reference it. Use this to understand how to join tables.",
        "input_schema": {
            "type": "object",
            "properties": {
                "database": {
                    "type": "string",
                    "description": "Name of the database",
                },
                "table": {
                    "type": "string",
                    "description": "Name of the table",
                },
            },
            "required": ["database", "table"],
        },
    },
    {
        "name": "get_sample_values",
        "description": "Get sample distinct values for a column. Use this to understand enum-like columns or data distribution.",
        "input_schema": {
            "type": "object",
            "properties": {
                "database": {
                    "type": "string",
                    "description": "Name of the database",
                },
                "table": {
                    "type": "string",
                    "description": "Name of the table",
                },
                "column": {
                    "type": "string",
                    "description": "Name of the column",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum distinct values to return",
                    "default": 10,
                },
            },
            "required": ["database", "table", "column"],
        },
    },
    {
        "name": "find_entity",
        "description": "Find all occurrences of an entity across schema and documents. Returns matching tables, columns, and document excerpts that mention the entity. Use this to get a complete picture of where a concept appears in both data structures and documentation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Entity name to search for (e.g., 'Customer', 'order_id', 'pricing')",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum document excerpts to return",
                    "default": 3,
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "search_all",
        "description": "Universal semantic search across ALL data sources: tables, APIs, documents, and the business glossary. Uses vector embeddings to find relevant resources. Glossary results include curated business definitions with connected physical resources. This is the PRIMARY discovery tool - use it FIRST to find what's relevant to your question before exploring specific resources.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language query (e.g., 'employee compensation', 'order history', 'performance review guidelines')",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum total results to return",
                    "default": 10,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "lookup_glossary_term",
        "description": "Look up a glossary term by name or alias. Returns full definition, aliases, parent/child hierarchy, SVO relationships, and physical resource connections. Use after search_all finds a relevant glossary term.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Term name or alias (e.g., 'compensation', 'MRR')",
                },
            },
            "required": ["name"],
        },
    },
]
