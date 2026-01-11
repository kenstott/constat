"""Schema discovery tools for databases and tables.

These tools allow the LLM to discover database schemas on-demand
rather than loading everything into the system prompt upfront.
"""

from typing import Optional

from constat.catalog.schema_manager import SchemaManager


class SchemaDiscoveryTools:
    """Tools for discovering database schemas on-demand."""

    def __init__(self, schema_manager: SchemaManager):
        self.schema_manager = schema_manager

    def list_databases(self) -> list[dict]:
        """
        List all available databases with their types and descriptions.

        Returns:
            List of database info dicts with name, type, description, table_count
        """
        results = []

        # Get database configs
        for db_name, db_config in self.schema_manager.config.databases.items():
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
        results = self.schema_manager.find_relevant_tables(query, top_k=limit)

        # Enhance with descriptions
        for result in results:
            full_name = result.get("full_name", f"{result['database']}.{result['table']}")
            if full_name in self.schema_manager.metadata_cache:
                meta = self.schema_manager.metadata_cache[full_name]
                if meta.comment:
                    result["description"] = meta.comment

        return results

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
]
