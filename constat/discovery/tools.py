"""Unified discovery tool registry for LLM integration.

This module provides a unified interface for all discovery tools,
making it easy to integrate with the planner and executor.
"""

from typing import Any, Callable, Optional

from constat.core.config import Config
from constat.catalog.schema_manager import SchemaManager
from constat.catalog.api_catalog import APICatalog
from constat.execution.fact_resolver import FactResolver

from .schema_tools import SchemaDiscoveryTools, SCHEMA_TOOL_SCHEMAS
from .api_tools import APIDiscoveryTools, API_TOOL_SCHEMAS
from .doc_tools import DocumentDiscoveryTools, DOC_TOOL_SCHEMAS
from .fact_tools import FactResolutionTools, FACT_TOOL_SCHEMAS


# All discovery tool schemas combined
DISCOVERY_TOOL_SCHEMAS = (
    SCHEMA_TOOL_SCHEMAS +
    API_TOOL_SCHEMAS +
    DOC_TOOL_SCHEMAS +
    FACT_TOOL_SCHEMAS
)


class DiscoveryTools:
    """
    Unified interface for all discovery tools.

    Provides:
    - Schema discovery (databases, tables)
    - API discovery (GraphQL, OpenAPI)
    - Document discovery (reference docs)
    - Fact resolution (multi-source)

    Usage:
        tools = DiscoveryTools(schema_manager, api_catalog, config)

        # Get tool schemas for LLM
        schemas = tools.get_tool_schemas()

        # Execute a tool call from LLM
        result = tools.execute("list_databases", {})
        result = tools.execute("search_tables", {"query": "customer purchases"})
    """

    def __init__(
        self,
        schema_manager: Optional[SchemaManager] = None,
        api_catalog: Optional[APICatalog] = None,
        config: Optional[Config] = None,
        fact_resolver: Optional[FactResolver] = None,
    ):
        self.config = config

        # Initialize tool classes
        self.schema_tools = SchemaDiscoveryTools(schema_manager) if schema_manager else None
        self.api_tools = APIDiscoveryTools(api_catalog, config) if api_catalog else None
        self.doc_tools = DocumentDiscoveryTools(config) if config else None
        self.fact_tools = FactResolutionTools(fact_resolver, self.doc_tools)

        # Build tool handler map
        self._handlers: dict[str, Callable[..., Any]] = {}
        self._build_handler_map()

    def _build_handler_map(self) -> None:
        """Build mapping from tool names to handler methods."""
        # Schema tools
        if self.schema_tools:
            self._handlers.update({
                "list_databases": self.schema_tools.list_databases,
                "list_tables": self.schema_tools.list_tables,
                "get_table_schema": self.schema_tools.get_table_schema,
                "search_tables": self.schema_tools.search_tables,
                "get_table_relationships": self.schema_tools.get_table_relationships,
                "get_sample_values": self.schema_tools.get_sample_values,
            })

        # API tools
        if self.api_tools:
            self._handlers.update({
                "list_apis": self.api_tools.list_apis,
                "list_api_operations": self.api_tools.list_operations,
                "get_operation_details": self.api_tools.get_operation_details,
                "search_operations": self.api_tools.search_operations,
            })

        # Document tools
        if self.doc_tools:
            self._handlers.update({
                "list_documents": self.doc_tools.list_documents,
                "get_document": self.doc_tools.get_document,
                "search_documents": self.doc_tools.search_documents,
                "get_document_section": self.doc_tools.get_document_section,
            })

        # Fact tools (always available)
        self._handlers.update({
            "resolve_fact": self.fact_tools.resolve_fact,
            "add_fact": self.fact_tools.add_fact,
            "extract_facts_from_text": self.fact_tools.extract_facts_from_text,
            "list_known_facts": self.fact_tools.list_known_facts,
            "get_unresolved_facts": self.fact_tools.get_unresolved_facts,
        })

    def get_tool_schemas(self, include_disabled: bool = False) -> list[dict]:
        """
        Get tool schemas for LLM integration.

        Args:
            include_disabled: Include schemas for tools that aren't available

        Returns:
            List of tool schemas in Anthropic format
        """
        if include_disabled:
            return DISCOVERY_TOOL_SCHEMAS.copy()

        # Filter to only available tools
        available = []
        for schema in DISCOVERY_TOOL_SCHEMAS:
            if schema["name"] in self._handlers:
                available.append(schema)

        return available

    def execute(self, tool_name: str, args: dict) -> Any:
        """
        Execute a discovery tool by name.

        Args:
            tool_name: Name of the tool to execute
            args: Tool arguments as a dict

        Returns:
            Tool result (typically a dict or list)

        Raises:
            ValueError: If tool not found
        """
        handler = self._handlers.get(tool_name)
        if not handler:
            raise ValueError(f"Unknown discovery tool: {tool_name}")

        return handler(**args)

    def is_available(self, tool_name: str) -> bool:
        """Check if a tool is available."""
        return tool_name in self._handlers

    def list_available_tools(self) -> list[str]:
        """List all available tool names."""
        return list(self._handlers.keys())


# Minimal system prompt for discovery-based planning
DISCOVERY_SYSTEM_PROMPT = """You are a data analysis assistant with access to databases, APIs, and reference documents.

IMPORTANT: You do NOT have schema information loaded upfront. Use discovery tools to find what you need.

## Discovery Tools

### Schema Discovery (Databases)
- list_databases() - See available databases
- list_tables(database) - See tables in a database
- get_table_schema(database, table) - Get column details
- search_tables(query) - Find tables by description
- get_table_relationships(database, table) - See foreign keys
- get_sample_values(database, table, column) - See example values

### API Discovery
- list_apis() - See available APIs
- list_api_operations(api) - See operations in an API
- get_operation_details(operation) - Get operation schema
- search_operations(query) - Find operations by description

### Document Discovery
- list_documents() - See reference documents
- get_document(name) - Read a document
- search_documents(query) - Search document content
- get_document_section(name, section) - Get specific section

### Fact Resolution
- resolve_fact(question) - Resolve facts from all sources
- add_fact(name, value) - Add user-provided facts
- list_known_facts() - See cached facts
- get_unresolved_facts() - See what couldn't be resolved

## Planning Process

1. UNDERSTAND the user's question
2. DISCOVER relevant resources using tools
3. CLARIFY unclear terms with resolve_fact()
4. PLAN the analysis steps
5. OUTPUT a structured plan

Always verify resources exist before planning to use them.
"""
