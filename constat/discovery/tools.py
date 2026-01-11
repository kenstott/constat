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
                "execute_graphql": self.api_tools.execute_graphql,
                "execute_rest": self.api_tools.execute_rest,
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


# Minimal system prompt for discovery-based planning (tool calling mode)
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


class PromptBuilder:
    """
    Builds system prompts based on LLM tool calling capability.

    Automatically detects whether tool calling is supported:
    - If supported → minimal prompt with discovery tools
    - If not supported → comprehensive prompt with full metadata

    This is NOT configurable - it's automatic based on the LLM model.
    """

    # Models known to support tool calling
    TOOL_CAPABLE_MODELS = {
        # Anthropic Claude models with tool support
        "claude-3", "claude-sonnet", "claude-opus", "claude-haiku",
        # OpenAI models with function calling
        "gpt-4", "gpt-3.5-turbo",
        # Google models
        "gemini",
    }

    # Models/providers known to NOT support tool calling
    NO_TOOL_MODELS = {
        "claude-2", "claude-instant",
        "gpt-3.5-turbo-instruct",
        "text-davinci", "text-curie", "text-babbage", "text-ada",
    }

    def __init__(self, discovery_tools: "DiscoveryTools"):
        self.tools = discovery_tools

    def supports_tools(self, model: str) -> bool:
        """
        Check if a model supports tool calling.

        Args:
            model: Model name/ID

        Returns:
            True if model supports tool calling
        """
        model_lower = model.lower()

        # Check against known non-tool models first
        for pattern in self.NO_TOOL_MODELS:
            if pattern in model_lower:
                return False

        # Check against known tool-capable models
        for pattern in self.TOOL_CAPABLE_MODELS:
            if pattern in model_lower:
                return True

        # Default: assume modern models support tools
        # This is a reasonable default as tool calling is becoming standard
        return True

    def build_prompt(
        self,
        model: str,
        custom_prompt: str = "",
    ) -> tuple[str, bool]:
        """
        Build appropriate system prompt based on model capabilities.

        Automatically determines whether to use tool-based or full-prompt mode.

        Args:
            model: Model name/ID being used
            custom_prompt: Additional custom prompt content

        Returns:
            Tuple of (system_prompt, use_tools) where use_tools indicates
            whether discovery tools should be included in the API call
        """
        use_tools = self.supports_tools(model)

        if use_tools:
            prompt = self._build_tool_prompt(custom_prompt)
        else:
            prompt = self._build_full_prompt(
                include_schema=True,
                include_apis=True,
                include_documents=True,
                custom_prompt=custom_prompt,
            )

        return prompt, use_tools

    def _build_tool_prompt(self, custom_prompt: str) -> str:
        """Build minimal prompt for tool-based discovery."""
        parts = [DISCOVERY_SYSTEM_PROMPT]
        if custom_prompt:
            parts.append(f"\n## Additional Context\n{custom_prompt}")
        return "\n".join(parts)

    def _build_full_prompt(
        self,
        include_schema: bool,
        include_apis: bool,
        include_documents: bool,
        custom_prompt: str,
    ) -> str:
        """Build comprehensive prompt with full metadata (no tool calling)."""
        parts = [
            "You are a data analysis assistant with access to databases, APIs, and reference documents.",
            "",
        ]

        # Add schema overview
        if include_schema and self.tools.schema_tools:
            schema_section = self._build_schema_section()
            if schema_section:
                parts.append(schema_section)

        # Add API documentation
        if include_apis and self.tools.api_tools:
            api_section = self._build_api_section()
            if api_section:
                parts.append(api_section)

        # Add document content
        if include_documents and self.tools.doc_tools:
            doc_section = self._build_document_section()
            if doc_section:
                parts.append(doc_section)

        # Add custom prompt
        if custom_prompt:
            parts.append(f"## Additional Context\n{custom_prompt}")

        return "\n".join(parts)

    def _build_schema_section(self) -> str:
        """Build comprehensive schema documentation."""
        lines = ["## Available Databases\n"]

        databases = self.tools.schema_tools.list_databases()

        for db in databases:
            lines.append(f"### {db['name']}")
            if db.get('description'):
                lines.append(f"{db['description']}\n")

            # Get tables for this database
            tables = self.tools.schema_tools.list_tables(db['name'])

            for table in tables:
                lines.append(f"#### {table['name']}")
                if table.get('description'):
                    lines.append(f"{table['description']}")
                lines.append(f"Rows: ~{table['row_count']:,}")

                # Get full schema
                schema = self.tools.schema_tools.get_table_schema(db['name'], table['name'])
                if 'columns' in schema:
                    lines.append("Columns:")
                    for col in schema['columns']:
                        col_line = f"  - {col['name']}: {col['type']}"
                        if col.get('primary_key'):
                            col_line += " (PK)"
                        if col.get('comment'):
                            col_line += f" - {col['comment']}"
                        lines.append(col_line)

                # Add foreign keys
                if schema.get('foreign_keys'):
                    lines.append("Foreign Keys:")
                    for fk in schema['foreign_keys']:
                        lines.append(f"  - {fk['from']} → {fk['to']}")

                lines.append("")

        return "\n".join(lines)

    def _build_api_section(self) -> str:
        """Build comprehensive API documentation."""
        lines = ["## Available APIs\n"]

        apis = self.tools.api_tools.list_apis()
        if not apis:
            return ""

        for api in apis:
            lines.append(f"### {api['name']} ({api['type']})")
            if api.get('description'):
                lines.append(f"{api['description']}")
            if api.get('url'):
                lines.append(f"URL: {api['url']}")
            lines.append("")

            # Get operations
            operations = self.tools.api_tools.list_operations(api['name'])

            for op in operations:
                lines.append(f"#### {op['name']} ({op['type']})")
                if op.get('description'):
                    lines.append(f"{op['description']}")

                # Get full details
                details = self.tools.api_tools.get_operation_details(op['name'])
                if 'arguments' in details and details['arguments']:
                    lines.append("Arguments:")
                    for arg in details['arguments']:
                        arg_line = f"  - {arg['name']}: {arg['type']}"
                        if arg.get('required'):
                            arg_line += " (required)"
                        if arg.get('description'):
                            arg_line += f" - {arg['description']}"
                        lines.append(arg_line)

                if details.get('return_type'):
                    lines.append(f"Returns: {details['return_type']}")

                lines.append("")

        return "\n".join(lines)

    def _build_document_section(self) -> str:
        """Build reference document content."""
        lines = ["## Reference Documents\n"]

        documents = self.tools.doc_tools.list_documents()
        if not documents:
            return ""

        for doc in documents:
            lines.append(f"### {doc['name']}")
            if doc.get('description'):
                lines.append(f"{doc['description']}")
            if doc.get('tags'):
                lines.append(f"Tags: {', '.join(doc['tags'])}")
            lines.append("")

            # Get full content
            content = self.tools.doc_tools.get_document(doc['name'])
            if 'content' in content:
                # Indent document content
                doc_content = content['content']
                # Limit very long documents
                if len(doc_content) > 5000:
                    doc_content = doc_content[:5000] + "\n\n[Document truncated...]"
                lines.append(doc_content)

            lines.append("")

        return "\n".join(lines)

    def estimate_tokens(self, model: str) -> dict:
        """
        Estimate token count for both prompt modes.

        Rough estimate: ~4 characters per token for English text.

        Args:
            model: Model name/ID

        Returns:
            Dict with token estimates and which mode will be used
        """
        use_tools = self.supports_tools(model)
        prompt, _ = self.build_prompt(model)

        # Also calculate what full prompt would be for comparison
        full_prompt = self._build_full_prompt(True, True, True, "")

        return {
            "model": model,
            "supports_tools": use_tools,
            "mode": "tool_discovery" if use_tools else "full_prompt",
            "prompt_tokens": len(prompt) // 4,
            "full_prompt_tokens": len(full_prompt) // 4,
            "savings_percent": round(
                (1 - len(prompt) / max(len(full_prompt), 1)) * 100
            ) if use_tools else 0,
        }
