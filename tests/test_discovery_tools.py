"""Tests for the discovery tools module."""

import pytest
from pathlib import Path

from constat.core.config import Config
from constat.catalog.schema_manager import SchemaManager
from constat.catalog.api_catalog import APICatalog
from constat.discovery import (
    DiscoveryTools,
    PromptBuilder,
    SchemaDiscoveryTools,
    APIDiscoveryTools,
    DocumentDiscoveryTools,
    DISCOVERY_TOOL_SCHEMAS,
    SCHEMA_TOOL_SCHEMAS,
    API_TOOL_SCHEMAS,
    DOC_TOOL_SCHEMAS,
)


FIXTURES_DIR = Path(__file__).parent.parent
CHINOOK_DB = FIXTURES_DIR / "data" / "chinook.db"


@pytest.fixture(scope="module")
def config() -> Config:
    """Create config with test database."""
    return Config(
        databases={
            "chinook": {"uri": f"sqlite:///{CHINOOK_DB}"},
        },
        documents={
            "test_doc": {
                "type": "inline",
                "content": """# Business Rules

## VIP Customers
A VIP customer is defined as one with lifetime value > $100,000.

## Revenue Thresholds
- Low: < $10,000
- Medium: $10,000 - $50,000
- High: > $50,000
""",
                "description": "Test business rules document",
                "tags": ["rules", "definitions"],
            }
        },
        # Use in-memory vector store for tests to avoid stale data
        storage={"vector_store": {"backend": "numpy"}},
    )


@pytest.fixture(scope="module")
def schema_manager(config) -> SchemaManager:
    """Create and initialize schema manager."""
    sm = SchemaManager(config)
    sm.initialize()
    return sm


class TestSchemaDiscoveryTools:
    """Test schema discovery tools."""

    def test_list_databases(self, schema_manager):
        """Test listing databases."""
        tools = SchemaDiscoveryTools(schema_manager)
        result = tools.list_databases()

        assert len(result) == 1
        assert result[0]["name"] == "chinook"
        assert result[0]["type"] == "sql"
        assert result[0]["table_count"] > 0

    def test_list_tables(self, schema_manager):
        """Test listing tables in a database."""
        tools = SchemaDiscoveryTools(schema_manager)
        result = tools.list_tables("chinook")

        assert len(result) > 0
        # Chinook has tables like albums, artists, tracks
        table_names = [t["name"] for t in result]
        assert "albums" in table_names or "Album" in table_names
        assert "artists" in table_names or "Artist" in table_names

        # Check structure
        for table in result:
            assert "name" in table
            assert "row_count" in table
            assert "column_count" in table

    def test_get_table_schema(self, schema_manager):
        """Test getting detailed table schema."""
        tools = SchemaDiscoveryTools(schema_manager)

        # Find the artists table (case may vary)
        tables = tools.list_tables("chinook")
        artists_table = next(
            (t["name"] for t in tables if "artist" in t["name"].lower()),
            None
        )
        assert artists_table is not None

        result = tools.get_table_schema("chinook", artists_table)

        assert "columns" in result
        assert "primary_keys" in result
        assert len(result["columns"]) > 0

        # Should have at least an ID and Name column
        col_names = [c["name"].lower() for c in result["columns"]]
        assert any("id" in name for name in col_names)
        assert any("name" in name for name in col_names)

    def test_search_tables(self, schema_manager):
        """Test semantic search for tables."""
        tools = SchemaDiscoveryTools(schema_manager)
        result = tools.search_tables("music genres and tracks", limit=5)

        assert len(result) > 0
        assert "relevance" in result[0]
        assert result[0]["relevance"] > 0

        # Should find genre-related tables
        table_names = [r["table"].lower() for r in result]
        has_relevant = any(
            "genre" in name or "track" in name
            for name in table_names
        )
        assert has_relevant, f"Expected genre/track tables, got: {table_names}"

    def test_get_table_relationships(self, schema_manager):
        """Test getting table relationships."""
        tools = SchemaDiscoveryTools(schema_manager)

        # Find a table with foreign keys
        tables = tools.list_tables("chinook")
        fk_table = next(
            (t["name"] for t in tables if t.get("has_foreign_keys")),
            None
        )

        if fk_table:
            result = tools.get_table_relationships("chinook", fk_table)
            assert "references" in result or "referenced_by" in result


class TestDocumentDiscoveryTools:
    """Test document discovery tools."""

    def test_list_documents(self, config):
        """Test listing documents."""
        tools = DocumentDiscoveryTools(config)
        result = tools.list_documents()

        assert len(result) == 1
        assert result[0]["name"] == "test_doc"
        assert result[0]["type"] == "inline"
        assert "rules" in result[0]["tags"]

    def test_get_document(self, config):
        """Test getting document content."""
        tools = DocumentDiscoveryTools(config)
        result = tools.get_document("test_doc")

        assert "content" in result
        assert "VIP" in result["content"]
        assert "Business Rules" in result["content"]

    def test_get_document_not_found(self, config):
        """Test getting non-existent document."""
        tools = DocumentDiscoveryTools(config)
        result = tools.get_document("nonexistent")

        assert "error" in result

    def test_search_documents(self, config):
        """Test searching documents."""
        tools = DocumentDiscoveryTools(config)
        result = tools.search_documents("VIP customer definition", limit=3)

        assert len(result) > 0
        assert "excerpt" in result[0]
        assert "relevance" in result[0]
        # Should find the VIP section
        assert "VIP" in result[0]["excerpt"] or "100,000" in result[0]["excerpt"]

    def test_get_document_section(self, config):
        """Test getting specific section."""
        tools = DocumentDiscoveryTools(config)
        result = tools.get_document_section("test_doc", "VIP Customers")

        assert "content" in result
        assert "VIP" in result["content"]
        assert "100,000" in result["content"]


class TestDiscoveryTools:
    """Test unified discovery tools interface."""

    def test_initialization(self, schema_manager, config):
        """Test creating DiscoveryTools."""
        tools = DiscoveryTools(
            schema_manager=schema_manager,
            config=config,
        )

        # Should have handlers registered
        assert len(tools.list_available_tools()) > 0

    def test_get_tool_schemas(self, schema_manager, config):
        """Test getting tool schemas."""
        tools = DiscoveryTools(
            schema_manager=schema_manager,
            config=config,
        )

        schemas = tools.get_tool_schemas()
        assert len(schemas) > 0

        # Each schema should have required fields
        for schema in schemas:
            assert "name" in schema
            assert "description" in schema
            assert "input_schema" in schema

    def test_execute_tool(self, schema_manager, config):
        """Test executing tools through unified interface."""
        tools = DiscoveryTools(
            schema_manager=schema_manager,
            config=config,
        )

        # Execute list_databases
        result = tools.execute("list_databases", {})
        assert len(result) > 0
        assert result[0]["name"] == "chinook"

        # Execute search_tables
        result = tools.execute("search_tables", {"query": "music artists"})
        assert len(result) > 0

    def test_execute_unknown_tool(self, schema_manager, config):
        """Test executing unknown tool raises error."""
        tools = DiscoveryTools(
            schema_manager=schema_manager,
            config=config,
        )

        with pytest.raises(ValueError, match="Unknown discovery tool"):
            tools.execute("nonexistent_tool", {})

    def test_is_available(self, schema_manager, config):
        """Test checking tool availability."""
        tools = DiscoveryTools(
            schema_manager=schema_manager,
            config=config,
        )

        assert tools.is_available("list_databases")
        assert tools.is_available("search_tables")
        assert not tools.is_available("nonexistent_tool")


class TestToolSchemas:
    """Test tool schema definitions."""

    def test_schema_tool_schemas_valid(self):
        """Test schema tool schemas are valid."""
        for schema in SCHEMA_TOOL_SCHEMAS:
            assert "name" in schema
            assert "description" in schema
            assert "input_schema" in schema
            assert schema["input_schema"]["type"] == "object"

    def test_api_tool_schemas_valid(self):
        """Test API tool schemas are valid."""
        for schema in API_TOOL_SCHEMAS:
            assert "name" in schema
            assert "description" in schema
            assert "input_schema" in schema

    def test_doc_tool_schemas_valid(self):
        """Test document tool schemas are valid."""
        for schema in DOC_TOOL_SCHEMAS:
            assert "name" in schema
            assert "description" in schema
            assert "input_schema" in schema

    def test_all_schemas_combined(self):
        """Test all schemas are in combined list."""
        all_names = [s["name"] for s in DISCOVERY_TOOL_SCHEMAS]

        # Should have schema tools
        assert "list_databases" in all_names
        assert "search_tables" in all_names

        # Should have API tools
        assert "list_apis" in all_names
        assert "search_operations" in all_names

        # Should have doc tools
        assert "list_documents" in all_names
        assert "search_documents" in all_names

        # Should have fact tools
        assert "resolve_fact" in all_names


class TestAPIDiscoveryTools:
    """Test API discovery tools."""

    def test_list_apis_empty(self, config):
        """Test listing APIs when none configured."""
        catalog = APICatalog()
        tools = APIDiscoveryTools(catalog, config)
        result = tools.list_apis()

        # Config has no APIs, so should be empty or inferred from catalog
        assert isinstance(result, list)

    def test_search_operations_empty(self, config):
        """Test searching operations in empty catalog."""
        catalog = APICatalog()
        tools = APIDiscoveryTools(catalog, config)
        result = tools.search_operations("get user")

        assert isinstance(result, list)
        assert len(result) == 0


class TestPromptBuilder:
    """Test automatic prompt building based on model capabilities."""

    def test_supports_tools_claude_3(self, schema_manager, config):
        """Claude 3 models should support tools."""
        tools = DiscoveryTools(schema_manager=schema_manager, config=config)
        builder = PromptBuilder(tools)

        assert builder.supports_tools("claude-3-opus-20240229")
        assert builder.supports_tools("claude-3-sonnet-20240229")
        assert builder.supports_tools("claude-3-haiku-20240307")
        assert builder.supports_tools("claude-sonnet-4-20250514")

    def test_supports_tools_gpt4(self, schema_manager, config):
        """GPT-4 models should support tools."""
        tools = DiscoveryTools(schema_manager=schema_manager, config=config)
        builder = PromptBuilder(tools)

        assert builder.supports_tools("gpt-4")
        assert builder.supports_tools("gpt-4-turbo")
        assert builder.supports_tools("gpt-3.5-turbo")

    def test_no_tools_legacy_models(self, schema_manager, config):
        """Legacy models should not support tools."""
        tools = DiscoveryTools(schema_manager=schema_manager, config=config)
        builder = PromptBuilder(tools)

        assert not builder.supports_tools("claude-2")
        assert not builder.supports_tools("claude-instant-1.2")
        assert not builder.supports_tools("text-davinci-003")
        assert not builder.supports_tools("gpt-3.5-turbo-instruct")

    def test_build_prompt_tool_mode(self, schema_manager, config):
        """Tool-capable model should get minimal prompt."""
        tools = DiscoveryTools(schema_manager=schema_manager, config=config)
        builder = PromptBuilder(tools)

        prompt, use_tools = builder.build_prompt("claude-3-opus-20240229")

        assert use_tools is True
        assert "Discovery Tools" in prompt
        assert "list_databases" in prompt
        # Should NOT contain full schema
        assert "Columns:" not in prompt

    def test_build_prompt_full_mode(self, schema_manager, config):
        """Non-tool model should get comprehensive prompt."""
        tools = DiscoveryTools(schema_manager=schema_manager, config=config)
        builder = PromptBuilder(tools)

        prompt, use_tools = builder.build_prompt("claude-2")

        assert use_tools is False
        # Should contain full schema documentation
        assert "Available Databases" in prompt
        assert "chinook" in prompt
        # Should have column details
        assert "Columns:" in prompt

    def test_build_prompt_includes_documents(self, schema_manager, config):
        """Full mode should include document content."""
        tools = DiscoveryTools(schema_manager=schema_manager, config=config)
        builder = PromptBuilder(tools)

        prompt, use_tools = builder.build_prompt("claude-2")

        assert use_tools is False
        # Should include document content
        assert "Reference Documents" in prompt
        assert "VIP" in prompt or "Business Rules" in prompt

    def test_estimate_tokens(self, schema_manager, config):
        """Token estimation should show savings for tool mode."""
        tools = DiscoveryTools(schema_manager=schema_manager, config=config)
        builder = PromptBuilder(tools)

        # Tool-capable model
        estimate = builder.estimate_tokens("claude-3-opus-20240229")
        assert estimate["supports_tools"] is True
        assert estimate["mode"] == "tool_discovery"
        assert estimate["savings_percent"] > 0
        assert estimate["prompt_tokens"] < estimate["full_prompt_tokens"]

        # Non-tool model
        estimate = builder.estimate_tokens("claude-2")
        assert estimate["supports_tools"] is False
        assert estimate["mode"] == "full_prompt"
        assert estimate["savings_percent"] == 0
