# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tests for API Catalog with vector search."""

import pytest

from constat.catalog.api_catalog import (
    APICatalog,
    OperationType,
    OperationMetadata,
    OperationArgument,
    ArgumentType,
    introspect_graphql_endpoint,
    create_constat_api_catalog,
)


class TestAPICatalog:
    """Tests for APICatalog core functionality."""

    @pytest.fixture
    def catalog(self):
        """Create a catalog with sample operations."""
        catalog = APICatalog()
        catalog.register_operations([
            OperationMetadata(
                name="getUser",
                operation_type=OperationType.QUERY,
                description="Get a user by their ID",
                arguments=[
                    OperationArgument("id", "ID!", "User ID"),
                ],
                return_type="User",
                use_cases=["Fetch user profile", "Get user details"],
                tags=["user", "read"],
            ),
            OperationMetadata(
                name="listUsers",
                operation_type=OperationType.QUERY,
                description="List all users with optional filtering",
                arguments=[
                    OperationArgument("filter", "UserFilter", requirement=ArgumentType.OPTIONAL),
                    OperationArgument("limit", "Int", default_value=10, requirement=ArgumentType.OPTIONAL),
                ],
                return_type="[User!]!",
                use_cases=["Browse users", "Search for users"],
                tags=["user", "list", "read"],
            ),
            OperationMetadata(
                name="createUser",
                operation_type=OperationType.MUTATION,
                description="Create a new user account",
                arguments=[
                    OperationArgument("input", "CreateUserInput!", "User data"),
                ],
                return_type="User!",
                use_cases=["Register new user", "Create account"],
                tags=["user", "create"],
            ),
            OperationMetadata(
                name="userCreated",
                operation_type=OperationType.SUBSCRIPTION,
                description="Subscribe to new user creation events",
                arguments=[],
                return_type="User!",
                use_cases=["Real-time user notifications"],
                tags=["user", "realtime"],
            ),
        ])
        catalog.build_index()
        return catalog

    def test_register_operations(self, catalog):
        """Test registering operations."""
        assert len(catalog.operations) == 4
        assert "query.getUser" in catalog.operations
        assert "mutation.createUser" in catalog.operations
        assert "subscription.userCreated" in catalog.operations

    def test_get_operation_by_name(self, catalog):
        """Test retrieving operation by name."""
        op = catalog.get_operation("getUser")
        assert op is not None
        assert op.name == "getUser"
        assert op.operation_type == OperationType.QUERY

        # Also works with full name
        op2 = catalog.get_operation("query.getUser")
        assert op2 is not None
        assert op2.name == "getUser"

    def test_get_operation_not_found(self, catalog):
        """Test that non-existent operation returns None."""
        assert catalog.get_operation("nonexistent") is None

    def test_find_relevant_operations(self, catalog):
        """Test semantic search for operations."""
        results = catalog.find_relevant_operations("get user information")

        assert len(results) > 0
        # getUser should be highly relevant
        op_names = [r.operation for r in results]
        assert "getUser" in op_names

    def test_find_relevant_operations_with_type_filter(self, catalog):
        """Test filtering by operation type."""
        mutations = catalog.find_relevant_operations(
            "user operations",
            operation_type=OperationType.MUTATION
        )

        assert len(mutations) > 0
        for result in mutations:
            assert result.operation_type == OperationType.MUTATION

    def test_find_relevant_operations_with_tags(self, catalog):
        """Test filtering by tags."""
        realtime_ops = catalog.find_relevant_operations(
            "user events",
            tags=["realtime"]
        )

        assert len(realtime_ops) > 0
        assert realtime_ops[0].operation == "userCreated"

    def test_list_operations(self, catalog):
        """Test listing all operations."""
        all_ops = catalog.list_operations()
        assert len(all_ops) == 4

        queries = catalog.list_operations(operation_type=OperationType.QUERY)
        assert len(queries) == 2

    def test_get_overview(self, catalog):
        """Test generating token-optimized overview."""
        overview = catalog.get_overview()

        assert "Available API Operations" in overview
        assert "Queries" in overview
        assert "Mutations" in overview
        assert "Subscriptions" in overview
        assert "getUser" in overview
        assert "createUser" in overview

    def test_operation_to_dict(self, catalog):
        """Test operation serialization."""
        op = catalog.get_operation("getUser")
        as_dict = op.to_dict()

        assert as_dict["name"] == "getUser"
        assert as_dict["type"] == "query"
        assert len(as_dict["arguments"]) == 1
        assert as_dict["arguments"][0]["name"] == "id"

    def test_operation_to_embedding_text(self, catalog):
        """Test generating embedding text."""
        op = catalog.get_operation("getUser")
        text = op.to_embedding_text()

        assert "getUser" in text
        assert "Get a user by their ID" in text
        assert "Fetch user profile" in text
        assert "id" in text

    def test_get_tools(self, catalog):
        """Test getting LLM tool definitions."""
        tools = catalog.get_tools()

        assert len(tools) == 2
        tool_names = [t["name"] for t in tools]
        assert "find_api_operations" in tool_names
        assert "get_api_operation" in tool_names

    def test_tool_handlers(self, catalog):
        """Test tool handler functions."""
        handlers = catalog.get_tool_handlers()

        # Test find_api_operations
        results = handlers["find_api_operations"]("get user", top_k=2)
        assert len(results) > 0

        # Test get_api_operation
        op = handlers["get_api_operation"]("getUser")
        assert op["name"] == "getUser"

        # Test not found
        error = handlers["get_api_operation"]("nonexistent")
        assert "error" in error


class TestConstatAPICatalog:
    """Tests for the pre-built Constat API catalog."""

    @pytest.fixture
    def catalog(self):
        """Create the Constat API catalog."""
        return create_constat_api_catalog()

    def test_has_session_operations(self, catalog):
        """Test that session operations are registered."""
        session_op = catalog.get_operation("session")
        assert session_op is not None
        assert session_op.operation_type == OperationType.QUERY

        create_session = catalog.get_operation("createSession")
        assert create_session is not None
        assert create_session.operation_type == OperationType.MUTATION

    def test_has_artifact_operations(self, catalog):
        """Test that artifact operations are registered."""
        artifacts_op = catalog.get_operation("artifacts")
        assert artifacts_op is not None

        save_artifact = catalog.get_operation("saveArtifact")
        assert save_artifact is not None

    def test_find_analysis_operations(self, catalog):
        """Test finding operations for data analysis."""
        results = catalog.find_relevant_operations("start a new data analysis")

        # createSession should be relevant
        op_names = [r.operation for r in results]
        assert "createSession" in op_names

    def test_find_followup_operations(self, catalog):
        """Test finding follow-up operations."""
        results = catalog.find_relevant_operations("ask a follow-up question")

        op_names = [r.operation for r in results]
        assert "followUp" in op_names

    def test_find_realtime_operations(self, catalog):
        """Test finding realtime/subscription operations."""
        results = catalog.find_relevant_operations(
            "monitor execution progress",
            operation_type=OperationType.SUBSCRIPTION
        )

        assert len(results) > 0
        assert results[0].operation_type == OperationType.SUBSCRIPTION

    def test_overview_includes_all_types(self, catalog):
        """Test that overview includes queries, mutations, and subscriptions."""
        overview = catalog.get_overview()

        assert "Queries" in overview
        assert "Mutations" in overview
        assert "Subscriptions" in overview


@pytest.mark.integration
class TestGraphQLIntrospection:
    """
    Integration tests for GraphQL introspection.

    These tests require network access and hit public APIs.
    Run with: pytest -m integration
    """

    def test_introspect_countries_api(self):
        """Test introspecting the Countries GraphQL API."""
        catalog = introspect_graphql_endpoint(
            "https://countries.trevorblades.com/graphql"
        )

        # Should have operations
        assert len(catalog.operations) > 0

        # Check for expected operations
        countries_op = catalog.get_operation("countries")
        assert countries_op is not None
        assert countries_op.operation_type == OperationType.QUERY

        country_op = catalog.get_operation("country")
        assert country_op is not None

    def test_introspect_and_search(self):
        """Test introspecting and then searching operations."""
        catalog = introspect_graphql_endpoint(
            "https://countries.trevorblades.com/graphql"
        )

        # Search for country-related operations
        results = catalog.find_relevant_operations("find countries in a continent")

        assert len(results) > 0
        # Should find country/countries operations
        op_names = [r.operation for r in results]
        assert any("countr" in name.lower() for name in op_names)

    def test_introspect_generates_overview(self):
        """Test that introspection generates a valid overview."""
        catalog = introspect_graphql_endpoint(
            "https://countries.trevorblades.com/graphql"
        )

        overview = catalog.get_overview()

        assert "Available API Operations" in overview
        assert "Queries" in overview
        # Countries API has queries like country, countries, continent, etc.
        assert "countr" in overview.lower() or "continent" in overview.lower()

    def test_introspect_rick_and_morty(self):
        """Test introspecting the Rick and Morty API."""
        catalog = introspect_graphql_endpoint(
            "https://rickandmortyapi.com/graphql"
        )

        # Should have character operations
        characters_op = catalog.get_operation("characters")
        assert characters_op is not None

        # Search for character-related
        results = catalog.find_relevant_operations("find characters by name")
        assert len(results) > 0
