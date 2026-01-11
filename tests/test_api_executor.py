"""Tests for API executor - actually executing queries against external APIs."""

import pytest

from constat.core.config import Config, APIConfig
from constat.catalog.api_executor import APIExecutor, APIExecutionError


class TestAPIExecutorGraphQL:
    """Tests for GraphQL execution against real APIs."""

    @pytest.fixture
    def config_with_countries(self):
        """Config with Countries GraphQL API."""
        return Config(
            apis={
                "countries": APIConfig(
                    type="graphql",
                    url="https://countries.trevorblades.com/graphql",
                    description="Countries GraphQL API",
                ),
            }
        )

    @pytest.fixture
    def config_with_rick_and_morty(self):
        """Config with Rick and Morty GraphQL API."""
        return Config(
            apis={
                "rickandmorty": APIConfig(
                    type="graphql",
                    url="https://rickandmortyapi.com/graphql",
                    description="Rick and Morty GraphQL API",
                ),
            }
        )

    def test_execute_countries_query(self, config_with_countries):
        """Test executing a simple GraphQL query against Countries API."""
        with APIExecutor(config_with_countries) as executor:
            result = executor.execute_graphql(
                "countries",
                "{ countries { name code } }"
            )

        assert "countries" in result
        assert len(result["countries"]) > 0
        # Check structure
        first_country = result["countries"][0]
        assert "name" in first_country
        assert "code" in first_country

    def test_execute_countries_with_filter(self, config_with_countries):
        """Test GraphQL query with variables."""
        with APIExecutor(config_with_countries) as executor:
            result = executor.execute_graphql(
                "countries",
                """
                query GetCountry($code: ID!) {
                    country(code: $code) {
                        name
                        capital
                        currency
                    }
                }
                """,
                variables={"code": "US"}
            )

        assert "country" in result
        assert result["country"]["name"] == "United States"

    def test_execute_continents_query(self, config_with_countries):
        """Test querying continents."""
        with APIExecutor(config_with_countries) as executor:
            result = executor.execute_graphql(
                "countries",
                "{ continents { name code } }"
            )

        assert "continents" in result
        continent_names = [c["name"] for c in result["continents"]]
        assert "Europe" in continent_names
        assert "Asia" in continent_names

    def test_execute_rick_and_morty_characters(self, config_with_rick_and_morty):
        """Test querying Rick and Morty API."""
        with APIExecutor(config_with_rick_and_morty) as executor:
            result = executor.execute_graphql(
                "rickandmorty",
                """
                {
                    characters(page: 1, filter: { name: "Rick" }) {
                        results {
                            name
                            status
                            species
                        }
                    }
                }
                """
            )

        assert "characters" in result
        assert "results" in result["characters"]
        # Should find Rick characters
        names = [c["name"] for c in result["characters"]["results"]]
        assert any("Rick" in name for name in names)

    def test_api_not_found(self, config_with_countries):
        """Test error when API doesn't exist."""
        with APIExecutor(config_with_countries) as executor:
            with pytest.raises(APIExecutionError, match="not found"):
                executor.execute_graphql("nonexistent", "{ test }")

    def test_invalid_query(self, config_with_countries):
        """Test error on invalid GraphQL query."""
        with APIExecutor(config_with_countries) as executor:
            with pytest.raises(APIExecutionError, match="GraphQL errors"):
                executor.execute_graphql(
                    "countries",
                    "{ invalidField }"
                )

    def test_generic_execute_method(self, config_with_countries):
        """Test the generic execute() method that auto-detects API type."""
        with APIExecutor(config_with_countries) as executor:
            result = executor.execute(
                "countries",
                "{ countries { name } }"
            )

        assert "countries" in result
        assert len(result["countries"]) > 0


class TestAPIExecutorREST:
    """Tests for REST API execution."""

    @pytest.fixture
    def config_with_jsonplaceholder(self):
        """Config with JSONPlaceholder REST API."""
        return Config(
            apis={
                "jsonplaceholder": APIConfig(
                    type="openapi",
                    url="https://jsonplaceholder.typicode.com",
                    description="JSONPlaceholder fake REST API",
                ),
            }
        )

    def test_execute_rest_get(self, config_with_jsonplaceholder):
        """Test executing a REST GET request."""
        with APIExecutor(config_with_jsonplaceholder) as executor:
            result = executor.execute_rest(
                "jsonplaceholder",
                "/posts/1"
            )

        assert result["id"] == 1
        assert "title" in result
        assert "body" in result

    def test_execute_rest_get_list(self, config_with_jsonplaceholder):
        """Test getting a list of resources."""
        with APIExecutor(config_with_jsonplaceholder) as executor:
            result = executor.execute_rest(
                "jsonplaceholder",
                "/posts",
                query_params={"_limit": 5}
            )

        assert isinstance(result, list)
        assert len(result) == 5

    def test_execute_rest_with_path_params(self, config_with_jsonplaceholder):
        """Test REST call with path parameters."""
        with APIExecutor(config_with_jsonplaceholder) as executor:
            result = executor.execute_rest(
                "jsonplaceholder",
                "/users/{userId}/posts",
                path_params={"userId": 1}
            )

        assert isinstance(result, list)
        # All posts should be from user 1
        for post in result:
            assert post["userId"] == 1

    def test_execute_rest_post(self, config_with_jsonplaceholder):
        """Test executing a REST POST request."""
        with APIExecutor(config_with_jsonplaceholder) as executor:
            result = executor.execute_rest(
                "jsonplaceholder",
                "/posts",
                method="POST",
                body={
                    "title": "Test Post",
                    "body": "This is a test",
                    "userId": 1,
                }
            )

        # JSONPlaceholder returns the created resource with an ID
        assert "id" in result
        assert result["title"] == "Test Post"


class TestAPIExecutorAuth:
    """Tests for API authentication."""

    def test_bearer_auth_headers(self):
        """Test that bearer auth is added to headers."""
        config = Config(
            apis={
                "test_api": APIConfig(
                    type="graphql",
                    url="https://example.com/graphql",
                    auth_type="bearer",
                    auth_token="test_token_123",
                ),
            }
        )
        executor = APIExecutor(config)
        api_config = executor._get_api_config("test_api")
        headers = executor._build_headers(api_config)

        assert headers["Authorization"] == "Bearer test_token_123"

    def test_api_key_headers(self):
        """Test that API key is added to headers."""
        config = Config(
            apis={
                "test_api": APIConfig(
                    type="graphql",
                    url="https://example.com/graphql",
                    auth_type="api_key",
                    api_key="my_api_key",
                    api_key_header="X-Custom-Key",
                ),
            }
        )
        executor = APIExecutor(config)
        api_config = executor._get_api_config("test_api")
        headers = executor._build_headers(api_config)

        assert headers["X-Custom-Key"] == "my_api_key"

    def test_custom_headers(self):
        """Test that custom headers are included."""
        config = Config(
            apis={
                "test_api": APIConfig(
                    type="graphql",
                    url="https://example.com/graphql",
                    headers={"X-Custom": "value", "X-Another": "value2"},
                ),
            }
        )
        executor = APIExecutor(config)
        api_config = executor._get_api_config("test_api")
        headers = executor._build_headers(api_config)

        assert headers["X-Custom"] == "value"
        assert headers["X-Another"] == "value2"


class TestAPIExecutorIntegration:
    """Integration tests combining discovery and execution."""

    def test_discover_and_execute(self):
        """Test discovering APIs and then executing queries."""
        from constat.catalog.api_catalog import introspect_graphql_endpoint
        from constat.discovery.api_tools import APIDiscoveryTools
        from constat.catalog.api_catalog import APICatalog

        config = Config(
            apis={
                "countries": APIConfig(
                    type="graphql",
                    url="https://countries.trevorblades.com/graphql",
                ),
            }
        )

        # Introspect to discover operations
        catalog = introspect_graphql_endpoint(
            "https://countries.trevorblades.com/graphql"
        )

        # Create discovery tools with the catalog and config
        tools = APIDiscoveryTools(catalog, config)

        # List available APIs
        apis = tools.list_apis()
        assert len(apis) == 1
        assert apis[0]["name"] == "countries"

        # Search for relevant operations
        results = tools.search_operations("list all countries")
        assert len(results) > 0

        # Execute a query
        data = tools.execute_graphql(
            "countries",
            "{ countries { name code } }"
        )
        assert "countries" in data
        assert len(data["countries"]) > 0
