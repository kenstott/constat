from __future__ import annotations
# Copyright (c) 2025 Kenneth Stott
# Canary: a1cab75a-9db8-4ac5-807b-35ff1994dfc5
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Schema, mapping, initialization, and connection tests for ElasticsearchConnector."""

import pytest
from unittest.mock import MagicMock, patch

from constat.catalog.nosql.elasticsearch import ElasticsearchConnector
from constat.catalog.nosql.base import NoSQLType, CollectionMetadata


try:
    import elasticsearch
except ImportError:
    pytest.fail("elasticsearch is required but not installed — run: pip install elasticsearch")


# =============================================================================
# Test Initialization
# =============================================================================

class TestElasticsearchInitialization:
    """Test Elasticsearch connector initialization."""

    def test_init_with_defaults(self):
        """Test initialization with minimal parameters."""
        connector = ElasticsearchConnector()

        assert connector.hosts == ["http://localhost:9200"]
        assert connector.name == "elasticsearch"
        assert connector.cloud_id is None
        assert connector.api_key is None
        assert connector.basic_auth is None
        assert connector.use_ssl is False
        assert connector.verify_certs is True
        assert connector.sample_size == 100
        assert connector._connected is False

    def test_init_with_custom_hosts(self):
        """Test initialization with custom hosts."""
        connector = ElasticsearchConnector(
            hosts=["http://es1:9200", "http://es2:9200"],
            name="es_cluster",
        )

        assert connector.hosts == ["http://es1:9200", "http://es2:9200"]
        assert connector.name == "es_cluster"

    def test_init_with_cloud_id(self):
        """Test initialization with Elastic Cloud ID."""
        cloud_id = "deployment:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvJA..."
        connector = ElasticsearchConnector(
            cloud_id=cloud_id,
            api_key="api_key_here",
        )

        assert connector.cloud_id == cloud_id
        assert connector.api_key == "api_key_here"

    def test_init_with_basic_auth(self):
        """Test initialization with basic authentication."""
        connector = ElasticsearchConnector(
            hosts=["http://localhost:9200"],
            basic_auth=("elastic", "password"),
        )

        assert connector.basic_auth == ("elastic", "password")

    def test_init_with_ssl(self):
        """Test initialization with SSL settings."""
        connector = ElasticsearchConnector(
            hosts=["https://localhost:9200"],
            use_ssl=True,
            verify_certs=False,
        )

        assert connector.use_ssl is True
        assert connector.verify_certs is False

    def test_init_with_description(self):
        """Test initialization with description."""
        connector = ElasticsearchConnector(
            name="prod_es",
            description="Production Elasticsearch cluster",
        )

        assert connector.name == "prod_es"
        assert connector.description == "Production Elasticsearch cluster"

    def test_nosql_type(self):
        """Test that nosql_type returns SEARCH."""
        connector = ElasticsearchConnector()
        assert connector.nosql_type == NoSQLType.SEARCH


# =============================================================================
# Test Connection
# =============================================================================

class TestElasticsearchConnection:
    """Test Elasticsearch connection methods."""

    @patch("elasticsearch.Elasticsearch")
    def test_connect_simple(self, mock_es_class):
        """Test simple connection to localhost."""
        mock_client = MagicMock()
        mock_es_class.return_value = mock_client

        connector = ElasticsearchConnector()
        connector.connect()

        mock_es_class.assert_called_once_with(
            hosts=["http://localhost:9200"],
            verify_certs=True,
        )
        mock_client.info.assert_called_once()
        assert connector._connected is True

    @patch("elasticsearch.Elasticsearch")
    def test_connect_with_cloud_id(self, mock_es_class):
        """Test connection with Elastic Cloud ID."""
        mock_client = MagicMock()
        mock_es_class.return_value = mock_client

        connector = ElasticsearchConnector(
            cloud_id="test_cloud_id",
            api_key="test_api_key",
        )
        connector.connect()

        mock_es_class.assert_called_once_with(
            cloud_id="test_cloud_id",
            api_key="test_api_key",
        )
        assert connector._connected is True

    @patch("elasticsearch.Elasticsearch")
    def test_connect_with_api_key_no_cloud(self, mock_es_class):
        """Test connection with API key (no cloud ID)."""
        mock_client = MagicMock()
        mock_es_class.return_value = mock_client

        connector = ElasticsearchConnector(
            hosts=["http://localhost:9200"],
            api_key="test_api_key",
        )
        connector.connect()

        mock_es_class.assert_called_once_with(
            hosts=["http://localhost:9200"],
            api_key="test_api_key",
            verify_certs=True,
        )

    @patch("elasticsearch.Elasticsearch")
    def test_connect_with_basic_auth(self, mock_es_class):
        """Test connection with basic authentication."""
        mock_client = MagicMock()
        mock_es_class.return_value = mock_client

        connector = ElasticsearchConnector(
            hosts=["http://localhost:9200"],
            basic_auth=("elastic", "password"),
        )
        connector.connect()

        mock_es_class.assert_called_once_with(
            hosts=["http://localhost:9200"],
            basic_auth=("elastic", "password"),
            verify_certs=True,
        )

    def test_connect_raises_import_error_message(self):
        """Test that connect method has proper import error handling."""
        import inspect
        source = inspect.getsource(ElasticsearchConnector.connect)
        assert "elasticsearch" in source
        assert "pip install elasticsearch" in source


# =============================================================================
# Test Disconnect
# =============================================================================

class TestElasticsearchDisconnect:
    """Test Elasticsearch disconnection."""

    def test_disconnect_when_connected(self):
        """Test disconnecting when connected."""
        connector = ElasticsearchConnector()
        mock_client = MagicMock()
        connector._client = mock_client
        connector._connected = True

        connector.disconnect()

        mock_client.close.assert_called_once()
        assert connector._client is None
        assert connector._connected is False

    def test_disconnect_when_not_connected(self):
        """Test disconnecting when not connected."""
        connector = ElasticsearchConnector()
        connector._client = None
        connector._connected = False

        connector.disconnect()

        assert connector._connected is False


# =============================================================================
# Test Get Collection Schema
# =============================================================================

class TestElasticsearchGetCollectionSchema:
    """Test schema (mapping) retrieval."""

    def test_get_collection_schema_success(self):
        """Test getting index mapping."""
        connector = ElasticsearchConnector()
        connector._client = MagicMock()

        connector._client.indices.get_mapping.return_value = {
            "products": {
                "mappings": {
                    "properties": {
                        "name": {"type": "text"},
                        "price": {"type": "float"},
                        "category": {"type": "keyword"},
                    }
                }
            }
        }

        connector._client.count.return_value = {"count": 1000}

        connector._client.indices.get_settings.return_value = {
            "products": {
                "settings": {
                    "index": {
                        "number_of_shards": "5",
                        "number_of_replicas": "1",
                    }
                }
            }
        }

        schema = connector.get_collection_schema("products")

        assert schema.name == "products"
        assert schema.database == "elasticsearch"
        assert schema.nosql_type == NoSQLType.SEARCH
        assert schema.document_count == 1000
        assert len(schema.fields) == 3
        assert "Shards: 5" in schema.description

    def test_get_collection_schema_cached(self):
        """Test that schema is cached."""
        connector = ElasticsearchConnector()
        connector._client = MagicMock()

        cached_metadata = CollectionMetadata(
            name="products",
            database="elasticsearch",
            nosql_type=NoSQLType.SEARCH,
            fields=[],
        )
        connector._metadata_cache["products"] = cached_metadata

        schema = connector.get_collection_schema("products")

        assert schema is cached_metadata
        connector._client.indices.get_mapping.assert_not_called()

    def test_get_collection_schema_not_connected(self):
        """Test get_collection_schema raises when not connected."""
        connector = ElasticsearchConnector()
        connector._client = None

        with pytest.raises(RuntimeError, match="Not connected"):
            connector.get_collection_schema("products")

    def test_get_collection_schema_nested_fields(self):
        """Test schema with nested objects."""
        connector = ElasticsearchConnector()
        connector._client = MagicMock()

        connector._client.indices.get_mapping.return_value = {
            "orders": {
                "mappings": {
                    "properties": {
                        "customer": {
                            "properties": {
                                "name": {"type": "text"},
                                "email": {"type": "keyword"},
                            }
                        },
                        "total": {"type": "float"},
                    }
                }
            }
        }

        connector._client.count.return_value = {"count": 500}
        connector._client.indices.get_settings.return_value = {
            "orders": {"settings": {"index": {}}}
        }

        schema = connector.get_collection_schema("orders")

        field_names = [f.name for f in schema.fields]
        assert "customer.name" in field_names
        assert "customer.email" in field_names
        assert "total" in field_names

    def test_get_collection_schema_with_keyword_subfield(self):
        """Test schema detection of keyword sub-fields."""
        connector = ElasticsearchConnector()
        connector._client = MagicMock()

        connector._client.indices.get_mapping.return_value = {
            "logs": {
                "mappings": {
                    "properties": {
                        "message": {
                            "type": "text",
                            "fields": {
                                "keyword": {"type": "keyword"}
                            }
                        },
                    }
                }
            }
        }

        connector._client.count.return_value = {"count": 100}
        connector._client.indices.get_settings.return_value = {
            "logs": {"settings": {"index": {}}}
        }

        schema = connector.get_collection_schema("logs")

        message_field = next(f for f in schema.fields if f.name == "message")
        assert "keyword" in message_field.description.lower()


# =============================================================================
# Test Base Class Methods
# =============================================================================

class TestElasticsearchBaseClassMethods:
    """Test inherited base class methods."""

    def test_is_connected_property(self):
        """Test is_connected property."""
        connector = ElasticsearchConnector()

        assert connector.is_connected is False

        connector._connected = True
        assert connector.is_connected is True

    def test_sample_documents(self):
        """Test sample_documents uses query method."""
        connector = ElasticsearchConnector()
        connector._client = MagicMock()

        connector._client.search.return_value = {
            "hits": {
                "hits": [
                    {"_source": {"name": "Sample", "price": 10.0}}
                ]
            }
        }

        samples = connector.sample_documents("products", limit=5)

        assert samples == [{"name": "Sample", "price": 10.0}]
        call_kwargs = connector._client.search.call_args.kwargs
        assert call_kwargs["size"] == 5


# =============================================================================
# Test Mapping Parsing
# =============================================================================

class TestElasticsearchMappingParsing:
    """Test mapping property parsing."""

    def test_parse_simple_mapping(self):
        """Test parsing simple field mappings."""
        connector = ElasticsearchConnector()

        properties = {
            "name": {"type": "text"},
            "price": {"type": "float"},
            "active": {"type": "boolean"},
        }

        fields = connector._parse_mapping_properties(properties)

        assert len(fields) == 3
        names = {f.name: f for f in fields}
        assert names["name"].data_type == "text"
        assert names["price"].data_type == "float"
        assert names["active"].data_type == "boolean"

    def test_parse_nested_mapping(self):
        """Test parsing nested object mappings."""
        connector = ElasticsearchConnector()

        properties = {
            "user": {
                "properties": {
                    "name": {"type": "text"},
                    "email": {"type": "keyword"},
                }
            },
            "timestamp": {"type": "date"},
        }

        fields = connector._parse_mapping_properties(properties)

        names = {f.name for f in fields}
        assert "user.name" in names
        assert "user.email" in names
        assert "timestamp" in names

    def test_parse_deeply_nested_mapping(self):
        """Test parsing deeply nested mappings."""
        connector = ElasticsearchConnector()

        properties = {
            "order": {
                "properties": {
                    "customer": {
                        "properties": {
                            "address": {
                                "properties": {
                                    "city": {"type": "keyword"},
                                }
                            }
                        }
                    }
                }
            }
        }

        fields = connector._parse_mapping_properties(properties)

        names = {f.name for f in fields}
        assert "order.customer.address.city" in names

    def test_parse_mapping_with_index_setting(self):
        """Test parsing mapping with index: false."""
        connector = ElasticsearchConnector()

        properties = {
            "searchable": {"type": "text", "index": True},
            "not_searchable": {"type": "text", "index": False},
        }

        fields = connector._parse_mapping_properties(properties)

        field_dict = {f.name: f for f in fields}
        assert field_dict["searchable"].is_indexed is True
        assert field_dict["not_searchable"].is_indexed is False

    def test_parse_mapping_defaults_indexed(self):
        """Test that fields default to indexed=True."""
        connector = ElasticsearchConnector()

        properties = {
            "field_without_index": {"type": "text"},
        }

        fields = connector._parse_mapping_properties(properties)

        assert fields[0].is_indexed is True


# =============================================================================
# Test Error Handling
# =============================================================================

class TestElasticsearchErrorHandling:
    """Test error handling scenarios."""

    def test_get_collections_filters_system_indices(self):
        """Test that system indices are filtered out."""
        connector = ElasticsearchConnector()
        connector._client = MagicMock()

        connector._client.indices.get_alias.return_value = {
            "user_index": {},
            ".ds-logs": {},
            ".kibana": {},
            ".security": {},
            "products": {},
        }

        indices = connector.get_collections()

        assert "user_index" in indices
        assert "products" in indices
        assert ".ds-logs" not in indices
        assert ".kibana" not in indices
        assert ".security" not in indices

    def test_schema_handles_missing_properties(self):
        """Test schema handles mapping without properties."""
        connector = ElasticsearchConnector()
        connector._client = MagicMock()

        connector._client.indices.get_mapping.return_value = {
            "empty_index": {
                "mappings": {}
            }
        }

        connector._client.count.return_value = {"count": 0}
        connector._client.indices.get_settings.return_value = {
            "empty_index": {"settings": {"index": {}}}
        }

        schema = connector.get_collection_schema("empty_index")

        assert schema.fields == []
        assert schema.document_count == 0
