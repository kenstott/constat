"""Unit tests for Elasticsearch connector (mock-based, no Docker required)."""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from constat.catalog.nosql.elasticsearch import ElasticsearchConnector
from constat.catalog.nosql.base import NoSQLType, CollectionMetadata, FieldInfo


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

# Check if elasticsearch is available
try:
    import elasticsearch
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    ELASTICSEARCH_AVAILABLE = False


class TestElasticsearchConnection:
    """Test Elasticsearch connection methods."""

    @pytest.mark.skipif(not ELASTICSEARCH_AVAILABLE, reason="elasticsearch not installed")
    @patch("elasticsearch.Elasticsearch")
    def test_connect_simple(self, mock_es_class):
        """Test simple connection to localhost."""
        mock_client = MagicMock()
        mock_es_class.return_value = mock_client

        connector = ElasticsearchConnector()
        connector.connect()

        mock_es_class.assert_called_once_with(
            hosts=["http://localhost:9200"],
            use_ssl=False,
            verify_certs=True,
        )
        mock_client.info.assert_called_once()
        assert connector._connected is True

    @pytest.mark.skipif(not ELASTICSEARCH_AVAILABLE, reason="elasticsearch not installed")
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

    @pytest.mark.skipif(not ELASTICSEARCH_AVAILABLE, reason="elasticsearch not installed")
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
            use_ssl=False,
            verify_certs=True,
        )

    @pytest.mark.skipif(not ELASTICSEARCH_AVAILABLE, reason="elasticsearch not installed")
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
            use_ssl=False,
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

        # Should not raise
        connector.disconnect()

        assert connector._connected is False


# =============================================================================
# Test Get Collections
# =============================================================================

class TestElasticsearchGetCollections:
    """Test getting collection (index) list."""

    def test_get_collections_success(self):
        """Test listing indices."""
        connector = ElasticsearchConnector()
        connector._client = MagicMock()

        # Mock indices response
        connector._client.indices.get_alias.return_value = {
            "products": {},
            "users": {},
            "orders": {},
            ".kibana": {},  # System index, should be filtered
        }

        indices = connector.get_collections()

        assert "products" in indices
        assert "users" in indices
        assert "orders" in indices
        assert ".kibana" not in indices  # System indices filtered

    def test_get_collections_not_connected(self):
        """Test get_collections raises when not connected."""
        connector = ElasticsearchConnector()
        connector._client = None

        with pytest.raises(RuntimeError, match="Not connected"):
            connector.get_collections()

    def test_get_collections_empty(self):
        """Test get_collections with no indices."""
        connector = ElasticsearchConnector()
        connector._client = MagicMock()
        connector._client.indices.get_alias.return_value = {}

        indices = connector.get_collections()

        assert indices == []


# =============================================================================
# Test Get Collection Schema
# =============================================================================

class TestElasticsearchGetCollectionSchema:
    """Test schema (mapping) retrieval."""

    def test_get_collection_schema_success(self):
        """Test getting index mapping."""
        connector = ElasticsearchConnector()
        connector._client = MagicMock()

        # Mock mapping response
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

        # Mock count response
        connector._client.count.return_value = {"count": 1000}

        # Mock settings response
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

        # Pre-populate cache
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

        # Mock mapping with nested fields
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
# Test Query
# =============================================================================

class TestElasticsearchQuery:
    """Test query execution."""

    def test_query_with_match(self):
        """Test query with match query."""
        connector = ElasticsearchConnector()
        connector._client = MagicMock()

        connector._client.search.return_value = {
            "hits": {
                "hits": [
                    {"_source": {"name": "Product 1", "price": 10.0}},
                    {"_source": {"name": "Product 2", "price": 20.0}},
                ]
            }
        }

        results = connector.query("products", {"match": {"name": "Product"}})

        assert len(results) == 2
        assert results[0] == {"name": "Product 1", "price": 10.0}

    def test_query_empty_filter(self):
        """Test query with empty filter (match_all)."""
        connector = ElasticsearchConnector()
        connector._client = MagicMock()

        connector._client.search.return_value = {"hits": {"hits": []}}

        connector.query("products", {})

        call_kwargs = connector._client.search.call_args.kwargs
        assert call_kwargs["query"] == {"match_all": {}}

    def test_query_with_limit(self):
        """Test query respects size parameter."""
        connector = ElasticsearchConnector()
        connector._client = MagicMock()

        connector._client.search.return_value = {"hits": {"hits": []}}

        connector.query("products", {}, limit=50)

        call_kwargs = connector._client.search.call_args.kwargs
        assert call_kwargs["size"] == 50

    def test_query_not_connected(self):
        """Test query raises when not connected."""
        connector = ElasticsearchConnector()
        connector._client = None

        with pytest.raises(RuntimeError, match="Not connected"):
            connector.query("products", {})


# =============================================================================
# Test Search
# =============================================================================

class TestElasticsearchSearch:
    """Test text search methods."""

    def test_search_all_fields(self):
        """Test search across all fields."""
        connector = ElasticsearchConnector()
        connector._client = MagicMock()

        connector._client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_id": "1",
                        "_score": 5.5,
                        "_source": {"name": "Test Product", "price": 100},
                    }
                ]
            }
        }

        results = connector.search("products", "Test")

        assert len(results) == 1
        assert results[0]["name"] == "Test Product"
        assert results[0]["_score"] == 5.5
        assert results[0]["_id"] == "1"

        # Verify query_string was used
        call_kwargs = connector._client.search.call_args.kwargs
        assert "query_string" in call_kwargs["query"]

    def test_search_specific_fields(self):
        """Test search with specific fields."""
        connector = ElasticsearchConnector()
        connector._client = MagicMock()

        connector._client.search.return_value = {"hits": {"hits": []}}

        connector.search("products", "laptop", fields=["name", "description"])

        call_kwargs = connector._client.search.call_args.kwargs
        assert "multi_match" in call_kwargs["query"]
        assert call_kwargs["query"]["multi_match"]["fields"] == ["name", "description"]

    def test_search_with_limit(self):
        """Test search respects limit."""
        connector = ElasticsearchConnector()
        connector._client = MagicMock()

        connector._client.search.return_value = {"hits": {"hits": []}}

        connector.search("products", "test", limit=10)

        call_kwargs = connector._client.search.call_args.kwargs
        assert call_kwargs["size"] == 10

    def test_search_not_connected(self):
        """Test search raises when not connected."""
        connector = ElasticsearchConnector()
        connector._client = None

        with pytest.raises(RuntimeError, match="Not connected"):
            connector.search("products", "test")


# =============================================================================
# Test Aggregate
# =============================================================================

class TestElasticsearchAggregate:
    """Test aggregation queries."""

    def test_aggregate_terms(self):
        """Test terms aggregation."""
        connector = ElasticsearchConnector()
        connector._client = MagicMock()

        connector._client.search.return_value = {
            "aggregations": {
                "categories": {
                    "buckets": [
                        {"key": "Electronics", "doc_count": 100},
                        {"key": "Books", "doc_count": 50},
                    ]
                }
            }
        }

        aggs = {"categories": {"terms": {"field": "category"}}}
        results = connector.aggregate("products", aggs)

        assert "categories" in results
        assert len(results["categories"]["buckets"]) == 2

    def test_aggregate_with_filter(self):
        """Test aggregation with query filter."""
        connector = ElasticsearchConnector()
        connector._client = MagicMock()

        connector._client.search.return_value = {
            "aggregations": {"avg_price": {"value": 50.0}}
        }

        aggs = {"avg_price": {"avg": {"field": "price"}}}
        query = {"term": {"category": "Electronics"}}

        connector.aggregate("products", aggs, query=query)

        call_kwargs = connector._client.search.call_args.kwargs
        assert call_kwargs["query"] == query
        assert call_kwargs["size"] == 0  # No documents returned

    def test_aggregate_not_connected(self):
        """Test aggregate raises when not connected."""
        connector = ElasticsearchConnector()
        connector._client = None

        with pytest.raises(RuntimeError, match="Not connected"):
            connector.aggregate("products", {})


# =============================================================================
# Test Insert
# =============================================================================

class TestElasticsearchInsert:
    """Test document insertion."""

    def test_insert_single_document(self):
        """Test inserting a single document."""
        connector = ElasticsearchConnector()
        connector._client = MagicMock()

        connector._client.index.return_value = {"_id": "abc123"}

        documents = [{"name": "Product 1", "price": 10.0}]
        ids = connector.insert("products", documents)

        assert ids == ["abc123"]
        connector._client.index.assert_called_once_with(
            index="products",
            document={"name": "Product 1", "price": 10.0}
        )

    def test_insert_with_custom_id(self):
        """Test inserting document with custom ID."""
        connector = ElasticsearchConnector()
        connector._client = MagicMock()

        connector._client.index.return_value = {"_id": "custom-id"}

        documents = [{"_id": "custom-id", "name": "Product 1"}]
        ids = connector.insert("products", documents)

        assert ids == ["custom-id"]
        connector._client.index.assert_called_once_with(
            index="products",
            id="custom-id",
            document={"name": "Product 1"}
        )

    def test_insert_multiple_documents(self):
        """Test inserting multiple documents."""
        connector = ElasticsearchConnector()
        connector._client = MagicMock()

        connector._client.index.side_effect = [
            {"_id": "id1"},
            {"_id": "id2"},
            {"_id": "id3"},
        ]

        documents = [
            {"name": "Product 1"},
            {"name": "Product 2"},
            {"name": "Product 3"},
        ]
        ids = connector.insert("products", documents)

        assert ids == ["id1", "id2", "id3"]
        assert connector._client.index.call_count == 3

    def test_insert_not_connected(self):
        """Test insert raises when not connected."""
        connector = ElasticsearchConnector()
        connector._client = None

        with pytest.raises(RuntimeError, match="Not connected"):
            connector.insert("products", [{"name": "Test"}])


# =============================================================================
# Test Bulk Insert
# =============================================================================

class TestElasticsearchBulkInsert:
    """Test bulk document insertion."""

    @pytest.mark.skipif(not ELASTICSEARCH_AVAILABLE, reason="elasticsearch not installed")
    @patch("elasticsearch.helpers.bulk")
    def test_bulk_insert_success(self, mock_bulk):
        """Test successful bulk insert."""
        connector = ElasticsearchConnector()
        connector._client = MagicMock()

        mock_bulk.return_value = (100, [])

        documents = [{"name": f"Product {i}"} for i in range(100)]
        result = connector.bulk_insert("products", documents)

        assert result["success"] == 100
        assert result["errors"] == []
        mock_bulk.assert_called_once()

    @pytest.mark.skipif(not ELASTICSEARCH_AVAILABLE, reason="elasticsearch not installed")
    @patch("elasticsearch.helpers.bulk")
    def test_bulk_insert_with_errors(self, mock_bulk):
        """Test bulk insert with some errors."""
        connector = ElasticsearchConnector()
        connector._client = MagicMock()

        mock_bulk.return_value = (95, [{"error": "mapping error"}] * 5)

        documents = [{"name": f"Product {i}"} for i in range(100)]
        result = connector.bulk_insert("products", documents)

        assert result["success"] == 95
        assert len(result["errors"]) == 5

    def test_bulk_insert_not_connected(self):
        """Test bulk_insert raises when not connected."""
        connector = ElasticsearchConnector()
        connector._client = None

        with pytest.raises(RuntimeError, match="Not connected"):
            connector.bulk_insert("products", [{"name": "Test"}])


# =============================================================================
# Test Delete
# =============================================================================

class TestElasticsearchDelete:
    """Test document deletion."""

    def test_delete_by_query(self):
        """Test deleting documents by query."""
        connector = ElasticsearchConnector()
        connector._client = MagicMock()

        connector._client.delete_by_query.return_value = {"deleted": 10}

        query = {"match": {"status": "archived"}}
        deleted = connector.delete("products", query)

        assert deleted == 10
        connector._client.delete_by_query.assert_called_once_with(
            index="products",
            query=query,
        )

    def test_delete_not_connected(self):
        """Test delete raises when not connected."""
        connector = ElasticsearchConnector()
        connector._client = None

        with pytest.raises(RuntimeError, match="Not connected"):
            connector.delete("products", {"match_all": {}})


# =============================================================================
# Test Create Index
# =============================================================================

class TestElasticsearchCreateIndex:
    """Test index creation."""

    def test_create_index_simple(self):
        """Test creating a simple index."""
        connector = ElasticsearchConnector()
        connector._client = MagicMock()

        connector.create_index("new_index")

        connector._client.indices.create.assert_called_once_with(
            index="new_index",
            body=None,
        )

    def test_create_index_with_mappings(self):
        """Test creating index with mappings."""
        connector = ElasticsearchConnector()
        connector._client = MagicMock()

        mappings = {
            "properties": {
                "name": {"type": "text"},
                "price": {"type": "float"},
            }
        }

        connector.create_index("products", mappings=mappings)

        call_kwargs = connector._client.indices.create.call_args.kwargs
        assert call_kwargs["body"]["mappings"] == mappings

    def test_create_index_with_settings(self):
        """Test creating index with settings."""
        connector = ElasticsearchConnector()
        connector._client = MagicMock()

        settings = {
            "number_of_shards": 3,
            "number_of_replicas": 2,
        }

        connector.create_index("products", settings=settings)

        call_kwargs = connector._client.indices.create.call_args.kwargs
        assert call_kwargs["body"]["settings"] == settings

    def test_create_index_not_connected(self):
        """Test create_index raises when not connected."""
        connector = ElasticsearchConnector()
        connector._client = None

        with pytest.raises(RuntimeError, match="Not connected"):
            connector.create_index("new_index")


# =============================================================================
# Test Delete Index
# =============================================================================

class TestElasticsearchDeleteIndex:
    """Test index deletion."""

    def test_delete_index_success(self):
        """Test deleting an index."""
        connector = ElasticsearchConnector()
        connector._client = MagicMock()

        connector.delete_index("old_index")

        connector._client.indices.delete.assert_called_once_with(index="old_index")

    def test_delete_index_not_connected(self):
        """Test delete_index raises when not connected."""
        connector = ElasticsearchConnector()
        connector._client = None

        with pytest.raises(RuntimeError, match="Not connected"):
            connector.delete_index("old_index")


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

        # Mapping without properties key
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
