# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tests for NoSQL database connectors."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass

# Check for optional dependencies
try:
    import pymongo
    HAS_PYMONGO = True
except ImportError:
    HAS_PYMONGO = False

from constat.catalog.nosql import (
    NoSQLConnector,
    NoSQLType,
    CollectionMetadata,
    FieldInfo,
    MongoDBConnector,
    CassandraConnector,
    ElasticsearchConnector,
    DynamoDBConnector,
    CosmosDBConnector,
    FirestoreConnector,
)


class TestNoSQLBase:
    """Tests for base NoSQL connector functionality."""

    def test_nosql_type_enum(self):
        """Test NoSQLType enum values."""
        assert NoSQLType.DOCUMENT.value == "document"
        assert NoSQLType.WIDE_COLUMN.value == "wide_column"
        assert NoSQLType.KEY_VALUE.value == "key_value"
        assert NoSQLType.GRAPH.value == "graph"
        assert NoSQLType.SEARCH.value == "search"
        assert NoSQLType.TIME_SERIES.value == "time_series"

    def test_field_info_defaults(self):
        """Test FieldInfo dataclass defaults."""
        field = FieldInfo(name="test", data_type="string")
        assert field.name == "test"
        assert field.data_type == "string"
        assert field.nullable is True
        assert field.is_indexed is False
        assert field.is_unique is False
        assert field.sample_values == []
        assert field.description is None

    def test_collection_metadata_to_dict(self):
        """Test CollectionMetadata serialization."""
        fields = [
            FieldInfo(name="id", data_type="string", is_indexed=True),
            FieldInfo(name="name", data_type="string"),
        ]
        metadata = CollectionMetadata(
            name="users",
            database="testdb",
            nosql_type=NoSQLType.DOCUMENT,
            fields=fields,
            document_count=100,
        )

        result = metadata.to_dict()
        assert result["name"] == "users"
        assert result["database"] == "testdb"
        assert result["type"] == "document"
        assert len(result["fields"]) == 2
        assert result["document_count"] == 100

    def test_collection_metadata_to_embedding_text(self):
        """Test embedding text generation."""
        fields = [FieldInfo(name="id", data_type="string")]
        metadata = CollectionMetadata(
            name="users",
            database="testdb",
            nosql_type=NoSQLType.DOCUMENT,
            fields=fields,
            indexes=["id_idx"],
            description="User collection",
        )

        text = metadata.to_embedding_text()
        assert "Collection: testdb.users" in text
        assert "Type: document" in text
        assert "Description: User collection" in text
        assert "Fields: id" in text
        assert "Indexes: id_idx" in text


class TestMongoDBConnector:
    """Tests for MongoDB connector."""

    def test_init(self):
        """Test MongoDB connector initialization."""
        connector = MongoDBConnector(
            uri="mongodb://localhost:27017",
            database="testdb",
            name="test_mongo",
        )
        assert connector.uri == "mongodb://localhost:27017"
        assert connector.database_name == "testdb"
        assert connector.name == "test_mongo"
        assert connector.nosql_type == NoSQLType.DOCUMENT

    def test_default_name(self):
        """Test default name from database."""
        connector = MongoDBConnector(
            uri="mongodb://localhost:27017",
            database="mydb",
        )
        assert connector.name == "mydb"

    @pytest.mark.skipif(not HAS_PYMONGO, reason="pymongo not installed")
    @patch("pymongo.MongoClient")
    def test_connect(self, mock_client_class):
        """Test MongoDB connection."""
        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_client.__getitem__.return_value = mock_db
        mock_client_class.return_value = mock_client

        connector = MongoDBConnector(
            uri="mongodb://localhost:27017",
            database="testdb",
        )
        connector.connect()

        assert connector.is_connected
        mock_client_class.assert_called_once_with("mongodb://localhost:27017")

    @pytest.mark.skipif(not HAS_PYMONGO, reason="pymongo not installed")
    @patch("pymongo.MongoClient")
    def test_disconnect(self, mock_client_class):
        """Test MongoDB disconnection."""
        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_client.__getitem__.return_value = mock_db
        mock_client_class.return_value = mock_client

        connector = MongoDBConnector(
            uri="mongodb://localhost:27017",
            database="testdb",
        )
        connector.connect()
        connector.disconnect()

        assert not connector.is_connected
        mock_client.close.assert_called_once()

    @pytest.mark.skipif(not HAS_PYMONGO, reason="pymongo not installed")
    @patch("pymongo.MongoClient")
    def test_get_collections(self, mock_client_class):
        """Test listing collections."""
        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_db.list_collection_names.return_value = ["users", "orders"]
        mock_client.__getitem__.return_value = mock_db
        mock_client_class.return_value = mock_client

        connector = MongoDBConnector(
            uri="mongodb://localhost:27017",
            database="testdb",
        )
        connector.connect()

        collections = connector.get_collections()
        assert collections == ["users", "orders"]


class TestCassandraConnector:
    """Tests for Cassandra connector."""

    def test_init(self):
        """Test Cassandra connector initialization."""
        connector = CassandraConnector(
            keyspace="test_keyspace",
            hosts=["localhost"],
            name="test_cassandra",
        )
        assert connector.keyspace == "test_keyspace"
        assert connector.hosts == ["localhost"]
        assert connector.name == "test_cassandra"
        assert connector.nosql_type == NoSQLType.WIDE_COLUMN

    def test_cloud_config(self):
        """Test Cassandra cloud config (DataStax Astra)."""
        connector = CassandraConnector(
            keyspace="test_keyspace",
            cloud_config={"secure_connect_bundle": "/path/to/bundle.zip"},
            auth_provider=("client_id", "secret"),
        )
        assert connector.cloud_config is not None
        assert connector.auth_provider == ("client_id", "secret")


class TestElasticsearchConnector:
    """Tests for Elasticsearch connector."""

    def test_init(self):
        """Test Elasticsearch connector initialization."""
        connector = ElasticsearchConnector(
            hosts=["http://localhost:9200"],
            name="test_es",
        )
        assert connector.hosts == ["http://localhost:9200"]
        assert connector.name == "test_es"
        assert connector.nosql_type == NoSQLType.SEARCH

    def test_cloud_id_init(self):
        """Test Elasticsearch cloud initialization."""
        connector = ElasticsearchConnector(
            cloud_id="deployment:abc123",
            api_key="api_key_here",
            name="elastic_cloud",
        )
        assert connector.cloud_id == "deployment:abc123"
        assert connector.api_key == "api_key_here"


class TestDynamoDBConnector:
    """Tests for DynamoDB connector."""

    def test_init(self):
        """Test DynamoDB connector initialization."""
        connector = DynamoDBConnector(
            region="us-east-1",
            name="test_dynamodb",
        )
        assert connector.region == "us-east-1"
        assert connector.name == "test_dynamodb"
        assert connector.nosql_type == NoSQLType.KEY_VALUE

    def test_local_endpoint(self):
        """Test DynamoDB local endpoint configuration."""
        connector = DynamoDBConnector(
            endpoint_url="http://localhost:8000",
            name="dynamodb_local",
        )
        assert connector.endpoint_url == "http://localhost:8000"

    def test_type_map(self):
        """Test DynamoDB type mapping."""
        assert DynamoDBConnector.TYPE_MAP["S"] == "string"
        assert DynamoDBConnector.TYPE_MAP["N"] == "number"
        assert DynamoDBConnector.TYPE_MAP["B"] == "binary"
        assert DynamoDBConnector.TYPE_MAP["M"] == "map"
        assert DynamoDBConnector.TYPE_MAP["L"] == "list"
        assert DynamoDBConnector.TYPE_MAP["BOOL"] == "boolean"


class TestCosmosDBConnector:
    """Tests for Azure Cosmos DB connector."""

    def test_init_with_endpoint_and_key(self):
        """Test Cosmos DB initialization with endpoint and key."""
        connector = CosmosDBConnector(
            endpoint="https://account.documents.azure.com:443/",
            key="primary_key",
            database="testdb",
            name="test_cosmos",
        )
        assert connector.endpoint == "https://account.documents.azure.com:443/"
        assert connector.key == "primary_key"
        assert connector.database_name == "testdb"
        assert connector.nosql_type == NoSQLType.DOCUMENT

    def test_init_with_connection_string(self):
        """Test Cosmos DB initialization with connection string."""
        connector = CosmosDBConnector(
            connection_string="AccountEndpoint=https://...;AccountKey=...",
            database="testdb",
        )
        assert connector.connection_string is not None


class TestFirestoreConnector:
    """Tests for Google Firestore connector."""

    def test_init(self):
        """Test Firestore connector initialization."""
        connector = FirestoreConnector(
            project="my-gcp-project",
            name="test_firestore",
        )
        assert connector.project == "my-gcp-project"
        assert connector.database_id == "(default)"
        assert connector.name == "test_firestore"
        assert connector.nosql_type == NoSQLType.DOCUMENT

    def test_custom_database(self):
        """Test Firestore with custom database ID."""
        connector = FirestoreConnector(
            project="my-gcp-project",
            database="my-database",
        )
        assert connector.database_id == "my-database"

    def test_credentials_path(self):
        """Test Firestore with service account credentials."""
        connector = FirestoreConnector(
            project="my-gcp-project",
            credentials_path="/path/to/credentials.json",
        )
        assert connector.credentials_path == "/path/to/credentials.json"


class TestSchemaInference:
    """Tests for schema inference from samples."""

    def test_infer_field_type_string(self):
        """Test inferring string type."""
        connector = MongoDBConnector(
            uri="mongodb://localhost:27017",
            database="test",
        )
        assert connector.infer_field_type(["hello", "world"]) == "string"

    def test_infer_field_type_integer(self):
        """Test inferring integer type."""
        connector = MongoDBConnector(
            uri="mongodb://localhost:27017",
            database="test",
        )
        assert connector.infer_field_type([1, 2, 3]) == "integer"

    def test_infer_field_type_float(self):
        """Test inferring float type."""
        connector = MongoDBConnector(
            uri="mongodb://localhost:27017",
            database="test",
        )
        assert connector.infer_field_type([1.5, 2.5, 3.5]) == "float"

    def test_infer_field_type_boolean(self):
        """Test inferring boolean type."""
        connector = MongoDBConnector(
            uri="mongodb://localhost:27017",
            database="test",
        )
        assert connector.infer_field_type([True, False]) == "boolean"

    def test_infer_field_type_array(self):
        """Test inferring array type."""
        connector = MongoDBConnector(
            uri="mongodb://localhost:27017",
            database="test",
        )
        assert connector.infer_field_type([[1, 2], [3, 4]]) == "array"

    def test_infer_field_type_object(self):
        """Test inferring object type."""
        connector = MongoDBConnector(
            uri="mongodb://localhost:27017",
            database="test",
        )
        assert connector.infer_field_type([{"a": 1}, {"b": 2}]) == "object"

    def test_infer_field_type_mixed(self):
        """Test inferring mixed type."""
        connector = MongoDBConnector(
            uri="mongodb://localhost:27017",
            database="test",
        )
        assert connector.infer_field_type(["hello", 123, True]) == "mixed"

    def test_infer_field_type_null(self):
        """Test inferring null type."""
        connector = MongoDBConnector(
            uri="mongodb://localhost:27017",
            database="test",
        )
        assert connector.infer_field_type([None, None]) == "null"

    def test_infer_schema_from_samples(self):
        """Test full schema inference from documents."""
        connector = MongoDBConnector(
            uri="mongodb://localhost:27017",
            database="test",
        )

        samples = [
            {"name": "Alice", "age": 30, "active": True},
            {"name": "Bob", "age": 25, "active": False},
            {"name": "Charlie", "age": None, "active": True},
        ]

        metadata = connector.infer_schema_from_samples("users", samples)

        assert metadata.name == "users"
        assert len(metadata.fields) == 3

        field_map = {f.name: f for f in metadata.fields}
        assert field_map["name"].data_type == "string"
        assert field_map["age"].data_type == "integer"
        assert field_map["age"].nullable is True  # One None value
        assert field_map["active"].data_type == "boolean"

    def test_infer_nested_schema(self):
        """Test schema inference with nested objects."""
        connector = MongoDBConnector(
            uri="mongodb://localhost:27017",
            database="test",
        )

        samples = [
            {"user": {"name": "Alice", "email": "alice@example.com"}},
            {"user": {"name": "Bob", "email": "bob@example.com"}},
        ]

        metadata = connector.infer_schema_from_samples("nested", samples)

        field_names = [f.name for f in metadata.fields]
        assert "user.name" in field_names
        assert "user.email" in field_names


class TestNotConnectedErrors:
    """Tests for proper error handling when not connected."""

    def test_mongodb_not_connected(self):
        """Test MongoDB operations fail when not connected."""
        connector = MongoDBConnector(
            uri="mongodb://localhost:27017",
            database="test",
        )
        with pytest.raises(RuntimeError, match="Not connected"):
            connector.get_collections()

    def test_cassandra_not_connected(self):
        """Test Cassandra operations fail when not connected."""
        connector = CassandraConnector(
            keyspace="test",
            hosts=["localhost"],
        )
        with pytest.raises(RuntimeError, match="Not connected"):
            connector.get_collections()

    def test_elasticsearch_not_connected(self):
        """Test Elasticsearch operations fail when not connected."""
        connector = ElasticsearchConnector(
            hosts=["http://localhost:9200"],
        )
        with pytest.raises(RuntimeError, match="Not connected"):
            connector.get_collections()

    def test_dynamodb_not_connected(self):
        """Test DynamoDB operations fail when not connected."""
        connector = DynamoDBConnector(region="us-east-1")
        with pytest.raises(RuntimeError, match="Not connected"):
            connector.get_collections()

    def test_cosmosdb_not_connected(self):
        """Test Cosmos DB operations fail when not connected."""
        connector = CosmosDBConnector(
            endpoint="https://test.documents.azure.com:443/",
            key="key",
            database="test",
        )
        with pytest.raises(RuntimeError, match="Not connected"):
            connector.get_collections()

    def test_firestore_not_connected(self):
        """Test Firestore operations fail when not connected."""
        connector = FirestoreConnector(project="test-project")
        with pytest.raises(RuntimeError, match="Not connected"):
            connector.get_collections()
