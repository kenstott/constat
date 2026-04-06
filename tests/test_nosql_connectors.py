# Copyright (c) 2025 Kenneth Stott
# Canary: e76993c5-6e89-4c67-af0c-0ba7858bd963
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tests for NoSQL base types and shared connector functionality."""

from __future__ import annotations
import pytest

try:
    import pymongo
except ImportError:
    pytest.fail("pymongo is required but not installed — run: pip install pymongo")

try:
    import neo4j as _neo4j_mod
except ImportError:
    pytest.fail("neo4j is required but not installed — run: pip install neo4j")

from constat.catalog.nosql import (
    NoSQLType,
    CollectionMetadata,
    FieldInfo,
    MongoDBConnector,
    CassandraConnector,
    ElasticsearchConnector,
    DynamoDBConnector,
    CosmosDBConnector,
    FirestoreConnector,
    Neo4jConnector,
    JaegerConnector,
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
        assert NoSQLType.OBSERVABILITY.value == "observability"

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

    def test_neo4j_not_connected(self):
        """Test Neo4j operations fail when not connected."""
        connector = Neo4jConnector(uri="bolt://localhost:7687")
        with pytest.raises(RuntimeError, match="Not connected"):
            connector.get_collections()

    def test_neo4j_query_not_connected(self):
        """Test Neo4j query fails when not connected."""
        connector = Neo4jConnector(uri="bolt://localhost:7687")
        with pytest.raises(RuntimeError, match="Not connected"):
            connector.query("Person", {})

    def test_neo4j_cypher_not_connected(self):
        """Test Neo4j cypher fails when not connected."""
        connector = Neo4jConnector(uri="bolt://localhost:7687")
        with pytest.raises(RuntimeError, match="Not connected"):
            connector.cypher("MATCH (n) RETURN n")

    def test_neo4j_graph_schema_not_connected(self):
        """Test Neo4j graph_schema fails when not connected."""
        connector = Neo4jConnector(uri="bolt://localhost:7687")
        with pytest.raises(RuntimeError, match="Not connected"):
            connector.graph_schema()

    def test_neo4j_get_overview_not_connected(self):
        """Test Neo4j get_overview fails when not connected."""
        connector = Neo4jConnector(uri="bolt://localhost:7687")
        with pytest.raises(RuntimeError, match="Not connected"):
            connector.get_overview()

    def test_jaeger_not_connected(self):
        """Test Jaeger operations fail when not connected."""
        connector = JaegerConnector(uri="http://localhost:16686")
        with pytest.raises(RuntimeError, match="Not connected"):
            connector.get_collections()

    def test_jaeger_query_not_connected(self):
        """Test Jaeger query fails when not connected."""
        connector = JaegerConnector(uri="http://localhost:16686")
        with pytest.raises(RuntimeError, match="Not connected"):
            connector.query("frontend", {})
