# Copyright (c) 2025 Kenneth Stott
# Canary: e76993c5-6e89-4c67-af0c-0ba7858bd963
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

try:
    import neo4j as _neo4j_mod
    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False

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


class TestNeo4jConnector:
    """Tests for Neo4j connector."""

    def test_init(self):
        """Test Neo4j connector initialization."""
        connector = Neo4jConnector(
            uri="bolt://localhost:7687",
            database="movies",
            username="neo4j",
            password="password",
            name="test_neo4j",
        )
        assert connector.uri == "bolt://localhost:7687"
        assert connector.database_name == "movies"
        assert connector.username == "neo4j"
        assert connector.name == "test_neo4j"
        assert connector.nosql_type == NoSQLType.GRAPH

    def test_default_name(self):
        """Test default name from database."""
        connector = Neo4jConnector(
            uri="bolt://localhost:7687",
            database="movies",
        )
        assert connector.name == "movies"

    def test_default_database(self):
        """Test default database is 'neo4j'."""
        connector = Neo4jConnector()
        assert connector.database_name == "neo4j"

    @pytest.mark.skipif(not HAS_NEO4J, reason="neo4j not installed")
    @patch("neo4j.GraphDatabase")
    def test_connect(self, mock_gdb):
        """Test Neo4j connection."""
        mock_driver = MagicMock()
        mock_gdb.driver.return_value = mock_driver

        connector = Neo4jConnector(
            uri="bolt://localhost:7687",
            username="neo4j",
            password="password",
        )
        connector.connect()

        assert connector.is_connected
        mock_gdb.driver.assert_called_once_with(
            "bolt://localhost:7687", auth=("neo4j", "password")
        )
        mock_driver.verify_connectivity.assert_called_once()

    @pytest.mark.skipif(not HAS_NEO4J, reason="neo4j not installed")
    @patch("neo4j.GraphDatabase")
    def test_connect_no_auth(self, mock_gdb):
        """Test Neo4j connection without auth."""
        mock_driver = MagicMock()
        mock_gdb.driver.return_value = mock_driver

        connector = Neo4jConnector(uri="bolt://localhost:7687")
        connector.connect()

        mock_gdb.driver.assert_called_once_with("bolt://localhost:7687", auth=None)

    @pytest.mark.skipif(not HAS_NEO4J, reason="neo4j not installed")
    @patch("neo4j.GraphDatabase")
    def test_disconnect(self, mock_gdb):
        """Test Neo4j disconnection."""
        mock_driver = MagicMock()
        mock_gdb.driver.return_value = mock_driver

        connector = Neo4jConnector(uri="bolt://localhost:7687")
        connector.connect()
        connector.disconnect()

        assert not connector.is_connected
        mock_driver.close.assert_called_once()

    def _make_connected_connector(self):
        """Create a connector with a mocked driver."""
        connector = Neo4jConnector(
            uri="bolt://localhost:7687",
            database="movies",
            name="test_neo4j",
        )
        connector._driver = MagicMock()
        connector._connected = True
        return connector

    def _mock_session_run(self, connector, results):
        """Set up mock session.run to return given results."""
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.__iter__ = Mock(return_value=iter(results))
        mock_session.run.return_value = mock_result
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)
        connector._driver.session.return_value = mock_session
        return mock_session

    def test_get_collections(self):
        """Test listing node labels and relationship types."""
        connector = self._make_connected_connector()

        call_count = [0]
        label_results = [{"label": "Movie"}, {"label": "Person"}]
        rel_results = [{"relationshipType": "ACTED_IN"}, {"relationshipType": "DIRECTED"}]

        def run_side_effect(query, parameters=None):
            mock_result = MagicMock()
            if "db.labels" in query:
                mock_result.__iter__ = Mock(return_value=iter(label_results))
            else:
                mock_result.__iter__ = Mock(return_value=iter(rel_results))
            return mock_result

        mock_session = MagicMock()
        mock_session.run.side_effect = run_side_effect
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)
        connector._driver.session.return_value = mock_session

        collections = connector.get_collections()
        assert "Movie" in collections
        assert "Person" in collections
        assert "rel:ACTED_IN" in collections
        assert "rel:DIRECTED" in collections

    def test_query_nodes(self):
        """Test querying nodes by properties."""
        connector = self._make_connected_connector()

        # Mock Node-like object
        mock_node = MagicMock()
        mock_node.__class__.__name__ = "Node"
        type(mock_node).__name__ = "Node"
        mock_node.__iter__ = Mock(return_value=iter([("name", "Tom Hanks"), ("born", 1956)]))
        mock_node.keys = Mock(return_value=["name", "born"])

        results = [{"n": mock_node}]
        self._mock_session_run(connector, results)

        result = connector.query("Person", {"name": "Tom Hanks"}, limit=10)
        assert len(result) == 1

    def test_query_relationships(self):
        """Test querying relationships."""
        connector = self._make_connected_connector()

        mock_rel = MagicMock()
        type(mock_rel).__name__ = "Relationship"
        mock_rel.__iter__ = Mock(return_value=iter([("roles", ["Forrest"])]))
        mock_rel.keys = Mock(return_value=["roles"])

        results = [{"r": mock_rel, "source_label": "Person", "target_label": "Movie"}]
        self._mock_session_run(connector, results)

        result = connector.query("rel:ACTED_IN", {"roles": ["Forrest"]}, limit=10)
        assert len(result) == 1

    def test_cypher_execution(self):
        """Test raw Cypher execution."""
        connector = self._make_connected_connector()

        results = [{"name": "Tom Hanks", "title": "Forrest Gump"}]
        self._mock_session_run(connector, results)

        result = connector.cypher(
            "MATCH (p:Person)-[:ACTED_IN]->(m:Movie) RETURN p.name AS name, m.title AS title"
        )
        assert len(result) == 1
        assert result[0]["name"] == "Tom Hanks"

    def test_graph_schema(self):
        """Test graph schema extraction."""
        connector = self._make_connected_connector()

        call_number = [0]

        def run_side_effect(query, parameters=None):
            mock_result = MagicMock()
            call_number[0] += 1
            if "db.labels" in query:
                mock_result.__iter__ = Mock(return_value=iter([{"label": "Person"}]))
            elif "relationshipType" in query:
                mock_result.__iter__ = Mock(
                    return_value=iter([{"relationshipType": "ACTED_IN"}])
                )
            else:
                mock_result.__iter__ = Mock(
                    return_value=iter([
                        {"src": "Person", "rel": "ACTED_IN", "tgt": "Movie"}
                    ])
                )
            return mock_result

        mock_session = MagicMock()
        mock_session.run.side_effect = run_side_effect
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)
        connector._driver.session.return_value = mock_session

        schema = connector.graph_schema()
        assert "Person" in schema["labels"]
        assert "ACTED_IN" in schema["relationship_types"]
        assert "(:Person)-[:ACTED_IN]->(:Movie)" in schema["patterns"]

    def test_convert_value_primitives(self):
        """Test _convert_value with primitive types."""
        connector = Neo4jConnector()
        assert connector._convert_value("hello") == "hello"
        assert connector._convert_value(42) == 42
        assert connector._convert_value(3.14) == 3.14
        assert connector._convert_value(True) is True
        assert connector._convert_value(None) is None

    def test_convert_value_list(self):
        """Test _convert_value with lists."""
        connector = Neo4jConnector()
        assert connector._convert_value([1, "two", 3]) == [1, "two", 3]

    def test_convert_value_dict(self):
        """Test _convert_value with dicts."""
        connector = Neo4jConnector()
        assert connector._convert_value({"a": 1}) == {"a": 1}

    def test_convert_value_node(self):
        """Test _convert_value with Node-like object."""
        connector = Neo4jConnector()
        mock_node = MagicMock()
        type(mock_node).__name__ = "Node"
        mock_node.__iter__ = Mock(return_value=iter([("name", "Alice")]))
        mock_node.keys = Mock(return_value=["name"])
        mock_node.labels = {"Person"}

        result = connector._convert_value(mock_node)
        assert "_labels" in result

    def test_convert_value_relationship(self):
        """Test _convert_value with Relationship-like object."""
        connector = Neo4jConnector()
        mock_rel = MagicMock()
        type(mock_rel).__name__ = "Relationship"
        mock_rel.__iter__ = Mock(return_value=iter([("since", 2020)]))
        mock_rel.keys = Mock(return_value=["since"])
        mock_rel.type = "KNOWS"

        result = connector._convert_value(mock_rel)
        assert result["_type"] == "KNOWS"

    def test_convert_value_path(self):
        """Test _convert_value with Path-like object."""
        connector = Neo4jConnector()
        mock_path = MagicMock()
        type(mock_path).__name__ = "Path"
        mock_path.nodes = []
        mock_path.relationships = []

        result = connector._convert_value(mock_path)
        assert "_nodes" in result
        assert "_relationships" in result

    def test_rel_prefix_constant(self):
        """Test the relationship prefix constant."""
        assert Neo4jConnector.REL_PREFIX == "rel:"


class TestJaegerConnector:
    """Tests for Jaeger connector."""

    # Sample Jaeger API responses
    SERVICES_RESPONSE = {"data": ["frontend", "backend", "db-service"]}
    OPERATIONS_RESPONSE = {"data": ["GET /api/users", "POST /api/orders", "DB query"]}
    TRACES_RESPONSE = {
        "data": [
            {
                "traceID": "abc123",
                "processes": {
                    "p1": {
                        "serviceName": "frontend",
                        "tags": [{"key": "hostname", "value": "web-01"}],
                    },
                },
                "spans": [
                    {
                        "traceID": "abc123",
                        "spanID": "span1",
                        "operationName": "GET /api/users",
                        "processID": "p1",
                        "startTime": 1700000000000000,
                        "duration": 15000,
                        "tags": [
                            {"key": "http.method", "value": "GET"},
                            {"key": "http.status_code", "value": 200},
                        ],
                        "references": [],
                    },
                    {
                        "traceID": "abc123",
                        "spanID": "span2",
                        "operationName": "DB query",
                        "processID": "p1",
                        "startTime": 1700000000005000,
                        "duration": 8000,
                        "tags": [
                            {"key": "db.type", "value": "postgresql"},
                            {"key": "error", "value": True},
                        ],
                        "references": [
                            {"refType": "CHILD_OF", "spanID": "span1"},
                        ],
                    },
                ],
            },
        ],
    }

    def test_init(self):
        connector = JaegerConnector(
            uri="http://localhost:16686",
            name="test_jaeger",
        )
        assert connector.uri == "http://localhost:16686"
        assert connector.name == "test_jaeger"
        assert connector.nosql_type == NoSQLType.OBSERVABILITY

    def test_default_name(self):
        connector = JaegerConnector()
        assert connector.name == "jaeger"

    def test_uri_trailing_slash_stripped(self):
        connector = JaegerConnector(uri="http://localhost:16686/")
        assert connector.uri == "http://localhost:16686"

    @patch("constat.catalog.nosql.jaeger.requests.Session")
    def test_connect(self, mock_session_class):
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = self.SERVICES_RESPONSE
        mock_session.get.return_value = mock_resp
        mock_session_class.return_value = mock_session

        connector = JaegerConnector(uri="http://localhost:16686")
        connector.connect()

        assert connector.is_connected
        mock_session.get.assert_called_once_with("http://localhost:16686/api/services")

    @patch("constat.catalog.nosql.jaeger.requests.Session")
    def test_connect_with_auth(self, mock_session_class):
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = self.SERVICES_RESPONSE
        mock_session.get.return_value = mock_resp
        mock_session_class.return_value = mock_session

        connector = JaegerConnector(
            uri="http://localhost:16686",
            username="admin",
            password="secret",
        )
        connector.connect()

        assert mock_session.auth == ("admin", "secret")

    @patch("constat.catalog.nosql.jaeger.requests.Session")
    def test_disconnect(self, mock_session_class):
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = self.SERVICES_RESPONSE
        mock_session.get.return_value = mock_resp
        mock_session_class.return_value = mock_session

        connector = JaegerConnector()
        connector.connect()
        connector.disconnect()

        assert not connector.is_connected
        mock_session.close.assert_called_once()

    def _make_connected_connector(self):
        """Create a connector with a mocked session."""
        connector = JaegerConnector(
            uri="http://localhost:16686",
            name="test_jaeger",
            sample_size=50,
        )
        connector._session = MagicMock()
        connector._connected = True
        return connector

    def _mock_get(self, connector, responses):
        """Mock _session.get to return responses by URL pattern."""
        def side_effect(url, params=None):
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            for pattern, data in responses.items():
                if pattern in url:
                    mock_resp.json.return_value = data
                    return mock_resp
            mock_resp.json.return_value = {"data": []}
            return mock_resp
        connector._session.get.side_effect = side_effect

    def test_get_collections(self):
        connector = self._make_connected_connector()
        self._mock_get(connector, {"/api/services": self.SERVICES_RESPONSE})

        collections = connector.get_collections()
        assert collections == ["backend", "db-service", "frontend"]

    def test_get_collection_schema(self):
        connector = self._make_connected_connector()
        self._mock_get(connector, {"/api/traces": self.TRACES_RESPONSE})

        schema = connector.get_collection_schema("frontend")

        assert schema.name == "frontend"
        assert schema.database == "test_jaeger"
        assert schema.nosql_type == NoSQLType.OBSERVABILITY

        field_names = [f.name for f in schema.fields]
        # Fixed fields
        assert "traceID" in field_names
        assert "spanID" in field_names
        assert "operationName" in field_names
        assert "serviceName" in field_names
        assert "duration" in field_names
        assert "error" in field_names
        # Dynamic tag fields
        assert "tags.http.method" in field_names
        assert "tags.db.type" in field_names

    def test_get_collection_schema_cached(self):
        connector = self._make_connected_connector()
        self._mock_get(connector, {"/api/traces": self.TRACES_RESPONSE})

        schema1 = connector.get_collection_schema("frontend")
        schema2 = connector.get_collection_schema("frontend")
        assert schema1 is schema2
        # Only one HTTP call
        assert connector._session.get.call_count == 1

    def test_query(self):
        connector = self._make_connected_connector()
        self._mock_get(connector, {"/api/traces": self.TRACES_RESPONSE})

        results = connector.query("frontend", {"operation": "GET /api/users"}, limit=10)

        assert len(results) == 2
        assert results[0]["traceID"] == "abc123"
        assert results[0]["operationName"] == "GET /api/users"
        assert results[0]["serviceName"] == "frontend"

    def test_query_passes_params(self):
        connector = self._make_connected_connector()
        self._mock_get(connector, {"/api/traces": {"data": []}})

        connector.query("frontend", {
            "operation": "GET /api",
            "minDuration": "1ms",
            "tags": '{"http.method":"GET"}',
        }, limit=5)

        call_args = connector._session.get.call_args
        params = call_args[1].get("params") or call_args[0][1] if len(call_args[0]) > 1 else call_args[1]["params"]
        assert params["service"] == "frontend"
        assert params["operation"] == "GET /api"
        assert params["minDuration"] == "1ms"
        assert params["limit"] == 5

    def test_get_trace(self):
        connector = self._make_connected_connector()
        self._mock_get(connector, {"/api/traces/abc123": self.TRACES_RESPONSE})

        spans = connector.get_trace("abc123")

        assert len(spans) == 2
        assert spans[0]["traceID"] == "abc123"
        assert spans[1]["parentSpanID"] == "span1"

    def test_get_operations(self):
        connector = self._make_connected_connector()
        self._mock_get(connector, {"/operations": self.OPERATIONS_RESPONSE})

        ops = connector.get_operations("frontend")
        assert ops == ["GET /api/users", "POST /api/orders", "DB query"]

    def test_flatten_span_error_detection(self):
        """Test that error tag is properly detected."""
        connector = JaegerConnector()
        span = {
            "traceID": "t1",
            "spanID": "s1",
            "operationName": "op",
            "startTime": 1700000000000000,
            "duration": 1000,
            "tags": [{"key": "error", "value": True}],
            "references": [],
        }
        process = {"serviceName": "svc", "tags": []}

        flat = connector._flatten_span(span, process)
        assert flat["error"] is True

    def test_flatten_span_status_code(self):
        """Test that http.status_code is extracted."""
        connector = JaegerConnector()
        span = {
            "traceID": "t1",
            "spanID": "s1",
            "operationName": "op",
            "startTime": 1700000000000000,
            "duration": 1000,
            "tags": [{"key": "http.status_code", "value": 500}],
            "references": [],
        }
        process = {"serviceName": "svc", "tags": []}

        flat = connector._flatten_span(span, process)
        assert flat["statusCode"] == 500

    def test_flatten_span_parent_ref(self):
        """Test parent span ID extraction from CHILD_OF reference."""
        connector = JaegerConnector()
        span = {
            "traceID": "t1",
            "spanID": "s2",
            "operationName": "op",
            "startTime": 1700000000000000,
            "duration": 1000,
            "tags": [],
            "references": [{"refType": "CHILD_OF", "spanID": "s1"}],
        }
        process = {"serviceName": "svc", "tags": []}

        flat = connector._flatten_span(span, process)
        assert flat["parentSpanID"] == "s1"

    def test_get_overview(self):
        connector = self._make_connected_connector()
        self._mock_get(connector, {
            "/api/services": self.SERVICES_RESPONSE,
            "/operations": self.OPERATIONS_RESPONSE,
        })

        overview = connector.get_overview()
        assert "test_jaeger" in overview
        assert "observability" in overview


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


class TestParseGraphPattern:
    """Tests for _parse_graph_pattern helper."""

    def test_valid_pattern(self):
        from constat.discovery.relationship_extractor import _parse_graph_pattern
        result = _parse_graph_pattern("(:Person)-[:ACTED_IN]->(:Movie)")
        assert result == ("Person", "ACTED_IN", "Movie")

    def test_pattern_with_surrounding_text(self):
        from constat.discovery.relationship_extractor import _parse_graph_pattern
        result = _parse_graph_pattern("  (:Foo)-[:BAR]->(:Baz)  ")
        assert result == ("Foo", "BAR", "Baz")

    def test_invalid_pattern_returns_none(self):
        from constat.discovery.relationship_extractor import _parse_graph_pattern
        assert _parse_graph_pattern("not a pattern") is None
        assert _parse_graph_pattern("") is None
        assert _parse_graph_pattern("(:A)-(:B)") is None

    def test_underscore_names(self):
        from constat.discovery.relationship_extractor import _parse_graph_pattern
        result = _parse_graph_pattern("(:Node_A)-[:HAS_MANY]->(:Node_B)")
        assert result == ("Node_A", "HAS_MANY", "Node_B")


class TestSeedStructuralRelationships:
    """Tests for seed_structural_relationships (Phase -1)."""

    def _make_schema_manager(self, tables=None, nosql_connections=None):
        """Create a mock schema_manager."""
        sm = Mock()
        sm.metadata_cache = tables or {}
        sm.nosql_connections = nosql_connections or {}
        return sm

    def _make_table_meta(self, name, database, foreign_keys=None):
        """Create a mock TableMetadata."""
        from constat.catalog.schema_manager import TableMetadata, ForeignKey
        return TableMetadata(
            database=database,
            name=name,
            foreign_keys=foreign_keys or [],
        )

    def _make_vector_store(self):
        """Create a mock vector_store."""
        vs = Mock()
        vs.add_entity_relationship = Mock()
        return vs

    def test_fk_seeding(self):
        """Test FK constraints generate REFERENCES relationships."""
        from constat.catalog.schema_manager import ForeignKey
        from constat.discovery.relationship_extractor import seed_structural_relationships

        orders = self._make_table_meta("orders", "sales", foreign_keys=[
            ForeignKey(from_column="customer_id", to_table="customers", to_column="id"),
        ])
        sm = self._make_schema_manager(tables={"sales.orders": orders})
        vs = self._make_vector_store()

        rels = seed_structural_relationships("sess1", sm, vs)

        assert len(rels) == 1
        assert rels[0].subject_name == "orders"
        assert rels[0].verb == "REFERENCES"
        assert rels[0].object_name == "customers"
        assert rels[0].confidence == 0.95
        assert rels[0].verb_category == "association"
        assert "FK:" in rels[0].sentence
        vs.add_entity_relationship.assert_called_once()

    def test_multiple_fks(self):
        """Test multiple FKs from one table."""
        from constat.catalog.schema_manager import ForeignKey
        from constat.discovery.relationship_extractor import seed_structural_relationships

        orders = self._make_table_meta("orders", "sales", foreign_keys=[
            ForeignKey(from_column="customer_id", to_table="customers", to_column="id"),
            ForeignKey(from_column="product_id", to_table="products", to_column="id"),
        ])
        sm = self._make_schema_manager(tables={"sales.orders": orders})
        vs = self._make_vector_store()

        rels = seed_structural_relationships("sess1", sm, vs)
        assert len(rels) == 2
        targets = {r.object_name for r in rels}
        assert targets == {"customers", "products"}

    def test_no_fks_no_nosql(self):
        """Test with no FKs and no NoSQL connections returns empty."""
        from constat.discovery.relationship_extractor import seed_structural_relationships

        table = self._make_table_meta("users", "app")
        sm = self._make_schema_manager(tables={"app.users": table})
        vs = self._make_vector_store()

        rels = seed_structural_relationships("sess1", sm, vs)
        assert rels == []
        vs.add_entity_relationship.assert_not_called()

    def test_neo4j_graph_pattern_seeding(self):
        """Test Neo4j graph patterns create high-confidence relationships."""
        from constat.catalog.nosql.base import NoSQLType
        from constat.discovery.relationship_extractor import seed_structural_relationships

        connector = Mock()
        connector.nosql_type = NoSQLType.GRAPH
        connector.graph_schema.return_value = {
            "labels": ["Person", "Movie"],
            "relationship_types": ["ACTED_IN", "DIRECTED"],
            "patterns": [
                "(:Person)-[:ACTED_IN]->(:Movie)",
                "(:Person)-[:DIRECTED]->(:Movie)",
            ],
        }

        sm = self._make_schema_manager(nosql_connections={"movies": connector})
        vs = self._make_vector_store()

        rels = seed_structural_relationships("sess1", sm, vs)

        assert len(rels) == 2
        assert rels[0].subject_name == "Person"
        assert rels[0].verb == "ACTED_IN"
        assert rels[0].object_name == "Movie"
        assert rels[0].confidence == 0.98
        assert "Graph pattern:" in rels[0].sentence

        assert rels[1].verb == "DIRECTED"

    def test_skips_non_graph_nosql(self):
        """Test that non-graph NoSQL connectors are skipped."""
        from constat.catalog.nosql.base import NoSQLType
        from constat.discovery.relationship_extractor import seed_structural_relationships

        connector = Mock()
        connector.nosql_type = NoSQLType.DOCUMENT

        sm = self._make_schema_manager(nosql_connections={"mongo": connector})
        vs = self._make_vector_store()

        rels = seed_structural_relationships("sess1", sm, vs)
        assert rels == []
        connector.graph_schema.assert_not_called()

    def test_duplicate_handling(self):
        """Test that duplicate key errors are silently skipped."""
        from constat.catalog.schema_manager import ForeignKey
        from constat.discovery.relationship_extractor import seed_structural_relationships

        orders = self._make_table_meta("orders", "sales", foreign_keys=[
            ForeignKey(from_column="customer_id", to_table="customers", to_column="id"),
        ])
        sm = self._make_schema_manager(tables={"sales.orders": orders})
        vs = self._make_vector_store()
        vs.add_entity_relationship.side_effect = Exception("UNIQUE constraint failed")

        rels = seed_structural_relationships("sess1", sm, vs)
        assert rels == []

    def test_on_batch_callback(self):
        """Test on_batch callback is invoked with all relationships."""
        from constat.catalog.schema_manager import ForeignKey
        from constat.discovery.relationship_extractor import seed_structural_relationships

        orders = self._make_table_meta("orders", "sales", foreign_keys=[
            ForeignKey(from_column="customer_id", to_table="customers", to_column="id"),
        ])
        sm = self._make_schema_manager(tables={"sales.orders": orders})
        vs = self._make_vector_store()
        callback = Mock()

        rels = seed_structural_relationships("sess1", sm, vs, on_batch=callback)
        assert len(rels) == 1
        callback.assert_called_once_with(rels)

    def test_combined_fk_and_neo4j(self):
        """Test FK + Neo4j seeding produces combined results."""
        from constat.catalog.nosql.base import NoSQLType
        from constat.catalog.schema_manager import ForeignKey
        from constat.discovery.relationship_extractor import seed_structural_relationships

        orders = self._make_table_meta("orders", "sales", foreign_keys=[
            ForeignKey(from_column="customer_id", to_table="customers", to_column="id"),
        ])

        connector = Mock()
        connector.nosql_type = NoSQLType.GRAPH
        connector.graph_schema.return_value = {
            "labels": ["Person", "Movie"],
            "relationship_types": ["ACTED_IN"],
            "patterns": ["(:Person)-[:ACTED_IN]->(:Movie)"],
        }

        sm = self._make_schema_manager(
            tables={"sales.orders": orders},
            nosql_connections={"movies": connector},
        )
        vs = self._make_vector_store()

        rels = seed_structural_relationships("sess1", sm, vs)
        assert len(rels) == 2
        verbs = {r.verb for r in rels}
        assert verbs == {"REFERENCES", "ACTED_IN"}

    def test_graph_schema_error_graceful(self):
        """Test that graph_schema errors are logged but don't break seeding."""
        from constat.catalog.nosql.base import NoSQLType
        from constat.catalog.schema_manager import ForeignKey
        from constat.discovery.relationship_extractor import seed_structural_relationships

        orders = self._make_table_meta("orders", "sales", foreign_keys=[
            ForeignKey(from_column="customer_id", to_table="customers", to_column="id"),
        ])

        connector = Mock()
        connector.nosql_type = NoSQLType.GRAPH
        connector.graph_schema.side_effect = RuntimeError("Connection lost")

        sm = self._make_schema_manager(
            tables={"sales.orders": orders},
            nosql_connections={"movies": connector},
        )
        vs = self._make_vector_store()

        rels = seed_structural_relationships("sess1", sm, vs)
        # FK still succeeds despite Neo4j failure
        assert len(rels) == 1
        assert rels[0].verb == "REFERENCES"
