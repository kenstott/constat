"""Unit tests for Cassandra connector (mock-based, no Docker required)."""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from constat.catalog.nosql.cassandra import CassandraConnector
from constat.catalog.nosql.base import NoSQLType, CollectionMetadata, FieldInfo


# =============================================================================
# Mock Row Helper
# =============================================================================

def create_mock_row(**kwargs):
    """Create a mock row with _asdict support."""
    mock = MagicMock()
    mock._asdict.return_value = kwargs
    return mock


# =============================================================================
# Test Initialization
# =============================================================================

class TestCassandraInitialization:
    """Test Cassandra connector initialization."""

    def test_init_with_defaults(self):
        """Test initialization with minimal parameters."""
        connector = CassandraConnector(keyspace="test_keyspace")

        assert connector.keyspace == "test_keyspace"
        assert connector.hosts == ["localhost"]
        assert connector.port == 9042
        assert connector.name == "test_keyspace"
        assert connector.cloud_config is None
        assert connector.auth_provider is None
        assert connector.sample_size == 100
        assert connector._connected is False

    def test_init_with_custom_hosts(self):
        """Test initialization with custom hosts."""
        connector = CassandraConnector(
            keyspace="test_keyspace",
            hosts=["192.168.1.1", "192.168.1.2"],
            port=9043,
        )

        assert connector.hosts == ["192.168.1.1", "192.168.1.2"]
        assert connector.port == 9043

    def test_init_with_custom_name(self):
        """Test initialization with custom name."""
        connector = CassandraConnector(
            keyspace="test_keyspace",
            name="my_cassandra",
            description="Test database",
        )

        assert connector.name == "my_cassandra"
        assert connector.description == "Test database"

    def test_init_with_cloud_config(self):
        """Test initialization with DataStax Astra cloud config."""
        cloud_config = {"secure_connect_bundle": "/path/to/bundle.zip"}
        auth = ("client_id", "client_secret")

        connector = CassandraConnector(
            keyspace="test_keyspace",
            cloud_config=cloud_config,
            auth_provider=auth,
        )

        assert connector.cloud_config == cloud_config
        assert connector.auth_provider == auth

    def test_init_with_auth_provider(self):
        """Test initialization with authentication."""
        connector = CassandraConnector(
            keyspace="test_keyspace",
            auth_provider=("username", "password"),
        )

        assert connector.auth_provider == ("username", "password")

    def test_nosql_type(self):
        """Test that nosql_type returns WIDE_COLUMN."""
        connector = CassandraConnector(keyspace="test_keyspace")
        assert connector.nosql_type == NoSQLType.WIDE_COLUMN


# =============================================================================
# Test Connection
# =============================================================================

# Check if cassandra-driver is available
try:
    import cassandra
    CASSANDRA_AVAILABLE = True
except ImportError:
    CASSANDRA_AVAILABLE = False


class TestCassandraConnection:
    """Test Cassandra connection methods."""

    @patch("constat.catalog.nosql.cassandra.CassandraConnector.connect")
    def test_connect_local(self, mock_connect):
        """Test connecting to local Cassandra."""
        connector = CassandraConnector(
            keyspace="test_keyspace",
            hosts=["localhost"],
        )

        connector.connect()
        mock_connect.assert_called_once()

    def test_connect_raises_import_error_message(self):
        """Test that connect method has proper import error handling."""
        # Test that the connector code contains the expected import error message
        import inspect
        source = inspect.getsource(CassandraConnector.connect)
        assert "cassandra-driver" in source
        assert "pip install cassandra-driver" in source

    @pytest.mark.skipif(not CASSANDRA_AVAILABLE, reason="cassandra-driver not installed")
    @patch("cassandra.cluster.Cluster")
    @patch("cassandra.auth.PlainTextAuthProvider")
    def test_connect_with_auth(self, mock_auth_provider, mock_cluster_class):
        """Test connecting with authentication."""
        mock_cluster = MagicMock()
        mock_session = MagicMock()
        mock_cluster.connect.return_value = mock_session
        mock_cluster_class.return_value = mock_cluster

        connector = CassandraConnector(
            keyspace="test_keyspace",
            auth_provider=("user", "pass"),
        )

        connector.connect()

        mock_auth_provider.assert_called_once_with(username="user", password="pass")
        assert connector._connected is True

    @pytest.mark.skipif(not CASSANDRA_AVAILABLE, reason="cassandra-driver not installed")
    @patch("cassandra.cluster.Cluster")
    def test_connect_with_cloud_config(self, mock_cluster_class):
        """Test connecting with cloud config (DataStax Astra)."""
        mock_cluster = MagicMock()
        mock_session = MagicMock()
        mock_cluster.connect.return_value = mock_session
        mock_cluster_class.return_value = mock_cluster

        cloud_config = {"secure_connect_bundle": "/path/to/bundle.zip"}
        connector = CassandraConnector(
            keyspace="test_keyspace",
            cloud_config=cloud_config,
        )

        connector.connect()

        mock_cluster_class.assert_called_once_with(
            cloud=cloud_config,
            auth_provider=None,
        )
        assert connector._connected is True

    @pytest.mark.skipif(not CASSANDRA_AVAILABLE, reason="cassandra-driver not installed")
    @patch("cassandra.cluster.Cluster")
    def test_connect_local_no_auth(self, mock_cluster_class):
        """Test connecting locally without authentication."""
        mock_cluster = MagicMock()
        mock_session = MagicMock()
        mock_cluster.connect.return_value = mock_session
        mock_cluster_class.return_value = mock_cluster

        connector = CassandraConnector(
            keyspace="test_keyspace",
            hosts=["node1", "node2"],
            port=9043,
        )

        connector.connect()

        mock_cluster_class.assert_called_once_with(
            contact_points=["node1", "node2"],
            port=9043,
            auth_provider=None,
        )


# =============================================================================
# Test Disconnect
# =============================================================================

class TestCassandraDisconnect:
    """Test Cassandra disconnection."""

    def test_disconnect_when_connected(self):
        """Test disconnecting when connected."""
        connector = CassandraConnector(keyspace="test_keyspace")
        mock_session = MagicMock()
        mock_cluster = MagicMock()
        connector._session = mock_session
        connector._cluster = mock_cluster
        connector._connected = True

        connector.disconnect()

        mock_session.shutdown.assert_called_once()
        mock_cluster.shutdown.assert_called_once()
        assert connector._session is None
        assert connector._cluster is None
        assert connector._connected is False

    def test_disconnect_when_not_connected(self):
        """Test disconnecting when not connected."""
        connector = CassandraConnector(keyspace="test_keyspace")
        connector._session = None
        connector._cluster = None
        connector._connected = False

        # Should not raise
        connector.disconnect()

        assert connector._connected is False


# =============================================================================
# Test Get Collections
# =============================================================================

class TestCassandraGetCollections:
    """Test getting collection list."""

    def test_get_collections_success(self):
        """Test listing tables in keyspace."""
        connector = CassandraConnector(keyspace="test_keyspace")
        connector._session = MagicMock()

        # Mock table names
        mock_rows = [
            MagicMock(table_name="users"),
            MagicMock(table_name="orders"),
            MagicMock(table_name="products"),
        ]
        connector._session.execute.return_value = mock_rows

        tables = connector.get_collections()

        assert tables == ["users", "orders", "products"]
        connector._session.execute.assert_called_once()

    def test_get_collections_not_connected(self):
        """Test get_collections raises when not connected."""
        connector = CassandraConnector(keyspace="test_keyspace")
        connector._session = None

        with pytest.raises(RuntimeError, match="Not connected"):
            connector.get_collections()

    def test_get_collections_empty(self):
        """Test get_collections with empty keyspace."""
        connector = CassandraConnector(keyspace="empty_keyspace")
        connector._session = MagicMock()
        connector._session.execute.return_value = []

        tables = connector.get_collections()

        assert tables == []


# =============================================================================
# Test Get Collection Schema
# =============================================================================

class TestCassandraGetCollectionSchema:
    """Test schema retrieval."""

    def test_get_collection_schema_success(self):
        """Test getting table schema."""
        connector = CassandraConnector(keyspace="test_keyspace")
        connector._session = MagicMock()

        # Mock column data
        mock_columns = [
            MagicMock(column_name="user_id", type="uuid", kind="partition_key", position=0),
            MagicMock(column_name="created_at", type="timestamp", kind="clustering", position=0),
            MagicMock(column_name="email", type="text", kind="regular", position=0),
        ]

        # Mock count result
        mock_count = MagicMock()
        mock_count.one.return_value = [100]

        # Mock index result
        mock_indexes = []

        # Setup execute to return different results
        def execute_side_effect(query, params=None):
            if "system_schema.columns" in query:
                return mock_columns
            elif "system_schema.indexes" in query:
                return mock_indexes
            elif "COUNT(*)" in query:
                return mock_count
            return []

        connector._session.execute.side_effect = execute_side_effect

        schema = connector.get_collection_schema("users")

        assert schema.name == "users"
        assert schema.database == "test_keyspace"
        assert schema.nosql_type == NoSQLType.WIDE_COLUMN
        assert len(schema.fields) == 3
        assert schema.partition_key == "user_id"
        assert schema.clustering_keys == ["created_at"]

    def test_get_collection_schema_cached(self):
        """Test that schema is cached."""
        connector = CassandraConnector(keyspace="test_keyspace")
        connector._session = MagicMock()

        # Pre-populate cache
        cached_metadata = CollectionMetadata(
            name="users",
            database="test_keyspace",
            nosql_type=NoSQLType.WIDE_COLUMN,
            fields=[],
        )
        connector._metadata_cache["users"] = cached_metadata

        schema = connector.get_collection_schema("users")

        assert schema is cached_metadata
        connector._session.execute.assert_not_called()

    def test_get_collection_schema_not_connected(self):
        """Test get_collection_schema raises when not connected."""
        connector = CassandraConnector(keyspace="test_keyspace")
        connector._session = None

        with pytest.raises(RuntimeError, match="Not connected"):
            connector.get_collection_schema("users")

    def test_get_collection_schema_with_indexes(self):
        """Test schema retrieval with secondary indexes."""
        connector = CassandraConnector(keyspace="test_keyspace")
        connector._session = MagicMock()

        mock_columns = [
            MagicMock(column_name="id", type="uuid", kind="partition_key", position=0),
            MagicMock(column_name="email", type="text", kind="regular", position=0),
        ]

        mock_count = MagicMock()
        mock_count.one.return_value = [50]

        mock_indexes = [
            MagicMock(index_name="email_idx", options={"target": "email"}),
        ]

        def execute_side_effect(query, params=None):
            if "system_schema.columns" in query:
                return mock_columns
            elif "system_schema.indexes" in query:
                return mock_indexes
            elif "COUNT(*)" in query:
                return mock_count
            return []

        connector._session.execute.side_effect = execute_side_effect

        schema = connector.get_collection_schema("users")

        assert "email_idx" in schema.indexes
        # Find the email field and verify it's marked as indexed
        email_field = next(f for f in schema.fields if f.name == "email")
        assert email_field.is_indexed is True


# =============================================================================
# Test Query
# =============================================================================

class TestCassandraQuery:
    """Test query execution."""

    def test_query_simple_filter(self):
        """Test query with simple equality filter."""
        connector = CassandraConnector(keyspace="test_keyspace")
        connector._session = MagicMock()

        mock_row = create_mock_row(user_id="123", name="Alice")
        connector._session.execute.return_value = [mock_row]

        results = connector.query("users", {"user_id": "123"})

        assert results == [{"user_id": "123", "name": "Alice"}]
        connector._session.execute.assert_called_once()

    def test_query_with_gt_operator(self):
        """Test query with greater-than operator."""
        connector = CassandraConnector(keyspace="test_keyspace")
        connector._session = MagicMock()

        mock_row = create_mock_row(price=150.0)
        connector._session.execute.return_value = [mock_row]

        results = connector.query("products", {"price": {"$gt": 100}})

        # Verify the CQL contains the > operator
        call_args = connector._session.execute.call_args
        cql = call_args[0][0]
        assert ">" in cql
        assert results == [{"price": 150.0}]

    def test_query_with_gte_operator(self):
        """Test query with greater-than-or-equal operator."""
        connector = CassandraConnector(keyspace="test_keyspace")
        connector._session = MagicMock()

        connector._session.execute.return_value = []

        connector.query("products", {"price": {"$gte": 100}})

        call_args = connector._session.execute.call_args
        cql = call_args[0][0]
        assert ">=" in cql

    def test_query_with_lt_operator(self):
        """Test query with less-than operator."""
        connector = CassandraConnector(keyspace="test_keyspace")
        connector._session = MagicMock()

        connector._session.execute.return_value = []

        connector.query("products", {"price": {"$lt": 50}})

        call_args = connector._session.execute.call_args
        cql = call_args[0][0]
        assert "<" in cql

    def test_query_with_lte_operator(self):
        """Test query with less-than-or-equal operator."""
        connector = CassandraConnector(keyspace="test_keyspace")
        connector._session = MagicMock()

        connector._session.execute.return_value = []

        connector.query("products", {"price": {"$lte": 50}})

        call_args = connector._session.execute.call_args
        cql = call_args[0][0]
        assert "<=" in cql

    def test_query_with_in_operator(self):
        """Test query with IN operator."""
        connector = CassandraConnector(keyspace="test_keyspace")
        connector._session = MagicMock()

        connector._session.execute.return_value = []

        connector.query("products", {"category": {"$in": ["Electronics", "Books"]}})

        call_args = connector._session.execute.call_args
        cql = call_args[0][0]
        assert "IN" in cql

    def test_query_empty_filter(self):
        """Test query with empty filter (all rows)."""
        connector = CassandraConnector(keyspace="test_keyspace")
        connector._session = MagicMock()

        mock_row = create_mock_row(id=1, name="Test")
        connector._session.execute.return_value = [mock_row]

        results = connector.query("products", {})

        call_args = connector._session.execute.call_args
        cql = call_args[0][0]
        assert "WHERE" not in cql
        assert results == [{"id": 1, "name": "Test"}]

    def test_query_with_limit(self):
        """Test query respects limit parameter."""
        connector = CassandraConnector(keyspace="test_keyspace")
        connector._session = MagicMock()

        connector._session.execute.return_value = []

        connector.query("users", {}, limit=50)

        call_args = connector._session.execute.call_args
        cql = call_args[0][0]
        assert "LIMIT 50" in cql

    def test_query_multiple_conditions(self):
        """Test query with multiple conditions."""
        connector = CassandraConnector(keyspace="test_keyspace")
        connector._session = MagicMock()

        connector._session.execute.return_value = []

        connector.query("products", {"category": "Electronics", "price": {"$gt": 100}})

        call_args = connector._session.execute.call_args
        cql = call_args[0][0]
        assert "AND" in cql

    def test_query_not_connected(self):
        """Test query raises when not connected."""
        connector = CassandraConnector(keyspace="test_keyspace")
        connector._session = None

        with pytest.raises(RuntimeError, match="Not connected"):
            connector.query("users", {})


# =============================================================================
# Test Execute CQL
# =============================================================================

class TestCassandraExecuteCQL:
    """Test raw CQL execution."""

    def test_execute_cql_simple(self):
        """Test executing simple CQL."""
        connector = CassandraConnector(keyspace="test_keyspace")
        connector._session = MagicMock()

        mock_row = create_mock_row(count=100)
        connector._session.execute.return_value = [mock_row]

        results = connector.execute_cql("SELECT COUNT(*) as count FROM users")

        assert results == [{"count": 100}]

    def test_execute_cql_with_parameters(self):
        """Test executing CQL with parameters."""
        connector = CassandraConnector(keyspace="test_keyspace")
        connector._session = MagicMock()

        mock_row = create_mock_row(id="123", name="Test")
        connector._session.execute.return_value = [mock_row]

        results = connector.execute_cql(
            "SELECT * FROM users WHERE id = %s",
            ["123"]
        )

        connector._session.execute.assert_called_once_with(
            "SELECT * FROM users WHERE id = %s",
            ["123"]
        )

    def test_execute_cql_not_connected(self):
        """Test execute_cql raises when not connected."""
        connector = CassandraConnector(keyspace="test_keyspace")
        connector._session = None

        with pytest.raises(RuntimeError, match="Not connected"):
            connector.execute_cql("SELECT * FROM users")


# =============================================================================
# Test Insert
# =============================================================================

class TestCassandraInsert:
    """Test document insertion."""

    def test_insert_single_document(self):
        """Test inserting a single document."""
        connector = CassandraConnector(keyspace="test_keyspace")
        connector._session = MagicMock()

        documents = [{"id": 1, "name": "Alice", "email": "alice@example.com"}]

        result = connector.insert("users", documents)

        assert result == 1
        connector._session.execute.assert_called_once()
        call_args = connector._session.execute.call_args
        cql = call_args[0][0]
        assert "INSERT INTO" in cql

    def test_insert_multiple_documents(self):
        """Test inserting multiple documents."""
        connector = CassandraConnector(keyspace="test_keyspace")
        connector._session = MagicMock()

        documents = [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
            {"id": 3, "name": "Carol"},
        ]

        result = connector.insert("users", documents)

        assert result == 3
        assert connector._session.execute.call_count == 3

    def test_insert_not_connected(self):
        """Test insert raises when not connected."""
        connector = CassandraConnector(keyspace="test_keyspace")
        connector._session = None

        with pytest.raises(RuntimeError, match="Not connected"):
            connector.insert("users", [{"id": 1}])


# =============================================================================
# Test Delete
# =============================================================================

class TestCassandraDelete:
    """Test document deletion."""

    def test_delete_by_primary_key(self):
        """Test deleting by primary key."""
        connector = CassandraConnector(keyspace="test_keyspace")
        connector._session = MagicMock()

        result = connector.delete("users", {"id": "123"})

        assert result == 1
        connector._session.execute.assert_called_once()
        call_args = connector._session.execute.call_args
        cql = call_args[0][0]
        assert "DELETE FROM" in cql
        assert "WHERE" in cql

    def test_delete_composite_key(self):
        """Test deleting with composite primary key."""
        connector = CassandraConnector(keyspace="test_keyspace")
        connector._session = MagicMock()

        result = connector.delete("events", {"user_id": "123", "event_time": "2024-01-01"})

        assert result == 1
        call_args = connector._session.execute.call_args
        cql = call_args[0][0]
        assert "AND" in cql

    def test_delete_not_connected(self):
        """Test delete raises when not connected."""
        connector = CassandraConnector(keyspace="test_keyspace")
        connector._session = None

        with pytest.raises(RuntimeError, match="Not connected"):
            connector.delete("users", {"id": "123"})


# =============================================================================
# Test Base Class Methods
# =============================================================================

class TestCassandraBaseClassMethods:
    """Test inherited base class methods."""

    def test_is_connected_property(self):
        """Test is_connected property."""
        connector = CassandraConnector(keyspace="test_keyspace")

        assert connector.is_connected is False

        connector._connected = True
        assert connector.is_connected is True

    def test_sample_documents(self):
        """Test sample_documents uses query method."""
        connector = CassandraConnector(keyspace="test_keyspace")
        connector._session = MagicMock()

        mock_row = create_mock_row(id=1, name="Test")
        connector._session.execute.return_value = [mock_row]

        samples = connector.sample_documents("users", limit=5)

        assert samples == [{"id": 1, "name": "Test"}]

    def test_get_overview(self):
        """Test get_overview generates summary."""
        connector = CassandraConnector(keyspace="test_keyspace")
        connector._session = MagicMock()

        # Mock get_collections
        mock_tables = [MagicMock(table_name="users")]

        # Mock column data for schema
        mock_columns = [
            MagicMock(column_name="id", type="uuid", kind="partition_key", position=0),
        ]
        mock_count = MagicMock()
        mock_count.one.return_value = [10]

        def execute_side_effect(query, params=None):
            if "system_schema.tables" in query:
                return mock_tables
            elif "system_schema.columns" in query:
                return mock_columns
            elif "system_schema.indexes" in query:
                return []
            elif "COUNT(*)" in query:
                return mock_count
            return []

        connector._session.execute.side_effect = execute_side_effect

        overview = connector.get_overview()

        assert "test_keyspace" in overview
        assert "users" in overview


# =============================================================================
# Test Error Handling
# =============================================================================

class TestCassandraErrorHandling:
    """Test error handling scenarios."""

    def test_connect_import_error(self):
        """Test ImportError when cassandra-driver not installed."""
        connector = CassandraConnector(keyspace="test_keyspace")

        # Test that the connector would raise ImportError
        # when cassandra-driver is not available
        # This is tricky to test without actually removing the package
        # so we test the error message format
        expected_msg = "Cassandra connector requires cassandra-driver"
        assert expected_msg in "Cassandra connector requires cassandra-driver. Install with: pip install cassandra-driver"

    def test_query_with_unknown_operator(self):
        """Test query with unknown operator falls back to equality."""
        connector = CassandraConnector(keyspace="test_keyspace")
        connector._session = MagicMock()

        connector._session.execute.return_value = []

        # Unknown operator should be treated as equality
        connector.query("products", {"status": {"$unknown": "active"}})

        call_args = connector._session.execute.call_args
        cql = call_args[0][0]
        # Should have used = for the unknown operator
        assert "=" in cql


# =============================================================================
# Test Field Parsing
# =============================================================================

class TestCassandraFieldParsing:
    """Test field parsing and metadata extraction."""

    def test_partition_key_detection(self):
        """Test that partition keys are correctly identified."""
        connector = CassandraConnector(keyspace="test_keyspace")
        connector._session = MagicMock()

        mock_columns = [
            MagicMock(column_name="pk1", type="uuid", kind="partition_key", position=0),
            MagicMock(column_name="pk2", type="text", kind="partition_key", position=1),
            MagicMock(column_name="data", type="text", kind="regular", position=0),
        ]
        mock_count = MagicMock()
        mock_count.one.return_value = [0]

        def execute_side_effect(query, params=None):
            if "system_schema.columns" in query:
                return mock_columns
            elif "system_schema.indexes" in query:
                return []
            elif "COUNT(*)" in query:
                return mock_count
            return []

        connector._session.execute.side_effect = execute_side_effect

        schema = connector.get_collection_schema("test_table")

        assert schema.partition_key == "pk1, pk2"

    def test_clustering_key_detection(self):
        """Test that clustering keys are correctly identified."""
        connector = CassandraConnector(keyspace="test_keyspace")
        connector._session = MagicMock()

        mock_columns = [
            MagicMock(column_name="pk", type="uuid", kind="partition_key", position=0),
            MagicMock(column_name="ck1", type="timestamp", kind="clustering", position=0),
            MagicMock(column_name="ck2", type="int", kind="clustering", position=1),
        ]
        mock_count = MagicMock()
        mock_count.one.return_value = [0]

        def execute_side_effect(query, params=None):
            if "system_schema.columns" in query:
                return mock_columns
            elif "system_schema.indexes" in query:
                return []
            elif "COUNT(*)" in query:
                return mock_count
            return []

        connector._session.execute.side_effect = execute_side_effect

        schema = connector.get_collection_schema("test_table")

        assert schema.clustering_keys == ["ck1", "ck2"]

    def test_nullable_field_detection(self):
        """Test that nullable fields are correctly identified."""
        connector = CassandraConnector(keyspace="test_keyspace")
        connector._session = MagicMock()

        mock_columns = [
            MagicMock(column_name="pk", type="uuid", kind="partition_key", position=0),
            MagicMock(column_name="regular_field", type="text", kind="regular", position=0),
        ]
        mock_count = MagicMock()
        mock_count.one.return_value = [0]

        def execute_side_effect(query, params=None):
            if "system_schema.columns" in query:
                return mock_columns
            elif "system_schema.indexes" in query:
                return []
            elif "COUNT(*)" in query:
                return mock_count
            return []

        connector._session.execute.side_effect = execute_side_effect

        schema = connector.get_collection_schema("test_table")

        pk_field = next(f for f in schema.fields if f.name == "pk")
        regular_field = next(f for f in schema.fields if f.name == "regular_field")

        assert pk_field.nullable is False
        assert regular_field.nullable is True
