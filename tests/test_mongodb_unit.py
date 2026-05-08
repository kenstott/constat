# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Unit tests for MongoDB connector - no Docker required.

These tests use mocking to test MongoDB connector behavior without
requiring a running MongoDB instance.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from dataclasses import dataclass

# Check for optional dependencies
try:
    import pymongo
    from pymongo.errors import (
        ServerSelectionTimeoutError,
        OperationFailure,
        ConnectionFailure,
    )
    HAS_PYMONGO = True
except ImportError:
    HAS_PYMONGO = False
    # Create mock exceptions for tests that don't need pymongo
    class ServerSelectionTimeoutError(Exception):
        pass
    class OperationFailure(Exception):
        pass
    class ConnectionFailure(Exception):
        pass

from constat.catalog.nosql import (
    MongoDBConnector,
    NoSQLType,
    CollectionMetadata,
    FieldInfo,
)


class TestMongoDBConnectionErrors:
    """Tests for MongoDB connection error handling."""

    @pytest.mark.skipif(not HAS_PYMONGO, reason="pymongo not installed")
    @patch("pymongo.MongoClient")
    def test_connect_timeout_raises(self, mock_client_class):
        """Connection timeout should propagate as error."""
        mock_client_class.side_effect = ServerSelectionTimeoutError(
            "localhost:27017: [Errno 61] Connection refused"
        )

        connector = MongoDBConnector(
            uri="mongodb://localhost:27017",
            database="test",
        )

        with pytest.raises(ServerSelectionTimeoutError):
            connector.connect()

    @pytest.mark.skipif(not HAS_PYMONGO, reason="pymongo not installed")
    @patch("pymongo.MongoClient")
    def test_connect_auth_failure_on_db_access(self, mock_client_class):
        """Authentication failure on database access should raise."""
        mock_client = MagicMock()
        mock_client.__getitem__.side_effect = OperationFailure(
            "Authentication failed"
        )
        mock_client_class.return_value = mock_client

        connector = MongoDBConnector(
            uri="mongodb://localhost:27017",
            database="test",
        )

        with pytest.raises(OperationFailure, match="Authentication"):
            connector.connect()

    def test_db_property_raises_when_not_connected(self):
        """Accessing db property before connect() should raise RuntimeError."""
        connector = MongoDBConnector(
            uri="mongodb://localhost:27017",
            database="test",
        )

        with pytest.raises(RuntimeError, match="Not connected"):
            _ = connector.db

    def test_subscript_access_before_connect_raises(self):
        """Using connector['collection'] before connect should raise."""
        connector = MongoDBConnector(
            uri="mongodb://localhost:27017",
            database="test",
        )

        with pytest.raises(RuntimeError, match="Not connected"):
            _ = connector["users"]


class TestMongoDBQueryMethods:
    """Tests for MongoDB query execution."""

    def _setup_mock_connector(self, mock_client_class):
        """Helper to create a connected mock connector."""
        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_client.__getitem__.return_value = mock_db
        mock_client_class.return_value = mock_client

        connector = MongoDBConnector(
            uri="mongodb://localhost:27017",
            database="test",
        )
        connector.connect()

        return connector, mock_client, mock_db

    @pytest.mark.skipif(not HAS_PYMONGO, reason="pymongo not installed")
    @patch("pymongo.MongoClient")
    def test_query_returns_empty_list_for_no_matches(self, mock_client_class):
        """Query with no matches returns empty list, not None."""
        connector, mock_client, mock_db = self._setup_mock_connector(mock_client_class)

        mock_coll = MagicMock()
        mock_db.__getitem__.return_value = mock_coll
        mock_cursor = MagicMock()
        mock_coll.find.return_value = mock_cursor
        mock_cursor.limit.return_value = iter([])

        result = connector.query("users", {"name": "nonexistent"})

        assert result == []
        assert isinstance(result, list)

    @pytest.mark.skipif(not HAS_PYMONGO, reason="pymongo not installed")
    @patch("pymongo.MongoClient")
    def test_query_converts_objectid_to_string(self, mock_client_class):
        """ObjectId fields should be converted to strings."""
        from bson import ObjectId

        connector, mock_client, mock_db = self._setup_mock_connector(mock_client_class)

        doc_id = ObjectId()
        mock_coll = MagicMock()
        mock_db.__getitem__.return_value = mock_coll
        mock_cursor = MagicMock()
        mock_coll.find.return_value = mock_cursor
        mock_cursor.limit.return_value = iter([
            {"_id": doc_id, "name": "Alice"}
        ])

        result = connector.query("users", {})

        assert result[0]["_id"] == str(doc_id)
        assert isinstance(result[0]["_id"], str)

    @pytest.mark.skipif(not HAS_PYMONGO, reason="pymongo not installed")
    @patch("pymongo.MongoClient")
    def test_query_limit_is_respected(self, mock_client_class):
        """Limit parameter should be passed to MongoDB."""
        connector, mock_client, mock_db = self._setup_mock_connector(mock_client_class)

        mock_coll = MagicMock()
        mock_db.__getitem__.return_value = mock_coll
        mock_cursor = MagicMock()
        mock_coll.find.return_value = mock_cursor
        mock_cursor.limit.return_value = iter([])

        connector.query("users", {}, limit=50)

        mock_cursor.limit.assert_called_with(50)

    @pytest.mark.skipif(not HAS_PYMONGO, reason="pymongo not installed")
    @patch("pymongo.MongoClient")
    def test_query_default_limit_is_100(self, mock_client_class):
        """Default limit should be 100."""
        connector, mock_client, mock_db = self._setup_mock_connector(mock_client_class)

        mock_coll = MagicMock()
        mock_db.__getitem__.return_value = mock_coll
        mock_cursor = MagicMock()
        mock_coll.find.return_value = mock_cursor
        mock_cursor.limit.return_value = iter([])

        connector.query("users", {})

        mock_cursor.limit.assert_called_with(100)

    @pytest.mark.skipif(not HAS_PYMONGO, reason="pymongo not installed")
    @patch("pymongo.MongoClient")
    def test_query_preserves_other_fields(self, mock_client_class):
        """Query should preserve all document fields."""
        connector, mock_client, mock_db = self._setup_mock_connector(mock_client_class)

        mock_coll = MagicMock()
        mock_db.__getitem__.return_value = mock_coll
        mock_cursor = MagicMock()
        mock_coll.find.return_value = mock_cursor
        mock_cursor.limit.return_value = iter([
            {"_id": "abc", "name": "Alice", "age": 30, "active": True, "tags": ["a", "b"]}
        ])

        result = connector.query("users", {})

        assert result[0]["name"] == "Alice"
        assert result[0]["age"] == 30
        assert result[0]["active"] is True
        assert result[0]["tags"] == ["a", "b"]


class TestMongoDBAggregate:
    """Tests for MongoDB aggregation pipeline."""

    def _setup_mock_connector(self, mock_client_class):
        """Helper to create a connected mock connector."""
        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_client.__getitem__.return_value = mock_db
        mock_client_class.return_value = mock_client

        connector = MongoDBConnector(
            uri="mongodb://localhost:27017",
            database="test",
        )
        connector.connect()

        return connector, mock_client, mock_db

    @pytest.mark.skipif(not HAS_PYMONGO, reason="pymongo not installed")
    @patch("pymongo.MongoClient")
    def test_aggregate_returns_list(self, mock_client_class):
        """Aggregation should return a list."""
        connector, mock_client, mock_db = self._setup_mock_connector(mock_client_class)

        mock_coll = MagicMock()
        mock_db.__getitem__.return_value = mock_coll
        mock_coll.aggregate.return_value = iter([{"_id": "cat1", "count": 5}])

        result = connector.aggregate("products", [{"$group": {"_id": "$category", "count": {"$sum": 1}}}])

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["count"] == 5

    @pytest.mark.skipif(not HAS_PYMONGO, reason="pymongo not installed")
    @patch("pymongo.MongoClient")
    def test_aggregate_converts_objectid(self, mock_client_class):
        """Aggregation results should convert ObjectId to string."""
        from bson import ObjectId

        connector, mock_client, mock_db = self._setup_mock_connector(mock_client_class)

        group_id = ObjectId()
        mock_coll = MagicMock()
        mock_db.__getitem__.return_value = mock_coll
        mock_coll.aggregate.return_value = iter([
            {"_id": group_id, "total": 100}
        ])

        result = connector.aggregate("orders", [])

        assert isinstance(result[0]["_id"], str)
        assert result[0]["_id"] == str(group_id)

    @pytest.mark.skipif(not HAS_PYMONGO, reason="pymongo not installed")
    @patch("pymongo.MongoClient")
    def test_aggregate_empty_result(self, mock_client_class):
        """Empty aggregation result should return empty list."""
        connector, mock_client, mock_db = self._setup_mock_connector(mock_client_class)

        mock_coll = MagicMock()
        mock_db.__getitem__.return_value = mock_coll
        mock_coll.aggregate.return_value = iter([])

        result = connector.aggregate("empty_coll", [{"$match": {"nonexistent": True}}])

        assert result == []

    @pytest.mark.skipif(not HAS_PYMONGO, reason="pymongo not installed")
    @patch("pymongo.MongoClient")
    def test_aggregate_not_connected_raises(self, mock_client_class):
        """Aggregation when not connected should raise RuntimeError."""
        connector = MongoDBConnector(
            uri="mongodb://localhost:27017",
            database="test",
        )

        with pytest.raises(RuntimeError, match="Not connected"):
            connector.aggregate("users", [])


class TestMongoDBInsertUpdateDelete:
    """Tests for MongoDB write operations."""

    def _setup_mock_connector(self, mock_client_class):
        """Helper to create a connected mock connector."""
        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_client.__getitem__.return_value = mock_db
        mock_client_class.return_value = mock_client

        connector = MongoDBConnector(
            uri="mongodb://localhost:27017",
            database="test",
        )
        connector.connect()

        return connector, mock_client, mock_db

    @pytest.mark.skipif(not HAS_PYMONGO, reason="pymongo not installed")
    @patch("pymongo.MongoClient")
    def test_insert_single_document(self, mock_client_class):
        """Insert single document should use insert_one."""
        from bson import ObjectId

        connector, mock_client, mock_db = self._setup_mock_connector(mock_client_class)

        inserted_id = ObjectId()
        mock_coll = MagicMock()
        mock_db.__getitem__.return_value = mock_coll
        mock_result = MagicMock()
        mock_result.inserted_id = inserted_id
        mock_coll.insert_one.return_value = mock_result

        result = connector.insert("users", [{"name": "Alice"}])

        mock_coll.insert_one.assert_called_once_with({"name": "Alice"})
        assert result == [str(inserted_id)]

    @pytest.mark.skipif(not HAS_PYMONGO, reason="pymongo not installed")
    @patch("pymongo.MongoClient")
    def test_insert_multiple_documents(self, mock_client_class):
        """Insert multiple documents should use insert_many."""
        from bson import ObjectId

        connector, mock_client, mock_db = self._setup_mock_connector(mock_client_class)

        inserted_ids = [ObjectId(), ObjectId()]
        mock_coll = MagicMock()
        mock_db.__getitem__.return_value = mock_coll
        mock_result = MagicMock()
        mock_result.inserted_ids = inserted_ids
        mock_coll.insert_many.return_value = mock_result

        docs = [{"name": "Alice"}, {"name": "Bob"}]
        result = connector.insert("users", docs)

        mock_coll.insert_many.assert_called_once_with(docs)
        assert result == [str(id) for id in inserted_ids]

    @pytest.mark.skipif(not HAS_PYMONGO, reason="pymongo not installed")
    @patch("pymongo.MongoClient")
    def test_update_returns_modified_count(self, mock_client_class):
        """Update should return count of modified documents."""
        connector, mock_client, mock_db = self._setup_mock_connector(mock_client_class)

        mock_coll = MagicMock()
        mock_db.__getitem__.return_value = mock_coll
        mock_result = MagicMock()
        mock_result.modified_count = 5
        mock_coll.update_many.return_value = mock_result

        result = connector.update("users", {"active": False}, {"$set": {"active": True}})

        assert result == 5

    @pytest.mark.skipif(not HAS_PYMONGO, reason="pymongo not installed")
    @patch("pymongo.MongoClient")
    def test_delete_returns_deleted_count(self, mock_client_class):
        """Delete should return count of deleted documents."""
        connector, mock_client, mock_db = self._setup_mock_connector(mock_client_class)

        mock_coll = MagicMock()
        mock_db.__getitem__.return_value = mock_coll
        mock_result = MagicMock()
        mock_result.deleted_count = 3
        mock_coll.delete_many.return_value = mock_result

        result = connector.delete("users", {"status": "inactive"})

        assert result == 3

    @pytest.mark.skipif(not HAS_PYMONGO, reason="pymongo not installed")
    @patch("pymongo.MongoClient")
    def test_insert_not_connected_raises(self, mock_client_class):
        """Insert when not connected should raise RuntimeError."""
        connector = MongoDBConnector(
            uri="mongodb://localhost:27017",
            database="test",
        )

        with pytest.raises(RuntimeError, match="Not connected"):
            connector.insert("users", [{"name": "Alice"}])

    @pytest.mark.skipif(not HAS_PYMONGO, reason="pymongo not installed")
    @patch("pymongo.MongoClient")
    def test_update_not_connected_raises(self, mock_client_class):
        """Update when not connected should raise RuntimeError."""
        connector = MongoDBConnector(
            uri="mongodb://localhost:27017",
            database="test",
        )

        with pytest.raises(RuntimeError, match="Not connected"):
            connector.update("users", {}, {"$set": {"x": 1}})

    @pytest.mark.skipif(not HAS_PYMONGO, reason="pymongo not installed")
    @patch("pymongo.MongoClient")
    def test_delete_not_connected_raises(self, mock_client_class):
        """Delete when not connected should raise RuntimeError."""
        connector = MongoDBConnector(
            uri="mongodb://localhost:27017",
            database="test",
        )

        with pytest.raises(RuntimeError, match="Not connected"):
            connector.delete("users", {})


class TestMongoDBSchemaInference:
    """Tests for schema inference from document samples."""

    def test_empty_samples_returns_no_fields(self):
        """Empty sample list should return metadata with no fields."""
        connector = MongoDBConnector(
            uri="mongodb://localhost:27017",
            database="test",
        )
        metadata = connector.infer_schema_from_samples("empty_coll", [])

        assert metadata.name == "empty_coll"
        assert len(metadata.fields) == 0
        assert metadata.document_count == 0

    def test_single_document_sample(self):
        """Single document should produce correct field list."""
        connector = MongoDBConnector(
            uri="mongodb://localhost:27017",
            database="test",
        )
        samples = [{"name": "Alice", "age": 30}]
        metadata = connector.infer_schema_from_samples("users", samples)

        field_names = [f.name for f in metadata.fields]
        assert "name" in field_names
        assert "age" in field_names
        assert len(metadata.fields) == 2

    def test_deeply_nested_objects(self):
        """Deeply nested objects should produce dotted field paths."""
        connector = MongoDBConnector(
            uri="mongodb://localhost:27017",
            database="test",
        )
        samples = [
            {"address": {"city": {"name": "NYC", "zip": "10001"}}}
        ]
        metadata = connector.infer_schema_from_samples("nested", samples)

        field_names = [f.name for f in metadata.fields]
        assert "address.city.name" in field_names
        assert "address.city.zip" in field_names

    def test_mixed_types_across_documents(self):
        """Same field with different types should be 'mixed'."""
        connector = MongoDBConnector(
            uri="mongodb://localhost:27017",
            database="test",
        )
        samples = [
            {"value": 123},
            {"value": "string"},
            {"value": True},
        ]
        metadata = connector.infer_schema_from_samples("mixed", samples)

        field = next(f for f in metadata.fields if f.name == "value")
        assert field.data_type == "mixed"

    def test_all_null_values(self):
        """Field that is always null should have null type."""
        connector = MongoDBConnector(
            uri="mongodb://localhost:27017",
            database="test",
        )
        samples = [
            {"field": None},
            {"field": None},
        ]
        metadata = connector.infer_schema_from_samples("nulls", samples)

        field = next(f for f in metadata.fields if f.name == "field")
        assert field.data_type == "null"
        assert field.nullable is True

    def test_sparse_field_detection(self):
        """Field that exists in only some documents should be nullable."""
        connector = MongoDBConnector(
            uri="mongodb://localhost:27017",
            database="test",
        )
        samples = [
            {"name": "Alice", "optional": "present"},
            {"name": "Bob"},  # No 'optional' field
            {"name": "Carol", "optional": None},
        ]
        metadata = connector.infer_schema_from_samples("sparse", samples)

        # 'optional' should be marked nullable
        field = next((f for f in metadata.fields if f.name == "optional"), None)
        assert field is not None
        assert field.nullable is True

    def test_array_field_type(self):
        """Array fields should be typed as 'array'."""
        connector = MongoDBConnector(
            uri="mongodb://localhost:27017",
            database="test",
        )
        samples = [
            {"tags": ["python", "mongodb"]},
            {"tags": ["java"]},
        ]
        metadata = connector.infer_schema_from_samples("tagged", samples)

        field = next(f for f in metadata.fields if f.name == "tags")
        assert field.data_type == "array"

    def test_object_field_type(self):
        """Object fields should be typed as 'object'."""
        connector = MongoDBConnector(
            uri="mongodb://localhost:27017",
            database="test",
        )
        samples = [
            {"metadata": {"created": "2024-01-01"}},
            {"metadata": {"created": "2024-01-02", "updated": "2024-01-03"}},
        ]
        metadata = connector.infer_schema_from_samples("docs", samples)

        # The top-level 'metadata' field should have nested fields
        field_names = [f.name for f in metadata.fields]
        assert "metadata.created" in field_names

    def test_boolean_field_type(self):
        """Boolean fields should be typed correctly."""
        connector = MongoDBConnector(
            uri="mongodb://localhost:27017",
            database="test",
        )
        samples = [
            {"active": True},
            {"active": False},
        ]
        metadata = connector.infer_schema_from_samples("users", samples)

        field = next(f for f in metadata.fields if f.name == "active")
        assert field.data_type == "boolean"

    def test_integer_vs_float_distinction(self):
        """Integer and float types should be distinguished."""
        connector = MongoDBConnector(
            uri="mongodb://localhost:27017",
            database="test",
        )
        samples = [
            {"count": 10, "price": 19.99},
            {"count": 20, "price": 29.99},
        ]
        metadata = connector.infer_schema_from_samples("products", samples)

        field_map = {f.name: f for f in metadata.fields}
        assert field_map["count"].data_type == "integer"
        assert field_map["price"].data_type == "float"


class TestMongoDBSubscriptAccess:
    """Tests for db['collection'] style access."""

    @pytest.mark.skipif(not HAS_PYMONGO, reason="pymongo not installed")
    @patch("pymongo.MongoClient")
    def test_subscript_access_returns_collection(self, mock_client_class):
        """connector['collection'] should return pymongo collection."""
        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_coll = MagicMock()
        mock_client.__getitem__.return_value = mock_db
        mock_db.__getitem__.return_value = mock_coll
        mock_client_class.return_value = mock_client

        connector = MongoDBConnector(
            uri="mongodb://localhost:27017",
            database="test",
        )
        connector.connect()

        result = connector["users"]
        mock_db.__getitem__.assert_called_with("users")
        assert result is mock_coll

    @pytest.mark.skipif(not HAS_PYMONGO, reason="pymongo not installed")
    @patch("pymongo.MongoClient")
    def test_subscript_access_different_collections(self, mock_client_class):
        """Multiple subscript accesses should work correctly."""
        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_client.__getitem__.return_value = mock_db
        mock_client_class.return_value = mock_client

        connector = MongoDBConnector(
            uri="mongodb://localhost:27017",
            database="test",
        )
        connector.connect()

        _ = connector["users"]
        _ = connector["orders"]
        _ = connector["products"]

        assert mock_db.__getitem__.call_count == 3


class TestMongoDBCollectionSchema:
    """Tests for get_collection_schema method."""

    @pytest.mark.skipif(not HAS_PYMONGO, reason="pymongo not installed")
    @patch("pymongo.MongoClient")
    def test_get_collection_schema_caches_result(self, mock_client_class):
        """Schema should be cached after first call."""
        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_coll = MagicMock()
        mock_client.__getitem__.return_value = mock_db
        mock_db.__getitem__.return_value = mock_coll
        mock_coll.find.return_value.limit.return_value = [
            {"name": "Alice", "age": 30}
        ]
        mock_coll.estimated_document_count.return_value = 100
        mock_coll.list_indexes.return_value = []
        mock_client_class.return_value = mock_client

        connector = MongoDBConnector(
            uri="mongodb://localhost:27017",
            database="test",
        )
        connector.connect()

        # First call
        schema1 = connector.get_collection_schema("users")
        # Second call should use cache
        schema2 = connector.get_collection_schema("users")

        assert schema1 is schema2
        # find() should only be called once due to caching
        assert mock_coll.find.call_count == 1

    @pytest.mark.skipif(not HAS_PYMONGO, reason="pymongo not installed")
    @patch("pymongo.MongoClient")
    def test_get_collection_schema_includes_document_count(self, mock_client_class):
        """Schema should include document count."""
        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_coll = MagicMock()
        mock_client.__getitem__.return_value = mock_db
        mock_db.__getitem__.return_value = mock_coll
        mock_coll.find.return_value.limit.return_value = [{"name": "Alice"}]
        mock_coll.estimated_document_count.return_value = 1000
        mock_coll.list_indexes.return_value = []
        mock_client_class.return_value = mock_client

        connector = MongoDBConnector(
            uri="mongodb://localhost:27017",
            database="test",
        )
        connector.connect()

        schema = connector.get_collection_schema("users")

        assert schema.document_count == 1000

    @pytest.mark.skipif(not HAS_PYMONGO, reason="pymongo not installed")
    @patch("pymongo.MongoClient")
    def test_get_collection_schema_includes_indexes(self, mock_client_class):
        """Schema should include index information."""
        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_coll = MagicMock()
        mock_client.__getitem__.return_value = mock_db
        mock_db.__getitem__.return_value = mock_coll
        mock_coll.find.return_value.limit.return_value = [{"email": "test@example.com"}]
        mock_coll.estimated_document_count.return_value = 100
        mock_coll.list_indexes.return_value = [
            {"name": "_id_", "key": {"_id": 1}},
            {"name": "email_1", "key": {"email": 1}, "unique": True},
        ]
        mock_client_class.return_value = mock_client

        connector = MongoDBConnector(
            uri="mongodb://localhost:27017",
            database="test",
        )
        connector.connect()

        schema = connector.get_collection_schema("users")

        # Should exclude _id_ index but include email_1
        assert "email_1" in schema.indexes

    @pytest.mark.skipif(not HAS_PYMONGO, reason="pymongo not installed")
    @patch("pymongo.MongoClient")
    def test_get_collection_schema_marks_indexed_fields(self, mock_client_class):
        """Fields with indexes should be marked as indexed."""
        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_coll = MagicMock()
        mock_client.__getitem__.return_value = mock_db
        mock_db.__getitem__.return_value = mock_coll
        mock_coll.find.return_value.limit.return_value = [{"email": "test@example.com", "name": "Test"}]
        mock_coll.estimated_document_count.return_value = 100
        mock_coll.list_indexes.return_value = [
            {"name": "_id_", "key": {"_id": 1}},
            {"name": "email_1", "key": {"email": 1}, "unique": True},
        ]
        mock_client_class.return_value = mock_client

        connector = MongoDBConnector(
            uri="mongodb://localhost:27017",
            database="test",
        )
        connector.connect()

        schema = connector.get_collection_schema("users")

        field_map = {f.name: f for f in schema.fields}
        assert field_map["email"].is_indexed is True
        assert field_map["email"].is_unique is True
        assert field_map["name"].is_indexed is False


class TestMongoDBDisconnect:
    """Tests for MongoDB disconnect behavior."""

    @pytest.mark.skipif(not HAS_PYMONGO, reason="pymongo not installed")
    @patch("pymongo.MongoClient")
    def test_disconnect_clears_state(self, mock_client_class):
        """Disconnect should clear client and db references."""
        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_client.__getitem__.return_value = mock_db
        mock_client_class.return_value = mock_client

        connector = MongoDBConnector(
            uri="mongodb://localhost:27017",
            database="test",
        )
        connector.connect()
        assert connector.is_connected

        connector.disconnect()

        assert not connector.is_connected
        assert connector._client is None
        assert connector._db is None

    @pytest.mark.skipif(not HAS_PYMONGO, reason="pymongo not installed")
    @patch("pymongo.MongoClient")
    def test_disconnect_calls_close(self, mock_client_class):
        """Disconnect should call close on the client."""
        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_client.__getitem__.return_value = mock_db
        mock_client_class.return_value = mock_client

        connector = MongoDBConnector(
            uri="mongodb://localhost:27017",
            database="test",
        )
        connector.connect()
        connector.disconnect()

        mock_client.close.assert_called_once()

    def test_disconnect_when_not_connected_is_safe(self):
        """Disconnect when not connected should not raise."""
        connector = MongoDBConnector(
            uri="mongodb://localhost:27017",
            database="test",
        )

        # Should not raise
        connector.disconnect()
        assert not connector.is_connected


class TestMongoDBMetadata:
    """Tests for MongoDB connector metadata."""

    def test_nosql_type_is_document(self):
        """MongoDB connector should report DOCUMENT type."""
        connector = MongoDBConnector(
            uri="mongodb://localhost:27017",
            database="test",
        )
        assert connector.nosql_type == NoSQLType.DOCUMENT

    def test_name_defaults_to_database(self):
        """Default name should be the database name."""
        connector = MongoDBConnector(
            uri="mongodb://localhost:27017",
            database="mydb",
        )
        assert connector.name == "mydb"

    def test_custom_name(self):
        """Custom name should override default."""
        connector = MongoDBConnector(
            uri="mongodb://localhost:27017",
            database="mydb",
            name="production_mongo",
        )
        assert connector.name == "production_mongo"

    def test_sample_size_default(self):
        """Default sample size should be 100."""
        connector = MongoDBConnector(
            uri="mongodb://localhost:27017",
            database="test",
        )
        assert connector.sample_size == 100

    def test_custom_sample_size(self):
        """Custom sample size should be respected."""
        connector = MongoDBConnector(
            uri="mongodb://localhost:27017",
            database="test",
            sample_size=500,
        )
        assert connector.sample_size == 500
