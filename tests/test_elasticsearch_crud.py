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

"""CRUD operation tests for ElasticsearchConnector."""

import pytest
from unittest.mock import MagicMock, patch

from constat.catalog.nosql.elasticsearch import ElasticsearchConnector


try:
    import elasticsearch
except ImportError:
    pytest.fail("elasticsearch is required but not installed — run: pip install elasticsearch")


# =============================================================================
# Test Get Collections
# =============================================================================

class TestElasticsearchGetCollections:
    """Test getting collection (index) list."""

    def test_get_collections_success(self):
        """Test listing indices."""
        connector = ElasticsearchConnector()
        connector._client = MagicMock()

        connector._client.indices.get_alias.return_value = {
            "products": {},
            "users": {},
            "orders": {},
            ".kibana": {},
        }

        indices = connector.get_collections()

        assert "products" in indices
        assert "users" in indices
        assert "orders" in indices
        assert ".kibana" not in indices

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
