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

"""Query, search, and aggregation tests for ElasticsearchConnector."""

import pytest
from unittest.mock import MagicMock

from constat.catalog.nosql.elasticsearch import ElasticsearchConnector


try:
    import elasticsearch
except ImportError:
    pytest.fail("elasticsearch is required but not installed — run: pip install elasticsearch")


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
        assert call_kwargs["size"] == 0

    def test_aggregate_not_connected(self):
        """Test aggregate raises when not connected."""
        connector = ElasticsearchConnector()
        connector._client = None

        with pytest.raises(RuntimeError, match="Not connected"):
            connector.aggregate("products", {})
