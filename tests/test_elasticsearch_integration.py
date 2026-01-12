"""Integration tests for Elasticsearch connector (requires Docker)."""

import pytest

from constat.catalog.nosql.elasticsearch import ElasticsearchConnector
from constat.catalog.nosql.base import NoSQLType


# =============================================================================
# Basic Connection Tests
# =============================================================================

@pytest.mark.requires_elasticsearch
@pytest.mark.integration
class TestElasticsearchConnection:
    """Test Elasticsearch connection with real Docker container."""

    def test_connect_to_elasticsearch(self, elasticsearch_container):
        """Test connecting to Elasticsearch container."""
        connector = ElasticsearchConnector(
            hosts=[elasticsearch_container["url"]],
            name="test_es",
        )

        connector.connect()
        assert connector.is_connected is True

        connector.disconnect()
        assert connector.is_connected is False

    def test_connect_and_list_indices(self, elasticsearch_container):
        """Test listing indices from cluster."""
        connector = ElasticsearchConnector(
            hosts=[elasticsearch_container["url"]],
        )

        connector.connect()
        indices = connector.get_collections()

        # Should have some indices (even if empty initially)
        assert isinstance(indices, list)

        connector.disconnect()


# =============================================================================
# E-commerce Data Integration Tests
# =============================================================================

@pytest.mark.requires_elasticsearch
@pytest.mark.integration
class TestElasticsearchEcommerceData:
    """Test Elasticsearch with e-commerce sample data."""

    def test_list_ecommerce_indices(self, elasticsearch_with_ecommerce_data):
        """Test listing indices in e-commerce cluster."""
        connector = ElasticsearchConnector(
            hosts=[elasticsearch_with_ecommerce_data["url"]],
        )

        connector.connect()
        indices = connector.get_collections()

        assert "products" in indices
        assert "customers" in indices
        assert "orders" in indices
        assert "reviews" in indices

        connector.disconnect()

    def test_get_products_schema(self, elasticsearch_with_ecommerce_data):
        """Test getting schema for products index."""
        connector = ElasticsearchConnector(
            hosts=[elasticsearch_with_ecommerce_data["url"]],
        )

        connector.connect()
        schema = connector.get_collection_schema("products")

        assert schema.name == "products"
        assert schema.nosql_type == NoSQLType.SEARCH
        assert schema.document_count == 8

        field_names = [f.name for f in schema.fields]
        assert "name" in field_names
        assert "price" in field_names
        assert "category" in field_names
        assert "brand" in field_names

        connector.disconnect()

    def test_get_customers_schema(self, elasticsearch_with_ecommerce_data):
        """Test getting schema for customers index."""
        connector = ElasticsearchConnector(
            hosts=[elasticsearch_with_ecommerce_data["url"]],
        )

        connector.connect()
        schema = connector.get_collection_schema("customers")

        assert schema.name == "customers"
        assert schema.document_count == 5

        field_names = [f.name for f in schema.fields]
        assert "name" in field_names
        assert "email" in field_names
        assert "membership" in field_names

        connector.disconnect()

    def test_query_all_products(self, elasticsearch_with_ecommerce_data):
        """Test querying all products."""
        connector = ElasticsearchConnector(
            hosts=[elasticsearch_with_ecommerce_data["url"]],
        )

        connector.connect()
        products = connector.query("products", {}, limit=100)

        assert len(products) == 8
        product_names = [p["name"] for p in products]
        assert "MacBook Pro 16" in product_names
        assert "Kindle Paperwhite" in product_names

        connector.disconnect()

    def test_query_with_match(self, elasticsearch_with_ecommerce_data):
        """Test query with match filter."""
        connector = ElasticsearchConnector(
            hosts=[elasticsearch_with_ecommerce_data["url"]],
        )

        connector.connect()
        products = connector.query(
            "products",
            {"match": {"category": "Electronics"}}
        )

        assert len(products) >= 4
        for product in products:
            assert product["category"] == "Electronics"

        connector.disconnect()

    def test_query_with_term(self, elasticsearch_with_ecommerce_data):
        """Test query with term filter."""
        connector = ElasticsearchConnector(
            hosts=[elasticsearch_with_ecommerce_data["url"]],
        )

        connector.connect()
        products = connector.query(
            "products",
            {"term": {"brand": "Apple"}}
        )

        assert len(products) == 1
        assert products[0]["name"] == "MacBook Pro 16"

        connector.disconnect()


# =============================================================================
# Search Tests
# =============================================================================

@pytest.mark.requires_elasticsearch
@pytest.mark.integration
class TestElasticsearchSearch:
    """Test full-text search capabilities."""

    def test_search_products_by_text(self, elasticsearch_with_ecommerce_data):
        """Test searching products by text."""
        connector = ElasticsearchConnector(
            hosts=[elasticsearch_with_ecommerce_data["url"]],
        )

        connector.connect()
        results = connector.search("products", "laptop")

        assert len(results) >= 1
        assert any("laptop" in r["name"].lower() or "laptop" in r.get("description", "").lower()
                  for r in results)
        # Check that scores are included
        assert all("_score" in r for r in results)

        connector.disconnect()

    def test_search_products_by_description(self, elasticsearch_with_ecommerce_data):
        """Test searching products by description."""
        connector = ElasticsearchConnector(
            hosts=[elasticsearch_with_ecommerce_data["url"]],
        )

        connector.connect()
        results = connector.search("products", "noise-cancelling")

        assert len(results) >= 1
        assert any("headphones" in r["name"].lower() for r in results)

        connector.disconnect()

    def test_search_specific_fields(self, elasticsearch_with_ecommerce_data):
        """Test searching specific fields."""
        connector = ElasticsearchConnector(
            hosts=[elasticsearch_with_ecommerce_data["url"]],
        )

        connector.connect()
        results = connector.search(
            "products",
            "chair",
            fields=["name", "description"]
        )

        assert len(results) >= 1
        assert any("chair" in r["name"].lower() or "chair" in r.get("description", "").lower()
                  for r in results)

        connector.disconnect()

    def test_search_reviews(self, elasticsearch_with_ecommerce_data):
        """Test searching reviews."""
        connector = ElasticsearchConnector(
            hosts=[elasticsearch_with_ecommerce_data["url"]],
        )

        connector.connect()
        results = connector.search("reviews", "amazing")

        assert len(results) >= 1
        assert any("amazing" in r.get("body", "").lower() for r in results)

        connector.disconnect()


# =============================================================================
# Aggregation Tests
# =============================================================================

@pytest.mark.requires_elasticsearch
@pytest.mark.integration
class TestElasticsearchAggregations:
    """Test aggregation capabilities."""

    def test_aggregate_by_category(self, elasticsearch_with_ecommerce_data):
        """Test terms aggregation by category."""
        connector = ElasticsearchConnector(
            hosts=[elasticsearch_with_ecommerce_data["url"]],
        )

        connector.connect()
        result = connector.aggregate(
            "products",
            {"categories": {"terms": {"field": "category"}}}
        )

        assert "categories" in result
        buckets = result["categories"]["buckets"]
        assert len(buckets) >= 3

        category_counts = {b["key"]: b["doc_count"] for b in buckets}
        assert "Electronics" in category_counts
        assert category_counts["Electronics"] >= 4

        connector.disconnect()

    def test_aggregate_avg_price(self, elasticsearch_with_ecommerce_data):
        """Test average aggregation on price."""
        connector = ElasticsearchConnector(
            hosts=[elasticsearch_with_ecommerce_data["url"]],
        )

        connector.connect()
        result = connector.aggregate(
            "products",
            {"avg_price": {"avg": {"field": "price"}}}
        )

        assert "avg_price" in result
        avg = result["avg_price"]["value"]
        assert 100 < avg < 1500  # Reasonable range for our test data

        connector.disconnect()

    def test_aggregate_price_stats(self, elasticsearch_with_ecommerce_data):
        """Test stats aggregation on price."""
        connector = ElasticsearchConnector(
            hosts=[elasticsearch_with_ecommerce_data["url"]],
        )

        connector.connect()
        result = connector.aggregate(
            "products",
            {"price_stats": {"stats": {"field": "price"}}}
        )

        stats = result["price_stats"]
        assert "count" in stats
        assert "min" in stats
        assert "max" in stats
        assert "avg" in stats
        assert stats["count"] == 8
        assert stats["min"] < stats["max"]

        connector.disconnect()

    def test_aggregate_by_membership(self, elasticsearch_with_ecommerce_data):
        """Test aggregation of customers by membership level."""
        connector = ElasticsearchConnector(
            hosts=[elasticsearch_with_ecommerce_data["url"]],
        )

        connector.connect()
        result = connector.aggregate(
            "customers",
            {"memberships": {"terms": {"field": "membership"}}}
        )

        buckets = result["memberships"]["buckets"]
        memberships = {b["key"]: b["doc_count"] for b in buckets}

        assert "gold" in memberships
        assert "silver" in memberships
        assert "platinum" in memberships

        connector.disconnect()

    def test_aggregate_with_filter(self, elasticsearch_with_ecommerce_data):
        """Test aggregation with query filter."""
        connector = ElasticsearchConnector(
            hosts=[elasticsearch_with_ecommerce_data["url"]],
        )

        connector.connect()
        result = connector.aggregate(
            "products",
            {"avg_price": {"avg": {"field": "price"}}},
            query={"term": {"category": "Electronics"}}
        )

        avg = result["avg_price"]["value"]
        # Electronics should have higher average price
        assert avg > 100

        connector.disconnect()


# =============================================================================
# Insert and Delete Tests
# =============================================================================

@pytest.mark.requires_elasticsearch
@pytest.mark.integration
class TestElasticsearchInsertDelete:
    """Test insert and delete operations."""

    def test_insert_product(self, elasticsearch_with_ecommerce_data):
        """Test inserting a new product."""
        connector = ElasticsearchConnector(
            hosts=[elasticsearch_with_ecommerce_data["url"]],
        )

        connector.connect()

        # Insert new product
        new_product = {
            "_id": "prod-test",
            "name": "Test Product",
            "description": "A test product",
            "category": "Test",
            "brand": "TestBrand",
            "price": 99.99,
            "rating": 4.0,
            "in_stock": True,
        }
        ids = connector.insert("products", [new_product])
        assert ids == ["prod-test"]

        # Refresh to make searchable
        connector._client.indices.refresh(index="products")

        # Verify insertion
        products = connector.query(
            "products",
            {"term": {"_id": "prod-test"}}
        )
        # Note: Query doesn't filter by _id this way, use get instead
        # Let's search by name
        results = connector.search("products", "Test Product")
        assert any(r["name"] == "Test Product" for r in results)

        # Cleanup
        connector.delete("products", {"term": {"category": "Test"}})

        connector.disconnect()

    def test_delete_by_query(self, elasticsearch_with_ecommerce_data):
        """Test deleting products by query."""
        connector = ElasticsearchConnector(
            hosts=[elasticsearch_with_ecommerce_data["url"]],
        )

        connector.connect()

        # Insert product to delete
        new_product = {
            "name": "Delete Me",
            "category": "ToDelete",
            "price": 1.00,
        }
        connector.insert("products", [new_product])
        connector._client.indices.refresh(index="products")

        # Delete it
        deleted = connector.delete(
            "products",
            {"term": {"category": "ToDelete"}}
        )
        assert deleted >= 1

        # Verify deletion
        connector._client.indices.refresh(index="products")
        results = connector.search("products", "Delete Me")
        assert not any(r["name"] == "Delete Me" for r in results)

        connector.disconnect()


# =============================================================================
# Index Management Tests
# =============================================================================

@pytest.mark.requires_elasticsearch
@pytest.mark.integration
class TestElasticsearchIndexManagement:
    """Test index creation and deletion."""

    def test_create_and_delete_index(self, elasticsearch_container):
        """Test creating and deleting an index."""
        connector = ElasticsearchConnector(
            hosts=[elasticsearch_container["url"]],
        )

        connector.connect()

        # Create new index
        connector.create_index(
            "test_index",
            mappings={
                "properties": {
                    "name": {"type": "text"},
                    "value": {"type": "integer"},
                }
            }
        )

        # Verify creation
        indices = connector.get_collections()
        assert "test_index" in indices

        # Delete index
        connector.delete_index("test_index")

        # Verify deletion
        indices = connector.get_collections()
        assert "test_index" not in indices

        connector.disconnect()


# =============================================================================
# Schema Discovery Tests
# =============================================================================

@pytest.mark.requires_elasticsearch
@pytest.mark.integration
class TestElasticsearchSchemaDiscovery:
    """Test schema discovery capabilities."""

    def test_get_all_metadata(self, elasticsearch_with_ecommerce_data):
        """Test getting metadata for all indices."""
        connector = ElasticsearchConnector(
            hosts=[elasticsearch_with_ecommerce_data["url"]],
        )

        connector.connect()
        all_metadata = connector.get_all_metadata()

        assert len(all_metadata) >= 4
        index_names = [m.name for m in all_metadata]
        assert "products" in index_names
        assert "customers" in index_names
        assert "orders" in index_names
        assert "reviews" in index_names

        connector.disconnect()

    def test_get_overview(self, elasticsearch_with_ecommerce_data):
        """Test generating overview summary."""
        connector = ElasticsearchConnector(
            hosts=[elasticsearch_with_ecommerce_data["url"]],
        )

        connector.connect()
        overview = connector.get_overview()

        assert "elasticsearch" in overview
        assert "products" in overview or "customers" in overview

        connector.disconnect()

    def test_sample_documents(self, elasticsearch_with_ecommerce_data):
        """Test sampling documents from an index."""
        connector = ElasticsearchConnector(
            hosts=[elasticsearch_with_ecommerce_data["url"]],
        )

        connector.connect()
        samples = connector.sample_documents("products", limit=3)

        assert len(samples) == 3
        for sample in samples:
            assert "name" in sample
            assert "price" in sample

        connector.disconnect()


# =============================================================================
# NLQ Integration Tests (Natural Language Query scenarios)
# =============================================================================

@pytest.mark.requires_elasticsearch
@pytest.mark.integration
class TestElasticsearchNLQScenarios:
    """Test scenarios that would be used in natural language queries.

    These tests verify that queries typical in NLQ use cases work correctly.
    """

    def test_nlq_find_products_under_price(self, elasticsearch_with_ecommerce_data):
        """NLQ: Find all products under $200."""
        connector = ElasticsearchConnector(
            hosts=[elasticsearch_with_ecommerce_data["url"]],
        )

        connector.connect()

        products = connector.query(
            "products",
            {"range": {"price": {"lt": 200}}}
        )

        assert len(products) >= 4
        for product in products:
            assert product["price"] < 200

        connector.disconnect()

    def test_nlq_find_high_rated_products(self, elasticsearch_with_ecommerce_data):
        """NLQ: Show me products with rating above 4.5."""
        connector = ElasticsearchConnector(
            hosts=[elasticsearch_with_ecommerce_data["url"]],
        )

        connector.connect()

        products = connector.query(
            "products",
            {"range": {"rating": {"gt": 4.5}}}
        )

        assert len(products) >= 2
        for product in products:
            assert product["rating"] > 4.5

        connector.disconnect()

    def test_nlq_search_by_keyword(self, elasticsearch_with_ecommerce_data):
        """NLQ: Search for 'professional' products."""
        connector = ElasticsearchConnector(
            hosts=[elasticsearch_with_ecommerce_data["url"]],
        )

        connector.connect()

        results = connector.search("products", "professional")

        assert len(results) >= 1
        # Should find MacBook Pro which has "professional" in description

        connector.disconnect()

    def test_nlq_products_in_stock(self, elasticsearch_with_ecommerce_data):
        """NLQ: List all products that are in stock."""
        connector = ElasticsearchConnector(
            hosts=[elasticsearch_with_ecommerce_data["url"]],
        )

        connector.connect()

        products = connector.query(
            "products",
            {"term": {"in_stock": True}}
        )

        assert len(products) >= 7  # 7 out of 8 are in stock
        for product in products:
            assert product["in_stock"] is True

        connector.disconnect()

    def test_nlq_count_by_category(self, elasticsearch_with_ecommerce_data):
        """NLQ: How many products are in each category?"""
        connector = ElasticsearchConnector(
            hosts=[elasticsearch_with_ecommerce_data["url"]],
        )

        connector.connect()

        result = connector.aggregate(
            "products",
            {"by_category": {"terms": {"field": "category"}}}
        )

        buckets = result["by_category"]["buckets"]
        category_counts = {b["key"]: b["doc_count"] for b in buckets}

        assert "Electronics" in category_counts
        assert "Furniture" in category_counts
        assert "Books" in category_counts

        connector.disconnect()

    def test_nlq_average_spending_by_membership(self, elasticsearch_with_ecommerce_data):
        """NLQ: What's the average spending by membership level?"""
        connector = ElasticsearchConnector(
            hosts=[elasticsearch_with_ecommerce_data["url"]],
        )

        connector.connect()

        result = connector.aggregate(
            "customers",
            {
                "by_membership": {
                    "terms": {"field": "membership"},
                    "aggs": {
                        "avg_spent": {"avg": {"field": "total_spent"}}
                    }
                }
            }
        )

        buckets = result["by_membership"]["buckets"]
        for bucket in buckets:
            assert "avg_spent" in bucket
            assert bucket["avg_spent"]["value"] > 0

        connector.disconnect()

    def test_nlq_top_rated_products_by_reviews(self, elasticsearch_with_ecommerce_data):
        """NLQ: What are the top 3 most reviewed products?"""
        connector = ElasticsearchConnector(
            hosts=[elasticsearch_with_ecommerce_data["url"]],
        )

        connector.connect()

        # Use search with sort
        response = connector._client.search(
            index="products",
            query={"match_all": {}},
            sort=[{"reviews_count": {"order": "desc"}}],
            size=3
        )

        results = [hit["_source"] for hit in response["hits"]["hits"]]
        assert len(results) == 3

        # Verify they're sorted by reviews_count
        review_counts = [r["reviews_count"] for r in results]
        assert review_counts == sorted(review_counts, reverse=True)

        connector.disconnect()

    def test_nlq_customers_from_country(self, elasticsearch_with_ecommerce_data):
        """NLQ: How many customers are from the USA?"""
        connector = ElasticsearchConnector(
            hosts=[elasticsearch_with_ecommerce_data["url"]],
        )

        connector.connect()

        customers = connector.query(
            "customers",
            {"term": {"country": "USA"}}
        )

        assert len(customers) == 2  # Alice and David

        connector.disconnect()

    def test_nlq_recent_orders(self, elasticsearch_with_ecommerce_data):
        """NLQ: Show me orders from September 2023."""
        connector = ElasticsearchConnector(
            hosts=[elasticsearch_with_ecommerce_data["url"]],
        )

        connector.connect()

        orders = connector.query(
            "orders",
            {
                "range": {
                    "created_at": {
                        "gte": "2023-09-01",
                        "lte": "2023-09-30"
                    }
                }
            }
        )

        assert len(orders) >= 5

        connector.disconnect()

    def test_nlq_orders_by_status(self, elasticsearch_with_ecommerce_data):
        """NLQ: How many orders are in each status?"""
        connector = ElasticsearchConnector(
            hosts=[elasticsearch_with_ecommerce_data["url"]],
        )

        connector.connect()

        result = connector.aggregate(
            "orders",
            {"by_status": {"terms": {"field": "status"}}}
        )

        buckets = result["by_status"]["buckets"]
        status_counts = {b["key"]: b["doc_count"] for b in buckets}

        assert "delivered" in status_counts
        assert "shipped" in status_counts

        connector.disconnect()

    def test_nlq_total_revenue(self, elasticsearch_with_ecommerce_data):
        """NLQ: What's the total revenue from all orders?"""
        connector = ElasticsearchConnector(
            hosts=[elasticsearch_with_ecommerce_data["url"]],
        )

        connector.connect()

        result = connector.aggregate(
            "orders",
            {"total_revenue": {"sum": {"field": "total"}}}
        )

        revenue = result["total_revenue"]["value"]
        assert revenue > 4000  # Sum of all order totals

        connector.disconnect()

    def test_nlq_five_star_reviews(self, elasticsearch_with_ecommerce_data):
        """NLQ: Show me all 5-star reviews."""
        connector = ElasticsearchConnector(
            hosts=[elasticsearch_with_ecommerce_data["url"]],
        )

        connector.connect()

        reviews = connector.query(
            "reviews",
            {"term": {"rating": 5}}
        )

        assert len(reviews) >= 3
        for review in reviews:
            assert review["rating"] == 5

        connector.disconnect()

    def test_nlq_helpful_reviews(self, elasticsearch_with_ecommerce_data):
        """NLQ: Find reviews with more than 30 helpful votes."""
        connector = ElasticsearchConnector(
            hosts=[elasticsearch_with_ecommerce_data["url"]],
        )

        connector.connect()

        reviews = connector.query(
            "reviews",
            {"range": {"helpful_votes": {"gt": 30}}}
        )

        assert len(reviews) >= 2
        for review in reviews:
            assert review["helpful_votes"] > 30

        connector.disconnect()
