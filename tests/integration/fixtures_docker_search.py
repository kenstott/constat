from __future__ import annotations

"""Docker fixtures for search/analytics services (Elasticsearch)."""

import subprocess
import time
from typing import Generator

import pytest

from tests.integration.fixtures_docker_helpers import (
    get_unique_container_name,
    get_unique_port,
    is_container_running,
    stop_container,
)


# ---------------------------------------------------------------------------
# Elasticsearch helpers
# ---------------------------------------------------------------------------

def is_elasticsearch_ready(port: int = 9200) -> bool:
    """Check if Elasticsearch is ready."""
    try:
        import httpx
        response = httpx.get(f"http://localhost:{port}/_cluster/health", timeout=5.0)
        if response.status_code == 200:
            data = response.json()
            return data.get("status") in ["green", "yellow"]
    except Exception:
        pass  # Probe: Elasticsearch not reachable; return False as sentinel
    return False


def wait_for_elasticsearch(port: int = 9200, timeout: int = 90) -> bool:
    """Wait for Elasticsearch to be ready."""
    start = time.time()
    while time.time() - start < timeout:
        if is_elasticsearch_ready(port):
            return True
        time.sleep(3)
    return False


def load_elasticsearch_ecommerce_data(port: int) -> bool:
    """Load sample e-commerce data into Elasticsearch."""
    try:
        from elasticsearch import Elasticsearch
        from elasticsearch.helpers import bulk
    except ImportError:
        return False

    try:
        es = Elasticsearch([f"http://localhost:{port}"])

        for index in ["products", "customers", "orders", "reviews"]:
            try:
                es.indices.delete(index=index)
            except Exception:
                pass  # Index may not exist yet; swallowed intentionally

        es.indices.create(
            index="products",
            body={
                "mappings": {
                    "properties": {
                        "name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                        "description": {"type": "text"},
                        "category": {"type": "keyword"},
                        "brand": {"type": "keyword"},
                        "price": {"type": "float"},
                        "rating": {"type": "float"},
                        "reviews_count": {"type": "integer"},
                        "in_stock": {"type": "boolean"},
                        "tags": {"type": "keyword"},
                        "created_at": {"type": "date"},
                    }
                }
            }
        )

        es.indices.create(
            index="customers",
            body={
                "mappings": {
                    "properties": {
                        "name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                        "email": {"type": "keyword"},
                        "age": {"type": "integer"},
                        "country": {"type": "keyword"},
                        "membership": {"type": "keyword"},
                        "total_spent": {"type": "float"},
                        "joined_at": {"type": "date"},
                    }
                }
            }
        )

        es.indices.create(
            index="orders",
            body={
                "mappings": {
                    "properties": {
                        "customer_id": {"type": "keyword"},
                        "products": {
                            "properties": {
                                "product_id": {"type": "keyword"},
                                "quantity": {"type": "integer"},
                                "price": {"type": "float"},
                            }
                        },
                        "total": {"type": "float"},
                        "status": {"type": "keyword"},
                        "payment_method": {"type": "keyword"},
                        "created_at": {"type": "date"},
                    }
                }
            }
        )

        es.indices.create(
            index="reviews",
            body={
                "mappings": {
                    "properties": {
                        "product_id": {"type": "keyword"},
                        "customer_id": {"type": "keyword"},
                        "rating": {"type": "integer"},
                        "title": {"type": "text"},
                        "body": {"type": "text"},
                        "helpful_votes": {"type": "integer"},
                        "created_at": {"type": "date"},
                    }
                }
            }
        )

        products = [
            {"_index": "products", "_id": "prod-001", "_source": {"name": "MacBook Pro 16", "description": "Powerful laptop for professionals", "category": "Electronics", "brand": "Apple", "price": 2499.99, "rating": 4.8, "reviews_count": 1250, "in_stock": True, "tags": ["laptop", "apple", "professional"], "created_at": "2023-01-15"}},
            {"_index": "products", "_id": "prod-002", "_source": {"name": "Sony WH-1000XM5", "description": "Premium noise-cancelling headphones", "category": "Electronics", "brand": "Sony", "price": 349.99, "rating": 4.7, "reviews_count": 3420, "in_stock": True, "tags": ["headphones", "wireless", "noise-cancelling"], "created_at": "2023-02-20"}},
            {"_index": "products", "_id": "prod-003", "_source": {"name": "Ergonomic Office Chair", "description": "Comfortable chair with lumbar support", "category": "Furniture", "brand": "Herman Miller", "price": 1299.00, "rating": 4.5, "reviews_count": 890, "in_stock": True, "tags": ["chair", "office", "ergonomic"], "created_at": "2023-03-10"}},
            {"_index": "products", "_id": "prod-004", "_source": {"name": "Kindle Paperwhite", "description": "E-reader with adjustable warm light", "category": "Electronics", "brand": "Amazon", "price": 139.99, "rating": 4.6, "reviews_count": 15230, "in_stock": True, "tags": ["e-reader", "books", "portable"], "created_at": "2023-04-05"}},
            {"_index": "products", "_id": "prod-005", "_source": {"name": "Standing Desk", "description": "Electric height-adjustable desk", "category": "Furniture", "brand": "Uplift", "price": 599.00, "rating": 4.4, "reviews_count": 2100, "in_stock": False, "tags": ["desk", "standing", "adjustable"], "created_at": "2023-05-15"}},
            {"_index": "products", "_id": "prod-006", "_source": {"name": "Python Programming Book", "description": "Learn Python the hard way", "category": "Books", "brand": "O'Reilly", "price": 49.99, "rating": 4.3, "reviews_count": 520, "in_stock": True, "tags": ["programming", "python", "education"], "created_at": "2023-06-20"}},
            {"_index": "products", "_id": "prod-007", "_source": {"name": "Mechanical Keyboard", "description": "RGB mechanical keyboard with Cherry MX switches", "category": "Electronics", "brand": "Logitech", "price": 179.99, "rating": 4.6, "reviews_count": 4500, "in_stock": True, "tags": ["keyboard", "mechanical", "gaming"], "created_at": "2023-07-01"}},
            {"_index": "products", "_id": "prod-008", "_source": {"name": "Desk Lamp", "description": "LED desk lamp with adjustable brightness", "category": "Furniture", "brand": "BenQ", "price": 119.00, "rating": 4.5, "reviews_count": 1800, "in_stock": True, "tags": ["lamp", "led", "office"], "created_at": "2023-08-10"}},
        ]
        bulk(es, products)

        customers = [
            {"_index": "customers", "_id": "cust-001", "_source": {"name": "Alice Johnson", "email": "alice@example.com", "age": 32, "country": "USA", "membership": "gold", "total_spent": 5420.50, "joined_at": "2022-01-15"}},
            {"_index": "customers", "_id": "cust-002", "_source": {"name": "Bob Smith", "email": "bob@example.com", "age": 45, "country": "UK", "membership": "silver", "total_spent": 2150.00, "joined_at": "2022-03-20"}},
            {"_index": "customers", "_id": "cust-003", "_source": {"name": "Carol White", "email": "carol@example.com", "age": 28, "country": "Canada", "membership": "platinum", "total_spent": 12300.75, "joined_at": "2021-06-10"}},
            {"_index": "customers", "_id": "cust-004", "_source": {"name": "David Lee", "email": "david@example.com", "age": 38, "country": "USA", "membership": "gold", "total_spent": 4890.25, "joined_at": "2022-08-05"}},
            {"_index": "customers", "_id": "cust-005", "_source": {"name": "Emma Wilson", "email": "emma@example.com", "age": 25, "country": "Australia", "membership": "silver", "total_spent": 1560.00, "joined_at": "2023-01-20"}},
        ]
        bulk(es, customers)

        orders = [
            {"_index": "orders", "_id": "order-001", "_source": {"customer_id": "cust-001", "products": [{"product_id": "prod-001", "quantity": 1, "price": 2499.99}], "total": 2499.99, "status": "delivered", "payment_method": "credit_card", "created_at": "2023-09-01"}},
            {"_index": "orders", "_id": "order-002", "_source": {"customer_id": "cust-002", "products": [{"product_id": "prod-002", "quantity": 1, "price": 349.99}, {"product_id": "prod-004", "quantity": 1, "price": 139.99}], "total": 489.98, "status": "shipped", "payment_method": "paypal", "created_at": "2023-09-05"}},
            {"_index": "orders", "_id": "order-003", "_source": {"customer_id": "cust-003", "products": [{"product_id": "prod-003", "quantity": 1, "price": 1299.00}], "total": 1299.00, "status": "delivered", "payment_method": "credit_card", "created_at": "2023-09-10"}},
            {"_index": "orders", "_id": "order-004", "_source": {"customer_id": "cust-001", "products": [{"product_id": "prod-007", "quantity": 1, "price": 179.99}], "total": 179.99, "status": "processing", "payment_method": "credit_card", "created_at": "2023-09-15"}},
            {"_index": "orders", "_id": "order-005", "_source": {"customer_id": "cust-004", "products": [{"product_id": "prod-006", "quantity": 2, "price": 49.99}], "total": 99.98, "status": "delivered", "payment_method": "debit_card", "created_at": "2023-09-20"}},
        ]
        bulk(es, orders)

        reviews = [
            {"_index": "reviews", "_id": "rev-001", "_source": {"product_id": "prod-001", "customer_id": "cust-001", "rating": 5, "title": "Best laptop ever!", "body": "This MacBook Pro is amazing. Fast, beautiful display, and great battery life.", "helpful_votes": 42, "created_at": "2023-09-15"}},
            {"_index": "reviews", "_id": "rev-002", "_source": {"product_id": "prod-002", "customer_id": "cust-002", "rating": 4, "title": "Great headphones", "body": "Excellent noise cancellation but a bit pricey.", "helpful_votes": 28, "created_at": "2023-09-18"}},
            {"_index": "reviews", "_id": "rev-003", "_source": {"product_id": "prod-003", "customer_id": "cust-003", "rating": 5, "title": "Worth every penny", "body": "My back pain has significantly reduced since using this chair.", "helpful_votes": 65, "created_at": "2023-09-20"}},
            {"_index": "reviews", "_id": "rev-004", "_source": {"product_id": "prod-004", "customer_id": "cust-004", "rating": 4, "title": "Great for reading", "body": "Love the warm light feature, makes reading at night much easier.", "helpful_votes": 15, "created_at": "2023-09-22"}},
            {"_index": "reviews", "_id": "rev-005", "_source": {"product_id": "prod-001", "customer_id": "cust-003", "rating": 5, "title": "Professional grade", "body": "Perfect for software development and video editing.", "helpful_votes": 38, "created_at": "2023-09-25"}},
        ]
        bulk(es, reviews)

        es.indices.refresh(index="_all")
        es.close()
        return True
    except Exception as e:
        print(f"Error loading Elasticsearch data: {e}")
        return False


# ---------------------------------------------------------------------------
# Elasticsearch fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def elasticsearch_container(docker_available) -> Generator[dict, None, None]:
    """Start Elasticsearch container for the test session.

    Yields connection info dict with keys: host, port, url
    """
    if not docker_available:
        pytest.fail("Docker not available — install Docker to run this test")

    container_name = get_unique_container_name("constat_test_elasticsearch")
    port = get_unique_port(9200)

    if not is_container_running(container_name):
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True,
            timeout=30,
        )

        cmd = [
            "docker", "run", "-d",
            "--name", container_name,
            "-p", f"{port}:9200",
            "-e", "discovery.type=single-node",
            "-e", "xpack.security.enabled=false",
            "-e", "ES_JAVA_OPTS=-Xms256m -Xmx256m",
            "docker.elastic.co/elasticsearch/elasticsearch:9.0.0"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                pytest.fail(f"Failed to start Elasticsearch container: {result.stderr}")
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            pytest.fail(f"Error starting Elasticsearch container: {e}")

    if not wait_for_elasticsearch(port, timeout=90):
        stop_container(container_name)
        pytest.fail("Elasticsearch container failed to become ready")

    yield {
        "host": "localhost",
        "port": port,
        "url": f"http://localhost:{port}",
        "container_name": container_name,
    }

    stop_container(container_name)


@pytest.fixture(scope="session")
def elasticsearch_with_ecommerce_data(elasticsearch_container) -> Generator[dict, None, None]:
    """Elasticsearch container with sample e-commerce data pre-loaded."""
    if not load_elasticsearch_ecommerce_data(elasticsearch_container["port"]):
        pytest.fail("Failed to load Elasticsearch e-commerce data")

    yield {
        **elasticsearch_container,
        "indices": ["products", "customers", "orders", "reviews"],
    }


@pytest.fixture
def elasticsearch_url(elasticsearch_container) -> str:
    """Get Elasticsearch URL."""
    return elasticsearch_container["url"]
