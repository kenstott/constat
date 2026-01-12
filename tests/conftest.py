"""Pytest configuration and fixtures for integration tests."""

import os
import subprocess
import time
import pytest
from typing import Generator


def is_docker_available() -> bool:
    """Check if Docker is available and running."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=60,  # Docker can be slow on first call
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def is_container_running(name: str) -> bool:
    """Check if a container with the given name is running."""
    try:
        result = subprocess.run(
            ["docker", "ps", "-q", "-f", f"name={name}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return bool(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def start_container(
    name: str,
    image: str,
    port_mapping: str,
    env: dict[str, str] | None = None,
    volume: str | None = None,
    wait_seconds: int = 5,
) -> bool:
    """Start a Docker container if not already running."""
    if is_container_running(name):
        return True

    # Remove stopped container with same name if exists
    subprocess.run(
        ["docker", "rm", "-f", name],
        capture_output=True,
        timeout=30,
    )

    cmd = ["docker", "run", "-d", "--name", name, "-p", port_mapping]

    if env:
        for key, value in env.items():
            cmd.extend(["-e", f"{key}={value}"])

    if volume:
        cmd.extend(["-v", volume])

    cmd.append(image)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            print(f"Failed to start {name}: {result.stderr}")
            return False

        # Wait for container to be ready
        time.sleep(wait_seconds)
        return is_container_running(name)
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"Error starting {name}: {e}")
        return False


def stop_container(name: str) -> None:
    """Stop and remove a Docker container."""
    try:
        subprocess.run(
            ["docker", "rm", "-f", name],
            capture_output=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass


# Pytest markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "requires_mongodb: mark test as requiring MongoDB"
    )
    config.addinivalue_line(
        "markers", "requires_postgresql: mark test as requiring PostgreSQL"
    )
    config.addinivalue_line(
        "markers", "requires_docker: mark test as requiring Docker"
    )


# Docker availability check
DOCKER_AVAILABLE = is_docker_available()


@pytest.fixture(scope="session")
def docker_available() -> bool:
    """Check if Docker is available."""
    return DOCKER_AVAILABLE


@pytest.fixture(scope="session")
def mongodb_container(docker_available) -> Generator[dict, None, None]:
    """Start MongoDB container for the test session.

    Yields connection info dict with keys: host, port, uri
    """
    if not docker_available:
        pytest.skip("Docker not available")

    container_name = "constat_test_mongodb"
    port = 27017

    if not start_container(
        name=container_name,
        image="mongo:latest",
        port_mapping=f"{port}:27017",
        wait_seconds=10,  # Give MongoDB more time to initialize
    ):
        pytest.skip("Failed to start MongoDB container")

    yield {
        "host": "localhost",
        "port": port,
        "uri": f"mongodb://localhost:{port}",
        "container_name": container_name,
    }

    # Cleanup: stop container after all tests
    stop_container(container_name)


@pytest.fixture(scope="session")
def postgresql_container(docker_available) -> Generator[dict, None, None]:
    """Start PostgreSQL container for the test session.

    Yields connection info dict with keys: host, port, user, password, database, dsn
    """
    if not docker_available:
        pytest.skip("Docker not available")

    container_name = "constat_test_postgresql"
    port = 5432
    password = "testpassword"
    user = "postgres"
    database = "postgres"

    if not start_container(
        name=container_name,
        image="postgres:latest",
        port_mapping=f"{port}:5432",
        env={"POSTGRES_PASSWORD": password},
        wait_seconds=15,  # PostgreSQL needs more time to initialize
    ):
        pytest.skip("Failed to start PostgreSQL container")

    yield {
        "host": "localhost",
        "port": port,
        "user": user,
        "password": password,
        "database": database,
        "dsn": f"postgresql://{user}:{password}@localhost:{port}/{database}",
        "container_name": container_name,
    }

    # Cleanup: stop container after all tests
    stop_container(container_name)


@pytest.fixture
def mongodb_uri(mongodb_container) -> str:
    """Get MongoDB connection URI."""
    return mongodb_container["uri"]


@pytest.fixture
def postgresql_dsn(postgresql_container) -> str:
    """Get PostgreSQL connection DSN."""
    return postgresql_container["dsn"]


def load_mongodb_sample_data(uri: str, database: str = "sample_ecommerce") -> bool:
    """Load sample e-commerce data into MongoDB."""
    try:
        import pymongo
    except ImportError:
        return False

    # Sample data
    sample_data = {
        "customers": [
            {"_id": 1, "name": "Alice Johnson", "email": "alice@example.com", "country": "USA"},
            {"_id": 2, "name": "Bob Smith", "email": "bob@example.com", "country": "UK"},
            {"_id": 3, "name": "Carol White", "email": "carol@example.com", "country": "Canada"},
        ],
        "products": [
            {"_id": 101, "name": "Laptop Pro", "category": "Electronics", "price": 1299.99, "stock": 50},
            {"_id": 102, "name": "Wireless Mouse", "category": "Electronics", "price": 29.99, "stock": 200},
            {"_id": 103, "name": "Desk Chair", "category": "Furniture", "price": 249.99, "stock": 30},
        ],
        "orders": [
            {"_id": 1001, "customer_id": 1, "products": [{"product_id": 101, "quantity": 1}], "total": 1299.99, "status": "delivered"},
            {"_id": 1002, "customer_id": 2, "products": [{"product_id": 103, "quantity": 1}], "total": 249.99, "status": "shipped"},
        ],
    }

    client = pymongo.MongoClient(uri)
    db = client[database]

    for collection_name, documents in sample_data.items():
        db.drop_collection(collection_name)
        db[collection_name].insert_many(documents)

    client.close()
    return True


def load_postgresql_sample_data(dsn: str) -> bool:
    """Load sample e-commerce data into PostgreSQL."""
    try:
        import psycopg2
    except ImportError:
        return False

    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    cur = conn.cursor()

    # Create schema and load data
    cur.execute("""
        DROP TABLE IF EXISTS order_items, orders, reviews, products, customers CASCADE;

        CREATE TABLE customers (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            email VARCHAR(100) UNIQUE NOT NULL,
            country VARCHAR(50)
        );

        CREATE TABLE products (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            category VARCHAR(50),
            price DECIMAL(10, 2),
            stock INTEGER DEFAULT 0
        );

        CREATE TABLE orders (
            id SERIAL PRIMARY KEY,
            customer_id INTEGER REFERENCES customers(id),
            total DECIMAL(10, 2),
            status VARCHAR(20)
        );

        INSERT INTO customers (id, name, email, country) VALUES
        (1, 'Alice Johnson', 'alice@example.com', 'USA'),
        (2, 'Bob Smith', 'bob@example.com', 'UK'),
        (3, 'Carol White', 'carol@example.com', 'Canada');

        INSERT INTO products (id, name, category, price, stock) VALUES
        (101, 'Laptop Pro', 'Electronics', 1299.99, 50),
        (102, 'Wireless Mouse', 'Electronics', 29.99, 200),
        (103, 'Desk Chair', 'Furniture', 249.99, 30);

        INSERT INTO orders (id, customer_id, total, status) VALUES
        (1001, 1, 1299.99, 'delivered'),
        (1002, 2, 249.99, 'shipped');

        SELECT setval('customers_id_seq', 3);
        SELECT setval('products_id_seq', 103);
        SELECT setval('orders_id_seq', 1002);
    """)

    cur.close()
    conn.close()
    return True


@pytest.fixture(scope="session")
def mongodb_with_sample_data(mongodb_container) -> Generator[dict, None, None]:
    """MongoDB container with sample e-commerce data pre-loaded."""
    load_mongodb_sample_data(mongodb_container["uri"])
    yield {
        **mongodb_container,
        "database": "sample_ecommerce",
    }


@pytest.fixture(scope="session")
def postgresql_with_sample_data(postgresql_container) -> Generator[dict, None, None]:
    """PostgreSQL container with sample e-commerce data pre-loaded."""
    load_postgresql_sample_data(postgresql_container["dsn"])
    yield postgresql_container


# Auto-skip tests based on markers
def pytest_collection_modifyitems(config, items):
    """Skip tests that require Docker if it's not available."""
    if DOCKER_AVAILABLE:
        return

    skip_docker = pytest.mark.skip(reason="Docker not available")
    skip_mongodb = pytest.mark.skip(reason="MongoDB requires Docker")
    skip_postgresql = pytest.mark.skip(reason="PostgreSQL requires Docker")

    for item in items:
        if "requires_docker" in item.keywords:
            item.add_marker(skip_docker)
        if "requires_mongodb" in item.keywords:
            item.add_marker(skip_mongodb)
        if "requires_postgresql" in item.keywords:
            item.add_marker(skip_postgresql)
