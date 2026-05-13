from __future__ import annotations

"""Docker fixtures for relational and document databases (MongoDB, PostgreSQL, Cassandra)."""

import time
from typing import Generator

import pytest

from tests.integration.fixtures_docker_helpers import (
    get_unique_container_name,
    get_unique_port,
    start_container,
    stop_container,
)


# ---------------------------------------------------------------------------
# MongoDB
# ---------------------------------------------------------------------------

def load_mongodb_sample_data(uri: str, database: str = "sample_ecommerce") -> bool:
    """Load sample e-commerce data into MongoDB."""
    try:
        import pymongo
    except ImportError:
        return False

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
def mongodb_container(docker_available) -> Generator[dict, None, None]:
    """Start MongoDB container for the test session.

    Yields connection info dict with keys: host, port, uri
    """
    if not docker_available:
        pytest.fail("Docker not available — install Docker to run this test")

    container_name = get_unique_container_name("constat_test_mongodb")
    port = get_unique_port(27017)

    if not start_container(
        name=container_name,
        image="mongo:latest",
        port_mapping=f"{port}:27017",
        wait_seconds=10,
    ):
        pytest.fail("Failed to start MongoDB container")

    yield {
        "host": "localhost",
        "port": port,
        "uri": f"mongodb://localhost:{port}",
        "container_name": container_name,
    }

    stop_container(container_name)


@pytest.fixture(scope="session")
def postgresql_container(docker_available) -> Generator[dict, None, None]:
    """Start PostgreSQL container for the test session.

    Yields connection info dict with keys: host, port, user, password, database, dsn
    """
    if not docker_available:
        pytest.fail("Docker not available — install Docker to run this test")

    container_name = get_unique_container_name("constat_test_postgresql")
    port = get_unique_port(5432)
    password = "testpassword"
    user = "postgres"
    database = "postgres"

    if not start_container(
        name=container_name,
        image="postgres:latest",
        port_mapping=f"{port}:5432",
        env={"POSTGRES_PASSWORD": password},
        wait_seconds=15,
    ):
        pytest.fail("Failed to start PostgreSQL container")

    yield {
        "host": "localhost",
        "port": port,
        "user": user,
        "password": password,
        "database": database,
        "dsn": f"postgresql://{user}:{password}@localhost:{port}/{database}",
        "container_name": container_name,
    }

    stop_container(container_name)


@pytest.fixture
def mongodb_uri(mongodb_container) -> str:
    """Get MongoDB connection URI."""
    return mongodb_container["uri"]


@pytest.fixture
def postgresql_dsn(postgresql_container) -> str:
    """Get PostgreSQL connection DSN."""
    return postgresql_container["dsn"]


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


# ---------------------------------------------------------------------------
# Cassandra
# ---------------------------------------------------------------------------

def is_cassandra_ready(port: int = 9042) -> bool:
    """Check if Cassandra is ready to accept CQL connections."""
    import socket
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=2):
            pass
    except OSError:
        return False
    try:
        from cassandra.cluster import Cluster
        cluster = Cluster(
            ["127.0.0.1"],
            port=port,
            connect_timeout=5,
            control_connection_timeout=5,
        )
        session = cluster.connect()
        session.shutdown()
        cluster.shutdown()
        return True
    except Exception:
        return False


def wait_for_cassandra(port: int = 9042, timeout: int = 600) -> bool:
    """Wait for Cassandra to be ready (CQL handshake)."""
    start = time.time()
    while time.time() - start < timeout:
        if is_cassandra_ready(port):
            return True
        time.sleep(5)
    return False


def load_cassandra_iot_data(port: int, keyspace: str = "iot_data") -> bool:
    """Load sample IoT sensor data into Cassandra."""
    try:
        from cassandra.cluster import Cluster
    except ImportError:
        return False

    try:
        cluster = Cluster(["localhost"], port=port)
        session = cluster.connect()

        session.execute(f"""
            CREATE KEYSPACE IF NOT EXISTS {keyspace}
            WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': 1}}
        """)
        session.set_keyspace(keyspace)

        session.execute("""
            CREATE TABLE IF NOT EXISTS sensor_readings (
                device_id text,
                timestamp timestamp,
                sensor_type text,
                value double,
                unit text,
                PRIMARY KEY (device_id, timestamp)
            ) WITH CLUSTERING ORDER BY (timestamp DESC)
        """)

        session.execute("""
            CREATE TABLE IF NOT EXISTS devices (
                device_id text PRIMARY KEY,
                name text,
                location text,
                status text,
                installed_date date
            )
        """)

        session.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                device_id text,
                alert_time timestamp,
                alert_type text,
                severity text,
                message text,
                acknowledged boolean,
                PRIMARY KEY (device_id, alert_time)
            ) WITH CLUSTERING ORDER BY (alert_time DESC)
        """)

        devices = [
            ("sensor-001", "Temperature Sensor A", "Building 1 - Floor 1", "active", "2023-01-15"),
            ("sensor-002", "Humidity Sensor B", "Building 1 - Floor 2", "active", "2023-02-20"),
            ("sensor-003", "Pressure Sensor C", "Building 2 - Floor 1", "inactive", "2023-03-10"),
            ("sensor-004", "Temperature Sensor D", "Building 2 - Floor 2", "active", "2023-04-05"),
            ("sensor-005", "Motion Sensor E", "Building 1 - Floor 1", "active", "2023-05-15"),
        ]
        for d in devices:
            session.execute(
                "INSERT INTO devices (device_id, name, location, status, installed_date) VALUES (%s, %s, %s, %s, %s)",
                d
            )

        from datetime import datetime, timedelta
        base_time = datetime.now()
        readings = []
        for i in range(50):
            for device_id in ["sensor-001", "sensor-002", "sensor-003", "sensor-004"]:
                timestamp = base_time - timedelta(hours=i)
                if "Temperature" in device_id or device_id in ["sensor-001", "sensor-004"]:
                    value = 20.0 + (i % 10) + (hash(device_id) % 5)
                    unit = "celsius"
                    sensor_type = "temperature"
                else:
                    value = 40.0 + (i % 20)
                    unit = "percent"
                    sensor_type = "humidity"
                readings.append((device_id, timestamp, sensor_type, value, unit))

        for r in readings:
            session.execute(
                "INSERT INTO sensor_readings (device_id, timestamp, sensor_type, value, unit) VALUES (%s, %s, %s, %s, %s)",
                r
            )

        alerts = [
            ("sensor-001", base_time - timedelta(hours=2), "high_temperature", "warning", "Temperature exceeded 30C", False),
            ("sensor-002", base_time - timedelta(hours=5), "low_humidity", "info", "Humidity below 35%", True),
            ("sensor-003", base_time - timedelta(days=1), "device_offline", "critical", "Device not responding", False),
            ("sensor-001", base_time - timedelta(hours=10), "high_temperature", "warning", "Temperature exceeded 28C", True),
        ]
        for a in alerts:
            session.execute(
                "INSERT INTO alerts (device_id, alert_time, alert_type, severity, message, acknowledged) VALUES (%s, %s, %s, %s, %s, %s)",
                a
            )

        session.shutdown()
        cluster.shutdown()
        return True
    except Exception as e:
        print(f"Error loading Cassandra data: {e}")
        return False


@pytest.fixture(scope="session")
def cassandra_container(docker_available) -> Generator[dict, None, None]:
    """Start Cassandra container for the test session.

    Yields connection info dict with keys: host, port, keyspace
    """
    if not docker_available:
        pytest.fail("Docker not available — install Docker to run this test")

    container_name = get_unique_container_name("constat_test_cassandra")
    port = get_unique_port(9042)

    if not start_container(
        name=container_name,
        image="cassandra:latest",
        port_mapping=f"{port}:9042",
        env={
            "CASSANDRA_CLUSTER_NAME": "TestCluster",
            "MAX_HEAP_SIZE": "512M",
            "HEAP_NEWSIZE": "128M",
        },
        wait_seconds=10,
    ):
        pytest.fail("Failed to start Cassandra container")

    if not wait_for_cassandra(port, timeout=600):
        stop_container(container_name)
        pytest.fail("Cassandra container failed to become ready")

    yield {
        "host": "localhost",
        "port": port,
        "keyspace": "test_keyspace",
        "container_name": container_name,
    }

    stop_container(container_name)


@pytest.fixture(scope="session")
def cassandra_with_iot_data(cassandra_container) -> Generator[dict, None, None]:
    """Cassandra container with sample IoT data pre-loaded."""
    keyspace = "iot_data"
    if not load_cassandra_iot_data(cassandra_container["port"], keyspace):
        pytest.fail("Failed to load Cassandra IoT data")

    yield {
        **cassandra_container,
        "keyspace": keyspace,
        "tables": ["sensor_readings", "devices", "alerts"],
    }


@pytest.fixture
def cassandra_port(cassandra_container) -> int:
    """Get Cassandra port."""
    return cassandra_container["port"]
