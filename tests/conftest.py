# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Pytest configuration and fixtures for integration tests."""

import os
import subprocess
import tempfile
import shutil
import time
import pytest
from pathlib import Path
from typing import Generator

# Generate a unique session ID for this pytest run to allow parallel execution
# Each pytest process gets its own containers and ports
_SESSION_ID = os.getpid()
_PORT_OFFSET = _SESSION_ID % 1000  # Use PID mod 1000 for port offset


# =============================================================================
# Vector Store Isolation
# =============================================================================

@pytest.fixture(scope="session", autouse=True)
def isolated_vector_store(tmp_path_factory) -> Generator[Path, None, None]:
    """Create an isolated vector store for the test session.

    This prevents tests from polluting each other's vector embeddings.
    Each pytest session gets its own temporary vector store database.
    """
    # Create a temporary directory for the vector store
    vector_store_dir = tmp_path_factory.mktemp("vector_store")
    vector_store_path = vector_store_dir / "vectors.duckdb"

    # Set environment variable so DuckDBVectorStore uses this path
    old_value = os.environ.get("CONSTAT_VECTOR_STORE_PATH")
    os.environ["CONSTAT_VECTOR_STORE_PATH"] = str(vector_store_path)

    yield vector_store_path

    # Restore original environment
    # Restore original environment
    if old_value is not None:
        os.environ["CONSTAT_VECTOR_STORE_PATH"] = old_value
    else:
        os.environ.pop("CONSTAT_VECTOR_STORE_PATH", None)


@pytest.fixture
def clear_document_embeddings():
    """Clear document embeddings from the vector store before each test.

    Use this fixture for tests that load documents and need a clean slate.
    This prevents document content from one test polluting another.
    """
    def _clear():
        try:
            from constat.discovery.vector_store import DuckDBVectorStore
            vs = DuckDBVectorStore()
            # Clear document-sourced entities (preserves schema entities)
            vs.clear_entities(_source="document")
            # Clear all document chunks from embeddings table
            vs._conn.execute("DELETE FROM embeddings")
            # Clear chunk_entities links
            vs._conn.execute("DELETE FROM chunk_entities")
        except Exception:
            pass  # Vector store may not be initialized yet

    _clear()
    yield
    _clear()


def _get_unique_port(base_port: int) -> int:
    """Get a unique port for this session based on base port."""
    return base_port + _PORT_OFFSET


def _get_unique_container_name(base_name: str) -> str:
    """Get a unique container name for this session."""
    return f"{base_name}_{_SESSION_ID}"


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
def pytest_sessionfinish(session, exitstatus):
    """Close all DuckDB connections before process exit to avoid SIGABRT."""
    try:
        from constat.storage.duckdb_pool import close_all_pools
        close_all_pools()
    except Exception:
        pass


def pytest_unconfigure(config):
    """Suppress DuckDB SIGABRT at process exit.

    DuckDB's C++ destructors can trigger heap corruption detection (SIGABRT)
    during Python process teardown. This is cosmetic — all tests have already
    completed. Catch the signal so pytest can print its summary and exit 0.
    """
    import faulthandler, signal, os
    faulthandler.disable()
    signal.signal(signal.SIGABRT, lambda s, f: os._exit(0))


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
    config.addinivalue_line(
        "markers", "requires_ollama: mark test as requiring Ollama"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "requires_cassandra: mark test as requiring Cassandra"
    )
    config.addinivalue_line(
        "markers", "requires_elasticsearch: mark test as requiring Elasticsearch"
    )
    config.addinivalue_line(
        "markers", "requires_mistral: mark test as requiring Mistral container"
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

    Uses unique container name and port per pytest session to allow parallel runs.
    """
    if not docker_available:
        pytest.skip("Docker not available")

    container_name = _get_unique_container_name("constat_test_mongodb")
    port = _get_unique_port(27017)

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

    Uses unique container name and port per pytest session to allow parallel runs.
    """
    if not docker_available:
        pytest.skip("Docker not available")

    container_name = _get_unique_container_name("constat_test_postgresql")
    port = _get_unique_port(5432)
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


# =============================================================================
# Ollama Fixtures
# =============================================================================

def is_ollama_running(port: int = 11434) -> bool:
    """Check if Ollama server is responding."""
    try:
        import httpx
        response = httpx.get(f"http://localhost:{port}/api/tags", timeout=5.0)
        return response.status_code == 200
    except Exception:
        return False


def get_ollama_models(port: int = 11434) -> list[str]:
    """Get list of available Ollama models."""
    try:
        import httpx
        response = httpx.get(f"http://localhost:{port}/api/tags", timeout=5.0)
        if response.status_code == 200:
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        pass
    return []


def pull_ollama_model(model: str, port: int = 11434, timeout: int = 600) -> bool:
    """Pull an Ollama model. This can take several minutes for large models."""
    try:
        import httpx
        # Use streaming to handle long downloads
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                f"http://localhost:{port}/api/pull",
                json={"name": model, "stream": False},
                timeout=timeout,
            )
            return response.status_code == 200
    except Exception as e:
        print(f"Failed to pull model {model}: {e}")
        return False


def wait_for_ollama(port: int = 11434, timeout: int = 60) -> bool:
    """Wait for Ollama server to be ready."""
    import time
    start = time.time()
    while time.time() - start < timeout:
        if is_ollama_running(port):
            return True
        time.sleep(2)
    return False


@pytest.fixture(scope="session")
def ollama_container(docker_available) -> Generator[dict, None, None]:
    """Start Ollama container for the test session.

    Yields connection info dict with keys: host, port, base_url

    Note: The first run will pull the test model which can take several minutes.

    Uses a SHARED container that persists across test runs:
    - Container name: constat_ollama_shared
    - Port: 11434 (default)
    - Volume: ollama_test_data (persists models)

    The container is NOT stopped after tests - it can be reused by subsequent runs.
    Ollama handles multiple concurrent connections fine.
    """
    if not docker_available:
        pytest.skip("Docker not available")

    # Use fixed name/port for shared instance
    container_name = "constat_ollama_shared"
    port = 11434
    test_model = "llama3.2:1b"  # Small model for faster testing

    # Check if container already running on the expected port
    if is_ollama_running(port):
        # Already running - just use it
        models = get_ollama_models(port)
        actual_model = test_model
        for m in models:
            if m.startswith(test_model.split(":")[0]):
                actual_model = m
                break

        yield {
            "host": "localhost",
            "port": port,
            "base_url": f"http://localhost:{port}/v1",
            "model": actual_model,
            "models": models,
            "container_name": container_name,
        }
        return  # Don't stop - leave running for future tests

    # Not running - start it
    # Remove any stopped container with same name
    subprocess.run(
        ["docker", "rm", "-f", container_name],
        capture_output=True,
        timeout=30,
    )

    # Start Ollama container
    cmd = [
        "docker", "run", "-d",
        "--name", container_name,
        "-p", f"{port}:11434",
        "-v", "ollama_test_data:/root/.ollama",  # Persist models between runs
        "ollama/ollama"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            pytest.skip(f"Failed to start Ollama container: {result.stderr}")
    except subprocess.TimeoutExpired:
        pytest.skip("Ollama container start timed out - image may need to be pulled manually: docker pull ollama/ollama")
    except FileNotFoundError as e:
        pytest.skip(f"Error starting Ollama container: {e}")

    # Wait for Ollama to be ready
    if not wait_for_ollama(port, timeout=60):
        stop_container(container_name)
        pytest.skip("Ollama container failed to start")

    # Check if test model is available, pull if needed
    models = get_ollama_models(port)
    if test_model not in models and not any(m.startswith(test_model.split(":")[0]) for m in models):
        print(f"Pulling Ollama model {test_model}... (this may take a few minutes)")
        if not pull_ollama_model(test_model, port, timeout=600):
            stop_container(container_name)
            pytest.skip(f"Failed to pull Ollama model: {test_model}")

    # Refresh model list after pull
    models = get_ollama_models(port)

    # Find the actual model name (might have different tag)
    actual_model = test_model
    for m in models:
        if m.startswith(test_model.split(":")[0]):
            actual_model = m
            break

    yield {
        "host": "localhost",
        "port": port,
        "base_url": f"http://localhost:{port}/v1",
        "model": actual_model,
        "models": models,
        "container_name": container_name,
    }

    # NOTE: Do NOT stop container - leave it running for future test runs
    # Ollama handles concurrent connections fine and reusing the container
    # avoids the overhead of restarting and re-pulling models


@pytest.fixture
def ollama_model(ollama_container) -> str:
    """Get the test model name from Ollama container."""
    return ollama_container["model"]


@pytest.fixture
def ollama_base_url(ollama_container) -> str:
    """Get Ollama base URL."""
    return ollama_container["base_url"]


# =============================================================================
# Mistral Fixtures (via Ollama - prefers local, falls back to Docker)
# =============================================================================

@pytest.fixture(scope="session")
def mistral_container(docker_available) -> Generator[dict, None, None]:
    """Provide Mistral Nemo via Ollama for the test session.

    Yields connection info dict with keys: host, port, base_url, model

    Strategy:
    1. Check if Ollama is running locally (preferred - no Docker overhead)
    2. If not, start Ollama via Docker as fallback

    Model: mistral-nemo (12B open-weight model)

    Note: First run will download the model (~7GB) which can take a few minutes.
    """
    port = 11434
    test_model = os.environ.get("MISTRAL_TEST_MODEL", "mistral")  # Default to 7B
    container_name = "constat_ollama_shared"
    started_docker = False

    # Strategy 1: Check for local Ollama (preferred)
    if is_ollama_running(port):
        print("Using local Ollama instance")
    else:
        # Strategy 2: Fall back to Docker
        if not docker_available:
            pytest.skip("Ollama not running locally and Docker not available")

        print("Local Ollama not found, starting via Docker...")

        # Remove any stopped container with same name
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True,
            timeout=30,
        )

        # Start Ollama container
        cmd = [
            "docker", "run", "-d",
            "--name", container_name,
            "-p", f"{port}:11434",
            "-v", "ollama_test_data:/root/.ollama",
            "ollama/ollama"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                pytest.skip(f"Failed to start Ollama container: {result.stderr}")
            started_docker = True
        except subprocess.TimeoutExpired:
            pytest.skip("Ollama container start timed out")
        except FileNotFoundError as e:
            pytest.skip(f"Error starting Ollama container: {e}")

        # Wait for Ollama to be ready
        if not wait_for_ollama(port, timeout=60):
            stop_container(container_name)
            pytest.skip("Ollama container failed to start")

    # Now Ollama is running (either local or Docker)
    models = get_ollama_models(port)

    # Check if mistral-nemo is available, pull if needed
    if test_model not in models and not any(m.startswith(test_model) for m in models):
        print(f"Pulling {test_model} model... (this may take a few minutes)")
        if not pull_ollama_model(test_model, port, timeout=600):
            if started_docker:
                stop_container(container_name)
            pytest.skip(f"Failed to pull Ollama model: {test_model}")
        models = get_ollama_models(port)

    # Find actual model name (might have tag)
    actual_model = test_model
    for m in models:
        if m.startswith(test_model):
            actual_model = m
            break

    yield {
        "host": "localhost",
        "port": port,
        "base_url": f"http://localhost:{port}/v1",
        "model": actual_model,
        "models": models,
        "container_name": container_name if started_docker else None,
        "is_docker": started_docker,
    }

    # Only stop container if we started it via Docker
    # Leave local Ollama running


@pytest.fixture
def mistral_model(mistral_container) -> str:
    """Get the test model name from Mistral container."""
    return mistral_container["model"]


@pytest.fixture
def mistral_base_url(mistral_container) -> str:
    """Get Mistral base URL."""
    return mistral_container["base_url"]


# =============================================================================
# Cassandra Fixtures
# =============================================================================

def is_cassandra_ready(port: int = 9042, timeout: int = 5) -> bool:
    """Check if Cassandra is ready to accept connections."""
    try:
        from cassandra.cluster import Cluster
        cluster = Cluster(["localhost"], port=port)
        session = cluster.connect()
        session.shutdown()
        cluster.shutdown()
        return True
    except Exception:
        return False


def wait_for_cassandra(port: int = 9042, timeout: int = 120) -> bool:
    """Wait for Cassandra to be ready."""
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

        # Create keyspace
        session.execute(f"""
            CREATE KEYSPACE IF NOT EXISTS {keyspace}
            WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': 1}}
        """)
        session.set_keyspace(keyspace)

        # Create tables with partition keys and clustering keys
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

        # Insert sample data - devices
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

        # Insert sample data - sensor readings
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

        # Insert sample data - alerts
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

    Uses unique container name and port per pytest session to allow parallel runs.
    Note: Cassandra can take 60-90 seconds to fully initialize.
    """
    if not docker_available:
        pytest.skip("Docker not available")

    container_name = _get_unique_container_name("constat_test_cassandra")
    port = _get_unique_port(9042)

    if not start_container(
        name=container_name,
        image="cassandra:latest",
        port_mapping=f"{port}:9042",
        env={
            "CASSANDRA_CLUSTER_NAME": "TestCluster",
            "MAX_HEAP_SIZE": "256M",
            "HEAP_NEWSIZE": "64M",
        },
        wait_seconds=30,  # Initial wait
    ):
        pytest.skip("Failed to start Cassandra container")

    # Cassandra needs extra time to initialize
    if not wait_for_cassandra(port, timeout=120):
        stop_container(container_name)
        pytest.skip("Cassandra container failed to become ready")

    yield {
        "host": "localhost",
        "port": port,
        "keyspace": "test_keyspace",
        "container_name": container_name,
    }

    # Cleanup: stop container after all tests
    stop_container(container_name)


@pytest.fixture(scope="session")
def cassandra_with_iot_data(cassandra_container) -> Generator[dict, None, None]:
    """Cassandra container with sample IoT data pre-loaded."""
    keyspace = "iot_data"
    if not load_cassandra_iot_data(cassandra_container["port"], keyspace):
        pytest.skip("Failed to load Cassandra IoT data")

    yield {
        **cassandra_container,
        "keyspace": keyspace,
        "tables": ["sensor_readings", "devices", "alerts"],
    }


@pytest.fixture
def cassandra_port(cassandra_container) -> int:
    """Get Cassandra port."""
    return cassandra_container["port"]


# =============================================================================
# Elasticsearch Fixtures
# =============================================================================

def is_elasticsearch_ready(port: int = 9200) -> bool:
    """Check if Elasticsearch is ready."""
    try:
        import httpx
        response = httpx.get(f"http://localhost:{port}/_cluster/health", timeout=5.0)
        if response.status_code == 200:
            data = response.json()
            return data.get("status") in ["green", "yellow"]
    except Exception:
        pass
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

        # Delete existing indices if they exist
        for index in ["products", "customers", "orders", "reviews"]:
            try:
                es.indices.delete(index=index)
            except Exception:
                pass

        # Create products index with mapping
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

        # Create customers index
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

        # Create orders index
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

        # Create reviews index
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

        # Insert products
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

        # Insert customers
        customers = [
            {"_index": "customers", "_id": "cust-001", "_source": {"name": "Alice Johnson", "email": "alice@example.com", "age": 32, "country": "USA", "membership": "gold", "total_spent": 5420.50, "joined_at": "2022-01-15"}},
            {"_index": "customers", "_id": "cust-002", "_source": {"name": "Bob Smith", "email": "bob@example.com", "age": 45, "country": "UK", "membership": "silver", "total_spent": 2150.00, "joined_at": "2022-03-20"}},
            {"_index": "customers", "_id": "cust-003", "_source": {"name": "Carol White", "email": "carol@example.com", "age": 28, "country": "Canada", "membership": "platinum", "total_spent": 12300.75, "joined_at": "2021-06-10"}},
            {"_index": "customers", "_id": "cust-004", "_source": {"name": "David Lee", "email": "david@example.com", "age": 38, "country": "USA", "membership": "gold", "total_spent": 4890.25, "joined_at": "2022-08-05"}},
            {"_index": "customers", "_id": "cust-005", "_source": {"name": "Emma Wilson", "email": "emma@example.com", "age": 25, "country": "Australia", "membership": "silver", "total_spent": 1560.00, "joined_at": "2023-01-20"}},
        ]
        bulk(es, customers)

        # Insert orders
        orders = [
            {"_index": "orders", "_id": "order-001", "_source": {"customer_id": "cust-001", "products": [{"product_id": "prod-001", "quantity": 1, "price": 2499.99}], "total": 2499.99, "status": "delivered", "payment_method": "credit_card", "created_at": "2023-09-01"}},
            {"_index": "orders", "_id": "order-002", "_source": {"customer_id": "cust-002", "products": [{"product_id": "prod-002", "quantity": 1, "price": 349.99}, {"product_id": "prod-004", "quantity": 1, "price": 139.99}], "total": 489.98, "status": "shipped", "payment_method": "paypal", "created_at": "2023-09-05"}},
            {"_index": "orders", "_id": "order-003", "_source": {"customer_id": "cust-003", "products": [{"product_id": "prod-003", "quantity": 1, "price": 1299.00}], "total": 1299.00, "status": "delivered", "payment_method": "credit_card", "created_at": "2023-09-10"}},
            {"_index": "orders", "_id": "order-004", "_source": {"customer_id": "cust-001", "products": [{"product_id": "prod-007", "quantity": 1, "price": 179.99}], "total": 179.99, "status": "processing", "payment_method": "credit_card", "created_at": "2023-09-15"}},
            {"_index": "orders", "_id": "order-005", "_source": {"customer_id": "cust-004", "products": [{"product_id": "prod-006", "quantity": 2, "price": 49.99}], "total": 99.98, "status": "delivered", "payment_method": "debit_card", "created_at": "2023-09-20"}},
        ]
        bulk(es, orders)

        # Insert reviews
        reviews = [
            {"_index": "reviews", "_id": "rev-001", "_source": {"product_id": "prod-001", "customer_id": "cust-001", "rating": 5, "title": "Best laptop ever!", "body": "This MacBook Pro is amazing. Fast, beautiful display, and great battery life.", "helpful_votes": 42, "created_at": "2023-09-15"}},
            {"_index": "reviews", "_id": "rev-002", "_source": {"product_id": "prod-002", "customer_id": "cust-002", "rating": 4, "title": "Great headphones", "body": "Excellent noise cancellation but a bit pricey.", "helpful_votes": 28, "created_at": "2023-09-18"}},
            {"_index": "reviews", "_id": "rev-003", "_source": {"product_id": "prod-003", "customer_id": "cust-003", "rating": 5, "title": "Worth every penny", "body": "My back pain has significantly reduced since using this chair.", "helpful_votes": 65, "created_at": "2023-09-20"}},
            {"_index": "reviews", "_id": "rev-004", "_source": {"product_id": "prod-004", "customer_id": "cust-004", "rating": 4, "title": "Great for reading", "body": "Love the warm light feature, makes reading at night much easier.", "helpful_votes": 15, "created_at": "2023-09-22"}},
            {"_index": "reviews", "_id": "rev-005", "_source": {"product_id": "prod-001", "customer_id": "cust-003", "rating": 5, "title": "Professional grade", "body": "Perfect for software development and video editing.", "helpful_votes": 38, "created_at": "2023-09-25"}},
        ]
        bulk(es, reviews)

        # Refresh indices to make data searchable immediately
        es.indices.refresh(index="_all")

        es.close()
        return True
    except Exception as e:
        print(f"Error loading Elasticsearch data: {e}")
        return False


@pytest.fixture(scope="session")
def elasticsearch_container(docker_available) -> Generator[dict, None, None]:
    """Start Elasticsearch container for the test session.

    Yields connection info dict with keys: host, port, url

    Uses unique container name and port per pytest session to allow parallel runs.
    """
    if not docker_available:
        pytest.skip("Docker not available")

    container_name = _get_unique_container_name("constat_test_elasticsearch")
    port = _get_unique_port(9200)

    # Start Elasticsearch with single-node mode and reduced memory
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
            "elasticsearch:8.11.0"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                pytest.skip(f"Failed to start Elasticsearch container: {result.stderr}")
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            pytest.skip(f"Error starting Elasticsearch container: {e}")

    # Wait for Elasticsearch to be ready
    if not wait_for_elasticsearch(port, timeout=90):
        stop_container(container_name)
        pytest.skip("Elasticsearch container failed to become ready")

    yield {
        "host": "localhost",
        "port": port,
        "url": f"http://localhost:{port}",
        "container_name": container_name,
    }

    # Cleanup: stop container after all tests
    stop_container(container_name)


@pytest.fixture(scope="session")
def elasticsearch_with_ecommerce_data(elasticsearch_container) -> Generator[dict, None, None]:
    """Elasticsearch container with sample e-commerce data pre-loaded."""
    if not load_elasticsearch_ecommerce_data(elasticsearch_container["port"]):
        pytest.skip("Failed to load Elasticsearch e-commerce data")

    yield {
        **elasticsearch_container,
        "indices": ["products", "customers", "orders", "reviews"],
    }


@pytest.fixture
def elasticsearch_url(elasticsearch_container) -> str:
    """Get Elasticsearch URL."""
    return elasticsearch_container["url"]


# Auto-skip tests based on markers
def pytest_collection_modifyitems(config, items):
    """Skip tests that require Docker if it's not available."""
    if DOCKER_AVAILABLE:
        return

    skip_docker = pytest.mark.skip(reason="Docker not available")
    skip_mongodb = pytest.mark.skip(reason="MongoDB requires Docker")
    skip_postgresql = pytest.mark.skip(reason="PostgreSQL requires Docker")
    skip_ollama = pytest.mark.skip(reason="Ollama requires Docker")
    skip_cassandra = pytest.mark.skip(reason="Cassandra requires Docker")
    skip_elasticsearch = pytest.mark.skip(reason="Elasticsearch requires Docker")
    skip_mistral = pytest.mark.skip(reason="Mistral requires Docker with GPU")

    for item in items:
        if "requires_docker" in item.keywords:
            item.add_marker(skip_docker)
        if "requires_mongodb" in item.keywords:
            item.add_marker(skip_mongodb)
        if "requires_postgresql" in item.keywords:
            item.add_marker(skip_postgresql)
        if "requires_ollama" in item.keywords:
            item.add_marker(skip_ollama)
        if "requires_cassandra" in item.keywords:
            item.add_marker(skip_cassandra)
        if "requires_elasticsearch" in item.keywords:
            item.add_marker(skip_elasticsearch)
        if "requires_mistral" in item.keywords:
            item.add_marker(skip_mistral)
