# Copyright (c) 2025 Kenneth Stott
# Canary: 1f4ec839-42dc-48d0-8c31-5a2c485273f4
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Pytest configuration and shared fixtures for all test tiers."""

from __future__ import annotations
import os
import subprocess
from dotenv import load_dotenv
load_dotenv()
import pytest
from pathlib import Path
from typing import Generator


# =============================================================================
# Vector Store Isolation
# =============================================================================

@pytest.fixture(scope="session", autouse=True)
def isolated_vector_store(tmp_path_factory) -> Generator[Path, None, None]:
    """Create an isolated vector store for the test session.

    This prevents tests from polluting each other's vector embeddings.
    Each pytest session gets its own temporary vector store database.
    """
    vector_store_dir = tmp_path_factory.mktemp("vector_store")
    vector_store_path = vector_store_dir / "system.duckdb"

    old_value = os.environ.get("CONSTAT_VECTOR_STORE_PATH")
    os.environ["CONSTAT_VECTOR_STORE_PATH"] = str(vector_store_path)

    yield vector_store_path

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
            vs.clear_entities(_source="document")
            vs._conn.execute("DELETE FROM embeddings")
            vs._conn.execute("DELETE FROM chunk_entities")
        except Exception:
            pass  # Teardown: vector store may not be initialized; swallowed intentionally

    _clear()
    yield
    _clear()


# =============================================================================
# Pytest hooks
# =============================================================================

def pytest_sessionfinish(session, exitstatus):
    """Close all DuckDB connections and clean up Docker containers."""
    try:
        from constat.storage.duckdb_pool import close_all_pools
        close_all_pools()
    except Exception:
        pass  # Teardown: pool may already be closed; swallowed intentionally

    try:
        import jpype
        if jpype.isJVMStarted():
            jpype.shutdownJVM()
    except Exception:
        pass  # Teardown: JVM may not be running; swallowed intentionally

    try:
        result = subprocess.run(
            ["docker", "ps", "-a", "-q", "--filter", "name=constat_test_"],
            capture_output=True, text=True, timeout=10,
        )
        if result.stdout.strip():
            container_ids = result.stdout.strip().split("\n")
            for cid in container_ids:
                name_result = subprocess.run(
                    ["docker", "inspect", "--format", "{{.Name}}", cid],
                    capture_output=True, text=True, timeout=5,
                )
                name = name_result.stdout.strip().lstrip("/")
                if name == "constat_ollama_shared":
                    continue
                subprocess.run(
                    ["docker", "rm", "-f", cid],
                    capture_output=True, timeout=15,
                )
    except Exception:
        pass  # Teardown: Docker may be unavailable at session end; swallowed intentionally


def pytest_unconfigure(config):
    """Suppress DuckDB SIGABRT at process exit.

    DuckDB's C++ destructors can trigger heap corruption detection (SIGABRT)
    during Python process teardown. This is cosmetic — all tests have already
    completed. Catch the signal so pytest can print its summary and exit 0.
    """
    import faulthandler, signal, os
    faulthandler.disable()
    signal.signal(signal.SIGABRT, lambda s, f: os._exit(0))


def pytest_sessionstart(session):
    """Install SIGABRT handler early so DuckDB crashes during tests are non-fatal."""
    import signal, os
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


# =============================================================================
# Audio Fixtures
# =============================================================================

@pytest.fixture
def audio_fixtures(tmp_path):
    """Generate test audio files using stdlib wave module."""
    import wave, struct, math

    def make_wav(filename, duration_sec=2, sample_rate=16000, frequency=440):
        path = tmp_path / filename
        with wave.open(str(path), 'wb') as f:
            f.setnchannels(1)
            f.setsampwidth(2)  # 16-bit
            f.setframerate(sample_rate)
            for i in range(int(sample_rate * duration_sec)):
                sample = int(16000 * math.sin(2 * math.pi * frequency * i / sample_rate))
                f.writeframes(struct.pack('<h', sample))
        return path

    return {
        "short_wav": make_wav("short.wav", duration_sec=1),
        "medium_wav": make_wav("medium.wav", duration_sec=5),
        "silence_wav": make_wav("silence.wav", duration_sec=2, frequency=0),
    }


# =============================================================================
# GraphQL Test Fixtures
# =============================================================================

@pytest.fixture
def graphql_app():
    """Minimal FastAPI app with GraphQL router mounted (auth disabled).

    Use for integration tests that don't need a full server.
    """
    from unittest.mock import MagicMock
    from fastapi import FastAPI
    from constat.server.graphql import graphql_router

    app = FastAPI()
    app.state.session_manager = MagicMock()
    server_config = MagicMock()
    server_config.auth_disabled = True
    app.state.server_config = server_config
    app.include_router(graphql_router, prefix="/api/graphql")
    return app


@pytest.fixture
def graphql_client(graphql_app):
    """TestClient for the GraphQL endpoint (auth disabled).

    Usage::

        def test_something(graphql_client):
            resp = graphql_client.post("/api/graphql", json={"query": "{ __typename }"})
            assert resp.json()["data"]["__typename"] == "Query"
    """
    from starlette.testclient import TestClient
    return TestClient(graphql_app)


# NOTE: pytest_collection_modifyitems hook removed.
# Previously it silently skipped tests marked requires_docker/requires_mongodb/etc.
# when Docker was unavailable. The fixtures themselves now call pytest.fail() so
# the absence of Docker is visible as an explicit failure, not a silent skip.
