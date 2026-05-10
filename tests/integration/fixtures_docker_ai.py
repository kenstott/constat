from __future__ import annotations

"""Docker fixtures for AI inference services (Ollama, Mistral)."""

import os
import subprocess
import time
from typing import Generator

import pytest

from tests.integration.fixtures_docker_helpers import (
    stop_container,
)


# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------

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
        pass  # Probe: Ollama not reachable; return empty list as sentinel
    return []


def pull_ollama_model(model: str, port: int = 11434, timeout: int = 600) -> bool:
    """Pull an Ollama model. This can take several minutes for large models."""
    try:
        import httpx
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
    start = time.time()
    while time.time() - start < timeout:
        if is_ollama_running(port):
            return True
        time.sleep(2)
    return False


# ---------------------------------------------------------------------------
# Ollama fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def ollama_container(docker_available) -> Generator[dict, None, None]:
    """Start Ollama container for the test session.

    Yields connection info dict with keys: host, port, base_url, model, models

    Uses a SHARED container that persists across test runs:
    - Container name: constat_ollama_shared
    - Port: 11434 (default)
    - Volume: ollama_test_data (persists models)
    """
    if not docker_available:
        pytest.fail("Docker not available — install Docker to run this test")

    container_name = "constat_ollama_shared"
    port = 11434
    test_model = "llama3.2:1b"

    if is_ollama_running(port):
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
        return  # Don't stop — leave running for future tests

    subprocess.run(
        ["docker", "rm", "-f", container_name],
        capture_output=True,
        timeout=30,
    )

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
            pytest.fail(f"Failed to start Ollama container: {result.stderr}")
    except subprocess.TimeoutExpired:
        pytest.fail("Ollama container start timed out — pull ollama/ollama manually first")
    except FileNotFoundError as e:
        pytest.fail(f"Error starting Ollama container: {e}")

    if not wait_for_ollama(port, timeout=60):
        stop_container(container_name)
        pytest.fail("Ollama container failed to start")

    models = get_ollama_models(port)
    if test_model not in models and not any(m.startswith(test_model.split(":")[0]) for m in models):
        print(f"Pulling Ollama model {test_model}...")
        if not pull_ollama_model(test_model, port, timeout=600):
            stop_container(container_name)
            pytest.fail(f"Failed to pull Ollama model: {test_model}")

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
    # NOTE: Do NOT stop container — leave running for future test runs


@pytest.fixture
def ollama_model(ollama_container) -> str:
    """Get the test model name from Ollama container."""
    return ollama_container["model"]


@pytest.fixture
def ollama_base_url(ollama_container) -> str:
    """Get Ollama base URL."""
    return ollama_container["base_url"]


# ---------------------------------------------------------------------------
# Mistral fixture (via Ollama)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def mistral_container(docker_available) -> Generator[dict, None, None]:
    """Provide Mistral Nemo via Ollama for the test session.

    Yields connection info dict with keys: host, port, base_url, model

    Strategy:
    1. Check if Ollama is running locally (preferred)
    2. If not, start Ollama via Docker as fallback
    """
    port = 11434
    test_model = os.environ.get("MISTRAL_TEST_MODEL", "mistral")
    container_name = "constat_ollama_shared"
    started_docker = False

    if is_ollama_running(port):
        print("Using local Ollama instance")
    else:
        if not docker_available:
            pytest.fail("Ollama not running locally and Docker not available — install Docker or start Ollama")

        print("Local Ollama not found, starting via Docker...")

        subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True,
            timeout=30,
        )

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
                pytest.fail(f"Failed to start Ollama container: {result.stderr}")
            started_docker = True
        except subprocess.TimeoutExpired:
            pytest.fail("Ollama container start timed out")
        except FileNotFoundError as e:
            pytest.fail(f"Error starting Ollama container: {e}")

        if not wait_for_ollama(port, timeout=60):
            stop_container(container_name)
            pytest.fail("Ollama container failed to start")

    models = get_ollama_models(port)

    if test_model not in models and not any(m.startswith(test_model) for m in models):
        print(f"Pulling {test_model} model...")
        if not pull_ollama_model(test_model, port, timeout=600):
            if started_docker:
                stop_container(container_name)
            pytest.fail(f"Failed to pull Ollama model: {test_model}")
        models = get_ollama_models(port)

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
    # Only stop container if we started it via Docker; leave local Ollama running


@pytest.fixture
def mistral_model(mistral_container) -> str:
    """Get the test model name from Mistral container."""
    return mistral_container["model"]


@pytest.fixture
def mistral_base_url(mistral_container) -> str:
    """Get Mistral base URL."""
    return mistral_container["base_url"]
