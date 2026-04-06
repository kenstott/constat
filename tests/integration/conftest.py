# Copyright (c) 2025 Kenneth Stott
# Canary: c697f9c4-cf16-43ae-8b64-fd9fef2fff7f
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Integration test fixtures.

Manages a real Constat server for HTTP/GraphQL integration tests.
The server runs with auth_disabled=true so no Firebase credentials are needed.
Port is dynamically allocated to avoid conflicts.

Playwright/browser fixtures live in tests/e2e/conftest.py.
"""

from __future__ import annotations
import atexit
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

import pytest
import requests

from tests.integration.fixtures_docker_helpers import is_docker_available

# Register Docker fixture modules so their fixtures are available to all
# tests under tests/integration/ without explicit imports.
pytest_plugins = [
    "tests.integration.fixtures_docker_db",
    "tests.integration.fixtures_docker_ai",
    "tests.integration.fixtures_docker_search",
]

_SERVER_HOST = "127.0.0.1"
_STARTUP_TIMEOUT = 300  # seconds (warmup includes embedding model + whisper model + image/audio processing)


def _find_free_port() -> int:
    """Find a free TCP port by binding to port 0."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for_warmup(health_url: str, timeout: int = _STARTUP_TIMEOUT) -> bool:
    """Poll /health endpoint until warmup_complete is True or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(health_url, timeout=2)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("warmup_complete"):
                    return True
                if data.get("warmup_error"):
                    return False
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(0.5)
    return False


def _wait_for_http(url: str, timeout: int = _STARTUP_TIMEOUT) -> bool:
    """Poll an HTTP endpoint until it responds or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(url, timeout=2)
            if resp.status_code < 500:
                return True
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(0.5)
    return False


def _kill_process(proc):
    """Gracefully stop a process, then force-kill if needed."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=5)
        return
    except (ProcessLookupError, OSError):
        pass
    except subprocess.TimeoutExpired:
        pass
    # Force kill if SIGTERM didn't work
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, OSError):
        try:
            proc.kill()
        except (ProcessLookupError, OSError):
            pass
    try:
        proc.wait(timeout=5)
    except Exception:
        pass  # Teardown: proc.wait after SIGKILL may raise; swallowed intentionally


def _clean_stale_wal_files(data_dir: Path) -> None:
    """Remove stale .wal files left by crashed DuckDB processes."""
    constat_dir = data_dir / ".constat"
    if not constat_dir.is_dir():
        return
    for wal in constat_dir.rglob("*.wal"):
        wal.unlink(missing_ok=True)


@pytest.fixture(scope="session")
def docker_available() -> bool:
    """Check if Docker is available."""
    return is_docker_available()


@pytest.fixture(scope="session")
def integration_data_dir(tmp_path_factory):
    """Create a temporary .constat directory for integration tests."""
    data_dir = tmp_path_factory.mktemp("constat_integration")
    constat_dir = data_dir / ".constat"
    constat_dir.mkdir()
    return data_dir


@pytest.fixture(scope="session")
def server_port():
    """Allocate a random free port for the test server."""
    return _find_free_port()


@pytest.fixture(scope="session")
def server_process(integration_data_dir, server_port):
    """Start a Constat server for the test session.

    Uses a random port and registers atexit cleanup to guarantee
    the server is killed even if pytest crashes.
    """
    project_root = Path(__file__).parent.parent.parent
    demo_config = project_root / "demo" / "config.yaml"

    # Clean stale WAL files from previous test runs that may have crashed
    _clean_stale_wal_files(project_root)

    env = {
        **os.environ,
        "AUTH_DISABLED": "true",
        "CONSTAT_CONFIG": str(demo_config),
        "PYTHONPATH": str(project_root),
    }

    server_log = integration_data_dir / "server.log"
    log_fh = open(server_log, "w")
    proc = subprocess.Popen(
        [
            "python", "-m", "constat.cli", "serve",
            "-c", str(demo_config),
            "--host", _SERVER_HOST,
            "--port", str(server_port),
        ],
        cwd=str(project_root),
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,  # New process group for clean kill
    )

    # Register atexit to guarantee cleanup
    atexit.register(_kill_process, proc)

    base_url = f"http://{_SERVER_HOST}:{server_port}"
    if not _wait_for_http(f"{base_url}/api/sessions"):
        _kill_process(proc)
        log_fh.close()
        output = server_log.read_text(errors="replace")
        pytest.fail(f"Server failed to start within {_STARTUP_TIMEOUT}s.\nOutput:\n{output}")

    # Wait for warmup to complete (system DB must not be locked by warmup)
    if not _wait_for_warmup(f"{base_url}/health"):
        _kill_process(proc)
        log_fh.close()
        output = server_log.read_text(errors="replace")
        pytest.fail(f"Server warmup did not complete within {_STARTUP_TIMEOUT}s.\nOutput:\n{output}")

    yield proc

    _kill_process(proc)
    log_fh.close()


@pytest.fixture(scope="session")
def server_url(server_process, server_port):
    """Return the base URL of the running server."""
    return f"http://{_SERVER_HOST}:{server_port}"


# ---------------------------------------------------------------------------
# Glossary test fixtures (shared across integration test modules)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="class")
def session_id(server_url):
    """Create a session shared across tests in a class."""
    import uuid
    import requests as _requests
    body = {"session_id": str(uuid.uuid4())}
    resp = _requests.post(f"{server_url}/api/sessions", json=body)
    assert resp.status_code == 200, f"Create session failed: {resp.text}"
    sid = resp.json()["session_id"]
    yield sid
    _requests.delete(f"{server_url}/api/sessions/{sid}")
