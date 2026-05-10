from __future__ import annotations

"""Shared Docker helper utilities for integration test fixtures."""

import os
import subprocess
import time

# Generate a unique session ID for this pytest run to allow parallel execution
_SESSION_ID = os.getpid()
_PORT_OFFSET = _SESSION_ID % 1000  # Use PID mod 1000 for port offset


def get_unique_port(base_port: int) -> int:
    """Get a unique port for this session based on base port."""
    return base_port + _PORT_OFFSET


def get_unique_container_name(base_name: str) -> str:
    """Get a unique container name for this session."""
    return f"{base_name}_{_SESSION_ID}"


def is_docker_available() -> bool:
    """Check if Docker is available and running."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=60,
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
