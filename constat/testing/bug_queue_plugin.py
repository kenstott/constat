# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Pytest plugin that auto-files bugs on test failure into the BugQueue."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from constat.testing.bug_queue import BugQueue


def _get_commit_hash() -> str:
    """Get current git HEAD commit hash."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to get git commit hash: {result.stderr.strip()}")
    return result.stdout.strip()


def _detect_module(fspath: str | None) -> str | None:
    """Detect module from file path.

    Examples:
        constat/storage/foo.py -> storage
        constat/testing/bar.py -> testing
        tests/test_foo.py -> tests
    """
    if fspath is None:
        return None
    parts = Path(fspath).parts
    # Look for constat/ prefix
    for i, part in enumerate(parts):
        if part == "constat" and i + 1 < len(parts):
            next_part = parts[i + 1]
            if not next_part.endswith(".py"):
                return next_part
            return "constat"
    # Fallback: use first directory
    if len(parts) >= 2:
        return parts[-2]
    return None


def _build_environment() -> dict[str, Any]:
    """Capture environment metadata."""
    import duckdb

    return {
        "python_version": sys.version,
        "duckdb_version": duckdb.__version__,
        "platform": sys.platform,
    }


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register --bug-queue CLI flag."""
    parser.addoption(
        "--bug-queue",
        action="store_true",
        default=False,
        help="Enable automatic bug filing on test failures",
    )


def pytest_configure(config: pytest.Config) -> None:
    """If bug queue is active, instantiate BugQueue and attach to config."""
    active = config.getoption("--bug-queue", default=False) or os.environ.get(
        "CONSTAT_BUG_QUEUE"
    ) == "1"
    if not active:
        return

    db_path = Path(".constat") / "bugs.duckdb"
    bq = BugQueue(db_path)
    config._bug_queue = bq  # type: ignore[attr-defined]
    config._bug_queue_commit = _get_commit_hash()  # type: ignore[attr-defined]
    config._bug_queue_env = _build_environment()  # type: ignore[attr-defined]


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """File a bug on test call failures."""
    if report.when != "call" or not report.failed:
        return

    config = report.config if hasattr(report, "config") else None
    # pytest doesn't put config on report directly; get from the session
    if config is None:
        return

    bq: BugQueue | None = getattr(config, "_bug_queue", None)
    if bq is None:
        return

    # Extract error info from longreprtext
    longrepr = report.longreprtext if hasattr(report, "longreprtext") else ""
    error_type = "UnknownError"
    error_message = ""

    # Try to extract the exception class from the repr
    if longrepr:
        lines = longrepr.strip().splitlines()
        # Last line typically has "ErrorType: message"
        for line in reversed(lines):
            line = line.strip()
            if ": " in line and not line.startswith(" "):
                parts = line.split(": ", 1)
                error_type = parts[0]
                error_message = parts[1] if len(parts) > 1 else ""
                break
            elif line and not line.startswith(" "):
                error_type = line
                break

    fspath = str(report.fspath) if hasattr(report, "fspath") and report.fspath else None

    bq.upsert_bug(
        test_name=report.nodeid,
        test_file=fspath,
        module=_detect_module(fspath),
        error_type=error_type,
        error_message=error_message,
        traceback=longrepr,
        reproduction_cmd=f"python -m pytest {report.nodeid} -xvs",
        commit_hash=getattr(config, "_bug_queue_commit", "unknown"),
        environment=getattr(config, "_bug_queue_env", None),
    )


def pytest_unconfigure(config: pytest.Config) -> None:
    """Close bug queue connection on shutdown."""
    bq: BugQueue | None = getattr(config, "_bug_queue", None)
    if bq is not None:
        bq.close()
