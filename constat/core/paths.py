# Copyright (c) 2025 Kenneth Stott
# Canary: 4f8464c7-77cc-431d-8492-e6749aa02d1a
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Centralized path helpers for per-user vault directories."""

from pathlib import Path


def user_vault_dir(data_dir: Path, user_id: str) -> Path:
    """Return the vault directory for a user: {data_dir}/{user_id}.vault/"""
    return data_dir / f"{user_id}.vault"


def migrate_db_name(directory: Path, old_name: str, new_name: str) -> Path:
    """Return path to *new_name*, renaming *old_name* on first access.

    If *new_name* already exists, return it.  If only *old_name* exists,
    atomically rename it (and its .wal sidecar if present).  Otherwise
    return *new_name* (will be created fresh by the caller).
    """
    new_path = directory / new_name
    if new_path.exists():
        return new_path
    old_path = directory / old_name
    if old_path.exists():
        old_path.rename(new_path)
        # Rename WAL sidecar if present
        old_wal = old_path.with_suffix(old_path.suffix + ".wal")
        if old_wal.exists():
            old_wal.rename(new_path.with_suffix(new_path.suffix + ".wal"))
        return new_path
    return new_path
