# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Centralized path helpers for per-user vault directories."""

from pathlib import Path


def user_vault_dir(data_dir: Path, user_id: str) -> Path:
    """Return the vault directory for a user: {data_dir}/{user_id}.vault/"""
    return data_dir / f"{user_id}.vault"
