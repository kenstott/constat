# Copyright (c) 2025 Kenneth Stott
# Canary: 65293771-2689-429d-9702-db9368a8e9c5
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class ConstatConfig:
    server_url: str = "http://localhost:8000"
    token: str | None = None

    @classmethod
    def resolve(
        cls,
        server_url: str | None = None,
        token: str | None = None,
    ) -> ConstatConfig:
        """Resolve config from constructor args > environment variables."""
        return cls(
            server_url=server_url or os.environ.get("CONSTAT_SERVER_URL", "http://localhost:8000"),
            token=token or os.environ.get("CONSTAT_AUTH_TOKEN"),
        )
