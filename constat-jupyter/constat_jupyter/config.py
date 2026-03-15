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
