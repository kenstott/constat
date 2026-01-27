# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Server configuration for the Constat API server."""

import os

from pydantic import BaseModel, Field


def _get_bool_env(key: str, default: bool) -> bool:
    """Get a boolean value from environment variable."""
    value = os.environ.get(key)
    if value is None:
        return default
    return value.lower() in ("true", "1", "yes")


class ServerConfig(BaseModel):
    """Configuration for the Constat API server.

    Example YAML:
        server:
          host: 127.0.0.1
          port: 8000
          cors_origins:
            - http://localhost:5173
            - http://localhost:3000
          session_timeout_minutes: 60
          max_concurrent_sessions: 10
    """

    host: str = Field(
        default="127.0.0.1",
        description="Host address to bind the server to",
    )
    port: int = Field(
        default=8000,
        description="Port to listen on",
    )
    cors_origins: list[str] = Field(
        default=["http://localhost:5173", "http://localhost:3000"],
        description="Allowed CORS origins for the API",
    )
    session_timeout_minutes: int = Field(
        default=60,
        description="Session timeout in minutes before automatic cleanup",
    )
    max_concurrent_sessions: int = Field(
        default=10,
        description="Maximum number of concurrent sessions allowed",
    )
    require_plan_approval: bool = Field(
        default=True,
        description="Whether to require user approval for plans via WebSocket",
    )
    auth_disabled: bool = Field(
        default_factory=lambda: _get_bool_env("AUTH_DISABLED", True),
        description="Whether to disable authentication (uses 'default' user). Set AUTH_DISABLED=false to enable.",
    )
    firebase_project_id: str | None = Field(
        default_factory=lambda: os.environ.get("FIREBASE_PROJECT_ID"),
        description="Firebase project ID for JWT validation (required when auth enabled). Set FIREBASE_PROJECT_ID env var.",
    )
