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
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


def _get_bool_env(key: str) -> bool | None:
    """Get a boolean value from environment variable, or None if not set."""
    value = os.environ.get(key)
    if value is None:
        return None
    return value.lower() in ("true", "1", "yes")


class ServerConfig(BaseModel):
    """Configuration for the Constat API server.

    Can be configured via YAML file or environment variables.
    Environment variables take precedence over YAML values.

    Example YAML:
        server:
          host: 127.0.0.1
          port: 8000
          cors_origins:
            - http://localhost:5173
            - http://localhost:3000
          session_timeout_minutes: 60
          max_concurrent_sessions: 10
          auth_disabled: false
          firebase_project_id: my-firebase-project
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
        default=True,
        description="Whether to disable authentication (uses 'default' user)",
    )
    firebase_project_id: Optional[str] = Field(
        default=None,
        description="Firebase project ID for JWT validation (required when auth enabled)",
    )

    @model_validator(mode="after")
    def apply_env_overrides(self) -> "ServerConfig":
        """Apply environment variable overrides after model creation."""
        # AUTH_DISABLED env var overrides YAML/default
        auth_disabled_env = _get_bool_env("AUTH_DISABLED")
        if auth_disabled_env is not None:
            self.auth_disabled = auth_disabled_env

        # FIREBASE_PROJECT_ID env var overrides YAML/default
        firebase_env = os.environ.get("FIREBASE_PROJECT_ID")
        if firebase_env is not None:
            self.firebase_project_id = firebase_env

        return self

    @classmethod
    def from_yaml_data(cls, data: dict[str, Any] | None) -> "ServerConfig":
        """Create ServerConfig from parsed YAML data.

        Args:
            data: The 'server' section from the YAML config, or None

        Returns:
            ServerConfig with YAML values and env var overrides applied
        """
        if data is None:
            return cls()
        return cls.model_validate(data)
