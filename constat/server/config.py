# Copyright (c) 2025 Kenneth Stott
# Canary: 98e0cae7-2cac-43e6-a941-6ad83f76592d
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Server configuration for the Constat API server."""

import os
from pathlib import Path
from typing import Any, Optional

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class Persona(StrEnum):
    """Platform persona — controls UI visibility, write access, and feedback permissions."""

    PLATFORM_ADMIN = "platform_admin"
    DOMAIN_BUILDER = "domain_builder"
    SME = "sme"
    DOMAIN_USER = "domain_user"
    VIEWER = "viewer"


def _get_bool_env(key: str) -> bool | None:
    """Get a boolean value from environment variable, or None if not set."""
    value = os.environ.get(key)
    if value is None:
        return None
    return value.lower() in ("true", "1", "yes")


class LocalUser(BaseModel):
    """A local user for server-local authentication."""

    password_hash: str
    email: str = ""


class UserPermissions(BaseModel):
    """Permissions for a single user."""

    persona: str = Field(
        default=Persona.VIEWER,
        description="Platform persona (platform_admin, domain_builder, sme, domain_user, viewer)",
    )
    domains: list[str] = Field(
        default_factory=list,
        description="Domain filenames user can access (empty = none, unless platform_admin)",
    )

    databases: list[str] = Field(
        default_factory=list,
        description="Database names user can query (empty = none, unless platform_admin)",
    )
    documents: list[str] = Field(
        default_factory=list,
        description="Document names user can search (empty = none, unless platform_admin)",
    )
    apis: list[str] = Field(
        default_factory=list,
        description="API names user can call (empty = none, unless platform_admin)",
    )
    skills: list[str] = Field(
        default_factory=list,
        description="Skill names user can access",
    )
    agents: list[str] = Field(
        default_factory=list,
        description="Agent names user can access",
    )
    rules: list[str] = Field(
        default_factory=list,
        description="Rule IDs user can access",
    )
    facts: list[str] = Field(
        default_factory=list,
        description="Fact names user can access",
    )


class PermissionsConfig(BaseModel):
    """User permissions configuration."""

    users: dict[str, UserPermissions] = Field(
        default_factory=dict,
        description="Per-user permissions keyed by user ID (stable identifier)",
    )
    default: UserPermissions = Field(
        default_factory=UserPermissions,
        description="Default permissions for users not explicitly listed",
    )

    def get_user_permissions(self, user_id: str = "") -> UserPermissions:
        """Get permissions for a user by ID."""
        if user_id and user_id in self.users:
            return self.users[user_id]
        return self.default


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
        description="Whether to require user approval for plans via GraphQL subscription",
    )
    auth_disabled: bool = Field(
        default=True,
        description="Whether to disable authentication (uses 'default' user)",
    )
    firebase_project_id: Optional[str] = Field(
        default=None,
        description="Firebase project ID for JWT validation (required when auth enabled)",
    )
    firebase_api_key: Optional[str] = Field(
        default=None,
        description="Firebase Web API key for server-side email/password login",
    )
    admin_token: Optional[str] = Field(
        default=None,
        description="Admin token for local CLI/script access (bypasses Firebase auth)",
    )
    local_users: dict[str, LocalUser] = Field(
        default_factory=dict,
        description="Local users for server-local auth (keyed by username)",
    )

    # OAuth2 credentials for email (IMAP) browser flow
    google_email_client_id: Optional[str] = None
    google_email_client_secret: Optional[str] = None
    microsoft_email_client_id: Optional[str] = None
    microsoft_email_client_secret: Optional[str] = None
    microsoft_email_tenant_id: str = "common"

    # Microsoft SSO (Azure AD / Entra ID)
    microsoft_auth_client_id: Optional[str] = Field(default=None, description="Azure AD app registration client ID for SSO")
    microsoft_auth_client_secret: Optional[str] = Field(default=None, description="Azure AD client secret for SSO")
    microsoft_auth_tenant_id: str = Field(default="common", description="Azure AD tenant ID (or 'common' for multi-tenant)")

    source_refresh_interval_seconds: int = Field(
        default=900,
        description="Background source refresh interval in seconds",
    )
    runtime_dir: str = Field(
        default=".",
        description="Base directory for runtime data. Data is stored in {runtime_dir}/.constat/",
    )

    @property
    def data_dir(self) -> Path:
        """Get the full data directory path ({runtime_dir}/.constat)."""
        return Path(self.runtime_dir) / ".constat"
    permissions: PermissionsConfig = Field(
        default_factory=PermissionsConfig,
        description="User permissions configuration",
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

        # FIREBASE_API_KEY env var overrides YAML/default
        firebase_api_key_env = os.environ.get("FIREBASE_API_KEY")
        if firebase_api_key_env is not None:
            self.firebase_api_key = firebase_api_key_env

        # CONSTAT_ADMIN_TOKEN env var overrides YAML/default
        admin_token_env = os.environ.get("CONSTAT_ADMIN_TOKEN")
        if admin_token_env is not None:
            self.admin_token = admin_token_env

        # OAuth2 email credentials env var overrides
        google_email_client_id_env = os.environ.get("GOOGLE_EMAIL_CLIENT_ID")
        if google_email_client_id_env is not None:
            self.google_email_client_id = google_email_client_id_env

        google_email_client_secret_env = os.environ.get("GOOGLE_EMAIL_CLIENT_SECRET")
        if google_email_client_secret_env is not None:
            self.google_email_client_secret = google_email_client_secret_env

        microsoft_email_client_id_env = os.environ.get("MICROSOFT_EMAIL_CLIENT_ID")
        if microsoft_email_client_id_env is not None:
            self.microsoft_email_client_id = microsoft_email_client_id_env

        microsoft_email_client_secret_env = os.environ.get("MICROSOFT_EMAIL_CLIENT_SECRET")
        if microsoft_email_client_secret_env is not None:
            self.microsoft_email_client_secret = microsoft_email_client_secret_env

        microsoft_email_tenant_id_env = os.environ.get("MICROSOFT_EMAIL_TENANT_ID")
        if microsoft_email_tenant_id_env is not None:
            self.microsoft_email_tenant_id = microsoft_email_tenant_id_env

        for env_key, attr in [
            ("MICROSOFT_AUTH_CLIENT_ID", "microsoft_auth_client_id"),
            ("MICROSOFT_AUTH_CLIENT_SECRET", "microsoft_auth_client_secret"),
            ("MICROSOFT_AUTH_TENANT_ID", "microsoft_auth_tenant_id"),
        ]:
            val = os.environ.get(env_key)
            if val is not None:
                setattr(self, attr, val)

        source_refresh_env = os.environ.get("CONSTAT_SOURCE_REFRESH_INTERVAL")
        if source_refresh_env is not None:
            self.source_refresh_interval_seconds = int(source_refresh_env)

        # Merge persisted local_users.yaml (from self-registration) into local_users
        users_file = self.data_dir / "local_users.yaml"
        if users_file.exists():
            import yaml
            with open(users_file) as f:
                persisted = yaml.safe_load(f) or {}
            for uname, udata in persisted.items():
                if uname not in self.local_users:
                    self.local_users[uname] = LocalUser(**udata)

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
