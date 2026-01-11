"""Configuration loading with Pydantic validation and env var substitution."""

import os
import re
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class DatabaseCredentials(BaseModel):
    """Database credentials for authentication."""
    username: Optional[str] = None
    password: Optional[str] = None

    def is_complete(self) -> bool:
        """Check if both username and password are provided."""
        return self.username is not None and self.password is not None


class DatabaseConfig(BaseModel):
    """Database connection configuration."""
    # Note: 'name' is the dict key, not stored in the model
    uri: str  # SQLAlchemy URI (env vars already substituted)
    description: str = ""  # What this database contains/represents

    # Optional credentials (alternative to embedding in URI)
    # These can be provided in engine config or merged in from user config
    username: Optional[str] = None
    password: Optional[str] = None

    def get_connection_uri(self) -> str:
        """
        Get the connection URI with credentials applied.

        If username/password are set (from engine config or merged user config),
        they are injected into the URI. Otherwise returns URI as-is.

        Returns:
            Connection URI with credentials applied
        """
        # Use credentials if provided
        if self.username and self.password:
            return self._inject_credentials(self.username, self.password)

        # Return URI as-is (credentials embedded or no auth needed)
        return self.uri

    def _inject_credentials(self, username: str, password: str) -> str:
        """
        Inject credentials into the URI.

        Handles URIs like:
        - postgresql://localhost/db -> postgresql://user:pass@localhost/db
        - postgresql://@localhost/db -> postgresql://user:pass@localhost/db
        - postgresql://old:creds@localhost/db -> postgresql://user:pass@localhost/db
        """
        import urllib.parse
        from urllib.parse import quote_plus

        # Parse the URI
        parsed = urllib.parse.urlparse(self.uri)

        # Build new netloc with credentials
        # Quote special characters in username/password
        safe_user = quote_plus(username)
        safe_pass = quote_plus(password)

        if parsed.port:
            new_netloc = f"{safe_user}:{safe_pass}@{parsed.hostname}:{parsed.port}"
        else:
            new_netloc = f"{safe_user}:{safe_pass}@{parsed.hostname}"

        # Reconstruct URI
        new_parsed = parsed._replace(netloc=new_netloc)
        return urllib.parse.urlunparse(new_parsed)


class LLMTiersConfig(BaseModel):
    """Model tiering for cost optimization."""
    planning: str = "claude-sonnet-4-20250514"
    codegen: str = "claude-sonnet-4-20250514"
    simple: str = "claude-3-5-haiku-20241022"


class LLMConfig(BaseModel):
    """LLM provider configuration."""
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    api_key: Optional[str] = None
    tiers: Optional[LLMTiersConfig] = None

    def get_model(self, tier: str = "default") -> str:
        """
        Get the model for a specific tier.

        Args:
            tier: One of "planning", "codegen", "simple", or "default"

        Returns:
            Model name to use
        """
        if tier == "default" or self.tiers is None:
            return self.model

        if tier == "planning":
            return self.tiers.planning
        elif tier == "codegen":
            return self.tiers.codegen
        elif tier == "simple":
            return self.tiers.simple
        else:
            return self.model


class StorageConfig(BaseModel):
    """Storage configuration for artifact store."""
    # SQLAlchemy URI for the artifact store
    # Default: Uses DuckDB file per session
    # Options:
    #   - duckdb:///path/to/file.duckdb (embedded, single-user)
    #   - postgresql://user:pass@host:port/db (production, multi-user)
    #   - sqlite:///path/to/file.db (embedded alternative)
    artifact_store_uri: Optional[str] = None


class APIConfig(BaseModel):
    """External API configuration (GraphQL or REST)."""
    url: str  # Endpoint URL
    type: str = "graphql"  # graphql | rest
    description: str = ""  # What this API provides
    headers: dict[str, str] = Field(default_factory=dict)  # Auth headers, etc.


class ExecutionConfig(BaseModel):
    """Execution settings for generated code."""
    timeout_seconds: int = 60
    max_retries: int = 10
    allowed_imports: list[str] = Field(default_factory=list)


class Config(BaseModel):
    """Root configuration model."""
    model_config = {"extra": "ignore"}

    llm: LLMConfig = Field(default_factory=LLMConfig)

    # Databases keyed by name for easy merging
    # YAML format: databases: {main: {uri: ...}, analytics: {uri: ...}}
    databases: dict[str, DatabaseConfig] = Field(default_factory=dict)

    # External APIs keyed by name
    # YAML format: apis: {countries: {url: ..., type: graphql}}
    apis: dict[str, APIConfig] = Field(default_factory=dict)

    databases_description: str = ""  # Global context for all databases
    system_prompt: str = ""
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        user_config_path: Optional[str | Path] = None,
        user_config: Optional[dict] = None,
    ) -> "Config":
        """
        Load config from YAML file with env var substitution.

        User config can be provided via file or dict. It uses the same structure
        as the main config and is merged in (user values override engine values).

        Args:
            path: Path to the main config YAML file
            user_config_path: Optional path to user config YAML file
            user_config: Optional user config dict (same structure as YAML)

        Returns:
            Merged Config object

        Example:
            # With user config file
            config = Config.from_yaml("config.yaml", user_config_path="user-config.yaml")

            # With user config dict
            config = Config.from_yaml("config.yaml", user_config={
                "databases": [
                    {"name": "main", "username": "alice", "password": "secret"}
                ]
            })
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            raw_content = f.read()

        # Substitute environment variables: ${VAR_NAME}
        substituted = _substitute_env_vars(raw_content)
        data = yaml.safe_load(substituted)

        # Load user config from file if provided
        user_data = None
        if user_config_path:
            user_path = Path(user_config_path)
            if user_path.exists():
                with open(user_path) as f:
                    user_raw = f.read()
                user_substituted = _substitute_env_vars(user_raw)
                user_data = yaml.safe_load(user_substituted)
        elif user_config:
            user_data = user_config

        # Merge user config into engine config
        if user_data:
            data = cls._merge_configs(data, user_data)

        return cls.model_validate(data)

    @staticmethod
    def _merge_configs(engine: dict, user: dict) -> dict:
        """
        Merge user config into engine config.

        User values override engine values. Databases and APIs (dicts) are deep-merged by key.

        Args:
            engine: Engine config dict
            user: User config dict

        Returns:
            Merged config dict
        """
        merged = dict(engine)

        # Merge databases (dict keyed by name)
        if "databases" in user:
            engine_dbs = dict(merged.get("databases", {}))

            for db_name, user_db in user["databases"].items():
                if db_name in engine_dbs:
                    # Merge: user values override engine values
                    engine_dbs[db_name] = {**engine_dbs[db_name], **user_db}
                else:
                    # Add new database from user config
                    engine_dbs[db_name] = user_db

            merged["databases"] = engine_dbs

        # Merge APIs (dict keyed by name)
        if "apis" in user:
            engine_apis = dict(merged.get("apis", {}))

            for api_name, user_api in user["apis"].items():
                if api_name in engine_apis:
                    engine_apis[api_name] = {**engine_apis[api_name], **user_api}
                else:
                    engine_apis[api_name] = user_api

            merged["apis"] = engine_apis

        # Merge other top-level keys (user overrides engine)
        for key in user:
            if key not in ("databases", "apis"):
                if key in merged and isinstance(merged[key], dict) and isinstance(user[key], dict):
                    # Deep merge dicts
                    merged[key] = {**merged[key], **user[key]}
                else:
                    merged[key] = user[key]

        return merged

    def get_database(self, name: str) -> Optional[DatabaseConfig]:
        """Get database config by name."""
        return self.databases.get(name)


def _substitute_env_vars(content: str) -> str:
    """Replace ${VAR_NAME} with environment variable values."""
    pattern = re.compile(r'\$\{([^}]+)\}')

    def replacer(match: re.Match) -> str:
        var_name = match.group(1)
        value = os.environ.get(var_name)
        if value is None:
            raise ValueError(f"Environment variable not set: {var_name}")
        return value

    return pattern.sub(replacer, content)
