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
    name: str
    uri: str  # SQLAlchemy URI (env vars already substituted)
    description: str = ""  # What this database contains/represents

    # Optional credentials (alternative to embedding in URI)
    username: Optional[str] = None
    password: Optional[str] = None

    # If True, credentials must be provided at runtime (user session credentials)
    requires_user_credentials: bool = False

    def get_connection_uri(self, user_credentials: Optional[DatabaseCredentials] = None) -> str:
        """
        Get the connection URI with credentials applied.

        Credential priority:
        1. User session credentials (if requires_user_credentials is True)
        2. Database config credentials (username/password fields)
        3. URI as-is (credentials embedded or no auth needed)

        Args:
            user_credentials: Optional credentials from user session

        Returns:
            Connection URI with credentials applied
        """
        # If requires user credentials, they must be provided
        if self.requires_user_credentials:
            if user_credentials is None or not user_credentials.is_complete():
                raise ValueError(
                    f"Database '{self.name}' requires user credentials but none were provided"
                )
            return self._inject_credentials(user_credentials.username, user_credentials.password)

        # Use config-level credentials if provided
        if self.username and self.password:
            return self._inject_credentials(self.username, self.password)

        # Return URI as-is
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


class ExecutionConfig(BaseModel):
    """Execution settings for generated code."""
    timeout_seconds: int = 60
    max_retries: int = 10
    allowed_imports: list[str] = Field(default_factory=list)


class UserConfig(BaseModel):
    """
    User-level configuration that can be merged with the global config.

    This allows users to:
    - Provide their own database credentials
    - Override specific database settings
    - Add user-specific customizations
    """
    # Database credentials by database name
    database_credentials: dict[str, DatabaseCredentials] = Field(default_factory=dict)

    # Optional database config overrides (merge with global)
    databases: list[DatabaseConfig] = Field(default_factory=list)


class Config(BaseModel):
    """Root configuration model."""
    model_config = {"extra": "ignore"}

    llm: LLMConfig = Field(default_factory=LLMConfig)
    databases: list[DatabaseConfig] = Field(default_factory=list)
    databases_description: str = ""  # Global context for all databases (e.g., "Each database represents a company")
    system_prompt: str = ""
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)

    # User credentials cache (populated when merge_user_config is called)
    # Excluded from serialization
    user_credentials_cache: dict[str, DatabaseCredentials] = Field(
        default_factory=dict,
        exclude=True,
        description="Internal cache for user credentials, populated by merge_user_config"
    )

    @classmethod
    def from_yaml(cls, path: str | Path, user_config: Optional["UserConfig"] = None) -> "Config":
        """
        Load config from YAML file with env var substitution.

        Args:
            path: Path to the main config YAML file
            user_config: Optional user config to merge (for credentials/overrides)

        Returns:
            Merged Config object
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            raw_content = f.read()

        # Substitute environment variables: ${VAR_NAME}
        substituted = _substitute_env_vars(raw_content)

        data = yaml.safe_load(substituted)
        config = cls.model_validate(data)

        # Merge user config if provided
        if user_config:
            config = config.merge_user_config(user_config)

        return config

    def merge_user_config(self, user_config: "UserConfig") -> "Config":
        """
        Merge user configuration into this config.

        User config can:
        - Provide credentials for databases
        - Override database settings
        - Add new databases

        Args:
            user_config: User configuration to merge

        Returns:
            New Config with user settings applied
        """
        # Create a copy of the config
        merged = self.model_copy(deep=True)

        # Store user credentials
        merged.user_credentials_cache = dict(user_config.database_credentials)

        # Merge database overrides
        for user_db in user_config.databases:
            # Find matching database by name
            found = False
            for i, db in enumerate(merged.databases):
                if db.name == user_db.name:
                    # Merge: user values override, but keep base values for unset fields
                    merged_db = db.model_copy(update={
                        k: v for k, v in user_db.model_dump().items()
                        if v is not None and v != "" and v != []
                    })
                    merged.databases[i] = merged_db
                    found = True
                    break

            if not found:
                # Add new database from user config
                merged.databases.append(user_db)

        return merged

    def get_database_credentials(self, database_name: str) -> Optional[DatabaseCredentials]:
        """
        Get credentials for a database.

        Checks user credentials first, then database config credentials.

        Args:
            database_name: Name of the database

        Returns:
            DatabaseCredentials or None if no credentials available
        """
        # Check user credentials first
        if database_name in self.user_credentials_cache:
            return self.user_credentials_cache[database_name]

        # Check database config
        for db in self.databases:
            if db.name == database_name:
                if db.username and db.password:
                    return DatabaseCredentials(username=db.username, password=db.password)
                break

        return None

    def get_database_config(self, name: str) -> Optional[DatabaseConfig]:
        """Get database config by name."""
        for db in self.databases:
            if db.name == name:
                return db
        return None


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
