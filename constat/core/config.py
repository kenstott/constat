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
        """Check if both username and password are provided and non-empty."""
        return bool(self.username) and bool(self.password)


class DatabaseConfig(BaseModel):
    """Database connection configuration.

    Supports SQL databases (via SQLAlchemy URI) and NoSQL databases
    (via type-specific options).

    SQL Example:
        databases:
          main:
            uri: postgresql://localhost/mydb
            description: Main application database

    MongoDB Example:
        databases:
          mongo:
            type: mongodb
            uri: mongodb://localhost:27017
            database: mydb
            description: Document store

    Cassandra Example:
        databases:
          cassandra:
            type: cassandra
            keyspace: my_keyspace
            hosts: [node1, node2]
            username: ${CASSANDRA_USER}
            password: ${CASSANDRA_PASS}

    DynamoDB Example:
        databases:
          dynamo:
            type: dynamodb
            region: us-east-1
            profile_name: myprofile

    Elasticsearch Example:
        databases:
          elastic:
            type: elasticsearch
            hosts: [http://localhost:9200]
            api_key: ${ES_API_KEY}

    CosmosDB Example:
        databases:
          cosmos:
            type: cosmosdb
            endpoint: https://myaccount.documents.azure.com
            key: ${COSMOS_KEY}
            database: mydb
            container: mycontainer

    Firestore Example:
        databases:
          firestore:
            type: firestore
            project: my-gcp-project
            collection: users
            credentials_path: /path/to/credentials.json
    """
    model_config = {"extra": "allow"}  # Allow type-specific fields

    # Database type: sql (default), mongodb, cassandra, elasticsearch, dynamodb, cosmosdb, firestore
    type: str = "sql"

    # Common fields
    description: str = ""

    # SQL databases (SQLAlchemy)
    uri: Optional[str] = None

    # Credentials (used by SQL and some NoSQL)
    username: Optional[str] = None
    password: Optional[str] = None

    # MongoDB
    database: Optional[str] = None  # Also used by CosmosDB
    sample_size: int = 100  # For schema inference

    # Cassandra
    keyspace: Optional[str] = None
    hosts: Optional[list[str]] = None
    port: Optional[int] = None
    cloud_config: Optional[dict] = None  # For DataStax Astra

    # DynamoDB
    region: Optional[str] = None
    endpoint_url: Optional[str] = None  # For local DynamoDB
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_session_token: Optional[str] = None
    profile_name: Optional[str] = None

    # Elasticsearch
    api_key: Optional[str] = None
    # hosts: already defined above

    # CosmosDB
    endpoint: Optional[str] = None
    key: Optional[str] = None
    container: Optional[str] = None
    # database: already defined above

    # Firestore
    project: Optional[str] = None
    collection: Optional[str] = None
    credentials_path: Optional[str] = None

    def get_connection_uri(self) -> str:
        """
        Get the connection URI with credentials applied.

        Only valid for SQL and MongoDB databases.

        Returns:
            Connection URI with credentials applied
        """
        if self.type not in ("sql", "mongodb"):
            raise ValueError(f"get_connection_uri() not supported for type: {self.type}")

        if not self.uri:
            raise ValueError("URI not configured")

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
        - mongodb://host1:27017,host2:27017/db -> mongodb://user:pass@host1:27017,host2:27017/db
        """
        import urllib.parse
        from urllib.parse import quote_plus

        # Parse the URI
        parsed = urllib.parse.urlparse(self.uri)

        # Build new netloc with credentials
        # Quote special characters in username/password
        safe_user = quote_plus(username)
        safe_pass = quote_plus(password)

        # Extract the host portion from netloc (strip any existing credentials)
        netloc = parsed.netloc
        if "@" in netloc:
            # Remove existing credentials (everything before @)
            host_part = netloc.split("@", 1)[1]
        else:
            host_part = netloc

        # Build new netloc with credentials and host
        new_netloc = f"{safe_user}:{safe_pass}@{host_part}"

        # Reconstruct URI
        new_parsed = parsed._replace(netloc=new_netloc)
        return urllib.parse.urlunparse(new_parsed)

    def is_nosql(self) -> bool:
        """Check if this is a NoSQL database."""
        return self.type in ("mongodb", "cassandra", "elasticsearch", "dynamodb", "cosmosdb", "firestore")


class TierConfig(BaseModel):
    """Configuration for a single tier (provider + model).

    Can be specified as just a model string (uses default provider)
    or as a full config with provider override.

    Examples:
        # Just model (uses default provider)
        planning: "claude-opus-4-20250514"

        # Full config with provider override
        simple:
          provider: ollama
          model: llama3.2:3b
    """
    provider: Optional[str] = None  # None means use default provider
    model: str

    # Provider-specific options (e.g., base_url for Ollama)
    base_url: Optional[str] = None


class LLMTiersConfig(BaseModel):
    """Model tiering for cost optimization.

    Each tier can be:
    - A model string (uses default provider)
    - A TierConfig object with provider override

    Examples:
        tiers:
          planning: claude-opus-4-20250514  # Just model
          codegen: claude-sonnet-4-20250514
          simple:
            provider: ollama  # Provider override
            model: llama3.2:3b
    """
    planning: str | TierConfig = "claude-sonnet-4-20250514"
    codegen: str | TierConfig = "claude-sonnet-4-20250514"
    simple: str | TierConfig = "claude-3-5-haiku-20241022"

    def get_tier_config(self, tier: str) -> TierConfig:
        """Get the TierConfig for a tier, normalizing string to TierConfig."""
        value = getattr(self, tier, None)
        if value is None:
            raise ValueError(f"Unknown tier: {tier}")
        if isinstance(value, str):
            return TierConfig(model=value)
        return value


class LLMConfig(BaseModel):
    """LLM provider configuration."""
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    api_key: Optional[str] = None
    tiers: Optional[LLMTiersConfig] = None

    # Provider-specific options
    base_url: Optional[str] = None  # For Ollama or custom endpoints

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
            tier_config = self.tiers.get_tier_config("planning")
            return tier_config.model
        elif tier == "codegen":
            tier_config = self.tiers.get_tier_config("codegen")
            return tier_config.model
        elif tier == "simple":
            tier_config = self.tiers.get_tier_config("simple")
            return tier_config.model
        else:
            return self.model

    def get_tier_config(self, tier: str = "default") -> TierConfig:
        """
        Get the full tier configuration (provider + model) for a tier.

        Args:
            tier: One of "planning", "codegen", "simple", or "default"

        Returns:
            TierConfig with provider (or None for default) and model
        """
        if tier == "default" or self.tiers is None:
            return TierConfig(provider=None, model=self.model, base_url=self.base_url)

        if tier in ("planning", "codegen", "simple"):
            tier_config = self.tiers.get_tier_config(tier)
            # If tier doesn't override provider, it uses None (meaning default)
            return tier_config
        else:
            return TierConfig(provider=None, model=self.model, base_url=self.base_url)


class StorageConfig(BaseModel):
    """Storage configuration for artifact store."""
    # SQLAlchemy URI for the artifact store
    # Default: Uses DuckDB file per session
    # Options:
    #   - duckdb:///path/to/file.duckdb (embedded, single-user)
    #   - postgresql://user:pass@host:port/db (production, multi-user)
    #   - sqlite:///path/to/file.db (embedded alternative)
    artifact_store_uri: Optional[str] = None


class DocumentConfig(BaseModel):
    """Reference document configuration for inclusion in reasoning.

    Documents provide domain knowledge, business rules, API documentation,
    or any reference material the LLM should consider during analysis.

    File Example:
        documents:
          business_rules:
            type: file
            path: ./docs/business-rules.md
            description: "Revenue calculation rules and thresholds"

    HTTP Example (works for wiki pages, GitHub raw files, etc.):
        documents:
          wiki_page:
            type: http
            url: https://wiki.example.com/api/v2/pages/12345/export/view
            headers:
              Authorization: Bearer ${WIKI_TOKEN}
            description: "Data dictionary from wiki"

    Confluence Example:
        documents:
          confluence:
            type: confluence
            url: https://mycompany.atlassian.net
            space_key: ANALYTICS
            page_title: "Business Rules"
            username: ${CONFLUENCE_USER}
            api_token: ${CONFLUENCE_TOKEN}

    Inline Example:
        documents:
          glossary:
            type: inline
            content: |
              ## Key Terms
              - VIP: Customer with lifetime value > $100k
              - Churn: Customer inactive for 90+ days
            description: "Business glossary"
    """
    # Acquisition type: file | http | inline | confluence | notion
    type: str = "file"

    # For type=file
    path: Optional[str] = None  # Local file path (supports glob: ./docs/*.md)

    # For type=http
    url: Optional[str] = None  # URL to fetch
    headers: dict[str, str] = Field(default_factory=dict)  # HTTP headers

    # For type=confluence
    space_key: Optional[str] = None  # Confluence space key
    page_title: Optional[str] = None  # Page title to fetch
    page_id: Optional[str] = None  # Or page ID directly
    username: Optional[str] = None  # Confluence username
    api_token: Optional[str] = None  # Confluence API token

    # For type=notion
    page_url: Optional[str] = None  # Notion page URL
    notion_token: Optional[str] = None  # Notion integration token

    # For type=inline
    content: Optional[str] = None  # Inline content

    # Metadata
    description: str = ""  # What this document contains
    format: str = "auto"  # auto | markdown | text | html | pdf | docx | xlsx | pptx
    tags: list[str] = Field(default_factory=list)  # For categorization/search

    # PDF/Office extraction options
    extract_tables: bool = True  # Extract tables from PDFs/docs as markdown
    extract_images: bool = False  # Extract and describe images (requires vision model)
    page_range: Optional[str] = None  # e.g., "1-5" or "1,3,5-10" for PDFs

    # Loading behavior
    cache: bool = True  # Cache fetched content
    cache_ttl: Optional[int] = None  # Cache TTL in seconds (None = forever)

    # Link following (build corpus from linked documents)
    follow_links: bool = False  # Follow links in document to build corpus
    max_depth: int = 2  # Max link depth to follow (1 = direct links only)
    max_documents: int = 20  # Max documents to fetch via link following
    link_pattern: Optional[str] = None  # Regex to filter which links to follow
    same_domain_only: bool = True  # Only follow links to same domain


class APIConfig(BaseModel):
    """External API configuration (GraphQL or OpenAPI).

    GraphQL Example:
        apis:
          countries:
            type: graphql
            url: https://countries.trevorblades.com/graphql

    OpenAPI Example (auto-discovers endpoints from spec URL):
        apis:
          petstore:
            type: openapi
            spec_url: https://petstore.swagger.io/v2/swagger.json

    OpenAPI from local file:
        apis:
          internal:
            type: openapi
            spec_path: ./specs/internal-api.yaml

    OpenAPI inline (for simple APIs without a spec file):
        apis:
          simple_api:
            type: openapi
            url: https://api.example.com
            spec_inline:
              openapi: "3.0.0"
              info:
                title: Simple API
                version: "1.0"
              paths:
                /users/{userId}:
                  get:
                    operationId: getUser
                    parameters:
                      - name: userId
                        in: path
                        required: true
                        schema:
                          type: string
                    responses:
                      "200":
                        description: User details

    Auth can be provided at multiple levels:
    1. In headers (engine config) - shared across all users
    2. In user config - user-specific credentials override engine config
    """
    # API type: graphql | openapi
    type: str = "graphql"

    # Base URL for the API
    url: Optional[str] = None

    # OpenAPI spec location (for type=openapi)
    spec_url: Optional[str] = None  # URL to download OpenAPI spec
    spec_path: Optional[str] = None  # Local path to OpenAPI spec file
    spec_inline: Optional[dict] = None  # Inline OpenAPI spec (embedded in config)

    description: str = ""  # What this API provides
    headers: dict[str, str] = Field(default_factory=dict)  # Auth headers, etc.

    # Auth configuration (alternative to headers)
    auth_type: Optional[str] = None  # bearer | basic | api_key
    auth_token: Optional[str] = None  # Token for bearer auth
    auth_username: Optional[str] = None  # Username for basic auth
    auth_password: Optional[str] = None  # Password for basic auth
    api_key: Optional[str] = None  # API key value
    api_key_header: str = "X-API-Key"  # Header name for API key


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

    # Reference documents keyed by name
    # YAML format: documents: {rules: {type: file, path: ./docs/rules.md}}
    documents: dict[str, DocumentConfig] = Field(default_factory=dict)

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

        User values override engine values. Databases, APIs, and documents (dicts)
        are deep-merged by key.

        Security: Certain fields in database/api configs are protected and cannot
        be overwritten by user config (e.g., 'uri' to prevent connection hijacking).

        Args:
            engine: Engine config dict
            user: User config dict

        Returns:
            Merged config dict
        """
        merged = dict(engine)

        # Dict-keyed sections to deep merge
        dict_sections = ["databases", "apis", "documents"]

        # Fields that user config cannot override (security protection)
        protected_fields = {"uri", "hosts", "endpoint", "endpoint_url"}

        for section in dict_sections:
            if section in user:
                engine_items = dict(merged.get(section, {}))

                for name, user_item in user[section].items():
                    if name in engine_items:
                        # Merge: user values override engine values, except protected fields
                        merged_item = dict(engine_items[name])
                        for key, value in user_item.items():
                            if key not in protected_fields:
                                merged_item[key] = value
                        engine_items[name] = merged_item
                    else:
                        # Add new item from user config
                        engine_items[name] = user_item

                merged[section] = engine_items

        # Merge other top-level keys (user overrides engine)
        for key in user:
            if key not in dict_sections:
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
