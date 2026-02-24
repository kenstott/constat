# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Configuration loading with Pydantic validation and env var substitution."""

import glob
import os
import re
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator


def _resolve_refs(data: Any, base_dir: Path) -> Any:
    """Resolve $ref references in parsed YAML data (JSON Schema style).

    Usage in YAML:
        # Include a single file
        permissions:
          $ref: ./permissions.yaml

        # Include with glob pattern (returns a list)
        projects:
          $ref: ./projects/*.yaml

    Args:
        data: Parsed YAML data structure
        base_dir: Base directory for resolving relative paths

    Returns:
        Data with all $ref references resolved
    """
    if isinstance(data, dict):
        # Check if this is a $ref object
        if "$ref" in data and len(data) == 1:
            ref_path = data["$ref"]
            return _load_ref(ref_path, base_dir)
        else:
            # Recursively resolve refs in dict values
            return {k: _resolve_refs(v, base_dir) for k, v in data.items()}
    elif isinstance(data, list):
        # Recursively resolve refs in list items
        return [_resolve_refs(item, base_dir) for item in data]
    else:
        return data


def _load_ref(ref_path: str, base_dir: Path) -> Any:
    """Load content from a $ref path (file or glob pattern).

    For dict results, injects:
    - '_source_file': filename only (e.g., 'sales.yaml')
    - '_source_path': full absolute path (for editing)
    """
    full_pattern = base_dir / ref_path

    # Check if it's a glob pattern
    if "*" in ref_path or "?" in ref_path:
        matching_files = sorted(glob.glob(str(full_pattern)))
        if not matching_files:
            return []

        results = []
        for file_path in matching_files:
            with open(file_path) as f:
                content = f.read()
            content = _substitute_env_vars(content)
            data = yaml.safe_load(content)
            # Recursively resolve refs in included file
            data = _resolve_refs(data, Path(file_path).parent)
            # Inject source info for tracking (useful for projects)
            if isinstance(data, dict):
                data["_source_file"] = Path(file_path).name
                data["_source_path"] = str(Path(file_path).resolve())
            results.append(data)
        return results
    else:
        # Single file
        file_path = full_pattern
        if not file_path.exists():
            raise FileNotFoundError(f"Referenced file not found: {file_path}")

        with open(file_path) as f:
            content = f.read()
        content = _substitute_env_vars(content)
        data = yaml.safe_load(content)
        # Recursively resolve refs in included file
        data = _resolve_refs(data, file_path.parent)
        # Inject source info for tracking (useful for projects)
        if isinstance(data, dict):
            data["_source_file"] = file_path.name
            data["_source_path"] = str(file_path.resolve())
        return data


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
    read_only: bool = False  # If True, block write operations (INSERT, UPDATE, DELETE, etc.)

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

    # File-based data sources (csv, json, jsonl, parquet, arrow)
    # Supports local paths, s3://, https://, etc.
    path: Optional[str] = None

    def get_connection_uri(self, config_dir: Optional[str] = None) -> str:
        """
        Get the connection URI with credentials applied.

        Only valid for SQL and MongoDB databases.

        Args:
            config_dir: Directory containing config.yaml for resolving relative paths

        Returns:
            Connection URI with credentials applied
        """
        # Allow None type for SQL databases (inferred from URI)
        if self.type is not None and self.type not in ("sql", "mongodb"):
            raise ValueError(f"get_connection_uri() not supported for type: {self.type}")

        if not self.uri:
            raise ValueError("URI not configured")

        uri = self.uri

        # Resolve relative paths in SQLite URIs
        if uri.startswith("sqlite:///") and config_dir:
            # sqlite:///path means path after the 3 slashes
            db_path = uri[10:]  # Remove "sqlite:///"
            if db_path and not db_path.startswith("/"):
                # Relative path - resolve from config_dir
                from pathlib import Path
                resolved = (Path(config_dir) / db_path).resolve()
                uri = f"sqlite:///{resolved}"

        # Apply read-only mode for SQLite
        if self.read_only and uri.startswith("sqlite:///"):
            # SQLite read-only mode: file:path?mode=ro
            # Convert sqlite:///path to sqlite:///file:path?mode=ro
            db_path = uri[10:]  # Remove "sqlite:///"
            if "?" in db_path:
                uri = f"sqlite:///file:{db_path}&mode=ro"
            else:
                uri = f"sqlite:///file:{db_path}?mode=ro"

        # Use credentials if provided
        if self.username and self.password:
            return self._inject_credentials(self.username, self.password, uri)

        return uri

    def get_resolved_path(self, config_dir: Optional[str] = None) -> Optional[str]:
        """
        Get the file path resolved relative to config directory.

        For file-based data sources (csv, json, parquet, etc.).

        Args:
            config_dir: Directory containing config.yaml for resolving relative paths

        Returns:
            Resolved file path, or None if no path configured
        """
        if not self.path:
            return None

        # Remote paths (s3://, https://, etc.) - return as-is
        if "://" in self.path:
            return self.path

        # Absolute paths - return as-is
        from pathlib import Path
        path = Path(self.path)
        if path.is_absolute():
            return str(path)

        # Relative path - resolve from config_dir
        if config_dir:
            resolved = (Path(config_dir) / self.path).resolve()
            return str(resolved)

        return self.path

    def _inject_credentials(self, username: str, password: str, uri: Optional[str] = None) -> str:
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
        target_uri = uri if uri is not None else self.uri
        parsed = urllib.parse.urlparse(target_uri)

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
        # noinspection PyTypeChecker
        return urllib.parse.urlunparse(new_parsed)

    def is_nosql(self) -> bool:
        """Check if this is a NoSQL database."""
        return self.type in ("mongodb", "cassandra", "elasticsearch", "dynamodb", "cosmosdb", "firestore")

    def is_file_source(self) -> bool:
        """Check if this is a file-based data source."""
        return self.type in ("csv", "json", "jsonl", "parquet", "arrow", "feather")


class ModelSpec(BaseModel):
    """Specification for a model in a routing chain.

    Examples:
        # Full spec
        - provider: ollama
          model: sqlcoder:7b
          base_url: http://localhost:11434

        # Minimal (uses default provider)
        - model: claude-sonnet-4-20250514
    """
    provider: Optional[str] = None  # None means use default provider
    model: str

    # Provider-specific options
    base_url: Optional[str] = None
    timeout_seconds: Optional[int] = None
    max_tokens: Optional[int] = None


class TaskRoutingEntry(BaseModel):
    """Routing configuration for a single task type.

    Defines an ordered list of models to try for a task type.
    The router tries each model in order until success (escalation pattern).

    Examples:
        sql_generation:
          models:
            - provider: ollama
              model: sqlcoder:7b
            - provider: anthropic
              model: claude-3-5-haiku-20241022
    """
    # Ordered list of models to try (first = preferred, subsequent = fallback)
    models: list[ModelSpec]

    # Optional: models to use for high-complexity tasks
    high_complexity_models: Optional[list[ModelSpec]] = None


class TaskRoutingConfig(BaseModel):
    """Task-type to model chain mapping.

    Maps task types to ordered lists of models. The router tries each model
    in order until success (local-first with cloud fallback pattern).

    Example YAML:
        task_routing:
          sql_generation:
            models:
              - provider: ollama
                model: sqlcoder:7b
              - provider: anthropic
                model: claude-3-5-haiku-20241022
          planning:
            models:
              - provider: anthropic
                model: claude-sonnet-4-20250514
          python_analysis:
            models:
              - provider: ollama
                model: codellama:13b
              - provider: anthropic
                model: claude-sonnet-4-20250514
            high_complexity_models:
              - provider: anthropic
                model: claude-sonnet-4-20250514
    """
    # Dict of task_type name -> routing entry
    routes: dict[str, TaskRoutingEntry] = Field(default_factory=dict)

    def get_models_for_task(
        self,
        task_type: str,
        complexity: str = "medium",
    ) -> list[ModelSpec]:
        """
        Get ordered model list for a task type.

        Args:
            task_type: The task type name (e.g., "sql_generation")
            complexity: Complexity level ("low", "medium", "high")

        Returns:
            Ordered list of ModelSpecs to try
        """
        entry = self.routes.get(task_type)
        if not entry:
            return []

        if complexity == "high" and entry.high_complexity_models:
            return entry.high_complexity_models

        return entry.models


# Default task routing configuration
DEFAULT_TASK_ROUTING = {
    "planning": TaskRoutingEntry(
        models=[ModelSpec(model="claude-sonnet-4-20250514")]
    ),
    "replanning": TaskRoutingEntry(
        models=[ModelSpec(model="claude-sonnet-4-20250514")]
    ),
    "sql_generation": TaskRoutingEntry(
        models=[ModelSpec(model="claude-sonnet-4-20250514")]
    ),
    "python_analysis": TaskRoutingEntry(
        models=[ModelSpec(model="claude-sonnet-4-20250514")]
    ),
    "intent_classification": TaskRoutingEntry(
        models=[ModelSpec(model="claude-3-5-haiku-20241022")]
    ),
    "summarization": TaskRoutingEntry(
        models=[ModelSpec(model="claude-3-5-haiku-20241022")]
    ),
    "fact_resolution": TaskRoutingEntry(
        models=[ModelSpec(model="claude-sonnet-4-20250514")]
    ),
    "relationship_extraction": TaskRoutingEntry(
        models=[ModelSpec(model="claude-3-5-haiku-20241022")]
    ),
    "general": TaskRoutingEntry(
        models=[ModelSpec(model="claude-sonnet-4-20250514")]
    ),
}


class LLMConfig(BaseModel):
    """LLM provider configuration."""
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    api_key: Optional[str] = None

    # Task-type routing configuration
    task_routing: Optional[TaskRoutingConfig] = None

    # Provider-specific options
    base_url: Optional[str] = None  # For Ollama or custom endpoints

    def get_task_routing(self) -> TaskRoutingConfig:
        """
        Get the task routing config, with defaults applied.

        Returns:
            TaskRoutingConfig with all task types having at least default routing
        """
        if self.task_routing:
            # Merge user config with defaults
            merged_routes = dict(DEFAULT_TASK_ROUTING)
            merged_routes.update(self.task_routing.routes)
            return TaskRoutingConfig(routes=merged_routes)

        # Use defaults, but set provider/model from main config
        default_model = ModelSpec(
            provider=self.provider if self.provider != "anthropic" else None,
            model=self.model,
            base_url=self.base_url,
        )
        return TaskRoutingConfig(
            routes={
                task_type: TaskRoutingEntry(models=[default_model])
                for task_type in DEFAULT_TASK_ROUTING
            }
        )


class VectorStoreConfig(BaseModel):
    """Configuration for the vector store backend.

    The vector store handles document embedding storage and similarity search.

    Uses DuckDB VSS extension with array_cosine_similarity() for efficient
    vector search. Data persists across restarts.

    Example YAML:
        storage:
          vector_store:
            backend: duckdb
            db_path: ~/.constat/vectors.duckdb
    """
    # Backend type: "duckdb" or "numpy"
    backend: str = "duckdb"

    # Path to DuckDB database file (only for duckdb backend)
    # Default: ~/.constat/vectors.duckdb
    db_path: Optional[str] = None


class StorageConfig(BaseModel):
    """Storage configuration for artifact store and vector store."""
    # SQLAlchemy URI for the artifact store
    # Default: Uses DuckDB file per session
    # Options:
    #   - duckdb:///path/to/file.duckdb (embedded, single-user)
    #   - postgresql://user:pass@host:port/db (production, multi-user)
    #   - sqlite:///path/to/file.db (embedded alternative)
    artifact_store_uri: Optional[str] = None

    # Vector store configuration for document embeddings
    vector_store: Optional[VectorStoreConfig] = None


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

    # GraphQL implementation flavor (affects filter syntax hints)
    # Options: hasura | prisma | apollo | relay | custom
    # If not specified, generic hints are provided
    graphql_flavor: Optional[str] = None

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

    # Schema introspection (GraphQL introspection or OpenAPI spec parsing)
    introspect: bool = True  # Fetch and cache schema at startup
    _schema_cache: Optional[dict] = None  # Cached schema (internal, not serialized)


class ExecutionConfig(BaseModel):
    """Execution settings for generated code."""
    timeout_seconds: int = 60
    max_retries: int = 10
    allowed_imports: list[str] = Field(default_factory=list)
    print_file_refs: bool = True  # Print file:// URIs for saved files (set False for React UI)
    open_with_system_viewer: bool = False  # Auto-open saved files in the OS default app


class EmailConfig(BaseModel):
    """Email server configuration for sending results.

    Example:
        email:
          smtp_host: smtp.gmail.com
          smtp_port: 587
          smtp_user: ${EMAIL_USER}
          smtp_password: ${EMAIL_PASSWORD}
          from_address: noreply@company.com
          tls: true

    For Gmail, you need to use an App Password:
    https://support.google.com/accounts/answer/185833
    """
    smtp_host: str
    smtp_port: int = 587
    smtp_user: Optional[str] = None
    smtp_password: Optional[str] = None
    from_address: str
    from_name: str = "Constat"
    tls: bool = True
    timeout_seconds: int = 30


class SkillPathsConfig(BaseModel):
    """Configuration for skill file search paths.

    Skills are SKILL.md files following the Anthropic format that provide
    reusable instructions and domain knowledge.

    Default search paths (always included):
        - .constat/skills/ (project-level)
        - ~/.constat/skills/ (user-level)

    Example:
        skills:
          paths:
            - ./custom-skills
            - /shared/team-skills
    """
    # Additional paths to search for skills (beyond defaults)
    paths: list[str] = Field(default_factory=list)


class ContextPreloadConfig(BaseModel):
    """Configuration for preloading metadata into context.

    Seed patterns are used (once, at setup time) to identify which tables/columns
    should be cached and loaded into context at session start. This eliminates
    discovery tool calls for common query patterns.

    The cache is built once and persists until explicitly refreshed with /refresh.

    Example:
        context_preload:
          seed_patterns:
            - "sales"
            - "customer"
            - "revenue"
            - "inventory levels"
          similarity_threshold: 0.3

    This would preload metadata for tables/columns matching those patterns,
    so queries about sales, customers, etc. have schema info immediately available.
    """
    # Text patterns representing typical queries/domains
    # Used to match against table names, column names, and descriptions
    seed_patterns: list[str] = Field(default_factory=list)

    # Minimum similarity score (0-1) for a table to be included
    # Lower = more tables, higher = stricter matching
    similarity_threshold: float = 0.3

    # Maximum number of tables to preload (to avoid context overflow)
    max_tables: int = 50

    # Include column details for matched tables
    include_columns: bool = True

    # Maximum columns per table to include
    max_columns_per_table: int = 30


class DomainConfig(BaseModel):
    """Domain configuration - a reusable collection of data sources.

    Domains are defined in YAML files and can be shared across sessions.
    A session can select domains to load their databases, APIs, and documents.

    Example domain file (domains/sales-analytics.yaml):
        name: Sales Analytics
        description: Sales data from warehouse and CRM

        databases:
          snowflake_sales:
            uri: snowflake://...
            description: Sales data warehouse

        apis:
          salesforce:
            type: rest
            url: https://api.salesforce.com/...

        documents:
          sales_glossary:
            path: ./docs/sales-terms.md
    """
    model_config = {"extra": "ignore"}

    name: str
    description: str = ""
    owner: str = ""
    definition: str = ""
    filename: str = ""  # Source filename, autopopulated from _source_file
    source_path: str = ""  # Full path to source file, for editing
    path: str = ""  # Dot-delimited hierarchy path (e.g. "sales.north-america.retail")

    # Data sources (same structure as main config)
    databases: dict[str, DatabaseConfig] = Field(default_factory=dict)
    apis: dict[str, APIConfig] = Field(default_factory=dict)
    documents: dict[str, DocumentConfig] = Field(default_factory=dict)

    # Domain-specific config sections
    glossary: dict[str, Any] = Field(default_factory=dict)
    relationships: dict[str, Any] = Field(default_factory=dict)
    rights: dict[str, Any] = Field(default_factory=dict)
    facts: dict[str, Any] = Field(default_factory=dict)
    learnings: dict[str, Any] = Field(default_factory=dict)

    # NER stop list — terms to filter out during entity extraction
    ner_stop_list: list[str] = Field(default_factory=list)

    # Optional domain-scoped permissions (loaded from permissions.yaml in domain directory)
    permissions: Optional[Any] = Field(default=None, description="Domain-scoped PermissionsConfig (restricts global permissions)")

    # Optional domain-specific settings
    databases_description: str = ""
    system_prompt: str = ""

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DomainConfig":
        """Load domain config from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Domain file not found: {path}")

        with open(path) as f:
            raw_content = f.read()

        # Substitute environment variables
        substituted = _substitute_env_vars(raw_content)
        data = yaml.safe_load(substituted)
        data["filename"] = path.name
        data["source_path"] = str(path.resolve())
        # Default path to filename stem if not set
        if not data.get("path"):
            data["path"] = path.stem

        return cls.model_validate(data)

    @classmethod
    def from_dict(cls, data: dict) -> "DomainConfig":
        """Create DomainConfig from dict, handling _source_file/_source_path."""
        # Map _source_file to filename if present
        if "_source_file" in data and not data.get("filename"):
            data["filename"] = data["_source_file"]
        # Map _source_path to source_path if present
        if "_source_path" in data and not data.get("source_path"):
            data["source_path"] = data["_source_path"]
        return cls.model_validate(data)

    @classmethod
    def from_directory(cls, path: str | Path, parent_path: str = "") -> "DomainConfig":
        """Load domain config from a directory structure.

        Expected structure:
            <path>/config.yaml          — domain config
            <path>/permissions.yaml     — optional domain-scoped permissions
            <path>/skills/              — domain-specific skills
            <path>/domains/<sub>/       — nested sub-domains

        Sub-domains are merged alphabetically, then parent config overlays.

        Args:
            path: Path to domain directory
            parent_path: Dot-delimited parent hierarchy path

        Returns:
            DomainConfig with sub-domains merged in
        """
        path = Path(path)
        config_file = path / "config.yaml"
        if not config_file.exists():
            raise FileNotFoundError(f"Domain config not found: {config_file}")

        # Load parent config
        parent = cls.from_yaml(config_file)

        # Compute hierarchy path
        stem = path.stem
        parent.path = f"{parent_path}.{stem}" if parent_path else stem

        # Load domain-scoped permissions if present
        permissions_file = path / "permissions.yaml"
        if permissions_file.exists():
            from constat.server.config import PermissionsConfig
            with open(permissions_file) as f:
                raw_content = f.read()
            substituted = _substitute_env_vars(raw_content)
            perms_data = yaml.safe_load(substituted)
            if perms_data:
                parent.permissions = PermissionsConfig.model_validate(perms_data)

        # Load and merge sub-domains (alphabetically)
        sub_domains_dir = path / "domains"
        if sub_domains_dir.is_dir():
            merged_data: dict = {}
            for sub_dir in sorted(sub_domains_dir.iterdir()):
                if sub_dir.is_dir() and (sub_dir / "config.yaml").exists():
                    sub_domain = cls.from_directory(sub_dir, parent_path=parent.path)
                    sub_data = sub_domain.model_dump(exclude_defaults=True)
                    # Merge sub-domain data additively
                    for key, value in sub_data.items():
                        if isinstance(value, dict) and isinstance(merged_data.get(key), dict):
                            merged_data[key] = {**merged_data[key], **value}
                        elif value:
                            merged_data[key] = value

            # Parent overlays sub-domain merge
            parent_data = parent.model_dump(exclude_defaults=True)
            for key, value in parent_data.items():
                if isinstance(value, dict) and isinstance(merged_data.get(key), dict):
                    merged_data[key] = {**merged_data[key], **value}
                else:
                    merged_data[key] = value

            # Ensure required fields
            merged_data.setdefault("name", parent.name)
            result = cls.model_validate(merged_data)
            # Preserve permissions from parent (model_validate won't carry it through merged_data correctly)
            if parent.permissions is not None:
                result.permissions = parent.permissions
            return result

        return parent


# Backwards compatibility alias
ProjectConfig = DomainConfig


class Config(BaseModel):
    """Root configuration model."""
    model_config = {"extra": "ignore"}

    # Directory containing config.yaml (for resolving relative paths)
    config_dir: str = ""

    llm: LLMConfig = Field(default_factory=LLMConfig)

    # Domains keyed by filename
    # YAML format: domains: { sales.yaml: { $ref: ./domains/sales.yaml } }
    domains: dict[str, DomainConfig] = Field(default_factory=dict)

    @field_validator("domains", mode="before")
    @classmethod
    def _convert_domains(cls, v):
        """Convert raw domain dicts, mapping _source_file/_source_path."""
        if not v:
            return {}
        result = {}
        for key, item in v.items():
            if isinstance(item, dict):
                # Map _source_file to filename
                if "_source_file" in item and not item.get("filename"):
                    item["filename"] = item["_source_file"]
                # Map _source_path to source_path
                if "_source_path" in item and not item.get("source_path"):
                    item["source_path"] = item["_source_path"]
            result[key] = item
        return result

    # Databases keyed by name for easy merging
    # YAML format: databases: {main: {uri: ...}, analytics: {uri: ...}}
    databases: dict[str, DatabaseConfig] = Field(default_factory=dict)

    # External APIs keyed by name
    # YAML format: apis: {countries: {url: ..., type: graphql}}
    apis: dict[str, APIConfig] = Field(default_factory=dict)

    # Reference documents keyed by name
    # YAML format: documents: {rules: {type: file, path: ./docs/rules.md}}
    documents: dict[str, DocumentConfig] = Field(default_factory=dict)

    # Skill search paths configuration
    skills: SkillPathsConfig = Field(default_factory=SkillPathsConfig)

    # Context preload configuration (seed patterns for metadata caching)
    context_preload: ContextPreloadConfig = Field(default_factory=ContextPreloadConfig)

    databases_description: str = ""  # Global context for all databases
    system_prompt: str = ""
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    email: Optional[EmailConfig] = None  # Email configuration for send_email

    # Core facts that are always available in every session
    # YAML format: facts: {company_name: "Acme Corp", fiscal_year_start: "April 1"}
    facts: dict[str, Any] = Field(default_factory=dict)

    # New first-class config sections
    rights: dict[str, Any] = Field(default_factory=dict)
    glossary: dict[str, Any] = Field(default_factory=dict)
    relationships: dict[str, Any] = Field(default_factory=dict)

    # NER stop list — system-level terms to filter out during entity extraction
    ner_stop_list: list[str] = Field(default_factory=list)

    @property
    def projects(self) -> dict[str, DomainConfig]:
        """Backwards compatibility: access domains as projects."""
        return self.domains

    def list_domains(self) -> list[dict]:
        """List available domains.

        Returns:
            List of domain info dicts with 'filename', 'name', 'description'
        """
        return [
            {
                "filename": filename,
                "name": domain.name,
                "description": domain.description,
            }
            for filename, domain in self.domains.items()
        ]

    def load_domain(self, filename: str) -> Optional["DomainConfig"]:
        """Load a domain by filename.

        Args:
            filename: Domain YAML filename (e.g., 'sales-analytics.yaml')

        Returns:
            DomainConfig or None if not found
        """
        return self.domains.get(filename)

    # Backwards compatibility aliases
    list_projects = list_domains
    load_project = load_domain

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

        # Load .env file - search from config directory up to root
        config_dir = path.parent.resolve()
        env_loaded = False

        # Search up from config directory to find .env
        search_dir = config_dir
        while search_dir != search_dir.parent:  # Stop at filesystem root
            env_file = search_dir / ".env"
            if env_file.exists():
                load_dotenv(env_file)
                env_loaded = True
                break
            search_dir = search_dir.parent

        # Also check filesystem root
        if not env_loaded:
            root_env = search_dir / ".env"
            if root_env.exists():
                load_dotenv(root_env)
                env_loaded = True

        # Final fallback - try current working directory
        if not env_loaded:
            load_dotenv()

        with open(path) as f:
            raw_content = f.read()

        # Substitute environment variables: ${VAR_NAME}
        substituted = _substitute_env_vars(raw_content)
        # Parse YAML and resolve $ref references
        data = yaml.safe_load(substituted)
        data = _resolve_refs(data, config_dir)

        # Backwards compat: map 'projects' key to 'domains'
        if "projects" in data and "domains" not in data:
            data["domains"] = data.pop("projects")
        elif "projects" in data:
            data.pop("projects")

        # Load user config from file if provided
        user_data = None
        if user_config_path:
            user_path = Path(user_config_path)
            if user_path.exists():
                with open(user_path) as f:
                    user_raw = f.read()
                user_substituted = _substitute_env_vars(user_raw)
                user_data = yaml.safe_load(user_substituted)
                user_data = _resolve_refs(user_data, user_path.parent)
        elif user_config:
            user_data = user_config

        # Merge user config into engine config
        if user_data:
            data = cls._merge_configs(data, user_data)

        # Store config directory for resolving relative paths
        data["config_dir"] = str(config_dir)

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
        dict_sections = ["databases", "apis", "documents", "glossary", "relationships", "rights"]

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
