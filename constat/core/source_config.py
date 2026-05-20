# Copyright (c) 2025 Kenneth Stott
# Canary: f9c14643-c328-4f9e-9f90-4b8cf5778af2
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Data source configuration models: DatabaseConfig, DocumentConfig, APIConfig."""


from pydantic import BaseModel, Field


class DatabaseCredentials(BaseModel):
    """Database credentials for authentication."""
    username: str | None = None
    password: str | None = None

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

    Jaeger Example:
        databases:
          tracing:
            type: jaeger
            uri: http://localhost:16686
            description: Application tracing data
            sample_size: 50
            username: ${JAEGER_USER}
            password: ${JAEGER_PASS}
    """
    model_config = {"extra": "allow"}  # Allow type-specific fields

    # Database type: sql (default), mongodb, cassandra, elasticsearch, dynamodb, cosmosdb, firestore
    type: str = "sql"

    # Common fields
    description: str = ""
    read_only: bool = False  # If True, block write operations (INSERT, UPDATE, DELETE, etc.)

    # SQL databases (SQLAlchemy)
    uri: str | None = None

    # Credentials (used by SQL and some NoSQL)
    username: str | None = None
    password: str | None = None

    # MongoDB
    database: str | None = None  # Also used by CosmosDB
    sample_size: int = 100  # For schema inference

    # Cassandra
    keyspace: str | None = None
    hosts: list[str] | None = None
    port: int | None = None
    cloud_config: dict | None = None  # For DataStax Astra

    # DynamoDB
    region: str | None = None
    endpoint_url: str | None = None  # For local DynamoDB
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_session_token: str | None = None
    profile_name: str | None = None

    # Elasticsearch
    api_key: str | None = None
    # hosts: already defined above

    # CosmosDB
    endpoint: str | None = None
    key: str | None = None
    container: str | None = None
    # database: already defined above

    # Firestore
    project: str | None = None
    collection: str | None = None
    credentials_path: str | None = None

    # File-based data sources (csv, json, jsonl, parquet, arrow)
    # Supports local paths, s3://, https://, etc.
    path: str | None = None

    # JDBC (via JayDeBeApi)
    jdbc_driver: str | None = None
    jdbc_url: str | None = None
    jar_path: str | list[str] | None = None

    def is_jdbc(self) -> bool:
        return self.type == "jdbc"

    def get_connection_uri(self, config_dir: str | None = None) -> str:
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

    def get_resolved_path(self, config_dir: str | None = None) -> str | None:
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

    def _inject_credentials(self, username: str, password: str, uri: str | None = None) -> str:
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
        return self.type in ("mongodb", "cassandra", "elasticsearch", "dynamodb", "cosmosdb", "firestore", "neo4j", "jaeger")

    def is_file_source(self) -> bool:
        """Check if this is a file-based data source."""
        return self.type in ("csv", "json", "jsonl", "parquet", "arrow", "feather")

    # Glossary generation gating
    generate_definitions: bool | float | str = True  # True/False/float threshold/"auto"


class DocumentConfig(BaseModel):
    """Reference document configuration for inclusion in reasoning.

    Documents provide domain knowledge, business rules, API documentation,
    or any reference material the LLM should consider during analysis.

    File Example (type auto-detected from path/url/content):
        documents:
          business_rules:
            path: ./docs/business-rules.md
            description: "Revenue calculation rules and thresholds"

    HTTP Example (works for wiki pages, GitHub raw files, etc.):
        documents:
          wiki_page:
            url: https://wiki.example.com/api/v2/pages/12345/export/view
            headers:
              Authorization: Bearer ${WIKI_TOKEN}
            description: "Data dictionary from wiki"

    Confluence Example:
        documents:
          confluence:
            url: https://mycompany.atlassian.net
            space_key: ANALYTICS
            page_title: "Business Rules"
            username: ${CONFLUENCE_USER}
            api_token: ${CONFLUENCE_TOKEN}

    Inline Example:
        documents:
          glossary:
            content: |
              ## Key Terms
              - VIP: Customer with lifetime value > $100k
              - Churn: Customer inactive for 90+ days
            description: "Business glossary"

    S3 Example:
        documents:
          data_dict:
            url: s3://my-bucket/docs/data-dictionary.pdf
            aws_profile: analytics
            description: "Data dictionary from S3"
    """
    # Content type: auto | pdf | html | markdown | text | docx | xlsx | pptx
    # Also accepts full MIME: application/pdf, text/html, etc.
    type: str = "auto"

    # Source (transport inferred from field presence / URL scheme)
    path: str | None = None  # Local file path (supports glob: ./docs/*.md)
    url: str | None = None  # URL to fetch (http, https, s3, ftp, sftp)
    content: str | None = None  # Inline content
    headers: dict[str, str] = Field(default_factory=dict)  # HTTP headers

    # Credentials (all support ${ENV_VAR} via existing YAML substitution)
    username: str | None = None  # FTP/SFTP/Confluence
    password: str | None = None  # FTP/SFTP
    port: int | None = None  # FTP/SFTP custom port
    key_path: str | None = None  # SFTP SSH key path
    aws_profile: str | None = None  # S3 AWS profile
    aws_region: str | None = None  # S3 AWS region

    # Confluence
    space_key: str | None = None  # Confluence space key
    page_title: str | None = None  # Page title to fetch
    page_id: str | None = None  # Or page ID directly
    api_token: str | None = None  # Confluence API token

    # Notion
    page_url: str | None = None  # Notion page URL
    notion_token: str | None = None  # Notion integration token

    # IMAP
    mailbox: str = "INBOX"
    search_criteria: str = "ALL"
    since: str | None = None
    max_messages: int = 500
    include_headers: bool = True
    attachment_types: list[str] | None = None
    extract_attachments: bool = True

    # OAuth2 (IMAP)
    auth_type: str = "basic"  # "basic" | "oauth2" | "oauth2_refresh"
    oauth2_client_id: str | None = None
    oauth2_client_secret: str | None = None
    oauth2_tenant_id: str | None = None  # Azure AD
    oauth2_scopes: list[str] = Field(default_factory=list)
    oauth2_token_cache: str | None = None
    oauth2_cert_path: str | None = None  # Path to PFX file (env: SP_CERT_PATH)
    oauth2_cert_password: str | None = None  # PFX password (env: SP_CERT_PASSWORD)

    # Metadata
    description: str = ""  # What this document contains
    tags: list[str] = Field(default_factory=list)  # For categorization/search

    # PDF/Office extraction options
    extract_tables: bool = True  # Extract tables from PDFs/docs as markdown
    extract_images: bool = False  # Extract and describe images (requires vision model)
    page_range: str | None = None  # e.g., "1-5" or "1,3,5-10" for PDFs

    # Loading behavior
    cache: bool = True  # Cache fetched content
    cache_ttl: int | None = None  # Cache TTL in seconds (None = forever)

    # Link following (build corpus from linked documents)
    follow_links: bool = False  # Follow links in document to build corpus
    max_depth: int = 2  # Max link depth to follow (1 = direct links only)
    max_documents: int = 20  # Max documents to fetch via link following
    link_pattern: str | None = None  # Regex to filter which links to follow
    same_domain_only: bool = True  # Only follow links to same domain
    exclude_patterns: list[str] | None = None  # Regex patterns for URLs to skip

    # Glossary generation gating
    generate_definitions: bool | float | str = "auto"  # True/False/float threshold/"auto" (0.5)

    # Audio transcription
    whisper_model: str = "large-v3"  # faster-whisper model size
    diarize: bool = False  # Enable WhisperX speaker diarization
    language: str | None = None  # Language code (None = auto-detect)
    hf_token: str | None = None  # HuggingFace token for diarization (${HF_TOKEN})

    # Background refresh
    auto_refresh: bool = True  # Whether this source is eligible for background refresh
    refresh_interval: int | None = None  # Override default refresh interval (seconds)

    # MCP resource filtering
    resource_filter: str | None = None  # Regex to filter MCP resource URIs
    max_resources: int = 100  # Max resources to list from MCP server

    # Calendar source
    provider: str | None = None  # "google" | "microsoft"
    calendar_id: str | None = None  # Calendar ID (default: primary)
    until: str | None = None  # End date for event window (ISO format)
    max_events: int = 500
    expand_recurring: bool = True
    include_declined: bool = False
    include_cancelled: bool = False
    extract_body: bool = True  # Include event body/description
    calendars: list[str] = Field(default_factory=list)  # Multi-calendar IDs

    # Cloud drive source
    folder_id: str | None = None  # Root folder ID
    folder_path: str | None = None  # Root folder path (resolved to ID)
    recursive: bool = True  # Recurse into subfolders
    max_files: int = 200
    include_types: list[str] | None = None  # File extension allowlist [".pdf", ".docx"]
    include_trashed: bool = False
    site_id: str | None = None  # SharePoint site ID (for MS Graph)
    drive_id: str | None = None  # Specific drive ID

    # SharePoint source
    site_url: str | None = None  # SharePoint site URL
    discover_libraries: bool = True  # Index document libraries
    discover_lists: bool = False  # Index lists as tables
    discover_calendars: bool = False  # Index calendar lists
    discover_pages: bool = False  # Index site pages
    library_names: list[str] | None = None  # Allowlist of library names
    list_names: list[str] | None = None  # Allowlist of list names
    calendar_names: list[str] | None = None  # Allowlist of calendar names
    max_rows: int = 5000  # Max rows per list
    list_as_table: bool = True  # Render lists as markdown tables
    page_types: list[str] | None = None  # ["modern", "wiki"]


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
    graphql_flavor: str | None = None

    # Base URL for the API
    url: str | None = None

    # OpenAPI spec location (for type=openapi)
    spec_url: str | None = None  # URL to download OpenAPI spec
    spec_path: str | None = None  # Local path to OpenAPI spec file
    spec_inline: dict | None = None  # Inline OpenAPI spec (embedded in config)

    description: str = ""  # What this API provides
    headers: dict[str, str] = Field(default_factory=dict)  # Auth headers, etc.

    # Auth configuration (alternative to headers)
    auth_type: str | None = None  # bearer | basic | api_key
    auth_token: str | None = None  # Token for bearer auth
    auth_username: str | None = None  # Username for basic auth
    auth_password: str | None = None  # Password for basic auth
    api_key: str | None = None  # API key value
    api_key_header: str = "X-API-Key"  # Header name for API key

    # Schema introspection (GraphQL introspection or OpenAPI spec parsing)
    introspect: bool = True  # Fetch and cache schema at startup
    _schema_cache: dict | None = None  # Cached schema (internal, not serialized)

    # MCP tool filtering
    allowed_tools: list[str] | None = None  # Allowlist of MCP tool names
    denied_tools: list[str] | None = None  # Denylist of MCP tool names

    # Glossary generation gating
    generate_definitions: bool | float | str = True  # True/False/float threshold/"auto"


class ChonkModelSpec(BaseModel):
    """Model specification for a single chonk feature.

    Falls back to config.llm when omitted. Mirrors ModelSpec in config.py but is
    self-contained in source_config to avoid circular imports.
    """
    model: str
    provider: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    max_tokens: int | None = None


class ChonkConfig(BaseModel):
    """Optional chonk feature flags and per-feature model overrides.

    search_mode:
      "auto"         — pick graph_first if relationship_index available, else vector_first
      "vector_first" — standard vector→entity→MMR pipeline
      "graph_first"  — NER → RelationshipIndex → entity chunks → vector augment (requires DB schema)
      "global"       — community_summary chunks only (requires community_summaries=True)

    graph_first and global modes degrade gracefully to vector_first when prerequisites
    are absent (no relationship_index, no community chunks).

    Per-feature model overrides:
      ner_model          — spaCy model name for NerPipeline (e.g. "en_core_web_lg")
      community_llm      — LLM for CommunitySummarizer (falls back to config.llm)
      svo_llm            — LLM for SVOExtractor (falls back to config.llm)
      answer_llm         — LLM for AnswerGenerator (falls back to config.llm)
      embed_model        — sentence-transformers model for chunk embedding
    """
    search_mode: str = "auto"
    entity_expansion: bool = True
    community_summaries: bool = False
    lane_entity_min_sim: float | None = None

    # Path to a chonk.toml config file (relative to config.yaml or absolute)
    toml_path: str | None = None

    # Per-feature model specs
    ner_model: str = "en_core_web_sm"
    community_llm: ChonkModelSpec | None = None
    svo_llm: ChonkModelSpec | None = None
    answer_llm: ChonkModelSpec | None = None
    embed_model: str | None = None


# ---------------------------------------------------------------------------
# chonk.toml models — parsed from a separate TOML asset (e.g. demo/chonk.toml)
# ---------------------------------------------------------------------------

class ChonkVocabEntry(BaseModel):
    """One entry under [[vocab.entities]] in chonk.toml."""
    type: str                          # "static" | "db_query"
    entity_type: str
    domain: str | None = None       # constat domain_id; None = all domains
    # static:
    names: list[str] = []
    # db_query:
    connection: str | None = None
    sql: str | None = None


class ChonkIndexFeatures(BaseModel):
    """Feature flags under [index.features] in chonk.toml."""
    ner: bool = True
    schema_vocab: bool = True
    community: bool = False
    svo: bool = False
    context_graph: bool = False


class ChonkRetrieval(BaseModel):
    """Retrieval defaults under [retrieval] in chonk.toml."""
    search_mode: str = "auto"
    lane_entity_min_sim: float | None = None
    entity_ref_expansion: bool = False
    cluster: bool = False


class ChonkLLMOverride(BaseModel):
    """One LLM override under [llm.*] in chonk.toml."""
    provider: str | None = None
    model: str
    api_key: str | None = None
    base_url: str | None = None
    max_tokens: int | None = None


class ChonkLLMOverrides(BaseModel):
    """All LLM overrides under [llm] in chonk.toml."""
    svo: ChonkLLMOverride | None = None
    community: ChonkLLMOverride | None = None
    answer: ChonkLLMOverride | None = None


class ChonkIndex(BaseModel):
    """Settings under [index] in chonk.toml."""
    spacy_model: str = "en_core_web_sm"
    features: ChonkIndexFeatures = Field(default_factory=ChonkIndexFeatures)


class ChonkNamespaceConfig(BaseModel):
    """One namespace entry under [namespaces.<id>] in chonk.toml."""
    description: str = ""


class ChonkSourceConfig(BaseModel):
    """One entry under [[sources]] in chonk.toml."""
    name: str
    type: str  # "db_schema" | "glob" | "json_array" | "sql"
    namespace: str | None = None
    domain: str | None = None
    # db_schema
    uri: str | None = None
    # glob / document sources
    path: str | None = None
    pattern: str | None = None


class ChonkTomlConfig(BaseModel):
    """Top-level model for demo/chonk.toml.

    Load via ``ChonkTomlConfig.from_toml(path)``.
    """
    index: ChonkIndex = Field(default_factory=ChonkIndex)
    retrieval: ChonkRetrieval = Field(default_factory=ChonkRetrieval)
    llm: ChonkLLMOverrides = Field(default_factory=ChonkLLMOverrides)
    vocab: dict = Field(default_factory=dict)  # raw; .vocab_entries() parses it
    namespaces: dict[str, ChonkNamespaceConfig] = Field(default_factory=dict)
    sources: list[ChonkSourceConfig] = Field(default_factory=list)

    @classmethod
    def from_toml(cls, path: "str | Path") -> "ChonkTomlConfig":
        import tomllib
        from pathlib import Path as _Path
        with open(_Path(path), "rb") as f:
            raw = tomllib.load(f)
        vocab_raw = raw.pop("vocab", {})
        obj = cls.model_validate(raw)
        obj.vocab = vocab_raw
        return obj

    def vocab_entries(self) -> "list[ChonkVocabEntry]":
        entries = self.vocab.get("entities", [])
        return [ChonkVocabEntry.model_validate(e) for e in entries]

