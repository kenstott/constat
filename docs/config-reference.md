# Configuration Reference

Constat uses YAML configuration files with environment variable substitution (`${VAR_NAME}`) and `$ref` file inclusion.

## File Layout

```
project/
  config.yaml                  # Top-level config
  .env                         # Environment variables (auto-loaded)
  permissions.yaml             # Optional, included via $ref
  domains/
    sales-analytics/
      config.yaml              # Domain config
      permissions.yaml         # Optional domain-scoped permissions
      skills/                  # Domain-specific skills
      docs/                    # Domain documents (referenced by path)
    hr-reporting/
      config.yaml
      ...
```

**Top-level `config.yaml`** defines global settings: LLM provider, databases, APIs, documents, execution, storage, email, server, and the list of active domains.

**Domain `config.yaml`** files define domain-specific data sources, glossary, relationships, entity resolution, and task routing. They are merged into the session via tiered config.

## Environment Variables

Use `${VAR_NAME}` anywhere in YAML. Values are resolved from `.env` (searched upward from config directory) or the shell environment.

```yaml
api_key: ${ANTHROPIC_API_KEY}
```

Unset variables raise an error at load time.

## File Inclusion (`$ref`)

Include external YAML files or globs:

```yaml
# Single file
permissions:
  $ref: ./permissions.yaml

# Glob pattern (returns a list)
projects:
  $ref: ./projects/*.yaml
```

---

## Top-Level Sections

### `domains`

Defines which domain directories to activate. The filesystem under `domains/` is the authority; this list defines the activation order (DAG).

```yaml
domains:
  - sales-analytics
  - hr-reporting
```

Each entry is a directory name under `domains/`. The directory must contain a `config.yaml`.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| (list items) | `string` | yes | — | Domain directory names to activate |

---

### `llm`

LLM provider and task routing configuration.

```yaml
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: ${ANTHROPIC_API_KEY}
  base_url: null                    # For Ollama or custom endpoints

  task_routing:
    planning:
      models:
        - provider: anthropic
          model: claude-opus-4-20250514
          timeout_seconds: 180
      low_complexity_models:
        - provider: anthropic
          model: claude-sonnet-4-20250514
          timeout_seconds: 60
    sql_generation:
      models:
        - provider: together
          model: meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo
          timeout_seconds: 15
        - provider: anthropic
          model: claude-sonnet-4-20250514
          timeout_seconds: 120
```

**`llm` fields:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `provider` | `string` | no | `"anthropic"` | LLM provider: `anthropic`, `ollama`, `together`, etc. |
| `model` | `string` | no | `"claude-sonnet-4-20250514"` | Default model name |
| `api_key` | `string` | no | `null` | API key for the provider |
| `base_url` | `string` | no | `null` | Custom endpoint URL (for Ollama, etc.) |
| `task_routing` | `map` | no | built-in defaults | Task-type to model chain mapping |

**Task routing** maps task types to ordered model lists. The router tries each model in sequence (escalation pattern). Supported task types: `planning`, `replanning`, `sql_generation`, `python_analysis`, `intent_classification`, `clarification`, `summarization`, `fact_resolution`, `derivation_logic`, `relationship_extraction`, `structured_extraction`, `synthesis`, `glossary_generation`, `general`.

**Each task routing entry:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `models` | `list[ModelSpec]` | yes | — | Ordered fallback chain of models |
| `low_complexity_models` | `list[ModelSpec]` | no | `null` | Models for simple tasks |
| `high_complexity_models` | `list[ModelSpec]` | no | `null` | Models for complex tasks |

**ModelSpec fields:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `provider` | `string` | no | uses `llm.provider` | Provider override |
| `model` | `string` | yes | — | Model identifier |
| `api_key` | `string` | no | `null` | Per-model API key |
| `base_url` | `string` | no | `null` | Per-model endpoint |
| `timeout_seconds` | `int` | no | `null` | Request timeout |
| `max_tokens` | `int` | no | `null` | Max output tokens |

---

### `databases`

SQL databases, NoSQL databases, and file-based data sources keyed by name.

#### SQL databases

```yaml
databases:
  sales:
    uri: sqlite:///data/sales.db
    description: Sales CRM database
    read_only: false
    generate_definitions: true
```

```yaml
databases:
  warehouse:
    uri: postgresql://localhost/analytics
    username: ${PG_USER}
    password: ${PG_PASS}
    description: Data warehouse
```

#### File-based sources

Supported types: `csv`, `json`, `jsonl`, `parquet`, `arrow`, `feather`. Paths can be local, `s3://`, or `https://`.

```yaml
databases:
  web_metrics:
    type: csv
    path: data/website_metrics.csv
    description: Daily web analytics

  events:
    type: json
    path: data/events.json
    description: Clickstream events
```

#### NoSQL databases

Supported types: `mongodb`, `cassandra`, `elasticsearch`, `dynamodb`, `cosmosdb`, `firestore`, `neo4j`, `jaeger`.

```yaml
# MongoDB
databases:
  mongo:
    type: mongodb
    uri: mongodb://localhost:27017
    database: mydb
    sample_size: 100
    description: Document store

# Cassandra
  cassandra:
    type: cassandra
    keyspace: my_keyspace
    hosts: [node1, node2]
    port: 9042
    username: ${CASSANDRA_USER}
    password: ${CASSANDRA_PASS}

# DynamoDB
  dynamo:
    type: dynamodb
    region: us-east-1
    profile_name: myprofile
    endpoint_url: http://localhost:8000    # For local DynamoDB

# Elasticsearch
  elastic:
    type: elasticsearch
    hosts: [http://localhost:9200]
    api_key: ${ES_API_KEY}

# CosmosDB
  cosmos:
    type: cosmosdb
    endpoint: https://myaccount.documents.azure.com
    key: ${COSMOS_KEY}
    database: mydb
    container: mycontainer

# Firestore
  firestore:
    type: firestore
    project: my-gcp-project
    collection: users
    credentials_path: /path/to/credentials.json

# Jaeger
  tracing:
    type: jaeger
    uri: http://localhost:16686
    sample_size: 50
    username: ${JAEGER_USER}
    password: ${JAEGER_PASS}
```

**Common fields (all database types):**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `type` | `string` | no | `"sql"` | Database type (see above) |
| `description` | `string` | no | `""` | Human description of contents |
| `read_only` | `bool` | no | `false` | Block write operations |
| `generate_definitions` | `bool\|float\|string` | no | `true` | Glossary generation gating: `true`, `false`, float threshold, or `"auto"` |

**SQL fields:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `uri` | `string` | yes (for SQL) | `null` | SQLAlchemy connection URI |
| `username` | `string` | no | `null` | Credentials injected into URI |
| `password` | `string` | no | `null` | Credentials injected into URI |

**File fields:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `type` | `string` | yes | — | `csv`, `json`, `jsonl`, `parquet`, `arrow`, `feather` |
| `path` | `string` | yes (for files) | `null` | File path (local, `s3://`, `https://`) |

**MongoDB fields:** `uri`, `database`, `sample_size`, `username`, `password`

**Cassandra fields:** `keyspace`, `hosts`, `port`, `username`, `password`, `cloud_config`

**DynamoDB fields:** `region`, `endpoint_url`, `profile_name`, `aws_access_key_id`, `aws_secret_access_key`, `aws_session_token`

**Elasticsearch fields:** `hosts`, `api_key`

**CosmosDB fields:** `endpoint`, `key`, `database`, `container`

**Firestore fields:** `project`, `collection`, `credentials_path`

---

### `databases_description`

Global context string describing how databases relate to each other. Included in LLM context.

```yaml
databases_description: |
  Three interconnected business databases:
  - sales: CRM and order management
  - inventory: Multi-warehouse stock management
  - hr: Employee and performance tracking
```

---

### `apis`

External APIs keyed by name. Supports GraphQL and OpenAPI (REST).

#### GraphQL

```yaml
apis:
  countries:
    type: graphql
    url: https://countries.trevorblades.com/graphql
    graphql_flavor: hasura          # hasura | prisma | apollo | relay | custom
    description: Country data API
    headers:
      Authorization: Bearer ${TOKEN}
```

#### OpenAPI / REST

```yaml
apis:
  petstore:
    type: openapi
    url: https://petstore.swagger.io
    spec_url: https://petstore.swagger.io/v2/swagger.json
    description: Pet store API

  # From local spec file
  internal:
    type: openapi
    spec_path: ./specs/internal-api.yaml

  # Inline spec
  simple:
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
```

**API fields:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `type` | `string` | no | `"graphql"` | `graphql` or `openapi` |
| `url` | `string` | no | `null` | Base URL |
| `graphql_flavor` | `string` | no | `null` | GraphQL implementation hint |
| `spec_url` | `string` | no | `null` | URL to OpenAPI spec |
| `spec_path` | `string` | no | `null` | Local path to OpenAPI spec |
| `spec_inline` | `dict` | no | `null` | Inline OpenAPI spec |
| `description` | `string` | no | `""` | API description |
| `headers` | `map[string, string]` | no | `{}` | HTTP headers (auth, etc.) |
| `auth_type` | `string` | no | `null` | `bearer`, `basic`, or `api_key` |
| `auth_token` | `string` | no | `null` | Token for bearer auth |
| `auth_username` | `string` | no | `null` | Username for basic auth |
| `auth_password` | `string` | no | `null` | Password for basic auth |
| `api_key` | `string` | no | `null` | API key value |
| `api_key_header` | `string` | no | `"X-API-Key"` | Header name for API key |
| `introspect` | `bool` | no | `true` | Fetch schema at startup |
| `generate_definitions` | `bool\|float\|string` | no | `true` | Glossary generation gating |

---

### `documents`

Reference documents for LLM context. Supports local files, URLs, S3, Confluence, Notion, inline content, and web crawling.

#### Local file

```yaml
documents:
  business_rules:
    path: docs/business_rules.md
    description: Business policies for customer tiers
```

#### URL (HTTP/HTTPS)

```yaml
documents:
  wiki_page:
    url: https://wiki.example.com/api/v2/pages/12345/export/view
    headers:
      Authorization: Bearer ${WIKI_TOKEN}
    description: Data dictionary
```

#### Web crawling (follow links)

```yaml
documents:
  hr_docs:
    url: https://en.wikipedia.org/wiki/Human_resource_management
    follow_links: true
    max_depth: 1
    max_documents: 20
    same_domain_only: true
    exclude_patterns:
      - '/wiki/(Special|Wikipedia|Talk):'
      - \.(png|jpg|css|js)(\?|$)
    description: HR concepts
```

#### Confluence

```yaml
documents:
  confluence_page:
    url: https://mycompany.atlassian.net
    space_key: ANALYTICS
    page_title: "Business Rules"
    username: ${CONFLUENCE_USER}
    api_token: ${CONFLUENCE_TOKEN}
```

#### Notion

```yaml
documents:
  notion_page:
    page_url: https://notion.so/page-id
    notion_token: ${NOTION_TOKEN}
```

#### S3

```yaml
documents:
  data_dict:
    url: s3://my-bucket/docs/data-dictionary.pdf
    aws_profile: analytics
    aws_region: us-east-1
```

#### Inline content

```yaml
documents:
  glossary:
    content: |
      ## Key Terms
      - VIP: Customer with lifetime value > $100k
      - Churn: Customer inactive for 90+ days
    description: Business glossary
```

**Document fields:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `type` | `string` | no | `"auto"` | Content type: `auto`, `pdf`, `html`, `markdown`, `text`, `docx`, `xlsx`, `pptx` |
| `path` | `string` | no | `null` | Local file path (supports glob: `./docs/*.md`) |
| `url` | `string` | no | `null` | URL (http, https, s3, ftp, sftp) |
| `content` | `string` | no | `null` | Inline content |
| `headers` | `map[string, string]` | no | `{}` | HTTP headers |
| `description` | `string` | no | `""` | What this document contains |
| `tags` | `list[string]` | no | `[]` | Categorization tags |
| `extract_tables` | `bool` | no | `true` | Extract tables from PDFs/docs |
| `extract_images` | `bool` | no | `false` | Extract and describe images |
| `page_range` | `string` | no | `null` | PDF page range (e.g., `"1-5"`) |
| `cache` | `bool` | no | `true` | Cache fetched content |
| `cache_ttl` | `int` | no | `null` | Cache TTL in seconds |
| `follow_links` | `bool` | no | `false` | Crawl linked documents |
| `max_depth` | `int` | no | `2` | Max crawl depth |
| `max_documents` | `int` | no | `20` | Max documents to crawl |
| `link_pattern` | `string` | no | `null` | Regex to filter links |
| `same_domain_only` | `bool` | no | `true` | Only follow same-domain links |
| `exclude_patterns` | `list[string]` | no | `null` | URL regex patterns to skip |
| `generate_definitions` | `bool\|float\|string` | no | `"auto"` | Glossary generation gating |

**Credential fields (transport-specific):**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `username` | `string` | `null` | FTP/SFTP/Confluence username |
| `password` | `string` | `null` | FTP/SFTP password |
| `port` | `int` | `null` | FTP/SFTP custom port |
| `key_path` | `string` | `null` | SFTP SSH key path |
| `aws_profile` | `string` | `null` | S3 AWS profile |
| `aws_region` | `string` | `null` | S3 AWS region |
| `space_key` | `string` | `null` | Confluence space key |
| `page_title` | `string` | `null` | Confluence page title |
| `page_id` | `string` | `null` | Confluence page ID |
| `api_token` | `string` | `null` | Confluence API token |
| `page_url` | `string` | `null` | Notion page URL |
| `notion_token` | `string` | `null` | Notion integration token |

---

### `entity_resolution`

Maps custom entity types to data sources for NER and vector-based resolution. Each entry populates the entity resolver with known values from a database, API, or static list.

#### SQL shorthand (table + column)

```yaml
entity_resolution:
  - entity_type: CUSTOMER
    source: sales
    table: customers
    name_column: name
```

#### SQL custom query

```yaml
entity_resolution:
  - entity_type: EMPLOYEE
    source: hr
    query: "SELECT DISTINCT first_name || ' ' || last_name AS name FROM employees"
```

#### GraphQL

```yaml
entity_resolution:
  - entity_type: COUNTRY
    source: countries
    query: "{ countries { name } }"
    items_path: countries
    name_field: name
```

#### REST API

```yaml
entity_resolution:
  - entity_type: PRODUCT
    source: catalog_api
    endpoint: /products
    items_path: data.products
    name_field: title
```

#### Static list

```yaml
entity_resolution:
  - entity_type: REGION
    values:
      - North America
      - EMEA
      - APAC
```

**Entity resolution fields:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `entity_type` | `string` | yes | — | Custom NER label (e.g., `"CUSTOMER"`) |
| `source` | `string` | no | `""` | Database or API name from config |
| `table` | `string` | no | `null` | Table to query (SQL shorthand) |
| `name_column` | `string` | no | `null` | Column with entity names (SQL shorthand) |
| `query` | `string` | no | `null` | Custom SQL/GraphQL query returning a `name` column |
| `endpoint` | `string` | no | `null` | REST GET endpoint path |
| `items_path` | `string` | no | `null` | JSON path to items array |
| `name_field` | `string` | no | `"name"` | Field name in each item |
| `values` | `list[string]` | no | `null` | Inline static values |
| `max_values` | `int` | no | `10000` | Cap on distinct values |

---

### `glossary`

Pre-defined glossary terms loaded into the vector store for semantic search. Each entry can be a simple string definition or a structured object.

```yaml
glossary:
  MRR:
    definition: Monthly Recurring Revenue
    aliases: [monthly recurring revenue, recurring revenue]
    category: metrics
  churn:
    definition: Customer inactive for 90+ days
    aliases: [churned, customer churn]
  # Simple string form
  ARR: Annual Recurring Revenue
```

**Per-term fields (structured form):**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `definition` | `string` | no | `""` | Term definition |
| `aliases` | `list[string]` | no | `[]` | Alternative names |
| `category` | `string` | no | `""` | Categorization label |

---

### `relationships`

Pre-defined relationships and SVO extraction configuration.

#### Preferred verbs (SVO extraction)

Controls the canonical verb vocabulary for relationship extraction. Verbs use `UPPER_SNAKE_CASE`.

```yaml
relationships:
  preferred_verbs:
    hierarchy: [MANAGES, REPORTS_TO]
    action: [CREATES, PROCESSES, APPROVES, PLACES]
    flow: [SENDS, RECEIVES, TRANSFERS]
    causation: [DRIVES, REQUIRES, ENABLES]
    temporal: [PRECEDES, FOLLOWS, TRIGGERS]
    association: [REFERENCES, WORKS_IN, PARTICIPATES_IN, USES]
```

#### Pre-defined relationships

```yaml
relationships:
  customer_orders:
    definition: A customer places orders
    aliases: [purchase history, order history]
    entities: [customer, order]
```

**Per-relationship fields (structured form):**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `definition` | `string` | no | `""` | Relationship description |
| `aliases` | `list[string]` | no | `[]` | Alternative names |
| `entities` | `list[string]` | no | `[]` | Connected entity names |

---

### `execution`

Controls generated code execution.

```yaml
execution:
  timeout_seconds: 120
  max_retries: 5
  allowed_imports:
    - pandas
    - numpy
    - plotly
    - json
    - datetime
  print_file_refs: true
  open_with_system_viewer: false
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `timeout_seconds` | `int` | no | `60` | Max execution time per step |
| `max_retries` | `int` | no | `10` | Max retry attempts on failure |
| `allowed_imports` | `list[string]` | no | `[]` | Python modules permitted in generated code |
| `print_file_refs` | `bool` | no | `true` | Print `file://` URIs for saved files |
| `open_with_system_viewer` | `bool` | no | `false` | Auto-open saved files in OS default app |

---

### `storage`

Vector store and artifact store configuration.

```yaml
storage:
  artifact_store_uri: null           # Default: DuckDB file per session
  vector_store:
    backend: duckdb
    db_path: null                    # Default: ~/.constat/vectors.duckdb
    reranker_model: null             # e.g., "cross-encoder/ms-marco-MiniLM-L-6-v2"
    store_chunk_text: true
    cluster_min_terms: 2
    cluster_divisor: 5
    cluster_max_k: null
```

**`storage` fields:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `artifact_store_uri` | `string` | no | `null` | SQLAlchemy URI for artifact store (`duckdb:///`, `postgresql://`, `sqlite:///`) |
| `vector_store` | `VectorStoreConfig` | no | `null` | Vector store settings |

**`vector_store` fields:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `backend` | `string` | no | `"duckdb"` | Backend type: `duckdb` or `numpy` |
| `db_path` | `string` | no | `null` | DuckDB database file path |
| `reranker_model` | `string` | no | `null` | Cross-encoder model for reranking |
| `store_chunk_text` | `bool` | no | `true` | Store original chunk text (disable for large datasets) |
| `cluster_min_terms` | `int` | no | `2` | Minimum terms to trigger clustering |
| `cluster_divisor` | `int` | no | `5` | k = max(2, n_terms // divisor) |
| `cluster_max_k` | `int` | no | `null` | Optional cap on k |

---

### `email`

SMTP configuration for `send_email()` in generated code.

```yaml
email:
  smtp_host: smtp.gmail.com
  smtp_port: 587
  smtp_user: ${SMTP_USER}
  smtp_password: ${SMTP_PASSWORD}
  from_address: noreply@company.com
  from_name: Constat
  tls: true
  timeout_seconds: 30
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `smtp_host` | `string` | yes | — | SMTP server hostname |
| `smtp_port` | `int` | no | `587` | SMTP port |
| `smtp_user` | `string` | no | `null` | SMTP username |
| `smtp_password` | `string` | no | `null` | SMTP password |
| `from_address` | `string` | yes | — | Sender email address |
| `from_name` | `string` | no | `"Constat"` | Sender display name |
| `tls` | `bool` | no | `true` | Use TLS |
| `timeout_seconds` | `int` | no | `30` | Connection timeout |

---

### `system_prompt`

Global system prompt prepended to LLM context in every session.

```yaml
system_prompt: |
  You are a business analyst assistant analyzing company data.
```

---

### `ner_stop_list`

Terms filtered out during named entity extraction. Prevents common technical terms from being treated as entities.

```yaml
ner_stop_list:
  - api
  - string
  - integer
  - graphql
  - JSON
```

---

### `facts`

Key-value pairs always available in session context. Used for domain constants the LLM should know.

```yaml
facts:
  company_name: "Acme Corp"
  fiscal_year_start: "April 1"
  default_currency: USD
```

---

### `rights`

Business rules and access rights definitions (free-form dict, merged across tiers).

```yaml
rights:
  data_classification:
    pii: [email, phone, ssn]
    confidential: [salary, performance_rating]
```

---

### `skills`

Additional search paths for SKILL.md files. Default paths (`.constat/skills/` and `~/.constat/skills/`) are always included.

```yaml
skills:
  paths:
    - ./custom-skills
    - /shared/team-skills
```

---

### `context_preload`

Preload table/column metadata into context at session start to eliminate discovery tool calls for common patterns.

```yaml
context_preload:
  seed_patterns:
    - "sales"
    - "customer"
    - "revenue"
  similarity_threshold: 0.3
  max_tables: 50
  include_columns: true
  max_columns_per_table: 30
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `seed_patterns` | `list[string]` | no | `[]` | Text patterns to match tables/columns |
| `similarity_threshold` | `float` | no | `0.3` | Minimum similarity score (0-1) |
| `max_tables` | `int` | no | `50` | Max tables to preload |
| `include_columns` | `bool` | no | `true` | Include column details |
| `max_columns_per_table` | `int` | no | `30` | Max columns per table |

---

### `server`

API server configuration. Most fields can be overridden by environment variables.

```yaml
server:
  host: 127.0.0.1
  port: 8000
  cors_origins:
    - http://localhost:5173
    - http://localhost:3000
  session_timeout_minutes: 60
  max_concurrent_sessions: 10
  require_plan_approval: true
  auth_disabled: ${AUTH_DISABLED}
  firebase_project_id: ${FIREBASE_PROJECT_ID}
  firebase_api_key: ${FIREBASE_API_KEY}
  admin_token: "my-dev-token"
  runtime_dir: "."

  local_users:
    demo:
      password_hash: "scrypt:..."
      email: demo@localhost

  permissions:
    $ref: ./permissions.yaml
```

| Field | Type | Required | Default | Env Override | Description |
|-------|------|----------|---------|--------------|-------------|
| `host` | `string` | no | `"127.0.0.1"` | — | Bind address |
| `port` | `int` | no | `8000` | — | Listen port |
| `cors_origins` | `list[string]` | no | `["http://localhost:5173", "http://localhost:3000"]` | — | Allowed CORS origins |
| `session_timeout_minutes` | `int` | no | `60` | — | Session timeout |
| `max_concurrent_sessions` | `int` | no | `10` | — | Max concurrent sessions |
| `require_plan_approval` | `bool` | no | `true` | — | Require user plan approval |
| `auth_disabled` | `bool` | no | `true` | `AUTH_DISABLED` | Disable authentication |
| `firebase_project_id` | `string` | no | `null` | `FIREBASE_PROJECT_ID` | Firebase project for JWT validation |
| `firebase_api_key` | `string` | no | `null` | `FIREBASE_API_KEY` | Firebase Web API key |
| `admin_token` | `string` | no | `null` | `CONSTAT_ADMIN_TOKEN` | Admin token for CLI access |
| `runtime_dir` | `string` | no | `"."` | — | Base directory for `.constat/` data |
| `local_users` | `map[string, LocalUser]` | no | `{}` | — | Local auth users (keyed by username) |
| `permissions` | `PermissionsConfig` | no | `{}` | — | User permissions |

**`local_users` entry:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `password_hash` | `string` | yes | Scrypt hash (generate with `constat hash-password`) |
| `email` | `string` | no | User email |

---

### `permissions`

User access control. Typically included via `$ref` from the server section.

```yaml
permissions:
  users:
    uid_abc123:
      persona: platform_admin
      domains: []
      databases: []
      documents: []
      apis: []
      skills: []
      agents: []
      rules: []
      facts: []

    uid_def456:
      persona: domain_user
      domains:
        - sales-analytics
      databases:
        - inventory
      documents: []
      apis:
        - countries

  default:
    domains: []
    databases: []
    documents: []
    apis: []
```

**Personas** (from most to least privileged): `platform_admin`, `domain_builder`, `sme`, `domain_user`, `viewer`.

`platform_admin` has full access regardless of resource lists.

**Per-user fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `persona` | `string` | `"viewer"` | Platform persona |
| `domains` | `list[string]` | `[]` | Accessible domain names |
| `databases` | `list[string]` | `[]` | Accessible database names |
| `documents` | `list[string]` | `[]` | Accessible document names |
| `apis` | `list[string]` | `[]` | Accessible API names |
| `skills` | `list[string]` | `[]` | Accessible skill names |
| `agents` | `list[string]` | `[]` | Accessible agent names |
| `rules` | `list[string]` | `[]` | Accessible rule IDs |
| `facts` | `list[string]` | `[]` | Accessible fact names |

---

## Domain Config Files

Domain configs (`domains/<name>/config.yaml`) use the same structure as top-level for data sources and add domain-specific metadata.

```yaml
name: Sales Analytics
description: Sales and customer data for revenue analysis
owner: ""
steward: ""
definition: ""
path: ""                          # Dot-delimited hierarchy (auto-set from dir name)
tier: system                      # system | shared | user
active: true
order: 0

# Data sources (same structure as top-level)
databases:
  sales:
    uri: sqlite:///data/sales.db
    description: Sales CRM

apis: {}
documents:
  business_rules:
    path: docs/business_rules.md
    description: Business policies

# Domain-specific sections
glossary: {}
relationships: {}
rights: {}
facts: {}
learnings: {}
entity_resolution: []
ner_stop_list: []
system_prompt: ""
databases_description: ""

# Domain-scoped task routing (overrides global for this domain)
task_routing:
  sql_generation:
    models:
      - provider: together
        model: meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo
        timeout_seconds: 30
      - provider: anthropic
        model: claude-sonnet-4-20250514
        timeout_seconds: 120

# Child domain references (DAG composition)
domains: []

# Regression testing
golden_questions: []

# Domain-scoped permissions (from permissions.yaml in domain dir)
permissions: null
```

**Domain metadata fields:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `name` | `string` | yes | — | Display name |
| `description` | `string` | no | `""` | Domain description |
| `owner` | `string` | no | `""` | Domain owner |
| `steward` | `string` | no | `""` | Data steward |
| `definition` | `string` | no | `""` | Extended definition |
| `path` | `string` | no | dir name | Dot-delimited hierarchy path |
| `tier` | `string` | no | `"system"` | `system`, `shared`, or `user` |
| `active` | `bool` | no | `true` | Whether domain is active |
| `order` | `int` | no | `0` | Sort order |
| `domains` | `list[string]` | no | `[]` | Child domain names (DAG composition) |
| `golden_questions` | `list[GoldenQuestion]` | no | `[]` | Regression test questions |

**`golden_questions` entry:**

```yaml
golden_questions:
  - question: "What are the top 5 customers by revenue?"
    tags: [smoke, revenue]
    objectives:                        # Original question + follow-ups
      - "What are the top 5 customers by revenue?"
      - "Break down by quarter"
    system_prompt: |                   # Captured at test creation time
      You are a business analyst assistant analyzing company data.
    step_hints:                        # Reference code from exploratory session
      - step_number: 1
        code: "df = store.query('SELECT ...')"
    expect:
      terms:
        - name: revenue
          has_definition: true
      grounding:
        - entity: customers
          resolves_to: [sales.customers]
      relationships:
        - subject: customer
          verb: PLACES
          object: order
      expected_outputs:
        - name: top_customers
          type: table
          columns: [customer_name, total_revenue]
      end_to_end:
        result_contains: ["revenue"]
        judge_prompt: "..."            # Custom or default LLM judge prompt
        validator_code: |              # Python code; raise AssertionError to fail
          assert len(tables['top_customers']) == 5
        plan_min_steps: 2
        expect_success: true
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `question` | `string` | yes | — | The test question |
| `tags` | `list[string]` | no | `[]` | Tags for filtering (e.g., `smoke`, `e2e`) |
| `objectives` | `list[string]` | no | `[]` | Original question + follow-ups for multi-turn tests |
| `system_prompt` | `string` | no | `""` | Domain context captured at test creation time |
| `step_hints` | `list[dict]` | no | `[]` | Reference code from exploratory session for reproducibility |
| `expect` | `GoldenExpectations` | no | `{}` | Assertion layers (terms, grounding, relationships, expected_outputs, end_to_end) |

---

## Config Merge Behavior (Tiered Config)

Configuration is merged from 5 precedence tiers, highest wins:

| Tier | Source | Description |
|------|--------|-------------|
| 1 | `config.yaml` (system) | Base configuration |
| 2 | `domains/<name>/config.yaml` | System domain configs (one or more) |
| 3 | `.constat/<user_id>/config.yaml` | User overrides |
| 4 | `.constat/<user_id>/domains/<name>/config.yaml` | User domain overrides |
| 5 | Session runtime | Dynamic additions |

**Merge rules:**
- **Dict sections** (`databases`, `apis`, `documents`, `glossary`, `relationships`, `rights`, `facts`, `learnings`, `skills`): deep-merged by key. Higher tiers override individual keys.
- **Scalar values** (`system_prompt`, `databases_description`): higher tier replaces entirely.
- **Null values** in higher tiers delete the key from the result.
- **Task routing**: collected per-domain (not merged across domains). Each domain's routing is independent. User-level task routing overrides system-level.
- **Entity resolution**: list from each domain is collected independently.

**Security protection**: When merging user config into engine config, certain database/API fields are protected and cannot be overridden by user config: `uri`, `hosts`, `endpoint`, `endpoint_url`.

**Within a tier**, multiple domains merge alphabetically, then the parent config overlays.

### Attribution

The resolved config tracks which tier set each value via `source_of(path)`, returning one of: `system`, `system_domain`, `user`, `user_domain`, `session`.
