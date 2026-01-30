are# Constat

A multi-step reasoning engine for data analysis with verifiable, auditable logic.

## Overview

Constat enables LLM-powered data analysis with two key principles:

1. **Verifiable, Auditable Logic**: Every conclusion can be traced back to its source data and reasoning steps
2. **Universal Data Integration**: Connect to SQL databases, NoSQL stores, and cloud data services through a unified interface

## Quick Start

Try the included demo environment:

```bash
# Setup demo data (SQLite databases, CSV, JSON files)
python demo/setup_demo.py

# Set your API key
export ANTHROPIC_API_KEY=your_key_here

# View available data sources
constat schema -c demo/config.yaml

# Start interactive session
constat repl -c demo/config.yaml
```

Example queries to try:
- "Top 5 customers by total order value"
- "Which pages have the highest bounce rate?"
- "Average performance rating by department"

## Execution and Provenance

Constat operates in exploratory mode for data exploration, visualization, and iterative analysis. The LLM generates multi-step plans and executes code to answer questions.

```python
from constat import Session

session = Session(config)
result = session.solve("What are the top 10 customers by revenue this quarter?")
```

- Generates multi-step execution plans
- Each step produces code, output, and narrative
- Results include charts, tables, and insights

### On-Demand Proof Generation

At any point, use `/prove` to generate a formal proof of your results with full derivation traces. This is useful for compliance, financial reporting, and any scenario requiring provable conclusions.

The system automatically:
1. Analyzes the question to identify required facts
2. Determines how to derive each fact from your data sources
3. Executes queries and combines results
4. Returns the answer with full provenance

The derivation trace shows exactly how the conclusion was reached:

```
is_vip(customer_id=C001) = True
  derived_from:
    customer_revenue(customer_id=C001) = 150000
      source: DATABASE
      query: SELECT SUM(amount) FROM orders WHERE customer_id = 'C001'
      executed_at: 2025-01-11T10:23:45Z

    vip_threshold = 100000
      source: CONFIG (config.yaml)

  reasoning: Customer revenue ($150,000) exceeds VIP threshold ($100,000)
```

Every fact in the derivation chain is traceable to either:
- A database query (with the exact SQL and timestamp)
- A configuration value
- A derived calculation (with the facts it depends on)

## Data Store Integration

### SQL Databases (via SQLAlchemy)

Connect to any SQLAlchemy-supported database:

```yaml
# config.yaml
databases:
  sales:
    uri: postgresql://${DB_USER}:${DB_PASS}@localhost/sales
    description: "Sales transactions and customer data"

  analytics:
    uri: bigquery://my-project/analytics
    description: "Analytics warehouse"
```

Supported: PostgreSQL, MySQL, SQLite, Oracle, SQL Server, BigQuery, Snowflake, and more.

### NoSQL Databases

Configure NoSQL databases in YAML with type-specific options:

```yaml
# config.yaml
databases:
  # MongoDB
  mongo:
    type: mongodb
    uri: mongodb://localhost:27017
    database: mydb
    description: "Document store"

  # Cassandra
  cassandra:
    type: cassandra
    keyspace: my_keyspace
    hosts: [node1, node2, node3]
    port: 9042
    username: ${CASSANDRA_USER}
    password: ${CASSANDRA_PASS}

  # Elasticsearch
  elastic:
    type: elasticsearch
    hosts: [http://localhost:9200]
    api_key: ${ES_API_KEY}
    description: "Search index"

  # AWS DynamoDB
  dynamo:
    type: dynamodb
    region: us-east-1
    profile_name: myprofile  # or use aws_access_key_id/aws_secret_access_key

  # Azure Cosmos DB
  cosmos:
    type: cosmosdb
    endpoint: https://account.documents.azure.com:443/
    key: ${COSMOS_KEY}
    database: mydb
    container: mycontainer

  # Google Firestore
  firestore:
    type: firestore
    project: my-gcp-project
    collection: users
    credentials_path: /path/to/credentials.json
```

All connectors provide:
- Schema introspection (inferred for schema-less databases)
- Unified query interface
- Embedding text generation for vector search

### File-Based Data Sources

CSV, JSON, Parquet, and Arrow files can be configured as queryable data sources alongside SQL and NoSQL databases. They appear uniformly in schema discovery and semantic search.

```yaml
# config.yaml
databases:
  # CSV file
  web_metrics:
    type: csv
    path: data/website_metrics.csv
    description: "Daily web analytics with page views and bounce rates"

  # JSON file
  events:
    type: json
    path: data/events.json
    description: "Clickstream events with user actions"

  # JSON Lines (newline-delimited JSON)
  logs:
    type: jsonl
    path: data/application.jsonl
    description: "Application log entries"

  # Parquet file
  transactions:
    type: parquet
    path: data/transactions.parquet
    description: "Historical transaction records"

  # Arrow/Feather file
  features:
    type: arrow  # or 'feather'
    path: data/ml_features.arrow
    description: "ML feature vectors"
```

**Remote file support:** Paths can be local or remote URLs (s3://, https://, etc.):

```yaml
databases:
  remote_data:
    type: parquet
    path: s3://my-bucket/data/sales.parquet
    description: "Sales data from S3"
```

**Generated code access:**
- SQL databases: `db_<name>` (SQLAlchemy connection)
- File sources: `file_<name>` (path string for pandas)

```python
# Generated code for SQL
df = pd.read_sql("SELECT * FROM customers", db_sales)

# Generated code for files
df = pd.read_csv(file_web_metrics)
df = pd.read_json(file_events)
df = pd.read_parquet(file_transactions)
```

**Schema introspection:** File sources are automatically sampled to infer:
- Column names and types
- Row counts
- Sample values (for semantic search)

## Architecture

```
                    +-------------------+
                    |   User Request    |
                    +-------------------+
                            |
                    +-------v-------+
                    | Mode Selector |
                    +-------+-------+
                            |
           +----------------+----------------+
           |                                 |
    +------v------+                   +------v------+
    | Exploratory |                   |  Auditable  |
    |    Mode     |                   |    Mode     |
    +------+------+                   +------+------+
           |                                 |
    +------v------+                   +------v------+
    | Multi-Step  |                   |    Fact     |
    |   Planner   |                   |  Resolver   |
    +------+------+                   +------+------+
           |                                 |
           +----------------+----------------+
                            |
              +-------------v-------------+
              |     Discovery Tools       |
              |  (tool-based or prompt)   |
              +-------------+-------------+
                            |
    +-------+-------+-------+-------+-------+-------+
    |       |       |       |       |       |       |
   SQL   MongoDB Cassandra DynamoDB  Files  CosmosDB
                                  (CSV/JSON/
                                   Parquet)
```

### Discovery Mode (Automatic)

Constat automatically detects whether the LLM model supports tool calling:

| Model | Tool Calling | Prompt Mode |
|-------|-------------|-------------|
| Claude 3+ (Opus, Sonnet, Haiku) | Yes | Minimal prompt + discovery tools |
| GPT-4, GPT-3.5-turbo | Yes | Minimal prompt + discovery tools |
| Gemini | Yes | Minimal prompt + discovery tools |
| Claude 2, Claude Instant | No | Full schema/API/docs in prompt |
| GPT-3 Instruct | No | Full schema/API/docs in prompt |

**Tool-based discovery** (modern models):
- Minimal system prompt (~500 tokens)
- LLM uses tools to discover relevant schema, APIs, and documents on-demand
- On-demand loading reduces token usage by 80-95%

Discovery tools include:
- **Schema**: `list_tables`, `get_table_schema`, `search_tables`, `get_sample_values`
- **APIs**: `list_api_operations`, `get_operation_details`, `search_operations`
- **Documents**: `list_documents`, `search_documents`, `get_document`, `get_document_section`

Document discovery supports:
- **Unstructured docs**: Markdown, text, PDF files with semantic search (sentence-transformers)
- **Structured files**: CSV, JSON, JSONL, Parquet with automatic schema inference
- **Section extraction**: Retrieve specific sections from large documents

Documents are indexed into a persistent vector store using DuckDB VSS (lazy, on first access).
The index persists at `~/.constat/vectors.duckdb` by default, eliminating re-indexing on restart.
During fact resolution and knowledge synthesis, relevant document excerpts are automatically
retrieved via semantic search - no explicit discovery step required.

**Full prompt mode** (legacy models):
- Comprehensive system prompt with all metadata
- Complete schema, API docs, and document content embedded
- No tool calling required

## CLI Usage

```bash
# Solve a single problem
constat solve "What are the top 5 customers by revenue?" -c config.yaml

# Start interactive REPL
constat repl -c config.yaml

# View session history
constat history

# Resume a previous session
constat resume abc123 -c config.yaml

# Validate config file
constat validate -c config.yaml

# Show database schema
constat schema -c config.yaml

# Generate sample config
constat init
```

### REPL Commands

Once in the interactive REPL, these commands are available:

**Session & Navigation:**

| Command | Description |
|---------|-------------|
| `/help`, `/h` | Show all commands |
| `/quit`, `/q` | Exit |
| `/reset` | Clear session state and start fresh |
| `/redo [instruction]` | Retry last query (optionally with modifications) |
| `/user [name]` | Show or set current user |

**Data Inspection:**

| Command | Description |
|---------|-------------|
| `/tables` | List tables in session datastore |
| `/show <table>` | Show table contents |
| `/query <sql>` | Run SQL query on datastore |
| `/export <table> [file]` | Export table to CSV or XLSX |
| `/code [step]` | Show generated code (all or specific step) |
| `/state` | Show session state |
| `/artifacts [all]` | Show artifacts (use 'all' to include intermediate) |

**Data Sources:**

| Command | Description |
|---------|-------------|
| `/databases`, `/db` | List configured databases |
| `/files` | List all data files |
| `/doc <path> [name]` | Add a document to this session |
| `/discover [scope] <query>` | Search data sources (scope: database\|api\|document) |
| `/update`, `/refresh` | Refresh metadata and rebuild cache |

**Facts & Memory:**

| Command | Description |
|---------|-------------|
| `/facts` | Show cached facts from this session |
| `/remember <fact>` | Persist a session fact across sessions |
| `/forget <name>` | Forget a remembered fact |
| `/correct <text>` | Record a correction for future reference |
| `/learnings` | Show learnings and rules |
| `/compact-learnings` | Promote similar learnings into rules |

**Plans & History:**

| Command | Description |
|---------|-------------|
| `/save <name>` | Save current plan for replay |
| `/share <name>` | Save plan as shared (all users) |
| `/plans` | List saved plans |
| `/replay <name>` | Replay a saved plan |
| `/history`, `/sessions` | List recent sessions |
| `/resume <id>` | Resume a previous session |
| `/summarize <target>` | Summarize plan\|session\|facts\|<table> |

**Verification:**

| Command | Description |
|---------|-------------|
| `/prove` | Verify conversation claims with auditable proof |

**Settings:**

| Command | Description |
|---------|-------------|
| `/verbose [on\|off]` | Toggle verbose mode |
| `/raw [on\|off]` | Toggle raw output display |
| `/insights [on\|off]` | Toggle insight synthesis |
| `/preferences` | Show current preferences |
| `/context` | Show context size and token usage |
| `/compact` | Compact context to reduce token usage |

**Saved Plans & Replay:**
- `/save` stores the executed code (not just the plan) for deterministic replay
- `/replay` executes the stored code without regenerating it via LLM
- Relative terms ("today", "last month", "within policy") are evaluated dynamically on each replay
- Explicit values ("January 2006", "above 100 units") are hardcoded as specified

**Brief mode:** Use keywords like "briefly", "tl;dr", "just show" in your query to skip the synthesis step and get raw results faster.

### Interactive Visualizations

Constat can generate interactive visualizations that are saved as HTML files you can open in your browser:

```
> Show me an interactive map of countries using the Euro

Interactive map: /Users/you/.constat/outputs/euro_countries.html
```

**Supported visualization types:**

| Type | Library | Example |
|------|---------|---------|
| Interactive maps | Folium | Geographic data, markers, choropleth maps |
| Interactive charts | Plotly | Bar, line, scatter, pie, treemap, etc. |
| Statistical charts | Altair | Declarative statistical visualizations |
| Static plots | Matplotlib/Seaborn | Traditional Python plotting |

Generated visualizations are:
- Saved to `~/.constat/outputs/` as self-contained HTML files
- Stored as artifacts in the session datastore (for UI display)
- Fully interactive in your browser (zoom, hover, pan)

**Example queries:**
- "Create an interactive map showing customer locations"
- "Show me a bar chart of revenue by region"
- "Visualize the correlation between price and quantity"

### Dashboards

Request a "dashboard" to generate multi-panel visualizations automatically:

```
> Create a dashboard of sales performance

[Generates 2x2 grid with: revenue trend, breakdown by category, top products, KPI summary]
```

Dashboard layouts adapt to data:
- **Time series**: Trend + summary stats (1x2)
- **Categories**: Overview, breakdown, comparison, detail (2x2)
- **KPI-focused**: KPI cards on top, supporting charts below (3x2)

### Learning System

Constat learns from corrections and errors to improve over time. Learnings are stored per-user and persist across sessions.

**Explicit corrections:**
```
/correct "revenue" means gross sales minus returns
```

**Automatic learning:** When code generation fails and retries succeed, the error-to-fix pattern is captured automatically. These learnings are injected into future code generation prompts.

**Natural language detection:** Corrections in conversation are detected automatically:
- "That's wrong, active users means 30-day logins"
- "Actually, in our context, churn means 60 days inactive"

**Compaction:** Similar learnings are automatically promoted to rules when patterns emerge:
```
/learnings          # Show rules and pending learnings
/compact-learnings  # Manually trigger compaction
```

## Configuration

### Configuration Hierarchy

Constat uses a layered configuration system with merge semantics:

```
┌─────────────────────────────────────────────────────────────┐
│  Engine Config (config.yaml)                                │
│  - LLM settings, execution defaults, artifact storage       │
│  - Shared by all users, deployed with the application       │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ merged with (user overrides engine)
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  User Config (user-config.yaml or runtime dict)             │
│  - Same YAML structure as engine config                     │
│  - Provides user-specific database credentials              │
│  - Can override any engine setting                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Final Config                                               │
│  - Engine values as defaults                                │
│  - User values override where specified                     │
│  - Databases merged by name                                 │
└─────────────────────────────────────────────────────────────┘
```

**Merge Rules:**
- User config uses **the same YAML structure** as engine config
- User values **override** engine values
- Databases are **merged by name** (user credentials added to matching engine database)
- New databases in user config are **added** to the list

### Complete Configuration Reference

```yaml
# config.yaml - Complete reference with all options

#==============================================================================
# LLM CONFIGURATION
#==============================================================================
llm:
  # Required: LLM provider
  provider: anthropic                    # anthropic | openai | gemini | grok | ollama

  # Required: Model to use
  model: claude-sonnet-4-20250514

  # Required: API key (use env var for security)
  api_key: ${ANTHROPIC_API_KEY}

  # Optional: Task-type routing with automatic model escalation
  # Each task type maps to an ordered list of models to try (local-first with cloud fallback)
  task_routing:
    sql_generation:
      models:
        - provider: ollama                       # Try local SQLCoder first
          model: sqlcoder:7b
        - model: claude-sonnet-4-20250514        # Escalate to cloud on failure
    python_analysis:
      models:
        - model: claude-sonnet-4-20250514
      high_complexity_models:                    # Use for complex analysis
        - model: claude-opus-4-20250514
    planning:
      models:
        - model: claude-sonnet-4-20250514
    summarization:
      models:
        - model: claude-3-5-haiku-20241022       # Fast, cheap for summaries

  # Optional: Provider-specific settings
  # base_url: https://api.anthropic.com  # Custom API endpoint

#==============================================================================
# DATABASE CONFIGURATION
#==============================================================================

# Optional: Global context explaining relationship between databases
databases_description: |
  These databases represent different systems within a retail company.
  Sales data is in sales_db, inventory in inventory_db.

# Required: Databases keyed by name (dict format for easy merging)
databases:
  # SQL Database (SQLAlchemy URI) with credentials in env vars
  sales_db:
    uri: postgresql://${DB_USER}:${DB_PASS}@localhost:5432/sales
    description: "Customer transactions, orders, and revenue data"

  # With separate credentials (recommended - can be overridden by user config)
  inventory_db:
    uri: mysql+pymysql://localhost:3306/inventory
    username: ${INVENTORY_USER}
    password: ${INVENTORY_PASS}
    description: "Warehouse stock levels and shipments"

  # Without credentials (user config provides them)
  sensitive_db:
    uri: postgresql://localhost:5432/sensitive
    description: "Requires user authentication"

  # SQLite (local file)
  analytics:
    uri: sqlite:///./data/analytics.db
    description: "Historical metrics and trends"

  # BigQuery
  warehouse:
    uri: bigquery://my-project/dataset
    description: "Data warehouse"

  #----------------------------------------------------------------------------
  # File-Based Data Sources - CSV, JSON, Parquet, Arrow
  #----------------------------------------------------------------------------

  # CSV file
  web_metrics:
    type: csv
    path: data/website_metrics.csv          # Local path or remote URL
    description: "Web analytics data"
    sample_size: 100                        # Rows to sample for schema inference

  # JSON file
  events:
    type: json
    path: data/events.json
    description: "Event tracking data"

  # JSON Lines (newline-delimited JSON)
  logs:
    type: jsonl
    path: data/application.jsonl

  # Parquet file (supports s3://, https://, etc.)
  transactions:
    type: parquet
    path: s3://bucket/data/transactions.parquet
    description: "Transaction records"

  # Arrow/Feather file
  features:
    type: arrow                             # or 'feather'
    path: data/ml_features.arrow

  #----------------------------------------------------------------------------
  # NoSQL Databases - use 'type' to specify the database type
  #----------------------------------------------------------------------------

  # MongoDB
  documents:
    type: mongodb
    uri: mongodb://localhost:27017          # MongoDB connection URI
    database: myapp                         # Database name
    sample_size: 100                        # Docs to sample for schema inference
    description: "Document store"

  # Cassandra / DataStax Astra
  cassandra:
    type: cassandra
    keyspace: my_keyspace                   # Required
    hosts: [node1, node2, node3]            # Contact points (or use cloud_config)
    port: 9042                              # CQL port (default: 9042)
    username: ${CASSANDRA_USER}             # Optional auth
    password: ${CASSANDRA_PASS}
    # cloud_config:                         # For DataStax Astra
    #   secure_connect_bundle: /path/to/bundle.zip
    description: "Wide-column store"

  # Elasticsearch
  search:
    type: elasticsearch
    hosts: [http://localhost:9200]          # One or more hosts
    api_key: ${ES_API_KEY}                  # API key auth (or use username/password)
    # username: elastic
    # password: ${ES_PASSWORD}
    description: "Search and analytics"

  # AWS DynamoDB
  dynamo:
    type: dynamodb
    region: us-east-1                       # AWS region
    profile_name: myprofile                 # AWS profile (or use access keys)
    # aws_access_key_id: ${AWS_KEY}
    # aws_secret_access_key: ${AWS_SECRET}
    # endpoint_url: http://localhost:8000   # For local DynamoDB
    description: "Serverless key-value"

  # Azure Cosmos DB
  cosmos:
    type: cosmosdb
    endpoint: https://account.documents.azure.com:443/
    key: ${COSMOS_KEY}                      # Primary or secondary key
    database: mydb                          # Database name
    container: mycontainer                  # Container name
    description: "Globally distributed DB"

  # Google Firestore
  firestore:
    type: firestore
    project: my-gcp-project                 # GCP project ID
    collection: users                       # Collection to introspect
    credentials_path: /path/to/creds.json   # Service account credentials
    description: "Real-time document DB"

#==============================================================================
# EXTERNAL APIs
#==============================================================================

# Optional: External APIs keyed by name
apis:
  # GraphQL API (auto-introspects schema)
  countries:
    type: graphql
    url: https://countries.trevorblades.com/graphql
    description: "Country data for geographic enrichment"

  # OpenAPI from URL (auto-discovers all endpoints)
  petstore:
    type: openapi
    spec_url: https://petstore.swagger.io/v2/swagger.json
    description: "Pet store API"

  # OpenAPI from local file
  internal_api:
    type: openapi
    spec_path: ./specs/internal-api.yaml
    description: "Internal services"

  # OpenAPI inline (for simple APIs without a spec file)
  weather:
    type: openapi
    url: https://api.weather.gov              # Base URL
    spec_inline:
      openapi: "3.0.0"
      info:
        title: Weather API
        version: "1.0"
      paths:
        /points/{lat},{lon}:
          get:
            operationId: getPoint
            summary: Get forecast office and grid info for a location
            parameters:
              - name: lat
                in: path
                required: true
                schema:
                  type: number
              - name: lon
                in: path
                required: true
                schema:
                  type: number

  # API with authentication
  protected_api:
    type: openapi
    spec_url: https://api.example.com/openapi.json
    # Auth option 1: Bearer token
    auth_type: bearer
    auth_token: ${API_TOKEN}
    # Auth option 2: Basic auth
    # auth_type: basic
    # auth_username: ${API_USER}
    # auth_password: ${API_PASS}
    # Auth option 3: API key
    # auth_type: api_key
    # api_key: ${API_KEY}
    # api_key_header: X-API-Key            # Header name (default: X-API-Key)
    # Auth option 4: Raw headers
    # headers:
    #   Authorization: Bearer ${TOKEN}

#==============================================================================
# REFERENCE DOCUMENTS
#==============================================================================

# Optional: Documents to include in reasoning (keyed by name)
documents:
  # From local file
  business_rules:
    type: file
    path: ./docs/business-rules.md
    description: "Revenue calculation rules and thresholds"
    tags: [rules, revenue]

  # From HTTP (works for wikis, GitHub raw files, etc.)
  wiki_data_dictionary:
    type: http
    url: https://wiki.example.com/api/v2/pages/12345/export/view
    headers:
      Authorization: Bearer ${WIKI_TOKEN}
    description: "Data dictionary from internal wiki"
    cache_ttl: 3600                         # Refresh every hour

  # From Confluence
  analytics_guide:
    type: confluence
    url: https://mycompany.atlassian.net
    space_key: ANALYTICS
    page_title: "Metrics Definitions"
    username: ${CONFLUENCE_USER}
    api_token: ${CONFLUENCE_TOKEN}
    follow_links: true                      # Build corpus from linked pages
    max_depth: 2                            # Follow 2 levels deep
    max_documents: 10                       # Limit to 10 documents

  # Inline content
  glossary:
    type: inline
    content: |
      ## Key Terms
      - VIP: Customer with lifetime value > $100k
      - Churn: Customer inactive for 90+ days
      - MRR: Monthly Recurring Revenue
    description: "Business glossary"

#==============================================================================
# DOMAIN CONTEXT (SYSTEM PROMPT)
#==============================================================================

# Optional: Domain knowledge for the LLM
# This is included in every prompt and helps the LLM understand your data
system_prompt: |
  You are analyzing data for a retail company.

  Key business concepts:
  - customer_tier: Found in sales_db.customers.tier_level (gold/silver/bronze)
  - revenue: Found in sales_db.transactions, aggregate by SUM(amount)
  - region: Geographic sales regions (north/south/east/west)

  Common relationships:
  - Customers linked to transactions via customer_id
  - Targets set per-region per-quarter in sales_db.targets

  Business rules:
  - "Underperforming" means < 80% of target
  - "VIP customer" means tier_level = 'gold' OR lifetime_value > 100000
  - Q1 = Jan-Mar, Q2 = Apr-Jun, Q3 = Jul-Sep, Q4 = Oct-Dec

  Data quality notes:
  - Some older transactions have NULL region (default to 'unknown')
  - Customer tiers are updated monthly

#==============================================================================
# CONTEXT PRELOAD (Optional)
#==============================================================================
# Preload relevant table metadata into context at session start.
# This eliminates discovery tool calls for common query patterns.
# Cache is built once and persists until /refresh is run.

context_preload:
  # Seed patterns representing typical queries/domains
  # Used to match against table names, column names via vector similarity
  seed_patterns:
    - "sales"
    - "customer"
    - "revenue"
    - "inventory"
  similarity_threshold: 0.3        # Min similarity (0-1) for table inclusion
  max_tables: 50                   # Max tables to preload
  max_columns_per_table: 30        # Limit columns per table

#==============================================================================
# EXECUTION SETTINGS
#==============================================================================
execution:
  # Timeout per step in seconds (default: 60)
  timeout_seconds: 60

  # Max retry attempts per step (default: 10)
  max_retries: 10

  # Optional: Restrict imports in generated code
  allowed_imports:
    - pandas
    - numpy
    - scipy
    - sklearn
    - plotly
    - altair
    - matplotlib
    - seaborn
    - folium

#==============================================================================
# STORAGE SETTINGS
#==============================================================================
storage:
  # SQLAlchemy URI for artifact storage
  # Default: SQLite file per session in ~/.constat/sessions/

  # SQLite (simple, default)
  artifact_store_uri: sqlite:///~/.constat/artifacts.db

  # PostgreSQL (production, multi-user)
  # artifact_store_uri: postgresql://${DB_USER}:${DB_PASS}@localhost/constat

  # DuckDB (analytical workloads, requires duckdb-engine)
  # artifact_store_uri: duckdb:///~/.constat/artifacts.duckdb

  # Vector store for document embeddings (default: DuckDB VSS)
  vector_store:
    backend: duckdb  # "duckdb" (persistent) or "numpy" (in-memory)
    db_path: ~/.constat/vectors.duckdb

#==============================================================================
# EMAIL CONFIGURATION
#==============================================================================
email:
  # SMTP server settings
  smtp_host: ${SMTP_HOST}
  smtp_port: 587
  smtp_user: ${SMTP_USER}
  smtp_password: ${SMTP_PASSWORD}

  # Sender information
  from_address: noreply@company.com
  from_name: Constat

  # Use TLS encryption (recommended)
  tls: true

# Note: Email sending is protected by automatic data sensitivity detection.
# When plans involve sensitive data (PII, financial, health, HR data, etc.),
# email operations are blocked unless explicitly authorized with allow_sensitive=True.
# The planner classifies data sensitivity based on privacy regulations (GDPR, HIPAA, etc.).
```

### Environment Variable Substitution

Use `${VAR_NAME}` syntax to reference environment variables:

```yaml
databases:
  production:
    uri: postgresql://${DB_HOST}:${DB_PORT}/${DB_NAME}
    username: ${DB_USER}
    password: ${DB_PASSWORD}

llm:
  api_key: ${ANTHROPIC_API_KEY}
```

Environment variables are substituted at config load time. Missing variables raise an error.

### File References with $ref

Use `$ref` syntax (JSON Schema style) to include content from external files:

```yaml
# Include a single file
permissions:
  $ref: ./permissions.yaml

# Reference in nested structures
projects:
  sales-analytics.yaml:
    $ref: ./projects/sales-analytics.yaml
  hr-reporting.yaml:
    $ref: ./projects/hr-reporting.yaml
```

The `$ref` syntax:
- Follows JSON Schema/OpenAPI conventions
- Supports recursive references (included files can use `$ref`)
- Environment variables are substituted in included files
- Paths are relative to the including file's directory

### Projects Configuration

Projects are reusable collections of data sources (databases, APIs, documents) that can be activated per-session:

```yaml
# config.yaml
projects:
  sales-analytics.yaml:
    $ref: ./projects/sales-analytics.yaml
  hr-reporting.yaml:
    $ref: ./projects/hr-reporting.yaml
```

Each project file defines its data sources:

```yaml
# projects/sales-analytics.yaml
name: Sales Analytics
description: Sales and customer data for revenue analysis

databases:
  sales:
    uri: sqlite:///demo/data/sales.db
    description: Customer transactions and orders

apis:
  salesforce:
    type: rest
    url: https://api.salesforce.com/...

documents:
  sales_glossary:
    type: file
    path: ./docs/sales-terms.md
```

### Server Configuration

For API server mode, configure authentication and user permissions:

```yaml
# config.yaml
server:
  # Firebase authentication
  auth_disabled: false                    # Set true for development
  firebase_project_id: your-project-id

  # User permissions (or use $ref to separate file)
  permissions:
    $ref: ./permissions.yaml
```

### User Permissions

Control access to resources per-user. Permissions determine what metadata the LLM sees during analysis—resources without permission are hidden from discovery.

```yaml
# permissions.yaml
users:
  admin@company.com:
    admin: true          # Full access to ALL resources + can manage projects
    projects: []         # (ignored for admins)
    databases: []        # (ignored for admins)
    documents: []        # (ignored for admins)
    apis: []             # (ignored for admins)

  analyst@company.com:
    admin: false
    projects:
      - sales-analytics        # Can activate these projects (use project key, not filename)
    databases:
      - inventory              # Can query these databases (from core config)
      - web_metrics
    documents:
      - business_rules         # Can access these documents (from core config)
    apis:
      - countries              # Can call these APIs (from core config)

# Default permissions for unlisted users
default:
  admin: false
  projects: []
  databases: []
  documents: []
  apis: []
```

#### Permission Rules

**Admin Users (`admin: true`):**
- Full access to ALL resources across all projects and core config
- Can manage projects from the UI (create, edit, delete)
- Permission arrays are ignored—admins see everything

**Non-Admin Users:**
- Only see resources explicitly listed in their permission arrays
- Project access grants access to ALL resources within that project
- Core config resources require explicit permission in the user's arrays

**Unlisted Users:**
- Get the `default` permissions (typically empty = no access)

#### No Permissions = Full Access

If no `permissions` section is configured in `server` settings, ALL resources are available to everyone. This is intentional for development/single-user scenarios.

```yaml
# config.yaml without permissions = everything available
server:
  auth_disabled: true
  # No permissions configured → no filtering
```

#### Project Permissions and Resource Access

When a user has access to a project, they automatically get access to all resources defined within that project:

```yaml
# Example: analyst has access to sales-analytics project
# This grants them access to all databases, APIs, and documents in that project
projects:
  sales-analytics:
    $ref: ./projects/sales-analytics.yaml
```

```yaml
# projects/sales-analytics.yaml
name: Sales Analytics
databases:
  sales:           # analyst can query this
    uri: sqlite:///demo/data/sales.db
apis:
  salesforce:      # analyst can call this
    type: rest
    url: https://api.salesforce.com/...
documents:
  sales_glossary:  # analyst can search this
    type: file
    path: ./docs/sales-terms.md
```

The user's effective permissions are the **union** of:
1. Explicit permissions in their user entry
2. All resources from their accessible active projects

#### How Permission Filtering Works

Permissions are enforced at the **metadata level**, not runtime:

1. **Discovery Tools**: `list_tables`, `list_apis`, `list_documents` only return allowed resources
2. **Schema Manager**: Database summaries exclude databases the user cannot access
3. **System Prompts**: API/document/database descriptions only include allowed resources
4. **Semantic Search**: Only indexes and searches allowed documents

The LLM never sees metadata for resources the user cannot access. Since the LLM doesn't know about restricted resources, it cannot generate code to query them.

```
User Query → Permission Check → Filtered Metadata → LLM → Generated Code
                                     ↓
                            Only sees allowed
                            databases, APIs, docs
```

#### Permission Filtering by Resource Type

| Resource | Filtered In | Effect |
|----------|-------------|--------|
| Databases | `list_tables`, `search_tables`, schema summary | LLM doesn't see table schemas |
| APIs | `list_apis`, `list_api_operations`, system prompt | LLM doesn't see API endpoints |
| Documents | `list_documents`, `search_documents` | LLM can't search restricted docs |
| Projects | Project activation | User can't load restricted projects |

### Database Credentials

Three ways to provide database credentials:

**1. Embedded in URI (simple, not recommended for production):**
```yaml
databases:
  - name: main
    uri: postgresql://myuser:mypass@localhost/mydb
```

**2. Separate fields with environment variables (recommended for single-user):**
```yaml
databases:
  - name: main
    uri: postgresql://localhost/mydb
    username: ${DB_USER}
    password: ${DB_PASSWORD}
```

**3. User config file (multi-user deployments):**

Engine config defines databases without credentials:
```yaml
# config.yaml (engine config - no credentials)
databases:
  main:
    uri: postgresql://localhost/mydb
    description: "Main application database"
  analytics:
    uri: postgresql://localhost/analytics
```

Each user provides their credentials via user config (same structure):
```yaml
# user-config.yaml
databases:
  main:
    username: alice
    password: secret123
  analytics:
    username: alice
    password: secret456
  # Users can also add their own databases
  personal_db:
    uri: sqlite:///~/my-data.db
    description: "My personal analysis database"
```

```python
# Load with user config - credentials merged automatically
config = Config.from_yaml("config.yaml", user_config_path="user-config.yaml")

# Or provide as dict at runtime (e.g., from API request)
config = Config.from_yaml("config.yaml", user_config={
    "databases": {
        "main": {"username": "alice", "password": "secret123"}
    }
})
```

The configs are deep-merged by database name:
- User credentials fill in engine database definitions
- Users can add entirely new databases
- User values override engine values where both exist

## Discovery Tools

The discovery module provides on-demand access to schema, API, and document information.

### Available Tools

| Category | Tools |
|----------|-------|
| **Schema** | `list_databases`, `list_tables`, `get_table_schema`, `search_tables`, `get_table_relationships`, `get_sample_values` |
| **API** | `list_apis`, `list_api_operations`, `get_operation_details`, `search_operations` |
| **Documents** | `list_documents`, `get_document`, `search_documents`, `get_document_section` |
| **Facts** | `resolve_fact`, `add_fact`, `extract_facts_from_text`, `list_known_facts` |

### Usage

```python
from constat.discovery import DiscoveryTools, PromptBuilder

# Create discovery tools
tools = DiscoveryTools(
    schema_manager=schema_manager,
    api_catalog=api_catalog,
    config=config,
)

# Build prompt automatically based on model capabilities
builder = PromptBuilder(tools)
prompt, use_tools = builder.build_prompt("claude-sonnet-4-20250514")
# → use_tools=True for modern models (minimal prompt)
# → use_tools=False for legacy models (full metadata in prompt)

# Execute discovery tools directly
databases = tools.execute("list_databases", {})
relevant = tools.execute("search_tables", {"query": "customer purchases"})

# Check token savings
estimate = builder.estimate_tokens("claude-sonnet-4-20250514")
print(f"Savings: {estimate['savings_percent']}%")
```

### Artifact Store Configuration

The artifact store holds session state, execution history, and generated artifacts:

```yaml
storage:
  # SQLite (default, zero-config)
  artifact_store_uri: sqlite:///~/.constat/artifacts.db

  # PostgreSQL (production, multi-user)
  artifact_store_uri: postgresql://${DB_USER}:${DB_PASS}@localhost/constat

  # DuckDB (requires duckdb-engine package)
  artifact_store_uri: duckdb:///~/.constat/artifacts.duckdb
```

## GraphQL API

Constat provides a GraphQL API for integration with web applications:

```python
from constat.api.graphql import create_app

app = create_app(
    schema_manager=schema_manager,
    fact_resolver=fact_resolver,
    config=config,
)

# Run with: uvicorn constat.api.graphql.app:app
```

Available operations:
- `createSession`: Start a new analysis session
- `solve`: Execute a query in exploratory mode
- `resolveFact`: Resolve a fact with full provenance
- `sessionEvents`: Real-time updates via subscriptions

## LLM Provider Support

Multiple LLM providers for flexibility and cost optimization:

```python
from constat.providers import (
    AnthropicProvider,
    OpenAIProvider,
    GeminiProvider,
    GrokProvider,
    OllamaProvider,  # Local models
    TaskRouter,      # Task-type routing with automatic escalation
)

# Direct provider usage
provider = AnthropicProvider(model="claude-sonnet-4-20250514")

# Task-type routing (recommended for production)
from constat.core.config import LLMConfig
router = TaskRouter(llm_config)
result = router.execute(
    task_type=TaskType.SQL_GENERATION,
    system="Generate SQL...",
    user_message="Get top 5 customers",
)
```

### Task-Type Routing

Route tasks to the most appropriate model with automatic escalation on failure:

```yaml
# config.yaml
llm:
  provider: anthropic              # Default provider
  model: claude-sonnet-4-20250514
  api_key: ${ANTHROPIC_API_KEY}

  task_routing:
    # SQL generation: try local SQLCoder first, escalate to cloud on failure
    sql_generation:
      models:
        - provider: ollama
          model: sqlcoder:7b
        - model: claude-sonnet-4-20250514

    # Python analysis: use sonnet, escalate to opus for complex tasks
    python_analysis:
      models:
        - model: claude-sonnet-4-20250514
      high_complexity_models:
        - model: claude-opus-4-20250514

    # Planning: always use cloud (no local fallback)
    planning:
      models:
        - model: claude-sonnet-4-20250514

    # Summarization: use fast/cheap model
    summarization:
      models:
        - model: claude-3-5-haiku-20241022
```

**Task Types:**
| Task Type | Description | Typical Models |
|-----------|-------------|----------------|
| `planning` | Multi-step plan generation | claude-sonnet, claude-opus |
| `sql_generation` | Text-to-SQL queries | sqlcoder, claude-sonnet |
| `python_analysis` | DataFrame transformations | codellama, claude-sonnet |
| `summarization` | Result synthesis | claude-haiku, llama3.2 |
| `intent_classification` | User intent detection | phi3, claude-haiku |
| `fact_resolution` | Auditable fact derivation | claude-sonnet |

**Benefits:**
- **Automatic escalation**: Local model fails → cloud model tries automatically
- **Cost optimization**: Use cheaper/local models when they succeed
- **Latency optimization**: Fast local models for quick tasks
- **Privacy**: Route sensitive data through local models first
- **Observability**: Track escalation rates per task type

**Supported Providers:**
| Provider | Name | Tool Support |
|----------|------|--------------|
| Anthropic | `anthropic` | Yes |
| OpenAI | `openai` | Yes |
| Google Gemini | `gemini` | Yes |
| xAI Grok | `grok` | Yes |
| Ollama (local) | `ollama` | Yes (llama3.2+) |
| Together AI | `together` | Yes |
| Groq | `groq` | Yes |

## Skills

Skills are domain-specific knowledge modules that provide specialized context and guidance for analysis tasks. They are loaded dynamically based on the query context.

This follows the standard skill/prompt pattern used by Anthropic (Claude Code), OpenAI, and other AI providers for extending chatbot capabilities with domain-specific knowledge.

### Skill Structure

Skills are stored in directories following the pattern `skills/<skill-name>/SKILL.md`:

```
.constat/skills/
└── financial-analysis/
    └── SKILL.md
```

### SKILL.md Format

Each skill is defined as a Markdown file with YAML frontmatter:

```markdown
---
name: financial-analysis
description: Specialized instructions for financial data analysis
allowed-tools:
  - Read
  - Grep
  - list_tables
  - get_table_schema
---

# Financial Analysis Skill

## Key Concepts
- Revenue recognition principles
- Common financial metrics (Gross Margin, EBITDA, etc.)
- Time period handling (MTD, QTD, YTD)

## Analysis Guidelines
1. Start with data quality checks
2. Use appropriate aggregations
3. Present results clearly
```

### Skill Discovery Paths

Skills are discovered from multiple locations (in order of precedence):

1. **Project skills**: `.constat/skills/` in the project directory
2. **Global skills**: `~/.constat/skills/` in the user's home directory
3. **Config-specified paths**: Additional paths defined in `config.yaml`

### Configuring Additional Skill Paths

Add custom skill directories in your config file:

```yaml
# config.yaml
skills:
  paths:
    - /shared/team-skills/           # Team shared skills
    - /opt/constat/standard-skills/  # Standard library
    - ~/my-custom-skills/            # Personal skills (~ expanded)
```

Skills in config paths are searched after the default paths, so project and global skills take precedence.

### Link Following

Skills can reference additional files via markdown links. Links are parsed when the skill loads but content is fetched lazily (on-demand):

```markdown
# My Skill

See the [indicator definitions](references/indicators.md) for details.
For API docs, check [the official guide](https://example.com/docs.md).
```

**Supported link types:**
- **Relative paths**: `[text](references/file.md)` - resolved relative to the skill folder
- **URLs**: `[text](https://example.com/doc.md)` - fetched via HTTP

**How it works:**
1. When a skill loads, links are discovered and returned in the response
2. Content is NOT fetched until explicitly requested via `resolve_skill_link`
3. Fetched content is cached for subsequent calls

### Creating a Skill

1. Create a directory: `.constat/skills/my-skill/`
2. Add a `SKILL.md` file with YAML frontmatter
3. Define the skill's context, guidelines, and examples
4. Optionally add referenced files in subdirectories (e.g., `references/`)

Skills are automatically discovered and can be loaded when relevant to a query.

## Key Concepts

### Facts and Provenance

Every piece of information is a `Fact` with:
- **Value**: The actual data
- **Source**: Where it came from (DATABASE, CONFIG, RULE, LLM_KNOWLEDGE)
- **Confidence**: How certain we are (1.0 for database queries, lower for inferences)
- **Because**: The facts this was derived from

### Derivation Traces

When using `/prove`, every conclusion includes a full derivation trace showing:
- The logical chain of reasoning
- All data sources consulted
- The exact queries executed
- Confidence at each step

### Lazy Resolution

Facts are resolved only when needed:
1. Check cache
2. Check config
3. Apply rules
4. Query database
5. Fall back to LLM knowledge
6. Create sub-plan if complex

This ensures efficient execution while maintaining full traceability.

## Installation

```bash
pip install constat

# Optional dependencies for specific databases
pip install constat[postgresql]  # PostgreSQL
pip install constat[mongodb]     # MongoDB
pip install constat[dynamodb]    # AWS DynamoDB
pip install constat[cosmosdb]    # Azure Cosmos DB
pip install constat[firestore]   # Google Firestore
pip install constat[all]         # Everything
```

## License

PolyForm Small Business License 1.0.0

Free for individuals and organizations with <100 employees and <$1M revenue.
For commercial licensing, contact kennethstott@gmail.com.
