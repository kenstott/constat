# Constat

A multi-step reasoning engine for data analysis with verifiable, auditable logic.

## Overview

Constat enables LLM-powered data analysis with two key principles:

1. **Verifiable, Auditable Logic**: Every conclusion can be traced back to its source data and reasoning steps
2. **Universal Data Integration**: Connect to SQL databases, NoSQL stores, and cloud data services through a unified interface

### Design Tradeoffs

Constat decomposes problems into discrete, verifiable steps rather than generating a single monolithic response. This makes it slower than a traditional chatbot — each query goes through intent classification, planning, approval, multi-step execution, and synthesis. The payoff is that breaking problems into smaller tasks produces more accurate, auditable results. Each step can be inspected, retried, or corrected independently, and the full reasoning chain is preserved. Steps are organized as a DAG and parallelized where dependencies allow, recouping some of the overhead.

Scalability patterns like progressive discovery (on-demand schema loading, vector-indexed documents, context preloading) are built in but have not been stress-tested at scale.

## Quick Start

Try the included demo environment:

```bash
# Setup demo data (SQLite databases, CSV, JSON files)
python demo/setup_demo.py

# Set your API key
export ANTHROPIC_API_KEY=your_key_here

# Start both the API server and web UI
./scripts/dev.sh demo/config.yaml        # macOS / Linux
scripts\dev.bat demo\config.yaml          # Windows
```

Open http://localhost:5173 and try:
- "Top 5 customers by total order value"
- "Which pages have the highest bounce rate?"
- "Average performance rating by department"

Or use the terminal REPL instead:

```bash
constat repl -c demo/config.yaml
```

## Core Concepts

### Agentic Domains

Domains are the primary organizational unit. Everything — data sources, glossary terms, skills, agents, rules — is scoped to a domain. Domains form a strict DAG and are organized in three tiers:

| Tier | Location | Editable | Purpose |
|------|----------|----------|---------|
| **system** | `config.yaml` domains | No | Curated by admin, read-only |
| **shared** | `.constat/shared/domains/` | Owner only | Promoted from user, visible to all |
| **user** | `.constat/{user_id}/domains/` | Yes | Personal sandbox, persists across sessions |

Content flows upward: user → shared → system (read-only). User domains are persistent staging areas — experiments, draft skills, and what-if rules survive across sessions until promoted or deleted.

**Domain resources:** databases, APIs, documents, glossary terms, skills, agents, rules, permissions, golden questions, system prompts, NER stop lists.

**Move resources** between domains via drag-and-drop in the domain panel or the `/domains/move-*` endpoints. **Promote** user domains to shared via the UI or `POST /domains/{filename}/promote`.

### Recursive Structures

A session is a conversation with memory. Each user question triggers:

1. **Intent classification** — determines execution mode
2. **Planning** — generates a multi-step DAG with user approval
3. **Execution** — each step produces code, output, and narrative
4. **Synthesis** — combines step results into a coherent answer

Follow-up questions extend the session, building on previous results. Steps are organized as a DAG and parallelized where dependencies allow. Clarification questions with interactive widgets (choice, curation, ranking, table, mapping, tree, annotation) fire before planning when queries are ambiguous.

```python
from constat import Session

session = Session(config)
result = session.solve("What are the top 10 customers by revenue this quarter?")
```

### Execution Modes

| Mode | Trigger | Behavior |
|------|---------|----------|
| **Exploratory** | Default | Multi-step plans, code generation, charts, tables |
| **Auditable** | `/reason` | Formal reasoning chain with full derivation traces |
| **Knowledge** | `/reason` on knowledge queries | Retrieves from documents via semantic search |

### Facts and Provenance

Every piece of information is a `Fact` with:
- **Value**: The actual data
- **Source**: Where it came from (DATABASE, CONFIG, DERIVED, LLM_INFERENCE, LEARNED)
- **Confidence**: How certain we are (1.0 for database queries, lower for inferences)
- **Because**: The facts this was derived from

**Derivation traces** show exactly how conclusions were reached:

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

**Lazy resolution** ensures efficient execution — facts are resolved only when needed:
1. Check cache
2. Check config
3. Apply rules
4. Query database
5. Fall back to LLM knowledge
6. Create sub-plan if complex

### LLM Primitives

Generated code has access to LLM wrapper functions for in-step operations:

```python
# Map values to an allowed set
mapped = llm_map(df['category'], ['Electronics', 'Clothing', 'Food'])

# Classify text into categories
labels = llm_classify(df['comment'], ['positive', 'negative', 'neutral'])

# Score text on a numeric scale
scores = llm_score(df['review'], min_val=0, max_val=1, instruction="sentiment score")

# Extract structured fields from text
facts = llm_extract(text, fields=["company", "revenue", "date"])

# Summarize text
summary = llm_summarize(df['description'].tolist())
```

These auto-detect the provider from environment variables and handle deduplication for efficiency.

## Features

### Reasoning Chain

At any point, use `/reason` to generate a formal reasoning chain for your results with full derivation traces. This is useful for compliance, financial reporting, and any scenario requiring provable conclusions.

The system automatically:
1. Analyzes the question to identify required facts
2. Determines how to derive each fact from your data sources
3. Executes queries and combines results
4. Returns the answer with full provenance

### Agents and Skills

**Agents** are specialist personas that customize the LLM's behavior for specific analysis contexts. Each agent provides a custom system prompt that shapes how the LLM approaches queries. The planner can delegate steps to agents for isolated, expertise-specific execution. Agents are stored per-user at `.constat/{user_id}/agents.yaml`.

**Skills** are domain-specific knowledge modules (SKILL.md files) that provide specialized context and guidance. They follow the standard skill/prompt pattern used by Anthropic (Claude Code), OpenAI, and other AI providers. Skills are portable programs — they can run under other LLMs, not just the one that created them. A completed reasoning chain can be converted to a skill via "Save as Skill" in the DAG panel.

Skills are discovered from: project `.constat/skills/`, global `~/.constat/skills/`, and config-specified paths. They support link following for lazy-loaded references, and YAML frontmatter for metadata and allowed tools.

### Glossary Generator

The glossary is a unified view of auto-generated entities (from NER extraction) married with curated business definitions. Every extracted entity is a self-describing glossary term. Features include:

- **Definitions** — business meaning for terms, with AI-assisted generation and refinement
- **Taxonomy** — parent/child hierarchy (e.g., "retail customer" is a kind of "customer")
- **Aliases** — alternate names for the same concept, with AI suggestions
- **Tags** — key-value metadata for categorization, with AI generators
- **Relationships** — SVO triples between terms (e.g., customer PLACES order) with UPPER_SNAKE_CASE verb normalization
- **Status workflow** — draft → reviewed → approved
- **Domain scoping** — terms are owned by domains

### Regression Testing

Golden questions let you define expected outcomes for a domain and verify them automatically. Questions are defined in domain YAML alongside the resources they test.

**Five assertion layers:**

| Layer | What | Cost |
|-------|------|------|
| Entity extraction | Expected entities appear in NER output | Free |
| Grounding | Entities resolve to expected sources | Free |
| Glossary | Terms have definitions, correct domain, parent hierarchy | Free |
| Relationships | Expected SVO triples exist | Free |
| End-to-end | LLM generates plan, executes, answer matches reference | LLM call |

The first four layers are pure database lookups. End-to-end is opt-in (`--e2e`).

```bash
constat test -c config.yaml                    # All domains
constat test -c config.yaml -d sales-analytics # Specific domain
constat test -c config.yaml --tags smoke       # Filter by tag
constat test -c config.yaml --e2e              # Include end-to-end (LLM)
```

The UI provides a Regression panel for golden question CRUD and test execution with SSE streaming for real-time progress.

### Learning System

Constat learns from corrections and errors to improve over time. Learnings are stored per-user and persist across sessions.

- **Explicit corrections:** `/correct "revenue" means gross sales minus returns`
- **Automatic learning:** When code generation fails and retries succeed, the error-to-fix pattern is captured automatically
- **Natural language detection:** Corrections in conversation are detected automatically (e.g., "That's wrong, active users means 30-day logins")
- **Compaction:** Similar learnings are automatically promoted to rules when patterns emerge

### Session Replay

Any exploratory session can be replayed — stored scratchpad code is re-executed without LLM codegen. This is useful for demos, testing with updated data, or resuming work after a break. Each query's steps are tracked by `objective_index`, so individual objectives can be replayed independently.

In Jupyter notebooks, session replay is automatic. When you restart the kernel and Run All, each `%%constat` cell replays only its own query's stored code rather than re-generating via LLM.

### Interactive Visualizations

Constat generates interactive visualizations saved as HTML files:

| Type | Library | Example |
|------|---------|---------|
| Interactive maps | Folium | Geographic data, markers, choropleth maps |
| Interactive charts | Plotly | Bar, line, scatter, pie, treemap, etc. |
| Statistical charts | Altair | Declarative statistical visualizations |
| Static plots | Matplotlib/Seaborn | Traditional Python plotting |

Request a "dashboard" to generate multi-panel visualizations with adaptive layouts (time series 1x2, categories 2x2, KPI-focused 3x2).

### Discovery Tools

All supported LLM providers use tool-based discovery:
- Minimal system prompt (~500 tokens)
- LLM uses tools to discover relevant schema, APIs, and documents on-demand
- On-demand loading reduces token usage by 80-95%

| Category | Tools |
|----------|-------|
| **Schema** | `list_databases`, `list_tables`, `get_table_schema`, `search_tables`, `get_table_relationships`, `get_sample_values` |
| **API** | `list_apis`, `list_api_operations`, `get_operation_details`, `search_operations` |
| **Documents** | `list_documents`, `get_document`, `search_documents`, `get_document_section` |
| **Facts** | `resolve_fact`, `add_fact`, `extract_facts_from_text`, `list_known_facts` |

Documents are indexed into a persistent vector store using DuckDB VSS (lazy, on first access). The index persists at `~/.constat/vectors.duckdb` by default, eliminating re-indexing on restart.

## Data Integration

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
    +-------------+          +------------------+
    |   Web UI    |          |   Terminal REPL   |
    | (React SPA) |          | (Textual TUI)    |
    +------+------+          +--------+---------+
           |                          |
    +------v------+                   |
    |  REST API   |                   |
    | (FastAPI +  |                   |
    |  WebSocket) |                   |
    +------+------+                   |
           |                          |
           +------------+-------------+
                        |
              +---------v---------+
              | Session (mixin-   |
              |  based modules)   |
              +---------+---------+
                        |
                +-------v-------+
                | Intent        |
                | Classifier    |
                +-------+-------+
                        |
       +----------------+----------------+
       |                                 |
+------v------+                   +------v------+
| Exploratory |                   |  Reasoning  |
|    Mode     |                   |    Chain    |
+------+------+                   +------+------+
       |                                 |
+------v------+                   +------v------+
| Multi-Step  |                   |    Fact     |
| DAG Planner |                   |  Resolver   |
+------+------+                   +------+------+
       |                                 |
       +----------------+----------------+
                        |
          +-------------v-------------+
          |     Discovery Tools       |
          |  (tool-based, on-demand)  |
          +------+------+------+-----+
                 |             |
    +------------v--+    +----v-----------+
    | DuckDB Session|    | DuckDB Vector  |
    | Store (tables,|    | Store (embeds, |
    | views, meta)  |    |  FTS, BM25)   |
    +------+--------+    +----+-----------+
           |                   |
+-------+--+----+-------+-----+--+-------+
|       |       |       |        |       |
SQL   MongoDB Cassandra DynamoDB Files  APIs
                                (CSV/   (GraphQL/
                                 JSON/   OpenAPI)
                                 Parquet)
```

For detailed architecture, see [docs/architecture.md](docs/architecture.md).

## REST API

Constat provides a REST API via FastAPI with WebSocket support for real-time streaming.

```bash
constat serve -c config.yaml --port 8000
```

### Key Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check and server stats |
| `POST` | `/api/sessions` | Create a new session |
| `GET` | `/api/sessions` | List sessions |
| `GET` | `/api/sessions/{id}` | Get session details |
| `POST` | `/api/sessions/{id}/query` | Submit a query |
| `GET` | `/api/sessions/{id}/plan` | Get current execution plan |
| `POST` | `/api/sessions/{id}/plan/approve` | Approve an execution plan |
| `POST` | `/api/sessions/{id}/cancel` | Cancel execution |
| `WS` | `/api/sessions/{id}/ws` | WebSocket for real-time events |

### Additional Route Groups

| Prefix | Description |
|--------|-------------|
| `/api/sessions/{id}/tables` | Table CRUD, versions, starring |
| `/api/sessions/{id}/artifacts` | Artifact listing, versions, deletion |
| `/api/sessions/{id}/facts` | Fact CRUD, persistence |
| `/api/sessions/{id}/steps` | Step code listing (exploratory mode) |
| `/api/sessions/{id}/scratchpad` | Execution narrative (goal + narrative per step) |
| `/api/sessions/{id}/inference-codes` | Inference code (auditable mode) |
| `/api/sessions/{id}/glossary/...` | Glossary terms, relationships, taxonomy |
| `/api/sessions/{id}/sources` | Unified data sources (databases, APIs, documents) |
| `/api/sessions/{id}/proof-tree` | Proof DAG for auditable mode |
| `/api/sessions/{id}/proof-facts` | Proof fact persistence for session restore |
| `/api/schema` | Schema introspection |
| `/api/users` | User management, permissions |
| `/api/sessions/agents` | Agent management |
| `/api/skills` | Skill CRUD, activation |
| `/api/learnings` | Learnings, rules, exemplar generation |
| `/api/fine-tune/...` | Fine-tuning job lifecycle |
| `/api/testing/...` | Golden question CRUD and regression test execution |
| `/api/domains/...` | Domain CRUD, tree, move, promote |

Full OpenAPI documentation is available at `/docs` when the server is running.

## LLM Provider Support

Multiple LLM providers for flexibility and cost optimization:

```python
from constat.providers import (
    AnthropicProvider,
    OpenAIProvider,
    GeminiProvider,
    GrokProvider,
    MistralProvider,
    CodestralProvider,
    OllamaProvider,     # Local models
    TogetherProvider,    # Hosted inference
    GroqProvider,        # Fast inference
    TaskRouter,          # Task-type routing with automatic escalation
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
- **Automatic escalation**: Local model fails -> cloud model tries automatically
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
| Mistral AI | `mistral` | Yes |
| Codestral | `codestral` | Yes |
| Ollama (local) | `ollama` | Yes (llama3.2+) |
| Together AI | `together` | Yes |
| Groq | `groq` | Yes |

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
  provider: anthropic                    # anthropic | openai | gemini | grok | mistral | ollama

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

  # Admin token for auth bypass (CLI, scripts, CI)
  admin_token: ${CONSTAT_ADMIN_TOKEN}     # Bearer token that grants admin access

  # User permissions (or use $ref to separate file)
  permissions:
    $ref: ./permissions.yaml
```

### User Permissions

Access control uses **personas** (UI visibility + write/feedback permissions) and **resource-level RBAC** (data source access per UID).

#### Personas

Each user is assigned a persona that controls which UI sections they see and what they can modify. Personas are defined in `constat/server/personas.yaml`:

| Persona | Visibility | Writes | Feedback |
|---------|------------|--------|----------|
| `platform_admin` | All sections | All resources | flag_answers |
| `domain_builder` | All sections | All resources | flag_answers |
| `sme` | Results, learnings, facts, glossary, entities, query history | Glossary, facts, learnings, entities | flag_answers, auto_approve, suggest_entities |
| `domain_user` | Results, query history | None | flag_answers, suggest_glossary |
| `viewer` | Results only | None | None |

#### Resource-Level RBAC

Per-user resource access lists are configured by UID in `config.yaml`:

```yaml
permissions:
  users:
    8TgdzQHw7EbTHSJY9osIuCElbGF2:
      persona: platform_admin
      domains: []          # Empty = admin sees all
      databases: []
      documents: []
      apis: []

    xK9mPqR2wYnZaB4cD7eF1gH3iJ5:
      persona: domain_user
      domains:
        - sales-analytics.yaml
      databases:
        - sales
        - inventory
      documents: []
      apis: []

  default:
    persona: viewer
    domains: []
    databases: []
    documents: []
    apis: []
```

#### Permission Rules

- **Admins** (`platform_admin` persona): Full access to all resources, permission arrays ignored
- **Non-admin users**: Only see resources explicitly listed in their permission arrays
- **Domain access**: Grants access to all resources within that domain, intersected with domain-scoped `permissions.yaml` if present (least-privilege)
- **Unlisted users**: Get `default` permissions
- **No permissions configured**: Everything available (development mode)
- **Auth disabled** (`auth_disabled: true`): All permissions granted

#### How Permission Filtering Works

Permissions are enforced at the **metadata level** — the LLM never sees metadata for resources the user cannot access:

| Resource | Filtered In | Effect |
|----------|-------------|--------|
| Databases | `list_tables`, `search_tables`, schema summary | LLM doesn't see table schemas |
| APIs | `list_apis`, `list_api_operations`, system prompt | LLM doesn't see API endpoints |
| Documents | `list_documents`, `search_documents` | LLM can't search restricted docs |
| Domains | Domain activation | User can't load restricted domains |
| Skills | Skill selection | LLM can't use restricted skills |
| Agents | Agent assignment | Planner can't delegate to restricted agents |

#### Advanced Authorization

For more complex requirements (e.g., resolving data source rights from external identity providers), create an integration with Constat to map UIDs to data source access using your chosen third-party policy engine.

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

## Interfaces

Constat provides three interfaces, each suited to different workflows:

| Interface | Best For | Guide |
|-----------|----------|-------|
| **Terminal REPL** | Keyboard-driven iteration, full command set, scriptable | [docs/repl.md](docs/repl.md) |
| **Web UI** | Visual exploration, DAG visualization, domain management, team collaboration | [docs/web.md](docs/web.md) |
| **Jupyter Notebook** | Reproducible analysis, per-cell replay, DataFrame ecosystem, zero-code magics | [docs/jupyter.md](docs/jupyter.md) |

## Installation

```bash
pip install constat

# Optional dependencies for specific databases
pip install constat[mysql]        # MySQL
pip install constat[mongodb]      # MongoDB
pip install constat[dynamodb]     # AWS DynamoDB
pip install constat[cosmosdb]     # Azure Cosmos DB
pip install constat[firestore]    # Google Firestore
pip install constat[openai]       # OpenAI provider
pip install constat[extras]       # Polars, Excel, PDF generation
pip install constat[all]          # Everything
```

## License

Business Source License 1.1 (BSL 1.1)

Free for individuals and organizations with <100 employees and <$1M revenue.
Converts to Apache License 2.0 on 2029-01-21.
For commercial licensing, contact kennethstott@gmail.com.
