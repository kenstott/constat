# Constat

A multi-step reasoning engine for data analysis with verifiable, auditable logic.

## Overview

Constat enables LLM-powered data analysis with two key principles:

1. **Verifiable, Auditable Logic**: Every conclusion can be traced back to its source data and reasoning steps
2. **Universal Data Integration**: Connect to SQL databases, NoSQL stores, and cloud data services through a unified interface

## Execution Modes

Constat supports two execution modes to balance flexibility with auditability:

### Exploratory Mode

For data exploration, visualization, and iterative analysis. The LLM generates multi-step plans and executes code to answer questions.

```python
from constat import Session

session = Session(config, mode="exploratory")
result = session.solve("What are the top 10 customers by revenue this quarter?")
```

- Generates multi-step execution plans
- Each step produces code, output, and narrative
- Results include charts, tables, and insights
- Trace available but not formally verified

### Auditable Mode

For compliance, financial reporting, and any scenario requiring provable conclusions. Uses lazy fact resolution with full derivation traces.

The key difference from exploratory mode: the LLM generates the derivation logic automatically based on your question and the available schema. You don't write rules - the system figures out what data it needs and how to combine it.

```python
from constat import Session

session = Session(config, mode="auditable")

# The system automatically:
# 1. Analyzes the question to identify required facts
# 2. Determines how to derive each fact from your data sources
# 3. Executes queries and combines results
# 4. Returns the answer with full provenance

result = session.resolve("Is customer C001 a VIP?")
print(result.answer)       # "Yes, customer C001 is a VIP"
print(result.derivation)   # Full derivation trace
```

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

Native connectors for popular NoSQL stores:

```python
from constat.catalog.nosql import (
    MongoDBConnector,
    CassandraConnector,
    ElasticsearchConnector,
    DynamoDBConnector,
    CosmosDBConnector,
    FirestoreConnector,
)

# MongoDB
mongo = MongoDBConnector(
    uri="mongodb://localhost:27017",
    database="mydb",
)

# AWS DynamoDB
dynamo = DynamoDBConnector(region="us-east-1")

# Azure Cosmos DB
cosmos = CosmosDBConnector(
    endpoint="https://account.documents.azure.com:443/",
    key="...",
    database="mydb",
)

# Google Firestore
firestore = FirestoreConnector(project="my-gcp-project")
```

All connectors provide:
- Schema introspection (inferred for schema-less databases)
- Unified query interface
- Embedding text generation for vector search

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
                    +-------v-------+
                    | Schema Manager |
                    +-------+-------+
                            |
    +-------+-------+-------+-------+-------+
    |       |       |       |       |       |
   SQL   MongoDB Cassandra DynamoDB CosmosDB Firestore
```

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

  # Optional: Use different models for different tasks (cost optimization)
  tiers:
    planning: claude-sonnet-4-20250514   # Complex reasoning for plan generation
    codegen: claude-sonnet-4-20250514    # Accurate code generation
    simple: claude-3-5-haiku-20241022    # Fast, cheap for simple tasks

  # Optional: Provider-specific settings
  # base_url: https://api.anthropic.com  # Custom API endpoint
  # max_tokens: 4096                     # Max output tokens

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
```

### Environment Variable Substitution

Use `${VAR_NAME}` syntax to reference environment variables:

```yaml
databases:
  - name: production
    uri: postgresql://${DB_HOST}:${DB_PORT}/${DB_NAME}
    username: ${DB_USER}
    password: ${DB_PASSWORD}

llm:
  api_key: ${ANTHROPIC_API_KEY}
```

Environment variables are substituted at config load time. Missing variables raise an error.

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

### NoSQL Database Configuration

NoSQL databases are configured programmatically (not in YAML) because they have diverse connection options:

```python
from constat.catalog.nosql import (
    MongoDBConnector,
    DynamoDBConnector,
    CosmosDBConnector,
    FirestoreConnector,
)

# MongoDB
mongo = MongoDBConnector(
    uri="mongodb://${MONGO_HOST}:27017",
    database="mydb",
    name="mongo_main",
)

# MongoDB with authentication
mongo = MongoDBConnector(
    uri="mongodb://username:password@cluster.mongodb.net/mydb?retryWrites=true",
    database="mydb",
)

# AWS DynamoDB (uses AWS credentials from environment/IAM)
dynamo = DynamoDBConnector(
    region="us-east-1",
    # Optional: explicit credentials
    aws_access_key_id="${AWS_ACCESS_KEY_ID}",
    aws_secret_access_key="${AWS_SECRET_ACCESS_KEY}",
)

# Azure Cosmos DB
cosmos = CosmosDBConnector(
    endpoint="https://myaccount.documents.azure.com:443/",
    key="${COSMOS_KEY}",
    database="mydb",
)

# Google Firestore (uses ADC or service account)
firestore = FirestoreConnector(
    project="my-gcp-project",
    credentials_path="/path/to/service-account.json",  # Optional
)

# Register with schema manager
schema_manager.register_nosql(mongo)
schema_manager.register_nosql(dynamo)
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
)

# Anthropic (default)
provider = AnthropicProvider(model="claude-sonnet-4-20250514")

# OpenAI
provider = OpenAIProvider(model="gpt-4o")

# Local Ollama
provider = OllamaProvider(model="llama3.2", host="http://localhost:11434")
```

## Key Concepts

### Facts and Provenance

Every piece of information is a `Fact` with:
- **Value**: The actual data
- **Source**: Where it came from (DATABASE, CONFIG, RULE, LLM_KNOWLEDGE)
- **Confidence**: How certain we are (1.0 for database queries, lower for inferences)
- **Because**: The facts this was derived from

### Derivation Traces

In auditable mode, every conclusion includes a full derivation trace showing:
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
