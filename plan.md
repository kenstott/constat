# Multi-Step AI Reasoning Engine

## Overview

A hybrid Python/Prolog system that enables LLMs to solve complex problems through disciplined logical reasoning. The LLM generates Prolog facts and rules which are executed by an embedded Prolog engine, providing traceable explanations for all conclusions. Python handles data operations (pandas, DuckDB) while Prolog handles reasoning.

**Primary use case: Critical decisioning** - where conclusions must be explainable, auditable, and defensible.

## Critical Decisioning Requirements

This system is designed for high-stakes decisions where:

| Requirement | How This System Addresses It |
|-------------|------------------------------|
| **Explainability** | Every conclusion has a Prolog derivation trace showing exactly which facts and rules led to it |
| **Auditability** | Full session logs: plan, artifacts, fact resolutions, sources, confidence scores, user interactions |
| **Defensibility** | Can answer "why did the system recommend X?" with formal proof, not narrative |
| **Data lineage** | Every fact tagged with source (which DB, which table, which query - or LLM with confidence) |
| **Reproducibility** | Given same inputs + KB state, produces same conclusions |
| **Human oversight** | Interrupt at any point, inspect state, provide corrections, approve before action |
| **Confidence awareness** | Distinguishes "certain from DB" vs "likely from LLM knowledge" - surfaces uncertainty |

**Example critical decisioning scenarios:**
- Loan approval: "Why was this application flagged for manual review?"
- Compliance: "Prove this transaction meets regulatory requirements"
- Healthcare: "What factors led to this risk classification?"
- Fraud detection: "Show the evidence chain for this alert"
- Resource allocation: "Justify why Project A was prioritized over Project B"

**Audit log structure:**
```json
{
  "session_id": "sess_20240115_143022",
  "timestamp": "2024-01-15T14:30:22Z",
  "problem": "Identify high-risk loan applications from Q4 batch",
  "plan": [...],
  "execution": [
    {
      "step": 1,
      "type": "python",
      "code": "...",
      "artifacts": ["applications_table"],
      "timing_ms": 2340
    },
    {
      "step": 2,
      "type": "prolog",
      "rules": ["high_risk(App) :- ..."],
      "facts_resolved": [
        {"fact": "credit_score(app_123, X)", "source": "loans_db", "confidence": 1.0},
        {"fact": "industry_risk(tech, X)", "source": "llm_knowledge", "confidence": 0.7}
      ],
      "derivation_trace": "high_risk(app_123) :- credit_score(app_123, 580), threshold(credit, 620), 580 < 620."
    }
  ],
  "conclusions": [
    {"result": "high_risk(app_123)", "confidence": 0.85, "reasoning": "..."}
  ],
  "user_interactions": [
    {"timestamp": "...", "action": "interrupt", "feedback": "Exclude startups less than 2 years old"}
  ]
}
```

**Key differentiator from black-box AI:** When a regulator, auditor, or stakeholder asks "why?", the system produces a formal proof - not a narrative explanation generated after the fact.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User Problem                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Planning Phase                           │
│  LLM analyzes problem → generates English step-by-step plan │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Execution Loop                            │
│  For each step, LLM chooses:                                │
│                                                             │
│  ┌─────────────────────┐    ┌─────────────────────┐        │
│  │   PROLOG (reasoning)│    │  PYTHON (data ops)  │        │
│  │  - Generate facts   │    │  - Query databases  │        │
│  │  - Generate rules   │    │  - Transform data   │        │
│  │  - Query for proof  │    │  - Statistical ops  │        │
│  │  - Get derivation   │    │  - Load to DuckDB   │        │
│  │    trace            │    │                     │        │
│  └─────────────────────┘    └─────────────────────┘        │
│            │                          │                     │
│            └──────────┬───────────────┘                     │
│                       ▼                                     │
│              Results → Scratchpad + DuckDB                  │
│              Prolog trace → Explanation log                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Final Output                             │
│  - Summary from scratchpad                                  │
│  - Reasoning trace: "X because Y and Z"                     │
│  - Data tables available for inspection                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 Interactive Refinement                      │
│  User can:                                                  │
│    - Refine: "Actually, filter to just Q4 data"            │
│    - Extend: "Now compare this to last year"               │
│    - Challenge: "Why did you conclude X?"                  │
│    - Loop back to planning with full context preserved      │
└─────────────────────────────────────────────────────────────┘
```

## Configuration File

The system is configured via a YAML file that specifies database connections, LLM settings, and domain context.

### Config File Structure

```yaml
# config.yaml

# LLM provider configuration
llm:
  provider: anthropic                    # anthropic | openai | ollama
  model: claude-sonnet-4-20250514
  api_key: ${ANTHROPIC_API_KEY}          # env var substitution

  # Optional: model tiering for cost optimization (Phase 2)
  tiers:
    planning: claude-sonnet-4-20250514   # Needs reasoning
    codegen: claude-sonnet-4-20250514    # Needs accuracy
    simple: claude-3-5-haiku-20241022    # Fact routing, SQL gen

# Database connections (SQLAlchemy URIs)
databases:
  - name: sales_db
    uri: postgresql://${DB_USER}:${DB_PASS}@localhost:5432/sales

  - name: inventory_db
    uri: mysql+pymysql://${DB_USER}:${DB_PASS}@localhost:3306/inventory

  - name: analytics_db
    uri: sqlite:///./data/analytics.db

# Domain context for the LLM (included in system prompt)
system_prompt: |
  You are analyzing data for a retail company.

  Key concepts:
  - customer_tier: Found in sales_db.customers.tier_level (gold/silver/bronze)
  - revenue: Found in sales_db.transactions, aggregate by SUM(amount)
  - region: Geographic sales regions (north/south/east/west)

  Common relationships:
  - Customers linked to transactions via customer_id
  - Targets set per-region per-quarter in sales_db.targets

  Business rules:
  - "Underperforming" means < 80% of target
  - "VIP customer" means tier_level = 'gold' OR lifetime_value > 100000

# Execution settings
execution:
  timeout_seconds: 60              # per-step timeout
  max_retries: 10                  # per-step retry limit (0 = unlimited)

  # Python import allowlist for generated code
  allowed_imports:
    - pandas
    - numpy
    - scipy
    - sklearn
    - requests
```

### Environment Variable Substitution

Config values can reference environment variables using `${VAR_NAME}` syntax:

```yaml
uri: postgresql://${DB_USER}:${DB_PASS}@localhost:5432/sales
api_key: ${ANTHROPIC_API_KEY}
```

Variables are resolved at config load time. Missing variables raise an error.

### SQLAlchemy URI Format

Database URIs follow SQLAlchemy's URL format:

| Database | URI Format |
|----------|------------|
| PostgreSQL | `postgresql://user:pass@host:5432/dbname` |
| MySQL | `mysql+pymysql://user:pass@host:3306/dbname` |
| SQLite | `sqlite:///./path/to/file.db` |
| SQL Server | `mssql+pyodbc://user:pass@host/dbname?driver=...` |

### Config Loading (`config.py`)

```python
from pydantic import BaseSettings, Field
from typing import Optional

class DatabaseConfig(BaseSettings):
    name: str
    uri: str  # SQLAlchemy URI with env var substitution

class LLMConfig(BaseSettings):
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    api_key: Optional[str] = None
    tiers: Optional[dict[str, str]] = None

class ExecutionConfig(BaseSettings):
    timeout_seconds: int = 60
    max_retries: int = 10
    allowed_imports: list[str] = Field(default_factory=list)

class Config(BaseSettings):
    llm: LLMConfig
    databases: list[DatabaseConfig]
    system_prompt: str = ""
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load config from YAML file with env var substitution."""
        ...
```

---

## Core Components

### 1. Database Asset Manager (`assets.py`)

Manages database connections and provides schema information to the LLM via a **hybrid approach**: a compact overview in the system prompt plus on-demand detail retrieval.

**Connection management:**
- Reads config file (YAML) containing list of SQLAlchemy database URIs
- **Establishes live connections** to each database
- Uses SQLAlchemy + appropriate drivers (psycopg2, mysql-connector, etc.)
- Connections kept alive for generated code to query during execution

**Metadata extraction** via database introspection:
- `inspector.get_table_names()`
- `inspector.get_columns(table)`
- `inspector.get_pk_constraint(table)`
- `inspector.get_foreign_keys(table)`
- Approximate row counts via `SELECT COUNT(*)`

**Hybrid schema context strategy:**

Rather than dumping all schema details into every prompt (token-expensive) or relying purely on vector search (may miss relevant tables), use a three-tier approach:

| Tier | What | When | Tokens |
|------|------|------|--------|
| **Overview** | Table of contents in system prompt | Always included | ~200-500 |
| **On-demand** | Full schema via `get_table_schema()` tool | LLM calls when needed | ~50-100 per table |
| **Discovery** | Vector search via `find_relevant_tables()` | Complex/ambiguous queries | ~100-200 |

**Tier 1: Overview in system prompt** (~200-500 tokens)

```
Available databases:
  sales_db: customers, transactions, targets, products (4 tables, ~500K rows)
  inventory_db: warehouses, stock_levels, shipments (3 tables, ~50K rows)

Key relationships:
  customers.id → transactions.customer_id
  products.id → transactions.product_id
  warehouses.id → stock_levels.warehouse_id
```

This gives the LLM a "mental map" of what exists without token bloat.

**Tier 2: On-demand schema lookup** (LLM tool)

```python
def get_table_schema(self, table: str) -> dict:
    """
    Returns full schema for one table.
    Called by LLM as a tool when it needs column details.

    Example:
      get_table_schema("sales_db.customers")
      → {
          columns: [
            {name: "id", type: "int", pk: true},
            {name: "name", type: "str"},
            {name: "tier", type: "str", values: ["gold", "silver", "bronze"]},
            {name: "created_at", type: "datetime"}
          ],
          row_count: 52341,
          foreign_keys: [],
          referenced_by: ["transactions.customer_id"]
        }
    """
```

**Tier 3: Vector search for discovery** (optional, for large schemas)

```python
def find_relevant_tables(self, query: str, top_k: int = 5) -> list[TableMatch]:
    """
    Semantic search over table/column descriptions.
    Useful when LLM isn't sure where data lives.

    Example:
      find_relevant_tables("customer purchase history and refund rate")
      → [
          {table: "transactions", relevance: 0.89, reason: "purchase records"},
          {table: "refunds", relevance: 0.85, reason: "refund data"},
          {table: "customers", relevance: 0.72, reason: "customer info"}
        ]
    """
```

For vector search, embed a document per table:
```
Table: sales_db.transactions
Description: Customer purchase records
Columns: id, customer_id (FK→customers), product_id, amount, timestamp, status
Keywords: revenue, sales, purchases, orders, buying history
```

Use lightweight embeddings (e.g., `sentence-transformers/all-MiniLM-L6-v2` - fast, runs locally).

**When to use each tier:**

| Schema Size | Strategy |
|-------------|----------|
| <20 tables | Overview only (may skip tools entirely) |
| 20-100 tables | Overview + on-demand lookup |
| 100+ tables | All three tiers with vector search |

**Implementation:**

```python
class SchemaManager:
    def __init__(self, config: Config):
        self.connections: dict[str, Engine] = {}
        self.metadata_cache: dict[str, TableMetadata] = {}
        self.vector_index: Optional[VectorIndex] = None  # Lazy-built

    def connect_all(self) -> None:
        """Establish connections to all configured databases."""
        for db in config.databases:
            engine = create_engine(db.uri)
            self.connections[db.name] = engine
            self._introspect(db.name, engine)

    def get_overview(self) -> str:
        """Token-optimized summary for system prompt."""
        lines = ["Available databases:"]
        for db_name, tables in self._group_by_db().items():
            table_list = ", ".join(t.name for t in tables)
            total_rows = sum(t.row_count for t in tables)
            lines.append(f"  {db_name}: {table_list} ({len(tables)} tables, ~{total_rows:,} rows)")

        lines.append("\nKey relationships:")
        for fk in self._get_foreign_keys():
            lines.append(f"  {fk.from_table}.{fk.from_col} → {fk.to_table}.{fk.to_col}")

        return "\n".join(lines)

    def get_table_schema(self, table: str) -> dict:
        """Full schema for one table (LLM tool)."""
        return self.metadata_cache[table].to_dict()

    def find_relevant_tables(self, query: str, top_k: int = 5) -> list[dict]:
        """Vector search over table descriptions (LLM tool)."""
        if self.vector_index is None:
            self._build_vector_index()
        return self.vector_index.search(query, top_k)
```

**Token budget comparison:**

| Scenario | Static Approach | Hybrid Approach |
|----------|-----------------|-----------------|
| 10 tables, use 2 | ~800 tokens | ~300 tokens |
| 50 tables, use 3 | ~4000 tokens | ~450 tokens |
| 200 tables, use 5 | ~16000 tokens | ~750 tokens |

The hybrid approach scales much better as schema size grows.

### 2. LLM Provider Interface (`providers/`)
- Abstract base class for LLM providers
- Implementations for: OpenAI, Anthropic, Ollama (local)
- Configuration via environment variables or config file
- Streaming support for long generations

### 3. Planner (`planner.py`)
- Takes user problem as input
- **First output is always an English-language plan** displayed to the user:
  ```
  Plan:
  1. Load the sales data from sales_db.transactions table
  2. Calculate monthly revenue totals grouped by region
  3. Identify the top 3 performing regions
  4. Compare year-over-year growth for each top region
  5. Generate summary with recommendations
  ```
- Plan is shown before any code generation begins
- Each step: plain English goal, expected inputs, expected outputs
- Returns structured Plan object for execution phase

### 4. Step Generator (`stepgen.py`)
- For each plan step, LLM decides: **Prolog** (reasoning) or **Python** (data ops)
- Outputs structured response with `type: "prolog"` or `type: "python"`

**Prolog generation** (for reasoning steps):
```prolog
% Facts (from data or prior steps)
revenue(north, 2024, 2300000).
revenue(south, 2024, 1800000).
target(region, 2000000).

% Rules
above_target(Region) :-
    revenue(Region, 2024, Rev),
    target(region, Target),
    Rev > Target.

% Query
?- above_target(X).
```

**Python generation** (for data operations + SQL integration):
- **Primary role**: Source data from external databases via SQLAlchemy
- Query databases, transform data, statistical operations
- Load results into DuckDB for intermediate storage
- **Feed Prolog**: Convert query results to facts via `prolog.assert_facts()`
- Access: `pandas`, `numpy`, `scipy`, `sklearn`, `requests`, `db`, `prolog`, `llm`

**Three data sources for Prolog facts:**

1. **External databases** (SQL):
```python
df = pd.read_sql("SELECT region, revenue FROM sales", db_conn)
for _, row in df.iterrows():
    prolog.assertz(f"revenue({row['region']}, {row['revenue']})")
```

2. **LLM world knowledge** (background facts):
```python
# Ask LLM for domain knowledge not in databases
response = llm.query("""
    List European countries and their capitals as Prolog facts.
    Format: capital(country, city).
""")
# Returns: capital(france, paris). capital(germany, berlin). ...
prolog.consult_string(response)
```

3. **LLM domain expertise** (rules/heuristics):
```python
# Ask LLM for domain rules
response = llm.query("""
    What rules determine if a sales region is underperforming?
    Express as Prolog rules using: revenue/2, target/2, growth_rate/2
""")
# Returns: underperforming(R) :- revenue(R,X), target(R,T), X < T * 0.8.
prolog.consult_string(response)
```

This makes the LLM a **knowledge source** (world facts, domain expertise) not just a code generator.

**Fact provenance and confidence scoring:**

**Option A: ProbLog (recommended)** - Probabilistic logic programming with native confidence:
```prolog
% Confidence is built into the syntax
1.0 :: revenue(north, 2300000).          % Database fact - certain
0.7 :: underperforming_threshold(0.8).   % LLM heuristic
0.6 :: capital(france, paris).           % LLM world knowledge

% Rules work normally
above_target(Region) :-
    revenue(Region, Rev),
    target(Region, T),
    Rev > T.

% Query returns probability
?- above_target(north).
% Result: 0.85 (computed from input confidences)
```

ProbLog advantages:
- Confidence propagation is automatic (probabilistic inference)
- Native support for uncertainty, not bolted on
- Can ask "what's the probability of X?"
- Python library: `pip install problog`
- No SWI-Prolog system dependency

**Option B: Standard Prolog with meta-predicates** (manual tracking):
```prolog
fact(revenue(north, 2300000), source(database), 1.0).
fact(capital(france, paris), source(llm), 0.6).

% Must manually propagate confidence in rules
```

**Recommendation**: Use **ProbLog** - it's designed for exactly this use case and eliminates manual confidence bookkeeping.

Reasoning engine can:
- Query with probability thresholds: "conclusions with P ≥ 0.8"
- Explain confidence: "This conclusion is 0.72 because fact X was 0.8 and fact Y was 0.9"
- Flag low-confidence conclusions automatically
- Allow user to verify/promote LLM facts

**On-demand fact resolution (Prolog → Python → LLM/DB):**

When Prolog needs a fact it doesn't have, it calls back to Python:

```prolog
% Foreign predicate - calls Python when evaluated
:- extern(resolve_fact/2).

% Rule that may trigger fact resolution
customer_tier(Customer, Tier) :-
    resolve_fact(customer_tier(Customer, Tier), Confidence),
    Confidence > 0.5.
```

Python resolver decides how to source the fact:
```python
def resolve_fact(goal):
    """Called by Prolog when a fact is missing."""

    # 1. Check if we can get it from a database
    source_plan = llm.query(f"""
        I need to resolve: {goal}
        Available databases: {db_schemas}
        Can this be answered from a database? If so, write the SQL.
        If not, can you provide this from your knowledge?
    """)

    if source_plan.type == "sql":
        result = db.execute(source_plan.sql)
        return Fact(goal, value=result, confidence=1.0, source="database")

    elif source_plan.type == "llm_knowledge":
        return Fact(goal, value=source_plan.value,
                   confidence=0.6, source="llm")

    else:
        return Fact(goal, value=None, confidence=0.0, source="unknown")
```

**Flow:**
```
Prolog: "I need customer_tier(acme, X)"
   │
   ▼
Python resolver:
   │
   ├─→ LLM: "Can I get this from sales_db?"
   │      → "Yes: SELECT tier FROM customers WHERE name='acme'"
   │
   ├─→ Execute SQL → tier = 'gold'
   │
   └─→ Return to Prolog: 1.0 :: customer_tier(acme, gold).
```

Benefits:
- No need to pre-load all possible facts
- Facts fetched just-in-time as reasoning requires them
- LLM acts as "query planner" - figures out WHERE to get data
- Confidence assigned based on source
- Caches resolved facts for reuse

**Error handling and validation:**

Most SQL errors surface at runtime and trigger regeneration:
```
SQL: SELECT tier FROM customers WHERE name='acme'
Error: column "tier" does not exist
→ Retry with error context → SELECT customer_tier FROM customers...
```

Data validation layer (optional sanity checks):
```python
def validate_fact(goal, value):
    # Type checking
    if goal.predicate == "revenue" and not isinstance(value, (int, float)):
        return False, "revenue must be numeric"

    # Range checking
    if goal.predicate == "percentage" and not (0 <= value <= 100):
        return False, "percentage out of range"

    # Null handling
    if value is None:
        return False, "no data found"

    return True, None
```

**Domain-specific system prompt:**

The setup file includes domain hints for fact resolution:
```yaml
# config.yaml
databases:
  - name: sales_db
    uri: postgresql://localhost:5432/sales

system_prompt: |
  This system analyzes sales performance for a retail company.

  Key concepts:
  - customer_tier: Found in sales_db.customers.tier_level (gold/silver/bronze)
  - revenue: Found in sales_db.transactions, aggregate by SUM(amount)
  - region: Geographic sales regions (north/south/east/west)

  Common relationships:
  - Customers are linked to transactions via customer_id
  - Targets are set per-region per-quarter in sales_db.targets

  Heuristics:
  - "Underperforming" typically means < 80% of target
  - "VIP customer" typically means tier_level = 'gold' OR lifetime_value > 100000
```

This gives the LLM domain context for:
- Where to find specific facts
- How tables relate
- Business-specific terminology and thresholds
- Common resolution patterns

**Recursive fact resolution (mini-plans):**

Some facts aren't simple lookups - they require multi-step derivation:

```
Prolog: "I need customer_lifetime_value(acme, X)"
   │
   ▼
Resolver: "This is complex - needs a mini-plan"
   │
   ▼
┌─────────────────────────────────────────────┐
│  Mini-plan for customer_lifetime_value:     │
│  1. Query all transactions for customer     │
│  2. Calculate sum, adjusting for refunds    │
│  3. Apply time-decay weighting              │
│  4. Return as fact with confidence 1.0      │
└─────────────────────────────────────────────┘
   │
   ▼
Execute mini-plan (Python steps)
   │
   ▼
Return: 1.0 :: customer_lifetime_value(acme, 84750).
```

The resolver classifies fact requests:

| Complexity | Example | Resolution |
|------------|---------|------------|
| **Simple lookup** | `customer_name(123, X)` | Single SQL query |
| **Aggregation** | `total_revenue(region, X)` | SQL with GROUP BY |
| **Derived fact** | `lifetime_value(C, X)` | Mini-plan (multiple steps) |
| **World knowledge** | `capital(france, X)` | LLM knowledge |

Mini-plans are:
- Generated by LLM like top-level plans
- Executed in same Python/Prolog environment
- Results cached as facts for reuse
- Shown in derivation trace: "lifetime_value derived via mini-plan [1,2,3,4]"

This is **recursive reasoning** - the system can spawn sub-problems as needed, bottoming out at simple DB lookups or LLM knowledge.

**Validation loop** (both types):
- Prolog: Parse with PySwip, retry on syntax error until valid
- Python: `ast.parse()`, retry until valid
- No attempt limit - must produce valid code

### 5. Prolog Engine (`prolog_engine.py`)
- Embeds SWI-Prolog via **PySwip**
- Maintains persistent knowledge base across steps
- Key operations:
  - `assert_facts(facts)` - Add facts to KB
  - `assert_rules(rules)` - Add rules to KB
  - `query(goal)` - Execute query, return bindings
  - `query_with_trace(goal)` - Execute with derivation trace
  - `retract(clause)` - Remove from KB
  - `list_kb()` - Dump current knowledge base
- **Derivation traces** captured for explanation:
  ```
  above_target(north) :-
    revenue(north, 2024, 2300000),
    target(region, 2000000),
    2300000 > 2000000.
  ```
- Traces written to explanation log for user inspection

### 6. Python Executor (`executor.py`)
- Runs generated Python code in isolated subprocess
- Timeout handling (configurable, default 60s)
- Captures stdout, stderr, exceptions
- Resource limits (memory, CPU time)
- **On failure**: Returns full traceback to step generator for retry loop
- Execution loops until code runs successfully (no attempt limit)

### 7. Scratchpad (`scratchpad.py`)
- Markdown file that persists across steps
- Sections: `## Context`, `## Step N Results`, `## Notes`
- LLM can read previous sections, append new content
- Provides context continuity between steps

### 8. Data Store (`datastore.py`)
- DuckDB in-memory database
- Helper functions: `create_table()`, `query()`, `list_tables()`
- Each step can create named temp tables
- Tables persist across steps within a session
- Schema introspection for LLM context
- **Bridge to Prolog**: `to_prolog_facts(table)` exports rows as Prolog facts

### 9. Session Manager (`session.py`)
- Orchestrates the full workflow
- Maintains state: plan, current step, scratchpad, db, Prolog KB
- Routes steps to Prolog engine or Python executor
- Handles retries on execution failures
- Generates final summary with reasoning traces

### 10. Session History & Artifact Store (`history.py`)

**Purpose**: Persist complete session state for review, debugging, and resumption.

**Storage structure:**
```
.constat/
├── sessions/
│   ├── 2024-01-15_143022_abc123/
│   │   ├── session.json       # Metadata, config, timestamps
│   │   ├── queries.jsonl      # All queries in order
│   │   ├── artifacts/
│   │   │   ├── 001_code.py    # Generated code
│   │   │   ├── 001_output.txt # Execution output
│   │   │   ├── 002_code.py
│   │   │   └── ...
│   │   └── state.json         # Final state (for resumption)
│   └── ...
└── config.yaml
```

**Session metadata (`session.json`):**
```json
{
  "session_id": "2024-01-15_143022_abc123",
  "created_at": "2024-01-15T14:30:22Z",
  "config_hash": "sha256:...",
  "databases": ["chinook", "northwind"],
  "status": "completed",  // running, completed, failed, interrupted
  "total_queries": 5,
  "total_duration_ms": 45230
}
```

**Query record (`queries.jsonl`):**
```json
{"query_id": 1, "timestamp": "...", "question": "Top 5 genres by revenue?", "success": true, "attempts": 1, "duration_ms": 3200}
{"query_id": 2, "timestamp": "...", "question": "Compare to last year", "success": true, "attempts": 2, "duration_ms": 5100}
```

**Artifact tracking:**
Each query produces artifacts:
- Generated code (before and after retries)
- Execution output (stdout/stderr)
- Error tracebacks (if any)
- Tool calls made (get_table_schema, find_relevant_tables)

**Key operations:**

```python
class SessionHistory:
    def __init__(self, storage_dir: Path = ".constat/sessions"):
        ...

    def create_session(self, config: Config) -> str:
        """Create new session, return session_id."""

    def record_query(self, session_id: str, question: str, result: QueryResult):
        """Record a completed query with all artifacts."""

    def list_sessions(self, limit: int = 20) -> list[SessionSummary]:
        """List recent sessions with summary info."""

    def get_session(self, session_id: str) -> SessionDetail:
        """Get full session detail including all queries."""

    def get_artifacts(self, session_id: str, query_id: int) -> list[Artifact]:
        """Get artifacts for a specific query."""

    def resume_session(self, session_id: str) -> Session:
        """Load session state for resumption."""
```

**CLI integration:**
```bash
# List recent sessions
constat history

# View session detail
constat history abc123

# View specific query artifacts
constat history abc123 --query 2

# Resume interrupted session
constat resume abc123
```

**Use cases:**
- **Debugging**: Review what code was generated, what errors occurred
- **Audit**: Full trace of questions asked and answers provided
- **Resumption**: Continue an interrupted multi-step analysis
- **Comparison**: Compare runs with different prompts or configs

### 11. Live Feedback / Monitoring (`feedback.py`)

**Critical for user trust** - a 10-minute silent run feels broken. Every action must be visible.

**Real-time output stream:**
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 PLAN (5 steps)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Load customer data from sales_db
2. Calculate lifetime value per customer
3. Define VIP classification rules
4. Query for VIP customers below target
5. Generate recommendations

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
▶ Step 1/5: Load customer data from sales_db
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [generating python...]
  [executing...]
  ✓ Loaded 15,432 rows → customers table
  ⏱ 2.3s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
▶ Step 2/5: Calculate lifetime value per customer
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [generating python...]
  [executing...]
  ✗ Error: column 'total_amount' not found
  [retrying with error context...]
  [generating python...]
  [executing...]
  ✓ Calculated LTV for 15,432 customers
  ⏱ 4.1s (1 retry)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
▶ Step 3/5: Define VIP classification rules
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [generating prolog...]
  [loading rules into knowledge base...]
  📥 Resolving fact: revenue_threshold(vip, X)
     → LLM knowledge: 100000 (conf: 0.7)
  📥 Resolving fact: industry_standard(vip_ratio, X)
     → LLM knowledge: 0.05 (conf: 0.6)
  ✓ 3 rules added to KB
  ⏱ 3.2s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
▶ Step 4/5: Query for VIP customers below target
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [generating prolog query...]
  [executing query...]
  📥 Resolving fact: target(acme_corp, X)
     → SQL: SELECT target FROM targets WHERE customer='acme_corp'
     → Result: 500000 (conf: 1.0)
  📥 Resolving fact: target(globex, X)
     → SQL: SELECT target FROM targets WHERE customer='globex'
     → Result: 750000 (conf: 1.0)
  ✓ Query complete: 23 VIP customers below target
  ⏱ 5.8s (2 fact resolutions)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
▶ Step 5/5: Generate recommendations
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [generating synthesis...]
  ✓ Complete
  ⏱ 2.1s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ COMPLETE (5/5 steps, 17.5s total)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Every state transition is visible:**
- Plan displayed upfront (user knows what's coming)
- Current step highlighted (progress bar mental model)
- Sub-operations shown: generating, executing, resolving
- Fact resolutions shown with source and confidence
- Errors shown immediately with retry indication
- Timing per step (user can gauge pace)
- Running total at completion

**Verbose mode** (--verbose) additionally shows:
- Full generated code
- Complete Prolog rules
- SQL queries being executed
- Full derivation traces

Uses `rich` library for:
- Live updating (no scroll spam)
- Syntax highlighting for code
- Progress spinners during LLM calls
- Color-coded status (green/red/yellow)

**User interrupt and course correction:**

User can press `Ctrl+C` at any point to pause execution:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
▶ Step 3/5: Define VIP classification rules
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [generating prolog...]
  📥 Resolving fact: revenue_threshold(vip, X)
     → LLM knowledge: 100000 (conf: 0.7)

^C
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⏸ PAUSED after step 2 (steps 1-2 complete, step 3 in progress)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

What would you like to do?
  [c] Continue from step 3
  [r] Revise plan - provide new instructions
  [s] Skip step 3, move to step 4
  [b] Back up - redo step 2 with new context
  [i] Inspect - show current state (KB, scratchpad, tables)
  [q] Quit

> r

Enter correction or additional context:
> The VIP threshold should be $50,000 not $100,000. Also,
> only consider customers active in the last 12 months.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔄 REPLANNING with user feedback...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 REVISED PLAN (6 steps)
1. ✓ Load customer data from sales_db (complete)
2. ✓ Calculate lifetime value per customer (complete)
3. Filter to customers active in last 12 months  ← NEW
4. Define VIP rules (threshold: $50,000)         ← REVISED
5. Query for VIP customers below target
6. Generate recommendations

Continue? [Y/n] y

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
▶ Step 3/6: Filter to customers active in last 12 months
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [generating python...]
```

**Interrupt options:**
| Key | Action | Use case |
|-----|--------|----------|
| `c` | Continue | False alarm, keep going |
| `r` | Revise | Wrong approach, add context |
| `s` | Skip | This step isn't needed |
| `b` | Back up | Redo previous step differently |
| `i` | Inspect | Debug - see current state |
| `q` | Quit | Abort entirely |

**State preservation on interrupt:**
- Completed steps remain in scratchpad
- Facts in KB are retained
- DuckDB tables persist
- Revised plan incorporates completed work
- User feedback becomes part of context for remaining steps

### 11. API Server (`api.py`)

**REST + WebSocket API** for React UI integration:

```
reasoning_engine/
├── api.py               # FastAPI server
├── api_models.py        # Pydantic request/response schemas
└── api_events.py        # WebSocket event types
```

**Endpoints:**

```
POST /sessions
  → Create new reasoning session
  ← { session_id, status }

POST /sessions/{id}/solve
  → { problem: "Analyze VIP customers...", setup_file: "..." }
  ← { status: "started" }  (execution begins, updates via WebSocket)

POST /sessions/{id}/interrupt
  → { action: "revise", feedback: "Use $50k threshold" }
  ← { status: "replanning" }

GET /sessions/{id}/state
  ← { plan, current_step, scratchpad, kb_facts, tables, artifacts }

GET /sessions/{id}/artifacts/{artifact_id}
  ← { type: "python"|"prolog"|"sql", code, result, timing }

DELETE /sessions/{id}
  → Cleanup session resources
```

**WebSocket stream** (`/sessions/{id}/stream`):

```typescript
// React connects to WebSocket for real-time updates
type Event =
  | { type: "plan", steps: Step[] }
  | { type: "step_start", step: number, goal: string }
  | { type: "generating", step: number, language: "python" | "prolog" }
  | { type: "executing", step: number }
  | { type: "fact_resolving", fact: string }
  | { type: "fact_resolved", fact: string, source: string, confidence: number }
  | { type: "step_complete", step: number, result: StepResult, timing: number }
  | { type: "step_error", step: number, error: string, retry_count: number }
  | { type: "artifact", artifact: Artifact }
  | { type: "complete", summary: string, total_time: number }
  | { type: "paused", reason: string }
  | { type: "replanning", feedback: string }

// Artifact includes all intermediate outputs
type Artifact = {
  id: string
  step: number
  type: "python_code" | "prolog_rules" | "sql_query" | "table" | "derivation_trace"
  content: string
  result?: any
  timing?: number
}
```

**React UI can render:**

| Artifact Type | UI Component |
|---------------|--------------|
| `python_code` | Syntax-highlighted code block + output |
| `prolog_rules` | Rule viewer with KB state |
| `sql_query` | Query + result table |
| `table` | Interactive data grid (DuckDB results) |
| `derivation_trace` | Collapsible proof tree |
| `plan` | Checklist with progress indicators |
| `fact_resolution` | Source badge (DB/LLM) + confidence meter |

**Session state accessible for UI inspection:**

```python
GET /sessions/{id}/state

{
  "session_id": "abc123",
  "status": "running",
  "plan": {
    "steps": [...],
    "current_step": 3,
    "completed_steps": [1, 2]
  },
  "scratchpad": "## Step 1 Results\n...",
  "knowledge_base": {
    "facts": ["revenue(north, 2300000)", ...],
    "rules": ["above_target(R) :- ...", ...]
  },
  "tables": ["customers", "sales_2024", "regional_summary"],
  "artifacts": [
    { "id": "art_001", "step": 1, "type": "python_code", ... },
    { "id": "art_002", "step": 2, "type": "sql_query", ... }
  ]
}
```

**Server implementation:**

```python
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"])

sessions: dict[str, Session] = {}

@app.post("/sessions")
async def create_session(config: SessionConfig):
    session = Session(config)
    sessions[session.id] = session
    return {"session_id": session.id}

@app.websocket("/sessions/{session_id}/stream")
async def stream(websocket: WebSocket, session_id: str):
    await websocket.accept()
    session = sessions[session_id]

    # Session emits events, we forward to WebSocket
    async for event in session.events():
        await websocket.send_json(event.dict())
```

### 12. Interactive Refinement Loop (`repl.py`)
- After execution completes, enters interactive prompt mode
- User can provide follow-up prompts to:
  - **Refine**: Modify previous results ("Filter to only US customers")
  - **Extend**: Add new steps ("Now calculate the trend line")
  - **Re-plan**: Start fresh with a new approach
- Full context preserved:
  - Original problem
  - Previous plan and all executed steps
  - Scratchpad contents
  - All DuckDB tables from prior execution
- LLM receives refinement prompt + full history
- Generates incremental plan (new steps only) or revised plan
- Loops back to execution phase
- Exit with `quit` or `exit` command

## File Structure

```
reasoning_engine/
├── __init__.py
├── main.py              # CLI entry point
├── session.py           # Orchestration
├── planner.py           # Plan generation (English steps)
├── stepgen.py           # Step generator (Prolog or Python)
├── prolog_engine.py     # PySwip wrapper, KB management
├── executor.py          # Python code execution
├── scratchpad.py        # Markdown scratchpad
├── datastore.py         # DuckDB wrapper + Prolog bridge
├── assets.py            # Database asset manager
├── feedback.py          # Live monitoring output
├── repl.py              # Interactive refinement loop (CLI)
├── api.py               # FastAPI server
├── api_models.py        # Pydantic request/response schemas
├── api_events.py        # WebSocket event types
├── providers/
│   ├── __init__.py
│   ├── base.py          # Abstract LLM interface
│   ├── openai.py
│   ├── anthropic.py
│   └── ollama.py
├── prompts/
│   ├── planner.txt      # System prompt for planning
│   ├── codegen.txt      # System prompt for code generation
│   └── synthesize.txt   # System prompt for final output
├── config.py            # Configuration management
└── utils.py             # Shared utilities

# Example config.yaml (see Configuration File section for full spec)
databases:
  - name: sales_db
    uri: postgresql://${DB_USER}:${DB_PASS}@localhost:5432/sales
  - name: inventory_db
    uri: mysql+pymysql://${DB_USER}:${DB_PASS}@localhost:3306/inventory
```

## Key Design Decisions

### Code Execution Safety
- Subprocess isolation with timeout
- Restricted imports (configurable allowlist)
- No file system access outside designated dirs
- Memory limits via resource module

### LLM Context Management
- Each step receives: original problem, plan, scratchpad content, table schemas
- Token budget awareness - truncate old scratchpad sections if needed
- Structured output parsing (JSON mode where available)

### Error Handling
- **Syntax errors**: If generated code fails to compile, loop with error message until valid Python is produced (no limit)
- **Runtime errors**: If code throws an exception during execution, capture full traceback, feed back to LLM, regenerate and retry until it runs successfully (no limit)
- **Plan failures**: Option to re-plan from current state if a step is fundamentally blocked

### Data Flow Between Steps
- **Scratchpad**: Unstructured text, notes, intermediate reasoning
- **DuckDB tables**: Structured data, query results, datasets
- Both are included in LLM context for subsequent steps

## Example Usage

```python
from reasoning_engine import Session

session = Session(provider="anthropic", model="claude-sonnet-4-20250514")

# Initial problem - outputs plan, executes, shows results
result = session.solve("""
    Analyze the top 10 tech companies by market cap.
    Compare their P/E ratios and revenue growth.
    Identify which are undervalued.
""")

print(result.summary)
```

### Interactive Session (CLI)

```
$ reasoning-engine --setup databases.yaml

> Analyze sales by region for Q4 2024

Plan:
1. Query sales_db.transactions for Q4 2024
2. Aggregate totals by region
3. Generate summary report

[Step 1] Querying sales data...
[Step 1] ✓ Complete - 15,432 rows loaded

[Step 2] Aggregating by region...
[Step 2] ✓ Complete - results in table 'regional_sales'

[Step 3] Generating summary...
[Step 3] ✓ Complete

Summary: North region led Q4 with $2.3M revenue...

> Now compare this to Q4 2023

Plan (continued):
4. Query Q4 2023 data
5. Calculate year-over-year growth by region

[Step 4] Querying Q4 2023...
...

> quit
```

## Dependencies

**Core:**
- `duckdb` - Embedded SQL database
- `problog` - Probabilistic logic programming (pure Python, no system deps)
- `sqlalchemy` - Database connectivity, metadata extraction (supports JDBC-style URIs)
- `litellm` - Unified LLM provider interface (optional, simplifies multi-provider)
- `pydantic` - Data validation and settings
- `pyyaml` - Setup file parsing
- `rich` - Terminal output formatting
- `click` - CLI interface
- `fastapi` - REST API framework
- `uvicorn` - ASGI server
- `websockets` - WebSocket support for real-time streaming

**Standard Libraries for Generated Code:**
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `scipy` - Scientific computing, statistics
- `scikit-learn` - ML algorithms, vectorization, clustering
- `sentence-transformers` - Text embeddings
- `requests` - HTTP client

## Implementation Phases

The implementation is split into two phases. Phase 1 delivers a complete, functional agentic system with Python execution. Phase 2 adds the formal logic reasoning layer with ProbLog.

### Phase 1: Python-Only Execution Engine

**Goal**: A working system that can plan, execute Python/SQL, and explain in natural language.

**Deliverables**:
- Multi-step planning from natural language
- Python code generation and execution
- Database connectivity and querying
- Live progress feedback
- Interactive refinement loop
- Natural language explanations (not formal proofs)

**Implementation Order**:

1. **Config loading** - Parse YAML config with env var substitution
   - Pydantic models for validation
   - Database URIs, LLM settings, system prompt

2. **Database asset manager** - SQLAlchemy connections, schema introspection
   - **First-use initialization**: On startup, introspect all databases and cache:
     - Full metadata (columns, types, PKs, FKs, row counts)
     - Token-optimized overview for system prompt
     - Vector index for schema discovery (lazy-built on first search)
   - Provides LLM tools: `get_table_schema()`, `find_relevant_tables()`
   - Cache persists for session; rebuild on config change or explicit refresh

3. **Core data structures** - Plan, Step, Result, StepType models (Pydantic)
   - Design `StepType` as extensible enum: `PYTHON` now, `PROLOG` later
   - Include hooks in Step model for future `prolog_code`, `derivation_trace` fields

4. **LLM provider interface** - Start with one provider (e.g., Anthropic)
   - Abstract base class for easy provider addition

5. **Scratchpad** - Simple markdown read/write
   - Section-based structure for context management

6. **DuckDB datastore** - Basic table operations
   - Skip Prolog fact export for now
   - Design schema introspection to be reusable for fact generation later

7. **Python executor** - Subprocess code execution with safety
   - Timeout, resource limits, import allowlist
   - Capture stdout/stderr/exceptions for retry loop

8. **Planner** - Problem decomposition to English steps
   - Receives schema overview in system prompt
   - Can call `get_table_schema()` / `find_relevant_tables()` as tools
   - Returns structured Plan object

9. **Step generator** - Python-only initially
   - Design interface to accept `allowed_types: list[StepType]`
   - Phase 1: only `PYTHON` in allowed list
   - Phase 2: add `PROLOG` to allowed list

10. **Session orchestration** - Wire everything together
    - Route steps by type (currently only Python)
    - Design with plugin pattern for future Prolog engine

11. **Feedback system** - Live output with `rich`
    - Step progress, timing, retries
    - Extensible event types for future fact resolution events

12. **Interactive REPL** - Refinement loop with context preservation

13. **CLI** - Command-line interface with Click

14. **API server** - FastAPI + WebSocket for React UI (optional, can defer)

**Phase 1 Exit Criteria**:
- Can solve multi-step data analysis problems
- Provides natural language explanations
- Handles errors with retry loops
- Interactive refinement works
- Live feedback shows progress

---

### Phase 2: Prolog/ProbLog Integration

**Goal**: Add formal reasoning with auditable derivation traces for critical decisioning.

**Deliverables**:
- Probabilistic logic programming via ProbLog
- Fact provenance and confidence scoring
- On-demand fact resolution (Prolog → Python → LLM/DB)
- Derivation trace capture for formal proofs
- Hybrid Python/Prolog step execution

**Implementation Order**:

1. **ProbLog engine wrapper** (`prolog_engine.py`)
   - Pure Python via `problog` package (no SWI-Prolog dependency)
   - KB operations: `assert_facts()`, `assert_rules()`, `query()`, `query_with_trace()`
   - Confidence annotation: `0.8 :: fact(x, y).`

2. **Fact provenance system**
   - Tag facts with source: `database`, `llm_knowledge`, `llm_heuristic`
   - Confidence assignment by source type
   - Fact caching to avoid re-resolution

3. **On-demand fact resolution**
   - Resolver that routes to DB query or LLM knowledge
   - Mini-plan generation for complex derived facts
   - Batch resolution for efficiency

4. **DuckDB → Prolog bridge**
   - `to_prolog_facts(table)` exports rows as ProbLog facts
   - Automatic type mapping (strings quoted, numbers as-is)

5. **Derivation trace capture**
   - Capture proof trees from ProbLog queries
   - Format traces for human readability
   - Include in audit log

6. **Extend step generator**
   - Add `PROLOG` to `allowed_types`
   - LLM decides Python vs Prolog per step
   - Prolog generation: facts, rules, queries

7. **Extend session orchestration**
   - Route `PROLOG` steps to ProbLog engine
   - Maintain KB state across steps
   - Integrate fact resolution callbacks

8. **Extend feedback system**
   - Fact resolution events: `📥 Resolving fact: X`
   - Source and confidence display
   - Derivation trace in verbose mode

9. **Audit log structure**
   - Full session logging per spec
   - Derivation traces linked to conclusions
   - Export for compliance review

**Phase 2 Exit Criteria**:
- Can execute Prolog reasoning steps
- Generates formal derivation traces
- Confidence propagates through inference
- Audit log captures full provenance
- Hybrid Python/Prolog problems work end-to-end

---

### Design Considerations for Phasing

**Extensibility points to build into Phase 1**:

| Component | Phase 1 Design | Phase 2 Extension |
|-----------|----------------|-------------------|
| `StepType` | Enum with `PYTHON` | Add `PROLOG` |
| `Step` model | Optional `prolog_code`, `derivation_trace` fields | Populate these fields |
| Step generator | `allowed_types` parameter | Include `PROLOG` |
| Session | Plugin pattern for executors | Register ProbLog executor |
| Feedback | Event type enum | Add fact resolution events |
| Datastore | Schema introspection | Add `to_prolog_facts()` |

**Risk mitigation**:
- Phase 1 is valuable standalone for many use cases
- If ProbLog proves problematic, can explore alternatives (e.g., Clingo, custom inference)
- Clean interface boundaries allow swapping implementations

## Performance & Token Budget

**Target benchmarks:**
- Simple question (1-2 facts): < 30 seconds
- Medium problem (5-10 steps): 2-5 minutes
- Complex multi-phase (20+ steps): 10-15 minutes acceptable if explanation is complete

**LLM call inventory per problem:**

| Phase | LLM Calls | Tokens (est.) |
|-------|-----------|---------------|
| Planning | 1 | ~1K in, ~500 out |
| Per step (code gen) | 1 | ~2K in, ~500 out |
| Per fact resolution | 0-1 | ~1K in, ~200 out |
| Per retry (error) | 1 | ~1.5K in, ~500 out |
| Final synthesis | 1 | ~2K in, ~1K out |

**Example: 10-step problem with 15 fact resolutions, 3 retries**
- Planning: 1 call
- Steps: 10 calls
- Fact resolution: 15 calls
- Retries: 3 calls
- Synthesis: 1 call
- **Total: ~30 LLM calls**

At ~2 seconds/call (API latency + generation): **~60 seconds of LLM time**
Plus DB queries, code execution: **~30 seconds**
**Total: ~90 seconds** for a medium-complexity problem ✓

**Performance optimizations:**

1. **Caching** (biggest win)
   ```python
   @cache
   def resolve_fact(goal):
       # Same fact never resolved twice per session
   ```

2. **Batch fact resolution**
   ```python
   # Instead of 10 separate calls:
   missing_facts = problog.get_missing_facts(query)
   resolved = llm.resolve_batch(missing_facts)  # 1 call for all
   ```

3. **Model tiering**
   | Task | Model | Why |
   |------|-------|-----|
   | Planning | Claude Sonnet/Opus | Needs reasoning |
   | Code gen | Claude Sonnet | Needs accuracy |
   | Simple fact routing | Claude Haiku | Just classification |
   | SQL generation | Claude Haiku | Straightforward |

4. **Prefetch hints**
   ```python
   # Domain hints can suggest likely facts
   prefetch:
     - customer_tier(X, _) -> likely needs revenue(X, _) too
   ```

5. **Parallel resolution**
   ```python
   # Resolve independent facts concurrently
   facts = await asyncio.gather(*[resolve(f) for f in missing])
   ```

6. **Truncate context aggressively**
   - Old scratchpad sections: summarize or drop
   - KB state: only include relevant predicates
   - Mini-plan history: just results, not full trace

**Cost estimate (Claude pricing):**

| Problem Size | Calls | Input Tokens | Output Tokens | Est. Cost |
|--------------|-------|--------------|---------------|-----------|
| Simple | 5 | 5K | 1K | ~$0.02 |
| Medium | 30 | 40K | 10K | ~$0.15 |
| Complex | 100 | 150K | 40K | ~$0.60 |

**Red flags to watch:**
- Fact resolution chains > 5 deep (latency explosion)
- Retry loops > 3 per step (LLM struggling)
- Context > 50K tokens (summarize aggressively)

## Test Database

The **Chinook database** (`data/chinook.db`) is used for development and testing. It models a digital music store:

```
data/chinook.db (SQLite, ~1MB)

Tables:
  Artist (275 rows)          - Music artists
  Album (347 rows)           - Albums → Artist
  Track (3,503 rows)         - Songs → Album, Genre, MediaType
  Genre (25 rows)            - Rock, Jazz, Metal, etc.
  MediaType (5 rows)         - MPEG, AAC, etc.
  Playlist (18 rows)         - Curated playlists
  PlaylistTrack (8,715 rows) - Playlist ↔ Track mapping
  Customer (59 rows)         - Customers with addresses
  Employee (8 rows)          - Sales reps (hierarchical)
  Invoice (412 rows)         - Purchases → Customer
  InvoiceLine (2,240 rows)   - Line items → Invoice, Track
```

**Example test queries:**
- "What are the top 5 selling genres by revenue?"
- "Which artist has the most tracks?"
- "Show me customers who spent more than $40 total"
- "Compare sales performance of each employee"
- "What percentage of tracks are in playlists?"

**Test config:** `config.yaml` points to this database with appropriate domain context.

## Verification

- **Prolog engine tests**: KB operations, query execution, trace capture
- Integration test: pure reasoning problem (uses only Prolog)
- Integration test: pure data problem (uses only Python)
- Integration test: hybrid problem (data fetch → Prolog reasoning → conclusions)
- Verify derivation traces are human-readable
- Manual testing with Chinook database queries

## Market Differentiation

**vs PromptQL**: PromptQL provides reliable data retrieval but no formal reasoning or proofs. This system adds a reasoning layer with derivation traces.

**vs Traditional BRE (FICO, Drools)**: Those require manual rule authoring. This system uses LLM to generate rules from natural language.

**vs XAI Platforms**: Most provide post-hoc explanations (SHAP, LIME). This system provides formal proofs via Prolog derivations.

**Unique value proposition**: LLM-generated rules → ProbLog execution → auditable derivation traces for critical decisions.
