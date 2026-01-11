# Constat Architecture

Technical documentation of the system architecture and logic flow.

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              User Interface                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │    CLI      │  │    REPL     │  │ GraphQL API │  │   Python SDK        │ │
│  │ (cli.py)    │  │ (repl.py)   │  │ (api/)      │  │ (session.py)        │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
└─────────┼────────────────┼────────────────┼────────────────────┼────────────┘
          │                │                │                    │
          └────────────────┴────────────────┴────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Session Layer                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         Session (session.py)                             ││
│  │  - Orchestrates execution                                                ││
│  │  - Manages state and context                                             ││
│  │  - Handles retries and errors                                            ││
│  │  - Emits events for feedback                                             ││
│  └───────────────────────────────┬─────────────────────────────────────────┘│
└──────────────────────────────────┼──────────────────────────────────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          │                        │                        │
          ▼                        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────────┐
│ Mode: Exploratory│    │ Mode: Auditable │    │      Shared Services        │
│                 │    │                 │    │                             │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────────────────┐ │
│ │   Planner   │ │    │ │FactResolver │ │    │ │    SchemaManager        │ │
│ │(planner.py) │ │    │ │(fact_       │ │    │ │ (schema_manager.py)     │ │
│ └──────┬──────┘ │    │ │ resolver.py)│ │    │ └─────────────────────────┘ │
│        │        │    │ └──────┬──────┘ │    │ ┌─────────────────────────┐ │
│        ▼        │    │        │        │    │ │    LLM Provider         │ │
│ ┌─────────────┐ │    │        ▼        │    │ │ (providers/)            │ │
│ │  Executor   │ │    │ ┌─────────────┐ │    │ └─────────────────────────┘ │
│ │(executor.py)│ │    │ │ Derivation  │ │    │ ┌─────────────────────────┐ │
│ └─────────────┘ │    │ │   Trace     │ │    │ │    DataStore            │ │
└─────────────────┘    │ └─────────────┘ │    │ │ (datastore.py)          │ │
                       └─────────────────┘    │ └─────────────────────────┘ │
                                              └─────────────────────────────┘
                                                            │
                    ┌───────────────────────────────────────┴───────────────┐
                    │                   Data Sources                         │
                    │                                                        │
                    │  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
                    │  │   SQL    │  │  NoSQL   │  │   API    │            │
                    │  │(SQLAlch.)│  │Connectors│  │ Catalog  │            │
                    │  └──────────┘  └──────────┘  └──────────┘            │
                    └────────────────────────────────────────────────────────┘
```

## Request Processing Flow

### Exploratory Mode: Multi-Step Planning

Used for data exploration and analysis questions.

```
User Question: "What are the top 5 customers by revenue this quarter?"
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  1. PLANNING PHASE                                                          │
│                                                                             │
│  Input:                                                                     │
│    - User question                                                          │
│    - Schema overview (tables, columns, relationships)                       │
│    - System prompt (domain context)                                         │
│                                                                             │
│  LLM generates:                                                             │
│    Plan:                                                                    │
│      1. Query sales data for current quarter                                │
│      2. Aggregate revenue by customer                                       │
│      3. Rank and select top 5                                               │
│      4. Format results with customer details                                │
│                                                                             │
│  Output: Plan object with Step objects                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  2. STEP EXECUTION LOOP (for each step)                                     │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  2a. Code Generation                                                  │  │
│  │                                                                       │  │
│  │  Input:                                                               │  │
│  │    - Step goal                                                        │  │
│  │    - Schema details (on-demand via tools)                             │  │
│  │    - Scratchpad (previous step results)                               │  │
│  │    - Available tables in datastore                                    │  │
│  │                                                                       │  │
│  │  LLM generates: Python code                                           │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                            │                                                │
│                            ▼                                                │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  2b. Code Execution                                                   │  │
│  │                                                                       │  │
│  │  Executor runs code in sandboxed environment:                         │  │
│  │    - Database connections available (db, db_sales, etc.)              │  │
│  │    - DataStore for saving/loading results                             │  │
│  │    - pandas, numpy available                                          │  │
│  │    - Timeout enforced                                                 │  │
│  │    - Import whitelist checked                                         │  │
│  │                                                                       │  │
│  │  Output captured: stdout, stderr, exceptions                          │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                            │                                                │
│                            ▼                                                │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  2c. Error Handling / Retry Loop                                      │  │
│  │                                                                       │  │
│  │  If execution fails:                                                  │  │
│  │    1. Format error message with traceback                             │  │
│  │    2. Send error + previous code to LLM                               │  │
│  │    3. LLM generates corrected code                                    │  │
│  │    4. Retry execution                                                 │  │
│  │    5. Repeat until success or max_retries                             │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                            │                                                │
│                            ▼                                                │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  2d. State Persistence                                                │  │
│  │                                                                       │  │
│  │  - DataFrames auto-saved to DataStore                                 │  │
│  │  - Scratchpad updated with step summary                               │  │
│  │  - Artifacts recorded (code, output, errors)                          │  │
│  │  - Events emitted for UI feedback                                     │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  3. COMPLETION                                                              │
│                                                                             │
│  - All step outputs combined                                                │
│  - Session recorded in history                                              │
│  - DataStore persisted for future queries                                   │
│  - Result returned to caller                                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Auditable Mode: Fact Resolution

Used for compliance and scenarios requiring provable conclusions.

```
User Question: "Is customer C001 a VIP?"
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  1. QUESTION ANALYSIS                                                       │
│                                                                             │
│  FactResolver analyzes the question:                                        │
│    - Identifies target fact: is_vip(customer_id="C001")                     │
│    - Determines required sub-facts                                          │
│    - Checks what data sources are available                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  2. DERIVATION LOGIC GENERATION (LLM)                                       │
│                                                                             │
│  LLM generates derivation logic automatically:                              │
│                                                                             │
│    is_vip(customer_id) :=                                                   │
│      customer_revenue(customer_id) > vip_threshold()                        │
│      OR customer_tier(customer_id) == "gold"                                │
│                                                                             │
│  This is NOT pre-defined - the LLM figures it out based on:                 │
│    - The question                                                           │
│    - Available schema                                                       │
│    - System prompt (domain knowledge)                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  3. LAZY FACT RESOLUTION                                                    │
│                                                                             │
│  For each required fact, resolve in order:                                  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  Resolution Hierarchy:                                                  ││
│  │                                                                         ││
│  │  1. CACHE         - Already resolved this session?                      ││
│  │       │             (includes user-provided facts from prior turns)     ││
│  │       ▼ (miss)                                                          ││
│  │  2. CONFIG        - Defined in config.yaml?                             ││
│  │       │             (e.g., vip_threshold = 100000)                      ││
│  │       ▼ (miss)                                                          ││
│  │  3. DATABASE      - Can be queried from a database?                     ││
│  │       │             LLM generates SQL query                             ││
│  │       │             Confidence: 1.0                                     ││
│  │       ▼ (miss)                                                          ││
│  │  4. LLM KNOWLEDGE - LLM foundational knowledge?                         ││
│  │       │             (e.g., "Paris is capital of France")                ││
│  │       │             Confidence: 0.6-0.8                                 ││
│  │       ▼ (miss)                                                          ││
│  │  5. SUB-PLAN      - Requires multi-step derivation?                     ││
│  │       │             Generate mini-plan and execute                      ││
│  │       ▼ (fail)                                                          ││
│  │  6. UNRESOLVED    - Return with missing facts explanation               ││
│  │                     User can provide facts via follow-up                ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                             │
│  User-Provided Facts (follow-up flow):                                      │
│                                                                             │
│  When facts are unresolved, the user can provide them in natural language:  │
│    User: "There were 1 million people at the march"                         │
│    → LLM extracts: march_attendance = 1000000                               │
│    → Fact added to cache with source: USER_PROVIDED                         │
│    → Resolution re-attempted with new fact available                        │
│                                                                             │
│                                                                             │
│  Example resolution for is_vip(C001):                                       │
│                                                                             │
│    is_vip(C001) = ?                                                         │
│      └─ customer_revenue(C001) = ?                                          │
│           └─ DATABASE: SELECT SUM(amount) FROM orders WHERE customer_id='C001' │
│           └─ Result: 150000, Confidence: 1.0                                │
│      └─ vip_threshold = ?                                                   │
│           └─ CONFIG: config.yaml                                            │
│           └─ Result: 100000, Confidence: 1.0                                │
│      └─ 150000 > 100000 = True                                              │
│    is_vip(C001) = True                                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  4. DERIVATION TRACE GENERATION                                             │
│                                                                             │
│  Build complete trace showing:                                              │
│    - Each fact and its value                                                │
│    - Source of each fact (DATABASE, CONFIG, LLM_KNOWLEDGE, DERIVED)         │
│    - Confidence level                                                       │
│    - Exact query/timestamp for database facts                               │
│    - Reasoning at each step                                                 │
│                                                                             │
│  Output:                                                                    │
│    is_vip(customer_id=C001) = True                                          │
│      derived_from:                                                          │
│        customer_revenue(customer_id=C001) = 150000                          │
│          source: DATABASE                                                   │
│          query: SELECT SUM(amount) FROM orders WHERE customer_id = 'C001'   │
│          executed_at: 2025-01-11T10:23:45Z                                  │
│          confidence: 1.0                                                    │
│                                                                             │
│        vip_threshold = 100000                                               │
│          source: CONFIG (config.yaml)                                       │
│          confidence: 1.0                                                    │
│                                                                             │
│      reasoning: Customer revenue ($150,000) exceeds VIP threshold ($100,000)│
└─────────────────────────────────────────────────────────────────────────────┘
```

### Follow-Up Questions (Context Preservation)

```
Initial: "Show me Q4 revenue by region"
Follow-up: "Now compare this to last year"
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  CONTEXT AVAILABLE FROM PREVIOUS WORK                                       │
│                                                                             │
│  1. DataStore Tables:                                                       │
│     - q4_revenue (from step 1 of initial query)                             │
│     - regional_summary (from step 2)                                        │
│                                                                             │
│  2. State Variables:                                                        │
│     - top_regions = ['North', 'South', 'East']                              │
│     - total_revenue = 2300000                                               │
│                                                                             │
│  3. Scratchpad:                                                             │
│     - Step 1: Queried transactions for Q4 2024                              │
│     - Step 2: Aggregated by region, North led with $890K                    │
│                                                                             │
│  This context is automatically included in the follow-up plan generation.  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  FOLLOW-UP PLAN GENERATION                                                  │
│                                                                             │
│  LLM sees:                                                                  │
│    - Previous scratchpad context                                            │
│    - Available tables from previous steps                                   │
│    - New question: "compare this to last year"                              │
│                                                                             │
│  Generates incremental plan:                                                │
│    Step 3: Query Q4 2023 data (load_dataframe not needed - fresh query)     │
│    Step 4: Join with existing q4_revenue table                              │
│    Step 5: Calculate YoY growth percentages                                 │
│                                                                             │
│  Steps are numbered to continue from previous work.                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### Session (`session.py`)

The central orchestrator that manages the execution lifecycle.

**Responsibilities:**
- Initialize components (SchemaManager, LLM, Planner, Executor)
- Manage session state (datastore, scratchpad)
- Execute plans step-by-step
- Handle retries and errors
- Emit events for monitoring
- Record history for resumption

**Key Methods:**
- `solve(problem)` - Full execution from question to answer
- `follow_up(question)` - Continue with context preserved
- `resume(session_id)` - Load and continue a previous session
- `get_state()` - Inspect current session state

### Planner (`execution/planner.py`)

Converts natural language questions into executable plans.

**Input:**
- User question
- Schema overview
- System prompt (domain context)

**Output:**
- Plan object with ordered Step objects
- Each step has: goal, expected_inputs, expected_outputs

**LLM Interaction:**
- Uses schema tools to query detailed table info
- Considers available data sources
- Structures multi-step approach

### Executor (`execution/executor.py`)

Runs generated Python code in a sandboxed environment.

**Security:**
- Subprocess isolation
- Configurable timeout
- Import whitelist
- No filesystem access outside designated paths

**Available to Code:**
- `db`, `db_<name>` - Database connections
- `store` - DataStore for persistence
- `pd`, `np` - pandas, numpy

### DataStore (`storage/datastore.py`)

Persistent storage for session state.

**Capabilities:**
- SQLAlchemy backend (SQLite, PostgreSQL, DuckDB)
- DataFrame storage and retrieval
- State variable persistence
- Scratchpad entries
- Artifact logging

**Cross-Step State:**
- Only shared state between steps
- Each step runs in isolation
- Must explicitly save to share data

### SchemaManager (`catalog/schema_manager.py`)

Provides schema information to the LLM for both SQL and NoSQL databases.

**NoSQL Support:**

The SchemaManager handles both SQL and NoSQL databases transparently:

```yaml
databases:
  # SQL (default)
  sales:
    uri: postgresql://localhost/sales

  # MongoDB
  documents:
    type: mongodb
    uri: mongodb://localhost:27017
    database: myapp

  # DynamoDB
  events:
    type: dynamodb
    region: us-east-1
```

NoSQL databases are introspected by sampling documents to infer schema.
All databases appear uniformly in the schema overview and vector search.

### DiscoveryTools (`discovery/tools.py`)

Provides on-demand schema, API, and document discovery via tool calling.

**Automatic Mode Selection:**

The system automatically detects whether the LLM supports tool calling:

| Model Type | Tool Support | Prompt Mode |
|------------|--------------|-------------|
| Claude 3+, GPT-4, Gemini | Yes | Minimal (~500 tokens) + discovery tools |
| Claude 2, GPT-3 Instruct | No | Full metadata in prompt |

**Discovery Tools:**

| Category | Tools |
|----------|-------|
| Schema | `list_databases`, `list_tables`, `get_table_schema`, `search_tables`, `get_table_relationships`, `get_sample_values` |
| API | `list_apis`, `list_api_operations`, `get_operation_details`, `search_operations` |
| Documents | `list_documents`, `get_document`, `search_documents`, `get_document_section` |
| Facts | `resolve_fact`, `add_fact`, `extract_facts_from_text`, `list_known_facts`, `get_unresolved_facts` |

**Token Savings:**

| Scenario | Current (full prompt) | With Tools | Savings |
|----------|----------------------|------------|---------|
| 5 DBs, 50 tables | ~8,000 tokens | ~500 + ~800 discovered | 84% |
| 10 APIs, 100 ops | ~12,000 tokens | ~500 + ~600 discovered | 91% |
| 20 documents | ~15,000 tokens | ~500 + ~400 discovered | 94% |

**PromptBuilder Usage:**

```python
from constat.discovery import DiscoveryTools, PromptBuilder

tools = DiscoveryTools(schema_manager, api_catalog, config)
builder = PromptBuilder(tools)

# Automatic mode selection based on model
prompt, use_tools = builder.build_prompt("claude-sonnet-4-20250514")
# → use_tools=True, minimal prompt

prompt, use_tools = builder.build_prompt("claude-2")
# → use_tools=False, full metadata embedded

# Check token estimates
estimate = builder.estimate_tokens("claude-sonnet-4-20250514")
# → {"mode": "tool_discovery", "savings_percent": 85, ...}
```

### APICatalog (`catalog/api_catalog.py`)

Provides API operation metadata for external services.

**Supported API Types:**

| Type | Discovery | Example |
|------|-----------|---------|
| GraphQL | Auto-introspection | Countries API |
| OpenAPI | Spec parsing (URL, file, inline) | Petstore API |

**Configuration:**

```yaml
apis:
  # GraphQL - auto-introspects schema
  countries:
    type: graphql
    url: https://countries.trevorblades.com/graphql

  # OpenAPI - parses spec to discover endpoints
  petstore:
    type: openapi
    spec_url: https://petstore.swagger.io/v2/swagger.json

  # OpenAPI inline - for simple APIs without a spec file
  weather:
    type: openapi
    url: https://api.weather.gov
    spec_inline:
      openapi: "3.0.0"
      paths:
        /points/{lat},{lon}:
          get:
            operationId: getPoint
            parameters:
              - name: lat
                in: path
                required: true
```

**LLM Tools:**
- `find_api_operations(query)` - Semantic search for relevant operations
- `get_api_operation(name)` - Full metadata for an operation

### FactResolver (`execution/fact_resolver.py`)

Resolves facts with full provenance for auditable mode.

**Fact Sources and Confidence:**

| Source | Confidence | Example |
|--------|------------|---------|
| CACHE | Preserved | Previously resolved fact |
| CONFIG | 1.0 | Config value |
| DATABASE | 1.0 | Query result |
| LLM_KNOWLEDGE | Parsed (default 0.6) | World knowledge |
| LLM_HEURISTIC | Parsed (default 0.6) | Industry standard |
| SUB_PLAN | min(dependencies) | Computed from other facts |
| USER_PROVIDED | 1.0 | User stated in follow-up |
| UNRESOLVED | 0.0 | Could not resolve |

**Resolution Strategy:**
1. Check cache (includes user-provided facts from prior turns)
2. Check config
3. Query database (LLM generates SQL)
4. Ask LLM for knowledge/heuristics
5. Generate sub-plan for complex derivations
6. Return UNRESOLVED (user can provide via follow-up)

**User-Provided Facts:**

When facts are unresolved, users can provide them in natural language:

```python
# User provides missing fact
session.provide_facts("There were 1 million people at the march")
# → Extracts: march_attendance = 1000000
# → Added to cache with source: USER_PROVIDED
# → Resolution re-attempted
```

REPL commands: `/unresolved` to view missing facts, `/facts <text>` to provide them.

### FeedbackDisplay (`feedback.py`)

Rich-based terminal UI for real-time feedback.

**Events Displayed:**
- Plan overview with step checklist
- Step start/complete/error
- Retry indicators
- Timing information
- Tables created

### InteractiveREPL (`repl.py`)

Interactive command loop for exploration.

**Commands:**
- `/tables` - Show available tables
- `/query <sql>` - Run SQL on datastore
- `/state` - Inspect session state
- `/history` - List previous sessions
- `/resume <id>` - Continue a session

## Data Flow

### Step Isolation Model

```
Step N                              Step N+1
┌────────────────────────┐         ┌────────────────────────┐
│  Local namespace       │         │  Local namespace       │
│  (not persisted)       │         │  (fresh start)         │
│                        │         │                        │
│  df = pd.read_sql(...) │         │  # Can't see df!       │
│  result = df.sum()     │         │  # Must load from store│
│                        │         │                        │
│  # Persist explicitly: │         │  df = store.load_      │
│  store.save_dataframe( │ ──────▶ │    dataframe('sales')  │
│    'sales', df, step=N)│         │                        │
└────────────────────────┘         └────────────────────────┘
          │                                   │
          │                                   │
          ▼                                   ▼
┌─────────────────────────────────────────────────────────────┐
│                    DataStore (DuckDB/SQLite)                │
│                                                             │
│  Tables: sales (step N), customers (step N-1), ...          │
│  State: total_revenue=123456, regions=['N','S','E']         │
│  Scratchpad: Step 1: ..., Step 2: ...                       │
│  Artifacts: code, output, errors                            │
└─────────────────────────────────────────────────────────────┘
```

### Event Flow

```
Session                    FeedbackHandler            FeedbackDisplay
   │                              │                          │
   │  emit(step_start)            │                          │
   │─────────────────────────────▶│                          │
   │                              │   step_start(...)        │
   │                              │─────────────────────────▶│
   │                              │                          │ Print: "Step 1: ..."
   │                              │                          │
   │  emit(generating)            │                          │
   │─────────────────────────────▶│                          │
   │                              │   step_generating(...)   │
   │                              │─────────────────────────▶│
   │                              │                          │ Print: "generating..."
   │                              │                          │
   │  emit(executing)             │                          │
   │─────────────────────────────▶│                          │
   │                              │   step_executing(...)    │
   │                              │─────────────────────────▶│
   │                              │                          │ Print: "executing..."
   │                              │                          │
   │  emit(step_complete)         │                          │
   │─────────────────────────────▶│                          │
   │                              │   step_complete(...)     │
   │                              │─────────────────────────▶│
   │                              │                          │ Print: "OK 2.3s"
```

## Error Handling

### Retry Strategy

```
Code Execution
      │
      ▼
  ┌───────┐
  │Execute│
  └───┬───┘
      │
      ▼
  ┌───────────┐
  │ Success?  │──── Yes ────▶ Continue to next step
  └─────┬─────┘
        │ No
        ▼
  ┌─────────────────────────┐
  │  Format error message   │
  │  - Full traceback       │
  │  - Line numbers         │
  │  - Variable state       │
  └───────────┬─────────────┘
              │
              ▼
  ┌─────────────────────────┐
  │  Send to LLM:           │
  │  - Previous code        │
  │  - Error details        │
  │  - Available context    │
  └───────────┬─────────────┘
              │
              ▼
  ┌─────────────────────────┐
  │  LLM generates          │
  │  corrected code         │
  └───────────┬─────────────┘
              │
              ▼
  ┌───────────────────┐
  │ attempt < max?    │──── Yes ────▶ Loop back to Execute
  └─────────┬─────────┘
            │ No
            ▼
      Step Failed
```

### Error Categories

| Error Type | Retry? | Example |
|------------|--------|---------|
| Syntax error | Yes | `df.groupby('region).sum()` |
| Column not found | Yes | `df['revnue']` (typo) |
| Type error | Yes | `"hello" + 123` |
| Timeout | No | Infinite loop |
| Import blocked | No | `import os` |
| Connection failure | Configurable | Database down |

## Performance Considerations

### Token Budget

| Phase | Tokens (est.) |
|-------|---------------|
| Planning | ~1K in, ~500 out |
| Per step (code gen) | ~2K in, ~500 out |
| Per retry | ~1.5K in, ~500 out |
| Per fact resolution | ~1K in, ~200 out |

### Optimization Strategies

1. **Schema Caching** - Introspect once, cache for session
2. **Fact Caching** - Never resolve same fact twice
3. **Model Tiering** - Use cheaper models for simple tasks
4. **Batch Resolution** - Resolve multiple facts in one LLM call
5. **Context Truncation** - Summarize old scratchpad entries
