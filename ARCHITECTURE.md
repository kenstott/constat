# Constat Architecture

Technical documentation of the system architecture and logic flow.

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Client Access Layer                                 │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌───────────────────┐  │
│  │   Web UI     │ │     CLI      │ │  Textual     │ │    Python SDK     │  │
│  │ (constat-ui/)│ │   (cli.py)   │ │    REPL      │ │   (session.py)    │  │
│  │  React/TS    │ │              │ │(textual_     │ │                   │  │
│  │              │ │              │ │  repl.py)    │ │                   │  │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └─────────┬─────────┘  │
└─────────┼────────────────┼────────────────┼───────────────────┼────────────┘
          │                └────────────────┼───────────────────┘
          │   REST + WebSocket              │
          └──────────┐                      │
                     ▼                      │
┌─────────────────────────────────────┐     │
│  Server Layer (constat/server/)     │     │
│  ┌──────────────────────────────┐   │     │
│  │  FastAPI (app.py)            │   │     │
│  │  REST routes + WebSocket     │   │     │
│  │  SessionManager              │   │     │
│  │  Firebase Auth               │   │     │
│  └──────────────┬───────────────┘   │     │
└─────────────────┼───────────────────┘     │
                  └─────────────┬───────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Session Layer                                     │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         Session (session.py)                             ││
│  │  - Orchestrates execution                                                ││
│  │  - Manages state and context                                             ││
│  │  - Handles retries and errors                                            ││
│  │  - Emits events for feedback                                             ││
│  └───────────────────────────────┬─────────────────────────────────────────┘│
└──────────────────────────────────┼──────────────────────────────────────────┘
                                   │
               ┌───────────────────┼───────────────────────┐
               │                   │                       │
               ▼                   ▼                       ▼
┌──────────────────────┐ ┌──────────────────┐ ┌───────────────────────────────┐
│  Intent Classifier   │ │ Execution Modes  │ │      Shared Services          │
│ (intent_classifier.  │ │                  │ │                               │
│  py)                 │ │ ┌──────────────┐ │ │ ┌───────────────────────────┐ │
│                      │ │ │ Exploratory  │ │ │ │    SchemaManager          │ │
│ Embedding + LLM      │ │ │  Planner     │ │ │ │ (schema_manager.py)       │ │
│ classification →     │ │ │  Executor    │ │ │ └───────────────────────────┘ │
│ CLARIFY / PLAN /     │ │ │  DAG Sched.  │ │ │ ┌───────────────────────────┐ │
│ EXECUTE / PROVE      │ │ └──────────────┘ │ │ │    TaskRouter             │ │
│                      │ │ ┌──────────────┐ │ │ │ (providers/router.py)     │ │
│                      │ │ │  Auditable   │ │ │ └───────────────────────────┘ │
│                      │ │ │ FactResolver │ │ │ ┌───────────────────────────┐ │
│                      │ │ │ ProbLog      │ │ │ │    DataStore              │ │
│                      │ │ │ Proof Tree   │ │ │ │ (datastore.py)            │ │
│                      │ │ └──────────────┘ │ │ └───────────────────────────┘ │
└──────────────────────┘ └──────────────────┘ │ ┌───────────────────────────┐ │
                                              │ │    DiscoveryTools         │ │
                                              │ │ (discovery/)              │ │
                                              │ └───────────────────────────┘ │
                                              │ ┌───────────────────────────┐ │
                                              │ │    LearningStore          │ │
                                              │ │ (storage/learnings.py)    │ │
                                              │ └───────────────────────────┘ │
                                              └───────────────────────────────┘
                                                             │
                  ┌──────────────────────────────────────────┴──────────────────┐
                  │                    Data Sources                              │
                  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
                  │  │   SQL    │ │  NoSQL   │ │  Files   │ │ External │       │
                  │  │Databases │ │Databases │ │(CSV/JSON/│ │   APIs   │       │
                  │  │(SQLAlch.)│ │Connectors│ │ Parquet) │ │(GraphQL/ │       │
                  │  │          │ │(MongoDB, │ │          │ │ OpenAPI) │       │
                  │  │          │ │ DynamoDB,│ │          │ │          │       │
                  │  │          │ │ Elastic, │ │          │ │          │       │
                  │  │          │ │ CosmosDB,│ │          │ │          │       │
                  │  │          │ │Cassandra,│ │          │ │          │       │
                  │  │          │ │Firestore)│ │          │ │          │       │
                  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
                  └────────────────────────────────────────────────────────────┘
```

**Key distinction:**
- **Client Access Layer**: Ways to USE Constat (Web UI, CLI, Textual REPL, Python SDK)
- **Server Layer**: FastAPI REST API + WebSocket for the Web UI and external consumers
- **Data Sources**: External systems Constat QUERIES (SQL databases, NoSQL databases, file-based sources like CSV/JSON/Parquet, and external GraphQL/REST APIs via API Executor)

## Intent Classification (`execution/intent_classifier.py`)

All queries are classified before routing to an execution mode.

**IntentClassifier** uses a two-tier approach:
1. **Embedding similarity** against exemplar intents (BAAI/bge-large-en-v1.5, 1024 dimensions)
2. **LLM fallback** for low-confidence classifications

**Classification thresholds:**
- Primary intent: 0.80 confidence (determines code path)
- Sub-intent: 0.65 confidence (defaults to None if below)

**Intent types** (defined in `execution/intent.py`):

| Intent | Description |
|--------|-------------|
| NEW_QUESTION | Fresh query requiring planning |
| QUERY | Simple data lookup |
| LOOKUP | Knowledge/document lookup |
| DRILL_DOWN | Dig deeper into previous results |
| COMPARE | Compare datasets or results |
| SUMMARIZE | Summarize previous analysis |
| EXPORT | Export data/artifacts |
| EXTEND | Add to existing plan |
| REDO | Re-execute with changes |
| MODIFY_FACT | Change an assumed fact |
| STEER_PLAN | Redirect plan execution |
| REFINE_SCOPE | Narrow or broaden scope |
| CHALLENGE | Question a result |
| MODE_SWITCH | Switch execution mode |
| PROVENANCE | Ask about data lineage |
| CREATE_ARTIFACT | Generate chart/report/email |
| TRIGGER_ACTION | Execute a side-effect |
| PREDICT | Forecasting request |
| ALERT | Set up a monitor |
| RESET | Clear session state |

**Mode preservation:** Redo-like intents (REDO, PREDICT, MODIFY_FACT, REFINE_SCOPE, STEER_PLAN) preserve the previous execution mode unless MODE_SWITCH is explicit.

**Multi-intent support:** Messages are split on sentence delimiters (`.` and `;`) and each segment is classified independently.

## Execution Modes (`execution/mode.py`)

```python
class Mode(Enum):
    EXPLORATORY = "exploratory"  # Multi-step planner (default)
    PROOF = "proof"              # Fact resolver with derivation traces
```

```python
class Phase(Enum):
    IDLE = "idle"                        # Waiting for input
    PLANNING = "planning"                # Generating/revising plan
    AWAITING_APPROVAL = "awaiting_approval"  # Plan ready, needs approval
    EXECUTING = "executing"              # Execution in progress
    FAILED = "failed"                    # Execution failed
```

All queries run exploratory by default. Use `/prove` for auditable proofs.

## Request Processing Flow

### Exploratory Mode: Multi-Step Planning

Used for data exploration and analysis questions.

```
User Question: "What are the top 5 customers by revenue this quarter?"
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  0. INTENT CLASSIFICATION                                                   │
│                                                                             │
│  IntentClassifier analyzes query:                                           │
│    - Embedding similarity against exemplar intents                          │
│    - LLM fallback for low-confidence cases                                  │
│    - Result: NEW_QUESTION → route to planning                               │
│                                                                             │
│  For follow-ups, intent determines routing:                                 │
│    - DRILL_DOWN → follow_up() with context preservation                     │
│    - REDO → re-execute with modifications                                   │
│    - EXPORT → generate artifact from existing results                       │
│    - etc.                                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  1. CLARIFICATION PHASE (if needed)                                         │
│                                                                             │
│  System detects ambiguous requests that need clarification before planning: │
│    - Geographic scope: "how many bears?" → "In what region?"                │
│    - Time period: "show sales" → "For what date range?"                     │
│    - Threshold values: "top customers" → "How many - top 5, 10, 20?"        │
│    - Category filters: "product analysis" → "Which product categories?"     │
│                                                                             │
│  If ambiguity detected:                                                     │
│    1. Present clarifying questions to user                                  │
│    2. User provides answers (or skips to proceed anyway)                    │
│    3. Enhanced question passed to planning phase                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  2. PLANNING PHASE                                                          │
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
│  Output: Plan object with Step objects arranged as a DAG                    │
│  Phase transitions: PLANNING → AWAITING_APPROVAL                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  3. PLAN APPROVAL                                                           │
│                                                                             │
│  User reviews plan and can:                                                 │
│    - Approve as-is                                                          │
│    - Suggest modifications → re-plan with feedback                          │
│    - Reject entirely                                                        │
│                                                                             │
│  Phase transitions: AWAITING_APPROVAL → EXECUTING                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  4. DAG EXECUTION (for each step, parallelized where possible)              │
│                                                                             │
│  ParallelScheduler runs steps in waves based on dependency graph:           │
│    - Wave 0: All leaf steps (no dependencies) run in parallel               │
│    - Wave N: Steps depending on Wave 0..N-1 run when dependencies resolve   │
│    - Max concurrent: 5 steps, per-step timeout: 60s                         │
│    - fail_fast: stop all on first failure                                   │
│                                                                             │
│  For each step:                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  4a. Code Generation                                                  │  │
│  │  LLM generates Python code using:                                     │  │
│  │    - Step goal + schema tools (on-demand discovery)                   │  │
│  │    - Scratchpad (previous step results)                               │  │
│  │    - Available DataStore tables                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                            │                                                │
│                            ▼                                                │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  4b. Code Execution (sandboxed subprocess)                            │  │
│  │  Available: db connections, store, pandas, numpy                      │  │
│  │  Enforced: timeout, import whitelist, path restrictions               │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                            │                                                │
│                            ▼                                                │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  4c. Error Handling / Retry Loop                                      │  │
│  │  Error + traceback + previous code → LLM → corrected code → retry    │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                            │                                                │
│                            ▼                                                │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  4d. State Persistence                                                │  │
│  │  DataFrames auto-saved, scratchpad updated, artifacts recorded        │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  5. COMPLETION                                                              │
│                                                                             │
│  - All step outputs combined                                                │
│  - Session recorded in history                                              │
│  - DataStore persisted for future queries                                   │
│  - Learnings extracted and stored                                           │
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
│  3. LAZY FACT RESOLUTION (parallel within each level)                       │
│                                                                             │
│  FactNode DAG built with dependencies. Resolved level by level:             │
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
│  │       │             LLM generates SQL → transpiled to target dialect    ││
│  │       │             Confidence: 1.0                                     ││
│  │       ▼ (miss)                                                          ││
│  │  4. DOCUMENT      - Found in configured documents?                      ││
│  │       │             Semantic search via vector embeddings               ││
│  │       ▼ (miss)                                                          ││
│  │  5. LLM KNOWLEDGE - LLM foundational knowledge?                         ││
│  │       │             (e.g., "Paris is capital of France")                ││
│  │       │             Confidence: 0.6-0.8                                 ││
│  │       ▼ (miss)                                                          ││
│  │  6. SUB-PLAN      - Requires multi-step derivation?                     ││
│  │       │             Generate mini-plan and execute                      ││
│  │       ▼ (fail)                                                          ││
│  │  7. UNRESOLVED    - Return with missing facts explanation               ││
│  │                     User can provide facts via follow-up                ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                             │
│  Parallel execution:                                                        │
│    Level 0 (leaf facts) → resolved concurrently via asyncio.gather          │
│    Level N → resolved once Level 0..N-1 complete                            │
│    Sub-proofs resolved recursively with parallelization at each level       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  4. PROOF TREE + DERIVATION TRACE                                           │
│                                                                             │
│  ProofNode tree built showing:                                              │
│    - Each fact, its value, source, and confidence                           │
│    - Exact SQL query or code used                                           │
│    - Hierarchical children for sub-facts                                    │
│    - Status: PENDING → RESOLVING → RESOLVED / FAILED / CACHED              │
│                                                                             │
│  Rendered as Rich Tree in REPL or interactive DAG in Web UI                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

### ProbLog Resolver (`execution/problog_resolver.py`)

Alternative fact resolution engine using logic programming semantics:

- Facts are **predicates** (ground truths from database queries)
- Dependencies are **rules** (symbolic derivations)
- Resolution is **depth-first search** (Prolog-style)
- Proofs come automatically from the resolution trace
- Confidence values propagated as probabilities

Register SQL executors for database-backed predicates. `resolve_fact()` returns a value plus `proof.to_trace()` for the full derivation chain.

### Knowledge Mode: Document Lookup + LLM Synthesis

Used for explanation and knowledge requests that don't require data analysis.

```
User Question: "Explain the revenue recognition process"
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  1. MODE DETECTION                                                          │
│                                                                             │
│  Query analyzed for knowledge keywords:                                     │
│    - "explain", "describe", "what is", "how does"                          │
│    - "process", "procedure", "workflow", "policy"                          │
│    - "tell me about", "definition of", "overview of"                       │
│                                                                             │
│  If knowledge keywords dominate and no data analysis needed:                │
│    → Route to KNOWLEDGE mode                                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  2. DOCUMENT SEARCH                                                         │
│                                                                             │
│  Semantic search across configured documents:                               │
│    - Search query: "revenue recognition process"                            │
│    - Vector embeddings via sentence-transformers                            │
│    - Return top-k relevant excerpts with relevance scores                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  3. LLM SYNTHESIS                                                           │
│                                                                             │
│  LLM synthesizes explanation from document excerpts + world knowledge.      │
│  Cites specific documents. Single LLM call, no planning.                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  4. OUTPUT WITH SOURCES                                                     │
│                                                                             │
│  Returns synthesized explanation with sources consulted.                    │
│  No plan, no code execution, fast response time.                            │
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
│  LLM sees previous scratchpad context + available tables + new question.    │
│  Generates incremental plan continuing from previous work.                  │
│  Steps numbered to continue from previous work.                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### Session (`session.py`)

The central orchestrator (~9,500 lines) that manages the execution lifecycle.

**Responsibilities:**
- Initialize components (SchemaManager, TaskRouter, Planner, Executor, IntentClassifier)
- Route LLM requests via TaskRouter based on task type with automatic escalation
- Manage session state (datastore, scratchpad, context)
- Execute plans step-by-step with DAG-based parallel scheduling
- Handle retries, errors, and mode transitions
- Emit events for monitoring and feedback
- Record history for resumption
- Integrate with learning system (store and apply learnings/rules)
- Detect and handle meta-questions without data queries

**Key Methods:**
- `solve(problem)` - Full execution from question to answer
- `follow_up(question)` - Continue with context preserved
- `resume(session_id)` - Load and continue a previous session
- `get_state()` - Inspect current session state
- `set_clarification_callback(callback)` - Set handler for ambiguous questions
- `set_approval_callback(callback)` - Set handler for plan approval

### Server (`server/app.py`)

FastAPI application serving the REST API and WebSocket connections.

**App Factory** creates the FastAPI app with:
- CORS middleware
- Config hashing for incremental updates
- SessionManager for server-side session lifecycle
- Firebase authentication integration

**Routes** (`server/routes/`):

| Route Module | Endpoints |
|-------------|-----------|
| `queries.py` | Query execution, streaming results |
| `data.py` | Data management, table operations |
| `databases.py` | Database configuration CRUD |
| `schema.py` | Schema introspection |
| `sessions.py` | Session lifecycle (create, resume, list) |
| `files.py` | File upload, document management |
| `learnings.py` | Learning and rule storage |
| `roles.py` | Role management |
| `skills.py` | Skill discovery and management |
| `users.py` | User operations |

**WebSocket** (`server/websocket.py`): Real-time streaming of execution events (plan updates, step progress, results) to the Web UI.

**SessionManager** (`server/session_manager.py`): Server-side session lifecycle management, maps API sessions to Session instances.

### DAG Execution (`execution/dag.py`, `execution/parallel_scheduler.py`)

Plans are represented as directed acyclic graphs for parallel execution.

**FactNode** - Node in the execution DAG:
- `name`, `description`, `source` (database/document/knowledge/computed)
- `dependencies` - list of FactNodes this depends on
- Status: `PENDING` → `RUNNING` → `RESOLVED` / `FAILED`
- Execution metadata: `sql_query`, `code`, `row_count`, `confidence`
- Properties: `is_leaf` (no dependencies), `is_table` (row_count > 1)

**ParallelScheduler** - Level-based parallel execution:

```
Wave 0: [step_a, step_b, step_c]  ← all leaves, run in parallel
Wave 1: [step_d, step_e]          ← depend on wave 0, run when ready
Wave 2: [step_f]                  ← depends on wave 1
```

**SchedulerConfig:**
- `max_concurrent`: 5 steps
- `step_timeout`: 60 seconds
- `fail_fast`: True (stop all on first failure)

**ExecutionContext:** Manages cancellation with `is_cancelled()` check for running steps.

### Engine (`execution/engine.py`)

Single-shot query execution engine for simpler queries that don't need full planning.

**QueryResult:** `success`, `answer`, `code`, `attempts`, `error`, `attempt_history`

Uses SCHEMA_TOOLS + DOC_TOOLS for on-demand discovery. Automatic retry on errors via `format_error_for_retry`.

### Planner (`execution/planner.py`)

Converts natural language questions into executable plans.

**Input:**
- User question
- Schema overview
- System prompt (domain context)

**Output:**
- Plan object with ordered Step objects (DAG structure with dependencies)
- Each step has: goal, expected_inputs, expected_outputs

### Executor (`execution/executor.py`)

Runs generated Python code in a sandboxed environment.

**Security:**
- Subprocess isolation
- Configurable timeout
- Import whitelist
- No filesystem access outside designated paths

**Available to Code:**
- `db`, `db_<name>` - SQL database connections (SQLAlchemy engines)
- `file_<name>` - File source paths (for pandas read functions)
- `store` - DataStore for persistence
- `pd`, `np` - pandas, numpy

### SQL Transpiler (`catalog/sql_transpiler.py`)

Cross-dialect SQL compatibility layer using SQLGlot.

**How it works:**
1. LLM generates SQL in the **canonical dialect** (PostgreSQL)
2. SQLGlot transpiles to the target database's SQL dialect automatically

**Supported dialects:**

| SQLAlchemy Driver | SQLGlot Dialect |
|-------------------|-----------------|
| sqlite | sqlite |
| postgresql/psycopg2 | postgres |
| mysql/pymysql | mysql |
| duckdb | duckdb |
| snowflake | snowflake |
| bigquery | bigquery |
| redshift | redshift |
| mssql/pyodbc | tsql |
| oracle | oracle |
| clickhouse | clickhouse |
| databricks | databricks |
| spark | spark |
| trino | trino |
| presto | presto |
| awsathena | presto |
| hive | hive |

### Context Management (`context.py`)

Monitors and compacts session context to stay within token budgets.

**ContextStats** tracks:
- `total_tokens`, broken down by: scratchpad, state variables, table metadata, artifacts
- Largest items for targeted compaction
- Thresholds: WARNING at 50K tokens, CRITICAL at 100K tokens

**ContextCompactor** - Automatic compaction when context reaches critical size:
- Summarizes old scratchpad entries
- Samples large tables (MAX_TABLE_ROWS: 1,000)
- Truncates long narratives (MAX_NARRATIVE_CHARS: 2,000)

### Proof Tree (`proof_tree.py`)

Visualization of fact resolution derivation chains.

**ProofNode:**
- `name`, `description`, `value`, `source`, `confidence`
- `status`: PENDING → RESOLVING → RESOLVED / FAILED / CACHED
- `children`: sub-facts forming the derivation
- `query`: SQL or code used to resolve
- `result_summary`: intermediate result text

Rendered as a Rich Tree in the REPL or interactive D3 DAG in the Web UI.

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

### Storage Layer (`storage/`)

| Module | Purpose |
|--------|---------|
| `datastore.py` | Session data persistence (DataFrames, state, scratchpad) |
| `registry.py` | Central registry for tables and artifacts (TableRecord, ArtifactRecord) |
| `registry_datastore.py` | Registry-aware DataStore integration |
| `history.py` | Execution history tracking and retrieval |
| `learnings.py` | Learning and rule storage |
| `facts.py` | Fact storage and retrieval |
| `duckdb_pool.py` | ThreadLocal DuckDB connection pool |
| `monitors.py` | Monitoring and alert storage |
| `bookmarks.py` | Session bookmark management |
| `parquet_store.py` | Parquet file storage backend |
| `session_store.py` | Session persistence |

**ConstatRegistry** (`registry.py`): Central registry tracking all tables and artifacts across sessions. Records include user_id, session_id, description, row_count, columns, and provenance (role_id, is_published, is_final_step).

### SchemaManager (`catalog/schema_manager.py`)

Provides schema information to the LLM for SQL databases, NoSQL databases, and file-based data sources.

**Unified Data Source Support:**

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

  # CSV file
  web_metrics:
    type: csv
    path: data/metrics.csv

  # JSON file
  clickstream:
    type: json
    path: data/events.json

  # Parquet file (supports s3://, https://)
  transactions:
    type: parquet
    path: s3://bucket/data/transactions.parquet
```

**Schema introspection by type:**
- SQL: Uses SQLAlchemy inspector for table/column metadata
- NoSQL: Samples documents to infer schema (connectors in `catalog/nosql/`)
- Files: Reads sample rows to infer column names, types, and sample values (`catalog/file/connector.py`)

All data sources appear uniformly in the schema overview and vector search.

**NoSQL Connectors** (`catalog/nosql/`):

| Connector | Backend |
|-----------|---------|
| `mongodb.py` | MongoDB |
| `dynamodb.py` | AWS DynamoDB |
| `elasticsearch.py` | Elasticsearch |
| `cosmosdb.py` | Azure Cosmos DB |
| `cassandra.py` | Apache Cassandra |
| `firestore.py` | Google Firestore |

**FileConnector** (`catalog/file/connector.py`): Handles CSV, JSON, JSONL, Parquet, and Arrow/Feather files. Schema inference from samples, row count estimation, metadata generation for vector embeddings, support for local and remote paths.

### DiscoveryTools (`discovery/tools.py`)

Provides on-demand schema, API, and document discovery via tool calling.

**Automatic Mode Selection:**

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
| Skills | `list_skills`, `get_skill`, `search_skills` |

**Extended Discovery Modules:**

| Module | Purpose |
|--------|---------|
| `schema_tools.py` | SchemaDiscoveryTools for database schema |
| `api_tools.py` | APIDiscoveryTools for external APIs |
| `doc_tools/` | DocumentDiscoveryTools package (see sub-modules below) |
| `doc_tools/_core.py` | Document loading, chunking, and refresh orchestration |
| `doc_tools/_access.py` | Document listing and access (list_documents, get_document, search_documents) |
| `doc_tools/_transport.py` | Transport abstraction: file, HTTP, S3, FTP, SFTP, inline |
| `doc_tools/_mime.py` | MIME type detection and normalization |
| `doc_tools/_crawler.py` | BFS link-following crawler for HTML/markdown documents |
| `doc_tools/_file_extractors.py` | Binary format extraction: PDF, DOCX, XLSX, PPTX |
| `fact_tools.py` | FactResolutionTools |
| `skill_tools.py` | SkillManager for skill discovery |
| `unified_discovery.py` | Unified discovery interface |
| `vector_store.py` | DuckDB VSS vector embeddings, entity storage, glossary, clustering |
| `glossary_generator.py` | LLM-powered glossary generation + physical resource resolution |
| `concept_detector.py` | Concept detection in queries |
| `entity_extractor.py` | Named entity extraction (spaCy NER + schema/API patterns) |
| `relationship_extractor.py` | Two-phase SVO extraction: spaCy candidates → LLM refinement with Cypher verb vocabulary |

### SkillManager (`discovery/skill_tools.py`, `core/skills.py`)

Manages domain-specific skill modules following the [Agent Skills](https://agentskills.io) open standard.

**Skill Structure:**

```
.constat/{user_id}/skills/{skill-name}/
├── SKILL.md          (required - YAML frontmatter + Markdown)
├── scripts/          (optional executable code)
├── references/       (optional documentation)
└── assets/           (optional templates, icons)
```

**SKILL.md Format:**

```markdown
---
name: financial-analysis
description: Specialized instructions for financial data analysis
allowed-tools:
  - Read
  - Grep
  - list_tables
  - get_table_schema
user-invocable: false
---

# Financial Analysis Skill

## Key Concepts
- Revenue recognition principles
- Common financial metrics (Gross Margin, EBITDA, etc.)
...
```

**SKILL.md Frontmatter fields:**
- `name`, `description`, `allowed-tools`, `context`, `agent`, `model`
- `disable-model-invocation` - prevent LLM from auto-selecting this skill
- `user-invocable` - can be triggered directly by user
- `argument-hint` - usage hint for invocation

**Discovery Paths (in order of precedence):**
1. **Project skills**: `.constat/skills/` in the project directory
2. **User skills**: `.constat/{user_id}/skills/`
3. **Global skills**: `~/.constat/skills/` in the user's home directory
4. **Config-specified paths**: Additional paths defined in `config.yaml`

**Link Following (Lazy):** Skills can reference additional files via markdown links. Links are discovered on load but content is NOT fetched until needed. Supports relative paths and URLs with caching.

### Roles (`core/roles.py`)

User-defined personas that customize system prompts.

**Role dataclass:**
- `name` - Role identifier
- `prompt` - Text appended to system prompt
- `description` - Human-readable description

**RoleManager:** Loads roles from `.constat/{user_id}/roles.yaml`. Manages multiple roles with one `active_role` at a time.

```yaml
# .constat/default/roles.yaml
roles:
  - name: financial-analyst
    description: Focus on financial metrics and reporting
    prompt: |
      You are a senior financial analyst. Focus on revenue trends,
      margin analysis, and financial KPIs.
  - name: data-engineer
    description: Focus on data quality and pipeline concerns
    prompt: |
      You are a data engineer. Focus on data quality, schema design,
      and pipeline efficiency.
```

### Commands System (`commands/`)

Structured command framework for interactive REPL commands.

| Module | Commands |
|--------|----------|
| `data.py` | `/tables`, `/show`, `/query`, `/code`, `/artifacts`, `/export`, `/download-code` |
| `session_cmds.py` | `/state`, `/reset`, `/facts`, `/context`, `/preferences`, `/learnings`, `/rules`, `/roles`, `/skills` |
| `sources.py` | Data source management |
| `help.py` | `/help` |

**CommandContext:** Provides access to session state (has_datastore, has_plan).

**CommandResult types:** `TextResult`, `TableResult` with format specifications.

**Command Registry** (`registry.py`): Registration and dispatch of commands by name.

### APICatalog (`catalog/api_catalog.py`)

Provides API operation metadata for external services.

**Supported API Types:**

| Type | Discovery | Example |
|------|-----------|---------|
| GraphQL | Auto-introspection | Countries API |
| OpenAPI | Spec parsing (URL, file, inline) | Petstore API |

**API Executor** (`catalog/api_executor.py`): Executes API operations discovered via GraphQL/OpenAPI. Separate from APICatalog which handles metadata.

**API Schema Manager** (`catalog/api_schema_manager.py`): Advanced API schema management beyond APICatalog's basic metadata.

### API Detection & Summarization (`api/`)

Higher-level API layer wrapping Session for structured consumers.

| Module | Purpose |
|--------|---------|
| `impl.py` | ConstatAPIImpl - wraps Session, converts dicts to frozen dataclasses |
| `protocol.py` | ConstatAPI interface definition |
| `types.py` | API type definitions |
| `factory.py` | Provider factory |
| `summarization.py` | Summarization of facts, plans, sessions, tables |
| `learning.py` | Learning integration |
| `detection/` | Display overrides, NL correction detection |

### MetadataPreloadCache (`catalog/preload_cache.py`)

Caches relevant table metadata for faster session startup.

**Configuration:**

```yaml
context_preload:
  seed_patterns:
    - "sales"
    - "customer"
    - "revenue"
  similarity_threshold: 0.3
  max_tables: 50
```

On `/refresh`, seed patterns match against table/column names via vector similarity. Results cached to `.constat/metadata_preload.json` and loaded directly into context on session start.

### FactResolver (`execution/fact_resolver.py`)

Resolves facts with full provenance for auditable mode.

**Fact Sources and Confidence:**

| Source | Confidence | Example |
|--------|------------|---------|
| CACHE | Preserved | Previously resolved fact |
| CONFIG | 1.0 | Config value |
| DATABASE | 1.0 | Query result |
| DOCUMENT | 0.9 | Document excerpt |
| API | 1.0 | External API result |
| LLM_KNOWLEDGE | Parsed (default 0.6) | World knowledge |
| LLM_HEURISTIC | Parsed (default 0.6) | Industry standard |
| RULE | 1.0 | Registered function |
| DERIVED | min(dependencies) | Computed from other facts |
| SUB_PLAN | min(dependencies) | Multi-step derivation |
| USER_PROVIDED | 1.0 | User stated in follow-up |
| UNRESOLVED | 0.0 | Could not resolve |

**Tier2Strategy** - LLM assessment for unresolved facts:
- DERIVABLE: Can be computed from 2+ inputs
- KNOWN: LLM can provide directly
- USER_REQUIRED: Needs human input

**User-Provided Facts:**

When facts are unresolved, users can provide them in natural language:

```python
session.provide_facts("There were 1 million people at the march")
# → Extracts: march_attendance = 1000000
# → Added to cache with source: USER_PROVIDED
# → Resolution re-attempted
```

REPL commands: `/unresolved` to view missing facts, `/facts <text>` to provide them.

### Learning System (`learning/compactor.py`, `learning/exemplar_generator.py`, `storage/learnings.py`)

Accumulates raw learnings from user corrections, promotes them to rules, and generates fine-tuning exemplars.

**LearningCompactor:**
- Analyzes patterns in raw learnings
- Creates generalized rules when sufficient similar learnings accumulate

**CompactionResult:**
- `rules_created`, `rules_strengthened`, `rules_merged`
- `learnings_archived`, `learnings_expired`
- `groups_found`, `skipped_low_confidence`, `errors`

**ExemplarGenerator:**
- Generates fine-tuning conversation pairs from rules, glossary terms, and relationships
- Three coverage levels: `minimal` (high-confidence rules), `standard` (+ approved glossary), `comprehensive` (+ all relationships)
- Outputs OpenAI messages JSONL and Alpaca JSONL formats
- Batch sizes: rules=10, glossary=10, relationships=15
- Exemplar run metadata tracked in `LearningStore`

**Configuration:**
- `CONFIDENCE_THRESHOLD`: 0.60
- `AUTO_COMPACT_THRESHOLD`: 50 unpromoted learnings
- `MIN_GROUP_SIZE`: 2 learnings to form a rule

### Email Integration (`email.py`)

SMTP-based email with Markdown rendering and attachments.

- `SensitiveDataError` for unauthorized sensitive data emailing
- `SensitivityChecker` type for context-aware sensitivity checks
- MIME multipart with CSS-styled HTML from Markdown
- Base64-encoded attachments for results/artifacts

### FeedbackDisplay (`repl/feedback.py`)

Rich-based terminal UI for real-time feedback (~110KB).

**Events Displayed:**
- Plan overview with step checklist
- Step start/complete/error
- Retry indicators
- Timing information
- Tables created

### InteractiveREPL (`repl/interactive.py`)

Interactive command loop for exploration (~101KB).

**Features:**
- Automatic clarification prompts for ambiguous questions
- Plan approval workflow with suggest/reject options
- Follow-up suggestions after analysis completion
- Schema-aware typeahead completions
- Session history and resume capability

### TextualREPL (`textual_repl.py`)

Alternative Textual-based REPL (~5,375 lines) with:
- Persistent status bar
- Rich console rendering (panels, syntax highlighting, tables, markdown, trees)
- Async worker pattern for non-blocking execution
- Modal screens for interactions
- Keyboard bindings and follow-up suggestions

**REPL Commands:**
- `/tables` - Show available tables
- `/show <table>` - Display table contents
- `/query <sql>` - Run SQL on datastore
- `/code [step]` - Show generated code
- `/state` - Inspect session state
- `/facts` - Show cached facts from session
- `/refresh` - Refresh metadata, documents, and preload cache (incremental)
- `/context` - Show context token usage
- `/compact` - Manually compact context
- `/user [name]` - Show/set current user
- `/save <name>` - Save plan for replay
- `/share <name>` - Save as shared plan
- `/plans` - List saved plans
- `/replay <name>` - Replay a saved plan
- `/history` - List previous sessions
- `/resume <id>` - Continue a session
- `/prove` - Switch to auditable mode
- `/roles` - List/switch roles
- `/skills` - List available skills
- `/learnings` - View stored learnings
- `/rules` - View compacted rules
- `/export` - Export results
- `/artifacts` - List generated artifacts

## Web UI (`constat-ui/`)

React 18 SPA built with TypeScript, Vite, and Tailwind CSS.

**Key dependencies:**
- **State**: Zustand
- **Data fetching**: TanStack React Query
- **Tables**: TanStack React Table
- **Charts**: Plotly.js (via react-plotly.js)
- **Graphs**: D3 + d3-dag (proof DAG visualization)
- **Markdown**: react-markdown + remark-gfm
- **Code**: react-syntax-highlighter
- **Auth**: Firebase
- **UI**: Headless UI + Heroicons

Communicates with the server via REST API + WebSocket for real-time streaming.

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
Session                    FeedbackHandler            Display (REPL / WebSocket)
   │                              │                          │
   │  emit(step_start)            │                          │
   │─────────────────────────────▶│                          │
   │                              │   step_start(...)        │
   │                              │─────────────────────────▶│
   │                              │                          │ REPL: Rich panel
   │                              │                          │ Web: WS message
   │                              │                          │
   │  emit(generating)            │                          │
   │─────────────────────────────▶│                          │
   │                              │   step_generating(...)   │
   │                              │─────────────────────────▶│
   │                              │                          │
   │  emit(step_complete)         │                          │
   │─────────────────────────────▶│                          │
   │                              │   step_complete(...)     │
   │                              │─────────────────────────▶│
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
3. **Task-Type Routing** - Route tasks to specialized models (SQLCoder for SQL, haiku for summaries)
4. **Automatic Escalation** - Try local/cheap models first, escalate to cloud on failure
5. **Batch Resolution** - Resolve multiple facts in one LLM call
6. **Context Compaction** - Summarize old scratchpad entries, sample large tables
7. **Parallel Execution** - DAG-based step scheduling (up to 5 concurrent steps)
8. **Parallel Fact Resolution** - Resolve independent facts concurrently (3-5x speedup)
   - Top-level assumed facts have no dependencies → resolve in parallel
   - Rate-limited to avoid API throttling (semaphore + RPM tracking)
   - Sub-proofs resolved recursively with parallelization at each level

### TaskRouter (`providers/router.py`)

Routes tasks to appropriate models with automatic escalation on failure.

**Purpose:**
- Maps task types to ordered lists of models (escalation chains)
- Tries each model in order until success
- Tracks escalation statistics for observability
- Enables local-first with cloud fallback pattern

**Configuration:**

```yaml
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514

  task_routing:
    sql_generation:
      models:
        - provider: ollama
          model: sqlcoder:7b
        - model: claude-sonnet-4-20250514

    python_analysis:
      models:
        - model: claude-sonnet-4-20250514
      high_complexity_models:
        - model: claude-opus-4-20250514

    planning:
      models:
        - model: claude-sonnet-4-20250514
```

**Task Type Routing:**

| Task Type | Description | Typical Models |
|-----------|-------------|----------------|
| `planning` | Multi-step plan generation | claude-sonnet, claude-opus |
| `sql_generation` | Text-to-SQL queries | sqlcoder, claude-sonnet |
| `python_analysis` | DataFrame transformations | codellama, claude-sonnet |
| `summarization` | Result synthesis | claude-haiku, llama3.2 |
| `fact_resolution` | Auditable fact derivation | claude-sonnet |

**Escalation Flow:**

```
Task Request
     │
     ▼
┌─────────────────────────────────────┐
│  Try Model 1 (e.g., ollama/sqlcoder)│
└─────────────────┬───────────────────┘
                  │
          Success? ──Yes──▶ Return result
                  │
                 No
                  │
                  ▼
┌─────────────────────────────────────┐
│  Log escalation event               │
│  Try Model 2 (e.g., claude-sonnet)  │
└─────────────────┬───────────────────┘
                  │
          Success? ──Yes──▶ Return result
                  │
                 No (all models exhausted)
                  │
                  ▼
            Return failure
```

**Supported Providers:**
- `anthropic` - Anthropic Claude
- `openai` - OpenAI GPT
- `gemini` - Google Gemini
- `grok` - xAI Grok
- `mistral` - Mistral AI (Large, Small, Nemo)
- `codestral` - Mistral code-specialized models
- `ollama` - Local Ollama
- `together` - Together AI
- `groq` - Groq