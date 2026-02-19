# Proof-Solving Process Diagram

## High-Level Flow

```
User Question
    │
    ▼
┌─────────────────┐
│  Intent Classify │  (QUERY / PLAN_NEW / CONTROL)
└────────┬────────┘
         │ QUERY
         ▼
┌─────────────────┐
│  Auditable Mode  │  Session.solve() → _solve_auditable()
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│                 PLANNING PHASE                   │
│                                                  │
│  1. Build context (schemas, docs, APIs, cache)   │
│  2. LLM generates formal derivation plan:        │
│     • QUESTION: restated problem                 │
│     • PREMISES: base facts to retrieve (P1..Pn)  │
│     • INFERENCES: derived computations (I1..In)  │
│     • CONCLUSION: final answer                   │
│  3. Parse plan into premises + inferences        │
│  4. Validate structure                           │
│     ├─ Pass → continue                           │
│     └─ Fail → retry (up to 3×) with feedback     │
└────────┬────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│              DAG CONSTRUCTION                    │
│                                                  │
│  parse_plan_to_dag() → ExecutionDAG              │
│                                                  │
│  Premises → leaf FactNodes (no dependencies)     │
│  Inferences → internal FactNodes (with deps)     │
│                                                  │
│  Topological sort → execution levels:            │
│    Level 0: [P1, P2, P3]     ← all premises     │
│    Level 1: [I1, I2]         ← depend on L0     │
│    Level 2: [I3]             ← depends on L0+L1 │
│    Level 3: [CONCLUSION]     ← final answer     │
└────────┬────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│            PARALLEL EXECUTION                    │
│                                                  │
│  DAGExecutor processes level-by-level:           │
│                                                  │
│  ┌─── Level 0 (parallel) ───────────────────┐   │
│  │  P1: database  P2: document  P3: api     │   │
│  │    ↓              ↓            ↓          │   │
│  │  SQL query    doc read     API call       │   │
│  │    ↓              ↓            ↓          │   │
│  │  DataFrame    text value   JSON data      │   │
│  │  → datastore  → cache     → datastore    │   │
│  └──────────────────────────────────────────┘   │
│                     │                            │
│                     ▼                            │
│  ┌─── Level 1 (parallel) ───────────────────┐   │
│  │  I1: join(P1, P2)    I2: filter(P3)      │   │
│  │    ↓                    ↓                 │   │
│  │  LLM generates       LLM generates       │   │
│  │  Python code         Python code          │   │
│  │    ↓                    ↓                 │   │
│  │  Execute with        Execute with         │   │
│  │  store, pd, np       store, pd, np        │   │
│  │    ↓                    ↓                 │   │
│  │  Validate result     Validate result      │   │
│  │  → datastore         → datastore          │   │
│  └──────────────────────────────────────────┘   │
│                     │                            │
│                     ▼                            │
│  ┌─── Level N ──────────────────────────────┐   │
│  │  CONCLUSION: summarize(I1, I2)            │   │
│  │    ↓                                      │   │
│  │  Final derivation with confidence         │   │
│  └──────────────────────────────────────────┘   │
└────────┬────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│               COMPLETION                         │
│                                                  │
│  • Build derivation report with provenance       │
│  • Propagate confidence scores                   │
│  • Emit answer_ready + proof_complete events     │
│  • Persist facts to cache and datastore          │
└─────────────────────────────────────────────────┘
```

## Premise Resolution by Source Type

```
┌──────────────┬─────────────────────────────────────────┐
│ Source        │ Resolution Strategy                      │
├──────────────┼─────────────────────────────────────────┤
│ cache        │ Check fact cache → datastore tables      │
│ database     │ LLM → SQL → execute (retry ×7) → table  │
│ api          │ Fact resolver → GraphQL/REST call         │
│ document     │ doc_read() → text content                 │
│ llm_knowledge│ LLM general knowledge (confidence 0.7)   │
│ user         │ Prompt user for value                     │
└──────────────┴─────────────────────────────────────────┘
```

## Inference Execution Detail

```
Inference Node
    │
    ▼
Build exec context
  • scalars: resolved premise values
  • tables: DataFrames in datastore
  • db connections: for referenced tables
  • API functions: for API sources
    │
    ▼
LLM generates Python code
  • globals: store, pd, np, llm_map, db conns
    │
    ▼
Execute code ──────────────────┐
    │                          │ error
    ▼                          ▼
Validate result         Retry (up to 7×)
  • result exists?        with error feedback
  • DataFrame not empty?        │
  • no all-null columns?        │
  • row count OK?          ─────┘
    │
    ▼ pass
Save to datastore
Resolve node with value + confidence
```

## Confidence Propagation

```
P1 (0.95) ──┐
             ├──→ I1: min(own=0.90, P1=0.95, P2=0.85) = 0.85
P2 (0.85) ──┘
                        │
P3 (0.90) ──────────────┤
                        ▼
              CONCLUSION: min(own, I1=0.85, P3=0.90) = 0.85
```

## Failure Handling

```
Node fails
    │
    ├──→ Mark node FAILED with error
    ├──→ All transitive dependents → BLOCKED
    ├──→ Other independent branches continue
    └──→ Return partial result with failure point highlighted

Cancellation
    │
    ├──→ Check is_cancelled() between levels
    ├──→ Cancel remaining futures
    ├──→ Preserve completed facts
    └──→ Return ExecutionResult(cancelled=True)
```

## Key Files

| File | Role |
|---|---|
| `session/_solve.py` | Entry point: `Session.solve()` |
| `session/_auditable.py` | Plan generation, parsing, validation |
| `session/_dag.py` | Node execution logic (SQL, code gen, APIs) |
| `execution/dag.py` | DAG structures, topological sort, DAGExecutor |
| `execution/parallel_scheduler.py` | ExecutionContext, cancellation |
| `proof_tree.py` | ProofTree/ProofNode visualization |
| `execution/fact_resolver/` | Tiered fact resolution |
