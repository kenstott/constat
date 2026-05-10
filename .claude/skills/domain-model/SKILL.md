---
name: domain-model
description: Constat domain knowledge and conceptual model. Auto-triggers when working with core business logic.
---

# Constat Domain Model

## What Constat Does
Constat is a data analysis system that takes natural language questions, plans analytical steps, executes them against data sources, and synthesizes results with proof DAGs.

## Domain Structure
- Recursive: domains contain sub-domains (a session domain, inherits from the user domain, the user domain inherits from system domain and its active domains). Every domain can contain other child domains.
- Config defines available system domains (e.g., `sales-analytics`, `hr-reporting`)
- Each domain has schemas, tables, relationships, glossaries, tests, agents and skills.

## Builder Pipeline
```
Config → Session → Plan → Execute → Synthesize
```
1. **Config** — data sources, domains, LLM providers (`core/config.py`)
2. **Session** — orchestrates the pipeline (12+ mixins in `session/`)
3. **Plan** — LLM generates analytical steps (`execution/planner.py`)
4. **Execute** — runs SQL/code against data (`execution/query_engine.py`)
5. **Synthesize** — combines results into narrative with proof DAG

## Session Mixin Composition
The Session class in `constat/session/__init__.py` uses 12+ mixins:
- `_solve.py` — main solving pipeline
- `_dag.py` — DAG proof construction
- `_clarify.py` — clarification questions
- `_context.py` — context management
- Each mixin adds a focused set of methods

## Storage Layers
| Layer | Class | Purpose |
|-------|-------|---------|
| Session | DuckDBSessionStore | Per-session data, tables, metadata |
| Vector | DuckDBVectorBackend | Embeddings, FTS, BM25, reranking |
| Relational | RelationalStore | Entities, glossary, relationships, NER cache |

## DAG Proof Inferences
- Steps are numbered positively (1, 2, 3...)
- Inference nodes use negative step numbers: I1→-1, I5→-5
- DAG edges connect evidence steps to inference conclusions

## Key Abstractions
- `create_api()` — factory entry point for all API operations
- Schema federation via DuckDB `ATTACH` (source DBs mounted read-only)
- Arrow zero-copy for DataFrame ↔ DuckDB transfers
- Task routing selects LLM model per task type (classify → Haiku, reason → Sonnet/Opus)
