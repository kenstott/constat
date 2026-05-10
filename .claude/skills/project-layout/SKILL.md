---
name: project-layout
description: Module ownership map and architecture overview. Auto-triggers when navigating or modifying project structure.
---

# Project Layout

## Backend (`constat/`)
| Module | Purpose | Key Files |
|--------|---------|-----------|
| `api/` | API factory, protocol, impl | `__init__.py` |
| `catalog/` | Schema discovery, API catalog | `schema_manager.py` |
| `commands/` | Shared REPL/UI command handlers | `data_commands.py` |
| `core/` | Config, models, types | `config.py`, `models.py` |
| `discovery/` | Schema/API/doc/fact discovery | `vector_store.py` |
| `execution/` | Planning, code execution | `planner.py`, `query_engine.py` |
| `learning/` | Rule compaction, fine-tune | `compactor.py` |
| `llm/` | LLM primitives (enrich/score/summarize) | `__init__.py` |
| `prompts/` | Prompt/template loading | `loader.py` |
| `providers/` | LLM provider integrations | `base.py`, `anthropic_provider.py` |
| `server/` | FastAPI server, routes, websocket | `app.py`, `routes/` |
| `session/` | Session orchestration (12+ mixins) | `__init__.py`, `_solve.py`, `_dag.py` |
| `storage/` | DuckDB session store, history | `duckdb_session_store.py` |
| `testing/` | Test runner, golden questions | `runner.py` |

## Frontend (`constat-ui/src/`)
| Module | Purpose |
|--------|---------|
| `api/` | HTTP/WebSocket clients |
| `components/` | React components (artifacts, auth, common, conversation, proof, layout) |
| `store/` | Zustand stores (session, auth, artifact, glossary, ui, proof, test, toast) |
| `types/` | TypeScript type definitions |
| `hooks/` | Custom React hooks |

## Session Mixin Composition
`constat/session/__init__.py` composes 12+ mixins:
- `_solve.py` — solving pipeline
- `_dag.py` — DAG proof construction
- `_clarify.py` — clarification questions
- `_context.py` — context management
- Each mixin adds methods to the Session class

## Storage Architecture
| Layer | Implementation | File |
|-------|---------------|------|
| Session data | DuckDBSessionStore | `storage/duckdb_session_store.py` |
| Vector/FTS/BM25 | DuckDBVectorBackend | `storage/duckdb_backend.py` |
| Entities/glossary | RelationalStore | `storage/relational.py` |

## Entry Points
- `create_api()` factory in `constat/api/__init__.py`
- Server: `python -m constat.server -c demo/config.yaml`
- Demo config: `demo/config.yaml` (domains: sales-analytics, hr-reporting)

## Dependency Graph
- `server/routes/` → `session/`, `storage/`
- `session/` → `execution/`, `storage/`, `providers/`
- `execution/` → `storage/`, `llm/`
- Frontend: `components/` → `store/` → `api/` → `types/`