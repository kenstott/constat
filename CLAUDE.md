CRITICAL: Never add fallback values or silent error handling. This has caused repeated production issues.
CRITICAL: We are currently in version 1 development. Never add migrations.
CRITICAL:  Maximum brevity. No pleasantries. No explanations unless asked. Code and facts only.

# Architecture

## Backend (Python — `constat/`)
| Module | Purpose | Key files |
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

## Frontend (TypeScript/React — `constat-ui/src/`)
| Module | Purpose |
|--------|---------|
| `api/` | HTTP/WebSocket clients |
| `components/` | React components (artifacts, auth, common, conversation, proof, layout) |
| `store/` | Zustand stores (session, auth, artifact, glossary, ui, proof, test, toast) |
| `types/` | TypeScript type definitions |
| `hooks/` | Custom React hooks |

## Storage
- Session data: DuckDB file per session (`session.duckdb`)
- Vector store: separate DuckDB (embeddings, FTS, BM25)
- Relational store: SQLite (entities, glossary, relationships)
- No Parquet files — all native DuckDB tables

## Key Patterns
- Session uses mixin composition (12+ mixins in `session/`)
- `create_api()` factory in `constat/api/__init__.py` — entry point for all API operations
- DuckDB `ATTACH` for source database federation
- Arrow zero-copy for DataFrame ↔ DuckDB
- DAG proof inferences use negative step numbers (I1→-1, I5→-5)

# Verification Commands
- Backend tests: `python -m pytest tests/ -x -q`
- Frontend build: `cd constat-ui && npm run build`
- Frontend lint: `cd constat-ui && npm run lint`
- Type check: `cd constat-ui && npx tsc --noEmit`
- Server: `python -m constat.server -c demo/config.yaml`
- Demo config: `demo/config.yaml` (domains: sales-analytics, hr-reporting)
- Server logs: `.logs/server.log`, `.logs/ui.log`
- Session data: `.constat/{user-id}/sessions/{timestamped-dir}/` (session.duckdb, state.json, proof_facts.json, etc.)

# Module Boundaries (for parallel work)
- Backend and frontend are fully independent — safe to work in parallel
- `constat/server/routes/` depends on `constat/session/` and `constat/storage/`
- `constat/session/` depends on `constat/execution/`, `constat/storage/`, `constat/providers/`
- `constat/execution/` depends on `constat/storage/`, `constat/llm/`
- `constat-ui/src/store/` depends on `constat-ui/src/api/` and `constat-ui/src/types/`
- `constat-ui/src/components/` depends on `constat-ui/src/store/` and `constat-ui/src/hooks/`

# Swarm Mode (Self-Claim)

All agents operate autonomously. No lead assignment required.

## After completing any task:
1. Call `TaskList` to see all tasks
2. Find the first task (lowest ID) that is **unblocked** AND has **no owner**
3. Call `TaskUpdate` to set yourself as owner and status to "in_progress"
4. Begin work immediately

## Rules:
- Never wait for assignment — self-claim
- Prefer lowest task ID among unblocked/unowned tasks
- If no tasks available, report idle and stop
- Respect module boundaries (see below) — only claim tasks in your domain
- Run the verification command for your module when done
- After completing a task, loop back to step 1 above — always pull the next task

## TeammateIdle Pattern:
When a teammate finishes and becomes idle, they must immediately self-claim the next available task rather than reporting back to a lead. The lead spawns initial teammates; after that, agents are self-sustaining.

# Teammate Spawn Context
When spawning teammates for parallel work, include:
1. Which module(s) they own (from Module Boundaries above)
2. Specific files to read first
3. Verification command to run when done
4. What NOT to touch (other teammate's modules)
