# Requirements

## Error Handling & Reliability
- **REQ-001** (2026-03-29): Never add fallback values or silent error handling — all errors must be explicit and fail-fast.
- **REQ-002** (2026-03-29): No migrations in version 1 development — schema created on first access via CREATE TABLE IF NOT EXISTS.

## Data & Storage
- **REQ-003** (2026-03-29): Session data stored in single DuckDB file per session (`session.duckdb`) with all metadata (_constat_* tables) in same file.
- **REQ-004** (2026-03-29): Vector store and session store use separate DuckDB connections and files — no shared connection.
- **REQ-005** (2026-03-29): Arrow zero-copy used for DataFrame ↔ DuckDB conversions to avoid serialization overhead.
- **REQ-006** (2026-03-29): DuckDB DROP VIEW IF EXISTS raises CatalogException if name is a TABLE — use try/except in code.
- **REQ-007** (2026-03-29): ON CONFLICT DO UPDATE used for upserts — no savepoints needed.
- **REQ-008** (2026-03-29): Learnings and facts stored in DuckDB backend (`learnings` and `facts` tables in user vault).
- **REQ-009** (2026-03-29): Session history, steps, plans, and inferences stored in DuckDB (session.duckdb).
- **REQ-010** (2026-03-29): All per-user file paths use `constat.core.paths.user_vault_dir(data_dir, user_id)` → `{data_dir}/{uid}.vault/`.
- **REQ-012** (2026-03-29): DuckDB extension guard via `_get_extension_init_sql()` — INSTALL only on first process connection, LOAD after.
- **REQ-013** (2026-03-29): All per-user paths use vault directory structure (`.constat/{uid}.vault/`) managed by `user_vault_dir()` helper.
- **REQ-014** (2026-03-29): Files reserved for human-editable configuration (YAML) — runtime data goes in DB (data sources, permissions, skills, agents, golden questions, entity resolution queries, fine-tune artifacts).

## Domain Model
- **REQ-015** (2026-03-29): Every resource must have an explicit domain — `domain=None` is invalid on stored objects.
- **REQ-016** (2026-03-29): New resources start in session domain (`session_id`) by default.
- **REQ-017** (2026-03-29): Resources can move between any domains with permission — no fixed promotion hierarchy.
- **REQ-018** (2026-03-29): "draft" is an auto-generated status marker, not a domain concept.
- **REQ-019** (2026-03-29): Chunks inherit domain from their data source via FK to data_sources table.
- **REQ-020** (2026-03-29): Glossary lives in DB only — no glossary in YAML config files.
- **REQ-021** (2026-03-29): Five domain tiers valid: SYSTEM, SYSTEM_DOMAIN, USER, USER_DOMAIN, SESSION.
- **REQ-022** (2026-03-29): GlossaryTerm model validation raises ValueError if domain is None via __post_init__.
- **REQ-023** (2026-03-29): Glossary generation respects domain tier — drafts use session_id, approved terms move to target domain.
- **REQ-024** (2026-03-29): Chunk domain filtering via JOIN to data_sources table — no redundant domain_id column on chunks.

## Split Vector Store & Multi-Tenancy
- **REQ-025** (2026-03-29): Split vector store with system DB (`.constat/vectors.duckdb`) and per-user DBs (`.constat/{uid}.vault/vectors.duckdb`).
- **REQ-026** (2026-03-29): ATTACH system DB as `sys` — v_* TEMP views merge both DBs for transparent reads.
- **REQ-027** (2026-03-29): Writes route by domain tier: `main.*` (user) or `sys.*` (system) via `domain_tier_fn` callback.
- **REQ-028** (2026-03-29): Cross-DB domain moves use SELECT→DELETE→INSERT pattern (not UPDATE domain_id).
- **REQ-029** (2026-03-29): FTS/BM25 index on main.embeddings only; system chunks use vector search.
- **REQ-030** (2026-03-29): Both system and user DBs MUST have ALL tables from SPLIT_TABLES even if empty — no fallback views.
- **REQ-031** (2026-03-29): Empty table schema cloned via `CREATE TABLE sys.X AS SELECT * FROM main.X WHERE false`.
- **REQ-032** (2026-03-29): `_init_schema()` always runs all CREATE TABLE statements — no early-return optimization.
- **REQ-033** (2026-03-29): System DB path `.constat/vectors.duckdb` same as warmup default — no per-domain system DBs.

## Security & Encryption
- **REQ-034** (2026-03-29): Vault encryption optional via `vault_encrypt` field on ServerConfig (default false).
- **REQ-035** (2026-03-29): User vault encryption uses AES-256-GCM with HKDF-SHA256 key derivation (UserVault class).
- **REQ-036** (2026-03-29): WebAuthn (passkey) support via `constat/server/routes/passkey.py` for registration and authentication.

## Authentication & Authorization
- **REQ-037** (2026-03-29): Token validation order: `auth_disabled` → `admin_token` → local token → Firebase JWT.
- **REQ-038** (2026-03-29): Local auth uses scrypt password hashing with opaque token generation.
- **REQ-039** (2026-03-29): `GET /health` returns `auth.auth_methods: string[]` for frontend auth discovery.
- **REQ-040** (2026-03-29): Frontend local login must wire to `POST /api/auth/login` (username/password → token).
- **REQ-041** (2026-03-29): Frontend local signup must call `POST /api/auth/register` for user self-registration.
- **REQ-042** (2026-03-29): Local login uses username identifier; Firebase uses email — distinct UI forms.
- **REQ-043** (2026-03-29): Microsoft Entra ID (Azure AD) OAuth2/OIDC support via `POST /api/auth/microsoft-login` code-exchange.
- **REQ-044** (2026-03-29): Microsoft config endpoint `GET /api/auth/microsoft-config` returns client_id and tenant_id (no secrets).
- **REQ-045** (2026-03-29): Frontend dynamically renders login page based on available methods from `/health` auth_methods.
- **REQ-046** (2026-03-29): Login page shows "Sign in with Microsoft" button when `microsoft` in auth_methods (Microsoft branding guidelines).
- **REQ-047** (2026-03-29): Frontend `useAuthMethods()` hook fetches `/health` and caches auth methods for session lifetime.
- **REQ-048** (2026-03-29): Local/custom auth (username/password with server-side token) is supported as lightweight alternative for testing environments and simple customer deployments that don't need Firebase. Firebase remains primary auth platform for production use.

## API & Integration
- **REQ-048** (2026-03-29): `create_api()` factory in `constat/api/__init__.py` is entry point for all API operations.
- **REQ-049** (2026-03-29): DuckDB ATTACH used for source database federation.
- **REQ-050** (2026-03-29): DAG proof inferences use negative step numbers (I1→-1, I5→-5).

## Session & Orchestration
- **REQ-051** (2026-03-29): Session orchestration via 12+ mixin composition in `constat/session/` modules.
- **REQ-052** (2026-03-29): `SessionManager.get_user_db(user_id)` returns existing ThreadLocalDuckDB from user's vector store.
- **REQ-053** (2026-03-29): LearningStore constructor accepts `db: ThreadLocalDuckDB | None` — uses directly when provided.
- **REQ-054** (2026-03-29): FactStore follows same DB constructor pattern as LearningStore for backend flexibility.

## Discovery & Search
- **REQ-055** (2026-03-29): Relational store (RelationalStore) manages entities, glossary, relationships, hashes, clusters, NER cache.
- **REQ-056** (2026-03-29): DuckDB vector backend (DuckDBVectorBackend) provides embeddings, FTS, BM25, RRF, reranking.
- **REQ-057** (2026-03-29): Store class composes RelationalStore + DuckDBVectorBackend.
- **REQ-058** (2026-03-29): Vector store (`constat/discovery/vector_store.py`) is thin wrapper over DuckDB backend.

## UI & Frontend
- **REQ-059** (2026-03-29): Frontend modules: `api/` (HTTP/WebSocket clients), `components/` (React), `store/` (Zustand), `types/` (TypeScript), `hooks/` (custom hooks).
- **REQ-060** (2026-03-29): Frontend uses Zustand stores for session, auth, artifact, glossary, ui, proof, test, toast state.
- **REQ-061** (2026-03-29): Glossary subscription events use GraphQL (`glossaryChanged`), NOT WebSocket.
- **REQ-062** (2026-03-29): UI completely GraphQL-driven — no REST.
- **REQ-063** (2026-03-29): Session events (step_complete, plan_ready, entity_update, artifact_created, etc.) delivered via GraphQL subscriptions.
- **REQ-064** (2026-03-29): Session actions (approve, reject, cancel, replan_from, edit_objective, etc.) executed via GraphQL mutations.
- **REQ-100** (2026-04-01): Replace custom artifactStore (createStore.ts) with direct Apollo Client cache usage. Remove artifactStore Zustand store; components use useQuery() directly instead of fetch methods. Real-time updates via apolloClient.writeQuery() or cache.modify() instead of custom set() calls.
- **REQ-101** (2026-04-02): 3-section accordion panel for artifacts: (1) **Artifacts** — published/starred final output (charts, tables, answer), expanded by default, auto-expands during execution; (2) **Debug** — step code, intermediate tables, unpublished artifacts, raw output, collapsed by default for power users; (3) **Context** — sources, glossary, reasoning, configuration, collapsed by default for clean knowledge/config surface. Replaces 2-section design, separating "the answer" / "how it got there" / "what it knows."
- **REQ-102** (2026-04-06): Every source type needs a CRUD UI with operations gated by user rights/permissions.

## Infrastructure & Configuration
- **REQ-065** (2026-03-29): Demo config at `demo/config.yaml` with domains `sales-analytics` and `hr-reporting`.
- **REQ-066** (2026-03-29): Server invoked as `python -m constat.server -c demo/config.yaml` (no server run without config).
- **REQ-067** (2026-03-29): Server logs: `.logs/server.log` and `.logs/ui.log`.
- **REQ-068** (2026-03-29): Session data directory: `.constat/{user-id}/sessions/{timestamped-dir}/` with session.duckdb, state.json, proof_facts.json, etc.
- **REQ-069** (2026-03-29): Backend and frontend fully independent — safe to work in parallel.
- **REQ-070** (2026-03-29): `constat/server/routes/` depends on `constat/session/` and `constat/storage/`.
- **REQ-071** (2026-03-29): `constat/session/` depends on `constat/execution/`, `constat/storage/`, `constat/providers/`.
- **REQ-072** (2026-03-29): `constat/execution/` depends on `constat/storage/`, `constat/llm/`.
- **REQ-073** (2026-03-29): `constat-ui/src/store/` depends on `constat-ui/src/api/` and `constat-ui/src/types/`.
- **REQ-074** (2026-03-29): `constat-ui/src/components/` depends on `constat-ui/src/store/` and `constat-ui/src/hooks/`.

## Testing & Quality
- **REQ-075** (2026-03-29): Backend tests run via `python -m pytest tests/ -x -q` (approximately 7 minutes).
- **REQ-076** (2026-03-29): 860+ tests passing; known pre-existing failures: test_auditable_nlq (LLM-dependent), test_elasticsearch (infra), test_load_pdf_via_http_content_type (SSL).
- **REQ-077** (2026-03-29): Session store tests: `tests/test_duckdb_session_store.py` (49 tests).
- **REQ-079** (2026-03-29): Frontend build via `cd constat-ui && npm run build`.
- **REQ-080** (2026-03-29): Frontend lint via `cd constat-ui && npm run lint`.
- **REQ-081** (2026-03-29): Frontend type check via `cd constat-ui && npx tsc --noEmit`.
- **REQ-103** (2026-04-06): Tests organized in three tiers: `tests/unit/` (pure logic, zero I/O), `tests/integration/` (real services via Docker), `tests/e2e/` (full HTTP via Playwright). New tests must be placed in the correct tier.
- **REQ-104** (2026-04-06): `pytest.skip()` for infrastructure reasons is forbidden. Tests requiring services must start them via Docker in the fixture or call `pytest.fail()` if Docker is unavailable.
- **REQ-105** (2026-04-06): E2E test directory is `tests/e2e/` (Playwright tests requiring running backend + Vite dev server), distinct from `tests/integration/` (service-level tests without HTTP app layer).

## Architecture & Design Patterns
- **REQ-082** (2026-03-29): Learnings table combines corrections + archive (`archived_at IS NOT NULL` = archived).
- **REQ-083** (2026-03-29): JSON TEXT fields for `context`, `scope`, `tags`, `source_learnings` — small payloads, read wholesale.
- **REQ-084** (2026-03-29): Learnings/facts NOT in SPLIT_TABLES — always user-scoped, never replicated to system DB.
- **REQ-085** (2026-03-29): LearningStore uses DuckDB pool directly — no in-memory cache or file-based persistence.
- **REQ-086** (2026-03-29): LearningStore public method signatures stable — backend implementation transparent to callers.
- **REQ-087** (2026-03-29): `get_relevant_rules()` filters confidence in SQL then Python keyword scoring (rule set small).
- **REQ-088** (2026-03-29): FactStore `value` field JSON-serialized to preserve Python types across RPC/persistence.
- **REQ-089** (2026-03-29): Architecture docs in `docs/arch/` ARE the planning documents — update when requirements change, don't implement without planning.

## GraphQL Migration
- **REQ-093** (2026-03-29): All UI data fetching migrates from REST to GraphQL (52Q + 87M + 8S). 5 binary download endpoints stay REST.
- **REQ-094** (2026-03-29): Persisted GraphQL queries (APQ) — client sends SHA-256 hash instead of full query string. Reduces payload size, enables CDN caching, prevents arbitrary query execution in production allowlist mode.
- **REQ-095** (2026-03-29): Apollo is the only frontend state store. Zustand deleted entirely by Phase 10. Server data via queries/mutations/subscriptions, UI state via reactive variables (`makeVar`).
- **REQ-096** (2026-03-29): Each GraphQL migration phase: implement backend schema + resolvers → wire frontend hooks → write tests → delete REST route → next phase.
- **REQ-097** (2026-03-29): Phase 1 auth REST routes (`/auth/*`) deleted in Phase 1b after React consumer migration. Config/permissions REST routes deleted in Phase 4 when `artifactStore` is killed.
- **REQ-098** (2026-03-29): `authStore.ts` file deleted in Phase 10 — 7 non-React modules depend on `getState().getToken()` for auth headers until all Zustand stores are removed.

## User Sources & Configuration
- **REQ-099** (2026-03-30): User source management (list, add, update, remove per-user databases/documents/APIs) must have a UI and be GraphQL-driven. User sources stored in per-user config.yaml files and persist across sessions.

## Documentation & Process
- **REQ-090** (2026-03-29): Maximum brevity in communications — code and facts only, no pleasantries or explanations unless asked.
- **REQ-091** (2026-03-29): New requirements tracked via `requirements-tracker` agent appending to `docs/arch/requirements.md`.
- **REQ-092** (2026-03-29): Requirements capture: stated constraints, features, design decisions — not implementation details, bug reports, or questions.
