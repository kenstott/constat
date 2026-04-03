# Tier 6 — Cleanup, Verification & Documentation

## 1. Chunk Domain Cleanup — Remove Redundant domain_id from Embeddings

## Goal

Complete the domain normalization started in Tier 5. Remove the redundant `domain_id` column from the `embeddings` table, replacing all 88+ references with join-based domain resolution through the `data_sources` table.

## Current State (after Tier 5)

Both columns exist on `embeddings`:
- `domain_id` — direct, redundant (legacy)
- `data_source_id` — FK to `data_sources.domain_id` (normalized)

All new code populates both. Existing queries use `domain_id` directly.

## What Changes

### duckdb_backend.py (~20 function updates)
- `chunk_visibility_filter()` — change `domain_id IN (...)` to JOIN on data_sources
- `add_chunks()` — stop writing `domain_id`, only write `data_source_id`
- `get_domain_chunks()` — join-based filtering
- `clear_domain_embeddings()` — join-based deletion
- All search methods with domain filtering — use join

### relational.py (~15 function updates)
- Entity domain propagation — derive from data_sources join
- Entity visibility filters — join-based
- Glossary-entity domain mismatch detection — join-based
- Document URL filtering — join-based

### vector_store.py
- Remove `domain_id` from embeddings DDL
- Update UNION views in split_store to include data_sources join

### split_store.py
- Update UNION ALL views for embeddings to join through data_sources

## Migration Strategy
- On first access: if `domain_id` column exists and `data_source_id` is populated, backfill any missing `data_source_id` values, then drop `domain_id` column
- Or: simply ignore the column (DuckDB allows unused columns)

## Testing
- Verify all domain-scoped queries return same results via join as via direct column
- Performance: ensure join-based filtering doesn't regress search latency
- Test domain movement: update data_sources.domain_id → verify chunks reflect new domain

---

## 2. Frontend REST Cleanup — sessions.ts Migration

Migrate remaining 1,285 lines of REST client code in `constat-ui/src/api/sessions.ts` to GraphQL. Most functions already have GraphQL equivalents in `constat-ui/src/hooks/`. Remove dead REST functions, update remaining callers to use Apollo hooks.

---

## 3. Verification Gates

### Backend Test Suite
- All tests pass: `python -m pytest tests/ -x -q`
- No files over 1000 lines: `find constat/ -name "*.py" | xargs wc -l | awk '$1 > 1000'`
- No silent error handling: grep for bare `except:` or `except Exception: pass`

### Frontend Test Suite
- All tests pass: `cd constat-ui && npm test`
- Coverage >= 80%: `npx vitest run --coverage`
- Type check clean: `npx tsc --noEmit`
- Build succeeds: `npm run build`

### Integration Verification
- Server starts: `python -m constat.server -c demo/config.yaml`
- Session creation works (default user + authenticated user)
- DuckDB stores load (learnings, facts, history, preferences, monitors, plans)
- YAML→DuckDB one-time import triggers on first access
- DB rename migration (vectors.duckdb → user.duckdb/system.duckdb) works
- AGENT.md files load correctly (tiered: system < domain < user)
- Dynamic login page adapts to configured auth methods
- Microsoft SSO popup flow completes (requires Azure AD app registration)
- Personal accounts persist across sessions
- Deploy CLI: `constat deploy diff --source demo/ --target demo/` produces zero changes

### Source Type Verification
- MCP: mock MCP server → resources listed → documents indexed
- Calendar: Google/Microsoft OAuth → events fetched → indexed
- Drive: folder traversal → files downloaded → content extracted
- SharePoint: site discovery → libraries/lists/calendars/pages indexed

### Regression
- Golden questions pass: `constat test run --domain sales-analytics`
- Existing notebook demo works: `demo/notebook_demo.ipynb`

---

## 4. Documentation Updates

### README.md
- Add new data source types (MCP, Calendar, Drive, SharePoint)
- Add authentication methods (local, Firebase, Microsoft SSO)
- Add personal resources feature
- Add deploy CLI usage
- Update architecture diagram

### docs/config-reference.md (or create if missing)
- DocumentConfig fields: all new fields (calendar, drive, sharepoint, MCP)
- APIConfig fields: allowed_tools, denied_tools
- ServerConfig fields: microsoft_auth_*, google_oauth_*, account_encryption_secret
- Auth configuration examples (local-only, Firebase, Microsoft SSO, all methods)

### docs/arch/ Cleanup
- Mark completed plans: move implemented docs to `docs/arch/completed/`
  - auth.md → completed/
  - db-persistence.md → completed/
  - domains.md (Phase 1-3) → completed/
  - graphql.md → completed/
  - data-source-changes.md → completed/
  - email.md → completed/
  - mcp.md → completed/
  - calendars.md → completed/
  - cloud-drive.md → completed/
  - sp.md → completed/
  - personal.md → completed/
  - deploy.md → completed/
  - handbook.md → completed/
  - agents-as-files.md → completed/
- Keep value-proposition.md in place (reference doc, not implementation plan)

### CHANGELOG.md (create)
- Tier 1: DB rename, LearningStore/FactStore → DuckDB, GraphQL CRUD, frontend migration
- Tier 2: Auth (local/Firebase/Microsoft), data source contract, DB persistence Phase 2, agents as files, 89% frontend coverage
- Tier 3: MCP client, Calendar, Drive, SharePoint sources
- Tier 4: Personal resources, deploy tooling
- Tier 5: Domain handbook, chunk domain normalization

### API Documentation
- GraphQL schema documentation (introspection already available at /api/graphql)
- New endpoints: /api/oauth/*, /api/accounts/*, handbook queries
- Deploy CLI man page or --help output
