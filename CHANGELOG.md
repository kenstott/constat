# Changelog

## Tier 1 — Foundation

- **DB rename**: `vectors.duckdb` to `user.duckdb` (user vault) / `system.duckdb` (root) with one-time migration
- **LearningStore + FactStore to DuckDB**: YAML-backed stores rewritten with SQL backend (learnings, learning_rules, facts, exemplar_runs tables). Same public API. One-time YAML import.
- **GraphQL Skills/Agents CRUD**: 9 new mutations (createSkill, updateSkill, deleteSkill, draftSkill, createSkillFromProof, createAgent, updateAgent, deleteAgent, draftAgent), 1 new query (agent)
- **Frontend REST to GraphQL**: ArtifactContext, ReasoningSection, ProofDAGPanel migrated from REST to Apollo mutations
- **Domains Phase 1-2**: Already complete (GlossaryTerm domain validation, no config glossary)

## Tier 2 — Auth + Data Contract

- **Dynamic login page**: Fetches `/health` auth_methods, conditionally renders local/Firebase/Microsoft options
- **Local self-registration**: `register` mutation, persists to `local_users.yaml`, auto-login
- **Microsoft SSO**: `@azure/msal-browser` popup flow, ID token validation, `loginWithMicrosoft` mutation
- **Split source_resolvers.py**: 1179 to 612+344+304 lines (3 files)
- **Split history.py**: 1061 to 107+994 lines (facade + FileSessionHistory)
- **DuckDB session history**: `DuckDBSessionHistory` (920 lines, 9 tables, full API)
- **Preferences/Monitors/Plans to DuckDB**: `PreferencesStore`, `MonitorStore`, `PlansMixin` rewritten
- **Data source registry**: 6 providers registered, `unifiedSources` GraphQL query
- **Agents as AGENT.md files**: AgentManager loads from directories, tiered loading, one-time migration from agents.yaml
- **Frontend test coverage**: 105 to 483 tests, 33.5% to 89.4% line coverage

## Tier 3 — New Source Types

- **MCP client**: `constat/mcp/` package — McpClient (JSON-RPC), McpClientPool (refcounted), ChangeProbe (tiered sync), McpDocumentProvider, McpApiProvider. 6 files, 50 tests.
- **Calendar source**: CalendarFetcher (Google Calendar API v3 + Microsoft Graph). Recurring expansion, attachment extraction, OAuth2 reuse. 26 tests.
- **Cloud Drive source**: DriveFetcher (Google Drive + OneDrive). BFS folder traversal, Google Docs export, file type filtering. 20 tests.
- **SharePoint source**: SharePointClient (Graph API). Libraries, lists-to-markdown, calendar conversion, modern/wiki page extraction. 17 tests.
- **Split _core.py**: 1615 to 975+572+249 lines (_core + _loaders + _chunking)
- **Split config.py**: 1409 to 945+486 lines (config + source_config)

## Tier 4 — Integration Layer

- **Personal resources**: `accounts.yaml` per user, generalized OAuth routes (`/api/oauth/`), account CRUD API (`/api/accounts/`), Fernet token encryption, session auto-loading. 29 tests.
- **Deploy tooling**: `constat deploy diff/generate/apply` CLI. ConfigDiffer (structural YAML diff), DeployApplier (backup/rollback, dry-run, category filtering), sensitive path masking. 41 tests.

## Tier 5 — Polish

- **Domain handbook**: `HandbookMixin` with 8 section builders (overview, sources, entities, glossary, rules, patterns, agents/skills, limitations). Edit routing to underlying stores. GraphQL query + mutation. 27 tests.
- **Chunk domain normalization**: `data_sources` table + `data_source_id` FK on embeddings (additive). Join-based domain resolution. 11 tests. Full cleanup deferred to Tier 6.

## Tier 6 — Cleanup

- **Chunk domain_id migration**: Embeddings queries migrated from direct `domain_id` to join-based resolution through `data_sources`
- **Arch docs cleanup**: Completed implementation plans moved to `docs/arch/completed/`
- **Verification gates**: Backend tests, frontend coverage, integration checks documented
