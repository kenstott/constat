# Domain Model

## Principles

1. **Every resource has an explicit domain.** No resource may have `domain=None`.
2. **New resources start in session domain** (`session_id`).
3. **Resources move between any domains** with permission — no fixed promotion hierarchy.
4. **"draft" is a status marker**, not a domain concept. Approving = removing draft status.
5. **Chunks inherit domain from their data source**, not stored redundantly.
6. **Glossary lives in the DB only** — no glossary in YAML config files.

## Domain Hierarchy

The 5-tier config resolution system defines valid domain scopes:

| Tier | Domain Value | Storage | Description |
|------|-------------|---------|-------------|
| SYSTEM | `"system"` | vectors.duckdb | Platform-wide resources |
| SYSTEM_DOMAIN | domain filename (e.g. `"hr-reporting"`) | vectors.duckdb | Shared domain resources |
| USER | `user_id` | {uid}.vault/vectors.duckdb | User's personal resources |
| USER_DOMAIN | `"{user_id}/{domain}"` | {uid}.vault/vectors.duckdb | User's domain-specific resources |
| SESSION | `session_id` | {uid}.vault/vectors.duckdb | Session-scoped resources |

## Resource Domain Rules

| Resource | Default Domain | Stored Where | Notes |
|----------|---------------|-------------|-------|
| Glossary term | `session_id` | glossary_terms.domain | Moved to other domains via update |
| Data source | `session_id` | Config or dynamic | Domain determines chunk domain |
| Chunk/embedding | Inherited from data source | embeddings.domain_id (current) | Future: FK to data source |
| Entity | Inherited from chunk(s) | entities.domain_id | An entity without a glossary term can span multiple domains (cross-domain entity). Domain is derived from its chunks' data sources. Once a glossary term is associated, domain comes from the glossary term. |
| Relationship | `session_id` | entity_relationships | Session-scoped |
| Learning/rule | `session_id` | YAML tiers | Promoted via tier management |

## Domain Movement

Any resource can move from any domain to any other domain, provided the user has write permission to the target domain. Permission is checked via `can_write_glossary()` / domain ownership.

---

## Implementation Phases

### Phase 1: Glossary Domain Enforcement

Enforce the invariant that every `GlossaryTerm` has an explicit domain.

**Model validation** (`constat/discovery/models.py`):
- `__post_init__` raises `ValueError` if `domain is None`
- Docstring updated to document domain semantics

**Fix creation sites — all drafts use `session_id`**:

| File | Change |
|------|--------|
| `constat/textual_repl/_commands.py:1466` | `domain=self.session.session_id` |
| `constat/session/_execution.py:1578,1600` | `domain=session_id` |
| `constat/discovery/glossary_generator.py:464,757` | `domain=session_id` |
| `constat/server/graphql/resolvers.py:418,469` | `draft_domain=session_id` |

**Fix tests**: Add explicit `domain` to all `GlossaryTerm` constructors in test files.

### Phase 2: Remove Glossary from Config YAML

Glossary terms live exclusively in the database. Config YAML defines data sources, not glossary.

**Remove config model fields**:
- `Config.glossary` field (`constat/core/config.py:1180`)
- `DomainConfig.glossary` field (`constat/core/config.py:994`)
- `ResolvedConfig.glossary` field (`constat/core/tiered_config.py:73`)

**Remove tiered config glossary handling**:
- `_load_system_tier` glossary serialization (`tiered_config.py:303-304`)
- `_extract_domain_sections` glossary serialization (`tiered_config.py:335-336`)
- `_build_resolved` glossary merge (`tiered_config.py:390`)

**Remove seeding**:
- Delete `_seed_domain_glossary()` (`session_manager.py:1859-1910`)
- Remove caller (`session_manager.py:834-842`)

**Remove config glossary chunk building**:
- `_index_glossary_and_relationships()` config glossary path (`doc_tools/_core.py:218-226`)
- `build_glossary_chunks()` if only used for config glossary (`catalog/glossary_builder.py:27`)

**Remove NER config glossary extraction**:
- `get_glossary_terms_for_ner(resolved.glossary)` calls (`session_manager.py:1000-1003`)
- NER should source business terms from DB glossary_terms table

**Remove from tier management**:
- `"glossary"` from `MANAGEABLE_SECTIONS` (`tier_management.py:47`)

**Clean up `canonical_source="domain_config"` references.**

### Phase 3: Chunk Domain Normalization

Chunks should reference their data source via FK. Domain is derived via join, not stored redundantly on each chunk.

**Schema change** (`constat/storage/duckdb_backend.py`):
- Add `data_source_id` FK column to embeddings table
- Create `data_sources` table (id, name, type, domain_id, session_id)
- Index on `data_source_id`
- Domain filtering via join: `JOIN data_sources ds ON e.data_source_id = ds.id WHERE ds.domain_id IN (...)`

**Migration**: ~88 `domain_id` references across `duckdb_backend.py` and `relational.py` need updating to use join-based domain resolution.

### Phase 4: Universal Domain Enforcement

Extend domain enforcement beyond glossary to all resource types:
- Entities: validate `domain_id` is never None
- Relationships: add `domain` column (currently session-scoped only)
- Learnings/facts: domain tracking in DB storage
