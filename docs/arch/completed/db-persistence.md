# DB Persistence Migration

## Principle

**Files are for human-editable configuration. Everything else goes in the DB.**

Data sources and permissions stay as YAML — humans review and modify them. All runtime-generated data moves to the database, eliminating file-management complexity.

## What Stays as Files

| Resource | Format | Why |
|----------|--------|-----|
| Data source configs (databases, APIs, documents) | YAML | Human-reviewed, version-controlled |
| Permissions / rights | YAML | Security-sensitive, auditable by humans |
| Skills | YAML + code | Developer-authored, filesystem organization natural |
| Agents | YAML | Human-authored prompts and config |
| Golden questions (regression tests) | YAML | Domain-specific, human-authored test cases |
| Entity resolution queries | YAML | Human-authored SQL/GraphQL, domain-specific |
| Fine-tune output artifacts | JSONL/YAML | Portable, git-trackable training data |

## What Moves to DB

| Resource | Current Storage | Current Format | Target DB | Notes |
|----------|----------------|---------------|-----------|-------|
| Learnings/rules | `.constat/{uid}/learnings.yaml` | YAML (up to 1MB) | user vault user.duckdb | Heavy access, search/filter, concurrent writes |
| Facts | `.constat/{uid}/facts.yaml` | YAML | user vault user.duckdb | Role-based filtering, domain tracking |
| Session history | `.constat/{uid}/sessions/{sid}/*.json` | JSON/JSONL/TXT (50+ files per session) | session.duckdb | Complex hierarchy, audit trail, querying |
| Steps + code/output | `.constat/{uid}/sessions/{sid}/steps/` | .py/.txt/JSONL | session.duckdb | Currently scattered files, needs indexing |
| Plans (saved) | `.constat/{uid}/saved_plans.json` | JSON | user vault user.duckdb | Searchable, shareable, versioned |
| Plan iterations | `.constat/{uid}/sessions/{sid}/plan/` | JSON/TXT | session.duckdb | Per-session, tied to step execution |
| Inferences (auditable) | `.constat/{uid}/sessions/{sid}/inferences/` | .py/.txt/JSONL | session.duckdb | Proof chain, tied to session |
| Monitors + run history | `.constat/{uid}/monitors.json` + `monitor_runs/` | JSON/JSONL | user vault user.duckdb | Append-only runs, trigger state |
| User preferences | `.constat/{uid}/preferences.yaml` | YAML | user vault user.duckdb | Simple key-value |
| Glossary | Config YAML (being removed) | YAML | Already in DB | See docs/arch/domains.md Phase 2 |

## DB File Naming

Part of Phase 1. Rename for clarity as files grow beyond vectors:

| Current | New | Location |
|---------|-----|----------|
| `vectors.duckdb` | `user.duckdb` | `{uid}.vault/` |
| `vectors.duckdb` | `system.duckdb` | `.constat/` root |
| `session.duckdb` | `session.duckdb` | Per-session dir (unchanged) |

Note: The root `.constat/vectors.duckdb` migrates to `system.duckdb`.

One-time rename on first access: if old file exists and new doesn't, rename it.

## Implementation Phases

### Phase 1: Learnings + Facts + DB Rename

Highest impact — learnings is the largest file, most frequently accessed, and YAML parse/dump is a bottleneck. Also renames DB files (`vectors.duckdb` → `user.duckdb` in user vault, `vectors.duckdb` → `system.duckdb` at root).

**Target DB:** user vault `user.duckdb` (renamed from `vectors.duckdb`, managed by SplitVectorStore).

**Table schemas:**

```sql
CREATE TABLE IF NOT EXISTS learnings (
    id VARCHAR PRIMARY KEY,
    category VARCHAR NOT NULL,
    created TIMESTAMP NOT NULL,
    context TEXT,                    -- JSON dict
    correction TEXT NOT NULL,
    source VARCHAR NOT NULL,
    applied_count INTEGER NOT NULL DEFAULT 0,
    promoted_to VARCHAR,
    scope TEXT,                      -- JSON dict or NULL
    domain VARCHAR NOT NULL DEFAULT '',
    archived_at TIMESTAMP           -- NULL = active, non-NULL = archived
);

CREATE TABLE IF NOT EXISTS learning_rules (
    id VARCHAR PRIMARY KEY,
    category VARCHAR NOT NULL,
    summary TEXT NOT NULL,
    confidence FLOAT NOT NULL,
    source_learnings TEXT,          -- JSON array
    tags TEXT,                      -- JSON array
    applied_count INTEGER NOT NULL DEFAULT 0,
    created TIMESTAMP NOT NULL,
    updated_at TIMESTAMP,
    domain VARCHAR NOT NULL DEFAULT '',
    scope TEXT                      -- JSON dict or NULL
);

CREATE TABLE IF NOT EXISTS facts (
    name VARCHAR PRIMARY KEY,
    value TEXT NOT NULL,            -- JSON-serialized
    description TEXT NOT NULL DEFAULT '',
    context TEXT NOT NULL DEFAULT '',
    role_id VARCHAR,
    domain VARCHAR NOT NULL DEFAULT '',
    created TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS exemplar_runs (
    id VARCHAR PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    stats TEXT NOT NULL             -- JSON blob
);
```

**Design notes:**
- `learnings` combines corrections + archive (`archived_at IS NOT NULL` = archived)
- JSON TEXT for `context`, `scope`, `tags`, `source_learnings` — small payloads, read wholesale
- NOT added to `SPLIT_TABLES` — user learnings/facts are always user-scoped

**Rewrite LearningStore** (`constat/storage/learnings.py`):
- Constructor accepts `db: ThreadLocalDuckDB | None` — when provided, use directly; when None, open own connection
- Remove RLock (DuckDB pool has its own), `_data` cache, `_save()`/`_load()` YAML methods
- Keep `LearningCategory`, `LearningSource` enums, all public method signatures unchanged
- `get_relevant_rules()`: SQL filter by confidence → Python keyword scoring (rule set is small)

**Rewrite FactStore** (`constat/storage/facts.py`):
- Same pattern. `value` field JSON-serialized to preserve Python types.

**Connection access:**
- Add `get_user_db(user_id)` to `SessionManager` — returns existing `ThreadLocalDuckDB` from user's vector store
- Update call sites in `session/_core.py`, `session_manager.py`, `routes/learnings.py` to pass `db`

**One-time YAML import:**
- On first access: if table empty AND `.yaml` exists, import all entries
- Rename `learnings.yaml` → `learnings.yaml.imported` (non-destructive)
- Same for `facts.yaml`

**DDL registration:**
- Add CREATE TABLE IF NOT EXISTS to `_ensure_incremental_schema()` in `vector_store.py`

**Critical files:**

| File | Change |
|------|--------|
| `constat/storage/learnings.py` | Rewrite: YAML → DuckDB backend |
| `constat/storage/facts.py` | Rewrite: YAML → DuckDB backend |
| `constat/discovery/vector_store.py` | Add table DDL to schema init |
| `constat/server/session_manager.py` | Add `get_user_db()`, update store construction |
| `constat/session/_core.py` | Pass `db` to LearningStore/FactStore constructors |
| `constat/server/routes/learnings.py` | Use `session_manager.get_user_db()` |

### Phase 2: Session History + Steps + Plans

Biggest file proliferation — 50+ files per session scattered across filesystem.

**Session metadata** (`constat/storage/history.py`):
- Table: `session_history` (session_id, created_at, config_hash, databases, status, summary)
- Currently: `session.json` per session directory

**Steps** (`constat/session/` step execution):
- Table: `steps` (session_id, step_number, goal, code, output, error, prompt, metadata)
- Currently: `step_NNN_code.py`, `step_NNN_output.txt`, etc.
- `index.jsonl` metadata merges into the table

**Plan iterations**:
- Table: `plan_iterations` (session_id, iteration, raw_response, parsed_plan, reasoning, approval)
- Currently: `plan/v{N}_*.json|txt` files

**Inferences** (auditable mode):
- Table: `inferences` (session_id, inference_id, code, output, error, prompt, premises)
- Currently: `inferences/*.py|txt|jsonl` files

**Conversation state**:
- Tables or columns in session tables for `state.json`, `messages.json`, `proof_facts.json`

### Phase 3: Monitors + Preferences + Saved Plans

Lower urgency — smaller files, less frequent access.

**Monitors** (`constat/storage/monitors.py`):
- Tables: `monitors` (config), `monitor_runs` (append-only log)
- Currently: `monitors.json` + `monitor_runs/{id}.jsonl`

**Preferences** (`constat/server/user_preferences.py`):
- Table: `preferences` (key, value) or single JSON column
- Currently: `preferences.yaml`

**Saved plans** (`constat/session/_plans.py`):
- Table: `saved_plans` (name, problem, created_by, steps JSON)
- Currently: `saved_plans.json` (user) + `shared/saved_plans.json` (shared)

## Migration Strategy

- No schema migrations (v1 rule) — tables are created on first access if they don't exist
- File-based storage classes get a DB backend behind the same interface
- Old files are read on first access (one-time import), then ignored
- Session history: new sessions use DB, old sessions remain readable from files
