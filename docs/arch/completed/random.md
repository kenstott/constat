# Incremental Features

> **Status:** Planned. Not yet implemented.

---

## 1. Consistent Domain Badges Across All Resources

### Problem

Domain badges exist on some resource types but not others, and styling is inconsistent across the ones that have them.

**Current state:**

| Resource | Has Badge | Style | Component |
|----------|-----------|-------|-----------|
| Glossary terms | Yes | gray bg, `text-xs`, `px-1 py-0.5` | `GlossaryPanel.tsx` |
| Facts | Yes | blue bg, `text-[10px]`, `px-1 py-0.5` | `ArtifactPanel.tsx` |
| Skills | Yes | blue bg, `text-[10px]`, `px-1.5 py-0.5` | `ArtifactPanel.tsx` |
| Agents | Yes | blue bg, `text-[10px]`, `px-1.5 py-0.5` | `ArtifactPanel.tsx` |
| Rules | Yes | blue bg, unspecified size, `px-1.5 py-0.5` | `ArtifactPanel.tsx` |
| Databases | **No** | shows "source" label | `ArtifactPanel.tsx` |
| APIs | **No** | shows "source" label | `ArtifactPanel.tsx` |
| Documents | **No** | shows "source" label | `ArtifactPanel.tsx` |
| Tables | **No** | — | `ArtifactPanel.tsx` |
| Artifacts | **No** | — | `ArtifactPanel.tsx` |

Two problems:
1. Databases, APIs, and documents have a `source` field (session/config/domain-filename) but render it as a generic "source" label, not a domain badge.
2. Badge styling varies across the five resource types that have badges — different text sizes, padding, and colors.

### Plan

#### A. Extract a shared `DomainBadge` component

```tsx
// constat-ui/src/components/common/DomainBadge.tsx
interface DomainBadgeProps {
  domain: string | null | undefined
  domainPath?: string | null
}

export function DomainBadge({ domain, domainPath }: DomainBadgeProps) {
  if (!domain) return null
  return (
    <span
      className="text-[10px] px-1.5 py-0.5 rounded bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400 flex-shrink-0"
      title={domainPath || domain}
    >
      {domainPath || domain}
    </span>
  )
}
```

Single source of truth for badge rendering. Blue style (matches majority). `domainPath` for hierarchy tooltip (glossary already has this field).

#### B. Replace all inline badge spans

| File | Change |
|------|--------|
| `GlossaryPanel.tsx` | Replace inline `<span>` with `<DomainBadge domain={term.domain} domainPath={term.domain_path} />` |
| `ArtifactPanel.tsx` (facts) | Replace inline `<span>` with `<DomainBadge domain={fact.domain} />` |
| `ArtifactPanel.tsx` (skills) | Replace inline `<span>` with `<DomainBadge domain={skill.domain} />` |
| `ArtifactPanel.tsx` (agents) | Replace inline `<span>` with `<DomainBadge domain={agent.domain} />` |
| `ArtifactPanel.tsx` (rules) | Replace inline `<span>` with `<DomainBadge domain={rule.domain} />` |

#### C. Add domain badges to databases, APIs, and documents

These resources already carry domain affiliation via their `source` field. When `source` is a domain filename (not `"session"` or `"config"`), render a `<DomainBadge>`.

The backend already provides this — the `source` field on databases/APIs/documents in the resolved config contains the domain filename when the resource comes from a domain. No backend changes needed.

```tsx
// In database/API/document list items:
{source !== 'session' && source !== 'config' && (
  <DomainBadge domain={source} />
)}
```

#### D. Add `domain_path` to backend responses for databases, APIs, documents

Currently only glossary terms include `domain_path`. Extend the pattern:

**`constat/server/routes/sessions.py`** — When building database/API/document lists for a session, look up the domain's `path` field from `Config.projects[filename]` and include it.

**`constat/server/models.py`** — Add optional `domain_path` to database/API/document response models.

### Files to Modify

| File | Change |
|------|--------|
| `constat-ui/src/components/common/DomainBadge.tsx` | New component |
| `constat-ui/src/components/artifacts/GlossaryPanel.tsx` | Use `DomainBadge` |
| `constat-ui/src/components/artifacts/ArtifactPanel.tsx` | Use `DomainBadge` for facts, skills, agents, rules, databases, APIs, documents |
| `constat/server/routes/sessions.py` | Add `domain_path` to resource responses |
| `constat/server/models.py` | Add `domain_path` field to resource response models |

---

## 2. URI-Based Document Addition

### Problem

Users cannot add HTTP/URL-based documents to a session at runtime. The infrastructure exists (transport layer, crawler, `add_document_from_config()`) but is only used at startup for domain-configured documents.

**Root cause:** `Session.add_file()` in `constat/session/_resources.py:128-134` only handles `file://` URIs:

```python
file_path = uri
if uri.startswith("file://"):
    file_path = uri[7:]

path = Path(file_path)
if path.suffix.lower() in doc_extensions and path.exists():  # ← FAILS FOR HTTP
    # Index document...
```

HTTP URIs silently fall through — no error, no indexing. The file reference is stored but the content is never fetched or indexed.

**What works at startup but not at runtime:**

```python
# constat/server/app.py — startup only
doc_tools.add_document_from_config(doc_name, doc_config, domain_id=filename)
```

This method handles HTTP fetch, content extraction, chunking, embedding, and link crawling. It is never called from any runtime API endpoint.

### Plan

#### A. New API endpoint: `POST /{session_id}/documents/add-uri`

**`constat/server/routes/files.py`** — Add endpoint:

```python
class AddDocumentURIRequest(BaseModel):
    name: str
    url: str
    description: str = ""
    headers: dict[str, str] = {}
    follow_links: bool = False
    max_depth: int = 2
    max_documents: int = 20
    same_domain_only: bool = True
    exclude_patterns: list[str] = []
    type: str = "auto"  # auto, pdf, html, markdown, text

@router.post("/{session_id}/documents/add-uri")
async def add_document_uri(
    session_id: str,
    body: AddDocumentURIRequest,
    user_id: CurrentUserId,
):
    managed = session_manager.get_session(session_id, user_id)
    if not managed:
        raise HTTPException(404, "Session not found")

    # Build DocumentConfig from request
    doc_config = DocumentConfig(
        url=body.url,
        description=body.description,
        headers=body.headers,
        follow_links=body.follow_links,
        max_depth=body.max_depth,
        max_documents=body.max_documents,
        same_domain_only=body.same_domain_only,
        exclude_patterns=body.exclude_patterns,
        type=body.type,
    )

    doc_tools = managed.session.doc_tools
    success, msg = doc_tools.add_document_from_config(
        body.name,
        doc_config,
        session_id=session_id,
    )

    if not success:
        raise HTTPException(400, msg)

    # Track as file reference for persistence
    managed._file_refs.append({
        "name": body.name,
        "uri": body.url,
        "has_auth": bool(body.headers),
        "description": body.description,
        "added_at": datetime.utcnow().isoformat(),
        "document_config": doc_config.model_dump(exclude_defaults=True),
    })

    return {"status": "ok", "name": body.name, "message": msg}
```

#### B. Fix `Session.add_file()` for HTTP URIs

**`constat/session/_resources.py`** — Extend `add_file()` to detect HTTP URIs and delegate to `doc_tools.add_document_from_config()`:

```python
def add_file(self, name: str, uri: str, auth: str = "", description: str = ""):
    # ... existing file:// handling ...

    if uri.startswith(("http://", "https://")):
        headers = {}
        if auth:
            headers["Authorization"] = auth
        doc_config = DocumentConfig(
            url=uri,
            description=description,
            headers=headers,
        )
        success, msg = self.doc_tools.add_document_from_config(
            name, doc_config, session_id=self.session_id,
        )
        if not success:
            raise ValueError(msg)
        self.session_files[name] = {"uri": uri, "description": description}
        return

    # ... existing local file handling ...
```

#### C. Restore URI documents on session restore

**`constat/server/session_manager.py`** — `restore_resources()` currently restores file refs but does not re-index URI documents. When a file ref has a `document_config` dict, call `add_document_from_config()`:

```python
async def restore_resources(self, managed: ManagedSession):
    for ref in managed._file_refs:
        if "document_config" in ref:
            doc_config = DocumentConfig(**ref["document_config"])
            managed.session.doc_tools.add_document_from_config(
                ref["name"], doc_config, session_id=managed.session_id,
            )
        else:
            managed.session.add_file(ref["name"], ref["uri"], ...)
```

#### D. Frontend: Add URI input to document section

**`constat-ui/src/components/artifacts/ArtifactPanel.tsx`** — In the documents section, add a URL input alongside the existing file upload:

```
┌──────────────────────────────────────────────┐
│  Documents                                   │
│                                              │
│  [📎 Upload file]  [🔗 Add URL]             │
│                                              │
│  ── Add URL ───────────────────────────────  │
│  URL:  [ https://...                      ]  │
│  Name: [ my-doc                           ]  │
│  Description: [ optional                  ]  │
│                                              │
│  ── Crawling ──────────────────────────────  │
│  ☑ Follow links                             │
│    Max depth:     [ 2 ▾]                     │
│    Max documents: [20 ▾]                     │
│    ☑ Same domain only                       │
│    Exclude patterns:                         │
│    [ /wiki/Special:*, /api/*            ] ✕  │
│    [ + Add pattern ]                         │
│                                              │
│  ── Advanced ──────────────────────────────  │
│  Content type: [Auto ▾]                      │
│  Custom headers:                             │
│  [ Authorization ] [ Bearer ... ]         ✕  │
│  [ + Add header ]                            │
│                                              │
│  [Add]                                       │
└──────────────────────────────────────────────┘
```

The Crawling and Advanced sections are collapsible, hidden by default. The Follow links checkbox is the toggle — when unchecked, the crawling sub-options are hidden. Sensible defaults (`max_depth=2`, `max_documents=20`, `same_domain_only=true`) match `DocumentConfig`.

**`constat-ui/src/api/sessions.ts`** — Add API method:

```typescript
export async function addDocumentURI(
  sessionId: string,
  body: {
    name: string
    url: string
    description?: string
    follow_links?: boolean
    max_depth?: number
    same_domain_only?: boolean
  }
): Promise<{ status: string; name: string; message: string }> {
  return post(`/sessions/${sessionId}/documents/add-uri`, body)
}
```

### Files to Modify

| File | Change |
|------|--------|
| `constat/server/routes/files.py` | New `POST /{session_id}/documents/add-uri` endpoint |
| `constat/session/_resources.py` | Fix `add_file()` to handle HTTP URIs |
| `constat/server/session_manager.py` | Restore URI documents on session restore |
| `constat-ui/src/components/artifacts/ArtifactPanel.tsx` | Add URL input UI in documents section |
| `constat-ui/src/api/sessions.ts` | `addDocumentURI()` API method |

---

## 3. Simplified Exemplar Download for Model Fine-Tuning

### Problem

The existing exemplar system (`POST /learnings/generate-exemplars`, `GET /learnings/exemplars/download`) generates fine-tuning pairs from rules, glossary terms, and relationships via LLM-based question-answer pair generation. This is powerful but heavyweight — it requires an active session, runs LLM calls per batch, and produces synthetic Q&A pairs that may not reflect actual user interactions.

Users need a simpler option: download the raw learnings and rules in a format ready for fine-tuning, without LLM generation. The data already exists — corrections capture real user intent, rules capture compacted patterns, and glossary terms capture domain vocabulary. A direct export of this material in standard fine-tuning formats gives users a fast, deterministic, no-cost alternative.

### Current State

**ExemplarGenerator** (`constat/learning/exemplar_generator.py`):
- 3 coverage levels (minimal/standard/comprehensive) controlling which data is included
- LLM generates synthetic Q&A pairs from rules, glossary, relationships
- Outputs `exemplars_messages.jsonl` (OpenAI Messages) and `exemplars_alpaca.jsonl` (Alpaca)
- Stored at `.constat/{user_id}/exemplars_{format}.jsonl`

**What's missing:**
- No option to export the raw learnings/rules directly (without LLM synthesis)
- No inclusion of actual query history (real user questions + system answers)
- No filtering by domain, category, or date range

### Plan

#### A. New endpoint: `GET /learnings/exemplars/simple`

Returns learnings, rules, and optionally glossary terms as fine-tuning records — no LLM calls, deterministic, instant.

**`constat/server/routes/learnings.py`** — Add endpoint:

```python
class SimpleExemplarRequest(BaseModel):
    format: Literal["messages", "alpaca", "sharegpt"] = "messages"
    include: list[Literal[
        "corrections", "rules", "glossary", "query_history"
    ]] = ["corrections", "rules"]
    domain: str | None = None        # Filter by domain (None = all)
    min_confidence: float = 0.0      # Rules only: minimum confidence
    since: datetime | None = None    # Filter by created_at

@router.get("/learnings/exemplars/simple")
async def download_simple_exemplars(
    format: str = "messages",
    include: str = "corrections,rules",  # Comma-separated
    domain: str | None = None,
    min_confidence: float = 0.0,
    since: str | None = None,
    user_id: CurrentUserId,
):
    ...
```

#### B. Output formats

Three standard formats, all as JSONL:

**OpenAI Messages** (`messages`):
```jsonl
{"messages": [{"role": "system", "content": "You are a data analyst..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

**Alpaca** (`alpaca`):
```jsonl
{"instruction": "...", "input": "", "output": "..."}
```

**ShareGPT** (`sharegpt`):
```jsonl
{"conversations": [{"from": "system", "value": "..."}, {"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
```

#### C. Record conversion logic

Each data type maps to fine-tuning records differently:

**Corrections (raw learnings):**
```python
# A correction captures: "user said X was wrong, the answer should be Y"
# Convert to: system prompt + user question (from context) + corrected answer
{
    "role": "user",
    "content": context.get("query_text", ""),  # Original question
}
{
    "role": "assistant",
    "content": correction,                      # The correction
}
```

**Rules (compacted patterns):**
```python
# A rule is a general instruction: "When X, always do Y"
# Convert to system prompt augmentation
{
    "role": "system",
    "content": f"Rule: {rule.summary}",
}
```

Alternatively, for rules that have source learnings with query context, reconstruct the Q&A pair using the original query and the rule as the guiding principle.

**Glossary terms:**
```python
# Convert to: "What does X mean?" → definition
{
    "role": "user",
    "content": f"What does '{term.name}' mean in this context?",
}
{
    "role": "assistant",
    "content": term.definition,
}
```

**Query history** (if session available):
```python
# Real user interactions — most valuable for fine-tuning
# Pull from session message history
{
    "role": "user",
    "content": message.query_text,
}
{
    "role": "assistant",
    "content": message.answer_summary,
}
```

#### D. Frontend: Add "Simple Export" option to learnings UI

**`constat-ui/src/components/artifacts/ArtifactPanel.tsx`** — Add download button alongside existing exemplar generation:

```
┌────────────────────────────────────────────────────┐
│  Learnings                             [Compact]   │
│                                                    │
│  Pending (12)  │  Rules (8)  │  Export             │
│                                                    │
│  ── Export for Fine-Tuning ──────────────────────  │
│                                                    │
│  Format: [OpenAI Messages ▾]                       │
│                                                    │
│  Include:                                          │
│  ☑ Corrections (12)                               │
│  ☑ Rules (8)                                      │
│  ☐ Glossary terms (45)                            │
│  ☐ Query history                                  │
│                                                    │
│  Filter by domain: [All ▾]                         │
│  Min rule confidence: [0.6 ▾]                      │
│                                                    │
│  [⬇ Download JSONL]    [⚙ Generate with LLM]     │
│                                                    │
│  "Download" = instant, raw data                    │
│  "Generate" = LLM-synthesized Q&A pairs            │
└────────────────────────────────────────────────────┘
```

"Download JSONL" calls the new simple endpoint. "Generate with LLM" calls the existing `generate-exemplars` endpoint.

**`constat-ui/src/api/sessions.ts`** — Add API method:

```typescript
export async function downloadSimpleExemplars(params: {
  format?: 'messages' | 'alpaca' | 'sharegpt'
  include?: string[]
  domain?: string
  min_confidence?: number
  since?: string
}): Promise<Blob> {
  const query = new URLSearchParams()
  if (params.format) query.set('format', params.format)
  if (params.include) query.set('include', params.include.join(','))
  if (params.domain) query.set('domain', params.domain)
  if (params.min_confidence) query.set('min_confidence', String(params.min_confidence))
  if (params.since) query.set('since', params.since)
  return getBlob(`/learnings/exemplars/simple?${query}`)
}
```

### Files to Modify

| File | Change |
|------|--------|
| `constat/server/routes/learnings.py` | New `GET /learnings/exemplars/simple` endpoint |
| `constat/learning/simple_exporter.py` | New module: conversion logic (corrections/rules/glossary/history → JSONL) |
| `constat/storage/learnings.py` | Add `list_corrections_with_context()` if needed for richer export |
| `constat-ui/src/components/artifacts/ArtifactPanel.tsx` | Export tab UI with format/filter options |
| `constat-ui/src/api/sessions.ts` | `downloadSimpleExemplars()` API method |

---

## 4. Data Source Scoping for Learnings and Rules

### Problem

Learnings and rules lose data source specificity during capture and compaction, producing generic rules that are often wrong when applied to a different source.

**Example:** A Snowflake-specific SQL issue (e.g., `DATE_TRUNC('month', col)` vs PostgreSQL's `DATE_TRUNC('month', col)::date`) gets captured as a `CODEGEN_ERROR` with this context:

```python
# What's captured today (_execution.py:618-623)
pending_learning_context = {
    "error_message": "...",
    "original_code": "...",
    "step_goal": "...",
    "attempt": 2,
}
# No data source name. No database type. No connection info.
```

The compactor then groups all `CODEGEN_ERROR` learnings together and asks the LLM to find patterns. A Snowflake DATE_TRUNC fix and a PostgreSQL EXTRACT fix land in the same group, producing a generic rule like "use database-specific date functions" — which tells the codegen nothing actionable.

**Three cascading failures:**

1. **Capture** — `pending_learning_context` doesn't record which data source triggered the error
2. **Compaction** — Groups by `LearningCategory` (6 broad buckets), not by data source. A Snowflake fix and a DuckDB fix both in `CODEGEN_ERROR` get merged
3. **Retrieval** — `_prompts.py:190-208` filters rules by SQL-vs-Python keyword matching, but has no way to filter by the data source currently being queried

### Plan

#### A. Capture data source at learning creation time

**`constat/session/_execution.py`** — Enrich `pending_learning_context` with the data source(s) involved in the current step:

```python
# In _execute_step(), when building pending_learning_context:
pending_learning_context = {
    "error_message": last_error[:500] if last_error else "",
    "original_code": last_code[:500] if last_code else "",
    "step_goal": step.goal,
    "attempt": attempt,
    # NEW: data source scope
    "data_sources": self._get_step_data_sources(step),
}
```

**New helper `_get_step_data_sources()`:**

```python
def _get_step_data_sources(self, step) -> list[dict]:
    """Extract data source info from the step's database references.

    Returns list of:
      {"name": "sales_db", "type": "snowflake", "dialect": "snowflake"}
    """
    sources = []
    for db_name in step.databases_used or []:
        db_info = self.schema_manager.get_database_info(db_name)
        if db_info:
            sources.append({
                "name": db_name,
                "type": db_info.get("type", ""),       # snowflake, postgres, duckdb, etc.
                "dialect": db_info.get("dialect", ""),  # SQL dialect
            })
    return sources
```

The step already knows which databases it references (from the planner). If `databases_used` isn't populated on the step, infer from the generated code by matching known database names.

#### B. Add `scope` field to learnings and rules

**`constat/storage/learnings.py`** — Add `scope` to both tiers:

```yaml
# Learning (Tier 1)
corrections:
  learn_001:
    category: "codegen_error"
    correction: "Snowflake requires DATE_TRUNC('month', col) not EXTRACT(MONTH FROM col)"
    context: {error_message: "...", original_code: "...", ...}
    scope:                              # NEW
      data_sources:
        - {name: "sales_db", type: "snowflake"}
      domain: "sales-analytics.yaml"

# Rule (Tier 2)
rules:
  rule_001:
    summary: "In Snowflake, use DATE_TRUNC('month', col) instead of EXTRACT(MONTH FROM col)"
    scope:                              # NEW
      data_sources:
        - {type: "snowflake"}           # Instance-level OR type-level
      domain: "sales-analytics.yaml"
```

**Scope hierarchy (most specific → least):**

| Level | Scope | Example | When it applies |
|-------|-------|---------|-----------------|
| Instance | `{name: "sales_db", type: "snowflake"}` | "sales_db requires SET QUOTED_IDENTIFIERS ON" | Fix is specific to this database's configuration, schema, or data |
| Type | `{type: "snowflake"}` | "Snowflake needs DATE_TRUNC not EXTRACT" | Fix is a dialect/engine behavior — applies to all instances of this type |
| Global | `{}` | "Always validate column names before SELECT" | Fix is a general coding practice — applies regardless of data source |

**The scope level requires judgment, not mechanical assignment.** The data source present at capture time is recorded as context, but the applicable scope is determined by the LLM at two points:

1. **At capture** — `_summarize_error_fix()` already uses the LLM. Extend it to also classify scope (see section C below).
2. **At compaction** — The compactor validates/adjusts scope when creating rules from groups of learnings.

The `save_learning()` signature gains an optional `scope` parameter:

```python
def save_learning(
    self,
    category: LearningCategory,
    context: dict,
    correction: str,
    source: LearningSource = LearningSource.AUTO_CAPTURE,
    scope: dict | None = None,  # NEW: {"data_sources": [...], "domain": "..."}
) -> str:
```

And `save_rule()` gains the same:

```python
def save_rule(
    self,
    summary: str,
    category: LearningCategory,
    confidence: float,
    source_learnings: list[str],
    tags: list[str] | None = None,
    domain: str = "",
    scope: dict | None = None,  # NEW
) -> str:
```

#### C. LLM-based scope classification at capture

The data source present at error time is a hint, not the answer. The LLM must judge whether the fix is instance-specific, type-specific, or universal.

**`constat/session/_execution.py`** — Extend `_summarize_error_fix()` to also classify scope:

```python
def _summarize_error_fix(self, context: dict, fixed_code: str) -> tuple[str, dict]:
    """Summarize error fix AND classify its scope.

    Returns:
        (summary, scope) where scope is:
          {"level": "instance", "data_sources": [{"name": "sales_db", "type": "snowflake"}]}
          {"level": "type", "data_sources": [{"type": "snowflake"}]}
          {"level": "global", "data_sources": []}
    """
    data_sources = context.get("data_sources", [])
    source_desc = ", ".join(
        f"{s['name']} ({s['type']})" for s in data_sources
    ) if data_sources else "unknown"

    prompt = f"""Summarize what was learned from this error fix in ONE sentence.
Then classify how broadly this fix applies.

Error: {context.get('error_message', '')[:300]}
Original code: {context.get('original_code', '')[:200]}
Fixed code: {fixed_code[:200]}
Data source(s) involved: {source_desc}

Output JSON:
{{
  "summary": "...",
  "scope_level": "instance | type | global",
  "scope_reason": "one sentence why"
}}

Scope guidance:
- "instance": Fix is about THIS specific database's config, schema quirk, data issue,
  or connection setup. Would NOT apply to another database of the same type.
  Example: "sales_db requires SET QUOTED_IDENTIFIERS ON due to legacy schema"
- "type": Fix is about the database ENGINE's behavior or SQL dialect. Applies to ALL
  databases of this type but not others.
  Example: "Snowflake uses DATE_TRUNC('month', col) not EXTRACT(MONTH FROM col)"
- "global": Fix is a general coding practice unrelated to any specific database engine.
  Example: "Always check if column exists before referencing in SELECT"

Output ONLY valid JSON."""
    ...
```

The returned `scope` is passed to `save_learning()`. The LLM sees the error, the fix, and the data source — it can judge whether the fix is about Snowflake's dialect (type), sales_db's particular schema (instance), or a universal pattern (global).

#### D. Scope-aware compaction

**`constat/learning/compactor.py`** — Two changes:

**1. Pre-group by scope level before LLM similarity analysis:**

```python
# Current: groups by category only (line 100-104)
by_category = defaultdict(list)
for learning in all_learnings:
    cat = learning.get("category", "unknown")
    by_category[cat].append(learning)

# New: group by (category, scope_level, scope_key)
by_scope = defaultdict(list)
for learning in all_learnings:
    cat = learning.get("category", "unknown")
    scope = learning.get("scope", {})
    level = scope.get("level", "global")
    scope_key = _scope_group_key(scope)
    by_scope[(cat, level, scope_key)].append(learning)
```

```python
def _scope_group_key(scope: dict) -> str:
    """Grouping key from scope. Instance and type learnings for the same
    source type land in the same pre-group; the LLM then decides similarity."""
    sources = scope.get("data_sources", [])
    if sources:
        types = sorted(set(s.get("type", "") for s in sources if s.get("type")))
        if types:
            return ":".join(types)
    return "_global"
```

Instance-level and type-level learnings for the same database type (e.g., both Snowflake) land in the same pre-group. The LLM similarity analysis within that group then decides which are truly similar. This prevents cross-engine merging while still allowing the LLM to find patterns within an engine.

**2. Scope judgment at rule creation:**

When the compactor creates a rule from a group of learnings, the LLM also determines the rule's scope — which may differ from the individual learnings' scopes:

```python
prompt = f"""Create a single rule from these {len(group)} similar learnings.

Data source context: {scope_description}
Individual scope levels: {scope_levels}

Learnings:
{corrections}

Output JSON:
{{
  "summary": "...",
  "confidence": 0.85,
  "tags": [...],
  "scope_level": "instance | type | global",
  "scope_reason": "why this scope level"
}}

Scope guidance:
- If ALL learnings are about the same specific database instance and the pattern
  is tied to that instance's config/schema/data → "instance"
- If the pattern is about the database ENGINE's behavior (SQL dialect, type system,
  function support) → "type"
- If the pattern is a general coding practice that happens to have occurred on
  a particular database → "global"

A group of instance-level learnings about the same engine quirk should produce a
TYPE-level rule. A group of type-level learnings about a universal practice should
produce a GLOBAL rule. Judge the RULE's scope, not just echo the learnings' scopes.
"""
```

This is the key insight: the compactor doesn't just inherit scope from learnings. Three instance-level Snowflake DATE_TRUNC learnings from different databases should produce a **type-level** Snowflake rule. Three type-level learnings about column validation from Snowflake, PostgreSQL, and DuckDB should produce a **global** rule.

#### E. Scope-aware retrieval

**`constat/session/_prompts.py`** — When injecting rules into the codegen prompt, filter by the active data sources:

```python
def _build_learning_context(self, step_goal: str, task_type) -> str:
    # Determine active data sources for this step
    active_sources = self._get_active_data_source_types()  # e.g., ["snowflake"]

    rules = self.learning_store.list_rules(
        category=LearningCategory.CODEGEN_ERROR,
        min_confidence=0.6,
        scope_filter=active_sources,  # NEW: only rules matching active sources
    )
    ...
```

**`constat/storage/learnings.py`** — Add `scope_filter` to `list_rules()`:

```python
def list_rules(
    self,
    category=None,
    min_confidence=0.0,
    limit=None,
    domain=None,
    scope_filter: list[str] | None = None,  # NEW: data source types
) -> list[dict]:
    ...
    if scope_filter:
        filtered = []
        for rule in rules:
            rule_types = _get_scope_types(rule)
            # Include if: rule is unscoped (global) OR matches any active source
            if not rule_types or rule_types & set(scope_filter):
                filtered.append(rule)
        rules = filtered
    ...
```

Unscoped rules (no `scope` field — legacy or truly generic) are always included. Scoped rules are only included when the scope matches.

#### F. Scope display in UI

**`constat-ui/src/components/artifacts/ArtifactPanel.tsx`** — Show scope badges on learnings and rules:

```
┌────────────────────────────────────────────────────────┐
│  Rules (8)                                             │
│                                                        │
│  ● Use DATE_TRUNC('month', col) not EXTRACT(...)       │
│    [snowflake] [sales_db]  confidence: 0.85            │
│                                                        │
│  ● Always CAST varchar to date before comparison       │
│    [postgresql]  confidence: 0.78                      │
│                                                        │
│  ● Check column exists before referencing in SELECT    │
│    [global]  confidence: 0.92                          │
└────────────────────────────────────────────────────────┘
```

Scope badges use distinct colors:
- Instance-level (specific db name): orange
- Type-level (snowflake, postgresql): purple
- Global (no scope): gray

### Migration

Existing learnings and rules have no `scope` field. They are treated as global (unscoped) — always included in retrieval, groupable with any same-category learning. No data migration needed. New learnings automatically capture scope; old ones remain global.

### Files to Modify

| File | Change |
|------|--------|
| `constat/session/_execution.py` | Enrich `pending_learning_context` with data source info; add `_get_step_data_sources()` |
| `constat/storage/learnings.py` | Add `scope` param to `save_learning()` and `save_rule()`; add `scope_filter` to `list_rules()` |
| `constat/learning/compactor.py` | Group by `(category, scope_key)` instead of category alone; include scope in LLM prompts |
| `constat/session/_prompts.py` | Filter rules by active data source types in `_build_learning_context()` |
| `constat/server/models.py` | Add `scope` to `LearningInfo` and `RuleInfo` response models |
| `constat-ui/src/components/artifacts/ArtifactPanel.tsx` | Render scope badges on learnings and rules |
| `constat-ui/src/types/api.ts` | Add `scope` to `Learning` and `Rule` types |