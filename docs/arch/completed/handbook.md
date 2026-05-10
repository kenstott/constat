# Domain Handbook

## Goal

Auto-generated, readable page per domain that explains what the domain knows, can do, and where it struggles. New users understand a domain without reading config YAML or asking questions. Users can edit/annotate — corrections flow back as glossary updates or rule additions.

Key distinction from PromptQL's wiki: their wiki is the **input** (users write, system reads). The handbook is the **output** (system generates, users read and refine).

## Page Structure

Each domain handbook is assembled from existing stores — no new data collection required.

### 1. Overview

- Domain name, description, scope (from domain config YAML)
- Parent domains in the DAG (inheritance chain)
- LLM-generated summary paragraph synthesizing the sections below into 2-3 sentences

**Source**: `DomainConfig` + LLM summarization

### 2. Data Sources

Per source:
- Name, type (database/API/document/MCP), connection status
- Schema summary: table count, key tables with row counts, column highlights
- For documents: document count, total chunks, last refresh timestamp
- For APIs: endpoint count, top endpoints by usage
- For MCP: server name, resource count, tool count

**Source**: `DuckDBSessionStore` metadata, `SchemaManager`, `APISchemaManager`, MCP catalog

### 3. Key Entities

- Entity name, type, source grounding (which table/document/API)
- Relationship graph summary (top relationships by frequency)
- Entity clusters (related entities grouped)
- Spanning entities (entities that appear across multiple sources)

**Source**: `RelationalStore` (entities, relationships, clusters)

### 4. Glossary

- Term, definition, aliases, category
- Grounding status (grounded to source data vs. user-defined)
- Connected resources count
- Domain-scoped vs. inherited (from parent domain)

**Source**: `RelationalStore` (glossary), domain config

### 5. Learned Rules

- Rule text, scope (global / domain / source-specific), confidence score
- Source count (how many observations produced this rule)
- Applied count (how many times used in planning)
- Example: "When querying revenue by region, always join orders with regions table on region_id, not region_name"

**Source**: `LearningCompactor` rules store

### 6. Common Patterns

- Frequent query intents (from session history)
- Most-used tables/columns/endpoints
- Typical multi-step sequences (e.g., "users often ask revenue, then break down by region, then filter to top 5")

**Source**: Session history analysis, step metadata

### 7. Agents & Skills

- Available agents with descriptions and specializations
- Available skills with trigger patterns
- Domain-scoped vs. inherited

**Source**: Domain config (agents, skills)

### 8. Known Limitations

- Failed regression test assertions (what the domain gets wrong)
- Open bug queue items for this domain
- Sources with stale data (last refresh > threshold)
- Ungrounded glossary terms (defined but not connected to source data)

**Source**: `TestRunner` results, `BugQueue`, refresh metadata, grounding status

## API

```
GET /sessions/{session_id}/handbook
GET /sessions/{session_id}/handbook/{section}    # single section
```

### Response

```python
@dataclass
class DomainHandbook:
    domain: str
    generated_at: str                        # ISO timestamp
    summary: str                             # LLM-generated overview paragraph
    sections: dict[str, HandbookSection]     # keyed by section name

@dataclass
class HandbookSection:
    title: str
    content: list[HandbookEntry]
    last_updated: str                        # most recent data change in this section

@dataclass
class HandbookEntry:
    key: str                                 # entity name, rule id, source name, etc.
    display: str                             # rendered text
    metadata: dict                           # section-specific structured data
    editable: bool                           # whether user can modify this entry
```

## User Edits

Handbook entries with `editable: true` can be modified by users. Edits route to the appropriate store:

| Section | Edit target |
|---|---|
| Glossary | `RelationalStore` glossary (add/update term) |
| Learned Rules | Rule store (add/update/disable rule) |
| Overview | Domain config description |
| Known Limitations | Bug queue (acknowledge/dismiss) |
| Entities | Entity annotations (alias, correction) |

```
PUT /sessions/{session_id}/handbook/{section}/{key}
```

```python
@dataclass
class HandbookEdit:
    section: str
    key: str
    field: str          # which field changed
    old_value: str
    new_value: str
    reason: str | None  # optional user explanation
```

Edits are stored as handbook edit history for audit. The edit is applied to the underlying store immediately — the handbook re-renders on next fetch.

## Caching & Freshness

- Handbook is generated on demand, not stored as a static artifact
- Per-section staleness check: compare `last_updated` against underlying store timestamps
- LLM summary regenerated only when section content changes (hash comparison)
- Cached in session memory with TTL (default 5 minutes)

## Frontend

### DomainHandbookPage Component

```
/session/{id}/handbook
```

- Sidebar: section navigation (anchors)
- Main content: rendered sections with collapsible entries
- Edit mode: inline editing with save/cancel, reason field
- Freshness indicator per section (green = fresh, yellow = stale)
- Print/export: render as markdown or PDF for sharing

### Integration Points

- Link from domain selector dropdown → "View Handbook"
- Link from glossary panel → handbook glossary section
- Link from data sources panel → handbook sources section
- Link from regression test results → handbook limitations section

## Generation Pipeline

```
handbook_request
  → parallel fetch:
      config_store.get_domain_config(domain)
      relational_store.get_entities(domain)
      relational_store.get_glossary(domain)
      learning_store.get_rules(domain)
      session_store.get_source_metadata()
      session_history.get_patterns(domain)
      test_store.get_results(domain)
      bug_queue.get_open(domain)
  → assemble HandbookSection per section
  → if summary stale:
      llm.summarize(sections) → summary paragraph
  → return DomainHandbook
```

All fetches are parallel. LLM call only for summary generation, and only when content has changed.

## File Changes

| File | Change |
|---|---|
| `constat/server/routes/handbook.py` | New — handbook API routes |
| `constat/session/_handbook.py` | New — handbook generation mixin |
| `constat-ui/src/components/handbook/DomainHandbookPage.tsx` | New — handbook page component |
| `constat-ui/src/components/handbook/HandbookSection.tsx` | New — section renderer with inline edit |
| `constat-ui/src/api/sessions.ts` | Add `getHandbook()`, `updateHandbookEntry()` |
| `constat-ui/src/types/api.ts` | Add `DomainHandbook`, `HandbookSection`, `HandbookEntry`, `HandbookEdit` types |

## Testing

- Unit: mock stores → verify section assembly, edit routing, cache invalidation
- Integration: seed a domain with sources/glossary/rules → fetch handbook → verify all sections populated
- Edit round-trip: edit glossary term via handbook → verify `RelationalStore` updated → re-fetch handbook → verify change reflected
- Freshness: modify underlying store → verify staleness detection → verify summary regeneration
- Empty domain: domain with no sources/glossary/rules → verify graceful empty sections
