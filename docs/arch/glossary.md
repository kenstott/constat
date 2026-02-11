# Glossary Architecture

> **Status:** Design. No implementation yet.

## Problem

Entities are extracted from schemas, APIs, and documents but lack curated business definitions. A "customer" entity exists because NER found it in a table name — but nobody has written down what "customer" means to the business. The book's semantic model (Appendix B) makes this explicit: every term needs a `definition` property that describes business meaning, not storage location.

Entities already connect to physical elements (DB tables, API endpoints, documents via chunk references). What's missing is the semantic layer on top: curated definitions, taxonomy relationships (parent/child, aliases), and the editorial workflow to maintain them.

## Core Idea

There is one user-facing concept: **the glossary**. It subsumes what was previously the entities UI. Every extracted entity is a glossary term. Most are self-describing — physical metadata communicates their meaning. Some get curated definitions where physical metadata is insufficient.

Two tables, one view. Entity name is the natural key.

```
┌─────────────┐         ┌─────────────────┐
│  entities   │         │ glossary_terms   │
│  (automated)│◄────────│ (curated)        │
│             │  name   │                  │
│ name        │ session │ name             │
│ sources     │         │ definition       │
│ chunks      │         │ aliases          │
│ ner_type    │         │ parent_id        │
│ session_id  │         │ status           │
└─────────────┘         └─────────────────┘
        │                       │
        └───────┬───────────────┘
                │
    ┌───────────▼────────────┐
    │   unified glossary     │
    │   (view / join)        │
    │                        │
    │ LEFT JOIN on (name,    │
    │   session_id)          │
    └────────────────────────┘
                │
    ┌───────────┼───────────────────┐
    │           │                   │
    ▼           ▼                   ▼
 Defined     Self-describing     Deprecated
 entity +    entity only         glossary term
 glossary    (no definition      with no matching
 match       needed)             entity (physical
                                 grounding lost)
```

The LEFT JOIN produces three categories:
- **Defined** — entity + glossary term match. Has physical grounding AND curated definition.
- **Self-describing** — entity exists, no glossary term. Physical metadata is sufficient.
- **Deprecated** — glossary term exists, no matching entity. The physical thing it described was removed. Surfaces in a deprecation queue for handling.

The inner join (entities that HAVE glossary terms) is the "defined" subset. Glossary terms with no entity match are deprecated — the grounding constraint enforced at the database level.

Entity name as natural key means multiple entities named "customer" from different sources (SCHEMA, API, CONCEPT) all share one glossary definition. Correct: the business meaning of "customer" doesn't depend on which system it was found in.

## Data Model

### Existing Table: `entities` (unchanged)

The automated NER extraction layer. Cleared and rebuilt freely on session start, project add/remove. No curated data here.

### New Table: `glossary_terms`

```sql
CREATE TABLE glossary_terms (
    id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,              -- Singular canonical form (matches entity.name)
    display_name VARCHAR NOT NULL,      -- Title case singular for UI
    definition TEXT NOT NULL,           -- Business meaning (the whole point)
    parent_id VARCHAR,                  -- Taxonomy parent (self-referential)
    aliases TEXT,                       -- JSON array of alternate names
    semantic_type VARCHAR,              -- CONCEPT, ATTRIBUTE, ACTION, TERM
    cardinality VARCHAR DEFAULT 'many', -- many | distinct | singular
    plural VARCHAR,                     -- Irregular plural form (person->people)
    list_of VARCHAR,                    -- Glossary term ID if collection
    status VARCHAR DEFAULT 'draft',     -- draft | reviewed | approved
    provenance VARCHAR DEFAULT 'llm',   -- llm | human | hybrid
    session_id VARCHAR NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (parent_id) REFERENCES glossary_terms(id),
    FOREIGN KEY (list_of) REFERENCES glossary_terms(id)
);

CREATE INDEX idx_glossary_name ON glossary_terms(name);
CREATE INDEX idx_glossary_parent ON glossary_terms(parent_id);
CREATE INDEX idx_glossary_session ON glossary_terms(session_id);
CREATE INDEX idx_glossary_status ON glossary_terms(status);
```

No junction table. The join key is `(name, session_id)` between `entities` and `glossary_terms`.

### Unified Glossary View

```sql
-- The main glossary: all entities, with definitions where they exist
CREATE VIEW unified_glossary AS
SELECT
    e.id AS entity_id,
    e.name,
    COALESCE(g.display_name, e.display_name) AS display_name,
    e.semantic_type,
    e.ner_type,
    e.session_id,
    g.id AS glossary_id,
    g.definition,
    g.parent_id,
    g.aliases,
    g.cardinality,
    g.plural,
    g.list_of,
    g.status,
    g.provenance,
    CASE
        WHEN g.id IS NOT NULL THEN 'defined'
        ELSE 'self_describing'
    END AS glossary_status
FROM entities e
LEFT JOIN glossary_terms g
    ON e.name = g.name
    AND e.session_id = g.session_id;

-- Deprecated: glossary terms whose physical grounding was lost
CREATE VIEW deprecated_glossary AS
SELECT g.*
FROM glossary_terms g
LEFT JOIN entities e
    ON g.name = e.name
    AND g.session_id = e.session_id
WHERE e.id IS NULL;
```

### Dataclass

```python
@dataclass
class GlossaryTerm:
    id: str
    name: str                                       # Singular canonical form
    display_name: str                               # Title case singular
    definition: str
    parent_id: Optional[str] = None
    aliases: list[str] = field(default_factory=list)
    semantic_type: Optional[str] = None
    cardinality: str = "many"                       # many | distinct | singular
    plural: Optional[str] = None                    # Irregular plural (person->people)
    list_of: Optional[str] = None                   # Glossary term ID if collection
    status: str = "draft"                           # draft | reviewed | approved
    provenance: str = "llm"                         # llm | human | hybrid
    session_id: str = ""
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
```

### Plurals and Collections

Following the book's semantic model:

**Cardinality** — how many instances a term represents:
- `many` (default) — 0..N instances, duplicates allowed
- `distinct` — 0..N instances, all unique
- `singular` — exactly one in this domain (singleton, e.g., "CEO")

**Plural** — irregular plural form. Default pluralization adds 's'. Override for irregular forms:
- person -> people
- datum -> data
- criterion -> criteria
- index -> indices

The existing `singularize()` in `discovery/models.py` already handles these for entity normalization. Glossary terms store the singular canonical form in `name` (matching entity names) and the irregular plural in `plural` when needed.

**Collections (list_of)** — when a plural is a distinct business concept with its own definition, not just "0..N of the thing." From the book:

```yaml
# Simple plural: just cardinality
employee:
  definition: An individual employed by the organization
  cardinality: many
  plural: employees

# Collection: a distinct concept
team:
  definition: A group of employees working toward a shared objective
  list_of: employee   # references another glossary term
```

The `list_of` field references another glossary term by ID. A collection is grounded through its element type — if "employee" is grounded, "team" is grounded.

## When a Term Needs a Definition

Not every entity needs a curated definition. Most don't. Physical metadata is often self-describing and sufficient — a column named `email_address` in a `customers` table doesn't need a business definition. The starting assumption is that physical metadata speaks for itself. The glossary definition exists to fill in what physical metadata cannot.

From the book: *"Leave embedded when containment is sufficient. If the term only matters within that document's scope, if nothing else references it, if elevating it would create governance overhead without corresponding value — leave it."*

**Add a definition when:**

| Signal | Why it needs a definition |
|--------|--------------------------|
| Ambiguous | `cust_status` values 1, 2, 3 — what do they mean? |
| Contested | Two departments define "customer" differently |
| Cross-referenced | Same concept appears in DB, API, and policy docs |
| Opaque | Column `ltv_score` — no one outside analytics knows this |
| Policy-defined | "For purposes of this document, 'eligible' means..." |
| Abstract | Category that organizes concrete terms but has no table of its own |

**Leave self-describing when:**

| Signal | Why it doesn't need a definition |
|--------|----------------------------------|
| Self-describing | `email_address`, `created_at`, `first_name` |
| Single-scope | Only matters within one table or one API |
| Generic action | `get`, `create`, `delete` — verbs with obvious meaning |
| Low mention count | Appeared once in one chunk |

Self-describing entities are still in the glossary UI, still searchable, still discoverable. They just don't have (or need) a curated definition. Any user can add a definition to any self-describing term at any time via `[+ Define]`.

## Phase 1: Two-Output Discovery Pipeline

### 1a: Glossary Generation (selective definitions for entities)

Glossary generation runs automatically as the second step of entity extraction. It selectively adds definitions where physical metadata is insufficient.

**Trigger:** Entity extraction completes -> glossary generation runs as follow-on step.

**Candidate selection** — The LLM decides which entities need definitions. Filter out obvious non-candidates first, then let the LLM apply the elevation test to the rest.

Pre-filter (skip — never candidates):
- Generic actions: get, create, update, delete, list, set
- Single-mention entities with `ner_type` == SCHEMA and self-describing names
- Entities with `mention_count` == 1 and only one source type

Pass to LLM for elevation decision (candidates):
- `semantic_type` in (CONCEPT, TERM) — likely candidates
- `semantic_type` == ATTRIBUTE with mention_count >= 3
- Entities appearing across multiple source types (DB + API, DB + document, etc.)
- Entities with aliases or contested usage in chunks

**Context assembly** — For each candidate, gather:
- The entity itself (name, type, sources)
- Top 3 chunks by confidence (the text that mentions this entity)
- Co-occurring entities (what appears alongside it)
- Source metadata (which DB/API/doc it comes from)

**LLM call** — Batch candidates into groups of ~20. The LLM makes the final elevation decision per-entity:

```
You are building a business glossary from extracted entities. Not every entity
needs a glossary entry — most physical metadata is self-describing (email_address,
created_at, first_name). The glossary exists only for terms that NEED semantic
help: ambiguous, contested, cross-referenced, opaque, or policy-defined terms.

For each entity below, first decide: does this term need a glossary entry?
- YES if: the name is ambiguous, the concept is contested across teams, it appears
  across multiple systems, the name is opaque without context, or a policy defines
  it with specific meaning
- NO if: the name is self-describing, it only matters within one scope, or the
  physical metadata already communicates its meaning

For entities that need a glossary entry, write a business definition that describes
what this concept means — not where it is stored or what system it comes from.

Good definition: "An individual or organization with an active commercial
relationship, identified by account number and tracked through the customer
lifecycle."

Bad definition: "Customer data stored in the CRM PostgreSQL database."

For each entity, also suggest:
- A parent category (if one exists among the other entities)
- Aliases (other names people use for this concept)
- Confidence (high/medium/low) in the definition

Context for each entity includes source chunks showing how the term is used.

Entities:
{batched_entities_with_context}

Respond as JSON array (omit entities that don't need glossary entries):
[{
  "name": "...",
  "definition": "...",
  "parent": "..." or null,
  "aliases": ["..."],
  "confidence": "high|medium|low"
}]
```

**Storage** — Store results as `glossary_terms` with `provenance: "llm"`, `status: "draft"`. The name matches the entity name — no FK needed. Embed as `ChunkType.GLOSSARY` chunks immediately.

**Integration point** — Use existing `constat/providers/router.py` task routing:

```python
class TaskType(str, Enum):
    ...
    GLOSSARY_GENERATION = "glossary_generation"
```

Route to a capable model (Claude Sonnet or equivalent). Batch operation, not latency-sensitive.

### 1b: Glossary Embedding

Each glossary term is embedded as a chunk. New `ChunkType`:

```python
class ChunkType(str, Enum):
    ...
    GLOSSARY = "glossary"
```

The embedded content includes the definition plus the connected physical resources (resolved from entities sharing the same name). When the LLM retrieves a glossary chunk, it gets both the meaning AND the paths to follow.

```python
def glossary_term_to_chunk(
    term: GlossaryTerm,
    entity_sources: list[str],  # resolved from entities with matching name
) -> DocumentChunk:
    """Build an embeddable chunk from a glossary term with its physical resources."""
    parts = [f"{term.display_name}: {term.definition}"]

    if term.aliases:
        parts.append(f"Also known as: {', '.join(term.aliases)}")

    if term.parent_display_name:
        parts.append(f"Category: {term.parent_display_name}")

    # Physical resources from matching entities
    if entity_sources:
        parts.append("Connected resources:")
        for source in entity_sources:
            parts.append(f"  - {source}")

    return DocumentChunk(
        document_name=f"glossary:{term.id}",
        content="\n".join(parts),
        source="glossary",
        chunk_type=ChunkType.GLOSSARY,
    )
```

Example embedded content for "Customer":
```
Customer: An individual or organization with an active commercial relationship,
identified by account number and tracked through the customer lifecycle.
Also known as: client, account, buyer
Category: Stakeholder
Connected resources:
  - crm.public.customers (DB)
  - GET /api/v1/customers (API: user-service)
  - business_rules.md §3.1 Customer Policies (Document)
```

When vector search returns this glossary chunk, the LLM immediately has:
1. What the term means (definition)
2. Where the physical data lives (from matching entities)
3. Multiple paths to follow (query the DB, call the API, read the doc)

 ### 1c: Physical Resource Resolution

A glossary term may be concrete (direct entity match) or abstract (grounded through children or `list_of`). Resolving a term to its physical resources requires walking the taxonomy tree until entities are found. Ungrounded paths are pruned — the RDB equivalent of what would be a single Cypher traversal in a graph store.

```python
def resolve_physical_resources(
    term_name: str,
    session_id: str,
) -> list[dict]:
    """Walk from glossary term to physical resources, pruning ungrounded paths.

    Handles three grounding paths:
    1. Direct: term name matches entities -> return their sources
    2. Collection: term.list_of -> resolve the element type
    3. Taxonomy: term.children -> resolve each child recursively

    Returns empty list if ungrounded (prune signal).
    """
    # Direct match — concrete term, trivial case
    entities = get_entities_by_name(term_name, session_id)
    if entities:
        return [
            {
                "entity_name": e.display_name,
                "entity_type": e.semantic_type,
                "sources": get_entity_chunk_sources(e.id),
            }
            for e in entities
        ]

    # No direct entities — try indirect paths
    term = get_glossary_term_by_name(term_name, session_id)
    if not term:
        return []  # ungrounded, prune

    # Collection: follow list_of to element type
    if term.list_of:
        target = get_glossary_term(term.list_of)
        if target:
            resources = resolve_physical_resources(target.name, session_id)
            if resources:
                return resources

    # Taxonomy: follow children, collect their resources
    resources = []
    for child in get_child_terms(term.id):
        child_resources = resolve_physical_resources(child.name, session_id)
        resources.extend(child_resources)

    return resources  # empty = ungrounded, prune this term
```

Example resolution paths:

```
Stakeholder (abstract)            # no direct entities
├── Customer (concrete)           # entities found: crm.customers, GET /api/customers
│   └── [crm (DB), user-service (API)]
├── Employee (concrete)           # entities found: idp.employees
│   └── [corp-idp (IDP)]
└── Partner (concrete)            # entities found: crm.partners
    └── [crm (DB)]

Team (collection)                 # no direct entities
└── list_of: Employee             # resolved through element type
    └── [corp-idp (IDP)]

Forecast (abstract, UNGROUNDED)   # no direct entities
├── Revenue Forecast              # no direct entities, no children
└── (empty)                       # prune — returns []
```

The empty return is the RDB equivalent of the inner join's filtering. If the recursion bottoms out with no entities, the term is ungrounded and excluded from search results.

**Performance:** This is fine at glossary scale (tens to low hundreds of defined terms, shallow taxonomy trees — rarely more than 3 levels deep). If it becomes a bottleneck, materialize grounding paths into a `grounded_terms` table rebuilt after entity extraction — pre-computing the recursive walk. But that's optimization, not architecture.

### 1d: Search Result Shape

Search returns chunks. The chunk type determines what gets resolved and returned. The resolution logic uses `resolve_physical_resources` to walk multi-hop paths for abstract/collection terms.

**When a glossary chunk scores highest:**

```python
def resolve_glossary_result(chunk, score) -> dict:
    """Resolve a glossary chunk into a full result with physical resources."""
    term = get_glossary_term(chunk.document_name)  # "glossary:{id}"
    resources = resolve_physical_resources(term.name, term.session_id)

    if not resources:
        return None  # ungrounded — exclude from results

    return {
        "type": "glossary",
        "score": score,
        "term": {
            "name": term.display_name,
            "definition": term.definition,
            "aliases": term.aliases,
            "parent": term.parent_display_name,
            "status": term.status,
        },
        "connected_resources": resources,
    }
```

**When a physical chunk (schema/API/doc) scores highest:**

```python
def resolve_physical_result(chunk, score, entities) -> dict:
    """Resolve a physical chunk, attaching glossary definitions if they exist."""
    # Find glossary terms for entities in this chunk (name-based lookup)
    glossary_terms = get_glossary_for_entity_names(
        [e.name for e in entities],
        session_id,
    )
    return {
        "type": chunk.source,  # "schema", "api", "document"
        "score": score,
        "content": chunk.content,
        "document": chunk.document_name,
        "section": chunk.section,
        "entities": [entity_to_dict(e) for e in entities],
        "glossary": [
            {
                "name": t.display_name,
                "definition": t.definition,
                "aliases": t.aliases,
            }
            for t in glossary_terms
        ],
    }
```

### 1e: Unified Search Tool

One search tool. It searches the entire vector space — schema chunks, API chunks, document chunks, and glossary chunks compete together. The chunk type determines how each result is resolved, but the agent doesn't need to know or care. It just searches.

```python
def search(query: str, limit: int = 10) -> list[dict]:
    """Search across all sources: databases, APIs, documents, and glossary.

    Returns the highest-scoring results regardless of source type.
    Each result includes:
    - Physical results: content + glossary definition (if one exists)
    - Glossary results: definition + connected physical resources

    Follow-up actions depend on the resource type in the result:
    - DB sources: use get_table_schema, then write SQL
    - API sources: use get_operation_details, then call the API
    - Document sources: use get_document_section for full context
    """
    query_embedding = embed(query)
    raw_results = vector_store.search(query_embedding, limit=limit)

    resolved = []
    for chunk, score in raw_results:
        if chunk.chunk_type == ChunkType.GLOSSARY:
            result = resolve_glossary_result(chunk, score)
            if result:  # None = ungrounded, skip
                resolved.append(result)
        else:
            entities = get_entities_for_chunk(chunk.id)
            resolved.append(resolve_physical_result(chunk, score, entities))

    return resolved
```

The tool description tells the agent how to interpret results, not which tool to pick:

```
Search across all data sources and the business glossary.

Results come in two forms:
- "glossary" results: a curated business definition with connected physical
  resources. The definition tells you what the term means. The connected
  resources tell you where the data lives.
- Physical results ("schema", "api", "document"): raw content from a data
  source. May include a "glossary" field with the business definition if
  one exists for entities in this result.

To act on results, follow the resource type:
- DB -> get_table_schema -> SQL query
- API -> get_operation_details -> API call
- Document -> get_document_section -> text retrieval
```

This replaces `search_documents`, `search_glossary`, and `search_all` with a single tool. The vector space is unified; the resolution logic handles the rest.

## Phase 2: Unified Glossary UI

### 2a: Glossary Browser (replaces EntityAccordion)

One panel replaces both the entity list and the glossary. Every extracted entity appears. Terms with curated definitions show them. Terms without show physical metadata only.

**GlossaryPanel.tsx** — Top-level component:
- Tree view showing taxonomy hierarchy (parent -> children, defined terms only)
- Flat list view with search/filter (all terms)
- Toggle between tree and flat views
- Filter: [All | Defined | Self-describing]
- Filter by status (draft/reviewed/approved), semantic_type
- Search by name, definition text, aliases

```
┌─────────────────────────────────────────────┐
│ Glossary                      [Tree|List]   │
│ ┌─────────────────────────────────────────┐ │
│ │ Search...                               │ │
│ └─────────────────────────────────────────┘ │
│ [All] [Defined] [Self-describing]  [Clear]  │
│                                             │
│ ▼ Customer                       CONCEPT    │
│   "An individual or organization with       │
│    an active commercial relationship"       │
│   [crm (DB)] [user-api (API)]   approved   │
│                                             │
│ ▶ email_address                  SCHEMA     │
│   crm.public.customers.email    no defn    │
│                                [+ Define]   │
│                                             │
│ ▶ cust_status                    SCHEMA     │
│   "Customer lifecycle state: 1=active,      │
│    2=suspended, 3=churned"                  │
│   crm.public.customers.cust_status  draft  │
│                                             │
│ ▶ GET /api/customers             API        │
│   user-service                  no defn    │
│                                [+ Define]   │
└─────────────────────────────────────────────┘
```

Default view: **Defined** — shows only terms with curated definitions. Keeps noise low. Switch to **All** to see everything, **Self-describing** to browse physical metadata.

**Deprecation queue** — separate view (or filter) for glossary terms whose entities disappeared. Shown when `deprecated_glossary` view has rows:

```
┌─────────────────────────────────────────────┐
│ Deprecated Terms (3)            [Dismiss All]│
│                                              │
│ ▶ legacy_score                    SCHEMA     │
│   "Risk score from retired model"            │
│   ⚠ No matching entity — source removed    │
│   [Reassign] [Archive] [Delete]              │
└─────────────────────────────────────────────┘
```

### 2b: Glossary Editor

**Add definition to any term** — `[+ Define]` button on self-describing terms. Click opens inline editor. Creates a glossary_terms row with matching name.

**Inline editing** — Click to edit definition, aliases, parent, status on defined terms. Changes update `provenance` to "hybrid" if term was LLM-generated.

**Status workflow:**
- `draft` -> `reviewed` -> `approved` (forward only in normal flow)
- Any status can be sent back to `draft` for re-editing
- Definition edits trigger re-embedding of the glossary chunk

**AI editing assistance:**

1. **Refine Definition** — Button on each defined term. Sends current definition + entity context to LLM: "Improve this business definition. Keep it concise. Describe meaning, not storage."

2. **Suggest Taxonomy** — Button on glossary panel. LLM analyzes all defined terms and suggests parent/child relationships. User accepts/rejects each suggestion.

3. **Suggest Aliases** — Per-term button. LLM suggests alternate names based on chunk contexts where the entity appears under different names.

4. **Bulk Review** — Select multiple draft terms. LLM reviews definitions for consistency, flags issues (storage-centric language, vague definitions, missing relationships).

### API Endpoints

```
GET    /sessions/{id}/glossary                  — Unified view (filterable: all/defined/self_describing)
GET    /sessions/{id}/glossary/deprecated       — Deprecated terms (no matching entity)
GET    /sessions/{id}/glossary/{name}           — Term detail + physical resources
POST   /sessions/{id}/glossary/generate         — Re-trigger LLM generation
POST   /sessions/{id}/glossary                  — Add definition to a term (name + definition)
PUT    /sessions/{id}/glossary/{name}           — Update definition
DELETE /sessions/{id}/glossary/{name}           — Remove definition (term reverts to self-describing)
POST   /sessions/{id}/glossary/{name}/refine    — AI-assisted refinement
POST   /sessions/{id}/glossary/suggest-taxonomy — AI taxonomy suggestions
PATCH  /sessions/{id}/glossary/bulk-status      — Bulk status update
```

Note: endpoints use `name` not synthetic ID — consistent with the natural key join.

### State Management

Zustand store, same pattern as existing stores:

```typescript
interface GlossaryStore {
  terms: UnifiedGlossaryTerm[]       // From unified_glossary view
  deprecatedTerms: GlossaryTerm[]    // From deprecated_glossary view
  selectedName: string | null
  viewMode: 'tree' | 'list'
  filters: {
    scope?: 'all' | 'defined' | 'self_describing'
    status?: string
    type?: string
    search?: string
  }

  fetchTerms: (sessionId: string) => Promise<void>
  addDefinition: (sessionId: string, name: string, definition: string) => Promise<void>
  updateTerm: (sessionId: string, name: string, updates: Partial<GlossaryTerm>) => Promise<void>
  generateGlossary: (sessionId: string) => Promise<void>
  refineTerm: (sessionId: string, name: string) => Promise<void>
}
```

## Phase 3: Semantic Model Bridge (Future)

Once the glossary is curated, it becomes the basis for generating semantic model YAML (the book's format). Each approved glossary term maps to a `BusinessDefinition`:

```yaml
definitions:
  customer:
    definition: "An individual or organization with an active commercial relationship"
    aliases: [client, account, buyer]
    kinds:
      retail:
        definition: "Individual consumers purchasing for personal use"
        resourcePath: "public/v_retail_customers"  # from entity's chunk refs
      enterprise:
        definition: "Business entities under negotiated contracts"
        resourcePath: "public/v_enterprise_customers"
```

This phase connects the glossary back to the book's vision: curated business definitions that ground to physical resources.

## Implementation Order

| Phase | Scope | Depends On |
|-------|-------|------------|
| 1a | `glossary_terms` table + `GlossaryTerm` dataclass | Nothing |
| 1b | `unified_glossary` + `deprecated_glossary` views | 1a |
| 1c | LLM glossary generation as follow-on to entity extraction | 1a |
| 1d | Glossary embedding with physical resources in chunk content | 1a, 1c |
| 1e | Physical resource resolution (multi-hop walk, ungrounded pruning) | 1d |
| 1f | Unified search tool (single tool, chunk-type-based resolution) | 1e |
| 1g | API endpoints (unified glossary, CRUD, deprecation) | 1b |
| 2a | Unified GlossaryPanel (replaces EntityAccordion) | 1g |
| 2b | `[+ Define]`, inline editing, status workflow, re-embedding | 2a |
| 2c | Deprecation queue UI | 2a |
| 2d | AI editing assistance (refine, suggest taxonomy, suggest aliases) | 2b |
| 3  | Semantic model YAML export | 2b |

## The Grounding Constraint

From the book: *"All paths through the ontology must terminate in physical reality. No floating concepts."*

The natural key join enforces this at the database level. A glossary term is grounded if it has matching entities. No matching entities = deprecated.

For taxonomy terms (abstract parents), grounding works through children:

- **Concrete** — has matching entities (the join finds them)
- **Abstract** — a category/parent whose children are concrete (recursively)
- **Collection** — `list_of` target is grounded
- **Deprecated** — no matching entities, no grounded children

```
┌──────────────────────────────────────────────┐
│ ▼ Stakeholder                       CONCEPT  │
│   "A party with interest in..."              │
│                                              │
│   ⚠ No direct resources (abstract term)     │
│   Grounded through children:                 │
│   ├── Customer ✓  3 resources               │
│   ├── Employee ✓  2 resources               │
│   └── Partner  ✓  1 resource                │
│                                              │
│ ▼ Revenue Forecast                  CONCEPT  │
│   "Projected income..."                     │
│                                              │
│   ⚠ UNGROUNDED — no resources, no children  │
│   [Link to Entity] [Add Child]               │
└──────────────────────────────────────────────┘
```

The UI surfaces grounding status alongside editorial status:

| Status | Grounded | Meaning |
|--------|----------|---------|
| draft + grounded | Definition needs review, but physical connections exist |
| approved + grounded | Complete — definition curated, physical paths established |
| draft + ungrounded | Needs both definition work and entity linking |
| approved + ungrounded | Definition is good but term floats — needs children or entity links |

```python
def is_grounded(term_name: str, session_id: str) -> bool:
    """Check if a term is grounded in physical reality.

    A term is grounded if:
    - It has matching entities (name join finds them), OR
    - Any child in the taxonomy is grounded (abstract parent), OR
    - Its list_of target is grounded (collection)
    """
    if entities_exist_for_name(term_name, session_id):
        return True
    term = get_glossary_term_by_name(term_name, session_id)
    if term and term.list_of:
        target = get_glossary_term(term.list_of)
        if target and is_grounded(target.name, session_id):
            return True
    children = get_child_terms(term.id) if term else []
    return any(is_grounded(c.name, session_id) for c in children)
```

## Design Decisions

**Why entity name as natural key?** Eliminates the junction table. The join is `(name, session_id)` — no synthetic FKs. Multiple entities with the same name from different sources (SCHEMA, API, CONCEPT) all share one glossary definition. Entity re-extraction can freely clear and rebuild without touching glossary terms. Glossary terms persist independently.

**Why a view, not a merged table?** Entities have a destructive lifecycle (clear + rebuild). Glossary terms have a curative lifecycle (draft -> reviewed -> approved). Keeping them in separate tables with a view means each can follow its own lifecycle without interfering. The view is the merge point.

**Why deprecation instead of deletion?** When an entity disappears (data source removed, schema changed), the glossary definition may still be valuable — it represents institutional knowledge about a concept that existed. Deprecation preserves the definition while surfacing that its physical grounding is lost. The user decides: reassign it, archive it, or delete it.

**Why one UI, not two?** Entities and glossary terms are the same thing at different levels of curation. Showing them in separate panels forces users to context-switch between "what exists" and "what it means." One unified panel shows both, with filters to focus on what matters.

**Why selective definitions, not define-everything?** Physical metadata is self-describing. `email_address` doesn't need a business definition. The glossary exists to fill in what physical metadata cannot communicate. This avoids the overhead of maintaining definitions for self-evident terms while ensuring opaque, ambiguous, or contested terms get the semantic help they need.

**Why batch LLM generation, not per-entity?** Context. A term's definition improves when the LLM sees related terms together. "Customer" is better defined when "Order", "Product", and "Account" are in the same prompt. Batches of ~20 balance context quality against token limits.

**Why embed drafts?** The glossary is a pipeline output, not an editorial artifact that gates on approval. Search should surface all defined terms so the LLM can use them. Bad definitions get fixed through the editorial flow, not by hiding them from search.