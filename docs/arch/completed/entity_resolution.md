# Entity Resolution — Phase 1

## Problem

Documents reference both structural metadata ("the orders table") and specific data values ("France"). The system embeds metadata (schema, API specs) and document content, and NER recognizes table/column names and standard spaCy entities. But there is no way to:

1. Pull specific entity values from data sources (e.g., country names from a countries table)
2. Add those values as custom NER patterns so they're recognized in documents
3. Embed those values so semantic search finds them ("French" → "France")
4. Point back to the source for record-level resolution

## Solution

Configure `entity_resolution` entries in domain (or top-level) config that map entity types to data sources. At session startup, values are extracted, embedded as chunks in the vector store, and registered as NER patterns. NER then extracts entities from those chunks (and from documents that reference them), making every configured entity value appear in the glossary.

## Entity Management Overview

Entities come from three distinct sources, each with its own management approach:

### 1. Schema/API Entities (metadata)
- **Source**: Database table/column names, API endpoints
- **Discovery**: Automatic during schema introspection
- **NER label**: `SCHEMA`, `API`
- **entity_class**: `metadata_entity`
- **Config**: `databases`, `apis` sections

### 2. Document Entities (mixed)
- **Source**: spaCy NER on document text
- **Discovery**: Statistical NER (ORG, PERSON, GPE, etc.)
- **NER label**: spaCy labels (ORG, PERSON, GPE, EVENT, LAW, PRODUCT)
- **entity_class**: `mixed`
- **Config**: `documents` section

### 3. Data Entities (entity resolution)
- **Source**: Actual record values from databases, APIs, or static lists
- **Discovery**: Configured via `entity_resolution` — values extracted, embedded as chunks, then NER extracts them
- **NER label**: Custom per entity_type (CUSTOMER, COUNTRY, DEPARTMENT, etc.)
- **entity_class**: `data_entity`
- **Config**: `entity_resolution` section (see below)

## Config

```yaml
entity_resolution:
  # SQL shorthand — table + name_column
  - entity_type: CUSTOMER
    source: sales            # References a configured database
    table: customers
    name_column: customer_name
    max_values: 10000        # Optional cap (default 10000)

  # Custom SQL query
  - entity_type: EMPLOYEE
    source: hr
    query: "SELECT DISTINCT first_name || ' ' || last_name AS name FROM employees"

  # GraphQL API
  - entity_type: COUNTRY
    source: countries         # References a configured API (type: graphql)
    query: "{ countries { name } }"
    items_path: countries
    name_field: name

  # REST API
  - entity_type: BREED
    source: catfacts          # References a configured API (type: openapi)
    endpoint: /breeds
    items_path: data
    name_field: breed

  # NoSQL — Neo4j Cypher
  - entity_type: CUSTOMER
    source: customer_graph
    query: "MATCH (n:Customer) RETURN n.name AS name"

  # Static list
  - entity_type: CURRENCY
    values: [USD, EUR, GBP, JPY]
```

Multiple sources for the same entity type merge — NER patterns combine, each source gets its own summary chunk and individual value embeddings.

Discriminator priority: `values` → static. `endpoint` → REST API. `query` + API source → GraphQL. `query` + DB source → custom query. `table` → SQL shorthand.

### Config Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `entity_type` | string | required | Custom NER label (e.g., CUSTOMER, COUNTRY) |
| `source` | string | `""` | Database or API name from config |
| `table` | string | — | Table to query (SQL shorthand) |
| `name_column` | string | — | Column with entity names (SQL shorthand) |
| `query` | string | — | Custom query (SQL, Cypher, GraphQL, CQL, etc.) |
| `endpoint` | string | — | REST GET endpoint path |
| `items_path` | string | — | JSON path to items array in response |
| `name_field` | string | `"name"` | Field in each item containing the entity name |
| `values` | list | — | Inline static values (no query needed) |
| `max_values` | int | `10000` | Cap on distinct values extracted |

## Architecture

### Files Modified

| File | Change |
|------|--------|
| `constat/core/config.py` | `EntityResolutionConfig` model, `entity_resolution` field on `DomainConfig` |
| `constat/discovery/models.py` | `ChunkType.ENTITY_VALUE`, `EntityClass` constants, `entity_class` on `Entity` |
| `constat/discovery/vector_store.py` | `entity_class` column on `embeddings`, `entities`, `ner_cached_entities` tables |
| `constat/storage/duckdb_backend.py` | Accept `entity_resolution` source, set `entity_class` on insert |
| `constat/storage/relational.py` | `entity_class` in entity INSERT/upsert and NER scope cache save/restore |
| `constat/discovery/entity_extractor.py` | `entity_terms` parameter, custom EntityRuler patterns, `entity_class` on entities |
| `constat/discovery/doc_tools/_entities.py` | Pass `entity_terms` through to `EntityExtractor` |
| `constat/discovery/doc_tools/_core.py` | `embed_entity_values()` — summary + individual value chunks |
| `constat/discovery/ner_fingerprint.py` | Include `entity_terms` in fingerprint hash |
| `constat/server/session_manager.py` | Orchestration: extract values → embed chunks → NER extraction |
| `constat/catalog/schema_manager.py` | `extract_entity_values()` with SQL, NoSQL, GraphQL, REST, static dispatch |

### Data Flow

```
Session Init
  │
  ├─ Collect entity_resolution configs (top-level + active domains)
  │
  ├─ schema_manager.extract_entity_values(configs)
  │   ├─ SQL shorthand: SELECT DISTINCT name_column FROM table
  │   ├─ Custom SQL: execute query, extract first column
  │   ├─ NoSQL: cypher() / execute_cql() / query_sql() / query()
  │   ├─ GraphQL: POST query to API URL, navigate items_path
  │   ├─ REST: GET endpoint, navigate items_path, extract name_field
  │   └─ Static: use values directly
  │   → {entity_type: [value1, value2, ...]}
  │
  ├─ embed_entity_values()  ← BEFORE NER
  │   ├─ Summary chunk per type+source (index 0)
  │   └─ Individual value chunks (index 1+)
  │       → source="entity_resolution", entity_class="data_entity"
  │       → document_name = source reference (e.g., "sales.customers")
  │
  ├─ NER fingerprint includes entity_terms (cache invalidation)
  │
  └─ extract_entities_for_session()
      ├─ EntityExtractor with entity_terms (EntityRuler patterns, confidence 0.95)
      ├─ NER runs on ALL visible chunks (documents + entity_resolution chunks)
      └─ Entities created from matches → appear in glossary with entity_class
```

**Critical ordering**: Entity value chunks must be embedded **before** NER extraction runs. NER extracts entities from those chunks, which is how every configured entity value (e.g., "Corner Store") appears in the glossary — even if no document mentions it.

### Chunk Naming

Entity resolution chunks use the source reference directly as `document_name` — no prefix. The `source` column (`entity_resolution`) already distinguishes them:

| Source type | document_name | Example |
|-------------|---------------|---------|
| SQL (table) | `{db}.{table}` | `sales.customers` |
| Custom query | `{db}` | `hr` |
| GraphQL API | `{api_name}` | `countries` |
| REST API | `{api_name}` | `catfacts` |
| Static | `static` | `static` |

This allows the glossary UI to navigate directly to the source — clicking `sales.customers` opens the table view, same as schema links.

### Vector Store Classification

The `entity_class` column classifies records across both tables:

| Source | entity_class (embeddings) | entity_class (entities) |
|--------|--------------------------|------------------------|
| `schema`, `api` | `metadata_entity` | `metadata_entity` |
| `entity_resolution` | `data_entity` | `data_entity` |
| `document` | `mixed` | `mixed` |

The `Entity` model carries `entity_class` through the full pipeline: EntityExtractor → relational store → NER scope cache save/restore.

### NER Behavior

- EntityRuler patterns run before spaCy NER, so custom entity types (COUNTRY) take precedence over spaCy's statistical types (GPE)
- Custom entity types get confidence 0.95 (higher than schema 0.9 and spaCy 0.75)
- Entity resolution entities get `entity_class=data_entity`; schema/API get `metadata_entity`; spaCy NER gets `mixed`
- The `_label_to_semantic_type()` default fallback to CONCEPT handles unknown labels

### NER Scope Cache

The scope cache (`ner_cached_entities`) preserves `entity_class` through save/restore:
- `store_ner_scope_cache()` copies `entity_class` from `entities` to `ner_cached_entities`
- `restore_ner_scope_cache()` restores `entity_class` back to `entities`
- Fingerprint includes `entity_terms` so changes to entity resolution config invalidate the cache

### Query-Time Behavior

No changes needed to search tools. Existing vector search handles this naturally:
- Search for "France" → cosine similarity finds entity value chunks (exact match, highest score) + document chunks mentioning France
- Results include `source` field → callers distinguish entity resolution hits from document hits
- `document_name` like `"sales.customers"` identifies which data source has the record
- `search_all()`, `find_entity()`, `explore_entity()` all return entity resolution matches transparently
