# Entity Resolution — Phase 1

## Problem

Documents reference both structural metadata ("the orders table") and specific data values ("France"). The system embeds metadata (schema, API specs) and document content, and NER recognizes table/column names and standard spaCy entities. But there is no way to:

1. Pull specific entity values from data sources (e.g., country names from a countries table)
2. Add those values as custom NER patterns so they're recognized in documents
3. Embed those values so semantic search finds them ("French" → "France")
4. Point back to the source for record-level resolution

## Solution

Configure `entity_resolution` entries in domain (or top-level) config that map entity types to data sources. At session startup, values are extracted, registered as NER patterns, and embedded in the vector store.

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

## Architecture

### Files Modified

| File | Change |
|------|--------|
| `constat/core/config.py` | `EntityResolutionConfig` model, `entity_resolution` field on `DomainConfig` and `Config` |
| `constat/discovery/models.py` | `ChunkType.ENTITY_VALUE`, `EntityClass` constants |
| `constat/discovery/vector_store.py` | `entity_class` column on `embeddings` and `entities` tables |
| `constat/storage/duckdb_backend.py` | Accept `entity_resolution` source, set `entity_class` on insert |
| `constat/discovery/entity_extractor.py` | `entity_terms` parameter, custom EntityRuler patterns per type |
| `constat/discovery/doc_tools/_entities.py` | Pass `entity_terms` through to `EntityExtractor` |
| `constat/discovery/doc_tools/_core.py` | `embed_entity_values()` — summary + individual value chunks |
| `constat/discovery/ner_fingerprint.py` | Include `entity_terms` in fingerprint hash |
| `constat/server/session_manager.py` | Orchestration: collect configs, extract values, pass to NER + embed |
| `constat/catalog/schema_manager.py` | `extract_entity_values()` with SQL, NoSQL, GraphQL, REST, static dispatch |

### Data Flow

```
Session Init
  │
  ├─ Collect entity_resolution configs (top-level + active domains)
  │
  ├─ schema_manager.extract_entity_values(configs)
  │   ├─ SQL: SELECT DISTINCT via SQLAlchemy engine
  │   ├─ NoSQL: cypher() / execute_cql() / query_sql() / query()
  │   ├─ GraphQL: POST query to API URL
  │   ├─ REST: GET endpoint, navigate items_path
  │   └─ Static: use values directly
  │   → {entity_type: [value1, value2, ...]}
  │
  ├─ NER fingerprint includes entity_terms (cache invalidation)
  │
  ├─ EntityExtractor with entity_terms
  │   └─ EntityRuler patterns per type (confidence 0.95)
  │
  └─ embed_entity_values()
      ├─ Summary chunk per type+source (index 0)
      └─ Individual value chunks (index 1+)
          → stored as source="entity_resolution", entity_class="data_entity"
```

### Vector Store Classification

The `entity_class` column enables query-time filtering:

| Source | entity_class (embeddings) | entity_class (entities) |
|--------|--------------------------|------------------------|
| `schema`, `api` | `metadata_entity` | `metadata` (default) |
| `entity_resolution` | `data_entity` | `data` |
| `document` | `mixed` | varies by NER type |

### NER Behavior

- EntityRuler patterns run before spaCy NER, so custom entity types (COUNTRY) take precedence over spaCy's statistical types (GPE)
- Custom entity types get confidence 0.95 (higher than schema 0.9 and spaCy 0.75)
- The `_label_to_semantic_type()` default fallback to CONCEPT handles unknown labels

### Query-Time Behavior

No changes needed to search tools. Existing vector search handles this naturally:
- Search for "France" → cosine similarity finds entity value chunks (exact match, highest score) + document chunks mentioning France
- Results include `source` field → callers distinguish entity resolution hits from document hits
- `document_name` like `"entity:hr.departments"` identifies which data source has the record
- `search_all()`, `find_entity()`, `explore_entity()` all return entity resolution matches transparently
