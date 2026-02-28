# Scale-Out: SQLAlchemy + Pluggable Vector Backend

> **Status:** Planned. DuckDB remains the default embedded backend.

## Problem

`DuckDBVectorStore` is a 3300-line monolith that mixes relational CRUD (entities, glossary, relationships, hashes) with vector operations (embedding storage, cosine similarity, BM25, hybrid search). This creates three scaling walls:

1. **Single-writer file lock** — DuckDB allows one writer per file. No horizontal scaling.
2. **Brute-force vector search** — `array_cosine_similarity` scans every row. No HNSW index despite VSS extension being loaded.
3. **Single shared connection** — all threads serialize through one connection with no application-level locking. Concurrent FastAPI handlers contend silently.

The current architecture works for single-user/demo. It breaks at ~100K embeddings or >1 concurrent user.

## Design

Split the monolith into two layers:

```
┌──────────────────────────────────────────────────┐
│              DuckDBVectorStore (current)          │
│  3300 lines, 60+ public methods, 9 tables        │
└──────────────────────────────────────────────────┘
                        ↓ refactor into
┌──────────────────────────────────────────────────┐
│                   RelationalStore                 │
│  SQLAlchemy ORM — works with any SQL database     │
│  Tables: entities, glossary_terms, relationships, │
│  chunk_entities, glossary_clusters, source_hashes,│
│  resource_hashes, document_urls                   │
├──────────────────────────────────────────────────┤
│              VectorBackend (ABC)                  │
│  Backend-specific embedding storage + search      │
│  Table: embeddings (vectors + content + metadata) │
│  Methods: add_chunks, search, search_by_source,   │
│  clear, count, delete_by_document                 │
└──────────────────────────────────────────────────┘
```

### Why Two Layers

| Concern | Relational (SQLAlchemy) | Vector (backend-specific) |
|---------|------------------------|--------------------------|
| SQL dialect | Standard — SELECT, INSERT, JOIN | Non-standard — `array_cosine_similarity`, `<=>`, HNSW indexes |
| Portability | Any SQLAlchemy-supported DB | Each DB has its own vector extension |
| Schema | 8 tables, views, indexes | 1 table with vector column |
| Operations | CRUD, visibility filters, joins | Similarity search, FTS, hybrid RRF |
| Scaling axis | Connection pooling, read replicas | ANN indexes, sharding by domain |

Trying to abstract vector operations through SQLAlchemy would require writing custom dialect extensions for every backend. Keeping vector ops backend-specific is simpler and lets each backend use its native strengths.

## Tables: Relational vs Vector

### Relational (→ SQLAlchemy)

```
entities              — NER-extracted entities (session-scoped)
chunk_entities        — junction: chunks ↔ entities
glossary_terms        — curated business definitions (user-scoped)
entity_relationships  — SVO triples
glossary_clusters     — KMeans cluster assignments
source_hashes         — config-level cache invalidation
resource_hashes       — resource-level cache invalidation
document_urls         — persisted source URLs for crawled docs
```

### Vector (→ VectorBackend)

```
embeddings            — chunks + 1024-dim FLOAT vectors + FTS content
```

The `embeddings` table is the only table that requires vector-specific operations. Everything else is standard relational SQL.

## VectorBackend ABC

Expand the existing `VectorStoreBackend` ABC (currently 4 methods) to cover all vector operations:

```python
class VectorBackend(ABC):
    """Backend-specific vector storage and similarity search."""

    @abstractmethod
    def add_chunks(self, chunks: list[DocumentChunk], embeddings: np.ndarray,
                   source: str, session_id: str | None, domain_id: str | None) -> None: ...

    @abstractmethod
    def search(self, query_embedding: np.ndarray, limit: int,
               domain_ids: list[str] | None, session_id: str | None,
               chunk_types: list[str] | None, query_text: str | None,
               ) -> list[tuple[str, float, DocumentChunk]]: ...

    @abstractmethod
    def search_by_source(self, query_embedding: np.ndarray, source: str,
                         limit: int, min_similarity: float,
                         query_text: str | None) -> list[tuple[str, float, DocumentChunk]]: ...

    @abstractmethod
    def delete_by_document(self, document_name: str) -> int: ...

    @abstractmethod
    def delete_by_source(self, source: str, domain_id: str | None) -> int: ...

    @abstractmethod
    def get_chunk_ids(self, session_id: str | None) -> list[str]: ...

    @abstractmethod
    def get_chunks_for_ids(self, chunk_ids: list[str]) -> list[DocumentChunk]: ...

    @abstractmethod
    def clear(self) -> None: ...

    @abstractmethod
    def count(self, source: str | None) -> int: ...
```

### Backend Implementations

**DuckDBVectorBackend** (existing behavior, extracted):
- `array_cosine_similarity()` brute-force scan
- DuckDB FTS extension for BM25
- RRF hybrid merge
- Optional cross-encoder reranking
- Future: add `CREATE INDEX ... USING HNSW` for datasets >50K chunks

**PgVectorBackend** (new):
- pgvector `<=>` cosine distance operator
- HNSW index: `CREATE INDEX ON embeddings USING hnsw (embedding vector_cosine_ops)`
- PostgreSQL `tsvector`/`tsquery` for FTS (replaces DuckDB FTS pragma)
- Connection pooling via SQLAlchemy engine (same engine as relational layer)

**Future backends** (same ABC):
- SQLite + sqlite-vec
- Qdrant/Pinecone/Weaviate (external vector DB, relational stays in PG)

## RelationalStore: SQLAlchemy Models

### Models

```python
# constat/storage/models.py

class EntityModel(Base):
    __tablename__ = "entities"
    id: Mapped[str] = mapped_column(primary_key=True)
    name: Mapped[str]
    display_name: Mapped[str]
    semantic_type: Mapped[str]
    ner_type: Mapped[str | None]
    session_id: Mapped[str]
    domain_id: Mapped[str | None]
    created_at: Mapped[datetime]

class ChunkEntityModel(Base):
    __tablename__ = "chunk_entities"
    chunk_id: Mapped[str] = mapped_column(primary_key=True)
    entity_id: Mapped[str] = mapped_column(primary_key=True)
    confidence: Mapped[float] = mapped_column(default=1.0)

class GlossaryTermModel(Base):
    __tablename__ = "glossary_terms"
    id: Mapped[str] = mapped_column(primary_key=True)
    name: Mapped[str]
    display_name: Mapped[str]
    definition: Mapped[str]
    domain: Mapped[str | None]
    parent_id: Mapped[str | None]
    parent_verb: Mapped[str] = mapped_column(default="HAS_KIND")
    aliases: Mapped[str | None]          # JSON array
    semantic_type: Mapped[str | None]
    cardinality: Mapped[str] = mapped_column(default="many")
    plural: Mapped[str | None]
    tags: Mapped[str | None]             # JSON dict
    owner: Mapped[str | None]
    status: Mapped[str] = mapped_column(default="draft")
    provenance: Mapped[str] = mapped_column(default="llm")
    session_id: Mapped[str]
    user_id: Mapped[str] = mapped_column(default="default")
    ignored: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime]
    updated_at: Mapped[datetime]

class EntityRelationshipModel(Base):
    __tablename__ = "entity_relationships"
    id: Mapped[str] = mapped_column(primary_key=True)
    subject_name: Mapped[str]
    verb: Mapped[str]
    object_name: Mapped[str]
    sentence: Mapped[str | None]
    confidence: Mapped[float] = mapped_column(default=1.0)
    verb_category: Mapped[str] = mapped_column(default="other")
    session_id: Mapped[str | None]
    user_edited: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime]

class GlossaryClusterModel(Base):
    __tablename__ = "glossary_clusters"
    term_name: Mapped[str] = mapped_column(primary_key=True)
    cluster_id: Mapped[int]
    session_id: Mapped[str] = mapped_column(primary_key=True)

class SourceHashModel(Base):
    __tablename__ = "source_hashes"
    source_id: Mapped[str] = mapped_column(primary_key=True)
    db_hash: Mapped[str | None]
    api_hash: Mapped[str | None]
    doc_hash: Mapped[str | None]
    updated_at: Mapped[datetime]

class ResourceHashModel(Base):
    __tablename__ = "resource_hashes"
    resource_id: Mapped[str] = mapped_column(primary_key=True)
    resource_type: Mapped[str]
    resource_name: Mapped[str]
    source_id: Mapped[str]
    content_hash: Mapped[str]
    updated_at: Mapped[datetime]

class DocumentUrlModel(Base):
    __tablename__ = "document_urls"
    document_name: Mapped[str] = mapped_column(primary_key=True)
    source_url: Mapped[str]
```

### RelationalStore Class

```python
# constat/storage/relational.py

class RelationalStore:
    """SQLAlchemy-backed relational storage for entities, glossary, relationships."""

    def __init__(self, engine: Engine):
        self._engine = engine
        self._Session = sessionmaker(bind=engine)

    # --- Entity methods ---
    def add_entities(self, entities: list[Entity], session_id: str) -> None: ...
    def find_entity_by_name(self, name: str, ...) -> Entity | None: ...
    def get_entity_by_id(self, entity_id: str) -> Entity | None: ...
    def clear_session_entities(self, session_id: str) -> None: ...
    # ... (all entity methods from DuckDBVectorStore)

    # --- Glossary methods ---
    def add_glossary_term(self, term: GlossaryTerm) -> None: ...
    def update_glossary_term(self, name: str, session_id: str, updates: dict, ...) -> bool: ...
    def get_glossary_term(self, name: str, session_id: str, ...) -> GlossaryTerm | None: ...
    def list_glossary_terms(self, session_id: str, ...) -> list[GlossaryTerm]: ...
    def get_unified_glossary(self, session_id: str, ...) -> list[dict]: ...
    # ... (all glossary methods)

    # --- Relationship methods ---
    def add_entity_relationship(self, rel: EntityRelationship) -> None: ...
    def get_relationships_for_entity(self, name: str, session_id: str) -> list[dict]: ...
    # ... (all relationship methods)

    # --- Cache invalidation ---
    def get_source_hash(self, source_id: str, hash_type: str) -> str | None: ...
    def set_source_hash(self, source_id: str, hash_type: str, hash_val: str) -> None: ...
    # ... (all hash methods)

    # --- Chunk-entity junction ---
    def link_chunk_entities(self, links: list[tuple]) -> None: ...
    def get_chunks_for_entity(self, entity_id: str, ...) -> list[str]: ...
    # returns chunk_ids only — caller fetches chunk content from VectorBackend

    # --- Visibility filters ---
    def entity_visibility_filter(self, session_id, active_domains) -> ...: ...
```

### Key Change: chunk_entities Junction

The `chunk_entities` table joins data from both layers — `chunk_id` lives in the vector backend, `entity_id` lives in the relational store. Two options:

**Option A: Junction table in relational store (recommended)**
- `chunk_entities` is a pure relational table (two VARCHAR foreign keys + confidence float)
- `get_chunks_for_entity()` returns chunk_ids, caller fetches content from VectorBackend
- Simple, no cross-database joins needed

**Option B: Duplicate chunk metadata in relational store**
- Store `document_name`, `source`, `section` in a relational `chunk_metadata` table
- Avoids needing to call VectorBackend for non-search queries
- More storage, but eliminates cross-layer lookups for glossary resource resolution

Option A is cleaner. Option B is faster for the glossary connected-resources path. Start with A, promote to B if latency is a problem.

## Composed Store

The main entry point that callers use, replacing `DuckDBVectorStore`:

```python
# constat/storage/store.py

class Store:
    """Composed relational + vector storage."""

    def __init__(self, relational: RelationalStore, vector: VectorBackend):
        self.relational = relational
        self.vector = vector

    # Delegate relational methods
    def add_entities(self, *a, **kw): return self.relational.add_entities(*a, **kw)
    def add_glossary_term(self, *a, **kw): return self.relational.add_glossary_term(*a, **kw)
    # ...

    # Delegate vector methods
    def add_chunks(self, *a, **kw): return self.vector.add_chunks(*a, **kw)
    def search(self, *a, **kw): return self.vector.search(*a, **kw)
    # ...

    # Cross-layer methods (need both)
    def get_chunks_for_entity(self, entity_id, domain_ids=None):
        chunk_ids = self.relational.get_chunk_ids_for_entity(entity_id)
        return self.vector.get_chunks_for_ids(chunk_ids)

    def search_enriched(self, query_embedding, ...):
        results = self.vector.search(query_embedding, ...)
        for chunk_id, score, chunk in results:
            chunk.entities = self.relational.get_entities_for_chunk(chunk_id)
        return results

    def extract_entities_for_session(self, session_id, ...):
        chunks = self.vector.get_all_chunks(...)
        # run NER, then:
        self.relational.add_entities(entities, session_id)
        self.relational.link_chunk_entities(links)
```

## Configuration

```yaml
# constat.yaml
storage:
  # Relational backend (SQLAlchemy URL)
  database_url: "duckdb:///~/.constat/constat.duckdb"    # embedded default
  # database_url: "postgresql://localhost/constat"        # production

  vector_store:
    backend: "duckdb"          # "duckdb" | "pgvector" | "sqlite-vec"
    db_path: null              # null = use database_url's DB for pgvector
    reranker_model: null
```

When `database_url` is a PostgreSQL URL and `backend` is `pgvector`, both layers use the same PostgreSQL database (same connection pool). When `database_url` is `duckdb://`, both layers use the same DuckDB file (backward compatible).

## Migration Phases

### Phase 1: Extract RelationalStore (behind the scenes)

**Goal:** SQLAlchemy models + RelationalStore class, tested against DuckDB via `duckdb_engine`.

1. Create `constat/storage/models.py` — SQLAlchemy ORM models for all 8 relational tables
2. Create `constat/storage/relational.py` — RelationalStore wrapping SQLAlchemy session
3. Create `constat/storage/vector_backend.py` — extract VectorBackend ABC
4. Create `constat/storage/duckdb_backend.py` — DuckDBVectorBackend (embeddings table only)
5. Create `constat/storage/store.py` — composed Store class
6. **Do not change any callers yet** — DuckDBVectorStore delegates to Store internally

```python
# Phase 1: DuckDBVectorStore becomes a thin wrapper
class DuckDBVectorStore(VectorStoreBackend):
    def __init__(self, ...):
        engine = create_engine(f"duckdb:///{db_path}")
        self._store = Store(
            relational=RelationalStore(engine),
            vector=DuckDBVectorBackend(db_path),
        )

    # All existing methods delegate to self._store
    def add_entities(self, *a, **kw):
        return self._store.add_entities(*a, **kw)
```

**Test:** All existing tests pass unchanged. No caller changes.

### Phase 2: Migrate callers to Store interface

**Goal:** Replace `vs._conn.execute(...)` raw SQL calls with Store methods.

There are ~20 modules that use the vector store. Most go through public methods, but several reach into `vs._conn.execute()` directly:

- `constat/server/routes/data/glossary.py` — `_resolve_entity_domain()` uses raw SQL
- `constat/discovery/glossary_generator.py` — entity query uses raw `_conn.execute()`
- `constat/server/routes/data/entities.py` — raw queries for entity stats
- `constat/discovery/concept_detector.py` — raw embedding lookups

Each raw SQL call becomes a RelationalStore or VectorBackend method.

### Phase 3: PgVectorBackend

**Goal:** PostgreSQL + pgvector as a production vector backend.

1. Create `constat/storage/pgvector_backend.py`
2. Implement VectorBackend ABC using pgvector operators
3. HNSW index creation: `CREATE INDEX ON embeddings USING hnsw (embedding vector_cosine_ops)`
4. PostgreSQL FTS via `tsvector`/`tsquery` (replaces DuckDB FTS pragma)
5. Connection pooling via SQLAlchemy engine (shared with RelationalStore)

### Phase 4: Connection pooling + concurrency

**Goal:** Proper connection management for both backends.

- SQLAlchemy engine with pool_size, max_overflow, pool_timeout
- Scoped sessions for FastAPI request lifecycle
- Write serialization for DuckDB backend (maintain single-writer constraint)
- Read replicas for PostgreSQL (optional, via SQLAlchemy binds)

## Callers Inventory (20 modules)

Files that import or use the vector store, grouped by change complexity:

**Uses public methods only (no changes needed after Phase 1):**
- `discovery/doc_tools/_core.py` — add_chunks, count
- `discovery/doc_tools/_access.py` — search, search_enriched
- `discovery/doc_tools/_entities.py` — add_entities, link_chunk_entities
- `discovery/unified_discovery.py` — search_enriched, search_similar_entities
- `discovery/relationship_extractor.py` — relationship methods
- `catalog/schema_manager.py` — add_chunks, clear_chunks, hash methods
- `catalog/api_schema_manager.py` — add_chunks, search_by_source
- `core/api/entity_manager.py` — extract_entities, clear methods
- `session/_analysis.py` — list_glossary_terms
- `session/_execution.py` — search methods
- `session/_metadata.py` — entity/glossary lookups
- `session/_prompts.py` — glossary lookups

**Uses raw `_conn.execute()` (needs Phase 2 refactoring):**
- `discovery/glossary_generator.py` — entity query with doc_names subquery
- `server/routes/data/glossary.py` — `_resolve_entity_domain()` raw SQL
- `server/routes/data/entities.py` — entity stats queries
- `discovery/concept_detector.py` — raw embedding lookups

**Creates DuckDBVectorStore instances (needs factory update):**
- `server/app.py` — warmup creates DocumentDiscoveryTools
- `server/session_manager.py` — session creation
- `catalog/schema_manager.py` — lazy init
- `catalog/api_schema_manager.py` — lazy init

## Current Scaling Bottlenecks to Fix

These exist independently of the SQLAlchemy migration and should be addressed:

| Issue | Current | Fix |
|-------|---------|-----|
| No HNSW index | Brute-force cosine scan | Add `CREATE INDEX ... USING HNSW` for DuckDB VSS |
| Double cosine computation | `search_by_source` computes similarity twice per row | Compute once in CTE or subquery |
| N+1 entity lookups | `search_enriched` does N queries | Batch query with `IN (chunk_ids)` |
| One-at-a-time inserts | `add_entities` inserts in loop with try/except | Use `executemany` + `ON CONFLICT DO NOTHING` |
| FTS full rebuild | Every write sets `_fts_dirty`, next search rebuilds | Incremental FTS or debounce rebuilds |
| No query timeout | Single connection blocks indefinitely | DuckDB `SET timeout` or wrap with asyncio timeout |
| Session data leaks | Orphaned data if server crashes | TTL-based cleanup on startup |

## Risks

1. **DuckDB SQLAlchemy dialect maturity** — `duckdb_engine` exists but is less battle-tested than PostgreSQL. Test thoroughly.
2. **JSON columns** — `aliases` and `tags` are stored as JSON strings. SQLAlchemy's `JSON` type works differently across backends. Use `Text` with manual JSON serialization for now.
3. **Visibility filter complexity** — `entity_visibility_filter` and `chunk_visibility_filter` build dynamic WHERE clauses. These need careful translation to SQLAlchemy expressions.
4. **Unified glossary view** — complex SQL view joining entities + glossary_terms. May need to be a query builder method rather than a DB view.
5. **Transaction semantics** — DuckDB has different transaction isolation than PostgreSQL. Test concurrent access patterns.
