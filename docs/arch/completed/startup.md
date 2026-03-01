# Startup Performance Architecture

> **Status:** All optimizations implemented (O1-O4, cluster perf, batch inserts, scope-cached NER, frontend parallel restore, glossary timeout).

## Problem

First connection after a server restart takes ~60 seconds before the user can type a query. Subsequent reconnections are near-instant because the session cache hits. The latency is split across three sequential phases: server boot, session creation, and frontend restore.

## Startup Sequence

```
Server boot                          Session creation              Frontend
────────────────────────────────── ──────────────────────────────── ──────────────────

EmbeddingModelLoader.get_model() ─┐
                                  │ blocking
_warmup_vector_store()  ──────────┘
                                   POST /sessions ─────────────────┐
                                     CoreMixin.__init__             │
                                       SchemaManager.initialize()   │ blocking
                                       APISchemaManager.initialize()│ (HTTP 200
                                       MetadataPreloadCache         │  not sent
                                       DocumentDiscoveryTools       │  until done)
                                       ConceptDetector.initialize() │
                                     _load_domains_into_session()   │
                                       add_database_dynamic() × N ─┘
                                     refresh_entities_async() ──── background

                                                                    HTTP 200
                                                                    GET /messages
                                                                    GET /proof-facts
                                                                    GET /steps (parallel)
                                                                    GET /inference-codes
                                                                    WS connect
```

## Phase 1: Server Lifespan (once at boot)

**File:** `constat/server/app.py:430-449`

Two blocking operations run sequentially before the server accepts connections:

### 1a. Embedding Model Load

```python
EmbeddingModelLoader.get_instance().start_loading()
EmbeddingModelLoader.get_instance().get_model()   # blocks until ready
```

Loads `BAAI/bge-large-en-v1.5` (1024-dim). First run downloads the model (~1.2 GB). Subsequent boots load from HuggingFace cache.

### 1b. Vector Store Warmup

```python
_warmup_vector_store(config)
```

**File:** `constat/server/app.py:168-406`

Pre-indexes all schemas, APIs, and documents from base config + every domain. Uses two-level hash-based cache (source-level → resource-level) to skip unchanged resources. On a cold start with no `vectors.duckdb`, this introspects every database and embeds every document. On a warm start with nothing changed, it completes in < 1 second.

Flow per source (base + each domain):
1. Compute `db_hash`, `api_hash`, `doc_hash` from config
2. Compare against stored hashes in vector store
3. If unchanged → skip
4. If changed → check each resource individually, rebuild only what changed

## Phase 2: Session Creation (per connection)

**Endpoint:** `POST /api/sessions` → `constat/server/routes/sessions.py:239-296`

The HTTP response is not sent until all blocking steps complete. Three sub-phases execute sequentially.

### 2a. CoreMixin.__init__

**File:** `constat/session/_core.py:53-249`

| Step | Component | What it does | Blocking? |
|------|-----------|-------------|-----------|
| 1 | `EmbeddingModelLoader.start_loading()` | No-op if already loaded (server lifespan did it) | No |
| 2 | `SchemaManager(config).initialize()` | Connects to all base DBs, introspects schemas, builds overview | **Yes** |
| 3 | `APISchemaManager(config).initialize()` | Introspects all base APIs (GraphQL/REST) | **Yes** |
| 4 | `MetadataPreloadCache` | Caches schema metadata for LLM context | Fast |
| 5 | `DocumentDiscoveryTools(config)` | Initializes doc tools reference | Fast |
| 6 | `TaskRouter`, `Planner`, `PythonExecutor` | Lightweight object construction | Fast |
| 7 | `ConceptDetector.initialize()` | Builds concept patterns for prompt injection | Fast |
| 8 | `IntentClassifier` | Lightweight init | Fast |

Steps 2-3 dominate. Each opens live DB/API connections and introspects metadata. They run sequentially.

### 2b. Domain Loading

**File:** `constat/server/routes/sessions.py:94-236`

```python
_load_domains_into_session(managed, preferred_domains)
```

Three phases:

1. **Phase 1 — Databases:** For each domain, calls `schema_manager.add_database_dynamic(name, db_config)` sequentially. Each call connects, introspects, and builds chunks.
2. **Phase 2 — Documents:** No-op; documents were pre-indexed during server warmup.
3. **Phase 3 — NER update:** Passes all schema entity names to `doc_tools.set_schema_entities()` and runs schema metadata through NER.

Domain APIs are also registered but this is lightweight (no introspection at load time).

### 2c. Entity Extraction (async)

```python
session_manager.refresh_entities_async(session_id)
```

**File:** `constat/server/session_manager.py:615-649`

Spawns a background thread. Does **not** block the HTTP response. Runs NER over all visible chunks, extracts entities, builds clusters. Frontend receives `ENTITY_REBUILD_START` and `ENTITY_REBUILD_COMPLETE` events via WebSocket.

## Phase 3: Frontend Restore

**File:** `constat-ui/src/store/sessionStore.ts:161-246`

After receiving the session HTTP 200:

| Step | Call | Blocks UI? |
|------|------|-----------|
| 1 | `POST /sessions` | **Yes** (awaited) |
| 2 | `GET /sessions/{id}/messages` | **Yes** (sequential) |
| 3 | `GET /sessions/{id}/proof-facts` | **Yes** (sequential) |
| 4 | `GET /sessions/{id}/steps` + `GET /sessions/{id}/inference-codes` | **Yes** (parallel via `Promise.all`) |
| 5 | `wsManager.connect()` | No (fire-and-forget) |

Steps 2-3 are sequential. Step 4 runs two fetches in parallel. New sessions (no history to restore) complete steps 2-4 nearly instantly.

## Timing Breakdown

| Component | Estimated Cost | Blocking? | When |
|-----------|---------------|-----------|------|
| Embedding model load | 30-60s cold / <1s warm | Yes | Server boot |
| Vector store warmup | 1-120s (proportional to data) | Yes | Server boot |
| `SchemaManager.initialize()` | 2-15s (per base DB) | Yes | Session create |
| `APISchemaManager.initialize()` | 1-5s (per base API) | Yes | Session create |
| `add_database_dynamic()` × N | 2-15s per domain DB | Yes | Session create |
| NER metadata pass | 1-3s | Yes | Session create |
| Entity extraction + clustering | 5-60s | **No** (background) | After HTTP 200 |
| Frontend restore calls | <1s (new) / 1-3s (reconnect) | Yes (client) | After HTTP 200 |

**Typical first-session wall clock:** Embedding (warm) + warmup (warm) + SchemaManager + domains ≈ 5-30s depending on database count and network latency.

**Worst case (cold boot, no cache):** Embedding download + full index + schema introspection ≈ 60-120s.

## Optimization Opportunities

### O1. Parallel base initialization in CoreMixin

`SchemaManager.initialize()` and `APISchemaManager.initialize()` are independent. Running them concurrently (e.g., `concurrent.futures.ThreadPoolExecutor`) would save the full duration of the shorter one.

**Impact:** Save 1-5s per session. Low risk — they share no mutable state.

### O2. Parallel domain database loading

`_load_domains_into_session` calls `add_database_dynamic()` sequentially in a loop. Each call is independent (different DB connection). Parallelizing with a thread pool would reduce N × T to ~T.

**Impact:** Save (N-1) × 2-15s for N domain databases. Medium risk — SchemaManager internal state needs a lock around `metadata_cache` updates.

### O3. Frontend parallel restore

Messages and proof-facts fetches (`getMessages`, `getProofFacts`) are sequential but independent. Wrapping all four restore calls in a single `Promise.all` would save one round-trip.

**Impact:** Save ~100-300ms on reconnect. Zero risk.

### O4. Lazy ConceptDetector

`ConceptDetector.initialize()` runs during `__init__` but is only needed when the first query arrives. Deferring to first use would shave time off session creation.

**Impact:** Small (typically <1s). Zero risk — already has lazy init pattern in agent/skill matchers.

### O5. Connection pooling for domain databases

Each `add_database_dynamic` opens a new connection. A shared connection pool across sessions would avoid repeated TCP handshake + auth for the same database.

**Impact:** Save 0.5-2s per domain DB per session. Medium complexity — requires lifecycle management.

## Glossary Population: Latency and Stuck-State Bug

### Observed Behavior

- With ~2,500 entities, glossary takes 40+ seconds to appear after session creation.
- On new sessions, the glossary panel sometimes never populates (gets stuck empty).

### Timeline

```
T=0ms     POST /sessions returns HTTP 200
T=0ms     refresh_entities_async() spawns background thread
T=50ms    GlossaryPanel mounts → useEffect fires fetchTerms()
T=50ms    GET /glossary returns EMPTY (NER still running)
          ↓
          User sees empty glossary panel
          ↓
T=10-15s  NER extraction completes (entity_extractor.extract per chunk)
T=15s     _rebuild_clusters() starts KMeans
T=40-50s  KMeans finishes (k ≈ 250 clusters on 2,500 vectors × 1024 dims)
T=40-50s  ENTITY_REBUILD_COMPLETE pushed to event_queue
T=40-50s  Frontend receives event → fetchTerms() → glossary renders
```

### The "Stuck" Bug

**File:** `constat/server/session_manager.py:630-645`

If `_run_entity_extraction()` or `_rebuild_clusters()` throws an exception, the error is logged but `ENTITY_REBUILD_COMPLETE` is **never sent**:

```python
def _run():
    try:
        self._push_event(managed, EventType.ENTITY_REBUILD_START, {...})
        self._run_entity_extraction(session_id, managed.session)
        self._push_event(managed, EventType.ENTITY_REBUILD_COMPLETE, {...})  # skipped on exception
    except Exception as e:
        logger.exception(f"refresh_entities_async({session_id}): failed: {e}")
        # No ENTITY_REBUILD_COMPLETE or error event sent
```

The frontend only re-fetches the glossary on `entity_rebuild_complete` (line 1167-1176 in `sessionStore.ts`). If that event never arrives, the panel stays empty forever. There is no timeout, no retry, and no error event.

### Event Delivery

Events use an `asyncio.Queue` on the `ManagedSession` (line 65). Events pushed before WebSocket connects **accumulate in the queue** and drain once the WS consumer starts — so events are not lost due to WS timing. However, the queue is **cleared on reconnect** if the session is in a stuck status (lines 739-743 in `queries.py`).

### Why 40+ Seconds

The 40s is dominated by `_rebuild_clusters()` (`vector_store.py:3126-3222`):

| Step | Cost at 2,500 entities |
|------|----------------------|
| Fetch entity embeddings (SQL join) | ~1s |
| Accumulate + average vectors | ~1s |
| Glossary override loop (O(G × N) linear scan) | ~2-5s |
| KMeans fit (k=250, N=2500, dim=1024) | **25-35s** |
| Insert cluster assignments (row-by-row) | ~2-3s |

### Scale Projection

| Entity Count | k (clusters) | KMeans Est. | Glossary Override | Total Cluster Time |
|-------------|-------------|-------------|-------------------|-------------------|
| 2,500 | 250 | 25-35s | 2-5s | ~40s |
| 10,000 | 1,000 | 3-8 min | 30-60s | ~5-10 min |
| 100,000 | 10,000 | 30-60 min | hours (O(N²)) | hours |

The glossary override loop at `vector_store.py:3174-3180` is O(G × N) — for each glossary term, it linearly scans all entity keys for case-insensitive match. At 100k entities this becomes quadratic.

### Fixes

**Bug fix (stuck state):** Add a `finally` block to `refresh_entities_async._run()` that always sends a completion or error event:

```python
def _run():
    try:
        self._push_event(managed, EventType.ENTITY_REBUILD_START, {...})
        self._run_entity_extraction(session_id, managed.session)
        self._push_event(managed, EventType.ENTITY_REBUILD_COMPLETE, {...})
    except Exception as e:
        logger.exception(...)
        self._push_event(managed, EventType.ENTITY_REBUILD_COMPLETE, {
            "session_id": session_id, "error": str(e),
        })
```

**Performance — cluster rebuild:**

1. **Cap k:** `k = min(max(2, N // 10), 500)` — diminishing returns beyond ~500 clusters.
2. **Use MiniBatchKMeans:** Drop-in replacement, ~10× faster for large N.
3. **Hash-based glossary override:** Replace the O(G × N) linear scan with a `dict` keyed by `name.lower()`.
4. **Batch inserts:** Replace row-by-row cluster assignment inserts with `executemany()`.

**Frontend resilience:**

1. **Timeout + retry:** If no `entity_rebuild_complete` within 60s, re-fetch anyway.
2. **Progressive loading:** Fetch glossary from entities table before clusters are ready (clusters are only used for the "similar terms" sidebar, not the main list).

## Scope-Cached NER

### Problem

The current model runs full NER on every new session. With 2,500 entities this takes 40+ seconds; at 100k+ it would take hours. Yet two sessions with identical active domains produce identical entity results — the work is entirely redundant.

A per-session fingerprint cache exists (`ner_fingerprint.py`) but only helps **reconnects** to the same session. A new session with the same domains recomputes everything.

### Design

Cache NER results at the **scope level** (base config + active domain set), not per-session. On session creation, copy cached results into the session instead of re-running NER.

```
Server boot:
  _warmup_vector_store()           existing — hash-based chunk cache
  _warmup_entities(config)         NEW — run NER for base config, store as scope cache

Session create:
  Compute scope fingerprint = hash(chunk_ids + schema_terms + api_terms + business_terms)
  If fingerprint matches a cached scope:
    Copy cached entities → session-scoped rows         (SQL INSERT...SELECT, fast)
    Copy cached chunk_entity links → session-scoped    (SQL INSERT...SELECT, fast)
    Copy cached cluster assignments → session-scoped   (SQL INSERT...SELECT, fast)
    Send ENTITY_REBUILD_COMPLETE immediately
  Else:
    Run NER in background (current path)
    Store result as new scope cache entry

Domain add/remove (mid-session):
    Scope changed → new fingerprint → cache miss → re-run NER
    (incremental path via extract_entities_for_project still applies)

Manual refresh (user-initiated):
    DELETE scope cache for current fingerprint
    Re-introspect datasources (rebuild chunks with fresh schema/content)
    Re-run NER + clustering
    Store new scope cache entry
```

### Cache Key

The fingerprint from `compute_ner_fingerprint` already encodes the complete NER input surface:

```python
fingerprint = hash(sorted(chunk_ids) + sorted(schema_terms) + sorted(api_terms) + sorted(business_terms))
```

Any change to the datasource collection — domain added, database added, document modified, schema changed — alters at least one of these inputs, producing a different fingerprint. The cache is **safe by construction**: stale results cannot be served because the key itself changes.

### Invalidation Events

| Event | Chunks change? | Patterns change? | Fingerprint changes? | Action |
|---|---|---|---|---|
| Domain added | No (pre-indexed) | Yes (new schema terms) | **Yes** | Auto cache miss → re-run |
| Domain removed | No | Yes (terms removed) | **Yes** | Auto cache miss → re-run |
| Database added mid-session | Yes | Yes | **Yes** | Auto cache miss → re-run |
| Document added mid-session | Yes | No | **Yes** | Auto cache miss → re-run |
| API added mid-session | Yes | Yes | **Yes** | Auto cache miss → re-run |
| Schema change at boot | Yes (chunks rebuilt) | Yes | **Yes** | Auto cache miss → re-run |
| External datasource mutation | No (stale) | No (stale) | No | **Manual refresh required** |
| User edits glossary term | No | No | No | No re-run needed |
| User sets taxonomy parent | No | No | No | No re-run needed |

The only gap is **external datasource mutation between server restarts** — a table renamed, columns added, API spec updated, document edited on disk. The server cannot detect these without re-introspecting. The "Refresh entities" button covers this case.

### Manual Refresh

The user-facing "Refresh entities" action handles the case where the user knows a datasource has changed upstream but the server hasn't restarted:

1. Re-introspect affected datasources (SchemaManager, APISchemaManager re-connect and re-read)
2. Rebuild chunks for changed resources (same hash-based flow as server warmup)
3. Recompute fingerprint (now different, since chunks/terms changed)
4. Run full NER against new chunks
5. Rebuild clusters
6. Store as new scope cache entry
7. Send `ENTITY_REBUILD_COMPLETE`

This is intentionally user-initiated because re-introspection is expensive and the server has no signal that external state changed.

### Storage

Scope cache lives in the existing `vectors.duckdb` database:

```
ner_scope_cache
  fingerprint    TEXT PRIMARY KEY    -- scope fingerprint (hash)
  created_at     TIMESTAMP          -- when NER ran
  entity_count   INTEGER            -- for diagnostics

ner_cached_entities
  fingerprint    TEXT               -- FK to ner_scope_cache
  entity fields...                  -- same columns as entities table

ner_cached_chunk_entities
  fingerprint    TEXT               -- FK to ner_scope_cache
  chunk_id, entity_id, confidence  -- same columns as chunk_entities table

ner_cached_clusters
  fingerprint    TEXT               -- FK to ner_scope_cache
  term_name, cluster_id            -- same columns as glossary_clusters table
```

On session create with cache hit: `INSERT INTO entities SELECT ... FROM ner_cached_entities WHERE fingerprint = ?` with session_id substituted. Same for chunk_entities and clusters. This is a single SQL statement per table — no row-by-row iteration.

Cache eviction: LRU by `created_at`, keep last N fingerprints (default 10). Old entries cleaned up during `start_cleanup_task`.

### Expected Impact

| Scenario | Current | With scope cache |
|---|---|---|
| New session, same domains as previous | 40-60s (full NER) | 1-2s (SQL copy) |
| New session, never-seen domain combo | 40-60s | 40-60s (cache miss, same as today) |
| Reconnect to existing session | <1s (session cache hit) | <1s (unchanged) |
| Server restart, same config | 40-60s per first session | 1-2s (scope cache survives in DuckDB) |
| Manual refresh | N/A (not available) | 40-60s (full re-introspect + NER) |

## Design Decisions

### Why server warmup exists

Without warmup, the first session would pay the full indexing cost (embedding model + chunk generation for all sources). Moving this to server boot means the cost is paid once, not per-session.

### Why entity extraction is async

Entity extraction depends on which domains are active (session-scoped), so it cannot run at server boot. It runs in a background thread after the HTTP 200 returns so the user sees the session immediately. The glossary panel shows a loading spinner until `ENTITY_REBUILD_COMPLETE` arrives via WebSocket.

### Why scope-cached NER instead of per-session NER

NER inputs are deterministic given a set of chunks and pattern vocabulary. Two sessions with identical active domains produce identical entities. Running NER per-session is O(sessions × chunks × patterns) redundant work. Caching at the scope level reduces this to O(unique_scopes × chunks × patterns) with O(1) SQL copy per subsequent session.

The cache is invalidated automatically when the input surface changes (domain add/remove, source add/remove, schema change). The only uncovered case — external datasource mutations between server restarts — is handled by a user-initiated "Refresh entities" action, since the server has no way to detect these changes without re-introspection.

### Why CoreMixin.__init__ is synchronous

The session object must be fully constructed before any tool or planner can use it. SchemaManager and APISchemaManager results are consumed immediately by downstream components (Planner, FactResolver, doc_tools). Making init async would require guarding every downstream access with "is ready?" checks, adding complexity for marginal gain.
