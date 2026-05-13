# Chonk Capability Roadmap + constat Migration

All additions to Chonk are backwards-compatible unless marked otherwise. constat
deprecations follow each Chonk feature once it lands. Phases are ordered by
dependency — nothing in a later phase requires breaking changes to an earlier one.

---

## Baseline: What Chonk Already Ships (Benchmarked)

The following is the complete Implicit GraphRAG stack described in the paper
"Implicit GraphRAG: Knowledge Graph Signals Without LLM-Based Graph Construction."
**Zero index-time LLM calls.** All=**0.685** confirmed on GraphRAG-Bench
(Medical + Novel); **0.691** projected with redundancy pruning (pending full-set
verification).

| Component | Chonk location | Benchmark signal |
|---|---|---|
| Semantic boundary chunking | `chunk_document()` | baseline |
| Contextual prefix embedding | `enrich_chunks()` | baseline |
| NER → `EntityIndex` | `chonk.ner` | P1 |
| Co-occurrence edges | `EntityIndex` + `CooccurrenceMatrix` | P1 |
| Entity-ref-expansion + lane filtering | `EnhancedSearch` | P1 +0.017 |
| Louvain/Leiden community detection | `ClusterMap` | P2 |
| Community context injection | `EnhancedSearch` → `AnswerContext` | P2 +0.015 |
| Widened retrieval k=10 | `EnhancedSearch` | P3 +0.024 |
| Redundancy pruning (cosine ≥ 0.92) | `EnhancedSearch` | P3 +0.006 [VERIFY] |
| Vector + BM25 hybrid store | `Store` | — |

**Enterprise note (§6.5 of paper):** entity-dense corpora (legal, financial,
medical, technical) resemble the Medical benchmark domain, which gains 3× more
from k=10 than narrative corpora. Recommended defaults for enterprise: k=10–15,
lane-sim=0.45, coherence=0.50, pruning enabled.

---

## Implementation Status

| Phase | Component | Status |
|---|---|---|
| 1.1 | `namespace` param on chonk `Store`/`DuckDBVectorBackend` | NOT STARTED — constat uses `domain_ids` instead |
| 1.2 | `chonk/schema.py` (`TableMeta`, `ColumnMeta`, `EndpointMeta`, `FieldMeta`) | DONE — `chonk/schema.py` |
| 1.2 | `DocumentLoader.load_schema()` / `load_api()` | DONE — on `DocumentLoader` |
| 1.3 | `infer_parquet()` | DONE — `chonk/_struct_inference.py`, used by `constat/catalog/file/connector.py` |
| 1.3 | `DocumentLoader.load_structured_file()` | DONE |
| 1.4 | `SchemaMatcher` in `chonk/ner/_schema.py` | DONE — two-pass NER active in `vector_store._run_entity_extraction()` |
| 1.5 | `DocumentChunk` re-exported from `chonk.models` | DONE |
| 1.5 | `ChunkType` enum | DONE — kept as `str, Enum` in `constat/discovery/models.py` (not deleted) |
| 2 | `EnhancedSearch` wired into `find_relevant_tables` + `search_documents` | DONE — `constat/storage/_enhanced_adapter.py` |
| 2 | `chunk_types` pre-filtering via `_EnhancedStoreAdapter.search()` | DONE — `chunk_types` forwarded through adapter; not passed to `EnhancedSearch.__init__` |
| 2 | structural expansion | DISABLED — `structural_expansion=False`; constat chunk ID hash includes `section` |
| 2 | cluster expansion | DISABLED — `cluster_expansion=False`; no `ClusterMap` threaded through |
| 2 | Replace `vector_store.py` with chonk `Store` | NOT STARTED — `NumpyVectorStore`, `VectorStoreBackend` dead code still present |
| 3 | `chonk/generation/` (`AnswerGenerator`, `PromptBuilder`, `AnswerContext`) | DONE — `chonk/generation/` |
| 3 | `answer_from_documents()` wired to `AnswerGenerator` | DONE — `_access.py`; `_chonk_llm.py` adapts constat providers |
| 3 | `ChonkConfig` + per-feature `ChonkModelSpec` | DONE — `source_config.py`; `ner_model`, `answer_llm`, `community_llm`, `svo_llm`, `embed_model`, `search_mode` |
| 4.1 | `SVOTriple`, `VERB_SET`, `SVOExtractor`, `RelationshipIndex` in `chonk/graph/` | DONE — FK-derived triples in `schema_manager._build_relationship_index()`; LLM extraction via `SVOExtractor` available |
| 4.1 | SVO triples merged and wired into retrieval | DONE — `app.py` merges FK triples across schema managers; `doc_tools._relationship_index` set at warmup |
| 4.1 | `graph_first` / `auto` mode wired into constat | DONE — `_resolve_search_mode()` auto-selects; `mode` + `relationship_index` forwarded through `search_enhanced` → `run_enhanced_search` |
| 4.1 | LLM-driven SVO extraction via `SVOExtractor` | NOT STARTED — `svo_llm` config wired, no call site yet |
| 4.2 | LLM community summaries | DONE — `chonk/community/_summarizer.py` |
| 4.2 | Community summaries wired into constat | NOT STARTED — `community_llm` config wired, `community_summaries: bool` flag present |
| 4.3 | `global` mode on `EnhancedSearch` | DONE — `EnhancedSearch.search(mode="global")` |
| 4.3 | `global` mode wired into constat | NOT STARTED — blocked by 4.2 (no community_summary chunks yet) |

---

## Phase 1 — Core Primitives (no breaking changes)

These are purely additive. Existing Chonk consumers are unaffected.

### 1.1 Namespace scoping on `Store`

**Why:** constat's vector store partitions embeddings by `project_id` — a single
shared table, one WHERE clause at search time composes a domain from multiple
independently-indexed corpora without re-embedding. Chonk's `Store` currently
has no filter parameters.

**Change:** add optional `namespace: str | None` column to `embeddings` table.

```python
store.add_document(chunks, embeddings, namespace="__base__")
store.search(query, namespaces=["__base__", "project_abc"])
```

`namespace=None` (default) searches everything — fully backwards compatible.
Internally: `WHERE namespace IN (...)` added to the existing SQL; HNSW still
drives the search.

**constat migration:** replace `project_id` column logic in `vector_store.py`
with `namespace` parameter. `'__base__'` maps to `namespace="__base__"`;
project IDs map directly.

---

### 1.2 `load_schema()` and `load_api()`

**Why:** constat hand-builds `DocumentChunk` objects for DB tables/columns and
API endpoints/fields. This logic belongs in Chonk — structured metadata is just
another document type.

**New models** (`chonk/schema.py`):

```python
@dataclass
class ColumnMeta:
    name: str
    data_type: str
    description: str | None = None
    nullable: bool = True
    is_primary_key: bool = False
    is_foreign_key: bool = False
    foreign_key_ref: str | None = None     # "other_table.col"

@dataclass
class TableMeta:
    name: str
    schema_name: str | None = None
    description: str | None = None
    columns: list[ColumnMeta] = field(default_factory=list)
    row_count: int | None = None
    source_db: str | None = None

@dataclass
class FieldMeta:
    name: str
    field_type: str
    description: str | None = None
    required: bool = False

@dataclass
class EndpointMeta:
    path: str
    method: str | None = None
    description: str | None = None
    fields: list[FieldMeta] = field(default_factory=list)
    endpoint_type: str = "rest"            # "rest" | "graphql_query" | "graphql_mutation" | "graphql_type"
    source_api: str | None = None
```

**New methods** on `DocumentLoader`:

```python
loader.load_schema(tables: list[TableMeta]) -> list[DocumentChunk]
loader.load_api(endpoints: list[EndpointMeta]) -> list[DocumentChunk]
```

**Chunking strategy — split-tables pattern:** each table produces N+1
independently embedded chunks (one table chunk + one per column). Same for
endpoints + fields. Fine-grained embeddings mean a query about a specific column
retrieves that column's chunk directly.

- Table chunk: `chunk_type="db_table"`, `document_name=f"schema:{source_db}.{name}"`, `section=["table_description"]`
- Column chunks: `chunk_type="db_column"`, `document_name=f"schema:{source_db}.{name}.{col}"`, `section=["column_description"]`
- Endpoint chunk: `chunk_type="api_endpoint"` (or graphql variant), `document_name=f"api:{source_api}.{path}"`
- Field chunks: `chunk_type="api_field"`, `document_name=f"api:{source_api}.{path}.{field}"`

`enrich_chunks()` runs as normal — `embedding_content` gets the contextual prefix.

No new dependencies — pure Python, no I/O.

**constat migration:** delete chunk-building sections of `schema_manager.py` and
`api_schema_manager.py`. Replace with `loader.load_schema([...])` and
`loader.load_api([...])`. `section` field converts trivially: `list[str]` joins
on `" > "` for any display code that expects a string.

---

### 1.3 `ParquetExtractor` + `load_structured_file()`

**Why:** constat infers schema from CSV, JSON, JSONL, Parquet, Arrow, Feather
and embeds the result as table+column chunks. This is just another file type —
it belongs in Chonk.

**New extractor** (`chonk/extractors/_parquet.py`, requires `chonk[parquet]`):

```toml
parquet = ["pyarrow>=14.0"]
```

Two modes: `mode="schema"` (default) renders column names, types, row count,
and sample values as text. `mode="data"` renders full file as markdown table
(small files only). Handles `.parquet`, `.arrow`, `.feather`.

**New method** on `DocumentLoader`:

```python
loader.load_structured_file(path_or_uri: str, name: str | None = None) -> list[DocumentChunk]
```

Infers `TableMeta` from the file then calls `load_schema([table_meta])`
internally — produces identical N+1 split-tables output regardless of whether
the source was a SQL table, a CSV, or a Parquet file. Downstream is unaware of
the source type.

`loader.load(uri)` auto-dispatches `.parquet`, `.arrow`, `.feather`, `.csv`,
`.jsonl` through `load_structured_file()` — no extra call needed.

| Format | Inference | Dependency |
|---|---|---|
| CSV | pandas `read_csv`, sample 100 rows | pandas (already in `chonk[storage]`) |
| JSON | `json.load`, walk keys on sample | none |
| JSONL | read first 100 lines, walk keys | none |
| Parquet | `pyarrow` schema + row count | `chonk[parquet]` |
| Arrow / Feather | `pyarrow` schema | `chonk[parquet]` |

**constat migration:** delete `_infer_csv_schema()`, `_infer_json_schema()`,
`_infer_jsonl_schema()` from `doc_tools.py` (~260 lines) and
`_infer_parquet_arrow_schema()` from `file/connector.py`. Replace all call
sites with `loader.load_structured_file(path)`.

---

### 1.4 `SchemaMatcher`

**Why:** constat's `EntityExtractor` generates multiple surface-form variants
of each schema/api/business term and registers them as spaCy EntityRuler
patterns. This logic belongs in Chonk as a standalone NER primitive.

**Design — two-pass NER:**

1. **Pass 1 — `SchemaMatcher`** (pure Python, no spaCy): vocabulary-based
   matching with normalization. No `chonk[spacy]` dependency.
2. **Pass 2 — `SpacyMatcher`** (requires `chonk[spacy]`): built-in spaCy NER
   categories (ORG, PERSON, GPE, EVENT, LAW, PRODUCT).
3. **Merge** — `merge_matches(vocab_hits + spacy_hits)` resolves span overlaps.

**New class** (`chonk/ner/_schema.py`):

```python
class SchemaMatcher:
    def __init__(
        self,
        schema_terms: list[str] | None = None,   # table/column names
        api_terms: list[str] | None = None,       # endpoint/operation names
        business_terms: list[str] | None = None,  # glossary terms
    )
    def match(self, text: str) -> list[EntityMatch]
```

Variant generation per term (e.g. `"performance_reviews"`):
1. Normalized — underscores/camelCase → spaces → `"performance reviews"`
2. Singular — strip trailing `s` → `"performance review"`
3. Underscore form — `"performance_reviews"` (if `_` present)
4. Joined forms — `"performancereviews"`, `"performancereview"`

All variants matched case-insensitively. Returns `EntityMatch` with
`entity_type` = `"schema"` | `"api"` | `"term"`.

**Normalization helper** exported from `chonk.ner`:

```python
def normalize_schema_term(term: str, to_singular: bool = False) -> str
```

**Assembly — constat's responsibility:** Chonk ships primitives only. constat
owns the two-pass loop:

```python
from chonk.ner import SchemaMatcher, SpacyMatcher, EntityIndex, merge_matches

schema_matcher = SchemaMatcher(
    schema_terms=table_and_column_names,
    api_terms=endpoint_names,
    business_terms=glossary_terms,
)
spacy_matcher = SpacyMatcher()
index = EntityIndex()

for chunk in chunks:
    vocab_hits = schema_matcher.match(chunk.content)
    spacy_hits = spacy_matcher.match(chunk.content)
    index.index_chunk(chunk.document_name, chunk.content, merge_matches(vocab_hits + spacy_hits))
```

No higher-level wrapper added to Chonk until a second consumer needs it.

**constat migration:** delete `constat/discovery/entity_extractor.py` entirely.
Replace all call sites with the two-pass pattern above. Serialize `EntityIndex`
to pickle between sessions.

---

### 1.5 `DocumentChunk` consolidation

**Why:** constat defines its own `DocumentChunk` dataclass and `ChunkType` enum
that duplicate Chonk's model.

**Change:** constat's `DocumentChunk` import redirects to `chonk.DocumentChunk`.
`ChunkType` enum is deleted; all usages replaced with string literals
(`"db_table"`, `"db_column"`, `"api_endpoint"`, etc.).

`section` type: constat stores as `str`; Chonk stores as `list[str]`. Convert
on read: `section.split(" > ")`. Convert for display: `" > ".join(section)`.
Find-and-replace pass, not a risk.

**constat migration:** update `constat/discovery/models.py` to re-export
`chonk.DocumentChunk`. Delete local dataclass and `ChunkType` enum. Fix all
import sites.

---

## Phase 2 — constat-Specific Wiring

These are constat changes only — no new Chonk features required. Execute after
Phase 1 lands.

1. **Swap `_chunk_document()`** in `doc_tools.py` — replace custom paragraph
   splitter with `chunk_document(name, text, min_chunk_size=600, max_chunk_size=1500)`.
   Replace inline prefix logic with `enrich_chunks(chunks)`.

2. **Replace `vector_store.py`** (~1565 lines) — instantiate `Store` per
   project with namespace scoping. Update all call sites:
   - `add_chunks(chunks, embeddings)` → `store.add_document(chunks, embeddings, namespace=project_id)`
   - `search(query, project_ids=[...])` → `store.search(query, namespaces=[...])`
   - `delete_resource_chunks(name)` → `store.delete_document(name)`

3. **Hash-cache bridge** — new `constat/discovery/doc_cache.py`:
   `(namespace, document_name, content_hash)` table. Before ingesting, compare
   hash; if changed call `store.delete_document(name)` then re-ingest. ~30 lines.

4. **Delete dead code:**
   - `constat/discovery/vector_store.py`
   - `constat/discovery/entity_extractor.py`
   - `VectorStoreBackend`, `NumpyVectorStore`, `create_vector_store()`
   - `VectorStoreConfig` in `config.py`
   - `_infer_*_schema()` functions in `doc_tools.py`

5. **Update `config.py`** — remove `VectorStoreConfig`; add
   `store_dir: str = "~/.constat/stores"` to `StorageConfig`.

---

## Phase 3 — Answer Generation Primitives

**Why:** retrieval is only half the pipeline. `EnhancedSearch` produces ranked
`ScoredChunk` objects with provenance (`seed`, `structural`, `entity_adjacent`,
`cluster_adjacent`). The generator needs to know provenance to construct the
right prompt — a `seed` chunk is high-confidence context, a `cluster_adjacent`
chunk is speculative. Treating them the same wastes the signal retrieval worked
to produce.

**New components** (`chonk/generation/`):

```python
@dataclass
class AnswerContext:
    chunks: list[ScoredChunk]          # ranked, with provenance
    community_context: str | None      # injected community framing (P2)
    query: str
    active_entities: list[str]

class PromptBuilder:
    def build(self, context: AnswerContext, token_budget: int) -> str
    # Orders chunks by provenance tier, respects token budget,
    # prepends community context. No LLM dependency.

class AnswerGenerator:
    def __init__(self, llm_fn: Callable[[str], str])
    def generate(self, context: AnswerContext) -> Answer
    # llm_fn is BYOM — user provides the call. Chonk owns PromptBuilder only.

@dataclass
class Answer:
    text: str
    citations: list[ScoredChunk]       # source chunks supporting the answer
```

Chonk takes no LLM dependency. `AnswerGenerator` is a thin wrapper that calls
the user-supplied `llm_fn` with the prompt `PromptBuilder` assembles.

**constat migration:** wire `EnhancedSearch` → `AnswerContext` → `PromptBuilder`
→ `AnswerGenerator` into the session answer path. Replaces ad-hoc prompt
assembly scattered across `session.py`.

---

## Phase 4 — Deferred GraphRAG Extensions

**Status:** do not implement until Phase 1–3 are stable and retrieval quality
baselines are established. These are additive — they do not replace or break
anything in Phases 1–3.

### 4.1 SVO Triples + `RelationshipIndex`

Typed directed edges between entities extracted by LLM classification over a
**closed verb vocabulary**. Restricted vocabulary is critical — turns
open-ended extraction into classification, dramatically reducing hallucination
rate.

**Verb set:**

| Verb | Example | Source |
|---|---|---|
| `type_of` | `card_number type_of personal_data` | LLM |
| `references` | `orders references customers` | FK (deterministic) |
| `contains` | `payment_transactions contains card_number` | LLM / schema |
| `part_of` | `invoice_line part_of invoices` | FK (deterministic) |
| `governs` | `PCI_DSS governs card_number` | LLM |
| `requires` | `GDPR_Article_17 requires data_deletion` | LLM |
| `defined_by` | `retention_period defined_by data_retention_standard` | LLM |
| `equivalent_to` | `customer_id equivalent_to data_subject` | LLM |
| `created_by` | `PCI_DSS created_by PCI_SSC` | LLM |

`references` and `part_of` are derivable from FK constraints with zero LLM
cost. Only document-to-schema edges require LLM classification.

**Chonk primitives:**

```python
@dataclass
class SVOTriple:
    subject_id: str
    verb: str                          # must be in closed vocabulary
    object_id: str
    confidence: float
    source_chunk_id: str | None = None

class RelationshipIndex:
    def add(self, triple: SVOTriple) -> None: ...
    def get_objects(self, subject_id: str, verb: str | None = None) -> list[SVOTriple]: ...
    def get_subjects(self, object_id: str, verb: str | None = None) -> list[SVOTriple]: ...
```

LLM extraction stays in constat. Chonk ships primitives only.

**Error propagation risk:** wrong SVO triple corrupts its community, which
corrupts every global query touching that community. Mitigations: closed verb
set, confidence threshold, deterministic extraction for schema edges.

### 4.2 LLM-Generated Community Summaries

For each community: collect member chunks → LLM summarizes → store result as
`DocumentChunk(chunk_type="community_summary", namespace=active_namespace)` in
the same `Store`. Enables global search mode (see 4.3).

LLM call stays in constat. Chonk receives the resulting chunk like any other.

### 4.3 Retrieval Modes on `EnhancedSearch`

All three modes use the same `Store`, `RelationshipIndex`, `ClusterMap`, and
community summary chunks. No new storage or indexing.

```python
EnhancedSearch.search(query_embedding, k, query_text, mode="vector_first")
# mode: "vector_first" | "graph_first" | "global"
```

| Mode | Driver | Assists | Best for |
|---|---|---|---|
| `vector_first` | vector similarity | entity / cluster / graph expansion | specific questions, current default |
| `graph_first` | NER on query → `RelationshipIndex` traversal | vector rerank | relationship questions ("what does X govern?") |
| `global` | vector search over `community_summary` chunks only | LLM synthesis | thematic questions no single chunk answers |

`vector_first` is the default and requires no new features — it is the
benchmarked stack. `graph_first` requires Phase 4.1. `global` requires Phase 4.2.

---

## Dependency Order Summary

```
Phase 1.1  namespace on Store
Phase 1.2  load_schema / load_api          → unblocks constat schema_manager migration
Phase 1.3  ParquetExtractor / load_structured_file → unblocks constat file inference migration
Phase 1.4  SchemaMatcher                   → unblocks constat entity_extractor deletion
Phase 1.5  DocumentChunk consolidation     → unblocks all remaining constat migrations
Phase 2    constat wiring                  → requires all Phase 1 items
Phase 3    AnswerGenerator primitives      → requires Phase 2
Phase 4.1  SVOTriple / RelationshipIndex   → requires Phase 3 baseline
Phase 4.2  Community summaries             → requires Phase 4.1
Phase 4.3  graph_first / global modes      → requires Phase 4.1 + 4.2
```
