# Chunk and Entity Management Architecture

## Architecture Layers

```
┌─────────────┐  ┌─────────────┐
│    REPL     │  │     UI      │   Presentation layer
└──────┬──────┘  └──────┬──────┘
       │                │
       └───────┬────────┘
               │
        ┌──────▼──────┐
        │    Core     │            Business logic layer
        └─────────────┘
```

Each layer must be **fully functional independently**:

| Layer | Responsibility |
|-------|----------------|
| **Core** | All common functions: chunk management, entity extraction, term search, session state |
| **REPL** | Terminal presentation, command parsing, text formatting |
| **UI** | Web presentation, component rendering, user interactions |

**Principle**: REPL and UI are thin presentation layers. All business logic lives in Core. Neither REPL nor UI should implement chunk/entity logic directly.

## Overview

### Chunk Events (3)

| Event | Action |
|-------|--------|
| Server start | Build/rebuild chunks for base + all domains (hash-based invalidation) |
| Source add | Incrementally add chunks for that source |
| Source delete | Incrementally delete chunks for that source |

### Entity Events (5)

| Event | Action |
|-------|--------|
| New session | Extract entities for base + active domains |
| Domain add | Incrementally add entities for domain's chunks |
| Domain remove | Incrementally delete entities for that domain |
| Source add | Incrementally add entities for new source's chunks |
| Source delete | Incrementally delete entities for that source |

**Key distinction**: Domains are pre-built collections of sources. Activating/deactivating a domain changes entity visibility but not chunks (they already exist from server start).

## Data Model

### Chunks (embeddings table)

Chunks are text segments with vector embeddings stored in DuckDB.

| Column | Description |
|--------|-------------|
| `chunk_id` | Primary key (hash of content) |
| `document_name` | Reference to source (table name, endpoint path, document name) |
| `chunk_type` | Granular type (see below) |
| `content` | Full original text that was vectorized |
| `embedding` | Vector embedding (1024 dimensions) |
| `project_id` | Scope: `__base__` for base config, domain filename for domains |
| `session_id` | Scope: Session ID for session-added sources (NULL for server-built) |

#### Chunk Scope

| Scope | `project_id` | `session_id` | Created |
|-------|--------------|--------------|---------|
| Base | `__base__` or NULL | NULL | Server start |
| Domain | domain filename | NULL | Server start |
| Session | NULL | session ID | Mid-session (add_document) |

#### Chunk Types and Boundaries

Chunks align with **logical element boundaries**, not arbitrary text splits:

| chunk_type | Boundary | Example |
|------------|----------|---------|
| `db_table` | One chunk per table | Table name + description |
| `db_column` | One chunk per column | Column name + type + description |
| `api_endpoint` | One chunk per REST endpoint | Path + method + description |
| `api_schema` | One chunk per request/response schema | Schema name + fields |
| `graphql_type` | One chunk per GraphQL type | Type name + fields |
| `graphql_field` | One chunk per field | Field name + args + return type |
| `graphql_query` | One chunk per query/mutation | Operation name + args |
| `document` | Text-based chunking | ~500 tokens with overlap |

This preserves semantic boundaries and enables precise retrieval.

### Entities (entities table)

Entities are extracted via **single-pass NER** using spaCy with custom patterns.

| Column | Description |
|--------|-------------|
| `id` | Primary key |
| `name` | Normalized name for NER matching (lowercase, singular) |
| `display_name` | Title case name for display ("Customer Order") |
| `semantic_type` | Linguistic role (see below) |
| `ner_type` | spaCy NER type (NULL if not from spaCy) |
| `session_id` | Session that owns this entity (required) |
| `project_id` | Domain this entity came from |

#### Semantic Types

| Type | Role | Example |
|------|------|---------|
| `CONCEPT` | Noun/thing | customer, order, user, product |
| `ATTRIBUTE` | Modifier/property | active, pending, total, name |
| `ACTION` | Verb/operation | create, get, delete, update |
| `TERM` | Compound phrase | "customer lifetime value", "MRR" |

#### NER Types (from spaCy)

| Type | Description |
|------|-------------|
| `ORG` | Organizations ("Acme Corp", "Microsoft") |
| `PERSON` | People ("John Smith") |
| `PRODUCT` | Products ("iPhone", "Windows") |
| `GPE` | Geo-political entities ("New York", "France") |
| `EVENT` | Events ("World Cup", "Black Friday") |
| NULL | Not from spaCy (schema/API patterns) |

### Chunk-Entity Links (chunk_entities table)

Junction table linking chunks to their extracted entities.

| Column | Description |
|--------|-------------|
| `chunk_id` | FK to embeddings |
| `entity_id` | FK to entities |
| `confidence` | NER confidence score (0.0-1.0) |

#### Entity Extraction

Single-pass NER combines:
- **spaCy NER**: Standard entities (ORG, PERSON, PRODUCT, etc.)
- **Schema patterns**: Cleaned table/column names from active databases
- **API patterns**: Cleaned endpoint/type names from active APIs

Schema and API names are normalized before pattern matching:
- **Singularized**: `customers` → "customer", `orders` → "order"
- **Case splitting**:
  - snake_case: `customer_orders` → "customer order"
  - camelCase: `getUserById` → "get user by id"
  - PascalCase: `CustomerOrder` → "customer order"
- **Casing**: Acronyms and proper nouns retain case (`APIKey` → "API key"), otherwise lowercase
- **Path extraction**: `/api/v1/users/{id}` → "user"

### Term Search

Term searches execute as **similarity search across embeddings**, scoped to visible chunks:
- Base config chunks (`project_id` = `__base__` or NULL)
- Session-added chunks (`session_id` = current session)
- Active domain chunks (`project_id` IN active domains)

Results include:
1. **Matching chunks**: All chunks above similarity threshold (e.g., 0.4), ordered by score
2. **Related chunks**: For each match, other chunks that share entities with it

This entity-based expansion surfaces related context (e.g., finding a table also returns chunks mentioning its columns).

### Source Hashes (source_hashes table)

Centralized hash storage for cache invalidation at the **source level** (base config or domain).

| Column | Description |
|--------|-------------|
| `source_id` | `__base__` or domain filename |
| `db_hash` | Hash of all database configs combined |
| `api_hash` | Hash of all API configs combined |
| `doc_hash` | Hash of all document configs combined |

### Resource Hashes (resource_hashes table)

Per-resource hash storage for **incremental updates** within a source.

| Column | Description |
|--------|-------------|
| `resource_id` | Unique identifier: `{source_id}:{type}:{name}` |
| `resource_type` | Type: `database`, `api`, `document` |
| `resource_name` | Name of the resource (e.g., "chinook", "petstore", "guide.md") |
| `source_id` | Parent source (`__base__` or domain filename) |
| `content_hash` | Hash of the resource's content/config |
| `updated_at` | Last update timestamp |

**Examples of `resource_id`:**
- `__base__:database:chinook` - Database "chinook" from base config
- `mydomain:api:petstore` - API "petstore" from domain "mydomain"
- `__base__:document:guide.md` - Document "guide.md" from base config

#### Incremental Update Flow

When a source (base or domain) is checked at server start:

```
1. Compute source-level hashes (db_hash, api_hash, doc_hash)
2. If source hash unchanged → skip (all resources unchanged)
3. If source hash changed:
   a. For each resource in the source:
      - Compute resource content_hash
      - Compare with stored resource_hashes
      - If resource hash changed:
        * Delete chunks for this specific resource
        * Rebuild chunks for this resource
        * Update resource hash
      - If resource hash unchanged:
        * Skip (keep existing chunks)
4. Update source-level hash
```

This enables **fine-grained cache invalidation**:
- Adding a new database only rebuilds chunks for that database
- Modifying one document only rebuilds chunks for that document
- Unchanged resources retain their existing chunks and embeddings

#### Hash Computation

Resource hashes are computed from the resource's defining content:

| Resource Type | Hash Input |
|---------------|------------|
| **database** | Connection string + schema introspection (tables, columns, types) |
| **api** | OpenAPI/AsyncAPI spec content or URL + spec hash |
| **document** | File content hash or URL + Last-Modified/ETag |

For file-based documents, use file modification time + content hash for fast detection.

## Lifecycle

### Chunk Events

#### Server Start
```
For each source (base + domains):
  1. Compute db_hash, api_hash, doc_hash from config
  2. Compare with stored hashes in source_hashes table
  3. If source hash unchanged → skip entirely
  4. If source hash changed:
     For each resource in the changed type:
       a. Compute resource content_hash
       b. Compare with stored resource_hashes
       c. If resource changed:
          - Delete chunks for this resource (document_name = resource name)
          - Rebuild chunks for this resource
          - Update resource hash
       d. If resource unchanged → skip
  5. Update source-level hash
```

This **two-level hashing** provides:
- Fast skip when nothing changed (source-level check)
- Fine-grained updates when something changed (resource-level check)

#### Source Add
```
When a source is added mid-session (e.g., add_document tool):
  1. Chunk the source content
  2. Generate embeddings
  3. Store chunks with session_id + project_id
```

#### Source Delete
```
When a source is deleted mid-session:
  1. Delete chunks for that source
```

### Entity Events

#### New Session
```
On session creation:
  1. Load user's preferred projects
  2. Extract entities from visible chunks (base + active projects)
  3. Store entities with session_id
```

#### Domain Add
```
When a domain is activated via set_active_domains:
  1. Chunks already exist (from server start)
  2. Extract entities from domain's chunks
  3. Store entities with session_id + domain_id
```

#### Domain Remove
```
When a domain is deactivated via set_active_domains:
  1. Chunks remain (pre-built, may be used by other sessions)
  2. Delete entities for that domain in this session
```

#### Source Add
```
When a source is added mid-session:
  1. (Chunks added - see above)
  2. Extract entities from new chunks
  3. Store entities with session_id
```

#### Source Delete
```
When a source is deleted mid-session:
  1. (Chunks deleted - see above)
  2. Delete entities for that source
```

## Key Methods

### Vector Store (vector_store.py)

| Method | Purpose |
|--------|---------|
| `add_chunks()` | Add chunks with embeddings |
| `clear_project_embeddings(project_id)` | Delete all chunks + entities for a domain |
| `get_source_hash(source_id, hash_type)` | Get stored hash for invalidation check |
| `set_source_hash(source_id, hash_type, hash)` | Update stored hash |
| `extract_entities_for_session(session_id, ...)` | Full NER for session (clears first) |
| `extract_entities_for_project(session_id, project_id, ...)` | Incremental NER for one domain |
| `clear_project_session_entities(session_id, project_id)` | Remove entities for deactivated domain |
| `clear_session_entities(session_id)` | Remove all entities for a session |
| `list_glossary_terms(session_id, ...)` | List glossary terms (glossary_terms table only) |
| `get_unified_glossary(session_id, ...)` | Unified glossary: entities + terms union |
| `get_cluster_siblings(term_name, session_id)` | Other terms in the same KMeans cluster |
| `find_matching_clusters(query, session_id)` | Find clusters matching query text |

### Server Startup (app.py)

| Function | Purpose |
|----------|---------|
| `_compute_db_config_hash()` | Hash database config |
| `_compute_api_config_hash()` | Hash API config |
| `_compute_doc_config_hash()` | Hash document config |
| `_warmup_vector_store()` | Check hashes and rebuild changed chunks |

## Design Decisions

### Why three hash types per source

Different resource types change independently:
- Database schema changes shouldn't rebuild API chunks
- Document updates shouldn't rebuild schema chunks

### Why no migration code

This is v1 - if schema changes, delete `vectors.duckdb` and rebuild. Migration adds complexity without benefit for initial release.

### Why session_id in embeddings

Supports documents added during a session via the `add_document` tool. These are session-scoped and only visible to that session.

### Why entities are session-scoped

Entity extraction uses session-specific patterns (schema terms vary by active databases). Each session may have different active sources, so entities must be scoped to the session.

## Document Ingestion Pipeline

Document loading uses a three-layer architecture: **transport** (fetch raw bytes), **MIME detection** (determine document type), and **content extraction** (convert bytes to text).

### Transport Abstraction (`doc_tools/_transport.py`)

Transport is inferred from field presence on `DocumentConfig`, never configured explicitly:

| Field Set | Transport | Protocol |
|-----------|-----------|----------|
| `content` | inline | N/A — content is already in memory |
| `path` | file | Local filesystem read |
| `url` (http/https) | http | `requests.get()` with configurable headers |
| `url` (s3://) | s3 | `boto3` S3 client (optional dep) |
| `url` (ftp://) | ftp | `ftplib.FTP` |
| `url` (sftp://) | sftp | `paramiko.SSHClient` (optional dep) |

```python
@dataclass
class FetchResult:
    data: bytes                    # Raw document bytes
    detected_mime: str | None      # Content-Type from transport (HTTP, S3)
    source_path: str | None        # Resolved path/URL for extension detection

def fetch_document(config: DocumentConfig, config_dir: str | None) -> FetchResult:
    """Fetch raw bytes via appropriate transport."""
```

Credential fields on `DocumentConfig` support `${ENV_VAR}` substitution via existing YAML processing:
- `username`, `password`, `port` — FTP/SFTP
- `key_path` — SFTP SSH key
- `aws_profile`, `aws_region` — S3

### MIME Type Detection (`doc_tools/_mime.py`)

Document type is determined by a priority chain, not by the `type` field on `DocumentConfig`:

```
1. Explicit type (if user sets type != "auto"): normalize_type(user_type)
2. HTTP Content-Type header (from transport): "application/pdf" → "pdf"
3. File extension: ".docx" → "docx", ".md" → "markdown"
4. Fallback: "text"
```

The `type` field on `DocumentConfig` defaults to `"auto"`. Legacy transport values (`file`, `http`, `inline`) are accepted for backward compatibility and treated as `"auto"`.

Supported short types: `pdf`, `html`, `markdown`, `text`, `docx`, `xlsx`, `pptx`, `csv`, `json`, `yaml`.

### Document Crawler (`doc_tools/_crawler.py`)

When `follow_links: true` is set on a document config with a URL, the crawler performs BFS link extraction:

```
Root URL → fetch → extract links → filter → fetch children → ...
```

**Link extraction** by document type:
- **HTML**: regex for `href="..."` attributes
- **Markdown**: regex for `[text](url)` links
- **Office docs**: not crawlable (text extractors strip hyperlinks)

**Constraints** (from `DocumentConfig`):

| Field | Default | Purpose |
|-------|---------|---------|
| `max_depth` | 2 | Maximum crawl depth from root |
| `max_documents` | 10 | Maximum total documents to fetch |
| `link_pattern` | None | Regex filter on URLs (only follow matching links) |
| `same_domain_only` | True | Restrict to same netloc as root URL |

**Deduplication**: URLs are normalized (sorted query params, stripped fragments) before comparison.

### DocumentConfig

```python
class DocumentConfig(BaseModel):
    # Content type: auto | pdf | html | markdown | text | docx | xlsx | pptx
    # Also accepts full MIME: application/pdf, text/html, etc.
    type: str = "auto"

    # Source (transport inferred from field presence)
    path: Optional[str] = None          # Local file → file transport
    url: Optional[str] = None           # URL → http/s3/ftp/sftp transport
    content: Optional[str] = None       # Inline → inline transport
    headers: dict[str, str] = {}        # HTTP headers

    # Credentials
    username: Optional[str] = None      # FTP/SFTP
    password: Optional[str] = None      # FTP/SFTP
    port: Optional[int] = None          # FTP/SFTP
    key_path: Optional[str] = None      # SFTP SSH key
    aws_profile: Optional[str] = None   # S3
    aws_region: Optional[str] = None    # S3

    # Crawler
    follow_links: bool = False
    max_depth: int = 2
    max_documents: int = 10
    link_pattern: Optional[str] = None
    same_domain_only: bool = True
```

### Ingestion Flow

```
DocumentConfig
    │
    ├── infer_transport() → "file" | "http" | "s3" | ...
    ├── fetch_document()  → FetchResult(data, detected_mime, source_path)
    │
    ├── normalize_type(config.type) → "auto" | "pdf" | "html" | ...
    ├── detect_type_from_source(source_path, detected_mime) → resolved type
    │
    ├── if follow_links: crawl_document() → [(url, FetchResult), ...]
    │
    ├── _extract_content(result, doc_type) → text
    │   ├── pdf  → _extract_pdf_text_from_bytes
    │   ├── docx → _extract_docx_text_from_bytes
    │   ├── xlsx → _extract_xlsx_text_from_bytes
    │   ├── pptx → _extract_pptx_text_from_bytes
    │   └── *    → bytes.decode("utf-8")
    │
    ├── if html: _convert_html_to_markdown → markdown
    │
    └── chunk + embed → store in vector_store
```

## Configuration

Server-level configuration in `config.yaml`:

```yaml
vector_store:
  similarity_threshold: 0.4    # Minimum score for search results
  chunk_size: 500              # Tokens per document chunk
  chunk_overlap: 50            # Overlap between document chunks
  embedding_model: "BAAI/bge-large-en-v1.5"
  embedding_dim: 1024
```
