# Data Source Changes

## Unified Data Source Contract

### Why Not Just Use Claude Connectors / MCP?

[Claude connectors](https://claude.com/connectors) are built on the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) — an open standard where **MCP servers** expose tools and resources to **MCP clients** (like Claude) via JSON-RPC over HTTP.

MCP has three primitives:
- **Tools** — callable functions (input schema → result). Maps to constat's database queries and API operations.
- **Resources** — URI-identified content (`resources/list` → `resources/read`). Maps to constat's document sources.
- **Prompts** — reusable prompt templates. No constat equivalent.

**We cannot just use MCP connectors because:**

| Requirement | MCP | Constat |
|---|---|---|
| SQL query federation | Not supported — tools are opaque function calls | DuckDB ATTACH, Arrow zero-copy, cross-source JOINs |
| Schema discovery | Not a primitive — server may expose a "list_tables" tool but no standard | Built-in: introspect tables, columns, types, relationships |
| Content vectorization | Not a concern — resources return raw content | Full pipeline: fetch → extract → chunk → embed → store |
| Incremental sync | `resources/subscribe` for change notification only | Change detection, delta ingestion, TTL-based refresh |
| Local sources | HTTP only (remote servers) | Local files, direct DB connections, IMAP |
| Persistent connections | Stateless per request | Connection pooling, session-scoped connections |
| DuckDB table registration | N/A | Arrow tables registered in session DuckDB |

**What we borrow from MCP:**

1. **URI-identified resources** — MCP resources have `uri`, `name`, `description`, `mimeType`. We adopt the same shape for our unified `DataSourceInfo`.
2. **Capability declaration** — MCP servers declare `{ resources: { subscribe, listChanged } }`. We adopt capability flags per provider: `queryable`, `ingestible`, `refreshable`.
3. **Registry pattern** — MCP has `mcp_servers[]` + `tools[]` (toolset per server). We adopt a `DataSourceRegistry` that maps `kind` → `DataSourceProvider`.
4. **Auth separation** — MCP puts `authorization_token` on the server definition, not on each tool. We adopt `AuthConfig` separate from source-specific config.

**MCP as an optional provider type:**

A future `type: mcp` source would let constat consume any MCP server's resources as documents:

```yaml
documents:
  my-mcp-source:
    type: mcp
    url: https://mcp-server.example.com/sse
    auth:
      method: bearer
      token_ref: mcp-server-token
    description: "Content from MCP server"
```

Constat would call `resources/list` → `resources/read` for each resource, then feed content through the existing vectorization pipeline. MCP tools could also be mapped to API sources. But this is a future extension — the contract below is designed to accommodate it.

### The Contract

#### Source Kinds

Every data source belongs to one of three kinds, based on what it provides to the session:

```python
class DataSourceKind(str, Enum):
    DATABASE = "database"    # queryable tables — SQL, NoSQL, CSV, SharePoint lists
    DOCUMENT = "document"    # ingestible content — files, web, email, calendar, drive
    API = "api"              # callable operations — GraphQL, REST/OpenAPI
```

A single external system (e.g. SharePoint) may register as multiple kinds — lists as `DATABASE`, documents as `DOCUMENT`.

#### DataSourceProvider Protocol

```python
class DataSourceProvider(Protocol):
    """
    Standard contract for all data source types.
    Each provider handles one (kind, type) pair.
    """

    kind: DataSourceKind

    # --- Lifecycle ---

    def connect(self, name: str, config: dict, auth: AuthConfig | None) -> ConnectionResult:
        """Establish connection. Called once at registration time.
        Returns success/failure + discovered capabilities."""
        ...

    def disconnect(self, name: str) -> None:
        """Release connection resources."""
        ...

    def status(self, name: str) -> SourceStatus:
        """Health check. Returns connected/error/stale."""
        ...

    # --- Discovery ---

    def discover(self, name: str) -> DiscoveryResult:
        """Introspect the source. Returns schema, item list, or capabilities.
        - DATABASE: tables, columns, types
        - DOCUMENT: document list with types/sizes
        - API: operations, endpoints, schemas
        """
        ...

    # --- Data access ---

    def list_items(self, name: str) -> list[SourceItem]:
        """Enumerate available items.
        - DATABASE: table names
        - DOCUMENT: document names (expanded from globs/collections)
        - API: operation names
        """
        ...

    def fetch_item(self, name: str, item_id: str) -> FetchResult:
        """Retrieve a specific item's content.
        - DATABASE: not used (queries go through DuckDB)
        - DOCUMENT: document content (text or bytes)
        - API: not used (operations called via query engine)
        """
        ...

    # --- Sync ---

    def refresh(self, name: str) -> RefreshResult:
        """Re-sync the source. Returns count of new/updated/removed items."""
        ...

    def supports_incremental(self) -> bool:
        """Whether the provider can do delta sync vs full re-ingest."""
        ...
```

#### Supporting Types

```python
@dataclass
class ConnectionResult:
    success: bool
    error: str | None = None
    capabilities: set[str] = field(default_factory=set)
    # Capability flags: "queryable", "ingestible", "refreshable",
    # "subscribable", "browsable", "schema_introspection"

@dataclass
class SourceStatus:
    state: Literal["connected", "disconnected", "error", "stale"]
    error: str | None = None
    item_count: int | None = None
    last_refreshed: datetime | None = None

@dataclass
class SourceItem:
    id: str                           # unique within source
    name: str                         # display name
    description: str = ""
    mime_type: str | None = None
    size: int | None = None           # bytes, if known
    last_modified: datetime | None = None
    viewable: bool = True             # can be opened/displayed individually
    children: list[str] | None = None # for collections (folders, mailboxes)

@dataclass
class DiscoveryResult:
    items: list[SourceItem]
    schema: dict | None = None        # DATABASE: {table: {col: type}}
    operations: list[dict] | None = None  # API: [{name, input_schema, ...}]

@dataclass
class FetchResult:
    content: str | bytes
    mime_type: str = "text/plain"
    metadata: dict = field(default_factory=dict)

@dataclass
class RefreshResult:
    added: int = 0
    updated: int = 0
    removed: int = 0
    errors: list[str] = field(default_factory=list)
```

#### AuthConfig — Unified Authentication

Replaces the scattered `oauth2_*`, `password`, `api_key`, `username` fields across config classes:

```python
class AuthConfig(BaseModel):
    method: Literal["none", "basic", "bearer", "api_key", "oauth2", "ntlm"] = "none"

    # basic / ntlm
    username: str | None = None
    password: str | None = None
    domain: str | None = None         # NTLM domain

    # bearer
    token: str | None = None

    # api_key
    api_key: str | None = None
    api_key_header: str = "X-API-Key"

    # oauth2
    oauth2_provider: str | None = None  # "google", "microsoft", "custom"
    oauth2_scopes: list[str] = []
    oauth2_tenant_id: str | None = None
    token_ref: str | None = None        # → tokens.yaml (see OAuth Token Retention below)
```

#### DataSourceInfo — Unified Response Model

Replaces `SessionDatabaseInfo`, `SessionApiInfo`, `SessionDocumentInfo`:

```python
class DataSourceInfo(BaseModel):
    name: str
    kind: DataSourceKind                # "database", "document", "api"
    type: str                           # "sql", "mongodb", "imap", "graphql", "file", etc.
    description: str | None = None
    state: str = "connected"            # "connected", "disconnected", "error", "stale"
    error: str | None = None

    # Capability flags
    queryable: bool = False             # has tables/schema for SQL
    ingestible: bool = False            # has content for vectorization
    refreshable: bool = False           # supports re-sync
    viewable: bool = False              # top-level source can be opened directly

    # Counts
    item_count: int | None = None       # tables, documents, operations
    indexed_count: int | None = None    # how many items are vectorized

    # Provenance
    source: str = "config"              # "config", "session", "user"
    tier: str | None = None             # "system", "system_domain", "user", "user_domain", "session"
    is_dynamic: bool = False            # added via UI (editable)
    scope: str | None = None            # "session" | "user" (for dynamic sources)

    # Type-specific metadata (flat, not nested)
    uri: str | None = None              # database connection, file path
    base_url: str | None = None         # API base URL
    dialect: str | None = None          # SQL dialect
    path: str | None = None             # document file path
```

#### DataSourceRegistry

```python
class DataSourceRegistry:
    """Central dispatcher — maps (kind, type) to provider implementations."""

    def __init__(self):
        self._providers: dict[tuple[DataSourceKind, str], DataSourceProvider] = {}

    def register(self, kind: DataSourceKind, source_type: str, provider: DataSourceProvider):
        self._providers[(kind, source_type)] = provider

    def get_provider(self, kind: DataSourceKind, source_type: str) -> DataSourceProvider:
        key = (kind, source_type)
        if key not in self._providers:
            raise KeyError(f"No provider for {kind.value}:{source_type}")
        return self._providers[key]

    def add_source(self, name: str, kind: DataSourceKind, source_type: str,
                   config: dict, auth: AuthConfig | None = None) -> ConnectionResult:
        provider = self.get_provider(kind, source_type)
        return provider.connect(name, config, auth)

    def remove_source(self, name: str, kind: DataSourceKind, source_type: str) -> None:
        provider = self.get_provider(kind, source_type)
        provider.disconnect(name)

    def list_all(self) -> list[DataSourceInfo]:
        """Aggregate status from all registered providers."""
        ...

# Default registry setup
def create_registry() -> DataSourceRegistry:
    registry = DataSourceRegistry()
    registry.register(DataSourceKind.DATABASE, "sql", SqlDatabaseProvider())
    registry.register(DataSourceKind.DATABASE, "mongodb", MongoDatabaseProvider())
    registry.register(DataSourceKind.DOCUMENT, "file", FileDocumentProvider())
    registry.register(DataSourceKind.DOCUMENT, "http", HttpDocumentProvider())
    registry.register(DataSourceKind.DOCUMENT, "imap", ImapDocumentProvider())
    registry.register(DataSourceKind.API, "graphql", GraphQLApiProvider())
    registry.register(DataSourceKind.API, "openapi", OpenApiProvider())
    # Future:
    # registry.register(DataSourceKind.DOCUMENT, "calendar", CalendarProvider())
    # registry.register(DataSourceKind.DOCUMENT, "drive", DriveProvider())
    # registry.register(DataSourceKind.DOCUMENT, "sharepoint_docs", SharePointDocProvider())
    # registry.register(DataSourceKind.DATABASE, "sharepoint_lists", SharePointListProvider())
    # registry.register(DataSourceKind.DOCUMENT, "mcp", McpResourceProvider())
    # registry.register(DataSourceKind.API, "mcp", McpToolProvider())
    return registry
```

### Provider Mapping

How current implementations become providers:

| Provider | Kind | Type | Current Code → Provider Method |
|---|---|---|---|
| `SqlDatabaseProvider` | DATABASE | sql | `SchemaManager.add_connection()` → `connect()`, `SchemaManager.get_tables()` → `discover()` |
| `MongoDatabaseProvider` | DATABASE | mongodb | `_connect_mongodb()` → `connect()` |
| `FileDocumentProvider` | DOCUMENT | file | `infer_transport()` + `_load_document()` → `fetch_item()`, glob expansion → `list_items()` |
| `HttpDocumentProvider` | DOCUMENT | http | `HTTPFetcher` → `fetch_item()`, link following → `list_items()` |
| `ImapDocumentProvider` | DOCUMENT | imap | `IMAPFetcher` → `fetch_item()`, message listing → `list_items()` |
| `GraphQLApiProvider` | API | graphql | `_introspect_graphql()` → `discover()` |
| `OpenApiProvider` | API | openapi | OpenAPI spec parse → `discover()` |
| `CalendarProvider` | DOCUMENT | calendar | `CalendarFetcher` → `fetch_item()` (see `calendars.md`) |
| `DriveProvider` | DOCUMENT | drive | `DriveFetcher` → `fetch_item()` (see `cloud-drive.md`) |
| `SharePointDocProvider` | DOCUMENT | sharepoint | `SharePointClient` file ops → `fetch_item()` (see `sp.md`) |
| `SharePointListProvider` | DATABASE | sharepoint_list | `_list_to_arrow()` → `connect()` registers DuckDB table (see `sp.md`) |
| `McpResourceProvider` | DOCUMENT | mcp | `resources/list` → `list_items()`, `resources/read` → `fetch_item()` |

### Unified API Routes

Replace type-specific endpoints with a unified CRUD API:

```python
# constat/server/routes/sources.py

@router.post("/{session_id}/sources")
async def add_source(session_id: str, body: AddSourceRequest) -> DataSourceInfo:
    """Add any source type. Registry dispatches to the correct provider."""

@router.get("/{session_id}/sources")
async def list_sources(session_id: str) -> list[DataSourceInfo]:
    """List all sources (replaces SessionDataSourcesResponse with 3 lists)."""

@router.get("/{session_id}/sources/{name}")
async def get_source(session_id: str, name: str) -> DataSourceInfo:
    """Get source details and status."""

@router.put("/{session_id}/sources/{name}")
async def update_source(session_id: str, name: str, body: UpdateSourceRequest) -> DataSourceInfo:
    """Update a dynamic source's config."""

@router.delete("/{session_id}/sources/{name}")
async def remove_source(session_id: str, name: str) -> dict:
    """Remove a dynamic source from the session."""

@router.post("/{session_id}/sources/{name}/refresh")
async def refresh_source(session_id: str, name: str) -> RefreshResult:
    """Re-sync a source."""

class AddSourceRequest(BaseModel):
    name: str
    kind: DataSourceKind
    type: str
    config: dict                              # type-specific fields
    auth: AuthConfig | None = None
    scope: Literal["session", "user"] = "session"
    description: str = ""

class UpdateSourceRequest(BaseModel):
    config: dict | None = None                # partial update of type-specific fields
    auth: AuthConfig | None = None
    description: str | None = None
    scope: Literal["session", "user"] | None = None  # promote/demote
```

Existing `/databases`, `/files/uri`, `/files/email`, `/apis` routes remain as backward-compatible aliases that construct an `AddSourceRequest` and delegate to `add_source()`.

### Config YAML Mapping

The contract doesn't change the YAML structure — `databases:`, `documents:`, `apis:` sections remain as-is for backward compatibility. The registry determines `kind` from the YAML section:

```python
def _load_sources_from_config(config: dict, registry: DataSourceRegistry) -> list[DataSourceInfo]:
    """Load sources from resolved config, dispatching to providers."""
    sources = []
    for name, db_config in config.get("databases", {}).items():
        result = registry.add_source(name, DataSourceKind.DATABASE, db_config.get("type", "sql"),
                                     config=db_config, auth=_extract_auth(db_config))
        sources.append(...)
    for name, doc_config in config.get("documents", {}).items():
        source_type = _infer_document_type(doc_config)  # file, http, imap, calendar, drive, etc.
        result = registry.add_source(name, DataSourceKind.DOCUMENT, source_type,
                                     config=doc_config, auth=_extract_auth(doc_config))
        sources.append(...)
    for name, api_config in config.get("apis", {}).items():
        result = registry.add_source(name, DataSourceKind.API, api_config.get("type", "graphql"),
                                     config=api_config, auth=_extract_auth(api_config))
        sources.append(...)
    return sources
```

### Migration Path

1. **Phase 1 — Define contract**: Add `DataSourceProvider` protocol, `DataSourceRegistry`, `DataSourceInfo` to `constat/core/`. No behavior change — existing code continues working.
2. **Phase 2 — Wrap existing code**: Create provider wrappers around `SchemaManager`, `doc_tools`, `_resources.py`. Registry delegates to wrappers. Old routes still work.
3. **Phase 3 — Unified routes**: Add `/sources` routes. Old routes become aliases. Frontend migrates to unified API.
4. **Phase 4 — New providers**: Calendar, drive, SharePoint, MCP providers implement the contract directly. No new top-level branches needed.

### File Changes

| File | Change |
|---|---|
| `constat/core/sources.py` | **New**: `DataSourceKind`, `DataSourceProvider`, `DataSourceRegistry`, `AuthConfig`, `DataSourceInfo`, supporting dataclasses |
| `constat/providers/sql_provider.py` | **New**: `SqlDatabaseProvider` wrapping `SchemaManager` |
| `constat/providers/document_providers.py` | **New**: `FileDocumentProvider`, `HttpDocumentProvider`, `ImapDocumentProvider` wrapping `doc_tools` |
| `constat/providers/api_providers.py` | **New**: `GraphQLApiProvider`, `OpenApiProvider` wrapping existing introspection |
| `constat/server/routes/sources.py` | **New**: unified CRUD routes |
| `constat/server/models.py` | Add `DataSourceInfo`, keep existing Info models as aliases |
| `constat/session/_resources.py` | Refactor to use registry internally |
| `tests/test_source_registry.py` | **New**: provider registration, lifecycle, dispatch |

---

# Data Source Persistence: Session vs User Scope

## Goal

When adding a data source from the UI (database, document, API, email), the user chooses whether to persist it at **session** scope (current session only, discarded on close) or **user** scope (retained across all future sessions via Tier 3 user config). OAuth tokens for email sources are always retained at user scope so re-authorization is not required.

## Current State

### What works

- UI-added databases and documents are persisted to `.constat/{user_id}/config.yaml` via `_persist_dbs_to_user_config()` and `_persist_docs_to_user_config()` in `session_manager.py:202-269`.
- `TieredConfigLoader` loads Tier 3 user config from `.constat/{user_id}/config.yaml` and merges into session at lines 338-344.
- `_apply_resolved_source_overrides()` in `sessions.py:321-363` registers Tier 3 databases/APIs into the session.
- All persisted entries are tagged with `"source": "session"`.

### Problems

1. **No scope choice**: Every UI-added source is always persisted to user config. No way to add a session-only source that doesn't carry forward.
2. **Misleading source tag**: Everything is tagged `"source": "session"` regardless of actual intent.
3. **OAuth tokens not retained properly**: Email sources store `oauth2_client_secret` (the refresh token) in `document_config` which IS persisted to user config YAML — but as plaintext, and the `oauth2_client_id` is set to the server's client ID at add-time. On reload in a new session, the server's client ID must be re-injected. Currently this works if the server config hasn't changed, but there's no explicit token management.
4. **No UI to manage persisted sources**: User can't see which sources are persisted, toggle them, or remove them from user config without editing YAML manually.
5. **Stale cleanup is destructive**: `_persist_dbs_to_user_config()` removes user-config entries not in current `_dynamic_dbs`, so opening a session without a previously-persisted DB deletes it from user config.

## Design

### Persistence Scope

Every "Add" modal gains a scope selector:

```
[Session] [User]
```

- **Session** (default): source is written to `.constat/{user_id}/sessions/{session_id}/config.yaml` — the session's own config file. Loaded when the session is restored/reconnected (Tier 5 session overrides). NOT written to user config. Discarded when the session is deleted.
- **User**: source is written to `.constat/{user_id}/config.yaml` with `"source": "user"` tag. Loaded into all future sessions via Tier 3.

### Session Config File

Each session gets its own `config.yaml` alongside the existing `session.json`, `state.json`, and `session.duckdb`:

```
.constat/{user_id}/sessions/{session_id}/
├── session.json          # metadata, timestamps
├── state.json            # resumption state (resources, etc.)
├── session.duckdb        # session data tables
├── config.yaml           # NEW: session-scoped sources
├── queries.jsonl
└── ...
```

This file follows the same structure as domain or user config — `databases:`, `documents:`, `apis:` sections — and is loaded as Tier 5 (session overrides) by `TieredConfigLoader` on session restore.

```yaml
# .constat/{user_id}/sessions/{session_id}/config.yaml
databases:
  sales_csv:
    type: sql
    uri: file:///uploads/sales.csv
    description: "Uploaded CSV"
    source: session

documents:
  wiki_ref:
    url: https://wiki.example.com/page
    description: "Reference doc"
    source: session

apis:
  weather:
    type: openapi
    base_url: https://api.weather.gov
    source: session
```

### Data Flow

```
UI Add Modal
  → scope = "session" | "user"
  → POST /sessions/{id}/databases   { ..., scope: "session" }
  → POST /sessions/{id}/databases   { ..., scope: "user" }

Session scope:
  → _dynamic_dbs.append({ ..., scope: "session" })
  → save_resources()  → session config.yaml (for restore)

User scope:
  → _dynamic_dbs.append({ ..., scope: "user" })
  → save_resources()  → user config.yaml AND session config.yaml
  → user config entry tagged: source: "user"
```

### Source Tag Values

| Tag | Meaning | Stored in | Loaded on |
|-----|---------|-----------|-----------|
| `config` | From system config or domain YAML | System/domain YAML | Every session (Tier 1/2) |
| `session` | UI-added, session scope | Session `config.yaml` | Session restore only (Tier 5) |
| `user` | UI-added, user scope | User `config.yaml` | Every session (Tier 3) |

### Session Restore Flow

When a session is restored (reconnect, page refresh, explicit resume):

```
1. TieredConfigLoader.resolve()
   ├─ Tier 1: System config
   ├─ Tier 2: System domains
   ├─ Tier 3: User config (.constat/{user_id}/config.yaml)     ← user-scoped sources
   ├─ Tier 4: User domains
   └─ Tier 5: Session overrides (.constat/{user_id}/sessions/{id}/config.yaml)  ← session-scoped sources

2. _apply_resolved_source_overrides() registers Tier 3 + 5 sources
3. Session-scoped sources are back — fully restored
```

Currently `session_overrides` (Tier 5) is passed as a dict parameter to `TieredConfigLoader`. Change to also load from the session's `config.yaml` file:

```python
class TieredConfigLoader:
    def __init__(self, ..., session_dir: Optional[Path] = None):
        self._session_dir = session_dir  # NEW

    def resolve(self) -> ResolvedConfig:
        ...
        # --- Tier 5: Session overrides ---
        tier5 = dict(self._session_overrides)
        if self._session_dir:
            session_config = _load_yaml_file(self._session_dir / "config.yaml")
            tier5 = _deep_merge(tier5, _extract_mergeable_sections(session_config))
        if tier5:
            merged = _deep_merge(merged, tier5, "", attribution, ConfigSource.SESSION)
```

## Backend Changes

### Request models — add `scope` field

```python
class DatabaseAddRequest(BaseModel):
    name: str
    uri: str | None = None
    file_id: str | None = None
    type: str | None = None
    description: str | None = None
    scope: Literal["session", "user"] = "session"  # NEW

class AddDocumentURIRequest(BaseModel):
    name: str
    url: str
    description: str = ""
    ...
    scope: Literal["session", "user"] = "session"  # NEW

class AddEmailSourceRequest(BaseModel):
    name: str
    url: str
    username: str
    ...
    scope: Literal["session", "user"] = "session"  # NEW

class AddApiRequest(BaseModel):
    name: str
    ...
    scope: Literal["session", "user"] = "session"  # NEW
```

### `save_resources()` — scope-aware persistence

Replace the unconditional `_persist_dbs_to_user_config()` / `_persist_docs_to_user_config()` with dual-target persistence:

```python
def save_resources(self) -> None:
    # Save dynamic resource lists to state.json (for internal tracking)
    state["resources"] = {
        "dynamic_dbs": self._dynamic_dbs,
        "dynamic_apis": self._dynamic_apis,
        "file_refs": self._file_refs,
        "active_domains": self.active_domains or [],
    }
    history.save_state(history_id, state)

    # Write session-scoped sources to session config.yaml (Tier 5 — for session restore)
    self._persist_session_config()

    # Write user-scoped sources to user config.yaml (Tier 3 — for all sessions)
    self._persist_user_scoped_dbs()
    self._persist_user_scoped_docs()
    self._persist_user_scoped_apis()
```

### Session config.yaml (Tier 5)

All session-scoped dynamic sources are written to the session's own `config.yaml`. This replaces the current approach of encoding sources only in `state.json` resources — the session config file is a proper YAML config that `TieredConfigLoader` can load directly.

```python
def _persist_session_config(self) -> None:
    """Write session-scoped sources to session config.yaml."""
    session_dir = self._get_session_dir()
    if not session_dir:
        return

    config_path = session_dir / "config.yaml"
    config: dict = {}

    # Session-scoped databases
    session_dbs = {db["name"]: {
        "type": db.get("type", "sql"),
        "uri": db.get("uri", ""),
        "description": db.get("description", ""),
        "source": "session",
    } for db in self._dynamic_dbs if db.get("scope") != "user"}
    if session_dbs:
        config["databases"] = session_dbs

    # Session-scoped documents (from file_refs)
    session_docs = {}
    for ref in self._file_refs:
        if ref.get("scope") == "user":
            continue
        doc_entry = ref.get("document_config", {})
        if not doc_entry:
            doc_entry = {"url": ref.get("uri", "")}
        doc_entry["source"] = "session"
        if ref.get("description"):
            doc_entry["description"] = ref["description"]
        session_docs[ref["name"]] = doc_entry
    if session_docs:
        config["documents"] = session_docs

    # Session-scoped APIs
    session_apis = {api["name"]: {
        "type": api.get("type", "openapi"),
        "url": api.get("base_url", ""),
        "description": api.get("description", ""),
        "source": "session",
    } for api in self._dynamic_apis if api.get("scope") != "user"}
    if session_apis:
        config["apis"] = session_apis

    if config:
        config_path.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False))
    elif config_path.exists():
        config_path.unlink()  # Clean up empty config
```

### User config.yaml (Tier 3) — fix stale cleanup

The current `_persist_dbs_to_user_config()` removes entries tagged `"source": "session"` that aren't in `_dynamic_dbs`. This is wrong — it deletes user sources added from other sessions.

Fix: **never delete from user config during save**. Only add/update. Deletion happens explicitly via the management API.

```python
def _persist_user_scoped_dbs(self) -> None:
    """Write user-scoped databases to .constat/{user_id}/config.yaml."""
    config_path = Path(".constat") / self.user_id / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    existing = yaml.safe_load(config_path.read_text()) if config_path.exists() else {}
    databases = existing.get("databases", {})

    # Upsert user-scoped entries only — never delete
    for db in self._dynamic_dbs:
        if db.get("scope") == "user":
            databases[db["name"]] = {
                "type": db.get("type", "sql"),
                "uri": db.get("uri", ""),
                "description": db.get("description", ""),
                "source": "user",
            }

    existing["databases"] = databases
    config_path.write_text(yaml.dump(existing, default_flow_style=False, sort_keys=False))
```

### User source management API

```python
# New routes in constat/server/routes/user_sources.py

@router.get("/{user_id}/sources")
async def list_user_sources(user_id: str) -> dict:
    """List all user-scoped sources from .constat/{user_id}/config.yaml."""
    config = _load_user_config(user_id)
    return {
        "databases": {k: v for k, v in config.get("databases", {}).items() if v.get("source") == "user"},
        "documents": {k: v for k, v in config.get("documents", {}).items() if v.get("source") == "user"},
        "apis": {k: v for k, v in config.get("apis", {}).items() if v.get("source") == "user"},
    }

@router.delete("/{user_id}/sources/{source_type}/{name}")
async def remove_user_source(user_id: str, source_type: str, name: str) -> dict:
    """Remove a user-scoped source from config.yaml."""

@router.put("/{user_id}/sources/{source_type}/{name}")
async def update_user_source(user_id: str, source_type: str, name: str, body: dict) -> dict:
    """Update a user-scoped source (rename, change options, toggle active)."""
```

## Email OAuth Token Retention

### Current flow

1. User clicks "Connect Gmail/Outlook" in email modal
2. Browser OAuth popup → `/api/oauth/email/callback` → returns `refresh_token`
3. Frontend sends `refresh_token` in `addEmailSource()` request body
4. Backend stores it as `oauth2_client_secret` in `DocumentConfig`
5. `DocumentConfig` is serialized to `document_config` dict in `_file_refs`
6. `_persist_docs_to_user_config()` writes it to `.constat/{user_id}/config.yaml` — **including the refresh token in plaintext**

### What's broken

- Refresh token is stored as `oauth2_client_secret` in plaintext YAML
- `oauth2_client_id` is baked from server config at add-time — if server config changes, token stops working
- No way to re-authenticate without re-adding the source
- No way to distinguish "this is a user's refresh token" from "this is an app client secret"

### Fix

#### 1. Separate token storage

Store OAuth tokens in a dedicated file, not in the document config:

```
.constat/{user_id}/tokens.yaml
```

```yaml
tokens:
  my-gmail:
    provider: google
    email: ken@gmail.com
    refresh_token: "<token>"
    scopes: "https://mail.google.com/ email"
    created_at: "2026-03-24T10:00:00Z"
    last_used: "2026-03-24T14:30:00Z"

  my-outlook:
    provider: microsoft
    email: ken@company.com
    refresh_token: "<token>"
    tenant_id: "abc-123"
    scopes: "https://outlook.office365.com/IMAP.AccessAsUser.All offline_access"
    created_at: "2026-03-24T10:05:00Z"
    last_used: "2026-03-24T14:35:00Z"
```

#### 2. Document config references token by name

In `.constat/{user_id}/config.yaml`:

```yaml
documents:
  my-gmail:
    type: imap
    url: imaps://imap.gmail.com:993
    username: ken@gmail.com
    auth_type: oauth2_refresh
    oauth2_token_ref: my-gmail         # references tokens.yaml entry
    mailbox: INBOX
    since: "2026-01-01"
    source: user
```

No plaintext secrets in the document config.

#### 3. Token injection at session load

When `TieredConfigLoader` loads Tier 3 user config, document entries with `oauth2_token_ref` are resolved at runtime:

```python
def _inject_oauth_tokens(user_id: str, documents: dict) -> dict:
    """Resolve oauth2_token_ref entries by loading tokens.yaml."""
    tokens = _load_user_tokens(user_id)
    for name, doc in documents.items():
        ref = doc.pop("oauth2_token_ref", None)
        if ref and ref in tokens:
            token_entry = tokens[ref]
            server_config = _get_server_config()
            if token_entry["provider"] == "google":
                doc["oauth2_client_id"] = server_config.google_email_client_id
                doc["oauth2_client_secret"] = token_entry["refresh_token"]
                doc["password"] = server_config.google_email_client_secret
            elif token_entry["provider"] == "microsoft":
                doc["oauth2_client_id"] = server_config.microsoft_email_client_id
                doc["oauth2_client_secret"] = token_entry["refresh_token"]
                doc["oauth2_tenant_id"] = token_entry.get("tenant_id") or server_config.microsoft_email_tenant_id
            doc["auth_type"] = "oauth2_refresh"
    return documents
```

This means:
- Server OAuth client credentials are always injected fresh from current server config
- Refresh tokens survive server config changes (only the refresh token matters)
- Token storage is separate from source configuration

#### 4. Re-authentication flow

When a refresh token expires or is revoked:
- Session load catches the authentication error during source ingestion
- Pushes `source_auth_required` WebSocket event with source name
- UI shows "Re-authenticate" button on the source
- User clicks → same OAuth popup flow → token updated in `tokens.yaml`
- Source re-ingested with new token

```python
# WebSocket event
{
    "type": "source_auth_required",
    "source_name": "my-gmail",
    "provider": "google",
    "email": "ken@gmail.com",
    "error": "Token expired"
}
```

#### 5. Token refresh on use

Update `last_used` and store new access tokens when refresh succeeds:

```python
def _get_oauth_token_for_source(user_id: str, token_ref: str) -> str:
    """Get a fresh access token, refreshing if needed."""
    tokens = _load_user_tokens(user_id)
    entry = tokens[token_ref]
    # Use existing OAuth providers to get access token from refresh token
    access_token = _refresh_access_token(entry)
    entry["last_used"] = datetime.now(timezone.utc).isoformat()
    _save_user_tokens(user_id, tokens)
    return access_token
```

## Frontend Changes

### Scope toggle in modals

All "Add" modals (database, document, email, API) get a scope selector:

```tsx
<div className="flex gap-2 mb-4">
  <button
    onClick={() => setScope('session')}
    className={scope === 'session' ? 'bg-primary-100 ...' : 'bg-gray-50 ...'}
  >
    This session
  </button>
  <button
    onClick={() => setScope('user')}
    className={scope === 'user' ? 'bg-primary-100 ...' : 'bg-gray-50 ...'}
  >
    All sessions
  </button>
</div>
```

**Email sources**: default to `user` scope (tokens should be retained). Other sources: default to `session` scope.

### Source badges

In the Sources panel, show scope badge alongside domain badge:

```tsx
{source.scope === 'user' && (
  <span className="text-[10px] px-1.5 py-0.5 rounded bg-purple-100 text-purple-700">
    user
  </span>
)}
{source.scope === 'session' && (
  <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-100 text-blue-700">
    session
  </span>
)}
```

### "Promote" action: session → user

For session-scoped sources, show a "Keep for all sessions" action (pin icon or promote button):

```tsx
{source.scope === 'session' && (
  <button onClick={() => promoteToUser(source.name)} title="Keep for all sessions">
    <BookmarkIcon className="w-3.5 h-3.5" />
  </button>
)}
```

This calls `PUT /users/{user_id}/sources/{type}/{name}` to write the entry to user config.

### "Demote" action: user → session

For user-scoped sources, show "Remove from all sessions" (unpin):

```tsx
{source.scope === 'user' && (
  <button onClick={() => demoteToSession(source.name)} title="Remove from future sessions">
    <BookmarkSlashIcon className="w-3.5 h-3.5" />
  </button>
)}
```

This calls `DELETE /users/{user_id}/sources/{type}/{name}` to remove from user config. The source stays in the current session.

### User sources panel

Optional: a "My Sources" section in Settings or Sources showing all user-scoped sources across types:

```
My Sources
├─ Databases
│   └─ sales_csv (CSV file) [✕ remove]
├─ Documents
│   └─ wiki_page (HTTP) [✕ remove]
├─ Email
│   └─ Gmail (ken@gmail.com) [⟳ re-auth] [✕ remove]
└─ APIs
    └─ weather_api (OpenAPI) [✕ remove]
```

### API client changes

```typescript
// sessions.ts — add scope to request types
export async function addDatabase(
  sessionId: string,
  data: {
    name: string
    uri?: string
    file_id?: string
    type?: string
    description?: string
    scope?: 'session' | 'user'      // NEW
  }
): Promise<SessionDatabase>

export async function addEmailSource(
  sessionId: string,
  body: {
    ...existing fields...
    scope?: 'session' | 'user'      // NEW, defaults to 'user' for email
  }
): Promise<...>

// New: user source management
export async function listUserSources(userId: string): Promise<UserSources>
export async function removeUserSource(userId: string, sourceType: string, name: string): Promise<void>
export async function promoteSource(sessionId: string, sourceType: string, name: string): Promise<void>
```

## Data Source Editing

### Goal

Users with write rights (`canWrite('sources')`) can edit any non-config data source they can see — changing description, connection URI, options, etc. Config-tier sources (from domain YAML) are read-only in the UI.

### Editable Fields by Source Type

| Source Type | Editable Fields |
|-------------|----------------|
| Database | `uri`, `description`, `type` |
| Document (URI) | `url`, `description`, `type`, `headers`, `follow_links`, `max_depth`, `max_documents`, `same_domain_only` |
| Document (email) | `mailbox`, `since`, `max_messages`, `include_headers`, `extract_attachments`, `description` |
| API | `base_url`, `description`, `type`, `auth_type`, `auth_header` |

The source `name` is immutable (used as the config key and addressing prefix). To rename, delete and re-add.

### Backend

```python
# databases.py
@router.put("/{session_id}/databases/{name}")
async def update_database(
    session_id: str, name: str, body: DatabaseUpdateRequest,
    session_manager: SessionManager = Depends(get_session_manager),
) -> SessionDatabaseInfo:
    """Update a dynamic database's configuration."""
    managed = session_manager.get_session_or_none(session_id)
    if not managed:
        raise HTTPException(status_code=404, detail="Session not found")

    # Find in dynamic_dbs
    db = next((d for d in managed._dynamic_dbs if d["name"] == name), None)
    if not db:
        raise HTTPException(status_code=400, detail="Cannot edit config-tier databases")

    # Apply updates
    if body.uri is not None:
        db["uri"] = body.uri
    if body.description is not None:
        db["description"] = body.description
    if body.type is not None:
        db["type"] = body.type

    # Re-register connection if URI changed
    if body.uri is not None:
        session.schema_manager.remove_connection(name)
        session.schema_manager.add_connection(name, body.uri, db.get("type", "sql"))

    managed.save_resources()
    return _db_info(db)

class DatabaseUpdateRequest(BaseModel):
    uri: str | None = None
    description: str | None = None
    type: str | None = None
```

```python
# files.py
@router.put("/{session_id}/documents/{name}")
async def update_document(
    session_id: str, name: str, body: DocumentUpdateRequest,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict:
    """Update a dynamic document source's configuration."""
    managed = session_manager.get_session_or_none(session_id)
    if not managed:
        raise HTTPException(status_code=404, detail="Session not found")

    ref = next((r for r in managed._file_refs if r["name"] == name), None)
    if not ref:
        raise HTTPException(status_code=400, detail="Cannot edit config-tier documents")

    # Apply updates to file_ref and its document_config
    doc_config = ref.get("document_config", {})
    for field in ("description", "url", "mailbox", "since", "max_messages",
                   "include_headers", "extract_attachments", "type"):
        value = getattr(body, field, None)
        if value is not None:
            doc_config[field] = value
            if field == "description":
                ref["description"] = value
            if field == "url":
                ref["uri"] = value
    ref["document_config"] = doc_config

    managed.save_resources()
    return {"status": "updated", "name": name}
```

### Frontend

Add an edit (pencil) icon button on each editable source row. Clicking opens the existing modal pre-populated with current values:

```tsx
{/* Edit button — only for dynamic (non-config) sources */}
{!source.from_config && canWrite('sources') && (
  <button
    onClick={() => openEditModal(source)}
    className="opacity-0 group-hover:opacity-100 p-1 text-gray-400 hover:text-primary-600 transition-all"
    title="Edit source"
  >
    <PencilIcon className="w-3.5 h-3.5" />
  </button>
)}
```

The "Add" modal is reused in edit mode: pre-fill fields, change button text to "Save", call `PUT` instead of `POST`. The `scope` toggle shows the current scope and allows promotion/demotion inline.

### API client

```typescript
export async function updateDatabase(
  sessionId: string, name: string,
  data: { uri?: string; description?: string; type?: string }
): Promise<SessionDatabase>

export async function updateDocument(
  sessionId: string, name: string,
  data: { url?: string; description?: string; mailbox?: string; since?: string; ... }
): Promise<{ status: string; name: string }>

export async function updateApi(
  sessionId: string, name: string,
  data: { base_url?: string; description?: string; type?: string; auth_type?: string }
): Promise<SessionApiSource>
```

## Document Source Hyperlink Fixes

### Problem

Document source names in the Sources panel are rendered as clickable blue links that call `handleViewDocument()`. This breaks for:

1. **Folder/glob document sources**: A config entry like `path: ./docs/*.md` expands to children (`data_files:sales.md`, `data_files:rules.md`). Clicking the parent collection name has no single document to display. Clicking a child works but the parent is misleading.

2. **IMAP email sources**: The source name (e.g., `my-gmail`) represents an IMAP mailbox, not a viewable document. `handleViewDocument()` fails because there's no single content blob — individual messages are the documents (`my-gmail:msg_20260301_abc`). There's no meaningful target for the hyperlink.

3. **Multi-transport sources** (planned: calendar, drive, SharePoint): Same problem — the source name represents a collection, not a document.

### Fix

Change document source names from always-clickable links to conditionally-clickable based on whether the source has viewable content:

```tsx
// Determine if source name should be a clickable link
const isViewable = (doc: DocumentSourceInfo): boolean => {
  // Single-file documents with a path or URL → viewable
  if (doc.path && !doc.path.includes('*') && !doc.path.endsWith('/')) return true
  // Inline content → viewable
  if (doc.type === 'inline') return true
  // HTTP URL to a single page → viewable
  if (doc.url && !doc.url.startsWith('imap')) return true
  // Collections (glob, directory, IMAP, calendar, drive, sharepoint) → NOT viewable
  return false
}
```

Render:

```tsx
{isViewable(doc) ? (
  <button
    onClick={() => handleViewDocument(doc.name)}
    className="text-sm font-medium text-blue-600 dark:text-blue-400 hover:underline cursor-pointer"
  >
    {doc.name}
  </button>
) : (
  <span className="text-sm font-medium text-gray-800 dark:text-gray-200">
    {doc.name}
  </span>
)}
```

### Source-type indicators

Replace the generic blue link with source-type-aware display:

| Source Type | Display | Clickable? |
|-------------|---------|------------|
| Single file (`path: ./doc.pdf`) | Blue link → opens file viewer | Yes |
| HTTP URL (`url: https://...`) | Blue link → opens in viewer/iframe | Yes |
| Inline content | Blue link → opens content modal | Yes |
| Folder/glob (`path: ./docs/*.md`) | Plain text + folder icon + child count badge | No (children are individually clickable) |
| IMAP email (`type: imap`) | Plain text + envelope icon | No |
| Calendar (`type: calendar`) | Plain text + calendar icon | No |
| Drive (`type: drive`) | Plain text + folder icon | No |
| SharePoint (`type: sharepoint`) | Plain text + globe icon | No |

For collection types, show an expand chevron to reveal children (individual messages, files, events) which ARE clickable.

### Backend support

`list_documents()` in `_access.py` already distinguishes single files from expanded globs (lines 69-126). Add a `viewable` field to the response:

```python
# In list_documents() result dicts:
entry["viewable"] = True   # single file, inline, HTTP
entry["viewable"] = False  # collection parent, IMAP source, etc.
```

For IMAP sources, the parent entry should report its type so the frontend can render the correct icon:

```python
if doc_config.url and urlparse(doc_config.url).scheme in ("imap", "imaps"):
    entry["source_type"] = "imap"
    entry["viewable"] = False
```

## Migration

### Existing persisted entries

Current entries in `.constat/{user_id}/config.yaml` have `"source": "session"`. Migrate on first load:

```python
def _migrate_user_config(user_id: str) -> None:
    """One-time migration: rename source='session' to source='user'."""
    config = _load_user_config(user_id)
    changed = False
    for section in ("databases", "documents", "apis"):
        for name, entry in config.get(section, {}).items():
            if entry.get("source") == "session":
                entry["source"] = "user"
                changed = True
    if changed:
        _save_user_config(user_id, config)
```

### Existing email tokens

Current email entries have `oauth2_client_secret` containing the refresh token. Migrate to `tokens.yaml`:

```python
def _migrate_email_tokens(user_id: str) -> None:
    """Extract inline OAuth tokens to tokens.yaml."""
    config = _load_user_config(user_id)
    tokens = _load_user_tokens(user_id)
    for name, doc in config.get("documents", {}).items():
        if doc.get("auth_type") in ("oauth2_refresh", "oauth2") and doc.get("oauth2_client_secret"):
            # Move token to tokens.yaml
            tokens[name] = {
                "provider": "microsoft" if doc.get("oauth2_tenant_id") else "google",
                "email": doc.get("username", ""),
                "refresh_token": doc["oauth2_client_secret"],
                "tenant_id": doc.get("oauth2_tenant_id"),
                "scopes": "",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            # Replace inline token with reference
            doc.pop("oauth2_client_secret", None)
            doc.pop("oauth2_client_id", None)
            doc["oauth2_token_ref"] = name
    _save_user_config(user_id, config)
    _save_user_tokens(user_id, tokens)
```

## File Changes Summary

| File | Change |
|------|--------|
| `constat/server/session_manager.py` | Scope-aware `save_resources()`, `_persist_session_config()`, `_persist_user_scoped_*()` methods, stop deleting non-present entries |
| `constat/core/tiered_config.py` | Accept `session_dir` parameter, load session `config.yaml` as Tier 5, call `_inject_oauth_tokens()` after Tier 3 |
| `constat/server/routes/sessions.py` | Pass `session_dir` to `TieredConfigLoader` on session create/restore |
| `constat/server/routes/databases.py` | Add `scope` to `DatabaseAddRequest`, `AddApiRequest` |
| `constat/server/routes/files.py` | Add `scope` to `AddDocumentURIRequest`, `AddEmailSourceRequest` |
| `constat/server/routes/user_sources.py` | **New**: user source CRUD API (list, remove, update, promote) |
| `constat/server/user_tokens.py` | **New**: `load_user_tokens()`, `save_user_tokens()`, `_inject_oauth_tokens()`, `_migrate_email_tokens()` |
| `constat-ui/src/api/sessions.ts` | Add `scope` to add requests, add user source API calls |
| `constat-ui/src/components/artifacts/ArtifactPanel.tsx` | Add scope toggle to all "Add" modals, promote/demote buttons, scope badges |
| `constat-ui/src/store/accountStore.ts` | **New**: Zustand store for user sources (or extend sessionStore) |
| `tests/test_source_persistence.py` | Scope-aware save/load, session config round-trip, migration, token injection |

## Testing Strategy

1. **Unit tests** (`test_source_persistence.py`):
   - Session-scoped DB: written to session `config.yaml`, NOT to user config
   - User-scoped DB: written to BOTH session `config.yaml` AND user config with `source: "user"`
   - Session restore: session `config.yaml` loaded as Tier 5 → session-scoped sources restored
   - New session: loads user-scoped DB from Tier 3, does NOT load other sessions' sources
   - Session config round-trip: add sources → save → reload from `config.yaml` → sources match
   - Promote: session-scoped → user-scoped, written to user config, stays in session config
   - Demote: user-scoped → removed from user config, stays in session config
   - Save does not delete existing user config entries from other sessions
   - Empty session config: file deleted (not left as empty YAML)

2. **Email token tests**:
   - Add email with OAuth → token stored in `tokens.yaml`, config references by `oauth2_token_ref`
   - New session → token injected from `tokens.yaml` with current server client ID
   - Server client ID changes → token still works (only refresh token matters)
   - Token expired → `source_auth_required` event pushed → re-auth updates `tokens.yaml`
   - Migration: existing inline tokens → extracted to `tokens.yaml`

3. **Migration tests**:
   - Existing `source: "session"` entries → renamed to `source: "user"`
   - Existing inline `oauth2_client_secret` → moved to `tokens.yaml`
   - Already-migrated config → no-op

4. **Frontend tests** (Vitest):
   - Scope toggle defaults: email → user, others → session
   - Scope badge rendering: user/session/config
   - Promote/demote button visibility by scope
   - API calls include correct `scope` parameter

## Edge Cases

| Case | Handling |
|------|----------|
| User adds same-named source in session and user scope | Session config (Tier 5) wins over user config (Tier 3) — last tier takes precedence |
| User config YAML edited manually (no source tag) | Treat as `source: "user"` — any entry in user config is user-scoped |
| Session restored after server restart | Session `config.yaml` loaded as Tier 5 → all session sources restored |
| Session `config.yaml` deleted manually | Session sources lost; only user + domain sources remain on restore |
| Session with OAuth email source restored | Session config has `oauth2_token_ref` → resolved from `tokens.yaml` at load time |
| Server OAuth client credentials rotate | `_inject_oauth_tokens()` always uses current server config — transparent |
| Refresh token revoked by user in Google/Azure portal | Auth fails → `source_auth_required` event → user re-authenticates |
| Multiple sessions open, both add user-scoped sources | Each writes to same user config; last write wins for overlapping names |
| User removes user-scoped source while session is using it | Source stays in current session (in `_dynamic_dbs`), removed from future sessions |
| `tokens.yaml` deleted | All email sources fail auth → `source_auth_required` for each |
| YAML env var substitution in tokens | Not applied — tokens are stored and loaded as-is (not user-editable config) |

## Non-Goals (v1)

- Token encryption at rest (plaintext in `tokens.yaml`, file permissions only)
- Cross-user source sharing
- Source sync status tracking in user config
- Automatic token refresh in background (only refreshed on use)
- Personal resource picker / unified "+" flow (see `personal.md` — separate feature)
- Granular per-session source selection from user sources
