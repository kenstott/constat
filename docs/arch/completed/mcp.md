# MCP Data Source

## Goal

Consume any [MCP server](https://modelcontextprotocol.io/) as a constat data source. MCP servers expose two primitives relevant to constat:

- **Resources** (`resources/list`, `resources/read`) — URI-identified content → ingested as **document sources** (vectorized, searchable)
- **Tools** (`tools/list`, `tools/call`) — callable operations with JSON Schema inputs → registered as **API sources** (available to the query engine)

A single MCP server may expose both resources and tools. Constat registers each as the appropriate source kind.

## MCP Protocol Summary

MCP uses JSON-RPC 2.0 over HTTP (Streamable HTTP or SSE). Key operations:

```
resources/list  → [{ uri, name, description, mimeType, size }]
resources/read  → { contents: [{ uri, mimeType, text | blob }] }
tools/list      → [{ name, description, inputSchema, outputSchema }]
tools/call      → { content: [{ type, text | data }], isError }
```

Servers declare capabilities: `{ resources: { subscribe, listChanged }, tools: { listChanged } }`.

Auth: OAuth bearer token passed in server connection config.

## Config

### Document source (MCP resources)

```yaml
documents:
  confluence-mcp:
    type: mcp
    url: https://mcp-confluence.example.com/sse
    auth:
      method: bearer
      token_ref: confluence-mcp-token    # → tokens.yaml
    description: "Confluence pages via MCP"
    # Optional: filter to specific resources by URI pattern
    resource_filter: "confluence://space/ENG/*"
    # Optional: limit resource count
    max_resources: 100
    # Standard document options
    cache: true
    auto_refresh: true
    refresh_interval: 3600
```

### API source (MCP tools)

```yaml
apis:
  jira-mcp:
    type: mcp
    url: https://mcp-jira.example.com/sse
    auth:
      method: bearer
      token_ref: jira-mcp-token
    description: "Jira operations via MCP"
    # Optional: allowlist specific tools
    allowed_tools:
      - search_issues
      - get_issue
      - create_issue
    # Optional: denylist tools
    denied_tools:
      - delete_issue
```

### Combined (resources + tools from same server)

```yaml
documents:
  github-docs:
    type: mcp
    url: https://mcp-github.example.com/sse
    auth:
      method: bearer
      token_ref: github-mcp-token
    description: "GitHub repo docs"
    resource_filter: "github://repo/*/README.md"

apis:
  github-ops:
    type: mcp
    url: https://mcp-github.example.com/sse
    auth:
      method: bearer
      token_ref: github-mcp-token
    description: "GitHub operations"
    allowed_tools:
      - search_code
      - list_issues
      - get_pull_request
```

Both entries point to the same MCP server URL. The MCP client maintains a single connection, shared by both source registrations.

### Local subprocess servers

For MCP servers that run as local processes (stdio transport):

```yaml
documents:
  local-files:
    type: mcp
    url: stdio://python -m my_mcp_server --dir /data
    description: "Local file server via MCP stdio"
```

The `stdio://` scheme signals that constat should spawn the process and communicate via stdin/stdout JSON-RPC.

## Architecture

### McpClient

Manages the JSON-RPC connection to a single MCP server:

```python
class McpClient:
    """MCP client — handles JSON-RPC transport, capability negotiation, and caching."""

    def __init__(self, url: str, auth: AuthConfig | None = None, timeout: int = 30):
        self._url = url
        self._auth = auth
        self._timeout = timeout
        self._capabilities: dict = {}
        self._session: httpx.AsyncClient | None = None
        self._process: subprocess.Popen | None = None  # for stdio

    async def connect(self) -> dict:
        """Initialize MCP session. Returns server capabilities."""
        if self._url.startswith("stdio://"):
            self._process = self._spawn_process(self._url[8:])
            # JSON-RPC over stdin/stdout
        else:
            self._session = httpx.AsyncClient(
                base_url=self._url,
                headers=self._auth_headers(),
                timeout=self._timeout,
            )
        # Send initialize request
        result = await self._rpc("initialize", {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "constat", "version": "1.0"},
        })
        self._capabilities = result.get("capabilities", {})
        await self._rpc("notifications/initialized", {})
        return self._capabilities

    async def list_resources(self, cursor: str | None = None) -> tuple[list[dict], str | None]:
        """Paginated resource listing."""
        params = {"cursor": cursor} if cursor else {}
        result = await self._rpc("resources/list", params)
        return result["resources"], result.get("nextCursor")

    async def read_resource(self, uri: str) -> list[dict]:
        """Read resource contents. Returns list of {uri, mimeType, text|blob}."""
        result = await self._rpc("resources/read", {"uri": uri})
        return result["contents"]

    async def list_tools(self, cursor: str | None = None) -> tuple[list[dict], str | None]:
        """Paginated tool listing."""
        params = {"cursor": cursor} if cursor else {}
        result = await self._rpc("tools/list", params)
        return result["tools"], result.get("nextCursor")

    async def call_tool(self, name: str, arguments: dict) -> dict:
        """Invoke a tool. Returns {content, isError, structuredContent?}."""
        result = await self._rpc("tools/call", {"name": name, "arguments": arguments})
        return result

    async def subscribe_resource(self, uri: str) -> None:
        """Subscribe to resource change notifications (if supported)."""
        if self._capabilities.get("resources", {}).get("subscribe"):
            await self._rpc("resources/subscribe", {"uri": uri})

    async def disconnect(self) -> None:
        """Clean shutdown."""
        if self._session:
            await self._session.aclose()
        if self._process:
            self._process.terminate()
            self._process.wait(timeout=5)

    async def _rpc(self, method: str, params: dict) -> dict:
        """Send JSON-RPC request, return result."""
        ...
```

### McpClientPool

Shared connection pool — multiple document/API registrations pointing to the same URL reuse one client:

```python
class McpClientPool:
    """Reuse MCP connections across sources pointing to the same server."""

    def __init__(self):
        self._clients: dict[str, McpClient] = {}  # keyed by URL
        self._refcount: dict[str, int] = {}

    async def acquire(self, url: str, auth: AuthConfig | None = None) -> McpClient:
        if url not in self._clients:
            client = McpClient(url, auth)
            await client.connect()
            self._clients[url] = client
            self._refcount[url] = 0
        self._refcount[url] += 1
        return self._clients[url]

    async def release(self, url: str) -> None:
        self._refcount[url] -= 1
        if self._refcount[url] <= 0:
            await self._clients[url].disconnect()
            del self._clients[url]
            del self._refcount[url]
```

### McpDocumentProvider

Implements `DataSourceProvider` for MCP resources → documents:

```python
class McpDocumentProvider:
    """Consumes MCP resources as constat document sources."""

    kind = DataSourceKind.DOCUMENT

    def __init__(self, pool: McpClientPool):
        self._pool = pool
        self._sources: dict[str, McpDocSource] = {}

    async def connect(self, name: str, config: dict, auth: AuthConfig | None) -> ConnectionResult:
        url = config["url"]
        client = await self._pool.acquire(url, auth)

        # Check server has resources capability
        if "resources" not in client._capabilities:
            return ConnectionResult(success=False, error="MCP server has no resources capability")

        # List resources (paginate all)
        resources = await self._list_all_resources(client, config.get("resource_filter"))

        self._sources[name] = McpDocSource(
            client=client,
            url=url,
            resources=resources,
            config=config,
        )

        capabilities = {"ingestible", "refreshable"}
        if client._capabilities.get("resources", {}).get("subscribe"):
            capabilities.add("subscribable")

        return ConnectionResult(success=True, capabilities=capabilities)

    def list_items(self, name: str) -> list[SourceItem]:
        source = self._sources[name]
        return [
            SourceItem(
                id=r["uri"],
                name=r.get("name", r["uri"]),
                description=r.get("description", ""),
                mime_type=r.get("mimeType"),
                size=r.get("size"),
                viewable=True,
            )
            for r in source.resources
        ]

    async def fetch_item(self, name: str, item_id: str) -> FetchResult:
        source = self._sources[name]
        contents = await source.client.read_resource(item_id)
        if not contents:
            raise ValueError(f"Empty resource: {item_id}")

        content = contents[0]
        if "text" in content:
            return FetchResult(
                content=content["text"].encode("utf-8"),
                mime_type=content.get("mimeType", "text/plain"),
                metadata={"uri": content["uri"]},
            )
        elif "blob" in content:
            import base64
            return FetchResult(
                content=base64.b64decode(content["blob"]),
                mime_type=content.get("mimeType", "application/octet-stream"),
                metadata={"uri": content["uri"]},
            )

    async def refresh(self, name: str) -> RefreshResult:
        """Re-sync using ChangeProbe for tiered change detection."""
        source = self._sources[name]
        probe = ChangeProbe()
        result = await probe.probe(source.client, source.stored_meta,
                                   source.config.get("resource_filter"))

        # Fetch and index added resources
        for r in result.added:
            content, meta = await self._fetch_and_hash(source, r["uri"])
            source.stored_meta[r["uri"]] = meta
            # → feed content into vectorization pipeline

        # Re-fetch and re-index changed resources
        for r in result.changed:
            content, meta = await self._fetch_and_hash(source, r["uri"])
            source.stored_meta[r["uri"]] = meta
            # → delete old chunks, re-vectorize

        # Remove chunks for deleted resources
        for uri in result.removed:
            source.stored_meta.pop(uri, None)
            # → delete chunks from vector store

        # Update resource list
        source.resources = result.current_resources
        return RefreshResult(
            added=len(result.added),
            updated=len(result.changed),
            removed=len(result.removed),
        )

    async def _fetch_and_hash(self, source: "McpDocSource", uri: str) -> tuple[str, "ResourceMeta"]:
        """Fetch resource content and compute metadata for future probes."""
        contents = await source.client.read_resource(uri)
        text = "".join(c.get("text", "") for c in contents if "text" in c)
        content_hash = hashlib.sha256(text.encode()).hexdigest()

        # Find the resource entry for metadata
        r = next((r for r in source.resources if r["uri"] == uri), {})
        meta = ResourceMeta(
            last_modified=r.get("annotations", {}).get("lastModified"),
            size=r.get("size"),
            content_hash=content_hash,
        )
        return text, meta

    async def _list_all_resources(self, client: McpClient, filter_pattern: str | None) -> list[dict]:
        """Paginate through all resources, optionally filtering by URI pattern."""
        resources = []
        cursor = None
        while True:
            page, cursor = await client.list_resources(cursor)
            resources.extend(page)
            if not cursor:
                break
        if filter_pattern:
            import fnmatch
            resources = [r for r in resources if fnmatch.fnmatch(r["uri"], filter_pattern)]
        return resources

    async def disconnect(self, name: str) -> None:
        source = self._sources.pop(name, None)
        if source:
            await self._pool.release(source.url)

@dataclass
class McpDocSource:
    client: McpClient
    url: str
    resources: list[dict]
    config: dict
    stored_meta: dict[str, "ResourceMeta"] = field(default_factory=dict)
```

### McpApiProvider

Implements `DataSourceProvider` for MCP tools → API operations:

```python
class McpApiProvider:
    """Exposes MCP tools as constat API operations."""

    kind = DataSourceKind.API

    def __init__(self, pool: McpClientPool):
        self._pool = pool
        self._sources: dict[str, McpApiSource] = {}

    async def connect(self, name: str, config: dict, auth: AuthConfig | None) -> ConnectionResult:
        url = config["url"]
        client = await self._pool.acquire(url, auth)

        if "tools" not in client._capabilities:
            return ConnectionResult(success=False, error="MCP server has no tools capability")

        tools = await self._list_all_tools(client, config)

        self._sources[name] = McpApiSource(
            client=client,
            url=url,
            tools=tools,
            config=config,
        )
        return ConnectionResult(success=True, capabilities={"queryable"})

    def discover(self, name: str) -> DiscoveryResult:
        """Return MCP tools as API operations for the query engine."""
        source = self._sources[name]
        operations = []
        for tool in source.tools:
            operations.append({
                "name": tool["name"],
                "description": tool.get("description", ""),
                "input_schema": tool.get("inputSchema", {}),
                "output_schema": tool.get("outputSchema"),
            })
        return DiscoveryResult(
            items=[
                SourceItem(id=t["name"], name=t["name"], description=t.get("description", ""))
                for t in source.tools
            ],
            operations=operations,
        )

    async def call_operation(self, name: str, tool_name: str, arguments: dict) -> dict:
        """Invoke an MCP tool. Called by the query engine."""
        source = self._sources[name]
        result = await source.client.call_tool(tool_name, arguments)
        if result.get("isError"):
            error_text = self._extract_text(result.get("content", []))
            raise RuntimeError(f"MCP tool error: {error_text}")
        # Return structured content if available, else extract text
        if "structuredContent" in result:
            return result["structuredContent"]
        return {"result": self._extract_text(result.get("content", []))}

    async def _list_all_tools(self, client: McpClient, config: dict) -> list[dict]:
        """Paginate tools, apply allow/deny filters."""
        tools = []
        cursor = None
        while True:
            page, cursor = await client.list_tools(cursor)
            tools.extend(page)
            if not cursor:
                break
        allowed = set(config.get("allowed_tools", []))
        denied = set(config.get("denied_tools", []))
        if allowed:
            tools = [t for t in tools if t["name"] in allowed]
        if denied:
            tools = [t for t in tools if t["name"] not in denied]
        return tools

    def _extract_text(self, content: list[dict]) -> str:
        return "\n".join(c.get("text", "") for c in content if c.get("type") == "text")

    async def disconnect(self, name: str) -> None:
        source = self._sources.pop(name, None)
        if source:
            await self._pool.release(source.url)

@dataclass
class McpApiSource:
    client: McpClient
    url: str
    tools: list[dict]
    config: dict
```

## Integration Points

### Transport layer (`_transport.py`)

```python
def infer_transport(config: "DocumentConfig") -> str:
    if config.type == "mcp":
        return "mcp"
    # ... existing scheme detection ...
```

```python
async def _fetch_mcp(config: "DocumentConfig") -> FetchResult:
    """Fetch a single MCP resource for document loading."""
    pool = get_mcp_pool()  # singleton per session
    client = await pool.acquire(config.url, _extract_auth(config))
    try:
        if config.mcp_resource_uri:
            contents = await client.read_resource(config.mcp_resource_uri)
        else:
            # Fetch all resources, concatenate
            resources, _ = await client.list_resources()
            # Apply filter
            if config.resource_filter:
                resources = [r for r in resources if fnmatch.fnmatch(r["uri"], config.resource_filter)]
            contents = []
            for r in resources[:config.max_resources or 100]:
                contents.extend(await client.read_resource(r["uri"]))

        # Combine text contents
        text_parts = [c["text"] for c in contents if "text" in c]
        combined = "\n\n---\n\n".join(text_parts)
        return FetchResult(
            data=combined.encode("utf-8"),
            detected_mime="text/markdown",
            source_path=config.url,
        )
    finally:
        await pool.release(config.url)
```

### Source refresher (`source_refresher.py`)

```python
def _classify_source(file_ref: dict) -> str | None:
    doc_config = file_ref.get("document_config", {})
    if doc_config.get("type") == "mcp":
        return "mcp"
    # ... existing classification ...
```

```python
async def _refresh_mcp_source(managed, file_ref, name) -> tuple[bool, str, int]:
    """Re-fetch MCP resources, using ChangeProbe for efficient change detection."""
    pool = get_mcp_pool()
    config = file_ref.get("document_config", {})
    client = await pool.acquire(config["url"], _extract_auth(config))
    try:
        stored_meta = file_ref.get("_resource_meta", {})
        # Deserialize stored ResourceMeta dicts
        stored = {uri: ResourceMeta(**m) for uri, m in stored_meta.items()}

        probe = ChangeProbe()
        result = await probe.probe(client, stored, config.get("resource_filter"))

        changed_count = 0

        # Index new resources
        for r in result.added:
            contents = await client.read_resource(r["uri"])
            text = "".join(c.get("text", "") for c in contents if "text" in c)
            content_hash = hashlib.sha256(text.encode()).hexdigest()
            await _reindex_document(managed, name, r["uri"], text)
            stored_meta[r["uri"]] = {
                "last_modified": r.get("annotations", {}).get("lastModified"),
                "size": r.get("size"),
                "content_hash": content_hash,
            }
            changed_count += 1

        # Re-index changed resources
        for r in result.changed:
            contents = await client.read_resource(r["uri"])
            text = "".join(c.get("text", "") for c in contents if "text" in c)
            content_hash = hashlib.sha256(text.encode()).hexdigest()
            await _reindex_document(managed, name, r["uri"], text)
            stored_meta[r["uri"]] = {
                "last_modified": r.get("annotations", {}).get("lastModified"),
                "size": r.get("size"),
                "content_hash": content_hash,
            }
            changed_count += 1

        # Remove deleted resources
        for uri in result.removed:
            await _remove_document_chunks(managed, name, uri)
            stored_meta.pop(uri, None)
            changed_count += 1

        file_ref["_resource_meta"] = stored_meta
        return True, f"Refreshed {changed_count} resources", changed_count
    except Exception as e:
        return False, str(e), 0
    finally:
        await pool.release(config["url"])
```

### Query engine integration

MCP tools registered as API sources are available to the planner/query engine through the existing API operation dispatch. The query engine discovers operations via `discover()` and generates tool calls:

```python
# In query_engine.py or planner.py — existing API call dispatch
async def _execute_api_call(self, source_name: str, operation: str, params: dict) -> dict:
    provider = self._registry.get_provider(DataSourceKind.API, "mcp")
    if isinstance(provider, McpApiProvider):
        return await provider.call_operation(source_name, operation, params)
    # ... existing REST/GraphQL dispatch ...
```

The LLM sees MCP tools the same way it sees GraphQL mutations or REST endpoints — as named operations with typed inputs. The planner generates calls; constat dispatches them to the MCP server.

### DocumentConfig additions

```python
class DocumentConfig(BaseModel):
    # ... existing fields ...

    # MCP-specific (only used when type="mcp")
    resource_filter: str | None = None       # glob pattern for resource URIs
    max_resources: int = 100                 # cap on resources to ingest
    mcp_resource_uri: str | None = None      # single resource URI (skip listing)
```

### APIConfig additions

```python
class APIConfig(BaseModel):
    # ... existing fields ...

    # MCP-specific (only used when type="mcp")
    allowed_tools: list[str] | None = None   # allowlist tool names
    denied_tools: list[str] | None = None    # denylist tool names
```

## Change Detection (ChangeProbe)

MCP's `resources/list` returns optional metadata per resource — but `lastModified`, `size`, and `annotations` are all optional. There's no "give me resources changed since X" query. Most MCP servers won't include change metadata.

This is the key problem for vectorization: we need to know what's new, what's changed, and what's gone — without downloading every resource on every refresh cycle.

### The Problem

| What we need to detect | Difficulty |
|---|---|
| **New** resources (not yet indexed) | Easy — compare URI sets |
| **Removed** resources (need chunk deletion) | Easy — compare URI sets |
| **Changed** resources (need re-indexing) | Hard — no standard mechanism |

### Tiered Probe Strategy

`ChangeProbe` tries the cheapest detection first, falls back to more expensive methods only for resources that can't be resolved from metadata alone:

```python
@dataclass
class ResourceMeta:
    """Stored per-resource metadata from the last successful sync."""
    last_modified: str | None = None
    size: int | None = None
    content_hash: str | None = None    # always computed on first fetch

@dataclass
class ProbeResult:
    added: list[dict]                  # new resources (not in stored)
    changed: list[dict]                # content changed since last sync
    removed: list[str]                 # URIs no longer in resource list
    unchanged: list[str]               # confirmed unchanged (skipped)
    current_resources: list[dict]      # full resource list from server

class ChangeProbe:
    """Tiered change detection for MCP resources."""

    async def probe(
        self,
        client: McpClient,
        stored: dict[str, ResourceMeta],
        resource_filter: str | None = None,
    ) -> ProbeResult:
        # Step 1: List current resources (metadata only — no content fetch)
        resources = await _list_all_resources(client, resource_filter)
        current_uris = {r["uri"] for r in resources}
        stored_uris = set(stored.keys())

        # Step 2: New and removed — always detectable from URI sets
        added = [r for r in resources if r["uri"] not in stored_uris]
        removed = [uri for uri in stored_uris if uri not in current_uris]

        # Step 3: For resources that exist in both, try tiered detection
        changed = []
        unchanged = []
        needs_fetch = []   # couldn't resolve from metadata

        for r in resources:
            uri = r["uri"]
            if uri not in stored_uris:
                continue   # already in added
            prev = stored[uri]

            # Tier 1: lastModified (free — already in list response)
            last_mod = r.get("annotations", {}).get("lastModified")
            if last_mod and prev.last_modified:
                if last_mod != prev.last_modified:
                    changed.append(r)
                else:
                    unchanged.append(uri)
                continue   # resolved either way

            # Tier 2: size (free — already in list response)
            size = r.get("size")
            if size is not None and prev.size is not None:
                if size != prev.size:
                    changed.append(r)
                else:
                    # Same size doesn't guarantee same content, but
                    # combined with no lastModified change it's a good signal.
                    # Still mark for hash check if we have a stored hash.
                    if prev.content_hash:
                        needs_fetch.append(r)
                    else:
                        unchanged.append(uri)
                continue

            # Tier 3: no metadata available — must fetch content
            if prev.content_hash:
                needs_fetch.append(r)
            else:
                # No stored hash either — treat as changed (first sync had no hash)
                changed.append(r)

        # Step 4: Fetch and hash only the unresolved resources
        for r in needs_fetch:
            contents = await client.read_resource(r["uri"])
            text = "".join(c.get("text", "") for c in contents if "text" in c)
            current_hash = hashlib.sha256(text.encode()).hexdigest()
            if current_hash != stored[r["uri"]].content_hash:
                changed.append(r)
            else:
                unchanged.append(r["uri"])

        return ProbeResult(
            added=added,
            changed=changed,
            removed=removed,
            unchanged=unchanged,
            current_resources=resources,
        )
```

### Cost Analysis

```
Best case (server provides lastModified):
  resources/list  → 1 API call
  resources/read  → 0 calls (all resolved from metadata)
  Total: O(1) regardless of resource count

Typical case (server provides size but not lastModified):
  resources/list  → 1 API call
  resources/read  → only for size-ambiguous resources
  Total: O(ambiguous) — usually small

Worst case (server provides no metadata):
  resources/list  → 1 API call
  resources/read  → 1 call per resource (full content fetch + hash)
  Total: O(N) — but only on first sync; subsequent syncs have stored hashes
```

After the first sync, every resource has a stored `content_hash`. So even servers with no metadata become efficient on subsequent syncs — only resources whose hash actually changes trigger re-indexing.

### Stored Metadata Persistence

`ResourceMeta` is stored per-source in the session's `file_ref`:

```python
file_ref["_resource_meta"] = {
    "confluence://page/123": {
        "last_modified": "2026-03-25T10:00:00Z",
        "size": 4096,
        "content_hash": "a1b2c3d4..."
    },
    "confluence://page/456": {
        "last_modified": null,     # server didn't provide
        "size": null,
        "content_hash": "e5f6g7h8..."   # computed on fetch
    }
}
```

Persisted to session `config.yaml` (Tier 5) or user `config.yaml` (Tier 3) depending on source scope. Survives session restore.

### MCP Subscribe (Optional Enhancement)

If the server declares `{ resources: { subscribe: true } }`, constat can subscribe to change notifications instead of polling:

```python
# On connect, subscribe to all indexed resources
for uri in source.stored_meta:
    await client.subscribe_resource(uri)

# Server pushes: notifications/resources/updated { uri }
# → re-fetch only that resource, update hash
```

This eliminates polling entirely for servers that support it. But since it's optional, `ChangeProbe` is always the primary mechanism.

## Data Flow

### Resources → Documents

```
Config: type=mcp, url=https://mcp.example.com/sse
  │
  ├─ connect() → McpClient.connect() → capabilities negotiation
  │
  ├─ resources/list → [{uri, name, mimeType, size, annotations}]
  │   └─ apply resource_filter glob
  │
  ├─ resources/read (per resource) → {text | blob}
  │   └─ FetchResult(data, mime_type)
  │   └─ compute content_hash, store ResourceMeta
  │
  ├─ _extract_content() → markdown/text
  │   └─ standard extraction pipeline (HTML→md, PDF→text, etc.)
  │
  ├─ chunk → embed → store in DuckDB vector store
  │
  └─ auto_refresh → ChangeProbe.probe()
      ├─ Tier 1: lastModified comparison (free)
      ├─ Tier 2: size comparison (free)
      ├─ Tier 3: content hash comparison (fetch required)
      └─ re-index only added + changed resources
```

### Tools → API Operations

```
Config: type=mcp, url=https://mcp.example.com/sse
  │
  ├─ connect() → McpClient.connect() → capabilities negotiation
  │
  ├─ tools/list → [{name, description, inputSchema, outputSchema}]
  │   └─ apply allowed_tools / denied_tools filters
  │
  ├─ discover() → operations registered in query engine
  │   └─ LLM sees: "search_issues(query: str, status: str) → list[Issue]"
  │
  └─ query engine generates tool call
      └─ tools/call → {content, structuredContent, isError}
          └─ result returned to LLM for reasoning
```

## Auth

MCP servers use OAuth bearer tokens. Constat's auth flow:

1. **Config-provided token**: `auth.token` or `auth.token_ref` → `tokens.yaml`
2. **Server OAuth flow**: For MCP servers requiring interactive OAuth (like Claude connectors):
   - Server returns 401 → constat initiates OAuth via the MCP authorization spec
   - Browser popup (reuse existing `oauth_email.py` pattern)
   - Access token stored in `tokens.yaml` via `token_ref`
   - Refresh token handled transparently

```yaml
# tokens.yaml
tokens:
  confluence-mcp-token:
    provider: custom
    access_token: "<token>"
    refresh_token: "<refresh>"
    token_endpoint: "https://auth.atlassian.com/oauth/token"
    client_id: "${MCP_CONFLUENCE_CLIENT_ID}"
    scopes: "read:confluence-content.all"
    expires_at: "2026-03-25T12:00:00Z"
```

Token refresh on 401:

```python
async def _rpc(self, method: str, params: dict) -> dict:
    response = await self._send(method, params)
    if response.get("error", {}).get("code") == -32001:  # auth error
        await self._refresh_token()
        response = await self._send(method, params)
    return response.get("result", {})
```

## MCP Server Catalog

### Goal

Users adding an MCP data source need to discover available servers. Instead of manually entering URLs, the UI presents a searchable catalog of MCP servers with names, descriptions, icons, and auth requirements.

### Architecture

Three-tier catalog with fallback chain:

```
Remote directory (PulseMCP API)
  ↓ fetch + cache
Local cache (.constat/mcp_catalog.json)
  ↓ fallback if remote unavailable
Bundled seed (constat/data/mcp_catalog_seed.json)
```

### Bundled seed file

A static snapshot of popular MCP servers, shipped with the codebase. Updated periodically with releases.

```
constat/data/mcp_catalog_seed.json
```

```json
{
  "version": "2026-03-25",
  "servers": [
    {
      "name": "Gmail",
      "slug": "gmail",
      "url": "https://mcp-gmail.example.com/sse",
      "description": "Gmail email access — read, search, send, organize",
      "icon": "mail",
      "category": "email",
      "capabilities": ["resources", "tools"],
      "auth_type": "oauth2",
      "auth_provider": "google",
      "scopes": ["https://mail.google.com/"],
      "source_url": "https://github.com/marlinjai/email-mcp"
    },
    {
      "name": "Google Drive",
      "slug": "google-drive",
      "url": "https://mcp-gdrive.example.com/sse",
      "description": "Google Drive file access — list, read, search",
      "icon": "folder",
      "category": "drive",
      "capabilities": ["resources", "tools"],
      "auth_type": "oauth2",
      "auth_provider": "google",
      "scopes": ["https://www.googleapis.com/auth/drive.readonly"],
      "source_url": "https://github.com/modelcontextprotocol/servers"
    },
    {
      "name": "Slack",
      "slug": "slack",
      "url": "https://mcp-slack.example.com/sse",
      "description": "Slack workspace — channels, messages, search",
      "icon": "message-square",
      "category": "communication",
      "capabilities": ["resources", "tools"],
      "auth_type": "oauth2",
      "auth_provider": "custom",
      "source_url": "https://github.com/modelcontextprotocol/servers"
    },
    {
      "name": "Notion",
      "slug": "notion",
      "url": "https://mcp-notion.example.com/sse",
      "description": "Notion pages, databases, and blocks",
      "icon": "file-text",
      "category": "documents",
      "capabilities": ["resources", "tools"],
      "auth_type": "bearer",
      "source_url": "https://github.com/modelcontextprotocol/servers"
    },
    {
      "name": "Confluence",
      "slug": "confluence",
      "url": "https://mcp-confluence.example.com/sse",
      "description": "Confluence spaces and pages",
      "icon": "book-open",
      "category": "documents",
      "capabilities": ["resources"],
      "auth_type": "oauth2",
      "auth_provider": "atlassian",
      "source_url": "https://github.com/atlassian/mcp-confluence"
    },
    {
      "name": "Jira",
      "slug": "jira",
      "url": "https://mcp-jira.example.com/sse",
      "description": "Jira issues, projects, and boards",
      "icon": "check-square",
      "category": "project-management",
      "capabilities": ["tools"],
      "auth_type": "oauth2",
      "auth_provider": "atlassian",
      "source_url": "https://github.com/atlassian/mcp-jira"
    },
    {
      "name": "GitHub",
      "slug": "github",
      "url": "https://mcp-github.example.com/sse",
      "description": "GitHub repos, issues, PRs, and code search",
      "icon": "git-branch",
      "category": "development",
      "capabilities": ["resources", "tools"],
      "auth_type": "bearer",
      "source_url": "https://github.com/modelcontextprotocol/servers"
    }
  ]
}
```

### Backend: McpCatalog

```python
class McpCatalog:
    """MCP server catalog with remote fetch, local cache, and bundled seed fallback."""

    REMOTE_URL = "https://www.pulsemcp.com/api/servers"   # or similar directory API
    CACHE_TTL = 86400  # 24 hours

    def __init__(self, cache_dir: Path, seed_path: Path):
        self._cache_dir = cache_dir
        self._seed_path = seed_path
        self._cache_path = cache_dir / "mcp_catalog.json"

    async def list_servers(self, category: str | None = None, query: str | None = None) -> list[dict]:
        """Return catalog entries, filtered by category or search query."""
        catalog = await self._load()
        servers = catalog.get("servers", [])

        if category:
            servers = [s for s in servers if s.get("category") == category]
        if query:
            q = query.lower()
            servers = [s for s in servers if q in s["name"].lower() or q in s.get("description", "").lower()]

        return servers

    async def get_server(self, slug: str) -> dict | None:
        """Get a specific server entry by slug."""
        catalog = await self._load()
        return next((s for s in catalog.get("servers", []) if s["slug"] == slug), None)

    async def _load(self) -> dict:
        """Load catalog: try cache → remote (refresh cache) → seed."""

        # 1. Check cache freshness
        if self._cache_path.exists():
            cache = json.loads(self._cache_path.read_text())
            cached_at = cache.get("_cached_at", 0)
            if time.time() - cached_at < self.CACHE_TTL:
                return cache

        # 2. Try remote directory
        try:
            catalog = await self._fetch_remote()
            catalog["_cached_at"] = time.time()
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(json.dumps(catalog, indent=2))
            return catalog
        except Exception:
            pass  # remote unavailable — fall through

        # 3. Use stale cache if available (better than seed)
        if self._cache_path.exists():
            return json.loads(self._cache_path.read_text())

        # 4. Fall back to bundled seed
        return json.loads(self._seed_path.read_text())

    async def _fetch_remote(self) -> dict:
        """Fetch catalog from remote MCP directory."""
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(self.REMOTE_URL)
            resp.raise_for_status()
            return resp.json()
```

### API routes

```python
# constat/server/routes/mcp_catalog.py

@router.get("/mcp/catalog")
async def list_mcp_servers(
    category: str | None = None,
    q: str | None = None,
) -> list[dict]:
    """List MCP servers from the catalog."""
    catalog = get_mcp_catalog()
    return await catalog.list_servers(category=category, query=q)

@router.get("/mcp/catalog/{slug}")
async def get_mcp_server(slug: str) -> dict:
    """Get details for a specific MCP server."""
    catalog = get_mcp_catalog()
    server = await catalog.get_server(slug)
    if not server:
        raise HTTPException(status_code=404, detail="MCP server not found")
    return server
```

### Frontend: MCP Server Picker

When user clicks "Add MCP Source" in the Sources panel, a modal shows the catalog:

```tsx
function McpServerPicker({ onSelect }: { onSelect: (server: McpServer) => void }) {
  const [servers, setServers] = useState<McpServer[]>([])
  const [query, setQuery] = useState('')
  const [category, setCategory] = useState<string | null>(null)

  useEffect(() => {
    fetchMcpCatalog({ q: query, category }).then(setServers)
  }, [query, category])

  const categories = ['email', 'documents', 'drive', 'communication',
                      'project-management', 'development']

  return (
    <div>
      <input
        placeholder="Search MCP servers..."
        value={query}
        onChange={e => setQuery(e.target.value)}
      />
      <div className="flex gap-2 my-3">
        {categories.map(cat => (
          <button
            key={cat}
            onClick={() => setCategory(category === cat ? null : cat)}
            className={category === cat ? 'bg-primary-100 ...' : 'bg-gray-50 ...'}
          >
            {cat}
          </button>
        ))}
      </div>
      <div className="grid grid-cols-3 gap-3">
        {servers.map(server => (
          <button
            key={server.slug}
            onClick={() => onSelect(server)}
            className="p-3 border rounded hover:border-primary-300 text-left"
          >
            <Icon name={server.icon} className="w-6 h-6 mb-1" />
            <div className="font-medium text-sm">{server.name}</div>
            <div className="text-xs text-gray-500 line-clamp-2">{server.description}</div>
          </button>
        ))}
      </div>
      {/* Custom URL option */}
      <button
        onClick={() => onSelect({ slug: 'custom', name: 'Custom', url: '' })}
        className="w-full mt-3 p-3 border-dashed border-2 rounded text-center text-gray-500"
      >
        + Custom MCP Server URL
      </button>
    </div>
  )
}
```

After selecting a server, the existing "Add Source" modal pre-fills the URL and auth type, then prompts for OAuth or token entry.

### Catalog entry schema

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | string | yes | Display name |
| `slug` | string | yes | URL-safe identifier |
| `url` | string | yes | Default MCP server URL |
| `description` | string | yes | Short description |
| `icon` | string | no | Lucide icon name |
| `category` | string | no | Grouping: email, documents, drive, communication, etc. |
| `capabilities` | string[] | no | `["resources"]`, `["tools"]`, or both |
| `auth_type` | string | no | `"none"`, `"bearer"`, `"oauth2"` |
| `auth_provider` | string | no | `"google"`, `"microsoft"`, `"atlassian"`, `"custom"` |
| `scopes` | string[] | no | OAuth2 scopes (pre-filled in auth flow) |
| `source_url` | string | no | Link to source code / docs |

### File changes

| File | Change |
|---|---|
| `constat/mcp/catalog.py` | **New**: `McpCatalog` |
| `constat/data/mcp_catalog_seed.json` | **New**: bundled seed catalog |
| `constat/server/routes/mcp_catalog.py` | **New**: catalog API routes |
| `constat-ui/src/components/artifacts/McpServerPicker.tsx` | **New**: catalog picker modal |
| `constat-ui/src/api/sessions.ts` | Add `fetchMcpCatalog()` client function |
| `tests/test_mcp_catalog.py` | **New**: cache fallback, remote fetch, seed loading, search/filter |

### Testing

1. **Remote available**: fetch → cache written → returns remote data
2. **Remote down, cache fresh**: returns cached data
3. **Remote down, cache stale**: returns stale cache (better than seed)
4. **Remote down, no cache**: returns bundled seed
5. **Cache TTL**: expired cache triggers remote fetch
6. **Category filter**: returns only matching category
7. **Search query**: matches name and description
8. **Slug lookup**: returns specific server or 404

## Dependencies

```
httpx[http2]          # HTTP/SSE transport (already in deps)
```

No new dependencies required. `httpx` handles both Streamable HTTP and SSE. JSON-RPC is implemented directly (no SDK dependency — the protocol is simple enough).

For stdio transport, `subprocess` from stdlib.

## File Changes

| File | Change |
|---|---|
| `constat/mcp/__init__.py` | **New**: `McpClient`, `McpClientPool` |
| `constat/mcp/change_probe.py` | **New**: `ChangeProbe`, `ResourceMeta`, `ProbeResult` |
| `constat/mcp/document_provider.py` | **New**: `McpDocumentProvider` |
| `constat/mcp/api_provider.py` | **New**: `McpApiProvider` |
| `constat/core/config.py` | Add `resource_filter`, `max_resources`, `mcp_resource_uri` to `DocumentConfig`; add `allowed_tools`, `denied_tools` to `APIConfig` |
| `constat/discovery/doc_tools/_transport.py` | Add `mcp` transport detection and `_fetch_mcp()` |
| `constat/server/source_refresher.py` | Add `mcp` classification and `_refresh_mcp_source()` |
| `constat/core/sources.py` | Register `McpDocumentProvider` and `McpApiProvider` in `create_registry()` |
| `tests/test_mcp_client.py` | **New**: client connection, resource listing/reading, tool listing/calling |
| `tests/test_mcp_change_probe.py` | **New**: tiered detection, hash fallback, metadata persistence |
| `tests/test_mcp_providers.py` | **New**: provider lifecycle, document ingestion, API dispatch |

## Testing

### Mock MCP server

Tests use an in-process mock MCP server (FastAPI + SSE) that serves canned resources and tools:

```python
@pytest.fixture
def mcp_server():
    """Spawn a local MCP server for integration tests."""
    app = FastAPI()
    resources = [
        {"uri": "test://doc1", "name": "doc1", "mimeType": "text/plain"},
        {"uri": "test://doc2", "name": "doc2", "mimeType": "text/markdown"},
    ]
    tools = [
        {"name": "search", "description": "Search", "inputSchema": {"type": "object", "properties": {"q": {"type": "string"}}}},
    ]
    # JSON-RPC endpoint
    @app.post("/")
    async def rpc(request: Request):
        body = await request.json()
        method = body["method"]
        if method == "initialize":
            return {"jsonrpc": "2.0", "id": body["id"], "result": {"capabilities": {"resources": {}, "tools": {}}}}
        if method == "resources/list":
            return {"jsonrpc": "2.0", "id": body["id"], "result": {"resources": resources}}
        if method == "resources/read":
            uri = body["params"]["uri"]
            return {"jsonrpc": "2.0", "id": body["id"], "result": {"contents": [{"uri": uri, "mimeType": "text/plain", "text": f"Content of {uri}"}]}}
        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": body["id"], "result": {"tools": tools}}
        if method == "tools/call":
            return {"jsonrpc": "2.0", "id": body["id"], "result": {"content": [{"type": "text", "text": "result"}]}}
    # Run in background thread
    ...
```

### Test cases

1. **Client lifecycle**: connect → list resources → read resource → disconnect
2. **Resource ingestion**: MCP resources → FetchResult → chunks in vector store
3. **Tool discovery**: MCP tools → API operations visible to query engine
4. **Tool execution**: query engine → tools/call → result returned
5. **Resource filter**: `resource_filter` glob reduces ingested resources
6. **Tool allow/deny**: `allowed_tools` / `denied_tools` filter correctly
7. **Connection pooling**: two sources pointing to same URL share one client
8. **Auth token refresh**: 401 → refresh → retry succeeds
9. **ChangeProbe — Tier 1**: server provides `lastModified` → no content fetch needed
10. **ChangeProbe — Tier 2**: server provides `size` only → size change detected without fetch
11. **ChangeProbe — Tier 3**: no metadata → content hash computed, compared with stored hash
12. **ChangeProbe — first sync**: all resources fetched, hashes stored, subsequent syncs efficient
13. **ChangeProbe — metadata persistence**: `_resource_meta` survives session restore via config.yaml
14. **Refresh with subscribe**: server supports subscribe → push notification triggers single re-fetch
10. **stdio transport**: subprocess spawned, JSON-RPC over stdin/stdout
11. **Server without resources**: connect for tools-only source succeeds
12. **Server without tools**: connect for documents-only source succeeds
13. **Empty server**: connect returns error (no capabilities)
14. **Binary resources**: blob content decoded and passed to extraction pipeline

## Edge Cases

| Case | Handling |
|---|---|
| MCP server down at connect time | `ConnectionResult(success=False, error=...)` — source shows error state in UI |
| MCP server down at refresh time | Refresh fails silently, keeps cached content, retries next interval |
| Resource returns binary (blob) | Base64 decoded, passed to `_extract_content()` for format detection |
| Resource URI contains special chars | URI passed as-is to `resources/read` (server handles its own URI scheme) |
| Server has 1000+ resources | Paginated via `nextCursor`; capped by `max_resources` config |
| Tool call times out | 30s default timeout, returns error to query engine |
| Tool returns `isError: true` | `RuntimeError` raised, query engine retries or reports to user |
| Same URL in documents + apis config | Single `McpClient` via pool, refcount=2 |
| Server capabilities change | Re-discovered on refresh; tools/resources list may change |
| stdio server crashes | Detected via broken pipe; reconnect on next operation |
| OAuth token expired | 401 detected in `_rpc()`, automatic refresh via `tokens.yaml` |
