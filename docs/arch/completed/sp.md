# SharePoint Document Source (Modern + Legacy)

## Goal

Add SharePoint as a unified document source that discovers and indexes:
- **Document libraries** — files (PDF, DOCX, XLSX, PPTX, images, etc.)
- **Lists** — structured tabular data (exposed as queryable tables via DuckDB)
- **Calendars** — events from SharePoint calendar lists
- **Pages** — SharePoint modern/classic page content

Support both **modern SharePoint Online** (Microsoft Graph API) and **legacy SharePoint on-premises** (SharePoint REST API / CSOM). Auto-detect protocol from the site URL.

## MCP-First Strategy

**Default approach**: Use existing MCP servers for SharePoint Online rather than building custom integrations. The [MCP ecosystem](https://modelcontextprotocol.io/) includes SharePoint/Microsoft 365 connectors that expose documents as MCP resources and operations as MCP tools.

### MCP config (preferred — SharePoint Online only)

```yaml
documents:
  sp-docs:
    type: mcp
    url: https://mcp-sharepoint.example.com/sse
    auth:
      method: bearer
      token_ref: sp-mcp-token
    resource_filter: "sharepoint://sites/analytics/Shared Documents/*"
    description: "Analytics SharePoint documents"

apis:
  sp-ops:
    type: mcp
    url: https://mcp-sharepoint.example.com/sse
    auth:
      method: bearer
      token_ref: sp-mcp-token
    allowed_tools:
      - search_documents
      - get_list_items
    description: "SharePoint search and list operations"
```

MCP resources → document pipeline (vectorized). MCP tools → API operations (query engine). See `mcp.md`.

### When to fall back to custom `SharePointClient`

The custom implementation below is needed when MCP **cannot** cover the use case:

| Gap | Why MCP cannot help | Custom solution |
|---|---|---|
| **SharePoint on-premises** | MCP servers are HTTP-only; on-prem sites use REST API (`_api/`) with NTLM auth | `SharePointRestClient` with NTLM + `_api/` endpoints |
| **Lists → DuckDB tables** | MCP resources are content blobs; constat needs Arrow tables registered in DuckDB | `_list_to_arrow()` with column type mapping |
| **Calendar lists** | SharePoint calendars are lists (BaseTemplate=106), not standard calendar APIs | `_calendar_items_to_events()` conversion |
| **Site page extraction** | Modern pages use CanvasContent1 JSON, wiki pages use WikiField HTML | Custom page parsers per page type |
| **Site-wide auto-discovery** | MCP server may expose a flat file list, not site structure | `_discover_site()` enumerating libraries/lists/calendars |
| **Dual protocol (Graph + REST)** | MCP only works with one protocol | `_detect_sp_protocol()` auto-switches |

**Decision matrix**:

| Scenario | Approach |
|---|---|
| SharePoint Online, document libraries only | MCP (preferred) |
| SharePoint Online, documents + lists as tables | MCP for docs, custom for lists→DuckDB |
| SharePoint Online, full site discovery | Custom (MCP lacks site introspection) |
| SharePoint on-premises (any) | Custom (MCP doesn't support NTLM/REST API) |
| Hybrid (Online + on-prem) | MCP for Online, custom for on-prem |

---

The rest of this document specifies the custom implementation (required for on-premises and advanced scenarios).

## Protocol Detection

```
SharePoint Online  →  https://*.sharepoint.com/*    →  Microsoft Graph API
SharePoint On-Prem →  https://sp.company.local/*     →  SharePoint REST API (_api/)
```

```python
def _detect_sp_protocol(url: str) -> Literal["graph", "rest"]:
    parsed = urlparse(url)
    if "sharepoint.com" in parsed.hostname:
        return "graph"
    return "rest"
```

The user can also force the protocol via config:

```yaml
protocol: graph   # or "rest" — overrides auto-detection
```

## Addressing Scheme

```
<data_source>:<resource_type>:<resource_id>                  # top-level resource
<data_source>:<resource_type>:<resource_id>:<child>          # child (attachment, embedded image)
```

Resource types: `lib` (library file), `list` (list row/table), `cal` (calendar event), `page` (site page).

Examples:
```
# Document library files
sp-analytics:lib:f_abc123_quarterly_report
sp-analytics:lib:f_abc123_quarterly_report:page_2_img_1.png

# List as a table (the whole list becomes one DuckDB table)
sp-analytics:list:project_tracker

# Calendar events
sp-analytics:cal:evt_20260315_standup_abc

# Site pages
sp-analytics:page:pg_welcome_to_analytics
```

## Configuration

### Minimal — auto-discover everything on a site:

```yaml
documents:
  sp-analytics:
    type: sharepoint
    site_url: https://contoso.sharepoint.com/sites/analytics
    description: "Analytics team SharePoint site"

    oauth2_client_id: ${AZURE_CLIENT_ID}
    oauth2_client_secret: ${AZURE_CLIENT_SECRET}
    oauth2_tenant_id: ${AZURE_TENANT_ID}
    oauth2_scopes:
      - "https://graph.microsoft.com/.default"
```

### Targeted — specify what to index:

```yaml
documents:
  sp-analytics:
    type: sharepoint
    site_url: https://contoso.sharepoint.com/sites/analytics
    description: "Analytics team SharePoint site"

    # Auth
    oauth2_client_id: ${AZURE_CLIENT_ID}
    oauth2_client_secret: ${AZURE_CLIENT_SECRET}
    oauth2_tenant_id: ${AZURE_TENANT_ID}

    # What to discover (default: all)
    discover_libraries: true           # index document libraries (default: true)
    discover_lists: true               # index lists as tables (default: true)
    discover_calendars: true           # index calendar lists (default: true)
    discover_pages: true               # index site pages (default: true)

    # Library options
    library_names:                     # allowlist; omit = all libraries
      - "Shared Documents"
      - "Reports"
    library_folder: "/Quarterly"       # subfolder within library (optional)
    recursive: true                    # traverse subfolders (default: true)
    max_files: 500
    include_types:
      - ".pdf"
      - ".docx"
      - ".xlsx"
      - ".pptx"
    exclude_patterns:
      - "^~\\$"                        # Office temp files
      - "^\\."                         # hidden files

    # List options
    list_names:                        # allowlist; omit = all non-system lists
      - "Project Tracker"
      - "Inventory"
    max_rows: 10000                    # max rows per list (default: 10000)
    list_as_table: true                # expose as DuckDB table (default: true)
    list_as_document: false            # also index as document text (default: false)

    # Calendar options
    calendar_names:                    # allowlist; omit = all calendar lists
      - "Team Calendar"
    since: "2026-01-01"
    until: "2026-12-31"
    expand_recurring: true

    # Page options
    page_types:                        # "modern" | "classic" | "wiki" (default: all)
      - modern
      - wiki

    # Common
    since: "2026-01-01"                # modified-after filter for all resource types
    extract_images: true
    extract_attachments: true
```

### Legacy SharePoint (on-premises):

```yaml
documents:
  sp-legacy:
    type: sharepoint
    site_url: https://sp.company.local/sites/analytics
    protocol: rest                     # force REST API (auto-detected for non-.sharepoint.com)
    description: "On-prem SharePoint"

    # NTLM or basic auth for on-prem
    auth_type: ntlm                    # "ntlm" | "basic" | "oauth2"
    username: ${SP_USER}
    password: ${SP_PASS}

    # Same discovery options as above
    discover_libraries: true
    discover_lists: true
```

### `DocumentConfig` additions

```python
class DocumentConfig(BaseModel):
    ...
    # SharePoint fields
    site_url: Optional[str] = None            # SharePoint site URL
    protocol: Optional[str] = None            # "graph" | "rest" — auto-detect if omitted

    # Discovery toggles
    discover_libraries: bool = True
    discover_lists: bool = True
    discover_calendars: bool = True
    discover_pages: bool = True

    # Library options
    library_names: Optional[list[str]] = None
    library_folder: Optional[str] = None
    # recursive, max_files, include_types, exclude_patterns — already exist

    # List options
    list_names: Optional[list[str]] = None
    max_rows: int = 10000
    list_as_table: bool = True
    list_as_document: bool = False

    # Calendar options
    calendar_names: Optional[list[str]] = None
    # since, until, expand_recurring — reuse from calendar spec

    # Page options
    page_types: Optional[list[str]] = None    # "modern", "classic", "wiki"
```

## Pipeline Overview

```
Authenticate (OAuth2 / NTLM / Basic)
  → Resolve site ID (Graph) or site URL (REST)
  → Discover resources:
      ├─ enumerate document libraries → list files → download → extract → chunk → embed
      ├─ enumerate lists → fetch rows → convert to DataFrame → register as DuckDB table
      ├─ enumerate calendar lists → fetch events → render as documents → chunk → embed
      └─ enumerate site pages → fetch page content → chunk → embed
```

## Implementation

### New file: `constat/discovery/doc_tools/_sharepoint.py`

```python
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Literal

class SPProtocol(str, Enum):
    GRAPH = "graph"
    REST = "rest"

@dataclass
class SPSite:
    site_id: str             # Graph: site ID; REST: site URL
    display_name: str
    web_url: str
    protocol: SPProtocol

@dataclass
class SPLibrary:
    library_id: str          # Graph: drive ID; REST: list ID
    name: str
    web_url: str

@dataclass
class SPFile:
    file_id: str
    name: str
    mime_type: str
    size: int | None
    modified_time: datetime
    path: str                # path within library
    library_name: str
    web_url: str | None

@dataclass
class SPList:
    list_id: str
    name: str
    item_count: int
    columns: list[SPListColumn]
    is_calendar: bool
    is_hidden: bool

@dataclass
class SPListColumn:
    name: str
    internal_name: str       # SharePoint internal field name
    column_type: str         # text, number, datetime, choice, lookup, person, boolean, currency, url
    choices: list[str] | None

@dataclass
class SPListItem:
    item_id: str
    fields: dict             # column_internal_name → value

@dataclass
class SPPage:
    page_id: str
    title: str
    page_type: str           # "modern" | "classic" | "wiki"
    content_html: str
    modified_time: datetime
    web_url: str
```

### SharePoint Client (Protocol Abstraction)

```python
class SharePointClient:
    """Unified client abstracting Graph API and REST API."""

    def __init__(self, config: DocumentConfig, config_dir: Path | None = None):
        self._config = config
        self._config_dir = config_dir
        self._protocol = self._resolve_protocol()
        self._session = self._build_session()

    def _resolve_protocol(self) -> SPProtocol:
        if self._config.protocol:
            return SPProtocol(self._config.protocol)
        return SPProtocol.GRAPH if _detect_sp_protocol(self._config.site_url) == "graph" else SPProtocol.REST

    def _build_session(self) -> httpx.Client:
        """Build authenticated HTTP session."""
        if self._config.auth_type == "ntlm":
            return self._build_ntlm_session()
        # OAuth2 (default for Graph, optional for REST)
        token = self._get_access_token()
        return httpx.Client(headers={"Authorization": f"Bearer {token}"})

    # --- Site resolution ---

    def resolve_site(self) -> SPSite:
        if self._protocol == SPProtocol.GRAPH:
            return self._resolve_site_graph()
        return self._resolve_site_rest()

    def _resolve_site_graph(self) -> SPSite:
        """Resolve site URL to Graph site ID."""
        parsed = urlparse(self._config.site_url)
        hostname = parsed.hostname
        site_path = parsed.path.rstrip("/")
        url = f"https://graph.microsoft.com/v1.0/sites/{hostname}:{site_path}"
        resp = self._session.get(url)
        resp.raise_for_status()
        data = resp.json()
        return SPSite(
            site_id=data["id"],
            display_name=data.get("displayName", ""),
            web_url=data.get("webUrl", self._config.site_url),
            protocol=SPProtocol.GRAPH,
        )

    def _resolve_site_rest(self) -> SPSite:
        """Use REST API to get site info."""
        url = f"{self._config.site_url}/_api/web"
        resp = self._session.get(url, headers={"Accept": "application/json;odata=verbose"})
        resp.raise_for_status()
        data = resp.json()["d"]
        return SPSite(
            site_id=data["Id"],
            display_name=data.get("Title", ""),
            web_url=data.get("Url", self._config.site_url),
            protocol=SPProtocol.REST,
        )
```

### Document Libraries

```python
    # --- Libraries ---

    def list_libraries(self, site: SPSite) -> list[SPLibrary]:
        if self._protocol == SPProtocol.GRAPH:
            return self._list_libraries_graph(site)
        return self._list_libraries_rest(site)

    def _list_libraries_graph(self, site: SPSite) -> list[SPLibrary]:
        url = f"https://graph.microsoft.com/v1.0/sites/{site.site_id}/drives"
        resp = self._session.get(url)
        resp.raise_for_status()
        libraries = []
        for drive in resp.json().get("value", []):
            if drive.get("driveType") != "documentLibrary":
                continue
            name = drive.get("name", "")
            if self._config.library_names and name not in self._config.library_names:
                continue
            libraries.append(SPLibrary(
                library_id=drive["id"],
                name=name,
                web_url=drive.get("webUrl", ""),
            ))
        return libraries

    def _list_libraries_rest(self, site: SPSite) -> list[SPLibrary]:
        url = f"{self._config.site_url}/_api/web/lists?$filter=BaseTemplate eq 101"
        resp = self._session.get(url, headers={"Accept": "application/json;odata=verbose"})
        resp.raise_for_status()
        libraries = []
        for lst in resp.json()["d"]["results"]:
            name = lst.get("Title", "")
            if self._config.library_names and name not in self._config.library_names:
                continue
            libraries.append(SPLibrary(
                library_id=lst["Id"],
                name=name,
                web_url=lst.get("RootFolder", {}).get("ServerRelativeUrl", ""),
            ))
        return libraries

    def list_files(self, site: SPSite, library: SPLibrary) -> list[SPFile]:
        if self._protocol == SPProtocol.GRAPH:
            return self._list_files_graph(site, library)
        return self._list_files_rest(library)

    def _list_files_graph(self, site: SPSite, library: SPLibrary) -> list[SPFile]:
        """Recursive file listing via Graph API."""
        base_url = f"https://graph.microsoft.com/v1.0/drives/{library.library_id}"

        if self._config.library_folder:
            folder_path = self._config.library_folder.strip("/")
            root_url = f"{base_url}/root:/{folder_path}:/children"
        else:
            root_url = f"{base_url}/root/children"

        files = []
        folders_to_scan = [root_url]

        while folders_to_scan and len(files) < self._config.max_files:
            url = folders_to_scan.pop(0)
            params = {"$top": 200}

            while True:
                resp = self._session.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()

                for item in data.get("value", []):
                    if "folder" in item:
                        if self._config.recursive:
                            folders_to_scan.append(
                                f"{base_url}/items/{item['id']}/children"
                            )
                        continue
                    if not self._matches_file_filters(item["name"], item.get("file", {}).get("mimeType", "")):
                        continue
                    mod_time = datetime.fromisoformat(item["lastModifiedDateTime"].rstrip("Z")).replace(tzinfo=timezone.utc)
                    if self._config.since and mod_time < datetime.fromisoformat(self._config.since).replace(tzinfo=timezone.utc):
                        continue
                    files.append(SPFile(
                        file_id=item["id"],
                        name=item["name"],
                        mime_type=item.get("file", {}).get("mimeType", "application/octet-stream"),
                        size=item.get("size"),
                        modified_time=mod_time,
                        path=item.get("parentReference", {}).get("path", "") + "/" + item["name"],
                        library_name=library.name,
                        web_url=item.get("webUrl"),
                    ))

                next_link = data.get("@odata.nextLink")
                if not next_link or len(files) >= self._config.max_files:
                    break
                url = next_link
                params = {}

        return files[:self._config.max_files]

    def _list_files_rest(self, library: SPLibrary) -> list[SPFile]:
        """File listing via SharePoint REST API."""
        folder_url = library.web_url
        if self._config.library_folder:
            folder_url = f"{folder_url}/{self._config.library_folder.strip('/')}"

        url = f"{self._config.site_url}/_api/web/GetFolderByServerRelativeUrl('{folder_url}')/Files"
        resp = self._session.get(url, headers={"Accept": "application/json;odata=verbose"})
        resp.raise_for_status()

        files = []
        for item in resp.json()["d"]["results"]:
            name = item.get("Name", "")
            if not self._matches_file_filters(name, ""):
                continue
            files.append(SPFile(
                file_id=item["UniqueId"],
                name=name,
                mime_type="",  # REST doesn't return MIME; infer from extension
                size=item.get("Length"),
                modified_time=datetime.fromisoformat(item["TimeLastModified"]),
                path=item.get("ServerRelativeUrl", ""),
                library_name=library.name,
                web_url=None,
            ))

        # Recurse into subfolders
        if self._config.recursive:
            subfolder_url = f"{self._config.site_url}/_api/web/GetFolderByServerRelativeUrl('{folder_url}')/Folders"
            resp = self._session.get(subfolder_url, headers={"Accept": "application/json;odata=verbose"})
            resp.raise_for_status()
            for folder in resp.json()["d"]["results"]:
                if folder.get("Name", "").startswith("_"):
                    continue  # skip system folders (_cts, _private, etc.)
                # Recursive call via REST folder enumeration
                files.extend(self._list_files_rest_folder(folder["ServerRelativeUrl"]))

        return files[:self._config.max_files]

    def download_file(self, site: SPSite, file: SPFile, library: SPLibrary) -> bytes:
        if self._protocol == SPProtocol.GRAPH:
            url = f"https://graph.microsoft.com/v1.0/drives/{library.library_id}/items/{file.file_id}/content"
            resp = self._session.get(url, follow_redirects=True)
        else:
            url = f"{self._config.site_url}/_api/web/GetFileByServerRelativeUrl('{file.path}')/$value"
            resp = self._session.get(url)
        resp.raise_for_status()
        return resp.content
```

### Lists (Structured Data → DuckDB Tables)

```python
    # --- Lists ---

    def list_lists(self, site: SPSite) -> list[SPList]:
        if self._protocol == SPProtocol.GRAPH:
            return self._list_lists_graph(site)
        return self._list_lists_rest()

    def _list_lists_graph(self, site: SPSite) -> list[SPList]:
        url = f"https://graph.microsoft.com/v1.0/sites/{site.site_id}/lists"
        params = {"$expand": "columns", "$top": 100}
        resp = self._session.get(url, params=params)
        resp.raise_for_status()
        lists = []
        for lst in resp.json().get("value", []):
            template = lst.get("list", {}).get("template", "")
            is_calendar = template == "events"
            is_hidden = lst.get("list", {}).get("hidden", False)
            name = lst.get("displayName", "")

            # Skip system/hidden lists unless explicitly named
            if is_hidden and (not self._config.list_names or name not in self._config.list_names):
                continue
            # Filter by name allowlist
            if self._config.list_names and name not in self._config.list_names:
                if not (is_calendar and self._config.calendar_names and name in self._config.calendar_names):
                    continue

            columns = []
            for col in lst.get("columns", []):
                if col.get("readOnly") and col.get("name") not in ("Title",):
                    continue  # skip computed/system columns
                col_type = self._map_column_type(col)
                columns.append(SPListColumn(
                    name=col.get("displayName", col.get("name", "")),
                    internal_name=col.get("name", ""),
                    column_type=col_type,
                    choices=col.get("choice", {}).get("choices") if col_type == "choice" else None,
                ))

            lists.append(SPList(
                list_id=lst["id"],
                name=name,
                item_count=lst.get("list", {}).get("contentTypesEnabled", 0),
                columns=columns,
                is_calendar=is_calendar,
                is_hidden=is_hidden,
            ))
        return lists

    def _list_lists_rest(self) -> list[SPList]:
        url = f"{self._config.site_url}/_api/web/lists?$filter=Hidden eq false"
        resp = self._session.get(url, headers={"Accept": "application/json;odata=verbose"})
        resp.raise_for_status()
        lists = []
        for lst in resp.json()["d"]["results"]:
            template_id = lst.get("BaseTemplate", 0)
            is_calendar = template_id == 106  # Events list template
            is_doc_lib = template_id == 101   # Document library — handled separately
            if is_doc_lib:
                continue

            name = lst.get("Title", "")
            if self._config.list_names and name not in self._config.list_names:
                if not (is_calendar and self._config.calendar_names and name in self._config.calendar_names):
                    continue

            # Fetch columns
            fields_url = f"{self._config.site_url}/_api/web/lists(guid'{lst['Id']}')/fields?$filter=Hidden eq false and ReadOnlyField eq false"
            fields_resp = self._session.get(fields_url, headers={"Accept": "application/json;odata=verbose"})
            fields_resp.raise_for_status()
            columns = []
            for f in fields_resp.json()["d"]["results"]:
                columns.append(SPListColumn(
                    name=f.get("Title", ""),
                    internal_name=f.get("InternalName", ""),
                    column_type=self._map_rest_field_type(f.get("TypeAsString", "")),
                    choices=f.get("Choices", {}).get("results") if f.get("TypeAsString") == "Choice" else None,
                ))

            lists.append(SPList(
                list_id=lst["Id"],
                name=name,
                item_count=lst.get("ItemCount", 0),
                columns=columns,
                is_calendar=is_calendar,
                is_hidden=False,
            ))
        return lists

    def fetch_list_items(self, site: SPSite, sp_list: SPList) -> list[SPListItem]:
        if self._protocol == SPProtocol.GRAPH:
            return self._fetch_items_graph(site, sp_list)
        return self._fetch_items_rest(sp_list)

    def _fetch_items_graph(self, site: SPSite, sp_list: SPList) -> list[SPListItem]:
        url = f"https://graph.microsoft.com/v1.0/sites/{site.site_id}/lists/{sp_list.list_id}/items"
        params = {"$expand": "fields", "$top": 200}
        items = []
        while True:
            resp = self._session.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            for item in data.get("value", []):
                items.append(SPListItem(
                    item_id=item["id"],
                    fields=item.get("fields", {}),
                ))
            next_link = data.get("@odata.nextLink")
            if not next_link or len(items) >= self._config.max_rows:
                break
            url = next_link
            params = {}
        return items[:self._config.max_rows]

    def _fetch_items_rest(self, sp_list: SPList) -> list[SPListItem]:
        url = f"{self._config.site_url}/_api/web/lists(guid'{sp_list.list_id}')/items"
        params = {"$top": 5000}  # REST API max page size
        items = []
        while True:
            resp = self._session.get(url, params=params, headers={"Accept": "application/json;odata=verbose"})
            resp.raise_for_status()
            data = resp.json()["d"]
            for item in data.get("results", []):
                items.append(SPListItem(
                    item_id=str(item["Id"]),
                    fields={col.internal_name: item.get(col.internal_name) for col in sp_list.columns},
                ))
            next_link = data.get("__next")
            if not next_link or len(items) >= self._config.max_rows:
                break
            url = next_link
            params = {}
        return items[:self._config.max_rows]
```

### Column Type Mapping

```python
    @staticmethod
    def _map_column_type(col: dict) -> str:
        """Map Graph API column definition to simplified type."""
        if col.get("number"):
            return "number"
        if col.get("dateTime"):
            return "datetime"
        if col.get("choice"):
            return "choice"
        if col.get("lookup"):
            return "lookup"
        if col.get("personOrGroup"):
            return "person"
        if col.get("boolean"):
            return "boolean"
        if col.get("currency"):
            return "currency"
        if col.get("hyperlinkOrPicture"):
            return "url"
        return "text"

    @staticmethod
    def _map_rest_field_type(type_str: str) -> str:
        """Map REST API TypeAsString to simplified type."""
        mapping = {
            "Text": "text", "Note": "text",
            "Number": "number", "Currency": "currency",
            "DateTime": "datetime",
            "Choice": "choice", "MultiChoice": "choice",
            "Lookup": "lookup", "LookupMulti": "lookup",
            "User": "person", "UserMulti": "person",
            "Boolean": "boolean", "YesNo": "boolean",
            "URL": "url",
            "Calculated": "text",
        }
        return mapping.get(type_str, "text")
```

### Lists → DuckDB Tables

The key differentiator: SharePoint lists become queryable DuckDB tables, not just vectorized text.

```python
import pyarrow as pa

def _list_to_arrow(sp_list: SPList, items: list[SPListItem]) -> pa.Table:
    """Convert SharePoint list items to an Arrow table."""
    column_map = {col.internal_name: col for col in sp_list.columns}
    arrays = {}

    for col in sp_list.columns:
        values = [item.fields.get(col.internal_name) for item in items]
        if col.column_type == "number":
            arrays[col.name] = pa.array(values, type=pa.float64())
        elif col.column_type == "currency":
            arrays[col.name] = pa.array(values, type=pa.float64())
        elif col.column_type == "datetime":
            arrays[col.name] = pa.array(
                [datetime.fromisoformat(v) if v else None for v in values],
                type=pa.timestamp("us", tz="UTC"),
            )
        elif col.column_type == "boolean":
            arrays[col.name] = pa.array(values, type=pa.bool_())
        elif col.column_type == "person":
            # Extract display name from person field
            arrays[col.name] = pa.array(
                [v.get("LookupValue", str(v)) if isinstance(v, dict) else str(v) if v else None for v in values],
                type=pa.string(),
            )
        elif col.column_type == "lookup":
            arrays[col.name] = pa.array(
                [v.get("LookupValue", str(v)) if isinstance(v, dict) else str(v) if v else None for v in values],
                type=pa.string(),
            )
        else:
            arrays[col.name] = pa.array([str(v) if v is not None else None for v in values], type=pa.string())

    return pa.table(arrays)
```

Registration with session DuckDB:

```python
# In _load_document() integration:
if self._config.list_as_table:
    arrow_table = _list_to_arrow(sp_list, items)
    table_name = _sanitize_table_name(f"sp_{source_name}_{sp_list.name}")
    self.datastore.store_table(table_name, arrow_table)
    # Register as a source table for schema discovery
    self.datastore.set_table_meta(table_name, {
        "source": "sharepoint",
        "list_name": sp_list.name,
        "site_url": self._config.site_url,
        "row_count": len(items),
        "columns": [{"name": c.name, "type": c.column_type} for c in sp_list.columns],
    })
```

### Calendars

SharePoint calendar lists are just lists with `BaseTemplate == 106` (or Graph template `events`). Reuse list item fetching but render as calendar events.

```python
def _calendar_items_to_events(sp_list: SPList, items: list[SPListItem], source_name: str) -> list[CalendarEvent]:
    """Convert SharePoint calendar list items to CalendarEvent dataclass."""
    from constat.discovery.doc_tools._calendar import CalendarEvent, EventAttachment

    events = []
    for item in items:
        f = item.fields
        start = f.get("EventDate") or f.get("StartDate")
        end = f.get("EndDate")
        if not start:
            continue

        event_id = _make_event_id(f, datetime.fromisoformat(start))
        events.append(CalendarEvent(
            event_id=event_id,
            title=f.get("Title", "(No title)"),
            start=datetime.fromisoformat(start),
            end=datetime.fromisoformat(end) if end else datetime.fromisoformat(start),
            all_day=bool(f.get("fAllDayEvent", False)),
            location=f.get("Location"),
            organizer=_resolve_person(f.get("Author")),
            attendees=[],  # SP calendar lists don't have attendee fields by default
            status="confirmed",
            description=f.get("Description"),
            recurrence_id=f.get("RecurrenceID"),
            attachments=[],  # list item attachments handled separately
            html_link=None,
        ))
    return events
```

Calendar events are rendered as documents using the same `_render_event()` from the calendar spec.

### Site Pages

```python
    def list_pages(self, site: SPSite) -> list[SPPage]:
        if self._protocol == SPProtocol.GRAPH:
            return self._list_pages_graph(site)
        return self._list_pages_rest()

    def _list_pages_graph(self, site: SPSite) -> list[SPPage]:
        url = f"https://graph.microsoft.com/v1.0/sites/{site.site_id}/pages"
        resp = self._session.get(url)
        resp.raise_for_status()
        pages = []
        for item in resp.json().get("value", []):
            page_type = "modern"  # Graph only returns modern pages
            if self._config.page_types and page_type not in self._config.page_types:
                continue
            pages.append(SPPage(
                page_id=item["id"],
                title=item.get("title", ""),
                page_type=page_type,
                content_html=item.get("webParts", ""),  # need separate content fetch
                modified_time=datetime.fromisoformat(item["lastModifiedDateTime"].rstrip("Z")).replace(tzinfo=timezone.utc),
                web_url=item.get("webUrl", ""),
            ))
        return pages

    def _list_pages_rest(self) -> list[SPPage]:
        """Fetch pages via REST API (supports classic + wiki pages)."""
        url = f"{self._config.site_url}/_api/web/lists/getByTitle('Site Pages')/items"
        params = {"$select": "Id,Title,WikiField,CanvasContent1,Modified,FileRef"}
        resp = self._session.get(url, params=params, headers={"Accept": "application/json;odata=verbose"})
        resp.raise_for_status()
        pages = []
        for item in resp.json()["d"]["results"]:
            # Detect page type from available content fields
            if item.get("CanvasContent1"):
                page_type = "modern"
                content = item["CanvasContent1"]
            elif item.get("WikiField"):
                page_type = "wiki"
                content = item["WikiField"]
            else:
                page_type = "classic"
                content = ""

            if self._config.page_types and page_type not in self._config.page_types:
                continue

            pages.append(SPPage(
                page_id=str(item["Id"]),
                title=item.get("Title", ""),
                page_type=page_type,
                content_html=content,
                modified_time=datetime.fromisoformat(item["Modified"]),
                web_url=item.get("FileRef", ""),
            ))
        return pages
```

### Page Content Rendering

```python
def _render_page(page: SPPage, source_name: str) -> str:
    """Render SharePoint page as markdown for vectorization."""
    from constat.discovery.doc_tools._html import html_to_markdown  # existing util

    parts = [f"# {page.title}"]
    parts.append(f"Type: {page.page_type} page")
    parts.append("")

    if page.content_html:
        md = html_to_markdown(page.content_html)
        parts.append(md)

    return "\n".join(parts)
```

### Integration with `_core.py`

In `_load_document(name)`:

```python
if doc_config.type == "sharepoint":
    client = SharePointClient(doc_config, config_dir=self._config_dir)
    site = client.resolve_site()

    # 1. Document libraries
    if doc_config.discover_libraries:
        for library in client.list_libraries(site):
            for file in client.list_files(site, library):
                file_name = f"{name}:lib:{_make_file_id(file)}"
                if self._is_file_indexed(file_name, file.modified_time):
                    continue
                data = client.download_file(site, file, library)
                doc_type = detect_type_from_source(file.mime_type, file.name)
                content = _extract_content_from_bytes(data, doc_type)
                self._loaded_documents[file_name] = LoadedDocument(
                    name=file_name, content=content, doc_type=doc_type,
                )

    # 2. Lists
    if doc_config.discover_lists:
        for sp_list in client.list_lists(site):
            if sp_list.is_calendar:
                continue  # handled below
            items = client.fetch_list_items(site, sp_list)
            list_doc_name = f"{name}:list:{_sanitize_name(sp_list.name)}"

            if doc_config.list_as_table:
                arrow_table = _list_to_arrow(sp_list, items)
                table_name = _sanitize_table_name(f"sp_{name}_{sp_list.name}")
                self.datastore.store_table(table_name, arrow_table)

            if doc_config.list_as_document:
                content = _render_list_as_markdown(sp_list, items)
                self._loaded_documents[list_doc_name] = LoadedDocument(
                    name=list_doc_name, content=content, doc_type="markdown",
                )

    # 3. Calendars
    if doc_config.discover_calendars:
        calendar_lists = [l for l in client.list_lists(site) if l.is_calendar]
        if doc_config.calendar_names:
            calendar_lists = [l for l in calendar_lists if l.name in doc_config.calendar_names]
        for cal_list in calendar_lists:
            items = client.fetch_list_items(site, cal_list)
            events = _calendar_items_to_events(cal_list, items, name)
            for event in events:
                event_name = f"{name}:cal:{event.event_id}"
                body = _render_event(event, name)
                self._loaded_documents[event_name] = LoadedDocument(
                    name=event_name, content=body, doc_type="markdown",
                )

    # 4. Pages
    if doc_config.discover_pages:
        for page in client.list_pages(site):
            page_name = f"{name}:page:{_make_page_id(page)}"
            content = _render_page(page, name)
            self._loaded_documents[page_name] = LoadedDocument(
                name=page_name, content=content, doc_type="markdown",
            )
```

### Transport Layer (`_transport.py`)

```python
if config.type == "sharepoint":
    return "sharepoint"
```

### Source Refresher (`source_refresher.py`)

Add to `_classify_source()`:

```python
doc_type = doc_config.get("type", "")
if doc_type == "sharepoint":
    return "sharepoint"
```

Add `_refresh_sharepoint_source()` following the existing IMAP pattern:
- Libraries: re-list files, compare `modified_time`, re-index changed files
- Lists: re-fetch all rows, replace DuckDB table (full refresh — lists are small)
- Calendars: re-fetch events in time window, re-index changed events
- Pages: re-fetch pages, compare `modified_time`, re-index changed pages

## Authentication

### Modern SharePoint Online (Graph API)

Reuses existing `AzureOAuth2Provider` from `_imap.py`:

```python
def _get_access_token(self) -> str:
    from constat.discovery.doc_tools._imap import AzureOAuth2Provider, RefreshTokenOAuth2Provider

    if self._config.auth_type == "oauth2_refresh":
        return RefreshTokenOAuth2Provider(self._config).get_access_token()
    return AzureOAuth2Provider(self._config).get_access_token()
```

Required Azure AD permissions:
- `Sites.Read.All` — read site metadata, lists, list items
- `Files.Read.All` — read document library files
- `Pages.Read.All` — read site pages (beta, may require beta endpoint)

### Legacy SharePoint On-Premises (NTLM)

```python
def _build_ntlm_session(self) -> httpx.Client:
    """Build NTLM-authenticated session for on-prem SharePoint."""
    try:
        import httpx_ntlm
    except ImportError:
        raise ImportError("httpx-ntlm required for SharePoint NTLM auth: pip install httpx-ntlm")

    auth = httpx_ntlm.HttpNtlmAuth(self._config.username, self._config.password)
    return httpx.Client(auth=auth)
```

For on-prem SharePoint with ADFS or modern auth:
```yaml
auth_type: oauth2
oauth2_client_id: ${SP_CLIENT_ID}
oauth2_client_secret: ${SP_CLIENT_SECRET}
oauth2_tenant_id: ${SP_TENANT_ID}
oauth2_scopes:
  - "https://sp.company.local/.default"
```

## Incremental Sync

### Libraries
Same as cloud-drive spec: compare `lastModifiedDateTime` per file. Skip unchanged files.

### Lists
Full table replace on each sync. Lists are typically small (<10K rows). The `store_table()` call overwrites the existing DuckDB table atomically.

### Calendars
Filter by `Modified` field: fetch only items where `Modified > last_synced`.

### Pages
Compare `Modified` field per page. Re-index changed pages, remove deleted pages.

### Change tracking (future optimization)
Both Graph API and REST API support delta queries / change tokens:
- Graph: `GET /sites/{id}/lists/{id}/items/delta`
- REST: `GetListItemChangesSinceToken`

Not in v1 — full re-list with modification time comparison is sufficient for the expected scale.

## New Dependencies

| Package | Purpose | Install |
|---------|---------|---------|
| `httpx` | HTTP client | Already installed |
| `msal` | Azure OAuth2 | Optional, already used by IMAP |
| `httpx-ntlm` | NTLM auth for on-prem | Optional, `pip install httpx-ntlm` |
| `pyarrow` | Arrow tables for list→DuckDB | Already installed |

## File Changes Summary

| File | Change |
|------|--------|
| `constat/discovery/doc_tools/_sharepoint.py` | **New**: `SharePointClient`, `SPSite`, `SPLibrary`, `SPFile`, `SPList`, `SPListColumn`, `SPListItem`, `SPPage`, `_list_to_arrow()`, `_render_page()`, `_calendar_items_to_events()` |
| `constat/discovery/doc_tools/_core.py` | Add `sharepoint` branch in `_load_document()` |
| `constat/discovery/doc_tools/_transport.py` | Recognize `sharepoint` type |
| `constat/core/config.py` | Add SharePoint fields to `DocumentConfig` |
| `constat/server/source_refresher.py` | Add `sharepoint` to `_classify_source()` + `_refresh_sharepoint_source()` |
| `tests/test_sharepoint_ingestion.py` | Unit tests with mocked API responses |

## Testing Strategy

1. **Unit tests** (`test_sharepoint_ingestion.py`):
   - **Protocol detection**: `.sharepoint.com` → graph, other → rest
   - **Site resolution**: mock Graph and REST site endpoints
   - **Libraries**: list drives/lists with BaseTemplate filter, file listing with pagination
   - **Lists**: fetch columns + items via both protocols, column type mapping
   - **Arrow conversion**: `_list_to_arrow()` with mixed types (text, number, datetime, person, lookup, boolean)
   - **Calendars**: calendar list detection, item-to-CalendarEvent conversion
   - **Pages**: modern page (CanvasContent1), wiki page (WikiField), classic page
   - **Auth**: OAuth2 token acquisition, NTLM session setup
   - **File ID generation**: deterministic, unique
   - **Filter matching**: include_types, exclude_patterns, library_names, list_names

2. **Integration tests**:
   - Full pipeline: resolve site → discover libraries + lists → download files → index
   - List → DuckDB: verify table registered, queryable via SQL, correct column types
   - Calendar → documents: verify event rendering and addressing
   - Incremental: second run skips unchanged files, replaces list tables
   - Mixed: site with 2 libraries, 3 lists, 1 calendar → all indexed correctly

3. **Fixtures**: JSON response dicts matching both Graph and REST API schemas:
   - Graph: site response, drives list, drive items (folder + files), list with columns + items
   - REST: `_api/web` response, lists with BaseTemplate, folder/files, list items with `__next` pagination

## Edge Cases

| Case | Handling |
|------|----------|
| Site with 50+ libraries | Respect `library_names` allowlist; default includes all |
| System/hidden lists (e.g., "Style Library", "Composed Looks") | Filter by `Hidden eq false` (REST) or `hidden: false` (Graph); skip unless explicitly named |
| Large lists (>5000 items) | REST API has 5000 item threshold (list view threshold). Use `$skiptoken` pagination or indexed column filter. Graph API handles this via `@odata.nextLink`. |
| Lookup columns with IDs | Resolve to display value; store as string |
| Person columns | Extract `LookupValue` (display name), not `LookupId` |
| Multi-value columns (MultiChoice, LookupMulti, UserMulti) | Join values with `; ` separator for DuckDB string column |
| Calculated columns | Fetch computed value; map to text |
| Managed metadata (taxonomy) columns | Extract term label(s); store as text |
| Document libraries with versioning | Index latest version only; skip minor versions |
| Check-out locked files | Download latest published version; log if file is checked out |
| Subsites (legacy pattern) | Not traversed in v1; each subsite requires separate config entry |
| SharePoint Online throttling (429) | Respect `Retry-After` header; exponential backoff |
| REST API `__next` pagination | Follow `__next` URL until exhausted or `max_rows` reached |
| On-prem NTLM with Kerberos fallback | Not supported in v1; NTLM only |
| Files with special characters in names | URL-encode paths for REST API; Graph API uses ID-based access |
| Empty lists (0 items) | Register DuckDB table with schema only (0 rows) |
| Mixed page types on same site | Filter via `page_types` config; default indexes all |

## Legacy REST API Reference

| Resource | REST Endpoint | Graph Equivalent |
|----------|--------------|------------------|
| Site info | `_api/web` | `/sites/{hostname}:{path}` |
| Libraries | `_api/web/lists?$filter=BaseTemplate eq 101` | `/sites/{id}/drives` |
| Files in folder | `_api/web/GetFolderByServerRelativeUrl('{path}')/Files` | `/drives/{id}/root/children` |
| Download file | `_api/web/GetFileByServerRelativeUrl('{path}')/$value` | `/drives/{id}/items/{id}/content` |
| Lists | `_api/web/lists?$filter=Hidden eq false` | `/sites/{id}/lists` |
| List fields | `_api/web/lists(guid'{id}')/fields` | `/sites/{id}/lists/{id}/columns` |
| List items | `_api/web/lists(guid'{id}')/items` | `/sites/{id}/lists/{id}/items?$expand=fields` |
| Pages | `_api/web/lists/getByTitle('Site Pages')/items` | `/sites/{id}/pages` |

## Non-Goals (v1)

- SharePoint search (using `/_api/search/query`) — we index locally
- Subsites / site collections traversal — each site is a separate config entry
- Content types and content type hubs
- SharePoint workflows / Power Automate integration
- Write operations (creating lists, uploading files)
- SharePoint webhooks / real-time change notifications
- InfoPath forms
- SharePoint Designer workflows (legacy)
- Kerberos / claims-based auth (on-prem) — NTLM and OAuth2 only
- BCS (Business Connectivity Services) external content types
- Term store / managed metadata service administration
- SharePoint Framework (SPFx) web part rendering
