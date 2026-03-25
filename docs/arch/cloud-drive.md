# Cloud Drive Document Source (Google Drive / OneDrive)

## Goal

Add cloud document libraries (Google Drive, Microsoft OneDrive/SharePoint) as a document data source. Files in specified folders become documents. The pipeline reuses existing format extractors (PDF, DOCX, XLSX, PPTX, images, markdown, HTML) and routes everything through the standard vectorization pipeline.

## MCP-First Strategy

**Default approach**: Use existing MCP servers for Google Drive and OneDrive rather than building custom integrations. The [MCP ecosystem](https://modelcontextprotocol.io/) includes drive connectors that expose files as MCP resources.

### MCP config (preferred)

```yaml
documents:
  shared-drive:
    type: mcp
    url: https://mcp-gdrive.example.com/sse    # Google Drive MCP server
    auth:
      method: bearer
      token_ref: gdrive-mcp-token
    resource_filter: "gdrive://folder/Analytics/*"
    description: "Analytics shared drive"

  onedrive-docs:
    type: mcp
    url: https://mcp-onedrive.example.com/sse   # OneDrive MCP server
    auth:
      method: bearer
      token_ref: onedrive-mcp-token
    description: "OneDrive documentation folder"
```

MCP resources flow through the standard pipeline: `resources/read` → content bytes → extract → chunk → embed → DuckDB vector store. See `mcp.md` for full MCP client architecture.

### When to fall back to custom `DriveFetcher`

Build the custom implementation below only if MCP servers cannot handle:

| Gap | Why MCP may fall short | Custom solution |
|---|---|---|
| Google Docs native export | Google Docs/Sheets/Slides aren't files — need export API (Docs→DOCX) | `_download_google()` with export MIME mapping |
| Incremental sync | MCP `resources/list` has no `modifiedTime` filter | `modifiedTime > timestamp` in Drive API query |
| Deep folder traversal | MCP server may flatten or limit depth | Recursive `folders_to_scan` with BFS |
| File type filtering | MCP server may not support extension/MIME filtering | `_matches_filters()` with allowlist + regex |
| Large drives (10K+ files) | MCP pagination may be slow or incomplete | Direct API with `pageSize=1000` + cursor |

**Decision process**: Try MCP first. If the MCP server handles native format export and folder traversal, skip everything below. If not, implement only the missing pieces as a custom provider that wraps the MCP server or calls the API directly.

---

The rest of this document specifies the custom fallback implementation.

## Addressing Scheme

```
<data_source>:<file_id>                            # file content
<data_source>:<file_id>:<child>                    # embedded image from file
```

Examples:
```
shared-drive:f_abc123_quarterly_report              # PDF in Drive
shared-drive:f_abc123_quarterly_report:page_2_img_1.png  # image extracted from PDF
onedrive-docs:f_def456_onboarding_guide             # DOCX in OneDrive
```

File IDs are derived from the provider's native file ID + filename slug for readability:
```python
f"f_{provider_id[:8]}_{slug}"
# f_abc12345_quarterly_report
```

## Configuration

### In domain YAML or root config:

```yaml
documents:
  shared-drive:
    type: drive
    provider: google                     # "google" | "microsoft"
    description: "Analytics shared drive"

    # OAuth2 credentials (reuses existing oauth2_* fields)
    oauth2_client_id: ${GOOGLE_CLIENT_ID}
    oauth2_client_secret: ${GOOGLE_CLIENT_SECRET}
    oauth2_scopes:
      - "https://www.googleapis.com/auth/drive.readonly"

    # Drive-specific options
    folder_id: "1aBcDeFgHiJkLmNoPqRsTuVwXyZ"   # root folder to sync
    folder_path: "/Shared Drives/Analytics"       # alternative: path-based (Google only)
    recursive: true                               # traverse subfolders (default: true)
    max_files: 500                                # cap per sync
    include_types:                                # allowlist; omit = all supported types
      - ".pdf"
      - ".docx"
      - ".xlsx"
      - ".pptx"
      - ".md"
      - ".txt"
      - ".html"
      - ".png"
      - ".jpg"
    exclude_patterns:                             # regex patterns for filenames to skip
      - "^~\\$"                                   # Office temp files
      - "^\\."                                    # hidden files
    since: "2026-01-01"                           # only files modified after this date
    include_trashed: false                        # include trashed files (default: false)

  onedrive-docs:
    type: drive
    provider: microsoft
    description: "OneDrive documentation folder"

    oauth2_client_id: ${AZURE_CLIENT_ID}
    oauth2_client_secret: ${AZURE_CLIENT_SECRET}
    oauth2_tenant_id: ${AZURE_TENANT_ID}
    oauth2_scopes:
      - "https://graph.microsoft.com/Files.Read.All"

    folder_path: "/Documents/Analytics"           # path relative to drive root
    site_id: "contoso.sharepoint.com,guid,guid"   # SharePoint site (optional)
    drive_id: "b!aBcDeFgHi..."                    # specific drive (optional; default: user's OneDrive)
    recursive: true
    max_files: 500
```

### `DocumentConfig` additions

```python
class DocumentConfig(BaseModel):
    ...
    # Drive fields
    provider: Optional[str] = None            # "google" | "microsoft" (shared with calendar)
    folder_id: Optional[str] = None           # provider-specific folder ID
    folder_path: Optional[str] = None         # human-readable path
    recursive: bool = True                    # traverse subfolders
    max_files: int = 500
    include_types: Optional[list[str]] = None # file extension allowlist
    exclude_patterns: Optional[list[str]] = None  # already exists for link following
    include_trashed: bool = False
    site_id: Optional[str] = None             # SharePoint site ID
    drive_id: Optional[str] = None            # specific drive ID
```

Reuses existing fields: `since`, `extract_images`, `extract_tables`, `oauth2_*`, `description`, `tags`.

## Pipeline Overview

```
OAuth2 authentication
  → resolve folder (ID or path → folder reference)
  → list files (paginated, filtered by type/date/name)
  → for each file:
      ├─ check if already indexed (skip if unchanged)
      ├─ download file bytes (or export for Google Docs native formats)
      ├─ detect type via _mime.detect_type_from_source()
      ├─ route to existing extractor:
      │     ├─ PDF → _extract_pdf_text() + optional _extract_pdf_images()
      │     ├─ DOCX/PPTX/XLSX → existing Office extractors
      │     ├─ image → _extract_image() → image pipeline
      │     ├─ markdown/text/HTML → direct content
      │     └─ unsupported → skip with warning
      ├─ _chunk_document → encode → add_chunks    (existing pipeline)
      │
      └─ if extract_images and doc type supports it:
            └─ extract embedded images → image pipeline
```

## Implementation

### New file: `constat/discovery/doc_tools/_drive.py`

```python
from dataclasses import dataclass
from datetime import datetime, timezone

@dataclass
class DriveFile:
    file_id: str                         # provider native ID
    name: str                            # filename
    mime_type: str
    size: int | None
    modified_time: datetime
    path: str                            # full path from root folder
    parent_id: str | None
    web_url: str | None                  # link back to cloud UI
    is_google_native: bool = False       # Google Docs/Sheets/Slides (need export)

class DriveFetcher:
    """Fetches files from Google Drive or Microsoft OneDrive/SharePoint."""

    def __init__(self, config: DocumentConfig, config_dir: Path | None = None):
        self._config = config
        self._config_dir = config_dir

    def list_files(self) -> list[DriveFile]:
        if self._config.provider == "google":
            return self._list_google()
        elif self._config.provider == "microsoft":
            return self._list_microsoft()
        raise ValueError(f"Unknown drive provider: {self._config.provider}")

    def download_file(self, file: DriveFile) -> bytes:
        if self._config.provider == "google":
            return self._download_google(file)
        elif self._config.provider == "microsoft":
            return self._download_microsoft(file)
        raise ValueError(f"Unknown drive provider: {self._config.provider}")
```

### Google Drive (REST API v3)

```python
    def _resolve_google_folder(self) -> str:
        """Resolve folder_id or folder_path to a Google Drive folder ID."""
        if self._config.folder_id:
            return self._config.folder_id
        if self._config.folder_path:
            return self._resolve_google_path(self._config.folder_path)
        raise ValueError("Either folder_id or folder_path required for Google Drive")

    def _list_google(self) -> list[DriveFile]:
        token = self._get_access_token()
        root_id = self._resolve_google_folder()
        headers = {"Authorization": f"Bearer {token}"}
        files = []

        folders_to_scan = [root_id]
        while folders_to_scan and len(files) < self._config.max_files:
            folder_id = folders_to_scan.pop(0)
            query_parts = [f"'{folder_id}' in parents"]
            if not self._config.include_trashed:
                query_parts.append("trashed = false")
            if self._config.since:
                query_parts.append(f"modifiedTime > '{self._config.since}T00:00:00Z'")
            query = " and ".join(query_parts)

            params = {
                "q": query,
                "fields": "nextPageToken,files(id,name,mimeType,size,modifiedTime,parents,webViewLink)",
                "pageSize": 1000,
            }
            url = "https://www.googleapis.com/drive/v3/files"

            while True:
                resp = httpx.get(url, params=params, headers=headers)
                resp.raise_for_status()
                data = resp.json()

                for item in data.get("files", []):
                    if item["mimeType"] == "application/vnd.google-apps.folder":
                        if self._config.recursive:
                            folders_to_scan.append(item["id"])
                        continue

                    if not self._matches_filters(item["name"], item["mimeType"]):
                        continue

                    files.append(DriveFile(
                        file_id=item["id"],
                        name=item["name"],
                        mime_type=item["mimeType"],
                        size=int(item.get("size", 0)),
                        modified_time=datetime.fromisoformat(item["modifiedTime"].rstrip("Z")).replace(tzinfo=timezone.utc),
                        path=item["name"],  # simplified; full path requires parent traversal
                        parent_id=folder_id,
                        web_url=item.get("webViewLink"),
                        is_google_native=item["mimeType"].startswith("application/vnd.google-apps."),
                    ))

                page_token = data.get("nextPageToken")
                if not page_token or len(files) >= self._config.max_files:
                    break
                params["pageToken"] = page_token

        return files[:self._config.max_files]

    def _download_google(self, file: DriveFile) -> bytes:
        """Download or export a Google Drive file."""
        token = self._get_access_token()
        headers = {"Authorization": f"Bearer {token}"}

        if file.is_google_native:
            # Export Google Docs/Sheets/Slides to Office format
            export_map = {
                "application/vnd.google-apps.document": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "application/vnd.google-apps.spreadsheet": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "application/vnd.google-apps.presentation": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            }
            export_mime = export_map.get(file.mime_type)
            if not export_mime:
                raise ValueError(f"Cannot export Google native format: {file.mime_type}")
            url = f"https://www.googleapis.com/drive/v3/files/{file.file_id}/export"
            resp = httpx.get(url, params={"mimeType": export_mime}, headers=headers)
        else:
            url = f"https://www.googleapis.com/drive/v3/files/{file.file_id}"
            resp = httpx.get(url, params={"alt": "media"}, headers=headers)

        resp.raise_for_status()
        return resp.content
```

### Microsoft OneDrive / SharePoint (Graph API)

```python
    def _resolve_microsoft_base_url(self) -> str:
        """Build the base Graph API URL for the target drive."""
        if self._config.site_id:
            return f"https://graph.microsoft.com/v1.0/sites/{self._config.site_id}/drive"
        if self._config.drive_id:
            return f"https://graph.microsoft.com/v1.0/drives/{self._config.drive_id}"
        return "https://graph.microsoft.com/v1.0/me/drive"

    def _list_microsoft(self) -> list[DriveFile]:
        token = self._get_access_token()
        headers = {"Authorization": f"Bearer {token}"}
        base_url = self._resolve_microsoft_base_url()
        files = []

        # Resolve starting folder
        if self._config.folder_path:
            folder_url = f"{base_url}/root:/{self._config.folder_path.strip('/')}:/children"
        elif self._config.folder_id:
            folder_url = f"{base_url}/items/{self._config.folder_id}/children"
        else:
            folder_url = f"{base_url}/root/children"

        folders_to_scan = [folder_url]
        while folders_to_scan and len(files) < self._config.max_files:
            url = folders_to_scan.pop(0)
            params = {"$top": 200}

            while True:
                resp = httpx.get(url, params=params, headers=headers)
                resp.raise_for_status()
                data = resp.json()

                for item in data.get("value", []):
                    if "folder" in item:
                        if self._config.recursive:
                            child_url = f"{base_url}/items/{item['id']}/children"
                            folders_to_scan.append(child_url)
                        continue

                    if not self._matches_filters(item["name"], item.get("file", {}).get("mimeType", "")):
                        continue

                    mod_time = datetime.fromisoformat(item["lastModifiedDateTime"].rstrip("Z")).replace(tzinfo=timezone.utc)
                    if self._config.since and mod_time < datetime.fromisoformat(self._config.since).replace(tzinfo=timezone.utc):
                        continue

                    files.append(DriveFile(
                        file_id=item["id"],
                        name=item["name"],
                        mime_type=item.get("file", {}).get("mimeType", "application/octet-stream"),
                        size=item.get("size"),
                        modified_time=mod_time,
                        path=item.get("parentReference", {}).get("path", "") + "/" + item["name"],
                        parent_id=item.get("parentReference", {}).get("id"),
                        web_url=item.get("webUrl"),
                    ))

                next_link = data.get("@odata.nextLink")
                if not next_link or len(files) >= self._config.max_files:
                    break
                url = next_link
                params = {}

        return files[:self._config.max_files]

    def _download_microsoft(self, file: DriveFile) -> bytes:
        """Download a file from OneDrive/SharePoint."""
        token = self._get_access_token()
        base_url = self._resolve_microsoft_base_url()
        url = f"{base_url}/items/{file.file_id}/content"
        resp = httpx.get(url, headers={"Authorization": f"Bearer {token}"}, follow_redirects=True)
        resp.raise_for_status()
        return resp.content
```

### File ID Generation

```python
    def _make_file_id(self, file: DriveFile) -> str:
        """Deterministic short ID from provider file ID + name."""
        short_hash = hashlib.sha256(file.file_id.encode()).hexdigest()[:8]
        slug = re.sub(r"[^a-z0-9]", "_", Path(file.name).stem.lower())[:30]
        return f"f_{short_hash}_{slug}"
```

### Filter Matching

```python
    def _matches_filters(self, filename: str, mime_type: str) -> bool:
        """Check if a file passes include_types and exclude_patterns filters."""
        if self._config.include_types:
            ext = Path(filename).suffix.lower()
            # Also match Google native formats by mime type
            native_exts = {
                "application/vnd.google-apps.document": ".docx",
                "application/vnd.google-apps.spreadsheet": ".xlsx",
                "application/vnd.google-apps.presentation": ".pptx",
            }
            effective_ext = native_exts.get(mime_type, ext)
            if effective_ext not in self._config.include_types:
                return False

        if self._config.exclude_patterns:
            import re
            for pattern in self._config.exclude_patterns:
                if re.search(pattern, filename):
                    return False
        return True
```

### Integration with `_core.py`

In `_load_document(name)`:

```python
if doc_config.type == "drive":
    fetcher = DriveFetcher(doc_config, config_dir=self._config_dir)
    files = fetcher.list_files()
    for file in files:
        file_name = f"{name}:{fetcher._make_file_id(file)}"

        # Skip if already indexed and unchanged
        if self._is_file_indexed(file_name, file.modified_time):
            continue

        data = fetcher.download_file(file)
        doc_type = detect_type_from_source(file.mime_type, file.name)

        # Google native formats are exported as Office formats
        if file.is_google_native:
            export_types = {
                "application/vnd.google-apps.document": "docx",
                "application/vnd.google-apps.spreadsheet": "xlsx",
                "application/vnd.google-apps.presentation": "pptx",
            }
            doc_type = export_types.get(file.mime_type, doc_type)

        if doc_type.startswith("image/"):
            image_result = _extract_image(path=None, data=data)
            if image_result.category == "image-primary":
                desc = await self._describe_image(self._provider, data, doc_type)
                image_result.description = desc.description
                image_result.subcategory = desc.subcategory
                image_result.labels = desc.labels
            content = _render_image_result(image_result, file.name)
        else:
            content = _extract_content_from_bytes(data, doc_type)

        self._loaded_documents[file_name] = LoadedDocument(
            name=file_name, content=content, doc_type=doc_type,
        )

        # Extract embedded images if enabled
        if doc_config.extract_images and not doc_type.startswith("image/"):
            embedded = _extract_images_from_document(None, data, doc_type)
            for img in embedded:
                img_name = f"{file_name}:{img.name}"
                img_result = _extract_image(path=None, data=img.data)
                if img_result.category == "image-primary":
                    desc = await self._describe_image(self._provider, img.data, img.mime_type)
                    img_result.description = desc.description
                img_content = _render_image_result(img_result, img.name)
                self._loaded_documents[img_name] = LoadedDocument(
                    name=img_name, content=img_content, doc_type="markdown",
                )
```

### Transport Layer (`_transport.py`)

Add `drive` transport recognition:

```python
if config.type == "drive":
    return "drive"
```

Drive fetching is handled entirely by `DriveFetcher`, not the generic `fetch_document()` path, because drive APIs require folder traversal and provider-specific pagination.

### OAuth2 Authentication

Reuses the existing OAuth2 infrastructure from the IMAP/Calendar implementations:

- Google Drive: `https://www.googleapis.com/auth/drive.readonly`
- Microsoft OneDrive: `Files.Read.All`
- SharePoint: `Sites.Read.All` + `Files.Read.All`

## Incremental Sync

Track file modification times to avoid re-downloading unchanged files.

```python
def _is_file_indexed(self, file_name: str, modified_time: datetime) -> bool:
    """Check if file is already indexed and hasn't changed."""
    existing_meta = self._vector_store.get_document_meta(file_name)
    if not existing_meta:
        return False
    stored_modified = existing_meta.get("modified_time")
    return stored_modified and datetime.fromisoformat(stored_modified) >= modified_time
```

Strategy:
1. First sync: list all files in folder tree, download and index everything.
2. Subsequent syncs: list files again but skip downloads for files with unchanged `modifiedTime`.
3. For changed files: delete existing chunks, re-download, re-index.
4. Deleted files: if a previously indexed file is missing from the listing, remove its chunks.

Google Drive supports `modifiedTime > 'timestamp'` in the query filter. Microsoft Graph supports `$filter=lastModifiedDateTime ge timestamp`. Both reduce the listing API calls on incremental sync.

The existing `source_refresher.py` handles the refresh scheduling.

## Handling Google Native Formats

Google Docs, Sheets, and Slides have no downloadable bytes — they must be exported:

| Google MIME | Export As | Resulting Type |
|-------------|-----------|----------------|
| `application/vnd.google-apps.document` | DOCX | `docx` |
| `application/vnd.google-apps.spreadsheet` | XLSX | `xlsx` |
| `application/vnd.google-apps.presentation` | PPTX | `pptx` |
| `application/vnd.google-apps.drawing` | PNG | `image/png` |
| `application/vnd.google-apps.form` | — | Skip (not exportable to useful format) |

The export uses the Drive API v3 export endpoint with `mimeType` parameter.

## New Dependencies

| Package | Purpose | Install |
|---------|---------|---------|
| `httpx` | HTTP client for REST APIs | Already installed |
| `msal` | Microsoft OAuth2 (if using Microsoft) | Optional, `pip install msal` |
| `google-auth-oauthlib` | Google OAuth2 (if using Google) | Optional, `pip install google-auth-oauthlib` |

Same optional OAuth2 dependencies as IMAP and Calendar implementations.

## File Changes Summary

| File | Change |
|------|--------|
| `constat/discovery/doc_tools/_drive.py` | **New**: `DriveFetcher`, `DriveFile`, `_make_file_id()` |
| `constat/discovery/doc_tools/_core.py` | Add drive branch in `_load_document()` |
| `constat/discovery/doc_tools/_transport.py` | Recognize `drive` type |
| `constat/core/config.py` | Add drive fields to `DocumentConfig` |
| `tests/test_drive_ingestion.py` | Unit tests with mocked API responses |

## Testing Strategy

1. **Unit tests** (`test_drive_ingestion.py`):
   - Mock `httpx.get` with canned Google Drive API responses (file list, download, export)
   - Mock `httpx.get` with canned Microsoft Graph API responses (children listing, download)
   - `_list_google()` with flat folder, nested subfolders, mixed file types
   - `_list_microsoft()` with SharePoint site, OneDrive path-based folder
   - Filter matching: include_types allowlist, exclude_patterns regex
   - Google native format export: verify export MIME selection and download path
   - File ID generation: deterministic, unique per file
   - Incremental sync: file with same modifiedTime skipped, changed file re-indexed

2. **Integration tests**:
   - Full pipeline mock: list files → download → extract → index
   - Verify chunks in vector store with correct `document_name` addresses
   - Nested folder: `shared-drive:f_abc_report` from subfolder indexed correctly
   - Google Docs export: native doc → DOCX export → text extraction → chunks
   - Mixed content: PDF + image + DOCX in same folder → all indexed with correct types

3. **Fixtures**: JSON response dicts matching API schemas:
   - Google: folder with 3 files (PDF, Google Doc, PNG), subfolder with 1 PPTX
   - Microsoft: OneDrive folder with 2 files, SharePoint document library with 3 files

## Edge Cases

| Case | Handling |
|------|----------|
| Google Shared Drives (Team Drives) | Use `supportsAllDrives=true` and `includeItemsFromAllDrives=true` in API params |
| Google Forms / Maps / Sites | Skip — not exportable to useful document formats |
| Very large files (>100MB) | Skip with warning log; configurable `max_file_size` field (default: 100MB) |
| File name collisions (same name, different folders) | Address uses file ID hash — inherently unique |
| Moved files (same file, new path) | File ID is stable — re-index detects same content, updates path metadata |
| Permission denied on individual files | Log warning, skip file, continue with rest of folder |
| Deleted files between syncs | If file_name not in current listing but has chunks → remove chunks |
| SharePoint sites with nested document libraries | Use `site_id` + `drive_id` to target specific library |
| Google Workspace rate limits | Drive API: 12,000 queries/min. Respect `Retry-After` and exponential backoff |
| Microsoft Graph rate limits | 10,000 requests/10 min. Respect `Retry-After` header |
| Office temp files (~$*.docx) | Excluded by default via `exclude_patterns: ["^~\\$"]` |
| Shortcut files (Google) | Resolve shortcut to target file via `shortcutDetails.targetId` |
| Symlinks / OneDrive aliases | Follow to target; deduplicate by file ID |

## Non-Goals (v1)

- File upload or modification (read-only)
- Real-time push notifications (webhooks / change notifications) — sync is pull-based
- Sharing permissions management or ACL enforcement
- Google Drive search (full-text search via Drive API) — we index locally
- Version history (downloading previous file versions)
- Conflict resolution for files being edited during sync
- Box, Dropbox, or other cloud storage providers (separate implementations)
