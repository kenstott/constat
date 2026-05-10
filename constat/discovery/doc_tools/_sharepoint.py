# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""SharePoint document source — libraries, lists, calendars, and pages."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import httpx

if TYPE_CHECKING:
    from constat.core.config import DocumentConfig

logger = logging.getLogger(__name__)

_GRAPH_API = "https://graph.microsoft.com/v1.0"

# SharePoint list base templates
_TEMPLATE_DOCUMENT_LIBRARY = "documentLibrary"
_TEMPLATE_GENERIC_LIST = "genericList"

# File extensions treated as indexable documents when found as hyperlinks
_DOCUMENT_EXTENSIONS = {
    ".docx", ".doc", ".pdf", ".xlsx", ".xls", ".pptx", ".ppt",
    ".txt", ".csv", ".md", ".odt", ".ods", ".odp", ".rtf",
}
_TEMPLATE_EVENTS = "events"


@dataclass
class SPLibrary:
    """A SharePoint document library."""

    id: str
    name: str
    drive_id: str
    item_count: int


@dataclass
class SPList:
    """A SharePoint list."""

    id: str
    name: str
    item_count: int
    columns: list[dict] = field(default_factory=list)  # [{name, type}]


@dataclass
class SPPage:
    """A SharePoint site page."""

    id: str
    name: str
    content: str  # extracted markdown
    url: str | None


class SharePointClient:
    """Fetches content from SharePoint sites via Microsoft Graph or REST API."""

    def __init__(self, config: "DocumentConfig", config_dir: Path | None = None):
        self._config = config
        self._config_dir = config_dir
        self._site_url = config.site_url
        self._api = self._detect_api()

    # ------------------------------------------------------------------
    # API detection
    # ------------------------------------------------------------------

    def _detect_api(self) -> str:
        """Auto-detect API: Graph for *.sharepoint.com, REST otherwise."""
        if self._site_url and "sharepoint.com" in self._site_url:
            return "graph"
        return "rest"

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def _get_access_token(self) -> str:
        """Get an OAuth2 access token (Graph-scoped) via existing Azure provider."""
        from constat.discovery.doc_tools._imap import AzureOAuth2Provider

        return AzureOAuth2Provider(self._config).get_access_token()

    def _has_cert(self) -> bool:
        """Return True if a PFX certificate is configured for SharePoint REST auth."""
        _cfg_cert = getattr(self._config, "oauth2_cert_path", None)
        return bool((_cfg_cert if isinstance(_cfg_cert, str) else None) or os.environ.get("SP_CERT_PATH"))

    def _get_sharepoint_rest_token(self) -> str:
        """Get a SharePoint REST API token scoped to the tenant hostname.

        Certificate auth (SP_CERT_PATH / oauth2_cert_path) bypasses the tenant-level
        AllowAppOnlyPolicy requirement. Client-secret auth requires that policy enabled.
        """
        import msal

        parsed = urlparse(self._site_url)
        hostname = parsed.hostname
        sp_scope = f"https://{hostname}/.default"
        tenant_id = getattr(self._config, "oauth2_tenant_id", None)
        client_id = getattr(self._config, "oauth2_client_id", None)
        authority = f"https://login.microsoftonline.com/{tenant_id}"

        _cfg_cert = getattr(self._config, "oauth2_cert_path", None)
        cert_path = (_cfg_cert if isinstance(_cfg_cert, str) else None) or os.environ.get("SP_CERT_PATH")
        if cert_path:
            _cfg_pw = getattr(self._config, "oauth2_cert_password", None)
            cert_password_raw = (
                (_cfg_pw if isinstance(_cfg_pw, str) else None)
                or os.environ.get("SP_CERT_PASSWORD")
                or ""
            )
            credential = self._load_pfx_credential(cert_path, cert_password_raw)
            app = msal.ConfidentialClientApplication(
                client_id, authority=authority, client_credential=credential
            )
        else:
            client_secret = getattr(self._config, "oauth2_client_secret", None)
            app = msal.ConfidentialClientApplication(
                client_id, authority=authority, client_credential=client_secret
            )

        result = app.acquire_token_for_client(scopes=[sp_scope])
        if "access_token" not in result:
            raise RuntimeError(
                f"SharePoint REST token acquisition failed: {result.get('error_description', result)}"
            )
        return result["access_token"]

    @staticmethod
    def _load_pfx_credential(cert_path: str, password: str) -> dict:
        """Extract thumbprint and private key from a PFX file for MSAL."""
        from cryptography.hazmat.primitives.hashes import SHA1
        from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat
        from cryptography.hazmat.primitives.serialization.pkcs12 import load_pkcs12

        with open(cert_path, "rb") as f:
            pfx_data = f.read()
        pw_bytes = password.encode() if password else None
        pfx = load_pkcs12(pfx_data, pw_bytes)
        private_key_pem = pfx.key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
        thumbprint = pfx.cert.certificate.fingerprint(SHA1()).hex().upper()
        return {"thumbprint": thumbprint, "private_key": private_key_pem.decode()}

    def _get_auth_headers(self) -> dict[str, str]:
        """Build auth headers based on auth_type (ntlm, basic, or bearer)."""
        auth_type = getattr(self._config, "auth_type", "bearer")
        if auth_type in ("ntlm", "basic"):
            import base64
            username = getattr(self._config, "username", "") or ""
            password = getattr(self._config, "password", "") or ""
            creds = base64.b64encode(f"{username}:{password}".encode()).decode()
            return {"Authorization": f"Basic {creds}"}
        return {"Authorization": f"Bearer {self._get_access_token()}"}

    def _headers(self, token: str) -> dict[str, str]:
        """Build standard authorization headers."""
        return {"Authorization": f"Bearer {token}"}

    # ------------------------------------------------------------------
    # REST API methods (on-premises)
    # ------------------------------------------------------------------

    def _discover_rest(self, headers: dict[str, str]) -> dict:
        """SharePoint REST API discovery via /_api/web/lists."""
        url = f"{self._site_url.rstrip('/')}/_api/web/lists"
        resp = httpx.get(
            url,
            headers={**headers, "Accept": "application/json;odata=verbose"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("d", {}).get("results", [])
        return {"lists": results}

    def _list_items_rest(self, list_title: str, headers: dict[str, str]) -> list[dict]:
        """Fetch list items via /_api/web/lists/getbytitle()."""
        encoded_title = list_title.replace("'", "''")
        url = (
            f"{self._site_url.rstrip('/')}/_api/web/lists"
            f"/getbytitle('{encoded_title}')/items"
        )
        resp = httpx.get(
            url,
            headers={**headers, "Accept": "application/json;odata=verbose"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("d", {}).get("results", [])

    def _list_files_rest(self, library_path: str, headers: dict[str, str]) -> list[dict]:
        """List files via /_api/web/GetFolderByServerRelativeUrl()."""
        url = (
            f"{self._site_url.rstrip('/')}/_api/web"
            f"/GetFolderByServerRelativeUrl('{library_path}')/Files"
        )
        resp = httpx.get(
            url,
            headers={**headers, "Accept": "application/json;odata=verbose"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("d", {}).get("results", [])

    # ------------------------------------------------------------------
    # Site resolution
    # ------------------------------------------------------------------

    def _get_site_id(self, token: str) -> str:
        """Resolve site URL to a Graph site ID.

        Root URL  → GET /sites/{hostname}
        Sub-site  → GET /sites/{hostname}:{path}
        """
        parsed = urlparse(self._site_url)
        hostname = parsed.hostname
        site_path = parsed.path.rstrip("/")

        if site_path:
            url = f"{_GRAPH_API}/sites/{hostname}:{site_path}"
        else:
            url = f"{_GRAPH_API}/sites/{hostname}"
        resp = httpx.get(url, headers=self._headers(token), timeout=30)
        resp.raise_for_status()
        return resp.json()["id"]

    # ------------------------------------------------------------------
    # Site discovery
    # ------------------------------------------------------------------

    def discover_site(self) -> dict:
        """Discover libraries, lists, calendars, and pages on the site.

        Returns a dict with keys: libraries, lists, calendars, pages.
        """
        token = self._get_access_token()
        result: dict[str, list] = {
            "libraries": [],
            "lists": [],
            "calendars": [],
            "pages": [],
        }
        site_id = self._get_site_id(token)

        all_lists = self._list_site_lists(token, site_id)
        for lst in all_lists:
            template = lst.get("list", {}).get("template", "")
            list_name = lst.get("displayName", "")

            if template == _TEMPLATE_DOCUMENT_LIBRARY:
                if not self._config.discover_libraries:
                    continue
                if self._config.library_names and list_name not in self._config.library_names:
                    continue
                drive_info = lst.get("drive", {})
                result["libraries"].append(
                    SPLibrary(
                        id=lst["id"],
                        name=list_name,
                        drive_id=drive_info.get("id", ""),
                        item_count=lst.get("list", {}).get("contentTypesEnabled", 0),
                    )
                )
            elif template == _TEMPLATE_EVENTS:
                if not self._config.discover_calendars:
                    continue
                if self._config.calendar_names and list_name not in self._config.calendar_names:
                    continue
                columns = self._get_list_columns(token, site_id, lst["id"])
                result["calendars"].append(
                    SPList(
                        id=lst["id"],
                        name=list_name,
                        item_count=lst.get("list", {}).get("contentTypesEnabled", 0),
                        columns=columns,
                    )
                )
            elif template == _TEMPLATE_GENERIC_LIST:
                if not self._config.discover_lists:
                    continue
                if self._config.list_names and list_name not in self._config.list_names:
                    continue
                columns = self._get_list_columns(token, site_id, lst["id"])
                result["lists"].append(
                    SPList(
                        id=lst["id"],
                        name=list_name,
                        item_count=lst.get("list", {}).get("contentTypesEnabled", 0),
                        columns=columns,
                    )
                )

        if self._config.discover_pages:
            result["pages"] = self._list_pages(token, site_id)

        return result

    def _list_site_lists(self, token: str, site_id: str) -> list[dict]:
        """Fetch all lists on a site (paginated)."""
        url = f"{_GRAPH_API}/sites/{site_id}/lists"
        params: dict[str, str] = {"$top": "200", "$expand": "drive"}
        all_lists: list[dict] = []

        while True:
            resp = httpx.get(url, params=params, headers=self._headers(token), timeout=30)
            resp.raise_for_status()
            data = resp.json()
            all_lists.extend(data.get("value", []))

            next_link = data.get("@odata.nextLink")
            if not next_link:
                break
            url = next_link
            params = {}

        return all_lists

    def _get_list_columns(self, token: str, site_id: str, list_id: str) -> list[dict]:
        """Fetch column definitions for a list."""
        url = f"{_GRAPH_API}/sites/{site_id}/lists/{list_id}/columns"
        resp = httpx.get(url, headers=self._headers(token), timeout=30)
        resp.raise_for_status()
        columns: list[dict] = []
        for col in resp.json().get("value", []):
            # Skip hidden/system columns
            if col.get("readOnly") and col.get("name", "").startswith("_"):
                continue
            columns.append({
                "name": col.get("name", ""),
                "type": col.get("text", col.get("number", col.get("dateTime", "unknown"))),
            })
        return columns

    # ------------------------------------------------------------------
    # Document library files
    # ------------------------------------------------------------------

    def _make_drive_config(self, drive_id: str, **overrides):
        """Build a config object for DriveFetcher by copying SP config attrs."""
        from types import SimpleNamespace

        attrs = {k: getattr(self._config, k, None) for k in (
            "auth_type", "oauth2_client_id", "oauth2_client_secret", "oauth2_tenant_id",
            "oauth2_scopes", "oauth2_token_cache",
            "recursive", "max_files", "include_types",
            "exclude_patterns", "include_trashed", "since",
        )}
        attrs.update(
            provider="microsoft",
            drive_id=drive_id,
            site_id=None,
            folder_id=None,
            folder_path=getattr(self._config, "folder_path", None),
        )
        attrs.update(overrides)
        return SimpleNamespace(**attrs)

    def fetch_library_files(self, library: SPLibrary) -> list:
        """List files in a document library (reuses DriveFetcher pattern).

        Returns a list of DriveFile objects.
        """
        from constat.discovery.doc_tools._drive import DriveFetcher

        fetcher = DriveFetcher(
            self._make_drive_config(library.drive_id),
            config_dir=self._config_dir,
        )
        return fetcher.list_files()

    def download_library_file(self, library: SPLibrary, file) -> bytes:
        """Download a file from a document library."""
        from constat.discovery.doc_tools._drive import DriveFetcher

        fetcher = DriveFetcher(
            self._make_drive_config(library.drive_id),
            config_dir=self._config_dir,
        )
        return fetcher.download_file(file)

    # ------------------------------------------------------------------
    # List items
    # ------------------------------------------------------------------

    def fetch_list_items(self, sp_list: SPList) -> list[dict]:
        """Fetch all items from a SharePoint list (paginated).

        Returns raw Graph API item dicts with 'fields' expanded.
        """
        token = self._get_access_token()
        site_id = self._get_site_id(token)

        url = f"{_GRAPH_API}/sites/{site_id}/lists/{sp_list.id}/items"
        params: dict[str, str] = {"$expand": "fields", "$top": "200"}
        all_items: list[dict] = []
        max_rows = getattr(self._config, "max_rows", 5000)

        while True:
            resp = httpx.get(url, params=params, headers=self._headers(token), timeout=30)
            resp.raise_for_status()
            data = resp.json()
            all_items.extend(data.get("value", []))

            if len(all_items) >= max_rows:
                break

            next_link = data.get("@odata.nextLink")
            if not next_link:
                break
            url = next_link
            params = {}

        return all_items[:max_rows]

    def render_list_as_markdown(self, sp_list: SPList, items: list[dict]) -> str:
        """Render list items as a markdown table."""
        if not items:
            return f"# {sp_list.name}\n\n(empty list)"

        headers = [col["name"] for col in sp_list.columns]
        max_rows = getattr(self._config, "max_rows", 5000)
        lines = [
            f"# {sp_list.name}",
            "",
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for item in items[:max_rows]:
            fields = item.get("fields", item)
            row = [str(fields.get(h, "")) for h in headers]
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    def fetch_list_item_attachments(self, sp_list: SPList, item_id: str) -> list[dict]:
        """List attachments for a single list item via SharePoint REST API.

        Graph API v1.0 does not expose the /attachments navigation property for
        SharePoint list items; the SharePoint REST API is the only supported path.

        Returns list of dicts with keys: FileName, ServerRelativeUrl.

        Raises:
            httpx.HTTPStatusError: 401 if the Azure AD app-only policy is not
                enabled on the tenant. Tenant admin must run:
                Set-SPOTenant -AllowAppOnlyPolicy $true
        """
        if not self._has_cert():
            logger.warning(
                "SharePoint list item attachments require certificate auth "
                "(SP_CERT_PATH not set) — skipping attachment indexing."
            )
            return []
        token = self._get_sharepoint_rest_token()
        encoded_name = sp_list.name.replace("'", "''")
        url = (
            f"{self._site_url.rstrip('/')}/_api/web"
            f"/lists/getbytitle('{encoded_name}')/items({item_id})/AttachmentFiles"
        )
        resp = httpx.get(
            url,
            headers={**self._headers(token), "Accept": "application/json;odata=verbose"},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("d", {}).get("results", [])

    def download_list_item_attachment(
        self, sp_list: SPList, item_id: str, file_name: str
    ) -> bytes:
        """Download attachment content via SharePoint REST API.

        Raises:
            httpx.HTTPStatusError: 401 if AllowAppOnlyPolicy is not enabled.
        """
        if not self._has_cert():
            logger.warning(
                "SharePoint list item attachments require certificate auth "
                "(SP_CERT_PATH not set) — skipping attachment download."
            )
            return b""
        token = self._get_sharepoint_rest_token()
        encoded_name = sp_list.name.replace("'", "''")
        encoded_file = file_name.replace("'", "''")
        url = (
            f"{self._site_url.rstrip('/')}/_api/web"
            f"/lists/getbytitle('{encoded_name}')/items({item_id})"
            f"/AttachmentFiles('{encoded_file}')/$value"
        )
        resp = httpx.get(
            url,
            headers=self._headers(token),
            timeout=60,
        )
        resp.raise_for_status()
        return resp.content

    # ------------------------------------------------------------------
    # Linked file extraction
    # ------------------------------------------------------------------

    def _extract_sp_urls(self, text: str) -> list[str]:
        """Extract SharePoint document URLs from a text or HTML field value.

        Handles both absolute URLs (https://hostname/...) and server-relative
        hrefs in HTML anchor tags (/SiteAssets/...).
        """
        if not text or not isinstance(text, str):
            return []
        parsed = urlparse(self._site_url)
        hostname = re.escape(parsed.hostname or "")
        base = f"{parsed.scheme}://{parsed.hostname}"
        seen: set[str] = set()
        results: list[str] = []

        def _accept(url: str) -> None:
            url = url.rstrip(".,;)")
            ext = Path(urlparse(url).path).suffix.lower()
            if ext in _DOCUMENT_EXTENSIONS and url not in seen:
                seen.add(url)
                results.append(url)

        # Absolute URLs containing the SharePoint hostname
        for url in re.findall(rf'https?://{hostname}[^\s"\'<>]+', text):
            _accept(url)

        # Server-relative hrefs in HTML anchor tags: href="/path/to/file.docx"
        for rel in re.findall(r'href=["\']([^"\']+)["\']', text):
            if rel.startswith("/"):
                _accept(base + rel)

        return results

    def fetch_linked_file(self, absolute_url: str) -> bytes:
        """Download a SharePoint-hosted file by its absolute URL."""
        if not self._has_cert():
            logger.warning(
                "Fetching linked SharePoint files requires certificate auth "
                "(SP_CERT_PATH not set) — skipping linked file download."
            )
            return b""
        token = self._get_sharepoint_rest_token()
        server_relative = urlparse(absolute_url).path.replace("'", "''")
        url = (
            f"{self._site_url.rstrip('/')}/_api/web"
            f"/GetFileByServerRelativeUrl('{server_relative}')/$value"
        )
        resp = httpx.get(url, headers=self._headers(token), timeout=60)
        resp.raise_for_status()
        return resp.content

    def extract_linked_files_from_items(
        self, items: list[dict]
    ) -> list[tuple[str, str, bytes]]:
        """Scan all string fields of list items for embedded SharePoint file URLs.

        Returns list of (item_id, url, content) for each linked document found.
        Inaccessible URLs are skipped with a debug log entry.
        """
        results = []
        seen: set[str] = set()
        for item in items:
            item_id = item.get("id", "")
            fields = item.get("fields", item)
            for value in fields.values():
                for url in self._extract_sp_urls(str(value) if value else ""):
                    if url in seen:
                        continue
                    seen.add(url)
                    try:
                        content = self.fetch_linked_file(url)
                        results.append((item_id, url, content))
                    except httpx.HTTPStatusError:
                        logger.debug("Linked file not accessible: %s", url)
        return results

    # ------------------------------------------------------------------
    # Calendar events
    # ------------------------------------------------------------------

    def fetch_calendar_events(self, calendar_list: SPList) -> list:
        """Fetch calendar list items and convert to CalendarEvent objects.

        SharePoint calendars are lists with BaseTemplate=106 (events).
        """
        from constat.discovery.doc_tools._calendar import CalendarEvent

        items = self.fetch_list_items(calendar_list)
        events: list[CalendarEvent] = []

        for item in items:
            fields = item.get("fields", item)
            event_id = self._make_event_id(item)

            start_str = fields.get("EventDate", fields.get("StartDate", ""))
            end_str = fields.get("EndDate", "")

            start_dt = self._parse_sp_datetime(start_str)
            end_dt = self._parse_sp_datetime(end_str)
            all_day = fields.get("fAllDayEvent", False)

            events.append(
                CalendarEvent(
                    event_id=event_id,
                    title=fields.get("Title", "(No title)"),
                    start=start_dt,
                    end=end_dt,
                    all_day=all_day,
                    location=fields.get("Location"),
                    organizer=fields.get("Author", {}).get("email", "")
                    if isinstance(fields.get("Author"), dict)
                    else str(fields.get("Author", "")),
                    attendees=[],
                    status="confirmed",
                    description=fields.get("Description"),
                    recurrence_id=fields.get("RecurrenceID"),
                )
            )

        return events

    @staticmethod
    def _parse_sp_datetime(value: str) -> datetime:
        """Parse a SharePoint datetime string to a timezone-aware datetime."""
        if not value:
            return datetime(1970, 1, 1, tzinfo=timezone.utc)
        # SharePoint datetimes may have trailing Z or offset
        cleaned = value.rstrip("Z")
        dt = datetime.fromisoformat(cleaned)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    @staticmethod
    def _make_event_id(item: dict) -> str:
        """Deterministic short ID for a SharePoint calendar event."""
        raw_id = str(item.get("id", ""))
        short_hash = hashlib.sha256(raw_id.encode()).hexdigest()[:8]
        fields = item.get("fields", item)
        title_slug = re.sub(
            r"[^a-z0-9]", "",
            (fields.get("Title") or "event").lower(),
        )[:20]
        return f"evt_sp_{title_slug}_{short_hash}"

    # ------------------------------------------------------------------
    # Site pages
    # ------------------------------------------------------------------

    def fetch_pages(self) -> list[SPPage]:
        """Fetch site pages and extract content."""
        token = self._get_access_token()
        site_id = self._get_site_id(token)
        raw_pages = self._list_pages(token, site_id)
        return raw_pages

    def _list_pages(self, token: str, site_id: str) -> list[SPPage]:
        """List and parse site pages from Graph API."""
        url = f"{_GRAPH_API}/sites/{site_id}/pages"
        params: dict[str, str] = {"$top": "200"}
        pages: list[SPPage] = []
        page_types = getattr(self._config, "page_types", None)

        while True:
            resp = httpx.get(url, params=params, headers=self._headers(token), timeout=30)
            resp.raise_for_status()
            data = resp.json()

            for item in data.get("value", []):
                page_type = item.get("pageLayout", "article")

                # Filter by page_types if configured
                if page_types:
                    if page_type == "article" and "modern" not in page_types:
                        continue
                    if page_type == "wiki" and "wiki" not in page_types:
                        continue

                content = self._extract_page_content(item, token, site_id)
                page_id = self._make_page_id(item)

                pages.append(
                    SPPage(
                        id=page_id,
                        name=item.get("name", item.get("title", "")),
                        content=content,
                        url=item.get("webUrl"),
                    )
                )

            next_link = data.get("@odata.nextLink")
            if not next_link:
                break
            url = next_link
            params = {}

        return pages

    def _extract_page_content(self, page: dict, token: str, site_id: str) -> str:
        """Extract text content from a SharePoint page.

        Modern pages use CanvasContent1 JSON; wiki pages use WikiField HTML.
        """
        page_id = page.get("id", "")

        # Fetch full page content
        url = f"{_GRAPH_API}/sites/{site_id}/pages/{page_id}/microsoft.graph.sitePage"
        params = {"$select": "id,title,canvasLayout"}
        resp = httpx.get(url, params=params, headers=self._headers(token), timeout=30)

        if resp.status_code != 200:
            # Fallback: use description or title
            return page.get("description", page.get("title", ""))

        page_data = resp.json()

        # Modern page: canvasLayout
        canvas_layout = page_data.get("canvasLayout")
        if canvas_layout:
            return self._extract_canvas_layout(canvas_layout)

        # Wiki page: WikiField in page properties
        wiki_field = page_data.get("WikiField", "")
        if wiki_field:
            return re.sub(r"<[^>]+>", "", wiki_field)

        return page.get("description", page.get("title", ""))

    def _extract_canvas_layout(self, canvas_layout: dict) -> str:
        """Extract text from a modern page canvasLayout structure."""
        texts: list[str] = []
        for section in canvas_layout.get("horizontalSections", []):
            for column in section.get("columns", []):
                for webpart in column.get("webparts", []):
                    inner_html = webpart.get("innerHtml", "")
                    if inner_html:
                        texts.append(re.sub(r"<[^>]+>", "", inner_html))
        return "\n\n".join(texts)

    def _extract_canvas_content(self, canvas_json: str) -> str:
        """Extract text from CanvasContent1 JSON (legacy format).

        CanvasContent1 is a JSON array of sections, each with columns
        containing controls with innerHTML.
        """
        try:
            canvas = json.loads(canvas_json)
        except (json.JSONDecodeError, TypeError):
            return ""

        texts: list[str] = []
        if isinstance(canvas, list):
            for section in canvas:
                if not isinstance(section, dict):
                    continue
                for column in section.get("columns", []):
                    if not isinstance(column, dict):
                        continue
                    for control in column.get("controls", []):
                        if not isinstance(control, dict):
                            continue
                        inner = control.get("innerHTML", "")
                        if inner:
                            texts.append(re.sub(r"<[^>]+>", "", inner))
        return "\n\n".join(texts)

    @staticmethod
    def _make_page_id(page: dict) -> str:
        """Deterministic short ID for a SharePoint page."""
        raw_id = str(page.get("id", ""))
        short_hash = hashlib.sha256(raw_id.encode()).hexdigest()[:8]
        title_slug = re.sub(
            r"[^a-z0-9]", "_",
            (page.get("title") or page.get("name") or "page").lower(),
        )[:30]
        return f"pg_{short_hash}_{title_slug}"
