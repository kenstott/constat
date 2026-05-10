# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Cloud Drive document source — Google Drive and Microsoft OneDrive/SharePoint."""

from __future__ import annotations

import hashlib
import logging
import re
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from constat.core.config import DocumentConfig

logger = logging.getLogger(__name__)

_GOOGLE_DRIVE_API = "https://www.googleapis.com/drive/v3"
_GRAPH_API = "https://graph.microsoft.com/v1.0"

# Google native format export mapping
_GOOGLE_EXPORT_MAP: dict[str, str] = {
    "application/vnd.google-apps.document": (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ),
    "application/vnd.google-apps.spreadsheet": (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    ),
    "application/vnd.google-apps.presentation": (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    ),
}

# Google native MIME → effective file extension (for filter matching)
_GOOGLE_NATIVE_EXTS: dict[str, str] = {
    "application/vnd.google-apps.document": ".docx",
    "application/vnd.google-apps.spreadsheet": ".xlsx",
    "application/vnd.google-apps.presentation": ".pptx",
}

_GOOGLE_FOLDER_MIME = "application/vnd.google-apps.folder"
_GOOGLE_NATIVE_PREFIX = "application/vnd.google-apps."


@dataclass
class DriveFile:
    """Represents a file in a cloud drive."""

    file_id: str
    name: str
    mime_type: str
    size: int | None
    modified_time: datetime
    path: str
    parent_id: str | None
    web_url: str | None
    is_google_native: bool = False


class DriveFetcher:
    """Fetches files from Google Drive or Microsoft OneDrive/SharePoint."""

    def __init__(self, config: "DocumentConfig", config_dir: Path | None = None):
        self._config = config
        self._config_dir = config_dir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_files(self) -> list[DriveFile]:
        """List files from the configured cloud drive provider."""
        if self._config.provider == "google":
            return self._list_google()
        elif self._config.provider == "microsoft":
            return self._list_microsoft()
        raise ValueError(f"Unknown drive provider: {self._config.provider}")

    def download_file(self, file: DriveFile) -> bytes:
        """Download (or export) a single file."""
        if self._config.provider == "google":
            return self._download_google(file)
        elif self._config.provider == "microsoft":
            return self._download_microsoft(file)
        raise ValueError(f"Unknown drive provider: {self._config.provider}")

    # ------------------------------------------------------------------
    # Google Drive API v3
    # ------------------------------------------------------------------

    def _resolve_google_folder(self) -> str:
        """Resolve folder_id or folder_path to a Google Drive folder ID."""
        if self._config.folder_id:
            return self._config.folder_id
        if self._config.folder_path:
            return self._resolve_google_path(self._config.folder_path)
        raise ValueError("Either folder_id or folder_path required for Google Drive")

    def _resolve_google_path(self, folder_path: str) -> str:
        """Walk a path like '/Shared Drives/Analytics' to resolve to a folder ID."""
        import httpx

        token = self._get_access_token()
        headers = {"Authorization": f"Bearer {token}"}

        parts = [p for p in folder_path.strip("/").split("/") if p]
        current_id = "root"

        for part in parts:
            query = (
                f"'{current_id}' in parents"
                f" and name = '{part}'"
                f" and mimeType = '{_GOOGLE_FOLDER_MIME}'"
                " and trashed = false"
            )
            resp = httpx.get(
                f"{_GOOGLE_DRIVE_API}/files",
                params={
                    "q": query,
                    "fields": "files(id,name)",
                    "pageSize": 1,
                    "supportsAllDrives": "true",
                    "includeItemsFromAllDrives": "true",
                },
                headers=headers,
            )
            resp.raise_for_status()
            files = resp.json().get("files", [])
            if not files:
                raise ValueError(
                    f"Folder not found: '{part}' in path '{folder_path}'"
                )
            current_id = files[0]["id"]

        return current_id

    def _list_google(self) -> list[DriveFile]:
        """List files via Google Drive API v3 with BFS folder traversal."""
        import httpx

        token = self._get_access_token()
        root_id = self._resolve_google_folder()
        headers = {"Authorization": f"Bearer {token}"}
        files: list[DriveFile] = []
        max_files = self._config.max_files

        folders_to_scan: deque[tuple[str, str]] = deque()
        folders_to_scan.append((root_id, ""))

        while folders_to_scan and len(files) < max_files:
            folder_id, folder_prefix = folders_to_scan.popleft()

            query_parts = [f"'{folder_id}' in parents"]
            if not self._config.include_trashed:
                query_parts.append("trashed = false")
            if self._config.since:
                query_parts.append(
                    f"modifiedTime > '{self._config.since}T00:00:00Z'"
                )
            query = " and ".join(query_parts)

            params: dict = {
                "q": query,
                "fields": (
                    "nextPageToken,"
                    "files(id,name,mimeType,size,modifiedTime,parents,webViewLink)"
                ),
                "pageSize": 1000,
                "supportsAllDrives": "true",
                "includeItemsFromAllDrives": "true",
            }

            while True:
                resp = httpx.get(
                    f"{_GOOGLE_DRIVE_API}/files",
                    params=params,
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()

                for item in data.get("files", []):
                    if item["mimeType"] == _GOOGLE_FOLDER_MIME:
                        if self._config.recursive:
                            child_prefix = f"{folder_prefix}{item['name']}/"
                            folders_to_scan.append((item["id"], child_prefix))
                        continue

                    if not self._matches_filters(item["name"], item["mimeType"]):
                        continue

                    is_native = item["mimeType"].startswith(_GOOGLE_NATIVE_PREFIX)
                    file_path = f"{folder_prefix}{item['name']}"
                    mod_str = item["modifiedTime"]
                    modified = datetime.fromisoformat(
                        mod_str.rstrip("Z")
                    ).replace(tzinfo=timezone.utc)

                    files.append(
                        DriveFile(
                            file_id=item["id"],
                            name=item["name"],
                            mime_type=item["mimeType"],
                            size=int(item["size"]) if item.get("size") else None,
                            modified_time=modified,
                            path=file_path,
                            parent_id=folder_id,
                            web_url=item.get("webViewLink"),
                            is_google_native=is_native,
                        )
                    )

                    if len(files) >= max_files:
                        break

                page_token = data.get("nextPageToken")
                if not page_token or len(files) >= max_files:
                    break
                params["pageToken"] = page_token

        return files[:max_files]

    def _download_google(self, file: DriveFile) -> bytes:
        """Download or export a Google Drive file."""
        import httpx

        token = self._get_access_token()
        headers = {"Authorization": f"Bearer {token}"}

        if file.is_google_native:
            export_mime = _GOOGLE_EXPORT_MAP.get(file.mime_type)
            if not export_mime:
                raise ValueError(
                    f"Cannot export Google native format: {file.mime_type}"
                )
            resp = httpx.get(
                f"{_GOOGLE_DRIVE_API}/files/{file.file_id}/export",
                params={"mimeType": export_mime},
                headers=headers,
            )
        else:
            resp = httpx.get(
                f"{_GOOGLE_DRIVE_API}/files/{file.file_id}",
                params={"alt": "media"},
                headers=headers,
            )

        resp.raise_for_status()
        return resp.content

    # ------------------------------------------------------------------
    # Microsoft Graph API (OneDrive / SharePoint)
    # ------------------------------------------------------------------

    def _resolve_microsoft_base_url(self) -> str:
        """Build the base Graph API URL for the target drive."""
        if self._config.site_id:
            return f"{_GRAPH_API}/sites/{self._config.site_id}/drive"
        if self._config.drive_id:
            return f"{_GRAPH_API}/drives/{self._config.drive_id}"
        return f"{_GRAPH_API}/me/drive"

    def _list_microsoft(self) -> list[DriveFile]:
        """List files via Microsoft Graph API with BFS folder traversal."""
        import httpx

        token = self._get_access_token()
        headers = {"Authorization": f"Bearer {token}"}
        base_url = self._resolve_microsoft_base_url()
        files: list[DriveFile] = []
        max_files = self._config.max_files

        # Resolve starting folder URL
        if self._config.folder_path:
            stripped = self._config.folder_path.strip("/")
            start_url = f"{base_url}/root:/{stripped}:/children"
        elif self._config.folder_id:
            start_url = f"{base_url}/items/{self._config.folder_id}/children"
        else:
            start_url = f"{base_url}/root/children"

        since_dt = self._parse_since()

        folders_to_scan: deque[str] = deque()
        folders_to_scan.append(start_url)

        while folders_to_scan and len(files) < max_files:
            url = folders_to_scan.popleft()
            params: dict = {"$top": "200"}

            while True:
                resp = httpx.get(url, params=params, headers=headers)
                resp.raise_for_status()
                data = resp.json()

                for item in data.get("value", []):
                    if "folder" in item:
                        if self._config.recursive:
                            child_url = (
                                f"{base_url}/items/{item['id']}/children"
                            )
                            folders_to_scan.append(child_url)
                        continue

                    if not self._matches_filters(
                        item["name"],
                        item.get("file", {}).get("mimeType", ""),
                    ):
                        continue

                    mod_str = item["lastModifiedDateTime"]
                    modified = datetime.fromisoformat(
                        mod_str.rstrip("Z")
                    ).replace(tzinfo=timezone.utc)

                    if since_dt and modified < since_dt:
                        continue

                    parent_ref = item.get("parentReference", {})
                    file_path = (
                        parent_ref.get("path", "") + "/" + item["name"]
                    )

                    files.append(
                        DriveFile(
                            file_id=item["id"],
                            name=item["name"],
                            mime_type=item.get("file", {}).get(
                                "mimeType", "application/octet-stream"
                            ),
                            size=item.get("size"),
                            modified_time=modified,
                            path=file_path,
                            parent_id=parent_ref.get("id"),
                            web_url=item.get("webUrl"),
                        )
                    )

                    if len(files) >= max_files:
                        break

                next_link = data.get("@odata.nextLink")
                if not next_link or len(files) >= max_files:
                    break
                url = next_link
                params = {}

        return files[:max_files]

    def _download_microsoft(self, file: DriveFile) -> bytes:
        """Download a file from OneDrive/SharePoint."""
        import httpx

        token = self._get_access_token()
        base_url = self._resolve_microsoft_base_url()
        url = f"{base_url}/items/{file.file_id}/content"
        resp = httpx.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
            follow_redirects=True,
        )
        resp.raise_for_status()
        return resp.content

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _get_access_token(self) -> str:
        """Get an OAuth2 access token using the configured provider."""
        from ._imap import (
            AzureOAuth2Provider,
            GoogleOAuth2Provider,
            RefreshTokenOAuth2Provider,
        )

        if self._config.auth_type == "oauth2_refresh":
            return RefreshTokenOAuth2Provider(self._config).get_access_token()
        if self._config.oauth2_tenant_id:
            return AzureOAuth2Provider(self._config).get_access_token()
        return GoogleOAuth2Provider(self._config).get_access_token()

    def _matches_filters(self, filename: str, mime_type: str) -> bool:
        """Check if a file passes include_types and exclude_patterns filters."""
        if self._config.include_types:
            ext = Path(filename).suffix.lower()
            # Google native formats have no file extension — use effective ext
            effective_ext = _GOOGLE_NATIVE_EXTS.get(mime_type, ext)
            if effective_ext not in self._config.include_types:
                return False

        if self._config.exclude_patterns:
            for pattern in self._config.exclude_patterns:
                if re.search(pattern, filename):
                    return False

        return True

    def _parse_since(self) -> datetime | None:
        """Parse the 'since' config field into a timezone-aware datetime."""
        if not self._config.since:
            return None
        return datetime.fromisoformat(self._config.since).replace(
            tzinfo=timezone.utc
        )

    @staticmethod
    def make_file_id(file: DriveFile) -> str:
        """Deterministic short ID from provider file ID + name."""
        short_hash = hashlib.sha256(file.file_id.encode()).hexdigest()[:8]
        slug = re.sub(r"[^a-z0-9]", "_", Path(file.name).stem.lower())[:30]
        return f"f_{short_hash}_{slug}"
