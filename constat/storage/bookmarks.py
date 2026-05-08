# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Bookmark storage for databases and files.

Provides persistent storage for frequently used data sources that can be
recalled across sessions. Bookmarks are stored in .constat/bookmarks.yaml.
"""

import os
from pathlib import Path
from typing import Optional

import yaml


class BookmarkStore:
    """Manages persistent bookmarks for databases and files.

    Bookmarks are stored in YAML format:
    ```yaml
    databases:
      mydb:
        type: sql
        uri: sqlite:///./data/mydb.db
        description: "My local database"

    files:
      report:
        uri: file:///shared/reports/q4.pdf
        description: "Q4 2025 financial report"
        auth: ""  # Optional auth header for HTTP
    ```
    """

    def __init__(self, base_dir: Optional[Path] = None):
        """Initialize bookmark store.

        Args:
            base_dir: Directory for .constat. Defaults to current directory.
        """
        self.base_dir = Path(base_dir) if base_dir else Path(".constat")
        self.file_path = self.base_dir / "bookmarks.yaml"
        self._data: Optional[dict] = None

    def _load(self) -> dict:
        """Load bookmarks from YAML file."""
        if self._data is not None:
            return self._data

        if not self.file_path.exists():
            self._data = {"databases": {}, "files": {}}
            return self._data

        with open(self.file_path, "r") as f:
            self._data = yaml.safe_load(f) or {"databases": {}, "files": {}}

        # Ensure both sections exist
        if "databases" not in self._data:
            self._data["databases"] = {}
        if "files" not in self._data:
            self._data["files"] = {}

        return self._data

    def _save(self) -> None:
        """Save bookmarks to YAML file."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        with open(self.file_path, "w") as f:
            yaml.dump(self._data, f, default_flow_style=False, sort_keys=False)

    def _expand_env(self, value: str) -> str:
        """Expand environment variables in a string."""
        return os.path.expandvars(value)

    # --- Database Bookmarks ---

    def save_database(
        self,
        name: str,
        db_type: str,
        uri: str,
        description: str = "",
    ) -> None:
        """Save a database bookmark.

        Args:
            name: Bookmark name (used to recall later)
            db_type: Database type (sql, csv, json, parquet, mongodb, etc.)
            uri: Connection URI or file path
            description: Human-readable description
        """
        data = self._load()
        data["databases"][name] = {
            "type": db_type,
            "uri": uri,
            "description": description,
        }
        self._save()

    def get_database(self, name: str) -> Optional[dict]:
        """Get a database bookmark by name.

        Args:
            name: Bookmark name

        Returns:
            Dict with type, uri, description (with env vars expanded), or None
        """
        data = self._load()
        bookmark = data["databases"].get(name)
        if bookmark:
            return {
                "type": bookmark["type"],
                "uri": self._expand_env(bookmark["uri"]),
                "description": bookmark.get("description", ""),
            }
        return None

    def list_databases(self) -> dict[str, dict]:
        """List all database bookmarks.

        Returns:
            Dict of name -> {type, uri, description}
        """
        data = self._load()
        return {
            name: {
                "type": bm["type"],
                "uri": self._expand_env(bm["uri"]),
                "description": bm.get("description", ""),
            }
            for name, bm in data["databases"].items()
        }

    def delete_database(self, name: str) -> bool:
        """Delete a database bookmark.

        Args:
            name: Bookmark name

        Returns:
            True if deleted, False if not found
        """
        data = self._load()
        if name in data["databases"]:
            del data["databases"][name]
            self._save()
            return True
        return False

    # --- File Bookmarks ---

    def save_file(
        self,
        name: str,
        uri: str,
        description: str = "",
        auth: str = "",
    ) -> None:
        """Save a file bookmark.

        Args:
            name: Bookmark name (used to recall later)
            uri: File URI (file:// or http://)
            description: Human-readable description
            auth: Auth header for HTTP (e.g., "Bearer token123")
        """
        data = self._load()
        data["files"][name] = {
            "uri": uri,
            "description": description,
        }
        if auth:
            data["files"][name]["auth"] = auth
        self._save()

    def get_file(self, name: str) -> Optional[dict]:
        """Get a file bookmark by name.

        Args:
            name: Bookmark name

        Returns:
            Dict with uri, description, auth (with env vars expanded), or None
        """
        data = self._load()
        bookmark = data["files"].get(name)
        if bookmark:
            return {
                "uri": self._expand_env(bookmark["uri"]),
                "description": bookmark.get("description", ""),
                "auth": self._expand_env(bookmark.get("auth", "")),
            }
        return None

    def list_files(self) -> dict[str, dict]:
        """List all file bookmarks.

        Returns:
            Dict of name -> {uri, description, auth}
        """
        data = self._load()
        return {
            name: {
                "uri": self._expand_env(bm["uri"]),
                "description": bm.get("description", ""),
                "auth": self._expand_env(bm.get("auth", "")),
            }
            for name, bm in data["files"].items()
        }

    def delete_file(self, name: str) -> bool:
        """Delete a file bookmark.

        Args:
            name: Bookmark name

        Returns:
            True if deleted, False if not found
        """
        data = self._load()
        if name in data["files"]:
            del data["files"][name]
            self._save()
            return True
        return False
