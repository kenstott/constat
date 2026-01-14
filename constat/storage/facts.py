"""Persistent fact storage for user-scoped facts.

Provides storage for facts that persist across sessions, stored in
.constat/<user_id>/facts.yaml.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml


class FactStore:
    """Manages persistent user-scoped facts.

    Facts are stored in YAML format:
    ```yaml
    facts:
      user_role:
        value: CFO
        description: User's role for context-aware suggestions
        created: 2024-01-15T10:30:00Z
      fiscal_year_start:
        value: April
        description: When fiscal year begins
        created: 2024-01-15T10:31:00Z
    ```
    """

    def __init__(self, base_dir: Optional[Path] = None, user_id: str = "default"):
        """Initialize fact store.

        Args:
            base_dir: Base directory for .constat. Defaults to current directory.
            user_id: User ID for user-scoped storage.
        """
        self.base_dir = Path(base_dir) if base_dir else Path(".constat")
        self.user_id = user_id
        self.file_path = self.base_dir / user_id / "facts.yaml"
        self._data: Optional[dict] = None

    def _load(self) -> dict:
        """Load facts from YAML file."""
        if self._data is not None:
            return self._data

        if not self.file_path.exists():
            self._data = {"facts": {}}
            return self._data

        with open(self.file_path, "r") as f:
            self._data = yaml.safe_load(f) or {"facts": {}}

        if "facts" not in self._data:
            self._data["facts"] = {}

        return self._data

    def _save(self) -> None:
        """Save facts to YAML file."""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.file_path, "w") as f:
            yaml.dump(self._data, f, default_flow_style=False, sort_keys=False)

    def save_fact(
        self,
        name: str,
        value: Any,
        description: str = "",
    ) -> None:
        """Save a persistent fact.

        Args:
            name: Fact name (snake_case recommended)
            value: Fact value (string, number, etc.)
            description: Human-readable description
        """
        data = self._load()
        data["facts"][name] = {
            "value": value,
            "description": description,
            "created": datetime.now(timezone.utc).isoformat(),
        }
        self._save()

    def get_fact(self, name: str) -> Optional[dict]:
        """Get a fact by name.

        Args:
            name: Fact name

        Returns:
            Dict with value, description, created, or None if not found
        """
        data = self._load()
        return data["facts"].get(name)

    def list_facts(self) -> dict[str, dict]:
        """List all persistent facts.

        Returns:
            Dict of name -> {value, description, created}
        """
        data = self._load()
        return data["facts"].copy()

    def delete_fact(self, name: str) -> bool:
        """Delete a persistent fact.

        Args:
            name: Fact name

        Returns:
            True if deleted, False if not found
        """
        data = self._load()
        if name in data["facts"]:
            del data["facts"][name]
            self._save()
            return True
        return False

    def clear_all(self) -> int:
        """Clear all persistent facts.

        Returns:
            Number of facts cleared
        """
        data = self._load()
        count = len(data["facts"])
        data["facts"] = {}
        self._save()
        return count
