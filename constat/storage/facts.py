# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Persistent fact storage with role provenance tracking.

Provides storage for facts that persist across sessions, stored in
.constat/<user_id>/facts.yaml.

All facts are globally accessible. The role_id field is metadata indicating
which role created the fact (provenance), not access control:
- role_id=None: Created in shared context
- role_id="financial-analyst": Created by financial-analyst role

The role_id enables:
- Provenance tracking (which role created this fact)
- UI grouping (display facts by source role)
- Trust attribution (role-derived facts may carry different confidence)
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml


class FactStore:
    """Manages persistent facts with role provenance tracking.

    Facts are stored in YAML format with role_id for provenance:
    ```yaml
    facts:
      user_role:
        value: CFO
        description: User's role for context-aware suggestions
        context: "User text: my role is CFO"
        role_id: null  # Created in shared context
        created: 2024-01-15T10:30:00Z
      q3_revenue:
        value: 4200000
        description: Q3 2024 revenue
        context: "Calculated from sales data"
        role_id: financial-analyst  # Created by financial-analyst role
        created: 2024-01-15T10:31:00Z
    ```

    All facts are globally accessible regardless of role_id.
    The role_id is metadata for provenance and UI grouping.
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
        context: str = "",
        role_id: Optional[str] = None,
    ) -> None:
        """Save a persistent fact.

        Args:
            name: Fact name (snake_case recommended)
            value: Fact value (string, number, etc.)
            description: Human-readable description
            context: Creation context (code, prompt, query that created this fact)
            role_id: Role that created this fact (None = shared)
        """
        data = self._load()
        data["facts"][name] = {
            "value": value,
            "description": description,
            "context": context,
            "role_id": role_id,
            "created": datetime.now(timezone.utc).isoformat(),
        }
        self._save()

    def get_fact(self, name: str) -> Optional[dict]:
        """Get a fact by name.

        Args:
            name: Fact name

        Returns:
            Dict with value, description, context, created, or None if not found
        """
        data = self._load()
        return data["facts"].get(name)

    def list_facts(self, role_id: Optional[str] = None, include_shared: bool = True) -> dict[str, dict]:
        """List persistent facts, optionally filtered by role.

        Args:
            role_id: Filter to facts from this role. None = shared facts only.
            include_shared: If True and role_id is set, also include shared facts.

        Returns:
            Dict of name -> {value, description, context, role_id, created}
        """
        data = self._load()
        all_facts = data["facts"]

        if role_id is None:
            # Return only shared facts
            return {k: v for k, v in all_facts.items() if v.get("role_id") is None}

        # Return role-specific facts, optionally with shared
        result = {}
        for name, fact_data in all_facts.items():
            fact_role = fact_data.get("role_id")
            if fact_role == role_id:
                result[name] = fact_data
            elif include_shared and fact_role is None:
                result[name] = fact_data

        return result

    def list_all_facts(self) -> dict[str, dict]:
        """List all persistent facts regardless of role.

        Returns:
            Dict of name -> {value, description, context, role_id, created}
        """
        data = self._load()
        return data["facts"].copy()

    def get_shared_facts(self) -> dict[str, dict]:
        """List only shared facts (role_id=None).

        Returns:
            Dict of name -> {value, description, context, created}
        """
        return self.list_facts(role_id=None, include_shared=False)

    def get_role_facts(self, role_id: str) -> dict[str, dict]:
        """List only facts for a specific role (excludes shared).

        Args:
            role_id: Role to filter by

        Returns:
            Dict of name -> {value, description, context, role_id, created}
        """
        data = self._load()
        return {k: v for k, v in data["facts"].items() if v.get("role_id") == role_id}

    def promote_to_shared(self, name: str) -> bool:
        """Promote a role-scoped fact to shared context.

        Used when final results from a role should be available globally.

        Args:
            name: Fact name to promote

        Returns:
            True if promoted, False if not found
        """
        data = self._load()
        if name not in data["facts"]:
            return False

        data["facts"][name]["role_id"] = None
        self._save()
        return True

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

    def load_into_session(
        self,
        session: "Session",
        role_id: Optional[str] = None,
        include_shared: bool = True,
    ) -> int:
        """Load persistent facts into a session's fact resolver.

        Args:
            session: Session instance to load facts into
            role_id: Only load facts for this role (None = shared only)
            include_shared: If True and role_id is set, also load shared facts

        Returns:
            Number of facts loaded
        """
        import logging
        logger = logging.getLogger(__name__)

        persistent_facts = self.list_facts(role_id=role_id, include_shared=include_shared)
        logger.debug(f"[FactStore] Found {len(persistent_facts)} persistent facts to load (role={role_id})")

        if not persistent_facts:
            return 0

        loaded = 0
        for name, fact_data in persistent_facts.items():
            try:
                session.fact_resolver.add_user_fact(
                    fact_name=name,
                    value=fact_data.get("value"),
                    reasoning="Loaded from persistent storage",
                    description=fact_data.get("description", ""),
                    role_id=fact_data.get("role_id"),
                )
                loaded += 1
                logger.debug(f"[FactStore] Loaded fact: {name} = {fact_data.get('value')} (role={fact_data.get('role_id')})")
            except Exception as e:
                logger.warning(f"[FactStore] Failed to load fact {name}: {e}")

        logger.debug(f"[FactStore] Successfully loaded {loaded} facts into session")
        return loaded
