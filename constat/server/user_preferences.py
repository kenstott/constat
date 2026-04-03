# Copyright (c) 2025 Kenneth Stott
# Canary: ff656c13-17fb-4c51-89cb-8240f2afb329
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""User preferences management backed by DuckDB.

Stores user preferences in the user vault DuckDB database.
- selected_domains: List of domain IDs to auto-select on session create
- Other preferences can be added here in the future
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

import yaml

from constat.storage.duckdb_pool import ThreadLocalDuckDB

logger = logging.getLogger(__name__)

# Default preferences for new users
DEFAULT_PREFERENCES = {
    "selected_domains": [],
    "brief_mode": False,
    "default_mode": "exploratory",
}


class PreferencesStore:
    """User preferences storage backed by DuckDB."""

    _DDL = """
    CREATE TABLE IF NOT EXISTS preferences (
        user_id VARCHAR PRIMARY KEY,
        prefs TEXT NOT NULL
    )
    """

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        user_id: str = "default",
        db: Optional[ThreadLocalDuckDB] = None,
    ):
        self.base_dir = Path(base_dir) if base_dir else Path(".constat")
        self.user_id = user_id
        self._owns_db = db is None
        if db is not None:
            self._db = db
        else:
            from constat.core.paths import user_vault_dir, migrate_db_name
            vault = user_vault_dir(self.base_dir, user_id)
            vault.mkdir(parents=True, exist_ok=True)
            db_path = migrate_db_name(vault, "vectors.duckdb", "user.duckdb")
            self._db = ThreadLocalDuckDB(str(db_path))
        self._tables_ensured = False

    def _ensure_tables(self) -> None:
        if self._tables_ensured:
            return
        self._db.execute(self._DDL)
        self._tables_ensured = True
        self._import_yaml()

    def _import_yaml(self) -> None:
        """One-time import from legacy preferences.yaml."""
        from constat.core.paths import user_vault_dir
        yaml_path = user_vault_dir(self.base_dir, self.user_id) / "preferences.yaml"
        if not yaml_path.exists():
            return
        count = self._db.execute(
            "SELECT COUNT(*) FROM preferences WHERE user_id = ?",
            [self.user_id],
        ).fetchone()[0]
        if count > 0:
            return
        with open(yaml_path, "r") as f:
            prefs = yaml.safe_load(f) or {}
        merged = DEFAULT_PREFERENCES.copy()
        merged.update(prefs)
        self._db.execute(
            "INSERT INTO preferences (user_id, prefs) VALUES (?, ?)",
            [self.user_id, json.dumps(merged)],
        )
        yaml_path.rename(yaml_path.with_suffix(".yaml.imported"))
        logger.info("Imported preferences from %s", yaml_path)

    def load(self) -> dict[str, Any]:
        """Load user preferences.

        Returns:
            Dictionary of user preferences, with defaults for missing keys.
        """
        self._ensure_tables()
        row = self._db.execute(
            "SELECT prefs FROM preferences WHERE user_id = ?",
            [self.user_id],
        ).fetchone()
        if row is None:
            return DEFAULT_PREFERENCES.copy()
        result = DEFAULT_PREFERENCES.copy()
        result.update(json.loads(row[0]))
        return result

    def save(self, preferences: dict[str, Any]) -> bool:
        """Save user preferences.

        Args:
            preferences: Dictionary of preferences to save.

        Returns:
            True if saved successfully.
        """
        self._ensure_tables()
        self._db.execute(
            """INSERT INTO preferences (user_id, prefs) VALUES (?, ?)
               ON CONFLICT (user_id) DO UPDATE SET prefs = excluded.prefs""",
            [self.user_id, json.dumps(preferences)],
        )
        return True


# ---------------------------------------------------------------------------
# Module-level backward-compatible wrappers
# ---------------------------------------------------------------------------

def load_user_preferences(
    user_id: str,
    base_dir: Optional[Path] = None,
    db: Optional[ThreadLocalDuckDB] = None,
) -> dict[str, Any]:
    """Load user preferences from DuckDB.

    Args:
        user_id: The user's ID
        base_dir: Optional base directory override
        db: Optional existing DuckDB connection

    Returns:
        Dictionary of user preferences, with defaults for missing keys
    """
    store = PreferencesStore(base_dir=base_dir, user_id=user_id, db=db)
    return store.load()


def save_user_preferences(
    user_id: str,
    preferences: dict[str, Any],
    base_dir: Optional[Path] = None,
    db: Optional[ThreadLocalDuckDB] = None,
) -> bool:
    """Save user preferences to DuckDB.

    Args:
        user_id: The user's ID
        preferences: Dictionary of preferences to save
        base_dir: Optional base directory override
        db: Optional existing DuckDB connection

    Returns:
        True if saved successfully
    """
    store = PreferencesStore(base_dir=base_dir, user_id=user_id, db=db)
    return store.save(preferences)


def get_selected_domains(
    user_id: str,
    base_dir: Optional[Path] = None,
    db: Optional[ThreadLocalDuckDB] = None,
) -> list[str]:
    """Get the user's selected domains.

    Args:
        user_id: The user's ID

    Returns:
        List of domain IDs that should be pre-selected
    """
    prefs = load_user_preferences(user_id, base_dir=base_dir, db=db)
    return prefs.get("selected_domains", [])


# Backwards compatibility alias
get_selected_projects = get_selected_domains


def set_selected_domains(
    user_id: str,
    domain_ids: list[str],
    base_dir: Optional[Path] = None,
    db: Optional[ThreadLocalDuckDB] = None,
) -> bool:
    """Set the user's selected domains.

    Args:
        user_id: The user's ID
        domain_ids: List of domain IDs to save as selected

    Returns:
        True if saved successfully
    """
    store = PreferencesStore(base_dir=base_dir, user_id=user_id, db=db)
    prefs = store.load()
    prefs["selected_domains"] = domain_ids
    return store.save(prefs)


# Backwards compatibility alias
set_selected_projects = set_selected_domains


def update_preference(
    user_id: str,
    key: str,
    value: Any,
    base_dir: Optional[Path] = None,
    db: Optional[ThreadLocalDuckDB] = None,
) -> bool:
    """Update a single preference value.

    Args:
        user_id: The user's ID
        key: Preference key to update
        value: New value

    Returns:
        True if saved successfully
    """
    store = PreferencesStore(base_dir=base_dir, user_id=user_id, db=db)
    prefs = store.load()
    prefs[key] = value
    return store.save(prefs)


def get_preference(
    user_id: str,
    key: str,
    default: Any = None,
    base_dir: Optional[Path] = None,
    db: Optional[ThreadLocalDuckDB] = None,
) -> Any:
    """Get a single preference value.

    Args:
        user_id: The user's ID
        key: Preference key to get
        default: Default value if key not found

    Returns:
        The preference value, or default if not found
    """
    prefs = load_user_preferences(user_id, base_dir=base_dir, db=db)
    return prefs.get(key, default)
