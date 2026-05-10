# Copyright (c) 2025 Kenneth Stott
# Canary: 6713c1ce-cbda-4f3b-a0bb-0bdc16ffc9a2
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Persistent fact storage with role provenance tracking.

Provides storage for facts that persist across sessions, backed by
DuckDB in the user vault.

All facts are globally accessible. The role_id field is metadata indicating
which role created the fact (provenance), not access control.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

from constat.storage.duckdb_pool import ThreadLocalDuckDB

logger = logging.getLogger(__name__)


class FactStore:
    """Manages persistent facts with role provenance tracking.

    Facts are backed by a DuckDB table in the user vault database.
    All facts are globally accessible regardless of role_id.
    The role_id is metadata for provenance and UI grouping.
    """

    _FACTS_DDL = """
    CREATE TABLE IF NOT EXISTS facts (
        name VARCHAR PRIMARY KEY,
        value TEXT NOT NULL,
        description TEXT NOT NULL DEFAULT '',
        context TEXT NOT NULL DEFAULT '',
        role_id VARCHAR,
        domain VARCHAR NOT NULL DEFAULT '',
        created TIMESTAMP NOT NULL
    )
    """

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        user_id: str = "default",
        db: Optional[ThreadLocalDuckDB] = None,
    ):
        """Initialize fact store.

        Args:
            base_dir: Base directory for .constat. Defaults to current directory.
            user_id: User ID for user-scoped storage.
            db: Existing ThreadLocalDuckDB connection to reuse. If None,
                opens a standalone connection to the user vault.
        """
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
            self._db = ThreadLocalDuckDB(
                str(db_path),
                init_sql=["INSTALL vss", "LOAD vss", "INSTALL fts", "LOAD fts"],
            )
        self._tables_ensured = False

    def _ensure_tables(self) -> None:
        """Create table if it doesn't exist (idempotent)."""
        if self._tables_ensured:
            return
        conn = self._db
        conn.execute(self._FACTS_DDL)
        self._tables_ensured = True
        self._import_yaml()

    def _import_yaml(self) -> None:
        """One-time import from legacy YAML file."""
        from constat.core.paths import user_vault_dir
        yaml_path = user_vault_dir(self.base_dir, self.user_id) / "facts.yaml"
        if not yaml_path.exists():
            return
        conn = self._db
        count = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        if count > 0:
            return
        try:
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f) or {}
            facts = data.get("facts", {})
            for name, entry in facts.items():
                conn.execute(
                    "INSERT OR IGNORE INTO facts (name, value, description, context, role_id, domain, created) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [
                        name,
                        json.dumps(entry.get("value")),
                        entry.get("description", ""),
                        entry.get("context", ""),
                        entry.get("role_id"),
                        entry.get("domain", ""),
                        entry.get("created", datetime.now(timezone.utc).isoformat()),
                    ],
                )
            imported = yaml_path.with_suffix(".yaml.imported")
            yaml_path.rename(imported)
            logger.info(f"Imported {len(facts)} facts from {yaml_path} → DuckDB")
        except Exception as e:
            logger.warning(f"Failed to import facts YAML: {e}")

    _FACT_COLS = ["name", "value", "description", "context", "role_id", "domain", "created"]

    def _row_to_fact(self, row: tuple, cols: list[str] | None = None) -> dict:
        d = dict(zip(cols or self._FACT_COLS, row))
        d["value"] = json.loads(d["value"]) if isinstance(d.get("value"), str) else d.get("value")
        return d

    def _fetch_facts(self, sql: str, params: list | None = None) -> list[tuple]:
        """Execute a SELECT and return rows."""
        return self._db.execute(sql, params).fetchall()

    def save_fact(
        self,
        name: str,
        value: Any,
        description: str = "",
        context: str = "",
        role_id: Optional[str] = None,
        domain: str = "",
    ) -> None:
        self._ensure_tables()
        conn = self._db
        # DuckDB doesn't have INSERT OR REPLACE — use DELETE + INSERT
        conn.execute("DELETE FROM facts WHERE name = ?", [name])
        conn.execute(
            "INSERT INTO facts (name, value, description, context, role_id, domain, created) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [name, json.dumps(value), description, context, role_id, domain, datetime.now(timezone.utc).isoformat()],
        )

    def get_fact(self, name: str) -> Optional[dict]:
        self._ensure_tables()
        rows = self._fetch_facts("SELECT * FROM facts WHERE name = ?", [name])
        if not rows:
            return None
        d = self._row_to_fact(rows[0])
        d.pop("name", None)
        return d

    def list_facts(self, role_id: Optional[str] = None, include_shared: bool = True) -> dict[str, dict]:
        self._ensure_tables()
        if role_id is None:
            rows = self._fetch_facts("SELECT * FROM facts WHERE role_id IS NULL")
        elif include_shared:
            rows = self._fetch_facts("SELECT * FROM facts WHERE role_id = ? OR role_id IS NULL", [role_id])
        else:
            rows = self._fetch_facts("SELECT * FROM facts WHERE role_id = ?", [role_id])
        result = {}
        for row in rows:
            d = self._row_to_fact(row)
            name = d.pop("name")
            result[name] = d
        return result

    def list_all_facts(self) -> dict[str, dict]:
        self._ensure_tables()
        rows = self._fetch_facts("SELECT * FROM facts")
        result = {}
        for row in rows:
            d = self._row_to_fact(row)
            name = d.pop("name")
            result[name] = d
        return result

    def get_shared_facts(self) -> dict[str, dict]:
        return self.list_facts(role_id=None, include_shared=False)

    def get_role_facts(self, role_id: str) -> dict[str, dict]:
        self._ensure_tables()
        rows = self._fetch_facts("SELECT * FROM facts WHERE role_id = ?", [role_id])
        result = {}
        for row in rows:
            d = self._row_to_fact(row)
            name = d.pop("name")
            result[name] = d
        return result

    def promote_to_shared(self, name: str) -> bool:
        self._ensure_tables()
        conn = self._db
        row = conn.execute("SELECT name FROM facts WHERE name = ?", [name]).fetchone()
        if row is None:
            return False
        conn.execute("UPDATE facts SET role_id = NULL WHERE name = ?", [name])
        return True

    def move_fact(self, name: str, to_domain: str) -> bool:
        self._ensure_tables()
        conn = self._db
        row = conn.execute("SELECT name FROM facts WHERE name = ?", [name]).fetchone()
        if row is None:
            return False
        conn.execute("UPDATE facts SET domain = ? WHERE name = ?", [to_domain, name])
        return True

    def delete_fact(self, name: str) -> bool:
        self._ensure_tables()
        conn = self._db
        row = conn.execute("SELECT name FROM facts WHERE name = ?", [name]).fetchone()
        if row is None:
            return False
        conn.execute("DELETE FROM facts WHERE name = ?", [name])
        return True

    def clear_all(self) -> int:
        self._ensure_tables()
        conn = self._db
        count = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        conn.execute("DELETE FROM facts")
        return count

    def load_into_session(
        self,
        session: "Session",
        role_id: Optional[str] = None,
        include_shared: bool = True,
    ) -> int:
        """Load persistent facts into a session's fact resolver."""
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
            except Exception as e:
                logger.warning(f"[FactStore] Failed to load fact {name}: {e}")

        logger.debug(f"[FactStore] Successfully loaded {loaded} facts into session")
        return loaded
