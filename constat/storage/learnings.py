# Copyright (c) 2025 Kenneth Stott
# Canary: 3e4289e5-6d91-4419-b67a-d3cd25e9dad2
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Two-tier learning storage for corrections and patterns.

Provides storage for learnings (raw corrections) and rules (compacted patterns)
that persist across sessions, backed by DuckDB in the user vault.
"""

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml

from constat.storage.duckdb_pool import ThreadLocalDuckDB

logger = logging.getLogger(__name__)


class LearningCategory(Enum):
    """Categories of learnings.

    Categories:
    - USER_CORRECTION: User manually corrected an output
    - CODEGEN_ERROR: General code generation error (syntax, logic, column names, etc.)
    - EXTERNAL_API_ERROR: Error in code calling external REST/GraphQL APIs
    - HTTP_ERROR: HTTP 4xx/5xx errors from external API calls
    - NL_CORRECTION: Natural language interpretation correction
    - API_ERROR: (deprecated, alias for EXTERNAL_API_ERROR for backward compatibility)
    """
    USER_CORRECTION = "user_correction"
    API_ERROR = "api_error"  # Deprecated: kept for backward compatibility
    EXTERNAL_API_ERROR = "external_api_error"  # Clearer name for API integration errors
    HTTP_ERROR = "http_error"  # 4xx/5xx errors from external APIs
    CODEGEN_ERROR = "codegen_error"
    NL_CORRECTION = "nl_correction"
    GLOSSARY_REFINEMENT = "glossary_refinement"


class LearningSource(Enum):
    """How a learning was captured."""
    AUTO_CAPTURE = "auto_capture"
    EXPLICIT_COMMAND = "explicit_command"
    NL_DETECTION = "nl_detection"


def _get_scope_types(rule: dict) -> set[str]:
    """Extract data source types from a rule's scope."""
    scope = rule.get("scope")
    if not scope:
        return set()
    sources = scope.get("data_sources", [])
    return {s.get("type", "") for s in sources if s.get("type")}


def _json_loads(val: Any) -> Any:
    """Parse a JSON string, returning None for NULL/empty."""
    if val is None:
        return None
    if isinstance(val, str):
        return json.loads(val)
    return val


class LearningStore:
    """Two-tier learning storage: raw corrections + compacted rules.

    Backed by DuckDB tables in the user vault database.
    """

    _LEARNINGS_DDL = """
    CREATE TABLE IF NOT EXISTS learnings (
        id VARCHAR PRIMARY KEY,
        category VARCHAR NOT NULL,
        created TIMESTAMP NOT NULL,
        context TEXT,
        correction TEXT NOT NULL,
        source VARCHAR NOT NULL,
        applied_count INTEGER NOT NULL DEFAULT 0,
        promoted_to VARCHAR,
        scope TEXT,
        domain VARCHAR NOT NULL DEFAULT '',
        archived_at TIMESTAMP
    )
    """

    _RULES_DDL = """
    CREATE TABLE IF NOT EXISTS learning_rules (
        id VARCHAR PRIMARY KEY,
        category VARCHAR NOT NULL,
        summary TEXT NOT NULL,
        confidence FLOAT NOT NULL,
        source_learnings TEXT,
        tags TEXT,
        applied_count INTEGER NOT NULL DEFAULT 0,
        created TIMESTAMP NOT NULL,
        updated_at TIMESTAMP,
        domain VARCHAR NOT NULL DEFAULT '',
        scope TEXT
    )
    """

    _EXEMPLAR_DDL = """
    CREATE TABLE IF NOT EXISTS exemplar_runs (
        id VARCHAR PRIMARY KEY,
        timestamp TIMESTAMP NOT NULL,
        stats TEXT NOT NULL
    )
    """

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        user_id: str = "default",
        db: Optional[ThreadLocalDuckDB] = None,
    ):
        """Initialize learning store.

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
        """Create tables if they don't exist (idempotent)."""
        if self._tables_ensured:
            return
        conn = self._db
        conn.execute(self._LEARNINGS_DDL)
        conn.execute(self._RULES_DDL)
        conn.execute(self._EXEMPLAR_DDL)
        self._tables_ensured = True
        self._import_yaml()

    def _import_yaml(self) -> None:
        """One-time import from legacy YAML file."""
        from constat.core.paths import user_vault_dir
        yaml_path = user_vault_dir(self.base_dir, self.user_id) / "learnings.yaml"
        if not yaml_path.exists():
            return
        # Check if tables already have data
        conn = self._db
        count = conn.execute("SELECT COUNT(*) FROM learnings").fetchone()[0]
        if count > 0:
            return
        try:
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f) or {}

            # Handle old list-based format
            corrections = data.get("corrections", data.get("raw_learnings", {}))
            if isinstance(corrections, list):
                corrections = {item["id"]: {k: v for k, v in item.items() if k != "id"} for item in corrections if "id" in item}
            rules = data.get("rules", {})
            if isinstance(rules, list):
                rules = {item["id"]: {k: v for k, v in item.items() if k != "id"} for item in rules if "id" in item}
            archive = data.get("archive", {})
            if isinstance(archive, list):
                archive = {item["id"]: {k: v for k, v in item.items() if k != "id"} for item in archive if "id" in item}

            for lid, entry in corrections.items():
                self._insert_learning(lid, entry)
            for lid, entry in archive.items():
                self._insert_learning(lid, entry, archived=True)
            for rid, entry in rules.items():
                self._insert_rule(rid, entry)

            # Import exemplar runs
            for run in data.get("exemplar_runs", []):
                run_id = run.get("id", self._generate_id("exrun"))
                ts = run.get("timestamp", datetime.now(timezone.utc).isoformat())
                stats = {k: v for k, v in run.items() if k not in ("id", "timestamp")}
                conn.execute(
                    "INSERT OR IGNORE INTO exemplar_runs (id, timestamp, stats) VALUES (?, ?, ?)",
                    [run_id, ts, json.dumps(stats)],
                )

            imported = yaml_path.with_suffix(".yaml.imported")
            yaml_path.rename(imported)
            logger.info(f"Imported learnings from {yaml_path} → DuckDB ({len(corrections)} corrections, {len(rules)} rules, {len(archive)} archived)")
        except Exception as e:
            logger.warning(f"Failed to import learnings YAML: {e}")

    def _insert_learning(self, lid: str, entry: dict, archived: bool = False) -> None:
        conn = self._db
        conn.execute(
            "INSERT OR IGNORE INTO learnings (id, category, created, context, correction, source, applied_count, promoted_to, scope, domain, archived_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                lid,
                entry.get("category", ""),
                entry.get("created", datetime.now(timezone.utc).isoformat()),
                json.dumps(entry.get("context")) if entry.get("context") else None,
                entry.get("correction", ""),
                entry.get("source", "auto_capture"),
                entry.get("applied_count", 0),
                entry.get("promoted_to"),
                json.dumps(entry.get("scope")) if entry.get("scope") else None,
                entry.get("domain", ""),
                entry.get("archived_at") if archived or entry.get("archived_at") else None,
            ],
        )

    def _insert_rule(self, rid: str, entry: dict) -> None:
        conn = self._db
        conn.execute(
            "INSERT OR IGNORE INTO learning_rules (id, category, summary, confidence, source_learnings, tags, applied_count, created, updated_at, domain, scope) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                rid,
                entry.get("category", ""),
                entry.get("summary", ""),
                entry.get("confidence", 0.0),
                json.dumps(entry.get("source_learnings")) if entry.get("source_learnings") else None,
                json.dumps(entry.get("tags")) if entry.get("tags") else None,
                entry.get("applied_count", 0),
                entry.get("created", datetime.now(timezone.utc).isoformat()),
                entry.get("updated_at"),
                entry.get("domain", ""),
                json.dumps(entry.get("scope")) if entry.get("scope") else None,
            ],
        )

    @staticmethod
    def _generate_id(prefix: str = "learn") -> str:
        """Generate a unique ID."""
        return f"{prefix}_{uuid.uuid4().hex[:8]}"

    def _row_to_learning(self, row: tuple, cols: list[str]) -> dict:
        d = dict(zip(cols, row))
        d["context"] = _json_loads(d.get("context"))
        d["scope"] = _json_loads(d.get("scope"))
        return d

    def _row_to_rule(self, row: tuple, cols: list[str]) -> dict:
        d = dict(zip(cols, row))
        d["source_learnings"] = _json_loads(d.get("source_learnings")) or []
        d["tags"] = _json_loads(d.get("tags")) or []
        d["scope"] = _json_loads(d.get("scope"))
        return d

    # -------------------------------------------------------------------------
    # Raw Learnings / Corrections (Tier 1)
    # -------------------------------------------------------------------------

    def save_learning(
        self,
        category: LearningCategory,
        context: dict,
        correction: str,
        source: LearningSource = LearningSource.AUTO_CAPTURE,
        scope: dict | None = None,
    ) -> str:
        self._ensure_tables()
        learning_id = self._generate_id("learn")
        conn = self._db
        conn.execute(
            "INSERT INTO learnings (id, category, created, context, correction, source, applied_count, scope) VALUES (?, ?, ?, ?, ?, ?, 0, ?)",
            [
                learning_id,
                category.value,
                datetime.now(timezone.utc).isoformat(),
                json.dumps(context) if context else None,
                correction,
                source.value,
                json.dumps(scope) if scope else None,
            ],
        )
        return learning_id

    def get_learning(self, learning_id: str) -> Optional[dict]:
        self._ensure_tables()
        conn = self._db
        cursor = conn.execute("SELECT * FROM learnings WHERE id = ?", [learning_id])
        cols = [desc[0] for desc in cursor.description]
        result = cursor.fetchone()
        if result is None:
            return None
        return self._row_to_learning(result, cols)

    def update_learning_context(self, learning_id: str, updates: dict) -> bool:
        self._ensure_tables()
        conn = self._db
        row = conn.execute("SELECT context FROM learnings WHERE id = ? AND archived_at IS NULL", [learning_id]).fetchone()
        if row is None:
            return False
        ctx = _json_loads(row[0]) or {}
        if isinstance(ctx, dict):
            ctx.update(updates)
        else:
            ctx = dict(updates)
        conn.execute("UPDATE learnings SET context = ? WHERE id = ?", [json.dumps(ctx), learning_id])
        return True

    def list_raw_learnings(
        self,
        category: Optional[LearningCategory] = None,
        limit: Optional[int] = 50,
        include_promoted: bool = False,
        model_family: str | None = None,
    ) -> list[dict]:
        self._ensure_tables()
        conn = self._db
        conditions = ["archived_at IS NULL"]
        params: list[Any] = []
        if category:
            conditions.append("category = ?")
            params.append(category.value)
        if not include_promoted:
            conditions.append("promoted_to IS NULL")
        where = " AND ".join(conditions)
        sql = f"SELECT * FROM learnings WHERE {where} ORDER BY created DESC"
        if limit:
            sql += f" LIMIT {limit}"
        cursor = conn.execute(sql, params)
        cols = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        learnings = [self._row_to_learning(r, cols) for r in rows]
        if model_family:
            learnings = [
                l for l in learnings
                if not (l.get("context") or {}).get("error_provider")
                or l["context"]["error_provider"] == model_family
            ]
        return learnings

    def delete_learning(self, learning_id: str) -> bool:
        self._ensure_tables()
        conn = self._db
        result = conn.execute("DELETE FROM learnings WHERE id = ? AND archived_at IS NULL", [learning_id])
        return result.fetchone() is not None if hasattr(result, 'fetchone') else True

    def increment_applied(self, learning_id: str) -> None:
        self._ensure_tables()
        conn = self._db
        conn.execute("UPDATE learnings SET applied_count = applied_count + 1 WHERE id = ?", [learning_id])

    # -------------------------------------------------------------------------
    # Rules (Tier 2)
    # -------------------------------------------------------------------------

    def save_rule(
        self,
        summary: str,
        category: LearningCategory,
        confidence: float,
        source_learnings: list[str],
        tags: Optional[list[str]] = None,
        domain: str = "",
        scope: dict | None = None,
    ) -> str:
        self._ensure_tables()
        rule_id = self._generate_id("rule")
        conn = self._db
        conn.execute(
            "INSERT INTO learning_rules (id, category, summary, confidence, source_learnings, tags, applied_count, created, domain, scope) VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?)",
            [
                rule_id,
                category.value,
                summary,
                confidence,
                json.dumps(source_learnings),
                json.dumps(tags or []),
                datetime.now(timezone.utc).isoformat(),
                domain,
                json.dumps(scope) if scope else None,
            ],
        )
        return rule_id

    def list_rules(
        self,
        category: Optional[LearningCategory] = None,
        min_confidence: float = 0.0,
        limit: Optional[int] = None,
        domain: Optional[str] = None,
        scope_filter: list[str] | None = None,
    ) -> list[dict]:
        self._ensure_tables()
        conn = self._db
        conditions = ["confidence >= ?"]
        params: list[Any] = [min_confidence]
        if category:
            conditions.append("category = ?")
            params.append(category.value)
        if domain is not None:
            conditions.append("domain = ?")
            params.append(domain)
        where = " AND ".join(conditions)
        sql = f"SELECT * FROM learning_rules WHERE {where} ORDER BY confidence DESC"
        if limit is not None:
            sql += f" LIMIT {limit}"
        cursor = conn.execute(sql, params)
        cols = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        rules = [self._row_to_rule(r, cols) for r in rows]

        if scope_filter is not None:
            filter_set = set(scope_filter)
            rules = [r for r in rules if not _get_scope_types(r) or _get_scope_types(r) & filter_set]

        return rules

    def get_relevant_rules(
        self,
        context: str,
        min_confidence: float = 0.6,
        limit: int = 5,
    ) -> list[dict]:
        rules = self.list_rules(min_confidence=min_confidence)
        if not rules:
            return []
        context_lower = context.lower()
        context_words = set(re.findall(r'\w+', context_lower))
        scored_rules = []
        for rule in rules:
            summary_words = set(re.findall(r'\w+', rule["summary"].lower()))
            tag_words = set(t.lower() for t in rule.get("tags", []))
            all_rule_words = summary_words | tag_words
            overlap = len(context_words & all_rule_words)
            if overlap > 0:
                score = overlap + rule.get("confidence", 0) + (rule.get("applied_count", 0) * 0.1)
                scored_rules.append((score, rule))
        scored_rules.sort(key=lambda x: x[0], reverse=True)
        return [r[1] for r in scored_rules[:limit]]

    def increment_rule_applied(self, rule_id: str) -> None:
        self._ensure_tables()
        conn = self._db
        conn.execute("UPDATE learning_rules SET applied_count = applied_count + 1 WHERE id = ?", [rule_id])

    def update_rule(
        self,
        rule_id: str,
        summary: Optional[str] = None,
        tags: Optional[list[str]] = None,
        confidence: Optional[float] = None,
    ) -> bool:
        self._ensure_tables()
        conn = self._db
        row = conn.execute("SELECT id FROM learning_rules WHERE id = ?", [rule_id]).fetchone()
        if row is None:
            return False
        updates: list[str] = []
        params: list[Any] = []
        if summary is not None:
            updates.append("summary = ?")
            params.append(summary)
        if tags is not None:
            updates.append("tags = ?")
            params.append(json.dumps(tags))
        if confidence is not None:
            updates.append("confidence = ?")
            params.append(confidence)
        updates.append("updated_at = ?")
        params.append(datetime.now(timezone.utc).isoformat())
        params.append(rule_id)
        conn.execute(f"UPDATE learning_rules SET {', '.join(updates)} WHERE id = ?", params)
        return True

    def delete_rule(self, rule_id: str) -> bool:
        self._ensure_tables()
        conn = self._db
        before = conn.execute("SELECT COUNT(*) FROM learning_rules WHERE id = ?", [rule_id]).fetchone()[0]
        if before == 0:
            return False
        conn.execute("DELETE FROM learning_rules WHERE id = ?", [rule_id])
        return True

    # -------------------------------------------------------------------------
    # Archive
    # -------------------------------------------------------------------------

    def archive_learning(self, learning_id: str, rule_id: str) -> bool:
        self._ensure_tables()
        conn = self._db
        row = conn.execute("SELECT id FROM learnings WHERE id = ? AND archived_at IS NULL", [learning_id]).fetchone()
        if row is None:
            return False
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE learnings SET promoted_to = ?, archived_at = ? WHERE id = ?",
            [rule_id, now, learning_id],
        )
        return True

    def list_archive(self, limit: int = 50) -> list[dict]:
        self._ensure_tables()
        conn = self._db
        cursor = conn.execute(
            "SELECT * FROM learnings WHERE archived_at IS NOT NULL ORDER BY archived_at DESC LIMIT ?",
            [limit],
        )
        cols = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        return [self._row_to_learning(r, cols) for r in rows]

    # -------------------------------------------------------------------------
    # Stats
    # -------------------------------------------------------------------------

    def get_stats(self) -> dict:
        self._ensure_tables()
        conn = self._db
        corrections_total = conn.execute("SELECT COUNT(*) FROM learnings WHERE archived_at IS NULL").fetchone()[0]
        rules_total = conn.execute("SELECT COUNT(*) FROM learning_rules").fetchone()[0]
        archive_total = conn.execute("SELECT COUNT(*) FROM learnings WHERE archived_at IS NOT NULL").fetchone()[0]
        unpromoted = conn.execute("SELECT COUNT(*) FROM learnings WHERE archived_at IS NULL AND promoted_to IS NULL").fetchone()[0]

        raw_by_cat = {}
        for row in conn.execute("SELECT category, COUNT(*) FROM learnings WHERE archived_at IS NULL GROUP BY category").fetchall():
            raw_by_cat[row[0]] = row[1]
        rules_by_cat = {}
        for row in conn.execute("SELECT category, COUNT(*) FROM learning_rules GROUP BY category").fetchall():
            rules_by_cat[row[0]] = row[1]

        return {
            "total_raw": corrections_total,
            "total_rules": rules_total,
            "total_archived": archive_total,
            "unpromoted": unpromoted,
            "raw_by_category": raw_by_cat,
            "rules_by_category": rules_by_cat,
        }

    # -------------------------------------------------------------------------
    # Exemplar Runs
    # -------------------------------------------------------------------------

    def save_exemplar_run(self, stats: dict) -> str:
        self._ensure_tables()
        run_id = self._generate_id("exrun")
        conn = self._db
        conn.execute(
            "INSERT INTO exemplar_runs (id, timestamp, stats) VALUES (?, ?, ?)",
            [run_id, datetime.now(timezone.utc).isoformat(), json.dumps(stats)],
        )
        return run_id

    def get_exemplar_runs(self) -> list[dict]:
        self._ensure_tables()
        conn = self._db
        rows = conn.execute("SELECT * FROM exemplar_runs ORDER BY timestamp DESC").fetchall()
        result = []
        for row in rows:
            entry = {"id": row[0], "timestamp": str(row[1])}
            entry.update(_json_loads(row[2]) or {})
            result.append(entry)
        return result

    def clear_all(self) -> dict:
        self._ensure_tables()
        conn = self._db
        counts = {
            "corrections": conn.execute("SELECT COUNT(*) FROM learnings WHERE archived_at IS NULL").fetchone()[0],
            "rules": conn.execute("SELECT COUNT(*) FROM learning_rules").fetchone()[0],
            "archive": conn.execute("SELECT COUNT(*) FROM learnings WHERE archived_at IS NOT NULL").fetchone()[0],
        }
        conn.execute("DELETE FROM learnings")
        conn.execute("DELETE FROM learning_rules")
        return counts
