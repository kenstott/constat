# Copyright (c) 2025 Kenneth Stott
# Canary: 52c82f93-595f-40b9-8430-c218ad3eb493
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Split vector store managing user DB (main) + system DB (ATTACHed as sys).

Creates TEMP UNION ALL views (v_*) over both schemas so reads transparently
merge user and system data. Writes are routed via _write_schema().
"""

import logging
from pathlib import Path

from constat.storage.duckdb_pool import ThreadLocalDuckDB

logger = logging.getLogger(__name__)

# Tables that exist in both user and system DBs
SPLIT_TABLES = [
    "entities",
    "embeddings",
    "glossary_terms",
    "entity_relationships",
    "chunk_entities",
    "source_hashes",
    "resource_hashes",
    "document_urls",
    "entity_resolution_names",
    "glossary_clusters",
    "proven_grounding",
    "data_sources",
]


class SplitVectorStore:
    """Manages two DuckDB databases with ATTACH + UNION views.

    user_db_path: writable per-user database (main schema)
    system_db_path: shared system database (ATTACHed as 'sys')

    In warmup mode (system_db_path is None), only the user DB is used
    and no ATTACH or views are created.
    """

    def __init__(
        self,
        user_db_path: str | Path,
        system_db_path: str | Path | None = None,
        init_sql: list[str] | None = None,
    ):
        self._user_db_path = Path(user_db_path)
        self._system_db_path = Path(system_db_path) if system_db_path else None
        self._warmup = system_db_path is None

        self._db = ThreadLocalDuckDB(
            str(self._user_db_path),
            init_sql=init_sql or [],
        )

        if not self._warmup:
            self._attach_system_db()
            # Views are created by create_union_views() AFTER _init_schema()
            # populates the user DB tables. Caller must invoke explicitly.

    @property
    def db(self) -> ThreadLocalDuckDB:
        return self._db

    @property
    def warmup(self) -> bool:
        return self._warmup

    def _attach_system_db(self) -> None:
        """ATTACH system DB as 'sys'.

        Tries read-write first, falls back to read-only on file handle
        conflicts (DuckDB enforces process-wide unique file handles for
        read-write but permits concurrent read-only ATTACHes).
        """
        attached = [
            row[0]
            for row in self._db.conn.execute(
                "SELECT database_name FROM duckdb_databases()"
            ).fetchall()
        ]
        if "sys" in attached:
            return
        try:
            self._db.conn.execute(
                f"ATTACH '{self._system_db_path}' AS sys"
            )
        except Exception as e:
            if "already attached" in str(e):
                return
            if "file handle conflict" in str(e).lower():
                self._db.conn.execute(
                    f"ATTACH '{self._system_db_path}' AS sys (READ_ONLY)"
                )
                return
            raise

    def _create_union_views(self) -> None:
        """Create TEMP UNION ALL views over main and sys schemas.

        Each view includes a _source_schema column ('main' or 'sys') so
        downstream code can determine where a row is stored.

        Requires all SPLIT_TABLES to exist in both main and sys schemas.
        Use DuckDBVectorStore._ensure_sys_tables() before calling this.
        """
        conn = self._db.conn
        for table in SPLIT_TABLES:
            conn.execute(
                f"CREATE OR REPLACE TEMP VIEW v_{table} AS "
                f"SELECT *, 'main' AS _source_schema FROM main.{table} "
                f"UNION ALL "
                f"SELECT *, 'sys' AS _source_schema FROM sys.{table}"
            )

        # Composite views over union base views
        conn.execute("""
            CREATE OR REPLACE TEMP VIEW unified_glossary AS
            SELECT
                e.id AS entity_id,
                e.name,
                COALESCE(g.display_name, e.display_name) AS display_name,
                e.semantic_type,
                e.ner_type,
                e.session_id,
                g.id AS glossary_id,
                g.domain,
                g.definition,
                g.parent_id,
                g.parent_verb,
                g.aliases,
                g.cardinality,
                g.plural,
                g.status,
                g.provenance,
                CASE
                    WHEN g.id IS NOT NULL THEN 'defined'
                    ELSE 'self_describing'
                END AS glossary_status
            FROM v_entities e
            LEFT JOIN v_glossary_terms g
                ON e.name = g.name
                AND e.session_id = g.session_id
        """)

        conn.execute("""
            CREATE OR REPLACE TEMP VIEW deprecated_glossary AS
            SELECT g.*
            FROM v_glossary_terms g
            LEFT JOIN v_entities e
                ON g.name = e.name
                AND g.session_id = e.session_id
            WHERE e.id IS NULL
        """)

    def _write_schema(self, domain_tier: str) -> str:
        """Return the schema name for writes based on domain tier.

        'system' tier writes to 'sys', everything else writes to 'main'.
        In warmup mode, always returns 'main'.
        """
        if self._warmup:
            return "main"
        if domain_tier == "system":
            return "sys"
        return "main"

    def close(self) -> None:
        """Detach system DB and close user DB."""
        if not self._warmup:
            try:
                self._db.conn.execute("DETACH sys")
            except Exception:
                pass
        self._db.close()

    def __enter__(self) -> "SplitVectorStore":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
