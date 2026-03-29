# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Tests for SplitVectorStore — ATTACH + UNION views across user/system DBs."""

import duckdb
import pytest
from pathlib import Path

from constat.storage.split_store import SplitVectorStore, SPLIT_TABLES

EMBEDDING_DIM = 1024

# Minimal schema DDL for each table (matches vector_store._init_schema)
_SCHEMA_DDL = {
    "embeddings": f"""
        CREATE TABLE IF NOT EXISTS embeddings (
            chunk_id VARCHAR PRIMARY KEY,
            document_name VARCHAR NOT NULL,
            source VARCHAR NOT NULL DEFAULT 'document',
            chunk_type VARCHAR NOT NULL DEFAULT 'document',
            section VARCHAR,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            embedding FLOAT[{EMBEDDING_DIM}] NOT NULL,
            session_id VARCHAR,
            domain_id VARCHAR,
            entity_class VARCHAR DEFAULT 'mixed',
            source_offset INTEGER,
            source_length INTEGER
        )
    """,
    "entities": """
        CREATE TABLE IF NOT EXISTS entities (
            id VARCHAR PRIMARY KEY,
            name VARCHAR NOT NULL,
            display_name VARCHAR NOT NULL,
            semantic_type VARCHAR NOT NULL,
            ner_type VARCHAR,
            session_id VARCHAR NOT NULL,
            domain_id VARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            entity_class VARCHAR DEFAULT 'metadata'
        )
    """,
    "glossary_terms": """
        CREATE TABLE IF NOT EXISTS glossary_terms (
            id VARCHAR PRIMARY KEY,
            name VARCHAR NOT NULL,
            display_name VARCHAR NOT NULL,
            definition TEXT NOT NULL,
            domain VARCHAR,
            parent_id VARCHAR,
            parent_verb VARCHAR DEFAULT 'HAS_KIND',
            aliases TEXT,
            semantic_type VARCHAR,
            cardinality VARCHAR DEFAULT 'many',
            plural VARCHAR,
            tags TEXT,
            owner VARCHAR,
            status VARCHAR DEFAULT 'draft',
            provenance VARCHAR DEFAULT 'llm',
            session_id VARCHAR NOT NULL,
            user_id VARCHAR NOT NULL DEFAULT 'default',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ignored BOOLEAN DEFAULT FALSE,
            canonical_source VARCHAR
        )
    """,
    "entity_relationships": """
        CREATE TABLE IF NOT EXISTS entity_relationships (
            id VARCHAR PRIMARY KEY,
            subject_name VARCHAR NOT NULL,
            verb VARCHAR NOT NULL,
            object_name VARCHAR NOT NULL,
            sentence TEXT,
            confidence FLOAT DEFAULT 1.0,
            verb_category VARCHAR DEFAULT 'other',
            session_id VARCHAR,
            user_edited BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(subject_name, verb, object_name, session_id)
        )
    """,
    "chunk_entities": """
        CREATE TABLE IF NOT EXISTS chunk_entities (
            chunk_id VARCHAR NOT NULL,
            entity_id VARCHAR NOT NULL,
            confidence FLOAT DEFAULT 1.0,
            PRIMARY KEY (chunk_id, entity_id)
        )
    """,
    "source_hashes": """
        CREATE TABLE IF NOT EXISTS source_hashes (
            source_id VARCHAR PRIMARY KEY,
            db_hash VARCHAR,
            api_hash VARCHAR,
            doc_hash VARCHAR,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            er_hash VARCHAR
        )
    """,
    "resource_hashes": """
        CREATE TABLE IF NOT EXISTS resource_hashes (
            resource_id VARCHAR PRIMARY KEY,
            resource_type VARCHAR NOT NULL,
            resource_name VARCHAR NOT NULL,
            source_id VARCHAR NOT NULL,
            content_hash VARCHAR NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """,
    "document_urls": """
        CREATE TABLE IF NOT EXISTS document_urls (
            document_name VARCHAR PRIMARY KEY,
            source_url VARCHAR NOT NULL
        )
    """,
    "entity_resolution_names": """
        CREATE TABLE IF NOT EXISTS entity_resolution_names (
            source_id VARCHAR NOT NULL,
            entity_type VARCHAR NOT NULL,
            names TEXT NOT NULL,
            PRIMARY KEY (source_id, entity_type)
        )
    """,
    "glossary_clusters": """
        CREATE TABLE IF NOT EXISTS glossary_clusters (
            term_name VARCHAR NOT NULL,
            cluster_id INTEGER NOT NULL,
            session_id VARCHAR NOT NULL,
            PRIMARY KEY (term_name, session_id)
        )
    """,
    "proven_grounding": """
        CREATE TABLE IF NOT EXISTS proven_grounding (
            entity_name TEXT NOT NULL,
            source_pattern TEXT NOT NULL,
            PRIMARY KEY (entity_name, source_pattern)
        )
    """,
}


def _init_db(path: Path, seed: dict[str, list[str]] | None = None) -> None:
    """Create all SPLIT_TABLES in a fresh DuckDB file, optionally seed rows."""
    conn = duckdb.connect(str(path))
    for table in SPLIT_TABLES:
        conn.execute(_SCHEMA_DDL[table])
    if seed:
        for sql_list in seed.values():
            for sql in sql_list:
                conn.execute(sql)
    conn.close()


@pytest.fixture
def split_dbs(tmp_path):
    """Create user + system DuckDB files with matching schemas."""
    user_path = tmp_path / "user.duckdb"
    sys_path = tmp_path / "system.duckdb"
    _init_db(user_path)
    _init_db(sys_path)
    return user_path, sys_path


@pytest.fixture
def split_store(split_dbs):
    user_path, sys_path = split_dbs
    store = SplitVectorStore(user_db_path=user_path, system_db_path=sys_path)
    store._create_union_views()
    yield store
    store.close()


# ------------------------------------------------------------------
# ATTACH + view creation
# ------------------------------------------------------------------

class TestAttachAndViews:
    def test_views_created_for_all_split_tables(self, split_store):
        """All v_* views exist after init."""
        conn = split_store.db.conn
        rows = conn.execute(
            "SELECT view_name FROM duckdb_views() WHERE temporary = true"
        ).fetchall()
        view_names = {r[0] for r in rows}
        for table in SPLIT_TABLES:
            assert f"v_{table}" in view_names, f"Missing view v_{table}"

    def test_composite_views_created(self, split_store):
        """unified_glossary and deprecated_glossary views exist."""
        conn = split_store.db.conn
        rows = conn.execute(
            "SELECT view_name FROM duckdb_views() WHERE temporary = true"
        ).fetchall()
        view_names = {r[0] for r in rows}
        assert "unified_glossary" in view_names
        assert "deprecated_glossary" in view_names

    def test_sys_schema_attached(self, split_store):
        """System DB is ATTACHed as 'sys'."""
        conn = split_store.db.conn
        rows = conn.execute("SELECT database_name FROM duckdb_databases()").fetchall()
        db_names = {r[0] for r in rows}
        assert "sys" in db_names


# ------------------------------------------------------------------
# Reads merge from both DBs
# ------------------------------------------------------------------

class TestUnionReads:
    def test_entities_merged(self, tmp_path):
        user_path = tmp_path / "user.duckdb"
        sys_path = tmp_path / "system.duckdb"
        _init_db(user_path)
        _init_db(sys_path, seed={"entities": [
            "INSERT INTO entities VALUES "
            "('s1', 'sys_entity', 'Sys Entity', 'dimension', NULL, 's1', NULL, CURRENT_TIMESTAMP, 'metadata')"
        ]})

        with SplitVectorStore(user_db_path=user_path, system_db_path=sys_path) as store:
            store._create_union_views()
            conn = store.db.conn
            conn.execute(
                "INSERT INTO main.entities VALUES "
                "('u1', 'user_entity', 'User Entity', 'metric', NULL, 's1', NULL, CURRENT_TIMESTAMP, 'metadata')"
            )
            rows = conn.execute("SELECT id, name FROM v_entities ORDER BY id").fetchall()
            assert len(rows) == 2
            assert {r[0] for r in rows} == {"u1", "s1"}

    def test_document_urls_merged(self, tmp_path):
        user_path = tmp_path / "user.duckdb"
        sys_path = tmp_path / "system.duckdb"
        _init_db(user_path)
        _init_db(sys_path, seed={"document_urls": [
            "INSERT INTO document_urls VALUES ('doc2', 'http://sys/doc2')"
        ]})

        with SplitVectorStore(user_db_path=user_path, system_db_path=sys_path) as store:
            store._create_union_views()
            conn = store.db.conn
            conn.execute("INSERT INTO main.document_urls VALUES ('doc1', 'http://user/doc1')")
            rows = conn.execute("SELECT document_name FROM v_document_urls ORDER BY document_name").fetchall()
            assert [r[0] for r in rows] == ["doc1", "doc2"]

    def test_proven_grounding_merged(self, tmp_path):
        user_path = tmp_path / "user.duckdb"
        sys_path = tmp_path / "system.duckdb"
        _init_db(user_path)
        _init_db(sys_path, seed={"proven_grounding": [
            "INSERT INTO proven_grounding VALUES ('e2', 'p2')"
        ]})

        with SplitVectorStore(user_db_path=user_path, system_db_path=sys_path) as store:
            store._create_union_views()
            conn = store.db.conn
            conn.execute("INSERT INTO main.proven_grounding VALUES ('e1', 'p1')")
            rows = conn.execute("SELECT entity_name FROM v_proven_grounding ORDER BY entity_name").fetchall()
            assert [r[0] for r in rows] == ["e1", "e2"]


# ------------------------------------------------------------------
# Write routing
# ------------------------------------------------------------------

class TestWriteRouting:
    def test_write_schema_user_tier(self, split_store):
        assert split_store._write_schema("user") == "main"

    def test_write_schema_domain_tier(self, split_store):
        assert split_store._write_schema("domain") == "main"

    def test_write_schema_system_tier(self, split_store):
        assert split_store._write_schema("system") == "sys"

    def test_write_schema_warmup_always_main(self, tmp_path):
        user_path = tmp_path / "warmup.duckdb"
        _init_db(user_path)
        store = SplitVectorStore(user_db_path=user_path)
        try:
            assert store._write_schema("system") == "main"
            assert store._write_schema("user") == "main"
        finally:
            store.close()

    def test_qualified_write_to_main(self, split_store):
        schema = split_store._write_schema("user")
        conn = split_store.db.conn
        conn.execute(
            f"INSERT INTO {schema}.document_urls VALUES ('d1', 'http://example.com')"
        )
        row = conn.execute("SELECT source_url FROM main.document_urls WHERE document_name='d1'").fetchone()
        assert row[0] == "http://example.com"


# ------------------------------------------------------------------
# Warmup mode (single DB, no ATTACH)
# ------------------------------------------------------------------

class TestWarmupMode:
    def test_warmup_no_sys_attached(self, tmp_path):
        user_path = tmp_path / "warmup.duckdb"
        _init_db(user_path)
        store = SplitVectorStore(user_db_path=user_path)
        try:
            assert store.warmup is True
            rows = store.db.conn.execute("SELECT database_name FROM duckdb_databases()").fetchall()
            db_names = {r[0] for r in rows}
            assert "sys" not in db_names
        finally:
            store.close()

    def test_warmup_no_views_created(self, tmp_path):
        user_path = tmp_path / "warmup.duckdb"
        _init_db(user_path)
        store = SplitVectorStore(user_db_path=user_path)
        try:
            rows = store.db.conn.execute(
                "SELECT view_name FROM duckdb_views() WHERE temporary = true"
            ).fetchall()
            view_names = {r[0] for r in rows}
            for table in SPLIT_TABLES:
                assert f"v_{table}" not in view_names
        finally:
            store.close()

    def test_warmup_direct_table_access(self, tmp_path):
        user_path = tmp_path / "warmup.duckdb"
        _init_db(user_path)
        store = SplitVectorStore(user_db_path=user_path)
        try:
            store.db.conn.execute(
                "INSERT INTO document_urls VALUES ('d1', 'http://example.com')"
            )
            row = store.db.conn.execute(
                "SELECT source_url FROM document_urls WHERE document_name='d1'"
            ).fetchone()
            assert row[0] == "http://example.com"
        finally:
            store.close()


# ------------------------------------------------------------------
# Context manager
# ------------------------------------------------------------------

class TestContextManager:
    def test_context_manager_closes(self, split_dbs):
        user_path, sys_path = split_dbs
        with SplitVectorStore(user_db_path=user_path, system_db_path=sys_path) as store:
            store.db.conn.execute("SELECT 1").fetchone()
        # After exit, pool should be closed
        assert store.db.pool._closed


# ------------------------------------------------------------------
# Cross-DB move via RelationalStore
# ------------------------------------------------------------------

class TestMoveRow:
    def test_move_row_within_writable_schema(self, tmp_path):
        """_move_row transfers a row using SELECT→DELETE→INSERT pattern.

        Tests within main using a non-split single-DB to verify the pattern works.
        """
        user_path = tmp_path / "user.duckdb"
        _init_db(user_path)

        from constat.storage.duckdb_pool import ThreadLocalDuckDB
        from constat.storage.relational import RelationalStore

        db = ThreadLocalDuckDB(str(user_path))
        rel = RelationalStore(db, split_mode=False)
        try:
            db.conn.execute("INSERT INTO document_urls VALUES ('d1', 'http://example.com')")
            # _move_row with same schema is a no-op
            rel._move_row("document_urls", "document_name", "d1", "main", "main")
            row = db.conn.execute("SELECT source_url FROM document_urls WHERE document_name = 'd1'").fetchone()
            assert row[0] == "http://example.com"
        finally:
            db.close()

    def test_move_row_with_updates(self, tmp_path):
        """_move_row applies column overrides during transfer."""
        user_path = tmp_path / "user.duckdb"
        sys_path = tmp_path / "system.duckdb"
        # Seed a row in sys before attaching
        _init_db(user_path)
        _init_db(sys_path, seed={"document_urls": [
            "INSERT INTO document_urls VALUES ('d1', 'http://old.com')"
        ]})

        # ATTACH sys for this test
        from constat.storage.duckdb_pool import ThreadLocalDuckDB
        from constat.storage.relational import RelationalStore

        db = ThreadLocalDuckDB(str(user_path))
        db.conn.execute(f"ATTACH '{sys_path}' AS sys")
        rel = RelationalStore(db, split_mode=True)
        try:
            # Move from sys → main with updates
            rel._move_row(
                "document_urls", "document_name", "d1", "sys", "main",
                updates={"source_url": "http://new.com"},
            )
            row = db.conn.execute("SELECT source_url FROM main.document_urls WHERE document_name = 'd1'").fetchone()
            assert row[0] == "http://new.com"
            # Gone from sys
            row = db.conn.execute("SELECT 1 FROM sys.document_urls WHERE document_name = 'd1'").fetchone()
            assert row is None
        finally:
            db.close()

    def test_move_row_noop_same_schema(self, tmp_path):
        """_move_row is a no-op when from_schema == to_schema."""
        user_path = tmp_path / "user.duckdb"
        sys_path = tmp_path / "system.duckdb"
        _init_db(user_path)
        _init_db(sys_path)

        with SplitVectorStore(user_db_path=user_path, system_db_path=sys_path) as store:
            from constat.storage.relational import RelationalStore
            rel = RelationalStore(store.db, split_mode=True)

            store.db.conn.execute(
                "INSERT INTO main.document_urls VALUES ('d1', 'http://example.com')"
            )
            rel._move_row("document_urls", "document_name", "d1", "main", "main")

            # Row should still be in main
            row = store.db.conn.execute(
                "SELECT source_url FROM main.document_urls WHERE document_name = 'd1'"
            ).fetchone()
            assert row[0] == "http://example.com"

    def test_move_row_missing_pk(self, tmp_path):
        """_move_row is a no-op when PK not found in source."""
        user_path = tmp_path / "user.duckdb"
        sys_path = tmp_path / "system.duckdb"
        _init_db(user_path)
        _init_db(sys_path)

        # ATTACH sys as writable for this test
        from constat.storage.duckdb_pool import ThreadLocalDuckDB
        from constat.storage.relational import RelationalStore

        db = ThreadLocalDuckDB(str(user_path))
        db.conn.execute(f"ATTACH '{sys_path}' AS sys")
        rel = RelationalStore(db, split_mode=True)
        try:
            # No row to move — should be a silent no-op
            rel._move_row("document_urls", "document_name", "nonexistent", "main", "sys")
        finally:
            db.close()


class TestReconcileCrossDB:
    def test_reconcile_glossary_same_db_update(self, tmp_path):
        """reconcile_glossary_domains does a simple UPDATE when tier doesn't change."""
        user_path = tmp_path / "user.duckdb"
        sys_path = tmp_path / "system.duckdb"
        _init_db(user_path)
        _init_db(sys_path)

        def tier_fn(domain_id):
            return "user"  # Everything stays in user tier

        with SplitVectorStore(user_db_path=user_path, system_db_path=sys_path) as store:
            store._create_union_views()
            from constat.storage.relational import RelationalStore
            rel = RelationalStore(store.db, split_mode=True, domain_tier_fn=tier_fn)

            conn = store.db.conn

            conn.execute(
                "INSERT INTO main.glossary_terms "
                "(id, name, display_name, definition, domain, session_id, user_id) "
                "VALUES ('g1', 'term1', 'Term 1', 'A term', 'domain-a', 's1', 'default')"
            )
            conn.execute(
                "INSERT INTO main.entities "
                "(id, name, display_name, semantic_type, session_id, domain_id) "
                "VALUES ('e1', 'term1', 'Term 1', 'metric', 's1', 'domain-b')"
            )

            moved = rel.reconcile_glossary_domains("s1")

            assert len(moved) == 1
            # Term should still be in main, domain updated
            row = conn.execute(
                "SELECT domain FROM main.glossary_terms WHERE id = 'g1'"
            ).fetchone()
            assert row[0] == "domain-b"

    def test_reconcile_glossary_cross_db_move(self, tmp_path):
        """reconcile_glossary_domains moves rows from sys→main when tier changes.

        Verifies the move_row pattern works for cross-DB moves in both
        directions (sys→main and main→sys).
        """
        user_path = tmp_path / "user.duckdb"
        sys_path = tmp_path / "system.duckdb"
        _init_db(user_path)
        # Seed a glossary term in sys before attaching
        _init_db(sys_path, seed={"glossary_terms": [
            "INSERT INTO glossary_terms "
            "(id, name, display_name, definition, domain, session_id, user_id) "
            "VALUES ('g1', 'term1', 'Term 1', 'A term', 'shared-domain', 's1', 'default')"
        ]})

        # Tier: "user-domain" → user, "shared-domain" → system
        def tier_fn(domain_id):
            return "system" if domain_id == "shared-domain" else "user"

        # ATTACH sys as writable so cross-DB delete works for this test
        from constat.storage.duckdb_pool import ThreadLocalDuckDB
        from constat.storage.relational import RelationalStore

        db = ThreadLocalDuckDB(str(user_path))
        db.conn.execute(f"ATTACH '{sys_path}' AS sys")
        # Create union views manually
        for table in SPLIT_TABLES:
            db.conn.execute(
                f"CREATE OR REPLACE TEMP VIEW v_{table} AS "
                f"SELECT * FROM main.{table} UNION ALL SELECT * FROM sys.{table}"
            )
        rel = RelationalStore(db, split_mode=True, domain_tier_fn=tier_fn)
        try:
            conn = db.conn

            # Entity in main pointing to "user-domain" — triggers reconcile
            conn.execute(
                "INSERT INTO main.entities "
                "(id, name, display_name, semantic_type, session_id, domain_id) "
                "VALUES ('e1', 'term1', 'Term 1', 'metric', 's1', 'user-domain')"
            )

            moved = rel.reconcile_glossary_domains("s1")

            assert len(moved) == 1
            assert moved[0]["from_domain"] == "shared-domain"
            assert moved[0]["to_domain"] == "user-domain"

            # Term should now be in main
            row = conn.execute(
                "SELECT domain FROM main.glossary_terms WHERE id = 'g1'"
            ).fetchone()
            assert row[0] == "user-domain"

            # Gone from sys
            row = conn.execute(
                "SELECT 1 FROM sys.glossary_terms WHERE id = 'g1'"
            ).fetchone()
            assert row is None
        finally:
            db.close()

    def test_schema_for_domain_helper(self, tmp_path):
        """_schema_for_domain routes based on domain_tier_fn."""
        user_path = tmp_path / "user.duckdb"
        _init_db(user_path)

        from constat.storage.duckdb_pool import ThreadLocalDuckDB
        from constat.storage.relational import RelationalStore

        def tier_fn(domain_id):
            return "system" if domain_id == "sys-domain" else "user"

        db = ThreadLocalDuckDB(str(user_path))
        rel = RelationalStore(db, split_mode=True, domain_tier_fn=tier_fn)
        try:
            assert rel._schema_for_domain(None) == "main"
            assert rel._schema_for_domain("user-domain") == "main"
            assert rel._schema_for_domain("sys-domain") == "sys"
        finally:
            db.close()

    def test_schema_for_domain_no_split(self, tmp_path):
        """_schema_for_domain always returns 'main' when not in split mode."""
        user_path = tmp_path / "user.duckdb"
        _init_db(user_path)

        from constat.storage.duckdb_pool import ThreadLocalDuckDB
        from constat.storage.relational import RelationalStore

        db = ThreadLocalDuckDB(str(user_path))
        rel = RelationalStore(db, split_mode=False)
        try:
            assert rel._schema_for_domain("anything") == "main"
            assert rel._schema_for_domain(None) == "main"
        finally:
            db.close()
