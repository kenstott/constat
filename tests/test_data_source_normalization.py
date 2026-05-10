# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tests for Phase 3 data source normalization."""

from __future__ import annotations
import tempfile
from pathlib import Path

import numpy as np
import pytest

from constat.discovery.models import DocumentChunk, ChunkType
from constat.discovery.vector_store import DuckDBVectorStore


@pytest.fixture
def store():
    """Create a DuckDBVectorStore with a temp DB."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.duckdb"
        vs = DuckDBVectorStore(db_path=str(db_path))
        yield vs


class TestDataSourcesTableCreation:
    """Test that data_sources table is created during schema init."""

    def test_table_exists(self, store):
        row = store._conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_name = 'data_sources'"
        ).fetchone()
        assert row[0] == 1

    def test_table_columns(self, store):
        cols = store._conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'data_sources' ORDER BY ordinal_position"
        ).fetchall()
        col_names = [c[0] for c in cols]
        assert "id" in col_names
        assert "name" in col_names
        assert "type" in col_names
        assert "domain_id" in col_names
        assert "session_id" in col_names
        assert "created_at" in col_names


class TestEnsureDataSource:
    """Test ensure_data_source idempotency."""

    def test_creates_row(self, store):
        source_id = store.ensure_data_source("my_db", "schema", "domain-1", "sess-1")
        assert source_id.startswith("ds_")
        row = store._conn.execute(
            "SELECT name, type, domain_id, session_id FROM data_sources WHERE id = ?",
            [source_id],
        ).fetchone()
        assert row == ("my_db", "schema", "domain-1", "sess-1")

    def test_idempotent(self, store):
        id1 = store.ensure_data_source("my_db", "schema", "domain-1")
        id2 = store.ensure_data_source("my_db", "schema", "domain-1")
        assert id1 == id2
        count = store._conn.execute(
            "SELECT COUNT(*) FROM data_sources WHERE id = ?", [id1]
        ).fetchone()[0]
        assert count == 1

    def test_different_sources_different_ids(self, store):
        id1 = store.ensure_data_source("db_a", "schema", "domain-1")
        id2 = store.ensure_data_source("db_b", "api", "domain-1")
        assert id1 != id2


class TestDataSourceIdOnChunks:
    """Test that data_source_id is populated when adding chunks."""

    def test_data_source_id_populated(self, store):
        chunks = [
            DocumentChunk(
                document_name="test_doc",
                chunk_type=ChunkType.DOCUMENT,
                section="intro",
                chunk_index=0,
                content="Hello world",
            )
        ]
        embeddings = np.random.randn(1, DuckDBVectorStore.EMBEDDING_DIM).astype(np.float32)
        store.add_chunks(chunks, embeddings, source="document", domain_id="domain-1")

        row = store._conn.execute(
            "SELECT data_source_id FROM embeddings WHERE document_name = 'test_doc'"
        ).fetchone()
        assert row is not None
        assert row[0] is not None
        assert row[0].startswith("ds_")

    def test_data_source_id_null_without_domain(self, store):
        chunks = [
            DocumentChunk(
                document_name="no_domain_doc",
                chunk_type=ChunkType.DOCUMENT,
                section="intro",
                chunk_index=0,
                content="No domain here",
            )
        ]
        embeddings = np.random.randn(1, DuckDBVectorStore.EMBEDDING_DIM).astype(np.float32)
        store.add_chunks(chunks, embeddings, source="document")

        row = store._conn.execute(
            "SELECT data_source_id FROM embeddings WHERE document_name = 'no_domain_doc'"
        ).fetchone()
        assert row is not None
        assert row[0] is None


class TestDomainViaJoin:
    """Test domain resolution via data_sources join."""

    def test_get_domain_for_chunk(self, store):
        chunks = [
            DocumentChunk(
                document_name="joined_doc",
                chunk_type=ChunkType.DOCUMENT,
                section="body",
                chunk_index=0,
                content="Join test content",
            )
        ]
        embeddings = np.random.randn(1, DuckDBVectorStore.EMBEDDING_DIM).astype(np.float32)
        store.add_chunks(chunks, embeddings, source="schema", domain_id="sales")

        chunk_id = store._conn.execute(
            "SELECT chunk_id FROM embeddings WHERE document_name = 'joined_doc'"
        ).fetchone()[0]

        domain = store.get_domain_for_chunk(chunk_id)
        assert domain == "sales"

    def test_get_domain_for_chunk_no_source(self, store):
        """Chunk without data_source_id returns None."""
        chunks = [
            DocumentChunk(
                document_name="orphan_doc",
                chunk_type=ChunkType.DOCUMENT,
                section="body",
                chunk_index=0,
                content="Orphan content",
            )
        ]
        embeddings = np.random.randn(1, DuckDBVectorStore.EMBEDDING_DIM).astype(np.float32)
        store.add_chunks(chunks, embeddings, source="document")

        chunk_id = store._conn.execute(
            "SELECT chunk_id FROM embeddings WHERE document_name = 'orphan_doc'"
        ).fetchone()[0]

        domain = store.get_domain_for_chunk(chunk_id)
        assert domain is None


class TestBackwardCompat:
    """Test that existing domain_id column still works."""

    def test_domain_id_still_populated(self, store):
        chunks = [
            DocumentChunk(
                document_name="compat_doc",
                chunk_type=ChunkType.DOCUMENT,
                section="s1",
                chunk_index=0,
                content="Backward compat test",
            )
        ]
        embeddings = np.random.randn(1, DuckDBVectorStore.EMBEDDING_DIM).astype(np.float32)
        store.add_chunks(chunks, embeddings, source="document", domain_id="hr-domain")

        row = store._conn.execute(
            "SELECT domain_id, data_source_id FROM embeddings "
            "WHERE document_name = 'compat_doc'"
        ).fetchone()
        assert row[0] == "hr-domain"
        assert row[1] is not None

    def test_domain_id_filter_still_works(self, store):
        chunks = [
            DocumentChunk(
                document_name="filter_doc",
                chunk_type=ChunkType.DOCUMENT,
                section="s1",
                chunk_index=0,
                content="Filter test",
            )
        ]
        embeddings = np.random.randn(1, DuckDBVectorStore.EMBEDDING_DIM).astype(np.float32)
        store.add_chunks(chunks, embeddings, source="document", domain_id="my-domain")

        count = store._conn.execute(
            "SELECT COUNT(*) FROM embeddings WHERE domain_id = 'my-domain'"
        ).fetchone()[0]
        assert count == 1
