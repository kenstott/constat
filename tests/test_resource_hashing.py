# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tests for resource-level hashing functionality."""

import os
import tempfile
import pytest
import numpy as np

from constat.discovery.vector_store import DuckDBVectorStore
from constat.discovery.models import DocumentChunk, ChunkType


@pytest.fixture
def temp_db():
    """Create a temporary database path for testing."""
    # Create a temp directory and use a path within it
    # Don't create the file - DuckDB will create it
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_vectors.duckdb")

    # Set environment variable to use this path
    old_path = os.environ.get("CONSTAT_VECTOR_STORE_PATH")
    os.environ["CONSTAT_VECTOR_STORE_PATH"] = db_path

    yield db_path

    # Cleanup
    if old_path:
        os.environ["CONSTAT_VECTOR_STORE_PATH"] = old_path
    else:
        os.environ.pop("CONSTAT_VECTOR_STORE_PATH", None)

    try:
        os.unlink(db_path)
        os.rmdir(temp_dir)
    except OSError:
        pass


@pytest.fixture
def vector_store(temp_db):
    """Create a vector store for testing."""
    return DuckDBVectorStore(db_path=temp_db)


class TestResourceHashing:
    """Test resource-level hashing for incremental updates."""

    def test_get_set_resource_hash(self, vector_store):
        """Test getting and setting resource hashes."""
        # Initially no hash
        hash_val = vector_store.get_resource_hash("__base__", "database", "chinook")
        assert hash_val is None

        # Set hash
        vector_store.set_resource_hash("__base__", "database", "chinook", "abc123")

        # Now hash exists
        hash_val = vector_store.get_resource_hash("__base__", "database", "chinook")
        assert hash_val == "abc123"

        # Update hash
        vector_store.set_resource_hash("__base__", "database", "chinook", "def456")
        hash_val = vector_store.get_resource_hash("__base__", "database", "chinook")
        assert hash_val == "def456"

    def test_resource_hash_different_types(self, vector_store):
        """Test resource hashes for different resource types."""
        # Set hashes for different types
        vector_store.set_resource_hash("__base__", "database", "mydb", "hash1")
        vector_store.set_resource_hash("__base__", "api", "myapi", "hash2")
        vector_store.set_resource_hash("__base__", "document", "mydoc", "hash3")

        # Each is independent
        assert vector_store.get_resource_hash("__base__", "database", "mydb") == "hash1"
        assert vector_store.get_resource_hash("__base__", "api", "myapi") == "hash2"
        assert vector_store.get_resource_hash("__base__", "document", "mydoc") == "hash3"

    def test_resource_hash_different_sources(self, vector_store):
        """Test resource hashes for different source IDs."""
        # Same resource name but different sources
        vector_store.set_resource_hash("__base__", "database", "db1", "base_hash")
        vector_store.set_resource_hash("domain1", "database", "db1", "domain_hash")

        # Each source has its own hash
        assert vector_store.get_resource_hash("__base__", "database", "db1") == "base_hash"
        assert vector_store.get_resource_hash("domain1", "database", "db1") == "domain_hash"

    def test_delete_resource_hash(self, vector_store):
        """Test deleting resource hashes."""
        vector_store.set_resource_hash("__base__", "document", "doc1", "hash123")

        # Hash exists
        assert vector_store.get_resource_hash("__base__", "document", "doc1") == "hash123"

        # Delete it
        deleted = vector_store.delete_resource_hash("__base__", "document", "doc1")
        assert deleted is True

        # Now it's gone
        assert vector_store.get_resource_hash("__base__", "document", "doc1") is None

        # Deleting again returns False
        deleted = vector_store.delete_resource_hash("__base__", "document", "doc1")
        assert deleted is False

    def test_get_resource_hashes_for_source(self, vector_store):
        """Test getting all resource hashes for a source."""
        # Set multiple hashes
        vector_store.set_resource_hash("__base__", "database", "db1", "hash1")
        vector_store.set_resource_hash("__base__", "database", "db2", "hash2")
        vector_store.set_resource_hash("__base__", "api", "api1", "hash3")
        vector_store.set_resource_hash("domain1", "database", "db1", "hash4")

        # Get all for base
        all_hashes = vector_store.get_resource_hashes_for_source("__base__")
        assert len(all_hashes) == 3
        assert all_hashes["db1"] == "hash1"
        assert all_hashes["db2"] == "hash2"
        assert all_hashes["api1"] == "hash3"

        # Get only databases for base
        db_hashes = vector_store.get_resource_hashes_for_source("__base__", "database")
        assert len(db_hashes) == 2
        assert "db1" in db_hashes
        assert "db2" in db_hashes
        assert "api1" not in db_hashes

        # Get for domain1
        domain_hashes = vector_store.get_resource_hashes_for_source("domain1")
        assert len(domain_hashes) == 1
        assert domain_hashes["db1"] == "hash4"

    def test_clear_resource_hashes_for_source(self, vector_store):
        """Test clearing all resource hashes for a source."""
        vector_store.set_resource_hash("__base__", "database", "db1", "hash1")
        vector_store.set_resource_hash("__base__", "api", "api1", "hash2")
        vector_store.set_resource_hash("domain1", "database", "db1", "hash3")

        # Clear base hashes
        count = vector_store.clear_resource_hashes_for_source("__base__")
        assert count == 2

        # Base hashes gone
        assert vector_store.get_resource_hash("__base__", "database", "db1") is None
        assert vector_store.get_resource_hash("__base__", "api", "api1") is None

        # Domain hash still exists
        assert vector_store.get_resource_hash("domain1", "database", "db1") == "hash3"


class TestDeleteResourceChunks:
    """Test deleting chunks for specific resources."""

    def _add_test_chunks(self, vector_store, source_id, resource_name, source_type, count=3):
        """Helper to add test chunks."""
        chunks = []
        for i in range(count):
            chunks.append(DocumentChunk(
                document_name=resource_name,
                content=f"Test content {i} for {resource_name}",
                section="Test",
                chunk_index=i,
                source=source_type,
                chunk_type=ChunkType.DOCUMENT,
            ))

        # Create dummy embeddings
        embeddings = np.random.rand(count, vector_store.EMBEDDING_DIM).astype(np.float32)

        domain_id = None if source_id == "__base__" else source_id
        vector_store.add_chunks(chunks, embeddings, source=source_type, domain_id=domain_id)

        return chunks

    def test_delete_resource_chunks_base(self, vector_store):
        """Test deleting chunks for a base config resource."""
        # Add chunks for two documents
        self._add_test_chunks(vector_store, "__base__", "doc1", "document", count=3)
        self._add_test_chunks(vector_store, "__base__", "doc2", "document", count=2)

        # Initial count
        total = vector_store.count()
        assert total == 5

        # Delete doc1 chunks
        deleted = vector_store.delete_resource_chunks("__base__", "document", "doc1")
        assert deleted == 3

        # doc2 chunks still exist
        remaining = vector_store.count()
        assert remaining == 2

    def test_delete_resource_chunks_domain(self, vector_store):
        """Test deleting chunks for a domain resource."""
        # Add chunks to base and domain (different doc names due to add_chunks dedup logic)
        self._add_test_chunks(vector_store, "__base__", "base_doc", "document", count=2)
        self._add_test_chunks(vector_store, "domain1", "domain_doc", "document", count=3)

        total = vector_store.count()
        assert total == 5

        # Delete only domain1's chunks
        deleted = vector_store.delete_resource_chunks("domain1", "document", "domain_doc")
        assert deleted == 3

        # Base chunks still exist
        remaining = vector_store.count()
        assert remaining == 2

    def test_delete_resource_chunks_database(self, vector_store):
        """Test deleting chunks for a database resource."""
        # Add schema chunks
        self._add_test_chunks(vector_store, "__base__", "chinook", "schema", count=5)
        self._add_test_chunks(vector_store, "__base__", "northwind", "schema", count=3)

        # Delete chinook
        deleted = vector_store.delete_resource_chunks("__base__", "database", "chinook")
        assert deleted == 5

        # northwind remains
        remaining = vector_store.count()
        assert remaining == 3

    def test_delete_nonexistent_resource(self, vector_store):
        """Test deleting chunks for a resource that doesn't exist."""
        deleted = vector_store.delete_resource_chunks("__base__", "document", "nonexistent")
        assert deleted == 0


class TestResourceHashComputation:
    """Test resource hash computation functions."""

    def test_doc_resource_hash_includes_mtime(self, temp_db):
        """Test that document resource hash includes file modification time."""
        import time
        from constat.server.app import _compute_doc_resource_hash

        # Create a temporary file
        temp_dir = os.path.dirname(temp_db)
        test_file = os.path.join(temp_dir, "test_doc.md")

        with open(test_file, "w") as f:
            f.write("Test content")

        # Create a mock doc config
        class MockDocConfig:
            path = test_file
            description = "Test doc"
            format = "markdown"
            url = None
            type = "markdown"

        # Compute hash
        hash1 = _compute_doc_resource_hash("test_doc", MockDocConfig(), temp_dir)

        # Modify the file (need to wait a bit for mtime to change)
        time.sleep(0.1)
        with open(test_file, "w") as f:
            f.write("Modified content")

        # Hash should be different after file modification
        hash2 = _compute_doc_resource_hash("test_doc", MockDocConfig(), temp_dir)

        assert hash1 != hash2, "Hash should change when file is modified"

        # Clean up
        os.unlink(test_file)

    def test_doc_resource_hash_same_for_unchanged_file(self, temp_db):
        """Test that document resource hash is stable for unchanged file."""
        from constat.server.app import _compute_doc_resource_hash

        temp_dir = os.path.dirname(temp_db)
        test_file = os.path.join(temp_dir, "stable_doc.md")

        with open(test_file, "w") as f:
            f.write("Stable content")

        class MockDocConfig:
            path = test_file
            description = "Stable doc"
            format = "markdown"
            url = None
            type = "markdown"

        # Compute hash twice without changes
        hash1 = _compute_doc_resource_hash("stable_doc", MockDocConfig(), temp_dir)
        hash2 = _compute_doc_resource_hash("stable_doc", MockDocConfig(), temp_dir)

        assert hash1 == hash2, "Hash should be stable for unchanged file"

        # Clean up
        os.unlink(test_file)


class TestIncrementalUpdateFlow:
    """Test the full incremental update flow."""

    def _add_test_chunks(self, vector_store, source_id, resource_name, source_type, count=3):
        """Helper to add test chunks."""
        chunks = []
        for i in range(count):
            chunks.append(DocumentChunk(
                document_name=resource_name,
                content=f"Test content {i} for {resource_name}",
                section="Test",
                chunk_index=i,
                source=source_type,
                chunk_type=ChunkType.DOCUMENT,
            ))

        embeddings = np.random.rand(count, vector_store.EMBEDDING_DIM).astype(np.float32)
        domain_id = None if source_id == "__base__" else source_id
        vector_store.add_chunks(chunks, embeddings, source=source_type, domain_id=domain_id)

    def test_incremental_update_single_resource(self, vector_store):
        """Test updating a single resource without affecting others."""
        # Initial state: 3 documents with hashes
        for doc in ["doc1", "doc2", "doc3"]:
            self._add_test_chunks(vector_store, "__base__", doc, "document", count=2)
            vector_store.set_resource_hash("__base__", "document", doc, f"hash_{doc}")

        assert vector_store.count() == 6

        # Simulate doc2 changed
        old_hash = vector_store.get_resource_hash("__base__", "document", "doc2")
        new_hash = "hash_doc2_v2"

        assert old_hash != new_hash

        # Delete old chunks for doc2
        deleted = vector_store.delete_resource_chunks("__base__", "document", "doc2")
        assert deleted == 2

        # Add new chunks for doc2
        self._add_test_chunks(vector_store, "__base__", "doc2", "document", count=3)

        # Update hash
        vector_store.set_resource_hash("__base__", "document", "doc2", new_hash)

        # Verify: 2 + 3 + 2 = 7 chunks
        assert vector_store.count() == 7
        assert vector_store.get_resource_hash("__base__", "document", "doc2") == new_hash

        # Other hashes unchanged
        assert vector_store.get_resource_hash("__base__", "document", "doc1") == "hash_doc1"
        assert vector_store.get_resource_hash("__base__", "document", "doc3") == "hash_doc3"

    def test_two_level_hash_check(self, vector_store):
        """Test the two-level hash check pattern."""
        # Set up initial state
        source_id = "__base__"

        # Set source-level hash (combined)
        vector_store.set_source_hash(source_id, "doc", "combined_hash_v1")

        # Set resource-level hashes
        vector_store.set_resource_hash(source_id, "document", "doc1", "doc1_hash")
        vector_store.set_resource_hash(source_id, "document", "doc2", "doc2_hash")

        # Add chunks
        self._add_test_chunks(vector_store, source_id, "doc1", "document", count=2)
        self._add_test_chunks(vector_store, source_id, "doc2", "document", count=2)

        # Simulate server restart with same config
        # 1. Check source-level hash
        stored_source_hash = vector_store.get_source_hash(source_id, "doc")
        new_source_hash = "combined_hash_v1"  # Same as before

        if stored_source_hash == new_source_hash:
            # Fast path: nothing changed, skip all resource checks
            pass

        # Verify chunks unchanged
        assert vector_store.count() == 4

        # Now simulate config change (doc2 modified)
        new_source_hash = "combined_hash_v2"  # Different

        if stored_source_hash != new_source_hash:
            # Slow path: check each resource
            resource_hashes = vector_store.get_resource_hashes_for_source(source_id, "document")

            # doc1 unchanged
            if resource_hashes.get("doc1") == "doc1_hash":
                pass  # Skip doc1

            # doc2 changed
            new_doc2_hash = "doc2_hash_v2"
            if resource_hashes.get("doc2") != new_doc2_hash:
                # Delete and rebuild doc2
                vector_store.delete_resource_chunks(source_id, "document", "doc2")
                self._add_test_chunks(vector_store, source_id, "doc2", "document", count=3)
                vector_store.set_resource_hash(source_id, "document", "doc2", new_doc2_hash)

            # Update source hash
            vector_store.set_source_hash(source_id, "doc", new_source_hash)

        # Verify: doc1 (2) + doc2 (3) = 5
        assert vector_store.count() == 5
        assert vector_store.get_resource_hash(source_id, "document", "doc2") == "doc2_hash_v2"
        assert vector_store.get_source_hash(source_id, "doc") == "combined_hash_v2"
