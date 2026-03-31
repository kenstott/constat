# Copyright (c) 2025 Kenneth Stott
# Canary: 348376dc-be59-43db-b138-415d60a23eca
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tests for entity resolution name cache (warmup + session read)."""

import os
import tempfile

import pytest

from constat.discovery.vector_store import DuckDBVectorStore


@pytest.fixture
def temp_db():
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_vectors.duckdb")
    old_path = os.environ.get("CONSTAT_VECTOR_STORE_PATH")
    os.environ["CONSTAT_VECTOR_STORE_PATH"] = db_path
    yield db_path
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
    return DuckDBVectorStore(db_path=temp_db)


class TestEntityResolutionNameCache:

    def test_store_and_get_entity_resolution_names(self, vector_store):
        """Round-trip: store names then read them back."""
        terms = {"COUNTRY": ["France", "Germany"], "CITY": ["Paris", "Berlin"]}
        vector_store.store_entity_resolution_names("__base__", terms)
        result = vector_store.get_entity_resolution_names(["__base__"])
        assert sorted(result["COUNTRY"]) == ["France", "Germany"]
        assert sorted(result["CITY"]) == ["Berlin", "Paris"]

    def test_get_names_filtered_by_source(self, vector_store):
        """Multi-source: only requested source_ids returned."""
        vector_store.store_entity_resolution_names("__base__", {"COUNTRY": ["France"]})
        vector_store.store_entity_resolution_names("sales", {"COUNTRY": ["USA"], "PRODUCT": ["Widget"]})
        vector_store.store_entity_resolution_names("hr", {"DEPT": ["Engineering"]})

        # Only base + sales
        result = vector_store.get_entity_resolution_names(["__base__", "sales"])
        assert sorted(result["COUNTRY"]) == ["France", "USA"]
        assert result.get("PRODUCT") == ["Widget"]
        assert "DEPT" not in result

    def test_get_names_all_sources(self, vector_store):
        """No filter returns all sources."""
        vector_store.store_entity_resolution_names("__base__", {"COUNTRY": ["France"]})
        vector_store.store_entity_resolution_names("sales", {"PRODUCT": ["Widget"]})
        result = vector_store.get_entity_resolution_names()
        assert "COUNTRY" in result
        assert "PRODUCT" in result

    def test_store_replaces_previous(self, vector_store):
        """Storing again for same source_id replaces old data."""
        vector_store.store_entity_resolution_names("__base__", {"COUNTRY": ["France"]})
        vector_store.store_entity_resolution_names("__base__", {"COUNTRY": ["Germany"]})
        result = vector_store.get_entity_resolution_names(["__base__"])
        assert result["COUNTRY"] == ["Germany"]

    def test_er_hash_round_trip(self, vector_store):
        """er_hash can be stored and retrieved via get/set_source_hash."""
        assert vector_store.get_source_hash("__base__", "er") is None
        vector_store.set_source_hash("__base__", "er", "abc123")
        assert vector_store.get_source_hash("__base__", "er") == "abc123"

    def test_er_hash_invalidation(self, vector_store):
        """Changing er_hash triggers detection of config change."""
        vector_store.set_source_hash("__base__", "er", "hash_v1")
        vector_store.store_entity_resolution_names("__base__", {"COUNTRY": ["France"]})

        # Simulate config change — new hash differs from cached
        cached = vector_store.get_source_hash("__base__", "er")
        new_hash = "hash_v2"
        assert cached != new_hash

        # After re-extraction, update hash
        vector_store.store_entity_resolution_names("__base__", {"COUNTRY": ["Germany", "Italy"]})
        vector_store.set_source_hash("__base__", "er", new_hash)
        assert vector_store.get_source_hash("__base__", "er") == new_hash
        result = vector_store.get_entity_resolution_names(["__base__"])
        assert sorted(result["COUNTRY"]) == ["Germany", "Italy"]

    def test_empty_source_returns_empty(self, vector_store):
        """Querying a source with no cached names returns empty dict."""
        result = vector_store.get_entity_resolution_names(["nonexistent"])
        assert result == {}
