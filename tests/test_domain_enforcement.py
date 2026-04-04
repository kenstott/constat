# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tests for Phase 4 universal domain enforcement."""

import tempfile
import pytest
from datetime import datetime

from constat.discovery.models import Entity, EntityRelationship, SemanticType
from constat.discovery.vector_store import DuckDBVectorStore
from constat.storage.learnings import LearningStore, LearningCategory, LearningSource


class TestEntityDomainEnforcement:
    """Validate that entities require a non-None domain_id."""

    def test_entity_insert_none_domain_raises(self, tmp_path):
        store = DuckDBVectorStore(str(tmp_path / "test.duckdb"))
        entity = Entity(
            id="e1",
            name="customer",
            display_name="Customer",
            semantic_type=SemanticType.CONCEPT,
            session_id="sess-1",
            domain_id=None,
        )
        with pytest.raises(ValueError, match="domain_id is required"):
            store.add_entities([entity], "sess-1")

    def test_entity_insert_empty_domain_raises(self, tmp_path):
        store = DuckDBVectorStore(str(tmp_path / "test.duckdb"))
        entity = Entity(
            id="e2",
            name="order",
            display_name="Order",
            semantic_type=SemanticType.CONCEPT,
            session_id="sess-1",
            domain_id="",
        )
        with pytest.raises(ValueError, match="domain_id is required"):
            store.add_entities([entity], "sess-1")

    def test_entity_insert_valid_domain_succeeds(self, tmp_path):
        store = DuckDBVectorStore(str(tmp_path / "test.duckdb"))
        entity = Entity(
            id="e3",
            name="product",
            display_name="Product",
            semantic_type=SemanticType.CONCEPT,
            session_id="sess-1",
            domain_id="sales-analytics",
        )
        store.add_entities([entity], "sess-1")
        found = store.find_entity_by_name(
            "product", domain_ids=["sales-analytics"], session_id="sess-1",
        )
        assert found is not None
        assert found.domain_id == "sales-analytics"


class TestRelationshipDomainColumn:
    """Validate that entity_relationships table has a domain column."""

    def test_relationship_domain_column_exists(self, tmp_path):
        store = DuckDBVectorStore(str(tmp_path / "test.duckdb"))
        rel = EntityRelationship(
            id="r1",
            subject_name="customer",
            verb="places",
            object_name="order",
            session_id="sess-1",
            domain="sales-analytics",
        )
        store.add_entity_relationship(rel)
        rows = store.get_relationships_for_entity(
            "customer", "sess-1",
        )
        assert len(rows) == 1
        assert rows[0]["subject_name"] == "customer"

    def test_relationship_domain_stored(self, tmp_path):
        store = DuckDBVectorStore(str(tmp_path / "test.duckdb"))
        rel = EntityRelationship(
            id="r2",
            subject_name="employee",
            verb="works_in",
            object_name="department",
            session_id="sess-1",
            domain="hr-reporting",
        )
        store.add_entity_relationship(rel)
        # Verify domain column is readable via raw SQL
        row = store._conn.execute(
            "SELECT domain FROM entity_relationships WHERE id = 'r2'"
        ).fetchone()
        assert row is not None
        assert row[0] == "hr-reporting"


class TestLearningDomainTracking:
    """Validate that learnings (corrections) track domain."""

    def test_learning_correction_has_domain_column(self, tmp_path):
        ls = LearningStore(base_dir=tmp_path, user_id="test-user")
        lid = ls.save_learning(
            category=LearningCategory.USER_CORRECTION,
            context={"query": "select *"},
            correction="Use explicit columns",
            source=LearningSource.AUTO_CAPTURE,
        )
        learning = ls.get_learning(lid)
        assert learning is not None
        # domain column exists and defaults to empty string
        assert "domain" in learning
        assert learning["domain"] == ""
