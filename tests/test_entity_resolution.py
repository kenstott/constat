# Copyright (c) 2025 Kenneth Stott
# Canary: e78a8f88-58a0-4e19-b6c6-0d98343038b5
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tests for entity resolution — Phase 1."""

from __future__ import annotations
import pytest

from constat.core.config import EntityResolutionConfig
from constat.discovery.entity_extractor import EntityExtractor
from constat.discovery.models import (
    ChunkType,
    DocumentChunk,
    EntityClass,
    SemanticType,
)
from constat.discovery.ner_fingerprint import compute_ner_fingerprint


class TestEntityResolutionConfig:
    """Test EntityResolutionConfig model."""

    def test_static_values(self):
        cfg = EntityResolutionConfig(
            entity_type="CURRENCY",
            values=["USD", "EUR", "GBP"],
        )
        assert cfg.entity_type == "CURRENCY"
        assert cfg.values == ["USD", "EUR", "GBP"]
        assert cfg.max_values == 10000

    def test_db_shorthand(self):
        cfg = EntityResolutionConfig(
            entity_type="COUNTRY",
            source="countries_db",
            table="countries",
            name_column="country_name",
        )
        assert cfg.source == "countries_db"
        assert cfg.table == "countries"
        assert cfg.name_column == "country_name"

    def test_custom_query(self):
        cfg = EntityResolutionConfig(
            entity_type="CUSTOMER",
            source="customer_graph",
            query="MATCH (n:Customer) RETURN n.name AS name",
        )
        assert cfg.query is not None
        assert cfg.table is None

    def test_api_source(self):
        cfg = EntityResolutionConfig(
            entity_type="COUNTRY",
            source="countries_api",
            endpoint="/countries",
            items_path="data",
            name_field="name",
        )
        assert cfg.endpoint == "/countries"
        assert cfg.items_path == "data"

    def test_max_values_default(self):
        cfg = EntityResolutionConfig(entity_type="X", values=["a"])
        assert cfg.max_values == 10000

    def test_max_values_override(self):
        cfg = EntityResolutionConfig(entity_type="X", values=["a"], max_values=500)
        assert cfg.max_values == 500


class TestEntityExtractorWithEntityTerms:
    """Test EntityExtractor recognizes entity resolution values."""

    def test_entity_terms_recognized(self):
        extractor = EntityExtractor(
            session_id="test",
            entity_terms={"COUNTRY": ["France", "Germany", "Japan"]},
        )
        chunk = DocumentChunk(
            document_name="test_doc",
            content="We expanded operations to France and Germany last quarter.",
        )
        results = extractor.extract(chunk)
        entity_names = {e.name for e, _ in results}
        assert "France" in entity_names or "france" in entity_names

    def test_entity_terms_high_confidence(self):
        extractor = EntityExtractor(
            session_id="test",
            entity_terms={"COUNTRY": ["France"]},
        )
        chunk = DocumentChunk(
            document_name="test_doc",
            content="Our main market is France.",
        )
        results = extractor.extract(chunk)
        for entity, link in results:
            if "france" in entity.name.lower():
                assert link.confidence == 0.95
                break
        else:
            pytest.fail("France entity not found in results")

    def test_entity_terms_label_preserved(self):
        extractor = EntityExtractor(
            session_id="test",
            entity_terms={"COUNTRY": ["France"]},
        )
        chunk = DocumentChunk(
            document_name="test_doc",
            content="Revenue from France grew 20%.",
        )
        results = extractor.extract(chunk)
        for entity, _ in results:
            if "france" in entity.name.lower():
                assert entity.ner_type == "COUNTRY"
                break
        else:
            pytest.fail("France entity not found")

    def test_multiple_entity_types(self):
        extractor = EntityExtractor(
            session_id="test",
            entity_terms={
                "COUNTRY": ["France", "Germany"],
                "CURRENCY": ["EUR", "USD"],
            },
        )
        chunk = DocumentChunk(
            document_name="test_doc",
            content="France uses EUR as its currency.",
        )
        results = extractor.extract(chunk)
        labels = {e.ner_type for e, _ in results}
        # Should find at least COUNTRY
        assert "COUNTRY" in labels or len(results) > 0

    def test_short_values_filtered(self):
        """Values with length <= 1 should not create patterns."""
        extractor = EntityExtractor(
            session_id="test",
            entity_terms={"LETTER": ["A", "B", "France"]},
        )
        chunk = DocumentChunk(
            document_name="test_doc",
            content="A is for France B is not recognized.",
        )
        results = extractor.extract(chunk)
        entity_names = {e.name for e, _ in results}
        # "A" and "B" should NOT be recognized (too short)
        assert "a" not in entity_names
        assert "b" not in entity_names

    def test_empty_entity_terms(self):
        """Empty or None entity_terms should not cause errors."""
        extractor = EntityExtractor(
            session_id="test",
            entity_terms=None,
        )
        chunk = DocumentChunk(
            document_name="test_doc",
            content="This is a plain document about nothing.",
        )
        results = extractor.extract(chunk)
        # Should still work, just no custom patterns
        assert isinstance(results, list)


class TestChunkTypeEntityValue:
    """Test ENTITY_VALUE chunk type."""

    def test_entity_value_enum(self):
        assert ChunkType.ENTITY_VALUE.value == "entity_value"

    def test_document_chunk_with_entity_value(self):
        chunk = DocumentChunk(
            document_name="entity:hr.departments",
            content="Engineering",
            chunk_index=1,
            source="entity_resolution",
            chunk_type=ChunkType.ENTITY_VALUE,
        )
        assert chunk.chunk_type == ChunkType.ENTITY_VALUE
        assert chunk.source == "entity_resolution"


class TestEntityClass:
    """Test EntityClass constants."""

    def test_constants(self):
        assert EntityClass.METADATA_ENTITY == "metadata_entity"
        assert EntityClass.DATA_ENTITY == "data_entity"
        assert EntityClass.MIXED == "mixed"


class TestNerFingerprintWithEntityTerms:
    """Test that entity_terms affect the NER fingerprint."""

    def test_fingerprint_changes_with_entity_terms(self):
        fp1 = compute_ner_fingerprint(["c1"], ["t1"], ["a1"])
        fp2 = compute_ner_fingerprint(
            ["c1"], ["t1"], ["a1"],
            entity_terms={"COUNTRY": ["France"]},
        )
        assert fp1 != fp2

    def test_fingerprint_stable_without_entity_terms(self):
        fp1 = compute_ner_fingerprint(["c1"], ["t1"], ["a1"])
        fp2 = compute_ner_fingerprint(["c1"], ["t1"], ["a1"])
        assert fp1 == fp2

    def test_fingerprint_changes_with_different_values(self):
        fp1 = compute_ner_fingerprint(
            ["c1"], ["t1"], ["a1"],
            entity_terms={"COUNTRY": ["France"]},
        )
        fp2 = compute_ner_fingerprint(
            ["c1"], ["t1"], ["a1"],
            entity_terms={"COUNTRY": ["France", "Germany"]},
        )
        assert fp1 != fp2
