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

from constat.core.config import EntityResolutionConfig
from constat.discovery.models import (
    ChunkType,
    DocumentChunk,
    EntityClass,
)


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
