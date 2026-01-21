"""Tests for entity extraction functionality."""

import pytest
from datetime import datetime

from constat.discovery.models import (
    DocumentChunk,
    Entity,
    ChunkEntity,
    EnrichedChunk,
    EntityType,
)
from constat.discovery.entity_extractor import (
    EntityExtractor,
    ExtractionConfig,
    create_schema_entities_from_catalog,
)


class TestEntityModels:
    """Test Entity and ChunkEntity dataclasses."""

    def test_entity_creation(self):
        """Test creating an Entity."""
        entity = Entity(
            id="abc123",
            name="customers",
            type=EntityType.TABLE,
            metadata={"source": "catalog"},
        )

        assert entity.id == "abc123"
        assert entity.name == "customers"
        assert entity.type == EntityType.TABLE
        assert entity.metadata == {"source": "catalog"}
        assert entity.created_at is not None

    def test_chunk_entity_creation(self):
        """Test creating a ChunkEntity link."""
        link = ChunkEntity(
            chunk_id="chunk_1",
            entity_id="entity_1",
            mention_count=3,
            confidence=0.95,
        )

        assert link.chunk_id == "chunk_1"
        assert link.entity_id == "entity_1"
        assert link.mention_count == 3
        assert link.confidence == 0.95

    def test_enriched_chunk_creation(self):
        """Test creating an EnrichedChunk."""
        chunk = DocumentChunk(
            document_name="test_doc",
            content="Some content",
            section="Introduction",
            chunk_index=0,
        )
        entities = [
            Entity(id="e1", name="customers", type=EntityType.TABLE),
            Entity(id="e2", name="revenue", type=EntityType.CONCEPT),
        ]

        enriched = EnrichedChunk(
            chunk=chunk,
            score=0.85,
            entities=entities,
        )

        assert enriched.chunk == chunk
        assert enriched.score == 0.85
        assert len(enriched.entities) == 2


class TestEntityExtractor:
    """Test EntityExtractor class."""

    def test_extract_schema_entities(self):
        """Test extracting schema entities by pattern matching."""
        config = ExtractionConfig(
            extract_schema=True,
            extract_ner=False,
            schema_entities=["customers", "orders", "product_id"],
        )
        extractor = EntityExtractor(config)

        chunk = DocumentChunk(
            document_name="business_rules",
            content="The customers table contains all customer records. Each customer can have multiple orders. Use product_id to join with products.",
            section="Data Model",
            chunk_index=0,
        )

        results = extractor.extract(chunk)

        # Should find customers, orders, product_id
        entity_names = {r[0].name.lower() for r in results}
        assert "customers" in entity_names
        assert "orders" in entity_names
        assert "product_id" in entity_names

    def test_extract_named_entities_pascal_case(self):
        """Test extracting PascalCase identifiers."""
        config = ExtractionConfig(
            extract_schema=False,
            extract_ner=True,
        )
        extractor = EntityExtractor(config)

        chunk = DocumentChunk(
            document_name="docs",
            content="The CustomerOrder class handles all OrderProcessing logic. It uses the PaymentGateway for transactions.",
            section="Classes",
            chunk_index=0,
        )

        results = extractor.extract(chunk)

        entity_names = {r[0].name for r in results}
        assert "CustomerOrder" in entity_names
        assert "OrderProcessing" in entity_names
        assert "PaymentGateway" in entity_names

    def test_extract_business_terms(self):
        """Test extracting known business terms."""
        config = ExtractionConfig(
            extract_schema=False,
            extract_ner=True,
            business_terms=["churn rate", "customer lifetime value", "MRR"],
        )
        extractor = EntityExtractor(config)

        chunk = DocumentChunk(
            document_name="metrics",
            content="The churn rate is calculated monthly. Customer lifetime value (CLV) and MRR are key metrics.",
            section="KPIs",
            chunk_index=0,
        )

        results = extractor.extract(chunk)

        entity_names = {r[0].name.lower() for r in results}
        assert "churn rate" in entity_names
        assert "mrr" in entity_names

    def test_mention_count(self):
        """Test that mention count is tracked correctly."""
        config = ExtractionConfig(
            extract_schema=True,
            extract_ner=False,
            schema_entities=["customers"],
        )
        extractor = EntityExtractor(config)

        chunk = DocumentChunk(
            document_name="docs",
            content="The customers table has many customers. Query customers by id. Update customers.",
            section="Guide",
            chunk_index=0,
        )

        results = extractor.extract(chunk)

        # Should have one entity with multiple mentions
        assert len(results) == 1
        entity, link = results[0]
        assert entity.name.lower() == "customers"
        assert link.mention_count == 4  # "customers" appears 4 times

    def test_entity_deduplication(self):
        """Test that entities are deduplicated across extractions."""
        config = ExtractionConfig(
            extract_schema=True,
            schema_entities=["users"],
        )
        extractor = EntityExtractor(config)

        chunk1 = DocumentChunk(
            document_name="doc1",
            content="The users table stores user data.",
            section="Schema",
            chunk_index=0,
        )
        chunk2 = DocumentChunk(
            document_name="doc2",
            content="Query the users table for active users.",
            section="Queries",
            chunk_index=0,
        )

        extractor.extract(chunk1)
        extractor.extract(chunk2)

        # Should only have one unique entity
        all_entities = extractor.get_all_entities()
        assert len(all_entities) == 1

    def test_confidence_levels(self):
        """Test that confidence levels are set appropriately."""
        config = ExtractionConfig(
            extract_schema=True,
            extract_ner=True,
            schema_entities=["orders"],
        )
        extractor = EntityExtractor(config)

        chunk = DocumentChunk(
            document_name="docs",
            content="The orders table uses CustomerOrder class.",
            section="Model",
            chunk_index=0,
        )

        results = extractor.extract(chunk)

        # Schema entities should have higher confidence than NER
        for entity, link in results:
            if entity.name.lower() == "orders":
                assert link.confidence >= 0.9  # Schema match
            elif entity.name == "CustomerOrder":
                assert link.confidence < 0.9  # NER extraction


class TestCreateSchemaEntities:
    """Test helper function for creating entities from catalog."""

    def test_create_from_catalog(self):
        """Test creating entities from table and column names."""
        entities = create_schema_entities_from_catalog(
            tables=["customers", "orders"],
            columns=["customer_id", "order_date"],
        )

        assert len(entities) == 4

        table_entities = [e for e in entities if e.type == EntityType.TABLE]
        column_entities = [e for e in entities if e.type == EntityType.COLUMN]

        assert len(table_entities) == 2
        assert len(column_entities) == 2

        table_names = {e.name for e in table_entities}
        assert "customers" in table_names
        assert "orders" in table_names


class TestExtractionConfig:
    """Test ExtractionConfig options."""

    def test_disable_all_extraction(self):
        """Test that disabling all extraction returns empty results."""
        config = ExtractionConfig(
            extract_schema=False,
            extract_ner=False,
            extract_concepts=False,
        )
        extractor = EntityExtractor(config)

        chunk = DocumentChunk(
            document_name="docs",
            content="Some content with CustomerOrder and tables.",
            section="Test",
            chunk_index=0,
        )

        results = extractor.extract(chunk)
        assert len(results) == 0

    def test_update_schema_entities(self):
        """Test dynamically updating schema entities."""
        config = ExtractionConfig(
            extract_schema=True,
            extract_ner=False,
        )
        extractor = EntityExtractor(config)

        chunk = DocumentChunk(
            document_name="docs",
            content="The products table contains items.",
            section="Schema",
            chunk_index=0,
        )

        # Initially no schema entities
        results = extractor.extract(chunk)
        assert len(results) == 0

        # Update with schema entities
        extractor.update_schema_entities(["products"])
        extractor.clear_cache()

        results = extractor.extract(chunk)
        assert len(results) == 1
        assert results[0][0].name.lower() == "products"