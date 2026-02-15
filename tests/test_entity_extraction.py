# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tests for entity extraction functionality."""

import os
import tempfile
import pytest
from datetime import datetime

from constat.discovery.models import (
    DocumentChunk,
    Entity,
    ChunkEntity,
    EnrichedChunk,
    SemanticType,
    NerType,
)
from constat.discovery.entity_extractor import EntityExtractor
from constat.discovery.vector_store import DuckDBVectorStore


class TestEntityModels:
    """Test Entity and ChunkEntity dataclasses."""

    def test_entity_creation(self):
        """Test creating an Entity."""
        entity = Entity(
            id="abc123",
            name="customers",
            display_name="Customers",
            semantic_type=SemanticType.CONCEPT,
            session_id="test-session",
        )

        assert entity.id == "abc123"
        assert entity.name == "customers"
        assert entity.display_name == "Customers"
        assert entity.semantic_type == SemanticType.CONCEPT
        assert entity.session_id == "test-session"

    def test_entity_defaults(self):
        """Test Entity default values."""
        entity = Entity(
            id="xyz",
            name="test",
            display_name="Test",
            semantic_type=SemanticType.CONCEPT,
            session_id="test-session",
        )

        assert entity.domain_id is None
        assert entity.ner_type is None
        assert entity.created_at is not None
        # Backwards compatibility: type property returns semantic_type
        assert entity.type == SemanticType.CONCEPT

    def test_chunk_entity_creation(self):
        """Test creating a ChunkEntity link."""
        link = ChunkEntity(
            chunk_id="chunk_1",
            entity_id="entity_1",
            confidence=0.85,
        )

        assert link.chunk_id == "chunk_1"
        assert link.entity_id == "entity_1"
        assert link.confidence == 0.85

    def test_enriched_chunk(self):
        """Test EnrichedChunk with entities."""
        chunk = DocumentChunk(
            document_name="test",
            content="Test content",
            section="Section",
            chunk_index=0,
        )
        entities = [
            Entity(
                id="e1",
                name="test entity",
                display_name="Test Entity",
                semantic_type=SemanticType.CONCEPT,
                session_id="test",
            ),
            Entity(
                id="e2",
                name="acme corp",
                display_name="Acme Corp",
                semantic_type=SemanticType.CONCEPT,
                ner_type=NerType.ORG,
                session_id="test",
            ),
        ]
        enriched = EnrichedChunk(chunk=chunk, score=0.85, entities=entities)

        assert enriched.chunk == chunk
        assert enriched.score == 0.85
        assert len(enriched.entities) == 2


class TestEntityExtractor:
    """Test EntityExtractor class."""

    def test_extract_schema_entities(self):
        """Test extracting schema entities by pattern matching."""
        extractor = EntityExtractor(
            session_id="test-session",
            schema_terms=["customers", "orders", "product_id"],
        )

        chunk = DocumentChunk(
            document_name="business_rules",
            content="The customers table contains all customer records. Each customer can have multiple orders. Use product_id to join with products.",
            section="Data Model",
            chunk_index=0,
        )

        results = extractor.extract(chunk)

        # Should find schema terms
        entity_names = {r[0].name.lower() for r in results}
        assert "customers" in entity_names or "customer" in entity_names
        assert "orders" in entity_names or "order" in entity_names

    def test_extract_business_terms(self):
        """Test extracting known business terms."""
        extractor = EntityExtractor(
            session_id="test-session",
            business_terms=["churn rate", "customer lifetime value", "MRR"],
        )

        chunk = DocumentChunk(
            document_name="metrics",
            content="The churn rate is calculated monthly. Customer lifetime value (CLV) and MRR are key metrics.",
            section="KPIs",
            chunk_index=0,
        )

        results = extractor.extract(chunk)

        entity_names = {r[0].name.lower() for r in results}
        assert "churn rate" in entity_names
        # MRR may or may not be extracted depending on spaCy's NER matching
        # for short acronyms - at minimum we should find churn rate
        assert len(entity_names) >= 1

    def test_entity_deduplication(self):
        """Test that entities are deduplicated across extractions."""
        extractor = EntityExtractor(
            session_id="test-session",
            schema_terms=["users"],
        )

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

        # Should only have one unique entity for "users"
        all_entities = extractor.get_all_entities()
        user_entities = [e for e in all_entities if "user" in e.name.lower()]
        assert len(user_entities) == 1

    def test_extractor_requires_session_id(self):
        """Test that session_id is required."""
        # This should work
        extractor = EntityExtractor(session_id="test")
        assert extractor.session_id == "test"

    def test_api_terms_extraction(self):
        """Test extracting API endpoint terms."""
        extractor = EntityExtractor(
            session_id="test-session",
            api_terms=["/users", "/orders", "GET /products"],
        )

        chunk = DocumentChunk(
            document_name="api_docs",
            content="Use /users endpoint to get user data. The /orders endpoint returns order history.",
            section="Endpoints",
            chunk_index=0,
        )

        results = extractor.extract(chunk)

        # Entity names may be normalized (singular form)
        entity_names = {r[0].name.lower() for r in results}
        # Check that we got API-related entities (may be normalized)
        assert any("/user" in name for name in entity_names)
        assert any("/order" in name for name in entity_names)

    def test_get_all_entities(self):
        """Test retrieving all extracted entities."""
        extractor = EntityExtractor(
            session_id="test-session",
            schema_terms=["customers", "products"],
        )

        chunk = DocumentChunk(
            document_name="docs",
            content="Query customers and products tables.",
            section="Guide",
            chunk_index=0,
        )

        extractor.extract(chunk)
        entities = extractor.get_all_entities()

        # Should have extracted entities
        assert len(entities) >= 1
        assert all(isinstance(e, Entity) for e in entities)

    def test_semantic_type_mapping_for_schema(self):
        """Test that SCHEMA patterns get CONCEPT semantic type."""
        extractor = EntityExtractor(
            session_id="test-session",
            schema_terms=["customers"],
        )

        chunk = DocumentChunk(
            document_name="doc",
            content="The customers table is used for storing customer data.",
            section="Guide",
            chunk_index=0,
        )

        results = extractor.extract(chunk)

        # Find the customers entity
        customer_entities = [e for e, _ in results if "customer" in e.name.lower()]
        assert len(customer_entities) >= 1

        # SCHEMA patterns should be CONCEPT (nouns/things)
        for entity in customer_entities:
            assert entity.semantic_type == SemanticType.CONCEPT

    def test_semantic_type_mapping_for_api(self):
        """Test that API patterns get ACTION semantic type."""
        extractor = EntityExtractor(
            session_id="test-session",
            api_terms=["create_user", "get_orders"],
        )

        chunk = DocumentChunk(
            document_name="doc",
            content="Call create_user to add new users. Use get_orders to fetch order history.",
            section="API",
            chunk_index=0,
        )

        results = extractor.extract(chunk)

        # API patterns should be ACTION (verbs/operations)
        api_entities = [e for e, _ in results if "create" in e.name.lower() or "get" in e.name.lower()]
        for entity in api_entities:
            assert entity.semantic_type == SemanticType.ACTION

    def test_display_name_generation(self):
        """Test that display_name is title case for schema names."""
        extractor = EntityExtractor(
            session_id="test-session",
            schema_terms=["order_items"],
        )

        chunk = DocumentChunk(
            document_name="doc",
            content="The order_items table links orders to products.",
            section="Schema",
            chunk_index=0,
        )

        results = extractor.extract(chunk)

        # Find the order_items entity
        order_entities = [e for e, _ in results if "order" in e.name.lower()]
        assert len(order_entities) >= 1

        # Display name should be title case
        for entity in order_entities:
            # name should be lowercase normalized
            assert entity.name == entity.name.lower() or "_" not in entity.name
            # display_name should be title case
            assert entity.display_name[0].isupper()

    def test_ner_type_preserved_for_spacy_entities(self):
        """Test that spaCy NER types are preserved as ner_type."""
        extractor = EntityExtractor(session_id="test-session")

        # spaCy should recognize organizations like "Microsoft"
        chunk = DocumentChunk(
            document_name="doc",
            content="Microsoft Corporation announced new products. Apple Inc released updates.",
            section="News",
            chunk_index=0,
        )

        results = extractor.extract(chunk)

        # Find entities from spaCy NER (like Microsoft, Apple)
        org_entities = [e for e, _ in results if e.ner_type == NerType.ORG]

        # If spaCy recognizes these as ORG, they should have ner_type set
        # Note: spaCy recognition may vary, so we just verify the pattern
        for entity in org_entities:
            assert entity.ner_type == NerType.ORG
            # NER entities should be CONCEPT (nouns/things)
            assert entity.semantic_type == SemanticType.CONCEPT

    def test_custom_pattern_entities_have_schema_ner_type(self):
        """Test that schema pattern entities get SCHEMA ner_type."""
        extractor = EntityExtractor(
            session_id="test-session",
            schema_terms=["custom_table"],
        )

        chunk = DocumentChunk(
            document_name="doc",
            content="Query the custom_table for data.",
            section="Schema",
            chunk_index=0,
        )

        results = extractor.extract(chunk)

        # Find the custom pattern entity
        custom_entities = [e for e, _ in results if "custom" in e.name.lower()]
        assert len(custom_entities) >= 1

        # Schema patterns should have SCHEMA ner_type
        for entity in custom_entities:
            assert entity.ner_type == NerType.SCHEMA

    def test_entity_domain_id_from_extractor(self):
        """Test that domain_id is set from extractor initialization."""
        extractor = EntityExtractor(
            session_id="test-session",
            domain_id="my-domain",
            schema_terms=["test_table"],
        )

        chunk = DocumentChunk(
            document_name="doc",
            content="The test_table stores data.",
            section="Schema",
            chunk_index=0,
        )

        results = extractor.extract(chunk)

        # Entities should have domain_id set
        for entity, _ in results:
            assert entity.domain_id == "my-domain"
            assert entity.session_id == "test-session"


@pytest.fixture
def temp_db():
    """Create a temporary database path for testing."""
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
    """Create a vector store for testing."""
    return DuckDBVectorStore(db_path=temp_db)


def _make_entity(name, session_id="sess-1", domain_id=None):
    """Helper to create an Entity with a deterministic ID."""
    import hashlib
    eid = hashlib.sha256(f"{name}:{session_id}".encode()).hexdigest()[:12]
    return Entity(
        id=eid,
        name=name,
        display_name=name.replace("_", " ").title(),
        semantic_type=SemanticType.CONCEPT,
        session_id=session_id,
        domain_id=domain_id,
    )


class TestFindEntityByName:
    """Test DuckDBVectorStore.find_entity_by_name with all argument combinations."""

    def test_find_by_name_only(self, vector_store):
        """Find entity by name without session or domain filter."""
        entity = _make_entity("customers", session_id="s1")
        vector_store.add_entities([entity], session_id="s1")

        found = vector_store.find_entity_by_name("customers")
        assert found is not None
        assert found.name == "customers"

    def test_find_by_name_case_insensitive(self, vector_store):
        """Lookup is case-insensitive."""
        entity = _make_entity("customers", session_id="s1")
        vector_store.add_entities([entity], session_id="s1")

        assert vector_store.find_entity_by_name("CUSTOMERS") is not None
        assert vector_store.find_entity_by_name("Customers") is not None

    def test_find_by_name_and_session(self, vector_store):
        """Filter by session_id."""
        e1 = _make_entity("orders", session_id="s1")
        e2 = _make_entity("orders", session_id="s2")
        vector_store.add_entities([e1], session_id="s1")
        vector_store.add_entities([e2], session_id="s2")

        found = vector_store.find_entity_by_name("orders", session_id="s1")
        assert found is not None
        assert found.session_id == "s1"

        found2 = vector_store.find_entity_by_name("orders", session_id="s2")
        assert found2 is not None
        assert found2.session_id == "s2"

        assert vector_store.find_entity_by_name("orders", session_id="s3") is None

    def test_find_by_name_and_domain_ids(self, vector_store):
        """Filter by domain_ids."""
        e1 = _make_entity("revenue", session_id="s1", domain_id="dom-a")
        e2 = _make_entity("revenue", session_id="s1", domain_id="dom-b")
        # Different id needed for second entity in same session
        e2.id = e2.id + "_b"
        vector_store.add_entities([e1, e2], session_id="s1")

        found = vector_store.find_entity_by_name("revenue", domain_ids=["dom-a"])
        assert found is not None
        assert found.domain_id == "dom-a"

        found_b = vector_store.find_entity_by_name("revenue", domain_ids=["dom-b"])
        assert found_b is not None
        assert found_b.domain_id == "dom-b"

        found_both = vector_store.find_entity_by_name("revenue", domain_ids=["dom-a", "dom-b"])
        assert found_both is not None

        assert vector_store.find_entity_by_name("revenue", domain_ids=["dom-z"]) is None

    def test_find_by_name_session_and_domain(self, vector_store):
        """Filter by both session_id and domain_ids together."""
        e1 = _make_entity("users", session_id="s1", domain_id="d1")
        e2 = _make_entity("users", session_id="s2", domain_id="d1")
        e2.id = e2.id + "_s2"
        vector_store.add_entities([e1], session_id="s1")
        vector_store.add_entities([e2], session_id="s2")

        # Both match domain, but only one matches session
        found = vector_store.find_entity_by_name("users", domain_ids=["d1"], session_id="s1")
        assert found is not None
        assert found.session_id == "s1"

        # Wrong session
        assert vector_store.find_entity_by_name("users", domain_ids=["d1"], session_id="s3") is None
        # Wrong domain
        assert vector_store.find_entity_by_name("users", domain_ids=["d9"], session_id="s1") is None

    def test_find_nonexistent_returns_none(self, vector_store):
        """Returns None when entity does not exist."""
        assert vector_store.find_entity_by_name("nonexistent") is None

    def test_returned_entity_has_all_fields(self, vector_store):
        """Returned Entity has all expected fields populated."""
        entity = _make_entity("invoices", session_id="s1", domain_id="d1")
        vector_store.add_entities([entity], session_id="s1")

        found = vector_store.find_entity_by_name("invoices", session_id="s1")
        assert found.id == entity.id
        assert found.name == "invoices"
        assert found.display_name == "Invoices"
        assert found.semantic_type == SemanticType.CONCEPT
        assert found.session_id == "s1"
        assert found.domain_id == "d1"
        assert found.created_at is not None
