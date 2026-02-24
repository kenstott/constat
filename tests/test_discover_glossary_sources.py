# Copyright (c) 2025 Kenneth Stott

"""Test that /discover glossary terms include connected resources."""

import numpy as np
import pytest

from constat.discovery.models import (
    ChunkType,
    ChunkEntity,
    DocumentChunk,
    Entity,
    GlossaryTerm,
)
from constat.discovery.vector_store import DuckDBVectorStore

DIM = DuckDBVectorStore.EMBEDDING_DIM
SESSION_ID = "test-session"
USER_ID = "test-user"


@pytest.fixture()
def vs():
    """In-memory DuckDB vector store with linked glossary/entity/chunk data."""
    store = DuckDBVectorStore(db_path=":memory:")

    # 1. API chunk (the connected resource)
    api_chunk = DocumentChunk(
        document_name="api:catfacts.CatFact",
        content="CatFact endpoint: CatFact",
        section="rest/schema",
        chunk_index=0,
        source="api",
        chunk_type=ChunkType.DOCUMENT,
    )
    api_embedding = np.random.randn(1, DIM).astype(np.float32)
    store.add_chunks([api_chunk], api_embedding, source="api")

    # 2. Glossary chunk (for vector search to find)
    glossary_chunk = DocumentChunk(
        document_name="glossary:term001",
        content="Cat Fact: An interesting piece of information about cats.",
        section="glossary",
        chunk_index=0,
        source="document",
        chunk_type=ChunkType.GLOSSARY_TERM,
    )
    glossary_embedding = np.random.randn(1, DIM).astype(np.float32)
    store.add_chunks([glossary_chunk], glossary_embedding, source="document")

    # 3. Glossary term in DB (what get_glossary_term_by_id returns)
    term = GlossaryTerm(
        id="term001",
        name="cat fact",
        display_name="Cat Fact",
        definition="An interesting piece of information about cats.",
        aliases=["feline fact"],
        status="draft",
        provenance="llm",
        session_id=SESSION_ID,
        user_id=USER_ID,
    )
    store.add_glossary_term(term)

    # 4. Entity matching the glossary term name
    entity = Entity(
        id="ent001",
        name="cat fact",
        display_name="Cat Fact",
        semantic_type="concept",
        session_id=SESSION_ID,
    )
    store.add_entities([entity], SESSION_ID)

    # 5. Link entity to the API chunk
    api_chunk_id = store._generate_chunk_id(api_chunk)
    link = ChunkEntity(
        chunk_id=api_chunk_id,
        entity_id="ent001",
        confidence=1.0,
    )
    store.link_chunk_entities([link])

    return store


class TestResolvePhysicalResources:
    """Test that resolve_physical_resources finds connected resources."""

    def test_finds_resources_with_session_id(self, vs):
        """Entity found when session_id matches."""
        from constat.discovery.glossary_generator import resolve_physical_resources

        resources = resolve_physical_resources(
            "cat fact", SESSION_ID, vs, user_id=USER_ID,
        )
        assert len(resources) > 0, "Should find connected resources"
        assert resources[0]["entity_name"] == "Cat Fact"
        sources = resources[0]["sources"]
        assert any("api:catfacts" in s["document_name"] for s in sources)

    def test_no_resources_without_session_id(self, vs):
        """Entity not found when session_id is wrong (visibility filter)."""
        from constat.discovery.glossary_generator import resolve_physical_resources

        resources = resolve_physical_resources(
            "cat fact", "wrong-session", vs, user_id="wrong-user",
        )
        assert len(resources) == 0

    def test_finds_resources_with_domain_ids(self, vs):
        """Adding domain_ids doesn't break when entity has no domain."""
        from constat.discovery.glossary_generator import resolve_physical_resources

        resources = resolve_physical_resources(
            "cat fact", SESSION_ID, vs,
            domain_ids=["some-domain"],
            user_id=USER_ID,
        )
        # Entity has no domain_id, but session_id matches → still visible
        assert len(resources) > 0


class TestSearchAllGlossarySources:
    """Test that search_all includes sources in glossary entries."""

    def test_glossary_entry_has_sources(self, vs):
        """search_all glossary entries should include connected resources."""
        from unittest.mock import MagicMock
        from constat.discovery.schema_tools import SchemaDiscoveryTools

        # Mock doc_tools with the real vector store
        doc_tools = MagicMock()
        doc_tools._vector_store = vs
        doc_tools._active_domain_ids = []
        doc_tools._model_lock = __import__("threading").Lock()
        # Model that returns a fixed embedding for any query
        mock_model = MagicMock()
        mock_model.encode = lambda texts, **kw: np.random.randn(len(texts), DIM).astype(np.float32)
        doc_tools._model = mock_model
        doc_tools.search_documents = MagicMock(return_value=[])

        # Mock schema_manager
        schema_manager = MagicMock()
        schema_manager.find_relevant_tables = MagicMock(return_value=[])

        tools = SchemaDiscoveryTools(
            schema_manager=schema_manager,
            doc_tools=doc_tools,
            session_id=SESSION_ID,
            user_id=USER_ID,
        )

        result = tools.search_all("cat facts", limit=10)

        glossary = result.get("glossary", [])
        assert len(glossary) > 0, f"Should find glossary terms, got: {result}"

        # Find the Cat Fact entry
        cat_fact = next((g for g in glossary if g.get("name") == "Cat Fact"), None)
        assert cat_fact is not None, f"Should find Cat Fact, got: {glossary}"
        assert cat_fact["definition"] == "An interesting piece of information about cats."
        assert "sources" in cat_fact, f"Cat Fact should have sources, got: {cat_fact}"
        assert len(cat_fact["sources"]) > 0
        assert cat_fact["sources"][0]["entity"] == "Cat Fact"
        locations = cat_fact["sources"][0]["locations"]
        assert any("api:catfacts" in loc for loc in locations)
