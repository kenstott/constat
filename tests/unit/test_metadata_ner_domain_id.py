# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Unit tests for process_metadata_through_ner domain_id fix.

Regression test: EntityExtractor was constructed without domain_id, causing
ValueError: Entity domain_id is required when entities were inserted into the
vector store. The fix adds domain_id="__base__" to the constructor call.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, call

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_nlp():
    """Return a minimal spaCy-like NLP mock that produces no entities."""
    doc_mock = MagicMock()
    doc_mock.ents = []

    pipe_mock = MagicMock()
    pipe_mock.pipe_names = []
    # add_pipe returns a ruler mock that accepts add_patterns
    ruler_mock = MagicMock()
    pipe_mock.add_pipe.return_value = ruler_mock
    pipe_mock.__call__ = lambda self_, text: doc_mock  # nlp(text) → doc

    return pipe_mock


def _make_mixin(captured_entities: list):
    """Build a minimal _EntityMixin instance with mocked collaborators."""
    from constat.discovery.doc_tools._entities import _EntityMixin

    class _Impl(_EntityMixin):
        """Concrete implementation for testing."""

        def __init__(self):
            self._schema_entities = ["order", "customer"]
            self._openapi_operations = []
            self._openapi_schemas = []
            self._graphql_types = []
            self._graphql_fields = []
            self._stop_list = set()
            self._vector_store = MagicMock()

            # Capture entities passed to add_entities
            def _capture_add_entities(entities, session_id=None):
                captured_entities.extend(entities)

            self._vector_store.add_entities.side_effect = _capture_add_entities
            self._vector_store.link_chunk_entities = MagicMock()

    return _Impl()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestProcessMetadataNerDomainId:
    """Tests that process_metadata_through_ner creates entities with domain_id."""

    def test_entity_extractor_constructed_with_base_domain_id(self):
        """EntityExtractor in process_metadata_through_ner must receive domain_id='__base__'."""
        captured_entities = []

        fake_nlp = _make_fake_nlp()

        with patch(
            "constat.discovery.entity_extractor.get_nlp",
            return_value=fake_nlp,
        ):
            from constat.discovery.entity_extractor import EntityExtractor

            constructed_with = {}

            original_init = EntityExtractor.__init__

            def recording_init(self, session_id, domain_id=None, **kwargs):
                constructed_with["domain_id"] = domain_id
                original_init(self, session_id=session_id, domain_id=domain_id, **kwargs)

            with patch.object(EntityExtractor, "__init__", recording_init):
                mixin = _make_mixin(captured_entities)
                mixin.process_metadata_through_ner([("orders_table", "order id customer reference")])

        assert "domain_id" in constructed_with, "EntityExtractor.__init__ was never called"
        assert constructed_with["domain_id"] == "__base__", (
            f"Expected domain_id='__base__', got {constructed_with['domain_id']!r}. "
            "The bug fix (domain_id='__base__') may have been reverted."
        )

    def test_entities_produced_have_non_empty_domain_id(self):
        """All entities produced by process_metadata_through_ner must have domain_id set."""
        from constat.discovery.models import Entity, SemanticType

        captured_entities = []

        # Inject a pre-fabricated entity directly into the extractor cache
        # to test the path from extractor → add_entities without needing spaCy.
        fake_entity = Entity(
            id="abc123",
            name="order",
            display_name="Order",
            semantic_type=SemanticType.CONCEPT,
            session_id="__metadata__",
            domain_id="__base__",
        )

        fake_nlp = _make_fake_nlp()

        with patch("constat.discovery.entity_extractor.get_nlp", return_value=fake_nlp):
            from constat.discovery.entity_extractor import EntityExtractor

            original_get_all = EntityExtractor.get_all_entities

            def injecting_get_all(self):
                return [fake_entity]

            with patch.object(EntityExtractor, "get_all_entities", injecting_get_all):
                mixin = _make_mixin(captured_entities)
                mixin.process_metadata_through_ner([("orders", "order management system")])

        assert captured_entities, "No entities were passed to add_entities"

        for entity in captured_entities:
            if not entity.domain_id:
                pytest.fail(
                    f"Entity {entity.name!r} has empty domain_id={entity.domain_id!r}. "
                    "This would trigger ValueError in the vector store."
                )

    def test_entities_would_pass_domain_id_validation(self):
        """Entities from process_metadata_through_ner pass the domain_id guard."""
        from constat.discovery.models import Entity, SemanticType

        captured_entities = []
        fake_nlp = _make_fake_nlp()

        # Simulate what the real extractor produces with domain_id="__base__"
        test_entity = Entity(
            id="def456",
            name="customer",
            display_name="Customer",
            semantic_type=SemanticType.CONCEPT,
            session_id="__metadata__",
            domain_id="__base__",
        )

        with patch("constat.discovery.entity_extractor.get_nlp", return_value=fake_nlp):
            from constat.discovery.entity_extractor import EntityExtractor

            with patch.object(EntityExtractor, "get_all_entities", return_value=[test_entity]):
                mixin = _make_mixin(captured_entities)
                mixin.process_metadata_through_ner([("customers", "customer data")])

        assert captured_entities, "No entities were passed to add_entities"

        # Replicate the validation guard from the vector store
        for entity in captured_entities:
            try:
                if not entity.domain_id:
                    raise ValueError(f"Entity domain_id is required, got {entity.domain_id!r}")
            except ValueError as exc:
                pytest.fail(f"Entity failed domain_id validation: {exc}")

    def test_empty_metadata_returns_early_without_error(self):
        """Empty metadata_texts list must return without touching the vector store."""
        captured_entities = []
        mixin = _make_mixin(captured_entities)
        mixin.process_metadata_through_ner([])

        mixin._vector_store.add_entities.assert_not_called()
        assert captured_entities == []

    def test_metadata_with_blank_text_skips_those_chunks(self):
        """Metadata entries with blank text must be skipped; no crash."""
        captured_entities = []
        fake_nlp = _make_fake_nlp()

        with patch("constat.discovery.entity_extractor.get_nlp", return_value=fake_nlp):
            mixin = _make_mixin(captured_entities)
            # All texts are blank — no chunks → early return
            mixin.process_metadata_through_ner([("source", ""), ("other", "   ")])

        # add_entities should not be called when no chunks are produced
        mixin._vector_store.add_entities.assert_not_called()
