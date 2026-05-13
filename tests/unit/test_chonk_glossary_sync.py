# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Unit tests for chonk-owned glossary term fields.

Covers:
  - sync writes empty definition/aliases/parent_id (chonk is authoritative)
  - sync skips human-provenance terms
  - _build_term_from_row falls back to chonk live for non-approved chonk_llm terms
  - Non-empty term field takes precedence over chonk (human edit wins)
  - Approved term uses stored values, not chonk
  - SVO parent resolved via get_entity_parents with correct verb mapping
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, call

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chonk_store(descriptions=None, aliases=None, parents=None):
    store = MagicMock()
    store.get_entity_descriptions.return_value = descriptions or {}
    store.get_entity_aliases_by_names.return_value = aliases or {}
    store.get_entity_parents.return_value = parents or {}
    return store


def _make_conn_execute(entity_domain_rows, entity_meta_rows, hierarchy_rows=None):
    """Return a mock _conn that serves the three queries sync_entity_descriptions_to_glossary makes."""
    conn = MagicMock()
    results = [
        MagicMock(fetchall=MagicMock(return_value=entity_domain_rows)),
        MagicMock(fetchall=MagicMock(return_value=entity_meta_rows)),
        MagicMock(fetchall=MagicMock(return_value=hierarchy_rows or [])),
    ]
    conn.execute.side_effect = results
    return conn


# ---------------------------------------------------------------------------
# sync_entity_descriptions_to_glossary
# ---------------------------------------------------------------------------

class TestSyncEntityDescriptionsToGlossary:

    def _run_sync(self, chonk_store, vector_store, domain_names=None, session_id="sess1", user_id="u1"):
        from constat.storage._chonk_glossary_sync import sync_entity_descriptions_to_glossary
        return sync_entity_descriptions_to_glossary(
            chonk_store, vector_store, domain_names or ["sales"], session_id, user_id
        )

    def test_writes_empty_definition_and_aliases(self):
        conn = _make_conn_execute(
            entity_domain_rows=[("eid1", "global:sales")],
            entity_meta_rows=[("eid1", "customer", "Customer", "concept")],
        )
        chonk_store = MagicMock()
        chonk_store.vector._conn = conn
        chonk_store.get_entity_descriptions.return_value = {"eid1": "A buyer of goods"}

        vector_store = MagicMock()
        vector_store.get_glossary_term.return_value = None

        self._run_sync(chonk_store, vector_store)

        added = vector_store.add_glossary_term.call_args[0][0]
        assert added.definition == ""
        assert added.aliases == []
        assert added.parent_id is None

    def test_writes_correct_metadata(self):
        conn = _make_conn_execute(
            entity_domain_rows=[("eid1", "global:sales")],
            entity_meta_rows=[("eid1", "customer", "Customer", "concept")],
        )
        chonk_store = MagicMock()
        chonk_store.vector._conn = conn
        chonk_store.get_entity_descriptions.return_value = {"eid1": "A buyer"}

        vector_store = MagicMock()
        vector_store.get_glossary_term.return_value = None

        self._run_sync(chonk_store, vector_store)

        added = vector_store.add_glossary_term.call_args[0][0]
        assert added.name == "customer"
        assert added.display_name == "Customer"
        assert added.provenance == "chonk_llm"
        assert added.status == "draft"

    def test_skips_human_provenance_terms(self):
        conn = _make_conn_execute(
            entity_domain_rows=[("eid1", "global:sales")],
            entity_meta_rows=[("eid1", "customer", "Customer", "concept")],
        )
        chonk_store = MagicMock()
        chonk_store.vector._conn = conn
        chonk_store.get_entity_descriptions.return_value = {"eid1": "A buyer"}

        existing = MagicMock()
        existing.provenance = "human"
        vector_store = MagicMock()
        vector_store.get_glossary_term.return_value = existing

        count = self._run_sync(chonk_store, vector_store)

        vector_store.add_glossary_term.assert_not_called()
        assert count == 0

    def test_returns_zero_for_empty_domains(self):
        chonk_store = MagicMock()
        vector_store = MagicMock()
        from constat.storage._chonk_glossary_sync import sync_entity_descriptions_to_glossary
        count = sync_entity_descriptions_to_glossary(chonk_store, vector_store, [], "s1", "u1")
        assert count == 0

    def test_returns_zero_when_no_descriptions(self):
        conn = _make_conn_execute(
            entity_domain_rows=[("eid1", "global:sales")],
            entity_meta_rows=[],
        )
        chonk_store = MagicMock()
        chonk_store.vector._conn = conn
        chonk_store.get_entity_descriptions.return_value = {}

        vector_store = MagicMock()
        count = self._run_sync(chonk_store, vector_store)
        assert count == 0


# ---------------------------------------------------------------------------
# _build_term_from_row — live chonk fallback
# ---------------------------------------------------------------------------

class TestBuildTermFromRowChonkFallback:

    def _row(self, **kwargs):
        defaults = {
            "name": "customer",
            "display_name": "Customer",
            "definition": "",
            "domain": "sales",
            "parent_id": None,
            "parent_verb": "HAS_KIND",
            "aliases": [],
            "semantic_type": "concept",
            "ner_type": None,
            "cardinality": "many",
            "status": "draft",
            "provenance": "chonk_llm",
            "glossary_status": "defined",
            "entity_id": "eid1",
            "glossary_id": "gid1",
            "tags": {},
            "ignored": False,
            "canonical_source": None,
        }
        defaults.update(kwargs)
        return defaults

    def _call(self, row, chonk_store=None):
        from constat.server.graphql.resolvers import _build_term_from_row
        vs = MagicMock()
        vs.find_entity_by_name.return_value = None
        return _build_term_from_row(row, "u1", {}, {}, vs, chonk_store)

    def test_live_description_from_chonk_when_empty(self):
        chonk_store = _make_chonk_store(descriptions={"customer": "A buyer of goods"})
        result = self._call(self._row(), chonk_store)
        assert result.definition == "A buyer of goods"

    def test_live_aliases_from_chonk_when_empty(self):
        chonk_store = _make_chonk_store(aliases={"customer": ["client", "buyer"]})
        result = self._call(self._row(), chonk_store)
        assert result.aliases == ["client", "buyer"]

    def test_live_parent_from_chonk_svo(self):
        chonk_store = _make_chonk_store(parents={"customer": ("person", "type_of")})
        result = self._call(self._row(), chonk_store)
        assert result.parent_id == "person"
        assert result.parent_verb == "HAS_KIND"

    def test_svo_part_of_maps_to_has_one(self):
        chonk_store = _make_chonk_store(parents={"customer": ("account", "part_of")})
        result = self._call(self._row(), chonk_store)
        assert result.parent_verb == "HAS_ONE"

    def test_human_definition_takes_precedence_over_chonk(self):
        chonk_store = _make_chonk_store(descriptions={"customer": "chonk description"})
        result = self._call(self._row(definition="human description"), chonk_store)
        assert result.definition == "human description"
        chonk_store.get_entity_descriptions.assert_not_called()

    def test_human_aliases_take_precedence_over_chonk(self):
        chonk_store = _make_chonk_store(aliases={"customer": ["chonk-alias"]})
        result = self._call(self._row(aliases=["human-alias"]), chonk_store)
        assert result.aliases == ["human-alias"]
        chonk_store.get_entity_aliases_by_names.assert_not_called()

    def test_human_parent_takes_precedence_over_chonk(self):
        chonk_store = _make_chonk_store(parents={"customer": ("chonk-parent", "type_of")})
        result = self._call(self._row(parent_id="human-parent"), chonk_store)
        assert result.parent_id == "human-parent"
        chonk_store.get_entity_parents.assert_not_called()

    def test_approved_term_does_not_query_chonk(self):
        chonk_store = _make_chonk_store(
            descriptions={"customer": "chonk desc"},
            aliases={"customer": ["chonk-alias"]},
        )
        result = self._call(self._row(status="approved", definition="", aliases=[]), chonk_store)
        chonk_store.get_entity_descriptions.assert_not_called()
        chonk_store.get_entity_aliases_by_names.assert_not_called()

    def test_non_chonk_llm_term_does_not_query_chonk(self):
        chonk_store = _make_chonk_store(descriptions={"customer": "chonk desc"})
        result = self._call(self._row(provenance="human", definition=""), chonk_store)
        chonk_store.get_entity_descriptions.assert_not_called()

    def test_no_chonk_store_returns_empty_fields(self):
        result = self._call(self._row())
        assert result.definition == "" or result.definition is None
        assert result.aliases == [] or result.aliases is None
