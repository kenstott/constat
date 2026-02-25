# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tests for glossary_term_to_chunk with relationship enrichment."""

import pytest

from constat.catalog.glossary_builder import glossary_term_to_chunk
from constat.discovery.models import GlossaryTerm


def _make_term(**overrides) -> GlossaryTerm:
    defaults = dict(
        id="t1",
        name="customer",
        display_name="Customer",
        definition="A person who buys products",
        session_id="s1",
    )
    defaults.update(overrides)
    return GlossaryTerm(**defaults)


class TestGlossaryTermToChunkRelationships:

    def test_with_relationships(self):
        term = _make_term()
        rels = [
            {"subject_name": "customer", "verb": "uses", "object_name": "product"},
            {"subject_name": "order", "verb": "belongs to", "object_name": "customer"},
        ]
        chunk = glossary_term_to_chunk(term, [], relationships=rels)
        assert "Related: " in chunk.content
        assert "customer USES product" in chunk.content
        assert "order BELONGS_TO customer" in chunk.content

    def test_filters_hierarchy_dupes(self):
        term = _make_term(parent_id="account")
        rels = [
            {"subject_name": "customer", "verb": "has_kind", "object_name": "account"},
            {"subject_name": "customer", "verb": "uses", "object_name": "product"},
        ]
        chunk = glossary_term_to_chunk(term, [], relationships=rels)
        assert "HAS_KIND" not in chunk.content
        assert "customer USES product" in chunk.content

    def test_hierarchy_dupe_reverse_direction(self):
        """Hierarchy verb with parent as subject is also filtered."""
        term = _make_term(parent_id="account")
        rels = [
            {"subject_name": "account", "verb": "has_a", "object_name": "customer"},
        ]
        chunk = glossary_term_to_chunk(term, [], relationships=rels)
        assert "Related:" not in chunk.content

    def test_hierarchy_verb_kept_for_non_parent(self):
        """HAS_KIND with a non-parent entity should NOT be filtered."""
        term = _make_term(parent_id="account")
        rels = [
            {"subject_name": "customer", "verb": "has_kind", "object_name": "vip customer"},
        ]
        chunk = glossary_term_to_chunk(term, [], relationships=rels)
        assert "HAS_KIND" in chunk.content

    def test_verb_upper_snake_case(self):
        term = _make_term()
        rels = [
            {"subject_name": "customer", "verb": "related to", "object_name": "order"},
        ]
        chunk = glossary_term_to_chunk(term, [], relationships=rels)
        assert "RELATED_TO" in chunk.content

    def test_no_relationships(self):
        term = _make_term()
        chunk = glossary_term_to_chunk(term, [])
        assert "Related:" not in chunk.content

    def test_empty_relationships_list(self):
        term = _make_term()
        chunk = glossary_term_to_chunk(term, [], relationships=[])
        assert "Related:" not in chunk.content

    def test_no_parent_no_filter(self):
        """Without a parent_id, hierarchy verbs are kept."""
        term = _make_term(parent_id=None)
        rels = [
            {"subject_name": "customer", "verb": "has_kind", "object_name": "enterprise customer"},
        ]
        chunk = glossary_term_to_chunk(term, [], relationships=rels)
        assert "HAS_KIND" in chunk.content
